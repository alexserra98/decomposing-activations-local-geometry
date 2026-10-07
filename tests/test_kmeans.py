"""KMeans model fitting, local geometry, rank selection, and persistence."""

import pytest
import torch

from dalg.init.centroid_artifact import compute_cluster_pca_directions
from dalg.init.neighborhood_pca import compute_neighborhood_pca
from dalg.init.projected_knn import KMeansTorch
from dalg.models.kmeans import KMeans, load_kmeans, save_kmeans


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def points_and_centers():
    gen = torch.Generator().manual_seed(18)
    centers = torch.tensor([[-20.0, 0, 0, 0], [20.0, 0, 0, 0]], dtype=torch.float64)
    X = torch.randn(2, 40, 4, generator=gen, dtype=torch.float64)
    X *= torch.tensor([2.0, 1.2, 0.7, 0.3])
    X += centers[:, None, :] + 0.2
    return X.reshape(-1, 4), centers


def test_seeded_fit_matches_existing_solver():
    X, _ = points_and_centers()
    kwargs = dict(restarts=2, tol=1e-6, seed=5, device="cpu", block_x=11, block_c=1)
    expected = KMeansTorch(2, n_iter=12, **kwargs).fit(X)
    model = KMeans(2, max_iter=12, **kwargs)
    assert model.fit(X) is model
    torch.testing.assert_close(model.mu, expected, rtol=0, atol=0)
    assert model.fit_metadata["n_samples"] == len(X)
    assert model.fit_metadata["inertia"] > 0
    assert model.q == 0
    assert not list(model.parameters())
    torch.testing.assert_close(model.predict(X), torch.cdist(X.float(), model.mu).argmin(1))
    repeat = KMeans(2, max_iter=12, **kwargs).fit(X)
    torch.testing.assert_close(model.mu, repeat.mu, rtol=0, atol=0)


def test_blocked_predictions_and_hard_responsibilities_keep_first_tie():
    model = KMeans.from_centroids(
        torch.tensor([[0.0, 0], [2.0, 0], [0.0, 0]]), block_x=1, block_c=1,
    )
    X = torch.tensor([[1.0, 0], [0.0, 0], [3.0, 0]])
    expected = torch.tensor([0, 0, 1])
    torch.testing.assert_close(model.predict(X), expected)
    for tau in (0.1, 1.0, 10.0):
        resp = model.responsibilities(X, tau=tau)
        assert resp.dtype == model.mu.dtype
        torch.testing.assert_close(resp.sum(1), torch.ones(3))
        torch.testing.assert_close(resp.argmax(1), expected)
        assert set(resp.unique().tolist()) == {0.0, 1.0}
    assert model.predict(X[:0]).shape == (0,)
    assert model.responsibilities(X[:0]).shape == (0, 3)
    for tau in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="tau"):
            model.responsibilities(X, tau=tau)


@pytest.mark.parametrize("method", ["cluster", "knn"])
def test_local_pca_matches_direct_centroid_centered_covariance(method):
    X, centers = points_and_centers()
    model = KMeans.from_centroids(centers, block_x=7)
    if method == "cluster":
        assert model.compute_pcs(X, threshold=1, chunk_elems=48, eig_batch_size=1) is model
        basis = model.W
    else:
        assert model.compute_init_pcs(X, rank=2, neighbors=17, eig_batch_size=1) is model
        basis = model.W_init
        assert model.q == 0
        with pytest.raises(RuntimeError, match="compute_pcs"):
            _ = model.W
    assert basis.shape == (2, 4, 1 if method == "cluster" else 2)
    if method == "cluster":
        assert model.eigenvalues.shape == (2, 4)
        assert model.eigenvalues.dtype == torch.float64
        torch.testing.assert_close(model.component_ranks, torch.tensor([1, 1]))
        assert model.pca_valid.all()
    else:
        assert model.init_pca_neighbors == 17
    for k in range(2):
        if method == "cluster":
            local = X[model.predict(X) == k]
        else:
            nearest = torch.cdist(X, centers[k:k + 1]).squeeze(1).argsort()[:17]
            local = X[nearest]
        residual = local - centers[k]
        covariance = residual.T @ residual / len(local)
        values, vectors = torch.linalg.eigh(covariance)
        rank = 1 if method == "cluster" else 2
        direct = vectors[:, -rank:]
        if method == "cluster":
            torch.testing.assert_close(model.eigenvalues[k], values.flip(0), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(basis[k] @ basis[k].T, direct @ direct.T, atol=1e-12, rtol=1e-12)
        active = basis[k, :, :rank]
        torch.testing.assert_close(active.T @ active, torch.eye(rank, dtype=torch.float64))
        assert not basis[k, :, rank:].any()
    if method == "cluster":
        assert torch.all(model.eigenvalues[:, :-1] >= model.eigenvalues[:, 1:])


def test_pca_helpers_keep_legacy_return_contract():
    X, centers = points_and_centers()
    labels = KMeans.from_centroids(centers).predict(X)
    direct = compute_cluster_pca_directions(X, labels, centers, rank=2)
    with_spectrum = compute_cluster_pca_directions(X, labels, centers, rank=2, return_eigenvalues=True)
    torch.testing.assert_close(direct, with_spectrum[0])
    indices = torch.stack([torch.arange(12), torch.arange(40, 52)])
    direct = compute_neighborhood_pca(X, centers, indices, rank=2, device=torch.device("cpu"), eig_batch_size=1)
    with_spectrum = compute_neighborhood_pca(X, centers, indices, rank=2, device=torch.device("cpu"), eig_batch_size=1, return_eigenvalues=True)
    torch.testing.assert_close(direct, with_spectrum[0].float(), rtol=1e-5, atol=1e-6)


def spectral_points_and_centers():
    centers = torch.tensor([[-20., 0., 0., 0.], [20., 0., 0., 0.]], dtype=torch.float64)
    spectra = torch.tensor([[10., 8., 5., 1.], [10., 6., 5., 4.]], dtype=torch.float64)
    axes = torch.diag_embed((4 * spectra).sqrt())
    points = torch.cat((axes, -axes), dim=1) + centers[:, None, :]
    return points.reshape(-1, 4), centers


def spectral_model():
    points, centers = spectral_points_and_centers()
    return KMeans.from_centroids(centers).compute_pcs(points, threshold=0.3)


def test_cattell_compact_storage_and_recomputation_after_loading(tmp_path):
    points, centers = spectral_points_and_centers()
    model = KMeans.from_centroids(centers)
    assert model.compute_pcs(points, threshold=0.3) is model
    assert model.component_ranks.tolist() == [3, 1]
    assert model.q == 3
    assert model.W.shape == (2, 4, 3)
    assert model.rank_mask.tolist() == [[True, True, True], [True, False, False]]
    assert not model.W[1, :, 1:].any()
    for k, rank in enumerate((3, 1)):
        expected = torch.diag(torch.tensor([1.] * rank + [0.] * (4 - rank), dtype=torch.float64))
        torch.testing.assert_close(model.W[k] @ model.W[k].T, expected)
    torch.testing.assert_close(model.W.transpose(1, 2) @ model.W, torch.diag_embed(model.rank_mask.double()))
    # Slicing must release the full basis rather than retain its backing storage.
    assert model.W.untyped_storage().nbytes() == model.W.numel() * model.W.element_size()
    original = model.W.clone()
    spectrum = model.eigenvalues.clone()
    path = tmp_path / "kmeans.pt"
    save_kmeans(model, path)
    restored = load_kmeans(path)
    assert restored.q == 3
    torch.testing.assert_close(restored.W, original, rtol=0, atol=0)
    torch.testing.assert_close(restored.eigenvalues, spectrum, rtol=0, atol=0)
    restored.compute_pcs(points, threshold=0.5)
    assert restored.component_ranks.tolist() == [1, 1]
    assert restored.W.shape == (2, 4, 1)
    save_kmeans(restored, path)
    restored = load_kmeans(path)
    assert restored.q == 1
    # Recovering discarded directions requires the original training points.
    restored.compute_pcs(points, threshold=0.3)
    torch.testing.assert_close(restored.W, original, rtol=0, atol=0)
    torch.testing.assert_close(restored.mu, centers, rtol=0, atol=0)


def test_cattell_ranks_are_monotone_and_zero_covariance_selects_one():
    points, centers = spectral_points_and_centers()
    model = KMeans.from_centroids(centers)
    previous = torch.full((model.K,), model.D)
    for threshold in (0.0, 0.1, 0.2, 0.3, 0.4, 1.0):
        model.compute_pcs(points, threshold=threshold)
        assert torch.all(model.component_ranks <= previous)
        assert model.q == int(model.component_ranks.max())
        previous = model.component_ranks.clone()
    model.compute_pcs(centers.repeat_interleave(2, dim=0), threshold=0)
    torch.testing.assert_close(model.component_ranks, torch.ones(2, dtype=torch.long))
    assert not model.eigenvalues.any()


def test_full_pca_selects_rank_from_all_gaps_without_a_configured_capacity():
    values = torch.tensor([10., 8., 5., 1.], dtype=torch.float64)
    axes = torch.diag((4 * values).sqrt())
    points = torch.cat((axes, -axes))
    model = KMeans.from_centroids(torch.zeros(1, 4, dtype=torch.float64))
    model.compute_pcs(points, threshold=0.3)
    assert model.D == 4
    assert model.q == 3
    assert model.component_ranks.tolist() == [3]
    torch.testing.assert_close(model.eigenvalues[0], values)
    torch.testing.assert_close(model.W[0] @ model.W[0].T, torch.diag(torch.tensor([1., 1., 1., 0.], dtype=torch.float64)))


@pytest.mark.parametrize("threshold", [None, -0.01, 1.01, float("nan"), float("inf")])
def test_invalid_surgery_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        points, centers = spectral_points_and_centers()
        KMeans.from_centroids(centers).compute_pcs(points, threshold=threshold)


def test_one_dimensional_pca_and_rank_selection():
    model = KMeans.from_centroids(torch.zeros(1, 1)).compute_pcs(
        torch.tensor([[-1.0], [1.0]]), threshold=0,
    )
    torch.testing.assert_close(model.component_ranks, torch.tensor([1]))


def test_undersized_clusters_and_invalid_neighborhoods_fail():
    X = torch.tensor([[-1.0, 0], [1.0, 0], [0.0, 1]])
    model = KMeans.from_centroids(torch.tensor([[0.0, 0], [10.0, 10]]))
    model.compute_pcs(X)
    assert model.pca_valid.tolist() == [True, False]
    assert model.component_ranks.tolist() == [1, 0]
    assert not model.W[1].any()
    for neighbors in (0, 1, 4):
        with pytest.raises(ValueError, match="neighbors"):
            model.compute_init_pcs(X, rank=1, neighbors=neighbors)
    with pytest.raises(TypeError, match="rank"):
        model.compute_pcs(X, rank=1)
    with pytest.raises(TypeError, match="method"):
        model.compute_pcs(X, method="unknown")
    assert model.q == 1
    assert model.q_init == 0


def test_refitting_invalidates_all_pca_state():
    X, centers = points_and_centers()
    model = KMeans.from_centroids(centers, max_iter=5, restarts=1)
    for attr in ("W", "component_ranks", "eigenvalues"):
        with pytest.raises(RuntimeError, match="compute_pcs"):
            getattr(model, attr)
    model.compute_pcs(X, threshold=1)
    model.compute_init_pcs(X, rank=3, neighbors=12)
    model.fit(X)
    assert model.q == 0
    assert model.q_init == 0
    assert model.init_pca_neighbors is None
    with pytest.raises(RuntimeError, match="compute_init_pcs"):
        _ = model.W_init
    assert model.surgery_threshold is None
    with pytest.raises(RuntimeError, match="compute_pcs"):
        _ = model.W


@pytest.mark.parametrize("method", [None, "cluster", "knn", "both"])
def test_checkpoint_round_trip_and_dtype(tmp_path, method):
    X, centers = points_and_centers()
    model = KMeans.from_centroids(centers, block_x=13, seed=11)
    model.fit_metadata = {"n_samples": 80, "inertia": 18.0}
    if method in ("cluster", "both"):
        model.compute_pcs(X, threshold=1)
    if method in ("knn", "both"):
        model.compute_init_pcs(X, rank=3, neighbors=12)
    path = tmp_path / "kmeans_model.pt"
    save_kmeans(model, path, extra={"split": "train", "seed": 4})
    restored = load_kmeans(path, map_location="cpu")
    assert restored.q == model.q
    assert restored.fit_metadata == model.fit_metadata
    assert restored.q_init == model.q_init
    assert restored.init_pca_neighbors == model.init_pca_neighbors
    assert restored.surgery_threshold == model.surgery_threshold
    assert restored.checkpoint_extra == {"split": "train", "seed": 4}
    assert not list(restored.parameters())
    assert restored.block_x == 13
    assert restored.seed == 11
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(restored.predict(X), model.predict(X))
    converted = load_kmeans(path, device="cpu", dtype=torch.float32)
    assert converted.mu.dtype == torch.float32
    assert converted._component_ranks.dtype == torch.long
    if method is None:
        with pytest.raises(RuntimeError, match="compute_pcs"):
            _ = restored.W
    payload = torch.load(path, weights_only=True)
    assert payload["meta"]["version"] == 3
    assert payload["meta"]["model_type"] == "kmeans"
    assert set(payload["state_dict"]) == {"mu", "_pcs", "_eigenvalues", "_component_ranks", "_init_pcs", "_pca_valid", "_cluster_counts"}


def test_legacy_centroids_and_unknown_versions_are_rejected(tmp_path):
    path = tmp_path / "centroids.pt"
    for legacy in (torch.zeros(2, 3), {"centroids": torch.zeros(2, 3)}):
        torch.save(legacy, path)
        with pytest.raises(ValueError, match="legacy centroid"):
            load_kmeans(path)
    save_kmeans(KMeans.from_centroids(torch.zeros(2, 3)), path)
    payload = torch.load(path, weights_only=True)
    payload["meta"]["version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="version"):
        load_kmeans(path)


def test_unfitted_and_invalid_inputs(tmp_path):
    model = KMeans(2, device="cpu")
    with pytest.raises(RuntimeError, match="fit KMeans"):
        model.predict(torch.zeros(3, 2))
    with pytest.raises(RuntimeError, match="fit KMeans"):
        save_kmeans(model, tmp_path / "kmeans_model.pt")
    with pytest.raises(ValueError, match="exceeds"):
        model.fit(torch.ones(1, 2))
    with pytest.raises(ValueError, match="finite"):
        model.fit(torch.full((3, 2), float("nan")))
    with pytest.raises(ValueError, match="nonempty"):
        KMeans.from_centroids(torch.empty(2, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("method", ["cluster", "knn"])
def test_cuda_model_device_round_trip(tmp_path, method):
    X, centers = points_and_centers()
    model = KMeans.from_centroids(centers, device="cuda")
    if method == "cluster":
        model.compute_pcs(X)
        assert model.W.is_cuda and model.eigenvalues.is_cuda and model.pca_valid.is_cuda
    else:
        model.compute_init_pcs(X, rank=2, neighbors=12)
        assert model.W_init.is_cuda
    assert model.predict(X).is_cuda
    path = tmp_path / "kmeans_model.pt"
    save_kmeans(model, path)
    restored = load_kmeans(path, device="cuda")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    torch.testing.assert_close(restored.predict(X), model.predict(X))


@pytest.mark.parametrize("corruption", ["pc_nan", "pc_scale", "spectrum_nan", "spectrum_negative", "spectrum_unsorted", "bad_rank"])
def test_malformed_numerical_checkpoint_buffers_are_rejected(tmp_path, corruption):
    model = spectral_model()
    path = tmp_path / "kmeans_model.pt"
    save_kmeans(model, path)
    payload = torch.load(path, weights_only=True)
    state = payload["state_dict"]
    if corruption == "pc_nan":
        state["_pcs"][0, 0, 0] = float("nan")
    elif corruption == "pc_scale":
        state["_pcs"][0, :, 0] *= 2
    elif corruption == "spectrum_nan":
        state["_eigenvalues"][0, 0] = float("nan")
    elif corruption == "spectrum_negative":
        state["_eigenvalues"][0, -1] = -0.1
    elif corruption == "spectrum_unsorted":
        state["_eigenvalues"][0, -1] = 12
    else:
        state["_component_ranks"][0] = model.q + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint"):
        load_kmeans(path)


def test_centroid_only_checkpoint_rejects_nonzero_component_ranks(tmp_path):
    path = tmp_path / "kmeans_model.pt"
    save_kmeans(KMeans.from_centroids(torch.zeros(2, 3)), path)
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["_component_ranks"][0] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="component ranks"):
        load_kmeans(path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectrum_rounding_tolerance_is_relative_to_covariance_scale(tmp_path, dtype):
    model = spectral_model().to(dtype=dtype)
    path = tmp_path / "kmeans_model.pt"
    save_kmeans(model, path)
    payload = torch.load(path, weights_only=True)
    values = payload["state_dict"]["_eigenvalues"]
    eps = torch.finfo(dtype).eps
    values[:] = torch.tensor([10.0, 1.0, -eps, -eps], dtype=dtype)
    torch.save(payload, path)
    restored = load_kmeans(path)
    torch.testing.assert_close(restored.eigenvalues, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_loading_validates_pcs_independently_of_tf32_setting(tmp_path):
    points = torch.randn(128, 64, generator=torch.Generator().manual_seed(9))
    model = KMeans.from_centroids(points.mean(0, keepdim=True)).compute_pcs(points)
    path = tmp_path / "kmeans_model.pt"
    save_kmeans(model, path)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        restored = load_kmeans(path, map_location="cuda")
        torch.testing.assert_close(restored.W.cpu(), model.W)
    finally:
        torch.set_float32_matmul_precision(previous)


def test_sparse_boundaries_and_independent_states(tmp_path):
    centers = torch.tensor([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]])
    points = torch.tensor([[0., -1., 0.], [0., 1., 0.],
                           [10., -1., 0.], [10., 1., 0.], [10., 0., 1.]])
    model = KMeans.from_centroids(centers).compute_init_pcs(points, rank=3, neighbors=5)
    initialization = model.W_init.clone()
    predicted = model.predict(points)
    model.compute_pcs(points)
    assert model.cluster_counts.tolist() == [2, 3, 0]
    assert model.pca_valid.tolist() == [True, True, False]
    assert model.component_ranks.tolist() == [1, 2, 0]
    assert not model.W[~model.pca_valid].any()
    assert not model.eigenvalues[~model.pca_valid].any()
    torch.testing.assert_close(model.W_init, initialization, rtol=0, atol=0)
    model.compute_pcs(points, threshold=.9)
    assert model.component_ranks.tolist() == [1, 1, 0]
    assert not model.W[:, :, 1:].any()
    geometry = model.W.clone()
    torch.testing.assert_close(model.W_init, initialization, rtol=0, atol=0)
    model.compute_init_pcs(points, rank=1, neighbors=4)
    torch.testing.assert_close(model.W, geometry, rtol=0, atol=0)
    torch.testing.assert_close(model.predict(points), predicted)
    path = tmp_path / 'sparse.pt'
    save_kmeans(model, path)
    restored = load_kmeans(path)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    model.compute_pcs(points)
    assert model.component_ranks.tolist() == [1, 2, 0]


def test_all_sparse_skips_eigendecomposition(tmp_path, monkeypatch):
    points = torch.eye(3)
    model = KMeans.from_centroids(points)
    monkeypatch.setattr(torch.linalg, 'eigh', lambda *a, **kw: pytest.fail('sparse clusters need no PCA'))
    model.compute_pcs(points, threshold=0)
    assert model.q == 1
    assert model.W.shape == (3, 3, 1)
    assert not model.rank_mask.any()
    assert not model.pca_valid.any()
    assert not model.component_ranks.any()
    assert not model.W.any()
    assert not model.eigenvalues.any()
    path = tmp_path / "all_sparse.pt"
    save_kmeans(model, path)
    restored = load_kmeans(path)
    assert restored.q == 1
    assert not restored.pca_valid.any()
    torch.testing.assert_close(restored.W, model.W, rtol=0, atol=0)


@pytest.mark.parametrize('field', ['_pca_valid', '_cluster_counts', '_init_pcs'])
def test_checkpoint_rejects_inconsistent_pca_state(tmp_path, field):
    X, centers = points_and_centers()
    model = KMeans.from_centroids(centers).compute_pcs(X).compute_init_pcs(X, rank=3, neighbors=12)
    path = tmp_path / 'bad.pt'
    save_kmeans(model, path)
    payload = torch.load(path, weights_only=True)
    value = payload['state_dict'][field]
    value[0] = False if field == '_pca_valid' else 0
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_kmeans(path)


@pytest.mark.parametrize("version", [1, 2])
def test_previous_checkpoint_format_requires_regeneration(tmp_path, version):
    path = tmp_path / 'v1.pt'
    torch.save({'format': f'dalg_kmeans_v{version}'}, path)
    with pytest.raises(ValueError, match='regenerate'):
        load_kmeans(path)
