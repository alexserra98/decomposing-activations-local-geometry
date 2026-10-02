from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dalg.analysis.cluster_assignments import compute_assignments
from dalg.cli.run_metrics import build_parser
from dalg.data.manifold_dataset import ToyManifoldConfig, save_toy_manifold_shards
from dalg.data.shard_activations import load_meta_index
from dalg.evaluation.toy_manifold_metrics import (
    _leading_covariance_eigenspaces,
    evaluate_toy_manifold_metrics,
)
from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling
from dalg.models.kmeans import KMeans, save_kmeans


def _build_kmeans_run(
    tmp_path: Path, method: str, cattell: bool, *, validation_stride: int | None = 4,
):
    shard_dir = save_toy_manifold_shards(
        tmp_path / "shards",
        ToyManifoldConfig(
            ambient_dim=4,
            n_samples=120,
            calibration_size=40,
            manifold_types=("segment",),
            manifolds_per_type=2,
            offset_radius=6.0,
            noise_ratio=None,
            seed=12,
        ),
        shard_size=30,
        layer=0,
    )
    points = torch.cat([
        torch.load(path, weights_only=True).reshape(-1, 4)
        for path in sorted((shard_dir / "layer00").glob("shard_*.pt"))
    ])
    meta_index = load_meta_index(shard_dir, layer=0)
    val_positions = list(range(0, len(meta_index), validation_stride)) if validation_stride else []
    validation = torch.zeros(len(points), dtype=torch.bool)
    validation[val_positions] = True
    model = KMeans(2, restarts=1, max_iter=15, seed=0, device="cpu")
    model.fit(points[~validation]).compute_pcs(
        points[~validation], threshold=0.1 if cattell else 1.0,
    )
    if method == "knn":
        model.compute_init_pcs(points[~validation], rank=3, neighbors=12)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_kmeans(model, run_dir / "kmeans_model.pt")
    (run_dir / "config.json").write_text(json.dumps({"model_kind": "kmeans"}))
    (run_dir / "val_indices.json").write_text(json.dumps({
        "train_rows": int((~validation).sum()),
        "val_rows": int(validation.sum()),
        "val_global_rows": [meta_index[p]["global_row"] for p in val_positions],
    }))
    args = build_parser().parse_args([
        "assignments", "--data-dir", str(run_dir), "--model-type", "kmeans",
        "--shard-dir", str(shard_dir), "--layer", "0", "--batch-size", "17",
        "--device", "cpu",
    ])
    args.func(args)
    return run_dir, shard_dir, model, points, validation


@pytest.mark.parametrize("method", ["cluster", "knn"])
@pytest.mark.parametrize("cattell", [False, True])
def test_kmeans_assignments_and_toy_evaluation(tmp_path: Path, method: str, cattell: bool):
    run_dir, shard_dir, model, points, validation = _build_kmeans_run(tmp_path, method, cattell)
    bundle = torch.load(run_dir / "kmeans_model_assignments.pt", weights_only=True)
    expected = model.predict(points)
    assert torch.equal(bundle["assignments"], expected)
    assert torch.equal(bundle["cluster_sizes"], torch.bincount(expected, minlength=model.K))
    assert torch.equal(bundle["max_responsibilities"], torch.ones(len(points)))
    assert torch.equal(bundle["peakedness"]["entropy"], torch.zeros(model.K))
    assert torch.equal(bundle["peakedness"]["one_minus_max"], torch.zeros(model.K))
    assert torch.equal(bundle["peakedness"]["top1_minus_top2"], torch.ones(model.K))
    assert bundle["model_type"] == "kmeans"
    assert bundle["model_path"] == str(run_dir / "kmeans_model.pt")
    assert bundle["source"] == {
        "shard_dir": str(shard_dir), "layer": 0, "drop_prefix": 0, "num_items": len(points),
    }
    assert not (run_dir / "centroids.pt").exists()

    metrics = evaluate_toy_manifold_tiling(
        run_dir, shard_dir=shard_dir, layer=0, model_kind="kmeans",
        batch_size=19, device="cpu", rank_threshold=-10.0,
    )
    assert metrics["schema_version"] == 2
    assert metrics["model_kind"] == "kmeans"
    assert "nll" not in metrics and "bic" not in metrics
    assert metrics["quantization"]["convention"] == "lower_is_better"
    for name, mask in (("train", ~validation), ("validation", validation)):
        errors = (points[mask].double() - model.mu[expected[mask]].double()).square().sum(1)
        result = metrics["quantization"][name]
        assert result["sum_squared_distance"] == pytest.approx(float(errors.sum()))
        assert result["mean_squared_distance"] == pytest.approx(float(errors.mean()))
        assert result["n"] == int(mask.sum())
    assert metrics["rank"]["definition"] == "kmeans_component_ranks"
    assert "threshold" not in metrics["rank"]
    assert metrics["rank"]["mean_learned"] == pytest.approx(float(model.component_ranks.float().mean()))
    assert metrics["association"]["associated_components"] == model.K
    assert metrics["tangent_alignment"]["subspace_overlap"]["mean"] == pytest.approx(1.0, abs=1e-5)
    if cattell:
        assert torch.equal(model.component_ranks, torch.ones(model.K, dtype=torch.long))
        assert metrics["tangent_containment"]["subspace_overlap"]["mean"] == pytest.approx(1.0, abs=1e-5)


def test_kmeans_assignment_stream_uses_integer_predictions(tmp_path: Path, monkeypatch):
    model = KMeans.from_centroids(torch.tensor([[0.0, 0.0], [2.0, 0.0], [100.0, 0.0]]))
    path = tmp_path / "kmeans_model.pt"
    save_kmeans(model, path)

    def dense_responsibilities_forbidden(*args, **kwargs):
        raise AssertionError("streaming assignments must not allocate a dense one-hot matrix")

    monkeypatch.setattr(KMeans, "responsibilities", dense_responsibilities_forbidden)
    sizes, assignments, confidence, peakedness = compute_assignments(
        path,
        [torch.tensor([[0.0, 0.0], [1.0, 0.0]]), (torch.tensor([[3.0, 0.0]]),)],
        model_type="kmeans", device="cpu", use_inference_cache=True,
    )
    assert assignments.tolist() == [0, 0, 1]
    assert sizes.tolist() == [2, 1, 0]
    assert confidence.tolist() == [1.0, 1.0, 1.0]
    assert peakedness["top1_minus_top2"].tolist() == [1.0, 1.0, 0.0]


def test_kmeans_geometry_uses_masked_pcs(tmp_path: Path):
    run_dir, shard_dir, model, points, validation = _build_kmeans_run(tmp_path, "cluster", False)
    values, vectors = _leading_covariance_eigenspaces(model, model.q)
    assert torch.equal(values, model.eigenvalues[:, :model.q + 1].double())
    assert torch.equal(vectors, model.W.double())
    assert model.q == int(model.component_ranks.max())

    disk_dir = save_toy_manifold_shards(
        tmp_path / "disk",
        ToyManifoldConfig(
            ambient_dim=3, n_samples=20, calibration_size=20,
            manifold_types=("flat_disk",), manifolds_per_type=1, noise_ratio=None,
        ),
        layer=0,
    )
    metadata = torch.load(disk_dir / "manifold_metadata.pt", weights_only=True)
    disk_points = torch.load(disk_dir / "layer00/shard_00000.pt", weights_only=True).reshape(-1, 3)
    disk_model = KMeans(1, restarts=1, device="cpu").fit(disk_points).compute_pcs(disk_points, threshold=1)
    result = evaluate_toy_manifold_metrics(disk_model, metadata, torch.ones(1, dtype=torch.bool))
    assert disk_model.component_ranks.tolist() == [1]
    assert result["rank"]["components"] == 1
    for summary in (result, result["per_manifold"][0]):
        for metric_name in ("tangent_alignment", "tangent_containment"):
            for score_name in ("subspace_overlap", "worst_direction_cosine"):
                assert summary[metric_name][score_name] == {
                    "mean": None, "valid_components": 0, "undefined_components": 1,
                }

    for summary in (result, result["per_manifold"][0]):
        assert summary["tangent_adjusted_alignment"]["subspace_overlap"] == {
            "mean": pytest.approx(0.5, abs=1e-6),
            "valid_components": 1, "undefined_components": 0,
        }
        for score_name in ("subspace_overlap", "worst_direction_cosine"):
            assert summary["tangent_partial_containment"][score_name] == {
                "mean": pytest.approx(1.0, abs=1e-6),
                "valid_components": 1, "undefined_components": 0,
            }

    # An undersized cluster cannot demand a tangent basis or be scored as rank zero.
    disk_model.compute_pcs(disk_points[:1])
    result = evaluate_toy_manifold_metrics(disk_model, metadata, torch.ones(1, dtype=torch.bool))
    assert result["association"]["associated_components"] == 1
    assert result["rank"]["components"] == 0
    assert result["pca_geometry"]["excluded_associated_components"] == 1
    assert result["tangent_alignment"]["subspace_overlap"]["mean"] is None
    for summary in (result, result["per_manifold"][0]):
        assert summary["tangent_adjusted_alignment"]["subspace_overlap"] == {
            "mean": None, "valid_components": 0, "undefined_components": 0,
        }
    for score_name in ("subspace_overlap", "worst_direction_cosine"):
        assert result["tangent_partial_containment"][score_name] == {
            "mean": None, "valid_components": 0, "undefined_components": 0,
        }


@pytest.mark.parametrize("source_only", [False, True])
def test_kmeans_evaluation_rejects_foreign_same_shape_assignments(tmp_path, source_only):
    run_dir, shard_dir, _, points, _ = _build_kmeans_run(tmp_path / "current", "cluster", False)
    foreign_dir, _, _, foreign_points, _ = _build_kmeans_run(tmp_path / "foreign", "cluster", False)
    assert points.shape == foreign_points.shape
    foreign_bundle = torch.load(foreign_dir / "kmeans_model_assignments.pt", weights_only=True)
    if source_only:
        foreign_bundle["model_path"] = str(run_dir / "kmeans_model.pt")
    torch.save(foreign_bundle, run_dir / "kmeans_model_assignments.pt")
    expected_error = "source.shard_dir" if source_only else "model_path"
    with pytest.raises(ValueError, match=expected_error):
        evaluate_toy_manifold_tiling(
            run_dir, shard_dir=shard_dir, layer=0, model_kind="kmeans", device="cpu",
        )


@pytest.mark.parametrize("field", ["assignments", "cluster_sizes"])
def test_kmeans_evaluation_rejects_fractional_assignment_values(tmp_path, field):
    run_dir, shard_dir, _, _, _ = _build_kmeans_run(tmp_path, "cluster", False)
    path = run_dir / "kmeans_model_assignments.pt"
    bundle = torch.load(path, weights_only=True)
    bundle[field] = bundle[field].float() + 0.25
    torch.save(bundle, path)
    with pytest.raises(ValueError, match=f"{field} must contain integer values"):
        evaluate_toy_manifold_tiling(
            run_dir, shard_dir=shard_dir, layer=0, model_kind="kmeans", device="cpu",
        )


@pytest.mark.parametrize("field", ["layer", "drop_prefix", "num_items"])
def test_kmeans_evaluation_rejects_wrong_source_stream(tmp_path, field):
    run_dir, shard_dir, _, _, _ = _build_kmeans_run(tmp_path, "cluster", False)
    path = run_dir / "kmeans_model_assignments.pt"
    bundle = torch.load(path, weights_only=True)
    bundle["source"][field] += 1
    torch.save(bundle, path)
    with pytest.raises(ValueError, match=f"source.{field}"):
        evaluate_toy_manifold_tiling(
            run_dir, shard_dir=shard_dir, layer=0, model_kind="kmeans", device="cpu",
        )



def test_kmeans_evaluation_without_validation_split(tmp_path):
    run_dir, shard_dir, model, points, _ = _build_kmeans_run(
        tmp_path, "cluster", False, validation_stride=None,
    )
    metrics = evaluate_toy_manifold_tiling(
        run_dir, shard_dir=shard_dir, layer=0, model_kind="kmeans", device="cpu",
    )
    assert metrics["quantization"]["validation"] == {
        "n": 0, "sum_squared_distance": 0.0, "mean_squared_distance": None,
    }
    errors = (points.double() - model.mu[model.predict(points)].double()).square().sum(1)
    assert metrics["quantization"]["train"]["n"] == len(points)
    assert metrics["quantization"]["train"]["sum_squared_distance"] == pytest.approx(float(errors.sum()))
    assert metrics["quantization"]["train"]["mean_squared_distance"] == pytest.approx(float(errors.mean()))
    assert metrics["dataset"]["validation_rows"] == 0


def test_kmeans_evaluation_requires_training_split(tmp_path):
    run_dir, shard_dir, _, points, _ = _build_kmeans_run(tmp_path, "cluster", False)
    meta_index = load_meta_index(shard_dir, layer=0)
    (run_dir / "val_indices.json").write_text(json.dumps({
        "train_rows": 0, "val_rows": len(points),
        "val_global_rows": [row["global_row"] for row in meta_index],
    }))
    with pytest.raises(ValueError, match="empty training split"):
        evaluate_toy_manifold_tiling(
            run_dir, shard_dir=shard_dir, layer=0, model_kind="kmeans", device="cpu",
        )
