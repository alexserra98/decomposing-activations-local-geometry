from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from dalg.init.centroid_artifact import (
    compute_cluster_pca_directions,
    load_centroid_artifact,
    save_centroid_artifact,
    validate_centroid_artifact,
)
from dalg.models.adaptive_q.mfa_ard import MFA_ARD
from dalg.models.adaptive_q.mfa_hddc import (
    ComponentShardedMFA_HDDC,
    MFA_HDDC,
)
from dalg.models.mfa import ComponentShardedMFA, MFA
from scripts.temporary.build_toy_kmeans_centroids import build_centroids


def _orthonormal_directions(K: int, D: int, q: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    out = []
    for _ in range(K):
        matrix = torch.randn(D, q, generator=generator)
        out.append(torch.linalg.qr(matrix, mode="reduced").Q)
    return torch.stack(out)


def test_exact_cluster_pca_recovers_planted_subspaces() -> None:
    generator = torch.Generator().manual_seed(3)
    n = 512
    centroids = torch.tensor(
        [[10.0, -4.0, 2.0, 1.0], [-3.0, 8.0, 5.0, -2.0]],
        dtype=torch.float64,
    )
    bases = torch.stack(
        [
            torch.eye(4, dtype=torch.float64)[:, [0, 1]],
            torch.eye(4, dtype=torch.float64)[:, [2, 3]],
        ]
    )
    points = []
    assignments = []
    for cluster in range(2):
        scores = torch.randn(n, 2, generator=generator, dtype=torch.float64)
        scores *= torch.tensor([4.0, 1.5], dtype=torch.float64)
        points.append(centroids[cluster] + scores @ bases[cluster].T)
        assignments.append(torch.full((n,), cluster, dtype=torch.long))

    directions = compute_cluster_pca_directions(
        torch.cat(points),
        torch.cat(assignments),
        centroids,
        rank=2,
        chunk_elems=128,
        eig_batch_size=1,
    )

    assert directions.shape == (2, 4, 2)
    gram = directions.transpose(1, 2) @ directions
    assert torch.allclose(
        gram,
        torch.eye(2, dtype=torch.float64).expand(2, 2, 2),
        atol=1e-10,
        rtol=0.0,
    )
    recovered_projectors = directions @ directions.transpose(1, 2)
    planted_projectors = bases @ bases.transpose(1, 2)
    assert torch.allclose(recovered_projectors, planted_projectors, atol=1e-10, rtol=0.0)
    leading_alignment = torch.abs(
        torch.einsum("kd,kd->k", directions[:, :, 0], bases[:, :, 0])
    )
    assert torch.all(leading_alignment > 0.98)


def test_cluster_pca_rejects_clusters_without_rank_plus_one_points() -> None:
    points = torch.randn(5, 4)
    assignments = torch.tensor([0, 0, 0, 1, 1])
    centroids = torch.zeros(2, 4)

    with pytest.raises(ValueError, match=r"rank\+1=3"):
        compute_cluster_pca_directions(
            points,
            assignments,
            centroids,
            rank=2,
        )


def test_centroid_artifact_supports_legacy_and_enriched_files(tmp_path: Path) -> None:
    centroids = torch.randn(3, 5)
    directions = _orthonormal_directions(3, 5, 2)
    legacy_path = tmp_path / "legacy.pt"
    bundle_path = tmp_path / "bundle.pt"
    torch.save(centroids, legacy_path)
    save_centroid_artifact(bundle_path, centroids, directions)

    legacy_centroids, legacy_directions = load_centroid_artifact(legacy_path)
    bundle_centroids, bundle_directions = load_centroid_artifact(bundle_path)

    assert torch.equal(legacy_centroids, centroids)
    assert legacy_directions is None
    assert torch.equal(bundle_centroids, centroids)
    assert torch.equal(bundle_directions, directions)
    validate_centroid_artifact(
        bundle_centroids,
        bundle_directions,
        expected_k=3,
        expected_d=5,
        required_pca_rank=2,
    )
    with pytest.raises(ValueError, match="stores 2 principal components"):
        validate_centroid_artifact(
            bundle_centroids,
            bundle_directions,
            required_pca_rank=3,
        )


def test_initial_directions_are_used_by_every_model_variant() -> None:
    K, D, q = 4, 6, 2
    centroids = torch.randn(K, D)
    directions = _orthonormal_directions(K, D, q)

    models = [
        MFA(centroids, rank=q, init_directions=directions),
        MFA_ARD(centroids, rank=q, init_directions=directions),
        MFA_HDDC(
            centroids,
            rank=q,
            init_directions=directions,
            isotropic_psi=True,
        ),
    ]
    for model in models:
        assert torch.equal(model.dir_raw.detach(), directions)
        assert torch.allclose(model._scale(), torch.ones(K, q), atol=1e-6, rtol=0.0)

    shard = ComponentShardedMFA.from_global_centroids(
        centroids,
        rank=q,
        dist_rank=1,
        world_size=2,
        init_directions=directions,
    )
    hddc_shard = ComponentShardedMFA_HDDC.from_global_centroids(
        centroids,
        rank=q,
        dist_rank=1,
        world_size=2,
        init_directions=directions,
        isotropic_psi=True,
    )
    assert shard.component_start == 2
    assert torch.equal(shard.dir_raw.detach(), directions[2:])
    assert torch.equal(hddc_shard.dir_raw.detach(), directions[2:])


def test_model_rejects_wrong_initial_direction_shape() -> None:
    with pytest.raises(ValueError, match="init_directions must have shape"):
        MFA(torch.zeros(2, 4), rank=2, init_directions=torch.zeros(2, 4, 3))


def test_toy_builder_pca_only_upgrades_existing_centroids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_dir = tmp_path / "shards"
    output_dir = tmp_path / "centroids"
    shard_dir.mkdir()
    output_dir.mkdir()
    (shard_dir / "config.json").write_text(json.dumps({"d_model": 3}))

    centroids = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    offsets = torch.tensor(
        [
            [-2.0, -1.0, 0.0],
            [-1.0, 2.0, 0.0],
            [1.0, -2.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    points = torch.cat([centroids[0] + offsets, centroids[1] + offsets])
    torch.save(centroids, output_dir / "centroids.pt")
    (output_dir / "config.json").write_text(
        json.dumps({"inertia": 2.5, "cluster_sizes": [4, 4]})
    )
    source_config = {
        "layers": [0],
        "num_rows": 8,
        "d_model": 3,
        "window": 1,
        "drop_prefix": 0,
    }
    monkeypatch.setattr(
        "scripts.temporary.build_toy_kmeans_centroids._load_activations",
        lambda *_args, **_kwargs: (points, source_config),
    )
    args = argparse.Namespace(
        shard_dir=shard_dir,
        layer=0,
        K=2,
        out_dir=output_dir,
        max_iter=10,
        restarts=1,
        tol=1e-6,
        seed=0,
        device="cpu",
        load_batch_size=16,
        block_x=16,
        block_c=16,
        pca_rank=2,
        pca_only=True,
        pca_chunk_elems=32,
        pca_eig_batch_size=1,
    )

    build_centroids(args)
    saved_centroids, saved_directions = load_centroid_artifact(
        output_dir / "centroids.pt"
    )
    metadata = json.loads((output_dir / "config.json").read_text())

    assert torch.equal(saved_centroids, centroids)
    assert saved_directions is not None
    assert saved_directions.shape == (2, 3, 2)
    assert metadata["principal_components"]["rank"] == 2
    assert metadata["principal_components"]["center"] == "stored_kmeans_centroid"

    # A second PCA-only invocation validates the stored rank and exits before
    # loading the activation dataset again.
    monkeypatch.setattr(
        "scripts.temporary.build_toy_kmeans_centroids._load_activations",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not load")),
    )
    build_centroids(args)
