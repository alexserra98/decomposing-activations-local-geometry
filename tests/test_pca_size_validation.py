from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from scripts.pca_size_validation import run_validation
from scripts.pca_size_validation.run_validation import (
    compare_pc_bases,
    compute_cluster_metrics,
    select_clusters,
    summarize_spectrum,
)


def test_select_clusters_is_deterministic_and_spans_population_rank() -> None:
    kmeans_sizes = torch.arange(100, 2100, 100)
    mfa_sizes = kmeans_sizes + 50

    first = select_clusters(
        kmeans_sizes, mfa_sizes, num_clusters=4, min_size=100, seed=7
    )
    second = select_clusters(
        kmeans_sizes, mfa_sizes, num_clusters=4, min_size=100, seed=7
    )

    assert torch.equal(first, second)
    assert len(torch.unique(first)) == 4
    population_ranks = torch.argsort(kmeans_sizes)[first]
    assert population_ranks[0] < 5
    assert 5 <= population_ranks[1] < 10
    assert 10 <= population_ranks[2] < 15
    assert 15 <= population_ranks[3] < 20


def test_spectrum_normalization_maps_rank_one_to_zero() -> None:
    D = 8
    n = 20
    variances = torch.zeros(D)
    variances[0] = 1.0

    participation_ratio, isotropy = summarize_spectrum(
        variances, sample_size=n, ambient_dim=D
    )

    assert participation_ratio == 1.0
    assert isotropy == 0.0


def test_restore_random_order_recovers_nested_sampling_priority() -> None:
    from scripts.pca_size_validation.run_validation import _restore_random_order

    buffers = [None, torch.tensor([[10.0], [20.0], [30.0]])]
    positions = {1: torch.tensor([30, 10, 20])}

    restored = _restore_random_order(buffers, positions)

    assert torch.equal(restored[1], torch.tensor([[30.0], [10.0], [20.0]]))


def test_pc_basis_comparison_handles_signs_and_orthogonal_spaces() -> None:
    identity = torch.eye(6)
    signed = identity.clone()
    signed[0] *= -1
    same = compare_pc_bases(identity[:3], signed[:3])
    orthogonal = compare_pc_bases(identity[:3], identity[3:])

    assert same["pc_mean_cos2"] == 1.0
    assert same["pc_median_angle_deg"] == 0.0
    assert orthogonal["pc_mean_cos2"] == 0.0
    assert orthogonal["pc_median_angle_deg"] == 90.0


def test_compute_cluster_metrics_uses_intrinsic_dim_module_top_pcs() -> None:
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(64, 12, generator=generator)

    result = compute_cluster_metrics(
        X, threshold=0.9, pca_device="cpu", top_pcs=5
    )

    assert 1 <= result["intrinsic_dim"] <= 12
    assert result["variances"].shape == (12,)
    assert result["components"].shape == (5, 12)
    assert result["participation_ratio"] > 1.0
    assert torch.isfinite(torch.tensor(result["sample_corrected_isotropy"]))


def test_validation_runner_smoke(tmp_path: Path, monkeypatch) -> None:
    from tests.synthetic_shards import build_multi_shard

    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=4)
    assignments = torch.arange(24) % 2
    bundle = {
        "assignments": assignments,
        "cluster_sizes": torch.bincount(assignments, minlength=2),
        "K": 2,
        "source": {"layer": 5, "drop_prefix": 0},
    }
    kmeans_assignments = tmp_path / "kmeans_assignments.pt"
    mfa_assignments = tmp_path / "mfa_assignments.pt"
    torch.save(bundle, kmeans_assignments)
    torch.save(bundle, mfa_assignments)

    centroids = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    kmeans_centroids = tmp_path / "kmeans_centroids.pt"
    mfa_centroids = tmp_path / "mfa_centroids.pt"
    torch.save(centroids, kmeans_centroids)
    torch.save(centroids, mfa_centroids)
    out_dir = tmp_path / "output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_validation.py",
            "--layer",
            "5",
            "--K",
            "2",
            "--rank",
            "1",
            "--num-clusters",
            "2",
            "--sample-sizes",
            "6",
            "8",
            "--top-pcs",
            "1",
            "--pca-device",
            "cpu",
            "--shard-dir",
            str(shard_dir),
            "--kmeans-assignments",
            str(kmeans_assignments),
            "--mfa-assignments",
            str(mfa_assignments),
            "--kmeans-centroids",
            str(kmeans_centroids),
            "--mfa-init-centroids",
            str(mfa_centroids),
            "--output-dir",
            str(out_dir),
        ],
    )

    run_validation.main()

    # The explicit overwrite path must support rerunning the same canonical output.
    sys.argv.append("--overwrite")
    run_validation.main()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (out_dir / "convergence_summary.csv").exists()
    for partition in ("kmeans", "mfa_responsibility"):
        for cap in (6, 8):
            cap_dir = out_dir / partition / f"n{cap}"
            result = torch.load(
                cap_dir / "intrinsic_dims.pt", map_location="cpu", weights_only=True
            )
            pcs = torch.load(
                cap_dir / "cluster_top_pcs.pt", map_location="cpu", weights_only=True
            )
            assert result["selected_clusters"].numel() == 2
            assert all(component.shape == (1, 2) for component in pcs["cluster_top_pcs"])
