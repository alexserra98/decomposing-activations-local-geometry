"""End-to-end tile-alignment checks on Gaussian collections with known geometry."""

import json

import pytest
import torch

from dalg.evaluation.tile_alignment import evaluate_tile_alignment, tile_alignment_report_valid
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC
from dalg.models.kmeans import KMeans


def _gaussians(kind, *, orthogonal):
    count, dimension, rank = 3, 7, 2
    centroids = torch.zeros(count, dimension, dtype=torch.float64)
    centroids[:, -1] = torch.tensor([0.0, 3.0, 9.0])
    directions = torch.stack([
        torch.eye(dimension, dtype=torch.float64)[:, 2 * i:2 * i + rank]
        if orthogonal else torch.eye(dimension, dtype=torch.float64)[:, :rank]
        for i in range(count)
    ])
    # Each Gaussian has two separated signal variances and isotropic noise.
    signal_variances = torch.tensor([4.0, 2.0], dtype=torch.float64)
    if kind == "hddc":
        model = MFA_HDDC(
            centroids, rank=rank, init_directions=directions, psi_init=0.1, shared_b=True,
        )
        with torch.no_grad():
            model.scale_rho.copy_(signal_variances.sqrt().expm1().log().expand(count, -1))
    else:
        # Store the exact population PCA of the same Gaussian collection.
        model = KMeans.from_centroids(centroids)
        model._pcs = directions
        model._eigenvalues = torch.full((count, dimension), 0.1, dtype=torch.float64)
        model._eigenvalues[:, :rank] += signal_variances
        model._component_ranks.fill_(rank)
        model._pca_valid.fill_(True)
        model._cluster_counts.fill_(100)
    return model


def _check_report(model, expected):
    report = evaluate_tile_alignment(model, torch.ones(model.K, dtype=torch.bool))
    # Exercise the saved JSON contract as well as the complete numerical metric.
    report = json.loads(json.dumps(report, allow_nan=False))
    assert tile_alignment_report_valid(report, components=model.K, neighbors=5)
    assert report["eligible_components"] == 3
    assert report["excluded_components"] == 0
    assert report["effective_neighbors"] == 2
    assert len(report["comparisons"]) == 6
    assert {(edge["source"], edge["target"]) for edge in report["comparisons"]} == {
        (i, j) for i in range(3) for j in range(3) if i != j
    }
    for name in ("leading_pc_alignment", "best_subset_pc_alignment"):
        assert report[name]["mean"] == pytest.approx(expected, abs=1e-12)
        for tile in report["per_tile"]:
            assert tile[name] == pytest.approx(expected, abs=1e-12)
        for edge in report["comparisons"]:
            assert edge[name] == pytest.approx(expected, abs=1e-12)
            assert edge["source_rank"] == edge["target_rank"] == 2
            assert edge["distance"] == pytest.approx(float(torch.linalg.vector_norm(
                model.mu.detach()[edge["source"]] - model.mu.detach()[edge["target"]],
            )))
    for source in range(3):
        distances = [edge["distance"] for edge in report["comparisons"] if edge["source"] == source]
        assert distances == sorted(distances)


@pytest.mark.parametrize("kind", ["hddc", "kmeans"])
def test_aligned_gaussian_tiles(kind):
    _check_report(_gaussians(kind, orthogonal=False), 1.0)


@pytest.mark.parametrize("kind", ["hddc", "kmeans"])
def test_orthogonal_gaussian_tiles(kind):
    _check_report(_gaussians(kind, orthogonal=True), 0.0)
