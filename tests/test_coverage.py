from __future__ import annotations

import pytest
import torch

from dalg.evaluation.coverage import (
    evaluate_heldout_distribution_coverage,
    nearest_live_centroid_distances,
    split_fingerprint,
    stratified_three_way_split,
    summarize_coverage_distances,
)


def test_heldout_coverage_reports_distance_distribution_and_curve() -> None:
    points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    centroids = torch.tensor([[0.0], [3.0]])

    summary, distances = evaluate_heldout_distribution_coverage(
        points,
        centroids,
        thresholds=(0.0, 0.5, 1.0),
        batch_size=2,
    )

    torch.testing.assert_close(distances, torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["root_mean_square"] == pytest.approx(2.0**-0.5)
    assert summary["median"] == pytest.approx(0.5)
    assert summary["r95"] == pytest.approx(1.0)
    assert summary["r99"] == pytest.approx(1.0)
    assert summary["maximum"] == pytest.approx(1.0)
    assert summary["components_total"] == 2
    assert summary["components_live"] == 2
    assert summary["coverage_curve"] == [
        {"radius": 0.0, "fraction": 0.5},
        {"radius": 0.5, "fraction": 0.5},
        {"radius": 1.0, "fraction": 1.0},
    ]


def test_heldout_coverage_excludes_dead_centroids() -> None:
    points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    centroids = torch.tensor([[0.0], [3.0]])

    distances = nearest_live_centroid_distances(
        points,
        centroids,
        live_components=torch.tensor([True, False]),
        batch_size=3,
    )

    torch.testing.assert_close(distances, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_heldout_coverage_is_stable_for_nearby_high_offset_points() -> None:
    dimension = 128
    centroids = torch.full((2, dimension), 10_000.0, dtype=torch.float32)
    centroids[1] += 2.0
    points = centroids[:1].clone()
    points[0, :32] += 0.125

    distances = nearest_live_centroid_distances(points, centroids)
    expected = (points[:, None, :] - centroids[None, :, :]).norm(dim=2).min(dim=1).values

    torch.testing.assert_close(distances, expected, rtol=1e-6, atol=1e-6)
    assert distances.item() == pytest.approx(32**0.5 * 0.125)


@pytest.mark.parametrize(
    ("points", "centroids", "live", "match"),
    [
        (torch.empty(0, 2), torch.ones(1, 2), None, "reference point"),
        (torch.ones(1, 2), torch.empty(0, 2), None, "centroid"),
        (torch.ones(1, 2), torch.ones(1, 3), None, "ambient dimension"),
        (torch.ones(1, 2), torch.ones(2, 2), torch.tensor([False, False]), "live"),
    ],
)
def test_heldout_coverage_rejects_invalid_populations(
    points: torch.Tensor,
    centroids: torch.Tensor,
    live: torch.Tensor | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        nearest_live_centroid_distances(
            points,
            centroids,
            live_components=live,
        )


def test_heldout_coverage_requires_sorted_unique_thresholds() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        evaluate_heldout_distribution_coverage(
            torch.zeros(2, 1),
            torch.zeros(1, 1),
            thresholds=(0.2, 0.1),
        )


def test_tail_radius_is_smallest_observed_radius_reaching_target() -> None:
    summary = summarize_coverage_distances(torch.tensor([0.0, 10.0]))
    deciles = summarize_coverage_distances(torch.arange(1.0, 11.0))

    assert summary["median"] == pytest.approx(5.0)
    assert summary["r90"] == pytest.approx(10.0)
    assert summary["r95"] == pytest.approx(10.0)
    assert summary["r99"] == pytest.approx(10.0)
    assert deciles["r90"] == pytest.approx(9.0)


def test_three_way_split_is_fixed_disjoint_and_stratified() -> None:
    labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)
    first = stratified_three_way_split(
        labels,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=17,
    )
    second = stratified_three_way_split(
        labels,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=17,
    )

    assert all(torch.equal(first[name], second[name]) for name in first)
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 36,
        "validation": 12,
        "test": 12,
    }
    combined = torch.cat(list(first.values()))
    assert torch.equal(combined.sort().values, torch.arange(60))
    assert combined.unique().numel() == 60
    for rows in first.values():
        assert torch.bincount(labels[rows], minlength=3).unique().numel() == 1
    assert split_fingerprint(first) == split_fingerprint(second)
