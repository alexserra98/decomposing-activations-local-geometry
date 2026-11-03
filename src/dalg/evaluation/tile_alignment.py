"""Local PC agreement on a directed nearest-centroid graph."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from dalg.evaluation.subspaces import (
    PC_RELATIVE_EIGENGAP_THRESHOLD,
    best_subset_alignment,
    effective_component_ranks,
    leading_covariance_eigenspaces,
    subspace_alignment,
    warn_tied_eigenvalues,
)
from dalg.models.kmeans import KMeans

_DEFINITIONS = {
    "leading_pc_alignment": "leading_min_rank_pc_subspace_overlap",
    "best_subset_pc_alignment": "best_min_rank_subset_of_larger_rank_pcs_overlap",
}


def validate_tile_alignment_settings(enabled: bool, neighbors: int) -> None:
    if type(enabled) is not bool:
        raise ValueError("evaluation.tile_alignment must be true or false")
    if type(neighbors) is not int or neighbors <= 0:
        raise ValueError("evaluation.tile_alignment_neighbors must be a positive integer")


@torch.no_grad()
def evaluate_tile_alignment(
    model,
    assignment_live: torch.Tensor,
    *,
    neighbors: int = 5,
    rank_threshold: float = 1.0,
    eigenspaces: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Average directed neighbor overlaps per tile, then equally across tiles.

    Optional precomputed eigenspaces allow toy geometry and tile alignment to
    share the same covariance decomposition. All returned IDs are model IDs.
    """
    validate_tile_alignment_settings(True, neighbors)
    live = assignment_live.detach().cpu().bool().reshape(-1)
    if live.numel() != model.K:
        raise ValueError("assignment-live mask does not match model K")
    ranks = effective_component_ranks(model, rank_threshold)
    eligible = live & (ranks > 0)
    if isinstance(model, KMeans):
        eligible &= model.pca_valid.detach().cpu()
    ids = torch.where(eligible)[0]
    count = len(ids)
    effective_neighbors = min(neighbors, max(0, count - 1))
    report = {
        "version": 1,
        "definitions": dict(_DEFINITIONS),
        "graph": "directed_euclidean_centroid_knn",
        "population": "live_positive_rank_pca_valid_components",
        "aggregation": "unweighted_mean_of_per_tile_neighbor_means",
        "neighbors": neighbors,
        "effective_neighbors": effective_neighbors,
        "eligible_components": count,
        "excluded_components": model.K - count,
        "relative_eigengap_tolerance": PC_RELATIVE_EIGENGAP_THRESHOLD,
        **{name: {"mean": None} for name in _DEFINITIONS},
        "per_tile": [],
        "comparisons": [],
    }
    if count:
        if eigenspaces is None:
            eigenspaces = leading_covariance_eigenspaces(model, int(ranks[eligible].max()))
        eigenvalues, pcs = eigenspaces
        warn_tied_eigenvalues(eigenvalues, ranks, eligible)
        centroids = model.mu.detach().cpu().double()[ids]
        # Direct distances avoid cancellation for nearby or identical centroids.
        distances = torch.cdist(centroids, centroids, compute_mode="donot_use_mm_for_euclid_dist")
        distances.fill_diagonal_(float("inf"))
        # IDs are ascending, so stable sorting resolves distance ties by model ID.
        nearest = torch.argsort(distances, dim=1, stable=True)[:, :effective_neighbors]
        for position, source in enumerate(ids.tolist()):
            values = {name: [] for name in _DEFINITIONS}
            for target_position in nearest[position].tolist():
                target = int(ids[target_position])
                source_rank, target_rank = int(ranks[source]), int(ranks[target])
                minimum = min(source_rank, target_rank)
                leading = subspace_alignment(pcs[source, :, :minimum], pcs[target, :, :minimum])[0]
                smaller, larger = (source, target) if source_rank <= target_rank else (target, source)
                best = best_subset_alignment(
                    pcs[smaller, :, :minimum], pcs[larger, :, :int(ranks[larger])],
                )[0]
                scores = {"leading_pc_alignment": leading, "best_subset_pc_alignment": best}
                report["comparisons"].append({
                    "source": source, "target": target,
                    "distance": float(distances[position, target_position]),
                    "source_rank": source_rank, "target_rank": target_rank,
                    **scores,
                })
                for name, score in scores.items():
                    values[name].append(score)
            report["per_tile"].append({
                "component_id": source, "rank": int(ranks[source]),
                **{name: sum(scores) / len(scores) if scores else None for name, scores in values.items()},
            })
    if effective_neighbors:
        for name in _DEFINITIONS:
            report[name]["mean"] = sum(tile[name] for tile in report["per_tile"]) / count
    return report


def tile_alignment_report_valid(report, *, components: int, neighbors: int) -> bool:
    """Check the saved graph, score populations, and requested neighbor setting."""
    if not isinstance(report, Mapping):
        return False
    count = report.get("eligible_components")
    if (report.get("version") != 1 or report.get("definitions") != _DEFINITIONS
            or report.get("graph") != "directed_euclidean_centroid_knn"
            or report.get("population") != "live_positive_rank_pca_valid_components"
            or report.get("aggregation") != "unweighted_mean_of_per_tile_neighbor_means"
            or report.get("relative_eigengap_tolerance") != PC_RELATIVE_EIGENGAP_THRESHOLD
            or report.get("neighbors") != neighbors
            or type(count) is not int or not 0 <= count <= components
            or report.get("excluded_components") != components - count):
        return False
    degree = min(neighbors, max(0, count - 1))
    tiles, edges = report.get("per_tile"), report.get("comparisons")
    if (report.get("effective_neighbors") != degree or not isinstance(tiles, list)
            or not isinstance(edges, list) or len(tiles) != count or len(edges) != count * degree):
        return False
    ranks = {}
    for tile in tiles:
        if not isinstance(tile, Mapping):
            return False
        index, rank = tile.get("component_id"), tile.get("rank")
        if (type(index) is not int or not 0 <= index < components or index in ranks
                or type(rank) is not int or rank <= 0):
            return False
        ranks[index] = rank
    seen = set()
    totals = {index: {name: [] for name in _DEFINITIONS} for index in ranks}
    for edge in edges:
        if not isinstance(edge, Mapping):
            return False
        source, target, distance = edge.get("source"), edge.get("target"), edge.get("distance")
        if (type(source) is not int or type(target) is not int
                or source not in ranks or target not in ranks or source == target
                or (source, target) in seen or edge.get("source_rank") != ranks[source]
                or edge.get("target_rank") != ranks[target]
                or type(distance) not in (int, float) or not math.isfinite(distance) or distance < 0):
            return False
        seen.add((source, target))
        for name in _DEFINITIONS:
            value = edge.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                return False
            totals[source][name].append(value)
    for name in _DEFINITIONS:
        means = []
        for tile in tiles:
            scores = totals[tile["component_id"]][name]
            if len(scores) != degree:
                return False
            expected = sum(scores) / degree if degree else None
            if tile.get(name) != expected:
                return False
            means.append(expected)
        summary = report.get(name)
        expected = sum(means) / count if degree else None
        if not isinstance(summary, Mapping) or summary.get("mean") != expected:
            return False
    return True
