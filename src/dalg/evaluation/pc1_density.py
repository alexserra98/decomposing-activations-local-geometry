"""PC1 density evaluation using saved training and validation partitions."""
from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import median

import torch
from torch.utils.data import DataLoader

from dalg.analysis.pc1_density import (
    KNNLogDensity, QUADRATURE_GRID_POINTS, fit_global_variance_pca, leading_components,
    project_pc1_segments, score_component,
)
from dalg.data.shard_activations import ActivationBatchDataset
from dalg.evaluation.common import EvaluationData

PC1_DENSITY_DEFAULTS = {
    "pc1_density": False,
    "pc1_density_space": "pca75",
    "pc1_density_neighbors": 20,
    "pc1_density_random_directions": 32,
    "pc1_density_seed": 42,
}
_DEFINITIONS = {
    "log_retention": "log_mean_segment_density_relative_to_center",
    "log_enrichment": "log_retention_minus_log_arithmetic_mean_random_retention",
}


def validate_pc1_density_settings(cfg: Mapping) -> None:
    if type(cfg.get("pc1_density", False)) is not bool:
        raise ValueError("evaluation.pc1_density must be true or false")
    if cfg.get("pc1_density_space", "pca75") not in ("pca75", "ambient"):
        raise ValueError("evaluation.pc1_density_space must be pca75 or ambient")
    for key in ("pc1_density_neighbors", "pc1_density_random_directions", "pc1_density_seed"):
        value = cfg.get(key, PC1_DENSITY_DEFAULTS[key])
        minimum = 0 if key == "pc1_density_seed" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"evaluation.{key} must be an integer >= {minimum}")


def _batches(data: EvaluationData, positions: list[int], layer: int, batch_size: int):
    dataset = ActivationBatchDataset(
        data.shard_dir, layer=layer, row_subset=positions, batch_size=batch_size,
        drop_prefix=data.drop_prefix, dtype=torch.float32,
        shuffle_shards=False, shuffle_within_shard=False,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)


@torch.no_grad()
def evaluate_pc1_density(model, data: EvaluationData, *, model_kind: str, layer: int,
                         batch_size: int, space: str = "pca75", neighbors: int = 20,
                         random_directions: int = 32, seed: int = 42) -> dict:
    """Fit optional training PCA and score original PC1 segments on validation density.

    CPU float64 preserves the pilot's numerical convention. The full validation
    reference is kept in memory; training covariance is accumulated in batches.
    """
    settings = dict(pc1_density=True, pc1_density_space=space, pc1_density_neighbors=neighbors,
                    pc1_density_random_directions=random_directions, pc1_density_seed=seed)
    validate_pc1_density_settings(settings)
    reference_points = len(data.val_positions) * data.tokens_per_row
    if reference_points < neighbors:
        raise ValueError("PC1 density requires at least pc1_density_neighbors saved validation points")
    geometry = leading_components(model, model_kind)
    projection = None
    if space == "pca75":
        pca = fit_global_variance_pca(_batches(data, data.train_positions, layer, batch_size), .75)
        projected = project_pc1_segments(geometry, pca["mean"], pca["basis"])
        projection = {key: pca[key] for key in (
            "dimension", "input_dimension", "training_count", "target_variance_fraction",
            "retained_variance_fraction", "previous_variance_fraction", "centering",
            "standardization", "whitening",
        )}
    else:
        projected = geometry
    references = []
    for batch in _batches(data, data.val_positions, layer, batch_size):
        x = batch.cpu().double()
        if space == "pca75":
            x = (x - pca["mean"]) @ pca["basis"].T
        references.append(x)
    reference = torch.cat(references)
    if len(reference) != reference_points:
        raise ValueError("PC1 density reference does not match the saved validation stream")
    estimator = KNNLogDensity(reference, k=neighbors)
    records, excluded = [], []
    for component, reason in enumerate(projected["exclusion_reasons"]):
        if reason:
            excluded.append({"component_id": component, "reason": reason})
            continue
        half_length = float(projected["half_lengths"][component]) if projection is not None else None
        score = score_component(
            estimator, projected["centers"][component], projected["vectors"][component],
            float(geometry["eigenvalues"][component]), n_random=random_directions,
            grid_points=101, seed=seed + component, adaptive=True, half_length=half_length,
        )
        record = {key: value for key, value in score.items() if not isinstance(value, torch.Tensor)}
        record.update(component_id=component, eigenvalue=float(geometry["eigenvalues"][component]),
                      eigengap=float(geometry["eigengap"][component]))
        if projection is not None:
            record.update(alpha=float(projected["alpha"][component]),
                          original_half_length=float(projected["original_half_lengths"][component]))
        records.append(record)
    return {
        "version": 2, "definitions": dict(_DEFINITIONS),
        "settings": {key: value for key, value in settings.items() if key != "pc1_density"},
        "population": "defined_covariance_pc1_components",
        "aggregation": "unweighted_component_log_scores",
        "interval_probability": .8, "pilot_points": 201,
        "quadrature_grid_points": list(QUADRATURE_GRID_POINTS),
        "integration": "adaptive_quad", "relative_tolerance": 1e-4,
        "density_dimension": estimator.dimension,
        "reference": {"split": "validation", "points": reference_points,
                      "global_rows": [data.meta_index[p]["global_row"] for p in data.val_positions]},
        "projection": projection,
        "eligible_components": len(records), "excluded_components": len(excluded),
        **{name: {"mean": sum(r[name] for r in records) / len(records) if records else None,
                  "median": median(r[name] for r in records) if records else None}
           for name in _DEFINITIONS},
        "per_component": records, "exclusions": excluded,
    }


def pc1_density_report_valid(report, *, components: int, cfg: Mapping, dataset: Mapping) -> bool:
    """Require settings, population, numerical scores, and the saved reference split."""
    if not isinstance(report, Mapping):
        return False
    settings = {k: cfg.get(k, v) for k, v in PC1_DENSITY_DEFAULTS.items() if k != "pc1_density"}
    if (report.get("version") != 2 or report.get("settings") != settings
            or report.get("definitions") != _DEFINITIONS
            or report.get("population") != "defined_covariance_pc1_components"
            or report.get("aggregation") != "unweighted_component_log_scores"
            or report.get("interval_probability") != .8 or report.get("pilot_points") != 201
            or report.get("quadrature_grid_points") != list(QUADRATURE_GRID_POINTS)
            or report.get("integration") != "adaptive_quad" or report.get("relative_tolerance") != 1e-4):
        return False
    reference = report.get("reference")
    if (not isinstance(reference, Mapping) or reference.get("split") != "validation"
            or type(reference.get("points")) is not int
            or reference["points"] != dataset.get("validation_points")
            or reference["points"] < settings["pc1_density_neighbors"]):
        return False
    rows = reference.get("global_rows")
    if (not isinstance(rows, list) or len(rows) != dataset.get("validation_rows")
            or any(type(row) is not int for row in rows) or len(set(rows)) != len(rows)):
        return False
    dimension = report.get("density_dimension")
    if type(dimension) is not int or dimension < 1:
        return False
    projection = report.get("projection")
    if settings["pc1_density_space"] == "pca75":
        if (not isinstance(projection, Mapping) or projection.get("dimension") != dimension
                or projection.get("training_count") != dataset.get("train_points")
                or projection.get("target_variance_fraction") != .75
                or projection.get("centering") is not True
                or projection.get("standardization") is not False or projection.get("whitening") is not False):
            return False
        previous, retained = (projection.get(k) for k in ("previous_variance_fraction", "retained_variance_fraction"))
        if (type(previous) not in (int, float) or type(retained) not in (int, float)
                or not 0 <= previous < .75 <= retained <= 1 + 1e-12):
            return False
    elif projection is not None:
        return False
    records, exclusions = report.get("per_component"), report.get("exclusions")
    if (not isinstance(records, list) or not isinstance(exclusions, list)
            or len(records) + len(exclusions) != components
            or report.get("eligible_components") != len(records)
            or report.get("excluded_components") != len(exclusions)):
        return False
    ids = []
    for record in records + exclusions:
        if not isinstance(record, Mapping) or type(record.get("component_id")) is not int:
            return False
        ids.append(record["component_id"])
    if sorted(ids) != list(range(components)):
        return False
    if any(not isinstance(r.get("reason"), str) or not r["reason"] for r in exclusions):
        return False
    for record in records:
        for key in (*_DEFINITIONS, "log_random_mean_retention", "half_length", "eigenvalue",
                    "center_log_density", "center_knn_radius", "adaptive_quadrature_relative_error"):
            value = record.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
        if (record["half_length"] <= 0 or record["eigenvalue"] <= 0 or record["center_knn_radius"] <= 0
                or not 0 <= record["adaptive_quadrature_relative_error"] <= 1e-4
                or not math.isclose(record["log_enrichment"], record["log_retention"] -
                                    record["log_random_mean_retention"], abs_tol=1e-10)):
            return False
    for name in _DEFINITIONS:
        expected = {"mean": sum(r[name] for r in records) / len(records) if records else None,
                    "median": median(r[name] for r in records) if records else None}
        if report.get(name) != expected:
            return False
    return True
