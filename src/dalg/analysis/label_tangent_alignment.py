"""Label-conditioned tangent alignment in the fitted activation space.

Numerical routines use CPU tensors: float64 for geometry and scores, int64
for labels/indices, and bool for masks. Inputs must not be mutated. Year labels
are integers; a segment is identified by its smaller endpoint y and ends at
y + label_step. Endpoints are matched by label value, not sorted position.
"""

from __future__ import annotations

import math

import torch


def label_block_starts(labels: torch.Tensor, block_size: int) -> torch.Tensor:
    """Map integer labels to zero-anchored blocks [b, b + block_size)."""
    if labels.ndim != 1 or labels.dtype != torch.int64 or labels.device.type != "cpu":
        raise ValueError("labels must be a 1D int64 CPU tensor")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    return torch.div(labels, block_size, rounding_mode="floor") * block_size


def estimate_label_tangents(
    activations: torch.Tensor,
    labels: torch.Tensor,
    templates: list[str] | None = None,
    *,
    template_id: str | None = None,
    min_samples: int = 1,
    block_size: int = 100,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> dict:
    """Estimate secants between successive block centroids.

    ``templates`` and ``template_id`` are accepted for existing notebook calls
    but do not select observations. Average all prompts in each zero-anchored
    block [b, b+block_size), weighting prompts equally, not years equally.
    Nonempty partial blocks are retained; empty blocks are not bridged.
    ``years`` contains block starts. ``endpoint_indices`` indexes centroid
    rows in ``means`` and ``years``, not individual activation rows.
    """
    result = estimate_label_mean_tangents(
        activations, label_block_starts(labels, block_size), min_samples=min_samples,
        label_step=block_size, atol=atol, rtol=rtol)
    return dict(result, block_size=block_size)


def estimate_label_mean_tangents(
    activations: torch.Tensor,
    labels: torch.Tensor,
    *,
    min_samples: int = 2,
    label_step: int = 1,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> dict:
    """Estimate unit secants between label means separated by label_step.

    Args:
        activations: Finite floating CPU values (N, D) in the fitted model's
            space. Means and directions are computed in float64.
        labels: Int64 CPU labels (N,); N and D must be positive.
        min_samples: Minimum reference observations at each endpoint, >= 1.
        label_step: Positive integer difference between endpoint labels.
        atol, rtol: Nonnegative tolerances in the criterion
            ||m_next - m|| <= atol + rtol * max(||m||, ||m_next||).

    Returns a dictionary with sorted ``years`` (Y,), ``counts`` (Y,),
    ``means`` (Y,D), ``endpoint_indices`` (S,2) indexing those centroid rows,
    valid ``segment_years`` (S,), ``tangents`` (S,D), and
    ``exclusions``. The latter contains integer counts ``missing_next_year``,
    ``insufficient_samples``, and ``degenerate_difference``. Every observed
    year with y+label_step <= the maximum is a candidate start; classify exclusions in
    that order. Missing integer years that were never observed are not starts.
    Zero valid segments returns correctly shaped empty tensors, not an error.
    """
    if activations.ndim != 2 or min(activations.shape) == 0:
        raise ValueError("activations must have nonempty shape (N, D)")
    if labels.ndim != 1 or len(labels) != len(activations):
        raise ValueError("labels must have shape (N,) matching activations")
    if activations.device.type != "cpu" or labels.device.type != "cpu":
        raise ValueError("activations and labels must be CPU tensors")
    if not activations.is_floating_point() or labels.dtype != torch.int64:
        raise ValueError("activations must be floating point and labels must be int64")
    if not torch.isfinite(activations).all():
        raise ValueError("activations must be finite")
    if type(min_samples) is not int or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    if type(label_step) is not int or label_step < 1:
        raise ValueError("label_step must be a positive integer")
    if any(not math.isfinite(tol) or tol < 0 for tol in (atol, rtol)):
        raise ValueError("atol and rtol must be finite and nonnegative")

    unique_labels, class_to_indices, class_count = labels.unique(
        sorted=True, return_inverse=True, return_counts=True,
    )
    values = activations.double()
    centroids_label = values.new_zeros((len(unique_labels), values.shape[1]))
    # Each sample stays paired with its group ID; initial zeros are not samples.
    centroids_label.scatter_reduce_(
        0, class_to_indices[:, None].expand_as(values), values,
        reduce="mean", include_self=False,
    )

    starts = (unique_labels + label_step <= unique_labels[-1]).nonzero(as_tuple=True)[0]
    targets = unique_labels[starts] + label_step
    ends = torch.searchsorted(unique_labels, targets)
    matched = unique_labels[ends] == targets
    missing = int((~matched).sum())
    starts, ends = starts[matched], ends[matched]
    centroids_diff = centroids_label[ends] - centroids_label[starts]
    lengths = torch.linalg.vector_norm(centroids_diff, dim=1)
    centroid_norms = torch.linalg.vector_norm(centroids_label, dim=1)
    sufficient = (class_count[starts] >= min_samples) & (class_count[ends] >= min_samples)
    degenerate = lengths <= atol + rtol * torch.maximum(centroid_norms[starts], centroid_norms[ends])
    valid = sufficient & ~degenerate

    return {
        "years": unique_labels,
        "counts": class_count,
        "means": centroids_label,
        "label_step": label_step,
        "endpoint_indices": torch.stack((starts[valid], ends[valid]), dim=1),
        "segment_years": unique_labels[starts[valid]],
        "tangents": centroids_diff[valid] / lengths[valid, None],
        "exclusions": {
            "missing_next_year": missing,
            "insufficient_samples": int((~sufficient).sum()),
            "degenerate_difference": int((sufficient & degenerate).sum()),
        },
    }


def segment_assignment_weights(
    labels: torch.Tensor,
    assignments: torch.Tensor,
    segment_years: torch.Tensor,
    n_components: int,
    *,
    label_step: int = 1,
) -> torch.Tensor:
    """Return endpoint-balanced assignment weights (S, K).

    ``labels`` and hard ``assignments`` are matching (N,) int64 CPU tensors;
    IDs lie in [0,K). ``segment_years`` contains S distinct y -> y+label_step
    segment starts, in the requested output order. Each endpoint must occur in
    the evaluation population; otherwise raise ValueError. Endpoint sample
    counts may differ. Preserve all K components, including unoccupied ones.
    Every nonempty output row sums to one. S=0 returns shape (0,K).
    The result is float64; inputs are not modified.
    """
    if any(t.ndim != 1 or t.dtype != torch.int64 or t.device.type != "cpu"
           for t in (labels, assignments, segment_years)):
        raise ValueError("labels, assignments, and segment_years must be 1D int64 CPU tensors")
    if len(labels) != len(assignments):
        raise ValueError("labels and assignments must have matching lengths")
    if type(n_components) is not int or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if type(label_step) is not int or label_step < 1:
        raise ValueError("label_step must be a positive integer")
    if ((assignments < 0) | (assignments >= n_components)).any():
        raise ValueError("assignment IDs must lie in [0, n_components)")
    if len(segment_years.unique()) != len(segment_years):
        raise ValueError("segment_years must contain distinct segment starts")
    if len(segment_years) == 0:
        return torch.empty((0, n_components), dtype=torch.float64)
    if len(labels) == 0:
        raise ValueError("Every segment endpoint must occur in the evaluation population")

    unique_labels, sample_to_group, class_count = labels.unique(
        sorted=True, return_inverse=True, return_counts=True,
    )
    # Count every occurrence of each (label group, component) pair.
    pair_indices = sample_to_group * n_components + assignments
    counts = torch.bincount(pair_indices, minlength=len(unique_labels) * n_components)
    fractions = counts.reshape(len(unique_labels), n_components).double() / class_count[:, None]

    endpoints = torch.stack((segment_years, segment_years + label_step), dim=1)
    endpoint_rows = torch.searchsorted(unique_labels, endpoints)
    if (endpoint_rows >= len(unique_labels)).any() or not torch.equal(unique_labels[endpoint_rows], endpoints):
        raise ValueError("Every segment endpoint must occur in the evaluation population")
    # Normalize each year separately before giving both endpoints equal weight.
    return fractions[endpoint_rows].mean(dim=1)


def _unit_rows(vectors: torch.Tensor, name: str) -> torch.Tensor:
    """Validate unit directions and use float64 for geometric reductions."""
    if (vectors.ndim != 2 or vectors.shape[1] == 0 or vectors.device.type != "cpu"
            or not vectors.is_floating_point() or not torch.isfinite(vectors).all()):
        raise ValueError(f"{name} must be a finite floating CPU matrix with nonzero dimension")
    vectors = vectors.double()
    if not torch.allclose(vectors.norm(dim=1), torch.ones(len(vectors), dtype=torch.float64),
                          atol=1e-8, rtol=1e-8):
        raise ValueError(f"{name} must contain unit directions")
    return vectors


def _assignment_weights(weights: torch.Tensor) -> torch.Tensor:
    """Validate a nonempty table of normalized segment weights."""
    if (weights.ndim != 2 or min(weights.shape) == 0 or weights.device.type != "cpu"
            or not weights.is_floating_point() or not torch.isfinite(weights).all()
            or (weights < 0).any()):
        raise ValueError("weights must be a nonempty finite nonnegative floating CPU matrix")
    weights = weights.double()
    if not torch.allclose(weights.sum(1), torch.ones(len(weights), dtype=torch.float64),
                          atol=1e-8, rtol=1e-8):
        raise ValueError("weights must sum to one for every segment")
    return weights


def _alignment_summary(segment_scores: torch.Tensor, eligible_mass: torch.Tensor) -> dict:
    """Summarize equal-weight segments, pooling the conditional denominator."""
    mass = eligible_mass.sum()
    return dict(segment_scores=segment_scores, eligible_mass=eligible_mass,
                score=float(segment_scores.mean()),
                unidentified_weight=float((1 - eligible_mass).mean().clamp(0, 1)),
                conditional_score=float((segment_scores.sum() / mass).clamp(0, 1))
                if mass > 0 else None)


def score_pc1_alignment(
    tangents: torch.Tensor,
    pc1: torch.Tensor,
    identifiable: torch.Tensor,
    weights: torch.Tensor,
) -> dict:
    """Score squared PC1 alignment using equal total weight per segment.

    Inputs: unit tangents (S,D), PC1 rows (K,D), bool eligibility (K,), and
    nonnegative normalized weights (S,K). S must be positive. Eligible PC1s
    are unit vectors; ineligible rows may contain zeros or nonfinite values
    and must never contaminate output. Reject non-unit eligible directions.

    Returns float64 tensors ``alignment`` and ``contributions`` (S,K),
    ``segment_scores`` and ``eligible_mass`` (S,), plus Python floats
    ``score``, ``unidentified_weight``, and ``conditional_score`` (or None
    when there is no eligible weight). Invalid PCs contribute zero without
    renormalizing the primary score. Conditional score pools eligible weight
    across segments; it is not an unweighted mean of conditional segment scores.
    All primary scores lie in [0,1]. Unidentified weight is averaged over
    segments, not over individual activations.
    """
    tangents = _unit_rows(tangents, "tangents")
    weights = _assignment_weights(weights)
    if pc1.ndim != 2 or pc1.shape != (weights.shape[1], tangents.shape[1]):
        raise ValueError("pc1 must have shape (K, D) matching weights and tangents")
    if pc1.device.type != "cpu" or not pc1.is_floating_point():
        raise ValueError("pc1 must be a floating CPU tensor")
    if (identifiable.shape != (len(pc1),) or identifiable.dtype != torch.bool
            or identifiable.device.type != "cpu" or len(tangents) != len(weights)):
        raise ValueError("identifiable and tangents must match the weight table axes")
    valid_pc1 = _unit_rows(pc1[identifiable], "identifiable pc1")
    # Exclude undefined directions before arithmetic; NaN multiplied by zero is NaN.
    alignment = torch.zeros_like(weights)
    alignment[:, identifiable] = (tangents @ valid_pc1.T).square().clamp(0, 1)
    contributions = alignment * weights
    eligible_mass = weights[:, identifiable].sum(1).clamp(0, 1)
    return dict(alignment=alignment, contributions=contributions,
                **_alignment_summary(contributions.sum(1).clamp(0, 1), eligible_mass))


def make_crossfit_folds(
    years: torch.Tensor,
    templates: list[str] | None = None,
    *,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint sorted row-index tensors A and B covering all N rows.

    Balance counts within each year to differ by at most one and, when
    templates are supplied, within each (year,template) stratum to differ by
    at most one. Allocate singleton strata across folds so they do not all
    accumulate in one fold. Odd counts cannot give exact balance. Years with
    only one observation are retained, then excluded by tangent eligibility.
    Use only a local seeded RNG; do not modify global random state or inputs.
    No activation values, assignments, or fitted geometry influence splitting.
    """
    if years.ndim != 1 or years.dtype != torch.int64 or years.device.type != "cpu":
        raise ValueError("years must be a 1D int64 CPU tensor")
    if templates is not None and (len(templates) != len(years)
                                   or any(not isinstance(t, str) for t in templates)):
        raise ValueError("templates must contain one string per observation")
    generator = torch.Generator().manual_seed(seed)
    by_year = {}
    for row, year in enumerate(years.tolist()):
        template = templates[row] if templates is not None else ""
        by_year.setdefault(year, {}).setdefault(template, []).append(row)
    folds = [[], []]
    for year in sorted(by_year):
        strata = list(by_year[year].values())
        year_folds = [[], []]
        for stratum in torch.randperm(len(strata), generator=generator).tolist():
            rows = strata[stratum]
            shuffled = [rows[i] for i in torch.randperm(len(rows), generator=generator).tolist()]
            half = len(rows) // 2
            year_folds[0].extend(shuffled[:half])
            year_folds[1].extend(shuffled[half:2 * half])
            if len(rows) % 2:
                # Balance odd strata within the year, then balance the overall split.
                balance = len(year_folds[0]) - len(year_folds[1])
                if balance == 0:
                    balance = len(folds[0]) - len(folds[1])
                side = int(balance > 0) if balance else int(torch.randint(2, (), generator=generator))
                year_folds[side].append(shuffled[-1])
        for side in (0, 1):
            folds[side].extend(year_folds[side])
    return tuple(torch.tensor(sorted(rows), dtype=torch.int64) for rows in folds)


def reference_agreement(tangents_a: torch.Tensor, tangents_b: torch.Tensor) -> torch.Tensor:
    """Return squared directional agreement (S,) for paired unit secants.

    Inputs are matching finite (S,D) tensors already restricted to the same
    ordered segment population. Agreement is invariant to direction signs,
    lies in [0,1], and is not a confidence interval or a model-quality score.
    """
    if tangents_a.shape != tangents_b.shape:
        raise ValueError("Reference tangents must have matching shapes and segment order")
    a = _unit_rows(tangents_a, "tangents_a")
    b = _unit_rows(tangents_b, "tangents_b")
    return (a * b).sum(1).square().clamp(0, 1)


def combine_crossfit_scores(fold_scores: list[dict]) -> dict:
    """Combine two directional scores over the same ordered segment population.

    Each dictionary follows ``score_pc1_alignment``'s output contract. Give
    both directions equal weight, regardless of their evaluation sample counts.
    Return ``segment_scores`` and ``eligible_mass`` (S,), plus ``score``,
    ``unidentified_weight``, and ``conditional_score`` (Python floats, with
    conditional_score None if all weight is ineligible). Conditional alignment
    pools eligible weight across directions and segments, rather than averaging
    the directional conditional scores. S must be positive and equal in both
    directions. The caller ensures matching segment identities and ordering.
    """
    if len(fold_scores) != 2:
        raise ValueError("Exactly two directional score dictionaries are required")
    shape = fold_scores[0]["segment_scores"].shape
    for fold in fold_scores:
        scores, mass = fold["segment_scores"], fold["eligible_mass"]
        if scores.ndim != 1 or scores.numel() == 0 or scores.shape != shape or mass.shape != shape:
            raise ValueError("Both folds must score the same nonempty segment population")
        if any(t.device.type != "cpu" or not t.is_floating_point() or not torch.isfinite(t).all()
               for t in (scores, mass)):
            raise ValueError("Fold scores and eligible masses must be finite floating CPU tensors")
        if ((mass < 0) | (mass > 1) | (scores < 0) | (scores > mass + 1e-8)).any():
            raise ValueError("Fold scores must lie between zero and their eligible mass")
    scores = torch.stack([fold["segment_scores"].double() for fold in fold_scores]).mean(0)
    mass = torch.stack([fold["eligible_mass"].double() for fold in fold_scores]).mean(0)
    return _alignment_summary(scores, mass)


def shuffled_reference_labels(years: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Permute reference labels without replacement using a local seeded RNG.

    Preserve the exact multiset, shape, and dtype. Permute only within the
    supplied reference fold, across templates. Evaluation labels, assignments,
    and fold membership remain fixed. A permutation may retain fixed points.
    """
    if years.ndim != 1 or years.dtype != torch.int64 or years.device.type != "cpu":
        raise ValueError("years must be a 1D int64 CPU tensor")
    generator = torch.Generator().manual_seed(seed)
    return years[torch.randperm(len(years), generator=generator)]


def component_alignment_diagnostics(
    assignments: torch.Tensor,
    weights: torch.Tensor,
    alignment: torch.Tensor,
    identifiable: torch.Tensor,
    ranks: torch.Tensor,
) -> dict:
    """Summarize component occupancy and contributions for one cross-fit fold.

    Inputs are evaluation hard IDs (N,), weights and masked alignment (S,K),
    eligibility (K,), and saved integer effective ranks (K,). N,S > 0.
    Return tensors (K,): ``rank`` (int64), ``identifiable`` (bool),
    ``assignment_mass`` (fraction of evaluation samples), ``segment_mass``
    (mean segment weight), ``contribution`` (mean weighted alignment), and
    ``conditional_alignment`` (contribution / segment_mass for identifiable,
    positive-mass PCs, NaN otherwise). Assignment mass and segment mass have
    different denominators. Contributions sum to the fold's primary score.
    Retain every component, including rank-zero and unoccupied components.
    """
    weights = _assignment_weights(weights)
    k = weights.shape[1]
    if (assignments.ndim != 1 or assignments.dtype != torch.int64
            or assignments.device.type != "cpu" or len(assignments) == 0
            or ((assignments < 0) | (assignments >= k)).any()):
        raise ValueError("assignments must be nonempty int64 CPU IDs in [0, K)")
    if (identifiable.shape != (k,) or identifiable.dtype != torch.bool
            or identifiable.device.type != "cpu" or ranks.shape != (k,)
            or ranks.dtype != torch.int64 or ranks.device.type != "cpu" or (ranks < 0).any()):
        raise ValueError("identifiable and nonnegative ranks must describe all K components")
    if (alignment.shape != weights.shape or alignment.device.type != "cpu"
            or not alignment.is_floating_point() or not torch.isfinite(alignment).all()
            or ((alignment < 0) | (alignment > 1)).any()
            or (alignment[:, ~identifiable] != 0).any()):
        raise ValueError("alignment must be finite in [0, 1], with zero for unidentified PCs")
    segment_mass = weights.mean(0)
    contribution = (weights * alignment.double()).mean(0)
    conditional = torch.full((k,), float("nan"), dtype=torch.float64)
    valid = identifiable & (segment_mass > 0)
    conditional[valid] = contribution[valid] / segment_mass[valid]
    return dict(rank=ranks.clone(), identifiable=identifiable.clone(),
                assignment_mass=torch.bincount(assignments, minlength=k).double() / len(assignments),
                segment_mass=segment_mass, contribution=contribution,
                conditional_alignment=conditional)


def summarize_paired_alignment(
    segment_years: torch.Tensor,
    hddc_scores: torch.Tensor,
    kmeans_scores: torch.Tensor,
) -> dict:
    """Describe paired scores on the same S > 0 segments.

    Inputs are (S,) tensors in matching order. Return Python scalars:
    ``mean_difference``, ``median_difference``, ``min_difference``,
    ``max_difference``, ``positive_fraction``, ``negative_fraction``,
    ``tie_fraction``. Differences are HDDC minus KMeans; a tie is abs(delta)
    <= 1e-12. Median uses the midpoint for even counts. Also return string
    ``summary`` describing whether the advantage is consistent, restricted
    to some segments, or absent. Describe the observed population only:
    these dependent segments do not justify a significance or coverage claim.
    """
    if (segment_years.ndim != 1 or segment_years.dtype != torch.int64
            or segment_years.device.type != "cpu" or len(segment_years) == 0
            or len(segment_years.unique()) != len(segment_years)):
        raise ValueError("segment_years must contain distinct int64 CPU segment starts")
    for scores in (hddc_scores, kmeans_scores):
        if (scores.shape != segment_years.shape or scores.device.type != "cpu"
                or not scores.is_floating_point() or not torch.isfinite(scores).all()
                or ((scores < 0) | (scores > 1)).any()):
            raise ValueError("Paired scores must match the segments and lie in [0, 1]")
    differences = hddc_scores.double() - kmeans_scores.double()
    positive = int((differences > 1e-12).sum())
    negative = int((differences < -1e-12).sum())
    total = len(differences)
    ties = total - positive - negative
    if positive == total:
        conclusion = "HDDC's observed advantage is consistent across all evaluated segments."
    elif positive:
        conclusion = "HDDC's observed advantage is restricted to some evaluated segments."
    else:
        conclusion = "No HDDC advantage is observed on the evaluated segments."
    summary = (f"{conclusion} HDDC is higher on {positive}/{total}, lower on {negative}/{total}, "
               f"and tied on {ties}/{total}; mean HDDC minus KMeans = {float(differences.mean()):.6g}. "
               "These are descriptive results for alignment with label secants.")
    return dict(mean_difference=float(differences.mean()),
                median_difference=float(differences.quantile(.5)),
                min_difference=float(differences.min()), max_difference=float(differences.max()),
                positive_fraction=positive / total, negative_fraction=negative / total,
                tie_fraction=ties / total, summary=summary)
