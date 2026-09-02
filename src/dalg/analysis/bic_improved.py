r"""Compute a utilization-adjusted BIC score for one trained MFA run.

This module deliberately does **not** redefine the standard BIC implemented in
``dalg.analysis.bic``.  It defines a separate research score for experiments in
which ``K`` is intended to be the number of local tiles, rather than merely an
upper bound on the number of mixture components.  In that setting, a model that
turns components off has failed to use part of the requested tiling capacity,
even if its likelihood and parameter count are otherwise competitive.

The score is

.. math::

    S_{active\text{-}BIC}
      = -\frac{BIC}{n} + K_{active}
      = 2\,\overline{\log L}
        - \frac{p\log n}{n}
        + K_{active},

and **higher is better**.  ``BIC`` is the standard minimizing training-set BIC
from :mod:`dalg.analysis.bic`, ``n`` is the number of training activation
vectors, and ``K_active`` is the number of components receiving at least one
hard MAP assignment on that same training split.  Consequently, if two models
have identical likelihood and parameter count, every additional active
component raises this score by exactly one.

Dividing standard BIC by ``n`` is essential: without it, an activity reward of
at most ``K`` would be numerically irrelevant beside a BIC that grows linearly
with dataset size.  The unit activity reward is an explicit research preference,
not a consequence of Bayesian model selection.  This score should therefore be
called a utilization-adjusted or active-BIC score, not standard BIC.

Comparisons are meaningful only between runs fitted to the same dataset and
split with the same nominal ``K``.  Increasing ``K`` changes both the model and
the maximum activity reward.  The score also treats a one-point MAP winner as
active; it intentionally answers the literal "was this Gaussian turned off?"
question rather than imposing an additional minimum-occupancy hyperparameter.

Assignment bundles cover the complete selected stream (training plus
validation).  This module reconstructs the recorded split and slices the bundle
at row boundaries, including activation windows with more than one retained
token, so activity and BIC are evaluated on exactly the same training samples.
It reads existing artifacts only and is not integrated into the evaluation
pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from dalg.analysis.bic import compute_bic
from dalg.data.shard_activations import load_meta_index
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec


def active_bic_from_standard(
    standard_bic: float,
    *,
    n: int,
    active_components: int,
    K: int,
) -> float:
    """Return ``-standard_bic / n + active_components``.

    Args:
        standard_bic: Standard minimizing BIC, ``-2 log L + p log n``.
        n: Number of activation vectors in the BIC training split.
        active_components: Components with at least one hard MAP assignment on
            those same activation vectors.
        K: Nominal number of mixture components.  It validates the activity
            count and records the comparison scale; it does not otherwise alter
            the formula.

    Returns:
        A finite utilization-adjusted score under the higher-is-better
        convention.
    """
    if not math.isfinite(standard_bic):
        raise ValueError("active-BIC requires a finite standard BIC")
    if n <= 0:
        raise ValueError("active-BIC requires at least one training sample")
    if K <= 0:
        raise ValueError("active-BIC requires K > 0")
    if not 0 <= active_components <= K:
        raise ValueError(
            "active_components must lie in [0, K]: "
            f"active_components={active_components}, K={K}"
        )
    return -float(standard_bic) / int(n) + int(active_components)


def _subset_spec(run_dir: Path) -> str | None:
    run_spec_path = run_dir / "run_spec.json"
    if not run_spec_path.exists():
        return None
    run_spec = json.loads(run_spec_path.read_text())
    _, subset_spec = split_shard_dir_spec(run_spec["dataset"]["shard_dir"])
    return subset_spec


def _training_cluster_sizes(
    run_dir: Path,
    assignments_path: Path,
) -> tuple[torch.Tensor, int]:
    """Return hard MAP counts on the run's recorded training split.

    The saved assignment array follows the canonical selected-row order and has
    ``window - drop_prefix`` consecutive entries per row.  We retain or discard
    each complete row-sized block using ``val_global_rows``; token assignments
    from a validation window can therefore never leak into the activity count.
    """
    config = json.loads((run_dir / "config.json").read_text())
    split = json.loads((run_dir / "val_indices.json").read_text())
    shard_dir = Path(config["shard_dir"])
    layer = int(config["layer"])
    window = int(config["window"])
    drop_prefix = int(config["drop_prefix"])
    items_per_row = window - drop_prefix
    if items_per_row <= 0:
        raise ValueError(
            "window - drop_prefix must be positive: "
            f"window={window}, drop_prefix={drop_prefix}"
        )

    meta_index = load_meta_index(shard_dir, layer=layer)
    subset_spec = _subset_spec(run_dir)
    selected_positions = resolve_spec_positions(
        meta_index,
        subset_spec,
        window=window,
        drop_prefix=drop_prefix,
    )

    bundle: dict[str, Any] = torch.load(
        assignments_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    required = {"assignments", "cluster_sizes", "K"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(
            f"assignment bundle is missing required fields: {sorted(missing)}"
        )
    if bundle.get("subset_spec") != subset_spec:
        raise ValueError(
            "assignment subset does not match the run: "
            f"assignments={bundle.get('subset_spec')!r}, run={subset_spec!r}"
        )

    assignments = bundle["assignments"].reshape(-1).long()
    saved_sizes = bundle["cluster_sizes"].reshape(-1).long()
    K = int(bundle["K"])
    configured_K = config.get("K")
    if configured_K is not None and int(configured_K) != K:
        raise ValueError(
            "assignment K does not match the trained run: "
            f"assignments={K}, config={int(configured_K)}"
        )
    expected_items = len(selected_positions) * items_per_row
    if assignments.numel() != expected_items:
        raise ValueError(
            "assignments do not cover the selected canonical stream: "
            f"assignments={assignments.numel()}, expected={expected_items}"
        )
    if saved_sizes.numel() != K:
        raise ValueError(
            f"cluster_sizes has {saved_sizes.numel()} entries, but K={K}"
        )
    if assignments.numel() and (
        int(assignments.min()) < 0 or int(assignments.max()) >= K
    ):
        raise ValueError("assignments contain component ids outside [0, K)")
    full_sizes = torch.bincount(assignments, minlength=K)
    if not torch.equal(full_sizes, saved_sizes):
        raise ValueError("cluster_sizes is inconsistent with assignments")

    validation_rows = {int(row) for row in split["val_global_rows"]}
    train_chunks: list[torch.Tensor] = []
    for row_index, position in enumerate(selected_positions):
        if int(meta_index[position]["global_row"]) in validation_rows:
            continue
        start = row_index * items_per_row
        train_chunks.append(assignments[start : start + items_per_row])

    train_assignments = (
        torch.cat(train_chunks)
        if train_chunks
        else torch.empty(0, dtype=torch.long)
    )
    expected_train_items = int(split["train_rows"]) * items_per_row
    if train_assignments.numel() != expected_train_items:
        raise ValueError(
            "recorded and reconstructed training splits disagree: "
            f"assignments={train_assignments.numel()}, "
            f"expected={expected_train_items}"
        )
    return torch.bincount(train_assignments, minlength=K), expected_train_items


def compute_improved_bic_details(
    run_dir: str | Path,
    *,
    assignments_path: str | Path | None = None,
    batch_size: int = 2_048,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Compute standard BIC, training activity, and active-BIC diagnostics."""
    run_dir = Path(run_dir)
    resolved_assignments_path = (
        Path(assignments_path)
        if assignments_path is not None
        else run_dir / "mfa_model_assignments.pt"
    )
    cluster_sizes, n = _training_cluster_sizes(
        run_dir,
        resolved_assignments_path,
    )
    K = int(cluster_sizes.numel())
    active_components = int((cluster_sizes > 0).sum().item())
    standard_bic = compute_bic(
        run_dir,
        batch_size=batch_size,
        device=device,
    )
    score = active_bic_from_standard(
        standard_bic,
        n=n,
        active_components=active_components,
        K=K,
    )
    return {
        "value": score,
        "standard_bic": float(standard_bic),
        "standard_bic_per_sample_reward": -float(standard_bic) / n,
        "activity_reward": active_components,
        "active_components": active_components,
        "inactive_components": K - active_components,
        "K": K,
        "n": n,
        "split": "train",
        "assignment_rule": "hard_map_count_greater_than_zero",
        "formula": "-standard_bic / n + active_components",
        "convention": "higher_is_better",
    }


def compute_improved_bic(
    run_dir: str | Path,
    *,
    assignments_path: str | Path | None = None,
    batch_size: int = 2_048,
    device: str | torch.device = "cpu",
) -> float:
    """Return the scalar higher-is-better active-BIC score for ``run_dir``."""
    return float(
        compute_improved_bic_details(
            run_dir,
            assignments_path=assignments_path,
            batch_size=batch_size,
            device=device,
        )["value"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--assignments-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the score decomposition instead of only the scalar value",
    )
    args = parser.parse_args()

    details = compute_improved_bic_details(
        args.run_dir,
        assignments_path=args.assignments_path,
        batch_size=args.batch_size,
        device=args.device,
    )
    if args.json:
        print(json.dumps(details, indent=2, sort_keys=True))
    else:
        print(details["value"])


if __name__ == "__main__":
    main()


__all__ = [
    "active_bic_from_standard",
    "compute_improved_bic",
    "compute_improved_bic_details",
]
