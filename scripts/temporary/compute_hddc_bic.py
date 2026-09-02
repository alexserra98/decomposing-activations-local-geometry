"""Compute post-hoc BIC for a folder of shared-b MFA-HDDC runs.

This one-off analysis targets the HDDC ``[a_kj b Q_k d_k]`` model: each
component has its own signal eigenvalues, orientation, and rank, while all
components share one isotropic noise floor.  It evaluates the saved best model
on the exact train/validation split recorded by the pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.models.adaptive_q.hddc_surgery import parameter_count
from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc


def _load_data(run_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    config = json.loads((run_dir / "config.json").read_text())
    split = json.loads((run_dir / "val_indices.json").read_text())
    shard_dir = Path(config["shard_dir"])
    layer = int(config["layer"])
    drop_prefix = int(config["drop_prefix"])

    meta_index = load_meta_index(shard_dir, layer=layer)
    val_rows = set(int(row) for row in split["val_global_rows"])
    val_positions = [
        position
        for position, entry in enumerate(meta_index)
        if int(entry["global_row"]) in val_rows
    ]
    train_positions = [
        position
        for position, entry in enumerate(meta_index)
        if int(entry["global_row"]) not in val_rows
    ]
    if len(train_positions) != int(split["train_rows"]):
        raise ValueError("recorded and reconstructed training splits disagree")
    if len(val_positions) != int(split["val_rows"]):
        raise ValueError("recorded and reconstructed validation splits disagree")

    def materialize(positions: list[int]) -> torch.Tensor:
        dataset = ActivationBatchDataset(
            shard_dir,
            layer=layer,
            row_subset=positions,
            batch_size=10_000,
            drop_prefix=drop_prefix,
            shuffle_shards=False,
            shuffle_within_shard=False,
            dtype=torch.float32,
        )
        return torch.cat(list(dataset), dim=0)

    train = materialize(train_positions)
    validation = materialize(val_positions)
    provenance = {
        "shard_dir": str(shard_dir.resolve()),
        "layer": layer,
        "split_seed": int(split["seed"]),
        "val_frac": float(split["val_frac"]),
        "n_train": int(train.shape[0]),
        "n_validation": int(validation.shape[0]),
        "n_full": int(train.shape[0] + validation.shape[0]),
    }
    return train, validation, provenance


@torch.no_grad()
def _log_likelihood(model, data: torch.Tensor, batch_size: int) -> float:
    total = torch.zeros((), dtype=torch.float64)
    model.eval()
    with model.inference_cache():
        for batch in data.split(batch_size):
            total += model.log_prob(batch).double().sum()
    return float(total.item())


def _hdclassif_parameter_count(model) -> int:
    """Parameter count for HDclassif model ``AKJBQKDK``.

    The implementation uses

      K*D + K - 1                         means and mixture weights
      sum_k d_k * (D - (d_k + 1) / 2)    component orientations
      sum_k d_k                           component signal eigenvalues
      K                                   selected component dimensions
      1                                   shared b
    """
    ranks = model.component_ranks.to(torch.float64)
    orientation = (ranks * (model.D - (ranks + 1.0) / 2.0)).sum()
    count = (
        model.K * model.D
        + model.K
        - 1
        + orientation.item()
        + ranks.sum().item()
        + model.K
        + 1
    )
    if not float(count).is_integer():
        raise ValueError(f"non-integral HDDC parameter count: {count}")
    return int(count)


def _sample_metrics(log_likelihood: float, n: int, parameters: int) -> dict:
    penalty = parameters * math.log(n)
    bic_max = 2.0 * log_likelihood - penalty
    return {
        "n": n,
        "log_likelihood": log_likelihood,
        "mean_nll": -log_likelihood / n,
        "penalty": penalty,
        "bic_hddc_maximize": bic_max,
        "bic_standard_minimize": -bic_max,
    }


def compute_run(
    run_dir: Path,
    train: torch.Tensor,
    validation: torch.Tensor,
    provenance: dict,
    *,
    batch_size: int,
) -> dict:
    spec = json.loads((run_dir / "run_spec.json").read_text())
    threshold = float(spec["training"]["arguments"]["surgery_threshold"])
    model = load_mfa_hddc(run_dir / "mfa_model.pt", map_location="cpu")
    if not model.shared_b:
        raise ValueError(f"expected shared-b HDDC model: {run_dir}")

    ranks = model.component_ranks.cpu()
    hdclassif_parameters = _hdclassif_parameter_count(model)
    repo_parameters = parameter_count(model)
    train_ll = _log_likelihood(model, train, batch_size)
    validation_ll = _log_likelihood(model, validation, batch_size)
    full_ll = train_ll + validation_ll
    train_metrics = _sample_metrics(
        train_ll, provenance["n_train"], hdclassif_parameters
    )
    full_metrics = _sample_metrics(
        full_ll, provenance["n_full"], hdclassif_parameters
    )
    train_metrics["bic_repo_helper_maximize"] = _sample_metrics(
        train_ll, provenance["n_train"], repo_parameters
    )["bic_hddc_maximize"]
    full_metrics["bic_repo_helper_maximize"] = _sample_metrics(
        full_ll, provenance["n_full"], repo_parameters
    )["bic_hddc_maximize"]

    return {
        "schema_version": 1,
        "run_dir": str(run_dir.resolve()),
        "run_id": run_dir.name,
        "surgery_threshold": threshold,
        "model": "AKJBQKDK",
        "criterion": "bic_hddc_maximize = 2 * log_likelihood - parameters * log(n)",
        "data": provenance,
        "ranks": {
            "sum": int(ranks.sum().item()),
            "mean": float(ranks.float().mean().item()),
            "min": int(ranks.min().item()),
            "median": int(ranks.median().item()),
            "max": int(ranks.max().item()),
            "histogram_0_to_q_max": torch.bincount(
                ranks, minlength=model.q + 1
            ).tolist(),
        },
        "parameters": {
            "hdclassif_akjbqkdk": hdclassif_parameters,
            "repo_parameter_count_helper": repo_parameters,
            "difference": hdclassif_parameters - repo_parameters,
        },
        "train": train_metrics,
        "validation": {
            "n": provenance["n_validation"],
            "log_likelihood": validation_ll,
            "mean_nll": -validation_ll / provenance["n_validation"],
        },
        "full": full_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write bic_metrics.json per run and bic_summary.json at the root",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run_dirs = sorted(path for path in args.root.iterdir() if path.is_dir())
    if not run_dirs:
        raise SystemExit(f"no run directories found under {args.root}")

    train, validation, provenance = _load_data(run_dirs[0])
    results = []
    for run_dir in run_dirs:
        result = compute_run(
            run_dir,
            train,
            validation,
            provenance,
            batch_size=args.batch_size,
        )
        results.append(result)
        print(
            f"threshold={result['surgery_threshold']:<7g} "
            f"p={result['parameters']['hdclassif_akjbqkdk']:,} "
            f"train_nll={result['train']['mean_nll']:.6f} "
            f"train_bic={result['train']['bic_hddc_maximize']:.3f} "
            f"val_nll={result['validation']['mean_nll']:.6f}"
        )
        if args.write:
            (run_dir / "bic_metrics.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )

    results.sort(key=lambda item: item["surgery_threshold"])
    summary = {
        "schema_version": 1,
        "criterion": "higher bic_hddc_maximize is better",
        "selected_by_train_bic": max(
            results, key=lambda item: item["train"]["bic_hddc_maximize"]
        )["surgery_threshold"],
        "selected_by_full_bic": max(
            results, key=lambda item: item["full"]["bic_hddc_maximize"]
        )["surgery_threshold"],
        "selected_by_train_bic_repo_helper": max(
            results, key=lambda item: item["train"]["bic_repo_helper_maximize"]
        )["surgery_threshold"],
        "runs": results,
    }
    if args.write:
        (args.root / "bic_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
