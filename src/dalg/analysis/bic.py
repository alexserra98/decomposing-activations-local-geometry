"""Compute BIC for a trained MFA-family run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec


def model_parameter_count(model, model_kind: str) -> int:
    """Return the identifiable parameter count for an MFA-family model."""
    model_kind = model_kind.lower()
    if model_kind == "hddc":
        ranks = model.component_ranks.detach().cpu().to(torch.float64)
        rank_parameters = model.K
        if model.shared_b:
            noise_parameters = 1
        elif model.isotropic_psi:
            noise_parameters = model.K
        else:
            noise_parameters = (
                model.K * model.D if model.psi_per_component else model.D
            )
    elif model_kind == "ard":
        ranks = model.effective_ranks().detach().cpu().to(torch.float64)
        rank_parameters = model.K
        noise_parameters = (
            model.K * model.D if model.psi_per_component else model.D
        )
    elif model_kind == "mfa":
        ranks = torch.full((model.K,), model.q, dtype=torch.float64)
        rank_parameters = 1
        noise_parameters = (
            model.K * model.D if model.psi_per_component else model.D
        )
    else:
        raise ValueError(f"unsupported model kind for BIC: {model_kind!r}")

    orientations = (ranks * (model.D - (ranks + 1.0) / 2.0)).sum()
    count = (
        model.K * model.D
        + model.K
        - 1
        + orientations.item()
        + ranks.sum().item()
        + rank_parameters
        + noise_parameters
    )
    if not float(count).is_integer():
        raise ValueError(f"non-integral model parameter count: {count}")
    return int(count)


def bic_from_mean_nll(
    model,
    model_kind: str,
    *,
    mean_nll: float,
    n: int,
) -> float:
    """Return standard minimizing BIC from a mean NLL and sample count."""
    if n <= 0:
        raise ValueError("BIC requires at least one sample")
    if not math.isfinite(mean_nll):
        raise ValueError("BIC requires a finite mean NLL")
    p = model_parameter_count(model, model_kind)
    return 2.0 * n * mean_nll + p * math.log(n)


def _model_kind(config: dict) -> str:
    name = str(config.get("model", config.get("model_kind", "mfa"))).lower()
    if "hddc" in name:
        return "hddc"
    if "ard" in name:
        return "ard"
    return "mfa"


def _load_model(run_dir: Path, model_kind: str, device: torch.device):
    model_path = run_dir / "mfa_model.pt"
    if model_kind == "hddc":
        from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc

        return load_mfa_hddc(model_path, map_location=device, device=device)
    if model_kind == "ard":
        from dalg.models.adaptive_q.mfa_ard import load_mfa_ard

        return load_mfa_ard(model_path, map_location=device, device=device)
    from dalg.models.mfa import load_mfa

    return load_mfa(model_path, map_location=device, device=device)


@torch.no_grad()
def compute_bic(
    run_dir: str | Path,
    *,
    batch_size: int = 2_048,
    device: str | torch.device = "cpu",
) -> float:
    """Return ``-2 log L + p log n`` on the run's recorded training split."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    split = json.loads((run_dir / "val_indices.json").read_text())

    shard_dir = Path(config["shard_dir"])
    layer = int(config["layer"])
    meta_index = load_meta_index(shard_dir, layer=layer)

    subset_spec = None
    run_spec_path = run_dir / "run_spec.json"
    if run_spec_path.exists():
        run_spec = json.loads(run_spec_path.read_text())
        _, subset_spec = split_shard_dir_spec(run_spec["dataset"]["shard_dir"])
    selected_positions = resolve_spec_positions(
        meta_index,
        subset_spec,
        window=int(config["window"]),
        drop_prefix=int(config["drop_prefix"]),
    )

    validation_rows = {int(row) for row in split["val_global_rows"]}
    train_positions = [
        position
        for position in selected_positions
        if int(meta_index[position]["global_row"]) not in validation_rows
    ]
    if len(train_positions) != int(split["train_rows"]):
        raise ValueError("recorded and reconstructed training splits disagree")

    data = ActivationBatchDataset(
        shard_dir,
        layer=layer,
        row_subset=train_positions,
        batch_size=batch_size,
        drop_prefix=int(config["drop_prefix"]),
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    resolved_device = torch.device(device)
    model_kind = _model_kind(config)
    model = _load_model(run_dir, model_kind, resolved_device).eval()

    log_likelihood = 0.0
    with model.inference_cache():
        for batch in data:
            log_likelihood += model.log_prob(batch.to(device)).double().sum().item()

    n = data.num_items
    mean_nll = -log_likelihood / n
    return bic_from_mean_nll(
        model,
        model_kind,
        mean_nll=mean_nll,
        n=n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(compute_bic(args.run_dir, batch_size=args.batch_size, device=args.device))


if __name__ == "__main__":
    main()


__all__ = ["bic_from_mean_nll", "compute_bic", "model_parameter_count"]
