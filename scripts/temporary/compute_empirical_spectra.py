r"""Export full soft empirical covariance spectra for a saved toy MFA/HDDC run.

Example (from the repository root)::

    PYTHONPATH=src python scripts/temporary/compute_empirical_spectra.py \
        --checkpoint /path/to/run/mfa_model.pt --output /path/to/spectra.pt

Relative shard paths in the saved config are resolved from the repository root.
The output describes responsibilities of this checkpoint, not an earlier E-step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dalg.analysis.empirical_spectra import compute_empirical_spectra
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index, stratified_split
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec
from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc
from dalg.models.mfa import load_mfa

REPO_ROOT = Path(__file__).resolve().parents[2]


def export_spectra(checkpoint, output, *, config=None, device="cpu", batch_size=1024,
                   eig_batch_size=32):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if batch_size <= 0 or eig_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    config = Path(config).expanduser() if config is not None else checkpoint.parent / "config.json"
    config = config.resolve(strict=True)
    cfg = json.loads(config.read_text())
    model_kind = {"MFA": "mfa", "MFA_HDDC": "hddc"}.get(cfg.get("model", "MFA"))
    if model_kind is None:
        raise ValueError("Only vanilla MFA and HDDC run configurations are supported")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "meta" not in payload:
        raise ValueError("Expected a saved model export, not an optimizer/resume checkpoint")
    if "ard" in payload["meta"] or ("rank_mask" in payload["state_dict"]) != (model_kind == "hddc"):
        raise ValueError("Checkpoint model family does not match the MFA/HDDC configuration")
    del payload
    load_model = load_mfa_hddc if model_kind == "hddc" else load_mfa
    model = load_model(checkpoint, map_location="cpu")
    if (model.K, model.D, model.q) != (cfg["K"], cfg["d_model"], cfg["rank"]):
        raise ValueError("Checkpoint dimensions do not match the saved configuration")

    shard_dir, subset_spec = split_shard_dir_spec(cfg["shard_dir"])
    if not shard_dir.is_absolute():
        shard_dir = REPO_ROOT / shard_dir
    shard_dir = shard_dir.resolve(strict=True)
    extraction = json.loads((shard_dir / "config.json").read_text())
    layer, window, drop_prefix = int(cfg["layer"]), int(cfg["window"]), int(cfg["drop_prefix"])
    if window != int(extraction["window"]) or model.D != int(extraction["d_model"]):
        raise ValueError("Activation dimensions do not match the saved configuration")
    if not 0 <= drop_prefix < window:
        raise ValueError("drop_prefix must be in [0, window)")
    val_frac, split_seed = float(cfg["val_frac"]), int(cfg["split_seed"])
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be in [0, 1)")
    meta = load_meta_index(shard_dir, layer)
    positions = resolve_spec_positions(meta, subset_spec, window=window, drop_prefix=drop_prefix)
    train_positions, _ = stratified_split(meta, val_frac=val_frac, seed=split_seed, positions=positions)
    dataset = ActivationBatchDataset(
        shard_dir, layer=layer, row_subset=train_positions, batch_size=batch_size,
        drop_prefix=drop_prefix, dtype=torch.float64,
        shuffle_shards=False, shuffle_within_shard=False,
    )
    if dataset.num_items == 0:
        raise ValueError("Training split is empty")
    print(f"{model_kind}: K={model.K}, D={model.D}; {dataset.num_items:,} training activations", flush=True)
    print(f"Float64 covariance accumulator: {model.K * model.D**2 * 8 / 2**20:.1f} MiB", flush=True)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    result = compute_empirical_spectra(
        model, tqdm(iter(loader), desc="Empirical covariance batches"),
        device=device, eig_batch_size=eig_batch_size,
    )
    if result["n_activations"] != dataset.num_items:
        raise ValueError("Processed activation count does not match the full training split")
    result.update({
        "format": "dalg_empirical_spectra_v1",
        "K": model.K, "D": model.D,
        "source": {
            "checkpoint": str(checkpoint), "config": str(config), "model_kind": model_kind,
            "shard_dir": str(shard_dir), "subset_spec": subset_spec, "layer": layer,
            "window": window, "drop_prefix": drop_prefix, "split": "train",
            "val_frac": val_frac, "split_seed": split_seed, "n_train_windows": len(train_positions),
        },
        "computation": {
            "memberships": "soft", "dtype": "float64", "device": str(device),
            "batch_size": batch_size, "eig_batch_size": eig_batch_size,
            "normalization": "sum of responsibilities (ML)",
            "centering": "responsibility-weighted empirical mean",
            "covariance": "sum_n r_nk (x_n - mean_k)(x_n - mean_k)^T / sum_n r_nk",
            "eigenvalue_order": "descending",
        },
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(result, stream)
    print(f"Saved {tuple(result['eigenvalues'].shape)} spectra to {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Saved mfa_model.pt export")
    parser.add_argument("--output", required=True, type=Path, help="New spectra .pt file; never overwritten")
    parser.add_argument("--config", type=Path, help="Run config.json (default: beside checkpoint)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eig-batch-size", type=int, default=32)
    export_spectra(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
