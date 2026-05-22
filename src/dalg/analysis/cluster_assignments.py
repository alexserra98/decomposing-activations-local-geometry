import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from tqdm import tqdm

from dalg.models.mfa import load_mfa
from dalg.models.vae import load_vae


ModelType = Literal["mfa", "vae"]


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"Requested device={requested}, but CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        print(f"Requested device={requested}, but MPS is not available; falling back to CPU.")
        return torch.device("cpu")
    return requested


PEAKEDNESS_METRICS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "entropy":         lambda r: -(r * (r + 1e-8).log()).sum(dim=1),
    "one_minus_max":   lambda r: 1.0 - r.max(dim=1).values,
    "top1_minus_top2": lambda r: ( # TODO: fix this metric, the results are non senical
        lambda s: s[:, 0] - s[:, -1]
    )(r.topk(min(2, r.shape[1]), dim=1).values), 
}


def load_model(
    model_path: Path,
    *,
    model_type: ModelType = "mfa",
    map_location: str | torch.device | None = None,
):
    """Load an assignment-compatible MFA or VAE model."""
    if model_type == "mfa":
        return load_mfa(model_path, map_location=map_location)
    if model_type == "vae":
        return load_vae(model_path, map_location=map_location)
    raise ValueError(f"Unsupported model_type={model_type!r}")


def _num_components(model, *, model_type: ModelType) -> int:
    if model_type == "mfa":
        return int(model.K)
    prior = getattr(model, "prior", None)
    return int(getattr(prior, "n_components", 1))


def _describe_model(model, *, model_type: ModelType, K: int) -> None:
    if model_type == "mfa":
        print(f"MFA: K={K} components  D={model.D}  rank={model.q}")
    else:
        prior = type(model.prior).__name__
        print(
            f"VAE: K={K} prior components  D={model.input_dim}  "
            f"latent_dim={model.latent_dim}  prior={prior}"
        )


def compute_assignments(
    model_path: Path,
    loader: Any,
    *,
    model_type: ModelType = "mfa",
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
    use_inference_cache: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Single-pass streaming over `loader`. Per point, takes the argmax of the
    model responsibilities and accumulates:
      - cluster sizes (K,)
      - hard assignments (N,)
      - max responsibility per sample (N,)
      - mean per-cluster peakedness for each metric in `PEAKEDNESS_METRICS`

    For ``model_type="vae"``, responsibilities are computed over prior
    components from the encoder posterior mean.
    """
    model_path = Path(model_path)
    device = _resolve_device(device)
    model = load_model(model_path, model_type=model_type, map_location="cpu").to(device)
    model.eval()
    K = _num_components(model, model_type=model_type)
    _describe_model(model, model_type=model_type, K=K)

    sizes = torch.zeros(K, dtype=torch.long, device=device)
    peakedness_sums = {
        name: torch.zeros(K, dtype=torch.float32, device=device)
        for name in PEAKEDNESS_METRICS
    }
    assignment_chunks: list[torch.Tensor] = []
    max_resp_chunks: list[torch.Tensor] = []
    cache = (
        model.inference_cache(enabled=use_inference_cache)
        if model_type == "mfa"
        else nullcontext(model)
    )

    with torch.no_grad(), cache:
        for batch_idx, batch in enumerate(tqdm(loader, desc="streaming assignments + peakedness")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=(device.type == "cuda"))
            r = model.responsibilities(x)           # (B, K)
            top = r.max(dim=1)
            assign = top.indices                    # stays on device
            max_resp = top.values                   # (B,)
            sizes += torch.bincount(assign, minlength=K)
            assignment_chunks.append(assign.cpu())
            max_resp_chunks.append(max_resp.cpu())
            for name, fn in PEAKEDNESS_METRICS.items():
                peakedness_sums[name].scatter_add_(0, assign, fn(r).float())

    assignments = torch.cat(assignment_chunks) if assignment_chunks else torch.empty(0, dtype=torch.long)
    max_responsibilities = torch.cat(max_resp_chunks) if max_resp_chunks else torch.empty(0, dtype=torch.float32)
    sizes = sizes.cpu()
    peakedness = {
        name: s.cpu() / sizes.float().clamp(min=1)
        for name, s in peakedness_sums.items()
    }

    print(f"\nCluster sizes — min={sizes.min().item()}  "
          f"max={sizes.max().item()}  "
          f"mean={sizes.float().mean():.1f}  "
          f"median={sizes.float().median():.1f}")
    print(f"Empty clusters: {(sizes == 0).sum().item()}")

    return sizes, assignments, max_responsibilities, peakedness

def main() -> None:
    import argparse

    from torch.utils.data import DataLoader

    from dalg.data.shard_activations import (
        ActivationBatchDataset,
        load_meta_index,
    )

    parser = argparse.ArgumentParser(description="Compute model cluster assignments for sharded activations")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to mfa_model.pt or vae_model.pt")
    parser.add_argument("--model-type", choices=["mfa", "vae"], default="mfa")
    parser.add_argument("--shard-dir", type=Path, required=True, help="Directory produced by extract-windows")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to stream from shard-dir")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-prefix", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument(
        "--no-inference-cache", "--slow-responsibilities",
        dest="use_inference_cache",
        action="store_false",
        default=True,
    )
    args = parser.parse_args()
    device = _resolve_device(args.device)

    shard_dir = args.shard_dir
    extract_cfg = json.loads((shard_dir / "config.json").read_text())
    drop_prefix = args.drop_prefix
    if drop_prefix is None:
        drop_prefix = int(extract_cfg.get("drop_prefix", 32))

    meta_index = load_meta_index(shard_dir, layer=args.layer)
    positions = list(range(len(meta_index)))
    print(f"shard_dir={shard_dir}  layer={args.layer}  rows={len(positions):,}")

    ds = ActivationBatchDataset(
        shard_dir,
        layer=args.layer,
        row_subset=positions,
        drop_prefix=drop_prefix,
        batch_size=args.batch_size,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    sizes, assignments, max_responsibilities, peakedness = compute_assignments(
        args.model_path,
        loader,
        model_type=args.model_type,
        device=device,
        max_batches=args.max_batches,
        use_inference_cache=args.use_inference_cache,
    )

    save_path = args.save_path
    if save_path is None:
        if args.max_batches is None:
            save_path = args.model_path.parent / f"{args.model_path.stem}_assignments.pt"
        else:
            save_path = args.model_path.parent / f"{args.model_path.stem}_assignments_first{args.max_batches}_batches.pt"

    torch.save({
        "cluster_sizes": sizes,
        "assignments": assignments,
        "max_responsibilities": max_responsibilities,
        "peakedness": peakedness,
        "K": int(sizes.numel()),
        "model_type": args.model_type,
    }, save_path)
    print(f"Assignments saved to {save_path}")


if __name__ == "__main__":
    main()
