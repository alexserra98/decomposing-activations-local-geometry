# %% [markdown]
# # Cluster Intrinsic Dimensionality
#
# Streaming pass over activations:
#   1. Hard-assign each point via argmax of MFA responsibilities.
#   2. Accumulate per-cluster peakedness metrics.
#   3. Keep at most `max_samples` activations per cluster via streaming
#      reservoir sampling.
# Then run PCA / SVD on the sampled activations for each cluster and define the
# intrinsic dimension as the smallest number of principal directions whose
# cumulative explained variance exceeds `variance_threshold`.

# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dalg.analysis.cluster_assignments import PEAKEDNESS_METRICS
from dalg.models.mfa import load_mfa


IntrinsicDimResults = dict[str, Any]


def intrinsic_dim_pca(
    X_cluster: torch.Tensor,
    *,
    threshold: float = 0.90,
    device: str | torch.device | None = None,
) -> tuple[int, torch.Tensor]:
    """
    Number of PCA directions needed to explain `threshold` of total variance.

    Returns `(intrinsic_dim, variances)` where `variances` is the descending
    variance spectrum estimated from the singular values of the centered data.
    """
    if X_cluster.shape[0] < 2:
        return 0, torch.zeros(0)

    if device is not None:
        X_cluster = X_cluster.to(device, non_blocking=True)

    X = X_cluster.float()
    X_c = X - X.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(X_c)
    var = (S ** 2).clamp(min=0)
    total = var.sum()
    if total <= 0:
        return 0, var
    cumvar = var.cumsum(0) / total
    above = (cumvar >= threshold).nonzero(as_tuple=True)[0]
    dim = int(above[0].item()) + 1 if len(above) > 0 else int(var.numel())
    return dim, var


def _update_reservoir(
    buffer: torch.Tensor | None,
    priorities: torch.Tensor | None,
    chunk: torch.Tensor,
    *,
    max_samples: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Priority-sampling reservoir update.

    Each item gets a random priority in (0, 1). Keeping the top `max_samples`
    priorities yields a uniform sample without replacement over the full stream.
    """
    chunk_priorities = torch.rand(chunk.shape[0], generator=generator)

    if buffer is None or priorities is None:
        merged_buffer = chunk
        merged_priorities = chunk_priorities
    else:
        merged_buffer = torch.cat([buffer, chunk], dim=0)
        merged_priorities = torch.cat([priorities, chunk_priorities], dim=0)

    if merged_buffer.shape[0] <= max_samples:
        return merged_buffer, merged_priorities

    keep_priorities, keep_idx = torch.topk(merged_priorities, k=max_samples, sorted=False)
    return merged_buffer[keep_idx], keep_priorities


def compute_intrinsic_dims_from_loader(
    model_path: Path,
    loader: Any,
    *,
    device: str | torch.device = "cpu",
    variance_threshold: float = 0.90,
    min_population: int = 100,
    max_samples: int = 10_000,
    store_dtype: torch.dtype = torch.float16,
    pca_device: str | torch.device | None = None,
    seed: int = 0,
    **_legacy,
) -> IntrinsicDimResults:
    """
    Streaming intrinsic-dim computation. Works with any DataLoader yielding
    either `(x, ...)` tuples or plain `x` tensors.

    The expensive full `(K, D, D)` covariance accumulation is avoided. Instead,
    each cluster keeps a capped reservoir of at most `max_samples` activations.
    """
    model_path = Path(model_path)
    model = load_mfa(model_path, map_location="cpu").to(device)
    model.eval()
    K, D, q = model.K, model.D, model.q
    print(f"MFA: K={K} components  D={D}  rank={q}")
    if pca_device is None:
        pca_device = device

    sizes = torch.zeros(K, dtype=torch.long)
    all_assignments: list[torch.Tensor] = []
    peakedness_sums = {name: torch.zeros(K) for name in PEAKEDNESS_METRICS}
    buffers: list[torch.Tensor | None] = [None for _ in range(K)]
    priorities: list[torch.Tensor | None] = [None for _ in range(K)]
    sample_sizes = torch.zeros(K, dtype=torch.long)
    rng = torch.Generator()
    rng.manual_seed(int(seed))

    with torch.no_grad():
        for batch in tqdm(loader, desc="streaming assignments + reservoir"):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True).float()
            r = model.responsibilities(x)          # (B, K)
            assign_dev = r.argmax(dim=1)
            assign = assign_dev.cpu()
            sizes += torch.bincount(assign, minlength=K)
            all_assignments.append(assign)

            for name, fn in PEAKEDNESS_METRICS.items():
                peakedness_sums[name].scatter_add_(0, assign, fn(r).cpu())

            x_cpu = x.cpu()
            for k in assign_dev.unique().tolist():
                chunk = x_cpu[assign == k].to(store_dtype)
                if chunk.shape[0] == 0:
                    continue
                buf, pri = _update_reservoir(
                    buffers[k], priorities[k], chunk,
                    max_samples=max_samples,
                    generator=rng,
                )
                buffers[k] = buf
                priorities[k] = pri
                sample_sizes[k] = buf.shape[0]

    assignments = torch.cat(all_assignments)
    peakedness = {
        name: s / sizes.float().clamp(min=1)
        for name, s in peakedness_sums.items()
    }

    torch.save({
        "cluster_sizes": sizes,
        "sample_sizes": sample_sizes,
        "assignments": assignments,
        "peakedness": peakedness,
        "K": K,
    }, model_path.parent / f"{model_path.stem}_assignments.pt")

    dims = torch.zeros(K, dtype=torch.long)
    cluster_variances: list[torch.Tensor] = [torch.zeros(0) for _ in range(K)]
    num_skipped = 0
    tag = str(pca_device)
    for k in tqdm(range(K), desc=f"per-cluster PCA ({tag})"):
        n = int(sizes[k])
        if n < min_population:
            num_skipped += 1
            continue
        if buffers[k] is None or sample_sizes[k] < 2:
            num_skipped += 1
            continue

        d, var = intrinsic_dim_pca(
            buffers[k],
            threshold=variance_threshold,
            device=pca_device,
        )
        dims[k] = d
        cluster_variances[k] = var.cpu()

    valid = dims > 0
    if valid.any():
        print(f"\nIntrinsic dims at {variance_threshold*100:.0f}% variance threshold:")
        print(f"  mean   = {dims[valid].float().mean():.2f}")
        print(f"  median = {dims[valid].float().median():.2f}")
        print(f"  min    = {dims[valid].min().item()}")
        print(f"  max    = {dims[valid].max().item()}")
        print(f"  MFA rank (q) = {q}  (reference)")
    print(f"Skipped {num_skipped} clusters with population < {min_population}")

    return {
        "intrinsic_dims": dims,
        "cluster_variances": cluster_variances,
        "cluster_sizes": sizes,
        "sample_sizes": sample_sizes,
        "assignments": assignments,
        "peakedness": peakedness,
        "variance_threshold": variance_threshold,
        "max_samples": max_samples,
        "K": K,
        "rank": q,
        "D": D,
    }


def compute_intrinsic_dims(
    model_path: Path,
    act_path: Path,
    tok_path: Path,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
    variance_threshold: float = 0.90,
    min_population: int = 100,
    max_samples: int = 10_000,
    store_dtype: torch.dtype = torch.float16,
    pca_device: str | torch.device | None = None,
    seed: int = 0,
    **_legacy,
) -> IntrinsicDimResults:
    """Monolithic-layout wrapper: loads activations.pt/tokens.pt and streams
    them through `compute_intrinsic_dims_from_loader`."""
    model_path = Path(model_path)
    act_path = Path(act_path)
    tok_path = Path(tok_path)
    X = torch.load(act_path, weights_only=True)
    tok = torch.load(tok_path, weights_only=True)
    print(f"Activations: {X.shape}  dtype={X.dtype}")

    loader = DataLoader(TensorDataset(X, tok), batch_size=batch_size, shuffle=False)
    return compute_intrinsic_dims_from_loader(
        model_path, loader,
        device=device,
        variance_threshold=variance_threshold,
        min_population=min_population,
        max_samples=max_samples,
        store_dtype=store_dtype,
        pca_device=pca_device,
        seed=seed,
    )


# ── CLI entry point ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Intrinsic dimensionality per MFA cluster")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to mfa_model.pt")
    parser.add_argument("--act-path", type=Path, required=True, help="Path to activations.pt")
    parser.add_argument("--tok-path", type=Path, required=True, help="Path to tokens.pt")
    parser.add_argument("--save-path", type=Path, default=None, help="Where to save results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pca-device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--variance-threshold", type=float, default=0.90)
    parser.add_argument("--min-population", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = compute_intrinsic_dims(
        args.model_path, args.act_path, args.tok_path,
        device=args.device,
        batch_size=args.batch_size,
        variance_threshold=args.variance_threshold,
        min_population=args.min_population,
        max_samples=args.max_samples,
        pca_device=args.pca_device,
        seed=args.seed,
    )

    save_path = args.save_path or args.model_path.parent / "intrinsic_dims.pt"
    torch.save(results, save_path)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
