import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm

from dalg.models.mfa import load_mfa


PEAKEDNESS_METRICS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "entropy":         lambda r: -(r * (r + 1e-8).log()).sum(dim=1),
    "one_minus_max":   lambda r: 1.0 - r.max(dim=1).values,
    "top1_minus_top2": lambda r: (
        lambda s: s[:, 0] - s[:, -1]
    )(r.topk(min(2, r.shape[1]), dim=1).values),
}


def compute_assignments(
    model_path: Path,
    loader: Any,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Single-pass streaming over `loader`. Per point, takes the argmax of the
    MFA responsibilities and accumulates:
      - cluster sizes (K,)
      - hard assignments (N,)
      - mean per-cluster peakedness for each metric in `PEAKEDNESS_METRICS`
    """
    model_path = Path(model_path)
    cache_path = model_path.parent / f"{model_path.stem}_assignments.pt"
    if cache_path.exists():
        data = torch.load(cache_path)
        if "peakedness" in data:
            return data["cluster_sizes"], data["assignments"], data["peakedness"]

    model = load_mfa(model_path, map_location="cpu").to(device)
    model.eval()
    K = model.K
    print(f"MFA: K={K} components  D={model.D}  rank={model.q}")

    sizes = torch.zeros(K, dtype=torch.long)
    all_assignments: list[torch.Tensor] = []
    peakedness_sums = {name: torch.zeros(K) for name in PEAKEDNESS_METRICS}

    with torch.no_grad():
        for batch in tqdm(loader, desc="streaming assignments + peakedness"):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            r = model.responsibilities(x)           # (B, K)
            assign = r.argmax(dim=1).cpu()
            sizes += torch.bincount(assign, minlength=K)
            all_assignments.append(assign)
            for name, fn in PEAKEDNESS_METRICS.items():
                peakedness_sums[name].scatter_add_(0, assign, fn(r).cpu())

    assignments = torch.cat(all_assignments)
    peakedness = {
        name: s / sizes.float().clamp(min=1)
        for name, s in peakedness_sums.items()
    }

    print(f"\nCluster sizes — min={sizes.min().item()}  "
          f"max={sizes.max().item()}  "
          f"mean={sizes.float().mean():.1f}  "
          f"median={sizes.float().median():.1f}")
    print(f"Empty clusters: {(sizes == 0).sum().item()}")

    torch.save({
        "cluster_sizes": sizes,
        "assignments": assignments,
        "peakedness": peakedness,
        "K": K,
    }, cache_path)

    return sizes, assignments, peakedness
