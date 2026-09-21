"""Initialize mixture proportions from the training split's KMeans partition."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from tqdm import tqdm

from dalg.init.projected_knn import KMeansTorch


@torch.no_grad()
def initialize_mixture_weights(
    model,
    centroids: torch.Tensor,
    train_loader: Iterable[torch.Tensor],
    *,
    n_train_tokens: int,
) -> torch.Tensor:
    """Set logits to log(max(n_k, 1)), normalized by the global softmax.

    Count nearest Euclidean centroids over the entire supplied training loader.
    For component sharding, pass the unwrapped base loader and global centroids;
    rank zero counts once and broadcasts counts before each rank takes its slice.
    Each empty cluster receives one pseudocount for weight initialization only.
    Returns the raw global counts on the model device, without pseudocounts.
    """
    if n_train_tokens <= 0:
        raise ValueError("mixture initialization requires a positive training-token count")
    device = model.pi_logits.device
    centers = centroids.to(device=device, dtype=model.mu.dtype)
    K = centers.shape[0]
    sharded = hasattr(model, "component_start")
    distributed = sharded and dist.is_available() and dist.is_initialized()
    is_main = not distributed or dist.get_rank() == 0
    counts = torch.zeros(K, dtype=torch.long, device=device)
    if is_main:
        kmeans = KMeansTorch(k=K, metric="euclidean", device=device)
        with tqdm(total=n_train_tokens, desc="Initial mixture proportions", unit="tokens") as progress:
            for batch in train_loader:
                points = batch.to(device=device, dtype=centers.dtype)
                labels = kmeans._assign_streamed(points, centers)
                counts += torch.bincount(labels, minlength=K)
                progress.update(points.shape[0])
    if distributed:
        dist.broadcast(counts, src=0)

    if int(counts.sum()) != n_train_tokens:
        raise ValueError(
            f"mixture initialization counted {int(counts.sum())} training tokens, "
            f"expected {n_train_tokens}"
        )
    empty = (counts == 0).nonzero(as_tuple=True)[0]
    if is_main and empty.numel():
        print(
            f"[init] giving {empty.numel()} empty training clusters one pseudocount each "
            "for mixture weights; "
            f"empty cluster ids (first 10): {empty[:10].tolist()}"
        )
    start = model.component_start if sharded else 0
    local_counts = counts[start:start + model.K].clamp_min(1)
    model.pi_logits.copy_(local_counts.to(model.pi_logits.dtype).log())
    if is_main:
        print(
            f"[init] pi_k = max(n_k, 1)/(N + n_empty) from {n_train_tokens:,} training tokens; "
            f"n_empty={empty.numel()}; raw cluster sizes={int(counts.min())}..{int(counts.max())}"
        )
    return counts
