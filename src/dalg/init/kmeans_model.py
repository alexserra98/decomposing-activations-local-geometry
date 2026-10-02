"""Read KMeans model state for MFA-family initialization."""

from __future__ import annotations

from pathlib import Path

import torch

from dalg.models.kmeans import KMeans, load_kmeans, save_kmeans


def load_kmeans_initialization(
    path: str | Path, *, expected_k: int, expected_d: int,
    rank: int | None = None, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return means and optional ordered PCs without exporting centroid bundles."""
    path = Path(path)
    if path.is_dir():
        path = path / "kmeans_model.pt"
    model = load_kmeans(path, map_location=device)
    if (model.K, model.D) != (expected_k, expected_d):
        raise ValueError(
            f"KMeans checkpoint {path} has shape {(model.K, model.D)}, "
            f"expected {(expected_k, expected_d)}"
        )
    directions = None
    if rank is not None:
        if model.q_init < rank:
            raise ValueError(
                f"KMeans checkpoint {path} stores {model.q_init} initialization PCs; "
                f"initialization requires {rank}"
            )
        directions = model.W_init[:, :, :rank].contiguous()
    return model.mu, directions


def save_reservoir_initialization(
    path: Path, centroids: torch.Tensor, *, data: dict, args,
) -> None:
    """Keep reservoir fitting unchanged while recording its result as a model."""
    from dalg.init.activation_selection import resolve_initialization_rows

    _, _, _, selection = resolve_initialization_rows(
        args.shard_dir, layer=args.layer,
        val_frac=data["val_frac"], split_seed=data["split_seed"],
        drop_prefix=data["drop_prefix"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    model = KMeans.from_centroids(centroids.detach().cpu())
    save_kmeans(model, path, extra={
        "method": "reservoir_kmeans",
        "source_shard_dir": str(Path(data["shard_dir"]).resolve()),
        "layer": args.layer,
        "selection": selection,
        "K": args.K,
        "principal_components": None,
        "proj_dim": args.proj_dim,
        "pool_size": args.pool_size,
        "max_pool_size": args.max_pool_size,
        "refine_epochs": args.refine_epochs,
        "seed": args.seed,
    })
