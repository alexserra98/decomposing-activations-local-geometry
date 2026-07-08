import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import concurrent.futures as _futures
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dalg.models.mfa import load_mfa


# Slow tqdm refreshes when stderr is a non-interactive sink (e.g. SLURM logs).
_LOG_TTY = sys.stderr.isatty()
_TQDM_MININTERVAL = 0.5 if _LOG_TTY else 30.0
_TQDM_MAXINTERVAL = 10.0 if _LOG_TTY else 60.0


IntrinsicDimResults = dict[str, Any]


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _default_assignments_path(model_path: Path) -> Path:
    return model_path.parent / f"{model_path.stem}_assignments.pt"


def _resolve_assignments_path(model_path: Path | None, assignments_path: Path | None) -> Path:
    if assignments_path is not None:
        return Path(assignments_path)
    if model_path is None:
        raise ValueError("assignments_path is required when model_path is not provided")
    return _default_assignments_path(Path(model_path))


def _loader_len(loader: Any) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def _batch_x(batch: Any) -> torch.Tensor:
    return batch[0] if isinstance(batch, (list, tuple)) else batch


def _build_shard_row_pairs(meta_index: list[dict]) -> dict[int, list[tuple[int, int]]]:
    by_shard: dict[int, list[tuple[int, int]]] = {}
    for row in meta_index:
        shard = int(row["shard"])
        by_shard.setdefault(shard, []).append(
            (int(row["row_in_shard"]), int(row["global_row"]))
        )
    for pairs in by_shard.values():
        pairs.sort()
    return by_shard


def _sample_positions_to_shard_requests(
    sample_positions: torch.Tensor,
    sample_clusters: torch.Tensor,
    meta_index: list[dict],
    *,
    window: int,
    drop_prefix: int,
    num_expected_items: int,
) -> dict[int, dict[str, list[int]]]:
    """Map canonical assignment positions to shard-local row/token coordinates."""
    tokens_per_row = int(window) - int(drop_prefix)
    if tokens_per_row <= 0:
        raise ValueError(f"drop_prefix={drop_prefix} must be smaller than window={window}")

    shard_row_pairs = _build_shard_row_pairs(meta_index)
    requests: dict[int, dict[str, list[int]]] = {}
    cursor = 0
    stream_offset = 0
    total_selected = int(sample_positions.numel())

    for shard_i in tqdm(
        sorted(shard_row_pairs),
        desc="mapping samples to shards",
        mininterval=_TQDM_MININTERVAL,
        maxinterval=_TQDM_MAXINTERVAL,
    ):
        pairs = shard_row_pairs[shard_i]
        next_offset = stream_offset + len(pairs) * tokens_per_row
        right = int(torch.searchsorted(sample_positions, next_offset, right=False).item())

        if right > cursor:
            positions = sample_positions[cursor:right]
            clusters = sample_clusters[cursor:right]
            shard_flat = (positions - stream_offset).long()
            row_offsets = torch.div(shard_flat, tokens_per_row, rounding_mode="floor")
            tok_pos = drop_prefix + (shard_flat % tokens_per_row)

            req = requests.setdefault(shard_i, {"rows": [], "tok_pos": [], "clusters": []})
            for row_offset, tok, cluster in zip(row_offsets.tolist(), tok_pos.tolist(), clusters.tolist()):
                row_in_shard, _global_row = pairs[int(row_offset)]
                req["rows"].append(int(row_in_shard))
                req["tok_pos"].append(int(tok))
                req["clusters"].append(int(cluster))

            cursor = right

        stream_offset = next_offset

    if stream_offset != num_expected_items:
        raise ValueError(
            f"Reconstructed stream length ({stream_offset:,}) does not match assignments "
            f"length ({num_expected_items:,}). Check drop_prefix and shard metadata."
        )
    if cursor != total_selected:
        raise ValueError(
            f"Mapped {cursor:,}/{total_selected:,} sampled positions. "
            "The reconstructed stream ended before all sampled positions were seen."
        )

    return requests


def _collect_sampled_shard_activations(
    shard_dir: Path,
    layer: int,
    requests: dict[int, dict[str, list[int]]],
    *,
    K: int,
    store_dtype: torch.dtype,
) -> list[torch.Tensor | None]:
    chunks: list[list[torch.Tensor]] = [[] for _ in range(K)]

    for shard_i in tqdm(
        sorted(requests),
        desc="loading sampled shards",
        mininterval=_TQDM_MININTERVAL,
        maxinterval=_TQDM_MAXINTERVAL,
    ):
        req = requests[shard_i]
        rows = torch.tensor(req["rows"], dtype=torch.long)
        tok_pos = torch.tensor(req["tok_pos"], dtype=torch.long)
        clusters = torch.tensor(req["clusters"], dtype=torch.long)

        shard_path = shard_dir / f"layer{layer:02d}" / f"shard_{shard_i:05d}.pt"
        acts = torch.load(shard_path, mmap=True, weights_only=True)
        x_selected = acts[rows, tok_pos].to(store_dtype).cpu()

        for k in torch.unique(clusters).tolist():
            mask = clusters == int(k)
            chunks[int(k)].append(x_selected[mask].contiguous())

        del acts, x_selected

    return [torch.cat(parts, dim=0) if parts else None for parts in chunks]


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


def _load_assignments(assignments_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], int, dict[str, Any]]:
    data = torch.load(assignments_path, map_location="cpu", weights_only=True)
    if "assignments" not in data or "cluster_sizes" not in data:
        raise ValueError(
            f"{assignments_path} must contain at least 'assignments' and 'cluster_sizes'."
        )

    assignments = data["assignments"].long().cpu()
    sizes = data["cluster_sizes"].long().cpu()
    peakedness = data.get("peakedness", {})
    K = int(data.get("K", sizes.numel()))

    if sizes.numel() != K:
        raise ValueError(f"K={K}, but cluster_sizes has shape {tuple(sizes.shape)}")
    if assignments.numel() != int(sizes.sum().item()):
        print(
            "Warning: assignments length does not match cluster_sizes.sum(); "
            "using assignments to choose samples and saved cluster_sizes for reporting."
        )

    metadata = {
        "subset_spec": data.get("subset_spec"),
        "source": data.get("source", {}),
        "centroids_path": data.get("centroids_path"),
    }
    return assignments, sizes, peakedness, K, metadata


def _assignment_subset_spec(metadata: dict[str, Any]) -> str | None:
    subset_spec = metadata.get("subset_spec")
    source = metadata.get("source")
    if subset_spec is None and isinstance(source, dict):
        subset_spec = source.get("subset_spec")
    return str(subset_spec) if subset_spec else None


def _validate_assignment_source(
    metadata: dict[str, Any],
    *,
    layer: int | None = None,
    drop_prefix: int | None = None,
) -> None:
    source = metadata.get("source")
    if not isinstance(source, dict):
        return

    if layer is not None and source.get("layer") is not None and int(source["layer"]) != int(layer):
        raise ValueError(
            f"Assignment file was computed for layer={source['layer']}, "
            f"but intrinsic-dim was requested for layer={layer}."
        )
    if (
        drop_prefix is not None
        and source.get("drop_prefix") is not None
        and int(source["drop_prefix"]) != int(drop_prefix)
    ):
        raise ValueError(
            f"Assignment file was computed with drop_prefix={source['drop_prefix']}, "
            f"but intrinsic-dim was requested with drop_prefix={drop_prefix}."
        )


def _load_model_metadata(model_path: Path | None, K: int) -> dict[str, Any]:
    if model_path is None:
        return {"model_kind": "assignments", "model_path": None, "D": None, "rank": None}

    mfa = load_mfa(model_path, map_location="cpu")
    if int(mfa.K) != K:
        raise ValueError(f"Assignment file has K={K}, but model has K={mfa.K}")

    return {
        "model_kind": "mfa",
        "model_path": str(model_path),
        "D": int(mfa.D),
        "rank": int(mfa.q),
    }


def _choose_sample_positions(
    assignments: torch.Tensor,
    sizes: torch.Tensor,
    *,
    K: int,
    max_samples: int,
    min_population: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Uniformly sample up to `max_samples` stream positions per cluster.

    The output positions are sorted by stream order. `sample_clusters[i]` is the
    cluster id for `sample_positions[i]`.
    """
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    counts = torch.bincount(assignments, minlength=K).long()
    order = torch.argsort(assignments)
    offsets = torch.zeros(K + 1, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)

    sample_sizes = torch.zeros(K, dtype=torch.long)
    position_chunks: list[torch.Tensor] = []
    cluster_chunks: list[torch.Tensor] = []
    rng = torch.Generator()
    rng.manual_seed(int(seed))

    for k in tqdm(
        range(K),
        desc="sampling assignment positions",
        mininterval=_TQDM_MININTERVAL,
        maxinterval=_TQDM_MAXINTERVAL,
    ):
        n = int(counts[k].item())
        if n < min_population or n < 2:
            continue

        start = int(offsets[k].item())
        end = int(offsets[k + 1].item())
        cluster_positions = order[start:end]

        if n > max_samples:
            keep = torch.randperm(n, generator=rng)[:max_samples]
            cluster_positions = cluster_positions[keep]

        sample_sizes[k] = int(cluster_positions.numel())
        position_chunks.append(cluster_positions)
        cluster_chunks.append(
            torch.full((cluster_positions.numel(),), k, dtype=torch.long)
        )

    if not position_chunks:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            sample_sizes,
        )

    sample_positions = torch.cat(position_chunks).long()
    sample_clusters = torch.cat(cluster_chunks).long()
    by_stream_order = torch.argsort(sample_positions)
    return sample_positions[by_stream_order], sample_clusters[by_stream_order], sample_sizes


def _collect_sampled_activations(
    loader: Any,
    sample_positions: torch.Tensor,
    sample_clusters: torch.Tensor,
    *,
    K: int,
    num_expected_items: int,
    store_dtype: torch.dtype,
) -> list[torch.Tensor | None]:
    """Stream activations once and collect only the sampled assignment positions."""
    chunks: list[list[torch.Tensor]] = [[] for _ in range(K)]
    cursor = 0
    stream_offset = 0
    total_selected = int(sample_positions.numel())

    if total_selected == 0:
        return [None for _ in range(K)]

    progress = tqdm(
        loader,
        total=_loader_len(loader),
        desc="collecting sampled activations",
        mininterval=_TQDM_MININTERVAL,
        maxinterval=_TQDM_MAXINTERVAL,
    )
    for batch in progress:
        x = _batch_x(batch)
        batch_size = int(x.shape[0])
        next_offset = stream_offset + batch_size

        right = int(torch.searchsorted(sample_positions, next_offset, right=False).item())
        if right > cursor:
            positions = sample_positions[cursor:right]
            rel = (positions - stream_offset).long()
            x_selected = x.detach().cpu().index_select(0, rel).to(store_dtype)
            c_selected = sample_clusters[cursor:right]

            for k in torch.unique(c_selected).tolist():
                mask = c_selected == int(k)
                chunks[int(k)].append(x_selected[mask].contiguous())

            cursor = right

        stream_offset = next_offset
        if cursor >= total_selected:
            # Keep consuming only if the caller needs length validation.
            pass

    if stream_offset != num_expected_items:
        raise ValueError(
            f"Activation stream length ({stream_offset:,}) does not match assignments "
            f"length ({num_expected_items:,}). Make sure batch size, num workers, "
            "drop_prefix, layer, and shard ordering match the assignment run."
        )
    if cursor != total_selected:
        raise ValueError(
            f"Collected {cursor:,}/{total_selected:,} sampled activations. "
            "The activation stream ended before all sampled positions were seen."
        )

    buffers: list[torch.Tensor | None] = []
    for parts in chunks:
        buffers.append(torch.cat(parts, dim=0) if parts else None)
    return buffers


def _run_cluster_pca(
    buffers: list[torch.Tensor | None],
    sizes: torch.Tensor,
    *,
    threshold: float,
    min_population: int,
    pca_device: str | torch.device | None,
    pca_workers: int,
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    K = len(buffers)
    dims = torch.zeros(K, dtype=torch.long)
    cluster_variances: list[torch.Tensor] = [torch.zeros(0) for _ in range(K)]

    valid_clusters = [
        k for k in range(K)
        if int(sizes[k]) >= min_population and buffers[k] is not None and buffers[k].shape[0] >= 2
    ]
    num_skipped = K - len(valid_clusters)

    def _one(k: int) -> tuple[int, int, torch.Tensor]:
        d, var = intrinsic_dim_pca(
            buffers[k],
            threshold=threshold,
            device=pca_device,
        )
        return k, d, var.cpu()

    use_threads = (
        pca_workers > 1
        and (pca_device is None or str(pca_device).startswith("cpu"))
    )

    if use_threads:
        old_threads = torch.get_num_threads()
        torch.set_num_threads(max(1, old_threads // pca_workers))
        try:
            with _futures.ThreadPoolExecutor(max_workers=pca_workers) as pool:
                futures = [pool.submit(_one, k) for k in valid_clusters]
                for fut in tqdm(
                    _futures.as_completed(futures),
                    total=len(futures),
                    desc=f"per-cluster PCA (cpu x{pca_workers})",
                    mininterval=_TQDM_MININTERVAL,
                    maxinterval=_TQDM_MAXINTERVAL,
                ):
                    k, d, var = fut.result()
                    dims[k] = d
                    cluster_variances[k] = var
        finally:
            torch.set_num_threads(old_threads)
    else:
        tag = str(pca_device)
        for k in tqdm(
            valid_clusters,
            desc=f"per-cluster PCA ({tag})",
            mininterval=_TQDM_MININTERVAL,
            maxinterval=_TQDM_MAXINTERVAL,
        ):
            _, d, var = _one(k)
            dims[k] = d
            cluster_variances[k] = var

    return dims, cluster_variances, num_skipped


def compute_intrinsic_dims_from_assignments(
    model_path: Path | None,
    loader: Any,
    assignments_path: Path | None = None,
    *,
    device: str | torch.device = "cpu",
    variance_threshold: float = 0.90,
    min_population: int = 100,
    max_samples: int = 10_000,
    store_dtype: torch.dtype = torch.float16,
    pca_device: str | torch.device | None = None,
    pca_workers: int = 1,
    seed: int = 0,
    **_legacy,
) -> IntrinsicDimResults:
    """
    Compute per-cluster intrinsic dimensions from precomputed assignments.

    `assignments` are interpreted as positions in the activation stream. For
    sharded data, prefer `compute_intrinsic_dims_from_shards`, which maps those
    positions through `meta_index` directly.
    """
    model_path = Path(model_path) if model_path is not None else None
    assignments_path = _resolve_assignments_path(model_path, assignments_path)
    if pca_device is None:
        pca_device = device

    assignments, sizes, peakedness, K, assignment_metadata = _load_assignments(assignments_path)
    model_metadata = _load_model_metadata(model_path, K)

    if model_metadata["model_kind"] == "mfa":
        print(
            f"MFA: K={K} components  D={model_metadata['D']}  "
            f"rank={model_metadata['rank']}"
        )
    else:
        print(f"Assignments-only clusters: K={K}")
    print(f"Assignments: {assignments_path}  N={assignments.numel():,}")
    print(
        f"Sampling up to {max_samples:,} activations per cluster "
        f"(min_population={min_population:,})."
    )

    sample_positions, sample_clusters, sample_sizes = _choose_sample_positions(
        assignments,
        sizes,
        K=K,
        max_samples=max_samples,
        min_population=min_population,
        seed=seed,
    )
    print(f"Selected {sample_positions.numel():,} activation vectors for PCA.")

    buffers = _collect_sampled_activations(
        loader,
        sample_positions,
        sample_clusters,
        K=K,
        num_expected_items=int(assignments.numel()),
        store_dtype=store_dtype,
    )

    dims, cluster_variances, num_skipped = _run_cluster_pca(
        buffers,
        sizes,
        threshold=variance_threshold,
        min_population=min_population,
        pca_device=pca_device,
        pca_workers=pca_workers,
    )

    valid = dims > 0
    if valid.any():
        print(f"\nIntrinsic dims at {variance_threshold*100:.0f}% variance threshold:")
        print(f"  mean   = {dims[valid].float().mean():.2f}")
        print(f"  median = {dims[valid].float().median():.2f}")
        print(f"  min    = {dims[valid].min().item()}")
        print(f"  max    = {dims[valid].max().item()}")
        if model_metadata["rank"] is not None:
            print(f"  MFA rank (q) = {model_metadata['rank']}  (reference)")
    print(f"Skipped {num_skipped} clusters with population < {min_population}")

    return {
        "intrinsic_dims": dims,
        "cluster_variances": cluster_variances,
        "cluster_sizes": sizes,
        "sample_sizes": sample_sizes,
        "peakedness": peakedness,
        "variance_threshold": variance_threshold,
        "max_samples": max_samples,
        "assignments_path": str(assignments_path),
        "K": K,
        "rank": model_metadata["rank"],
        "D": model_metadata["D"],
        "model_kind": model_metadata["model_kind"],
        "model_path": model_metadata["model_path"],
        "assignment_metadata": assignment_metadata,
    }

#TODO remove model dead code
def compute_intrinsic_dims_from_shards(
    model_path: Path | None,
    shard_dir: Path,
    *,
    layer: int,
    assignments_path: Path | None = None,
    drop_prefix: int | None = None,
    subset_spec: str | None = None,
    device: str | torch.device = "cpu",
    variance_threshold: float = 0.90,
    min_population: int = 100,
    max_samples: int = 10_000,
    store_dtype: torch.dtype = torch.float16,
    pca_device: str | torch.device | None = None,
    pca_workers: int = 1,
    seed: int = 0,
    **_legacy,
) -> IntrinsicDimResults:
    from dalg.data.shard_activations import load_meta_index
    from dalg.data.subset_spec import resolve_spec_positions

    model_path = Path(model_path) if model_path is not None else None
    shard_dir = Path(shard_dir)
    assignments_path = _resolve_assignments_path(model_path, assignments_path)
    if pca_device is None:
        pca_device = device

    extract_cfg = json.loads((shard_dir / "config.json").read_text())
    window = int(extract_cfg["window"])
    if drop_prefix is None:
        drop_prefix = int(extract_cfg.get("drop_prefix", 32))

    assignments, sizes, peakedness, K, assignment_metadata = _load_assignments(assignments_path)
    _validate_assignment_source(assignment_metadata, layer=layer, drop_prefix=drop_prefix)

    meta_index = load_meta_index(shard_dir, layer=layer)
    effective_subset_spec = subset_spec if subset_spec is not None else _assignment_subset_spec(assignment_metadata)
    if effective_subset_spec:
        keep = resolve_spec_positions(
            meta_index, effective_subset_spec, window=window, drop_prefix=drop_prefix
        )
        meta_index = [meta_index[i] for i in keep]
    model_metadata = _load_model_metadata(model_path, K)

    if model_metadata["model_kind"] == "mfa":
        print(
            f"MFA: K={K} components  D={model_metadata['D']}  "
            f"rank={model_metadata['rank']}"
        )
    else:
        print(f"Assignments-only clusters: K={K}")
    print(f"Assignments: {assignments_path}  N={assignments.numel():,}")
    print(
        f"shard_dir={shard_dir}  layer={layer}  rows={len(meta_index):,}  "
        f"window={window}  drop_prefix={drop_prefix}"
        + (f"  spec={effective_subset_spec!r}" if effective_subset_spec else "")
    )
    print("Mapping assignment indices in canonical shard order from meta_index.")

    sample_positions, sample_clusters, sample_sizes = _choose_sample_positions(
        assignments,
        sizes,
        K=K,
        max_samples=max_samples,
        min_population=min_population,
        seed=seed,
    )
    print(f"Selected {sample_positions.numel():,} activation vectors for PCA.")

    requests = _sample_positions_to_shard_requests(
        sample_positions,
        sample_clusters,
        meta_index,
        window=window,
        drop_prefix=drop_prefix,
        num_expected_items=int(assignments.numel()),
    )
    print(f"Sampled activations touch {len(requests):,} shard files.")

    buffers = _collect_sampled_shard_activations(
        shard_dir,
        layer,
        requests,
        K=K,
        store_dtype=store_dtype,
    )

    dims, cluster_variances, num_skipped = _run_cluster_pca(
        buffers,
        sizes,
        threshold=variance_threshold,
        min_population=min_population,
        pca_device=pca_device,
        pca_workers=pca_workers,
    )

    valid = dims > 0
    if valid.any():
        print(f"\nIntrinsic dims at {variance_threshold*100:.0f}% variance threshold:")
        print(f"  mean   = {dims[valid].float().mean():.2f}")
        print(f"  median = {dims[valid].float().median():.2f}")
        print(f"  min    = {dims[valid].min().item()}")
        print(f"  max    = {dims[valid].max().item()}")
        if model_metadata["rank"] is not None:
            print(f"  MFA rank (q) = {model_metadata['rank']}  (reference)")
    print(f"Skipped {num_skipped} clusters with population < {min_population}")

    return {
        "intrinsic_dims": dims,
        "cluster_variances": cluster_variances,
        "cluster_sizes": sizes,
        "sample_sizes": sample_sizes,
        "peakedness": peakedness,
        "variance_threshold": variance_threshold,
        "max_samples": max_samples,
        "assignments_path": str(assignments_path),
        "K": K,
        "rank": model_metadata["rank"],
        "D": model_metadata["D"],
        "model_kind": model_metadata["model_kind"],
        "model_path": model_metadata["model_path"],
        "assignment_metadata": assignment_metadata,
        "subset_spec": effective_subset_spec,
    }


def compute_intrinsic_dims_from_loader(
    model_path: Path | None,
    loader: Any,
    *,
    assignments_path: Path | None = None,
    device: str | torch.device = "cpu",
    variance_threshold: float = 0.90,
    min_population: int = 100,
    max_samples: int = 10_000,
    store_dtype: torch.dtype = torch.float16,
    pca_device: str | torch.device | None = None,
    pca_workers: int = 1,
    seed: int = 0,
    **legacy,
) -> IntrinsicDimResults:
    """Backward-compatible wrapper around the assignment-file implementation."""
    return compute_intrinsic_dims_from_assignments(
        model_path,
        loader,
        assignments_path,
        device=device,
        variance_threshold=variance_threshold,
        min_population=min_population,
        max_samples=max_samples,
        store_dtype=store_dtype,
        pca_device=pca_device,
        pca_workers=pca_workers,
        seed=seed,
        **legacy,
    )


def compute_intrinsic_dims(
    model_path: Path | None,
    act_path: Path,
    tok_path: Path | None = None,
    *,
    assignments_path: Path | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
    variance_threshold: float = 0.90,
    min_population: int = 100,
    max_samples: int = 10_000,
    store_dtype: torch.dtype = torch.float16,
    pca_device: str | torch.device | None = None,
    pca_workers: int = 1,
    seed: int = 0,
    **legacy,
) -> IntrinsicDimResults:
    """Monolithic-layout wrapper for activations.pt plus precomputed assignments."""
    X = torch.load(act_path, weights_only=True)
    print(f"Activations: {X.shape}  dtype={X.dtype}")

    if tok_path is not None and Path(tok_path).exists():
        tok = torch.load(tok_path, weights_only=True)
        loader = DataLoader(TensorDataset(X, tok), batch_size=batch_size, shuffle=False)
    else:
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

    return compute_intrinsic_dims_from_assignments(
        model_path,
        loader,
        assignments_path,
        device=device,
        variance_threshold=variance_threshold,
        min_population=min_population,
        max_samples=max_samples,
        store_dtype=store_dtype,
        pca_device=pca_device,
        pca_workers=pca_workers,
        seed=seed,
        **legacy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intrinsic dimensionality per cluster from saved assignments"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to mfa_model.pt for K validation and MFA rank metadata",
    )
    parser.add_argument(
        "--assignments-path",
        type=Path,
        default=None,
        help="Path to an assignments .pt file (default: next to --model-path)",
    )
    parser.add_argument("--shard-dir", type=Path, default=None, help="Shard directory from extract-windows")
    parser.add_argument("--layer", type=int, default=None, help="Layer index for --shard-dir")
    parser.add_argument("--drop-prefix", type=int, default=None)
    parser.add_argument("--act-path", type=Path, default=None, help="Monolithic activations.pt")
    parser.add_argument("--tok-path", type=Path, default=None, help="Optional monolithic tokens.pt")
    parser.add_argument("--save-path", type=Path, default=None, help="Where to save results")
    parser.add_argument("--device", default="cpu", help="Default PCA device if --pca-device is omitted")
    parser.add_argument("--pca-device", default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1024)
    parser.add_argument("--variance-threshold", type=float, default=0.90)
    parser.add_argument("--min-population", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--pca-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--store-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Dtype used to store sampled activations before PCA",
    )
    args = parser.parse_args()

    if args.model_path is None and args.assignments_path is None:
        raise SystemExit("--assignments-path is required when --model-path is omitted.")

    if args.shard_dir is not None:
        if args.layer is None:
            raise SystemExit("--layer is required with --shard-dir")
        results = compute_intrinsic_dims_from_shards(
            args.model_path,
            args.shard_dir,
            layer=args.layer,
            assignments_path=args.assignments_path,
            drop_prefix=args.drop_prefix,
            device=args.device,
            variance_threshold=args.variance_threshold,
            min_population=args.min_population,
            max_samples=args.max_samples,
            store_dtype=_dtype_from_name(args.store_dtype),
            pca_device=args.pca_device,
            pca_workers=args.pca_workers,
            seed=args.seed,
        )
    else:
        if args.act_path is None:
            raise SystemExit("Provide either --shard-dir/--layer or --act-path.")
        results = compute_intrinsic_dims(
            args.model_path,
            args.act_path,
            args.tok_path,
            assignments_path=args.assignments_path,
            device=args.device,
            batch_size=args.batch_size,
            variance_threshold=args.variance_threshold,
            min_population=args.min_population,
            max_samples=args.max_samples,
            store_dtype=_dtype_from_name(args.store_dtype),
            pca_device=args.pca_device,
            pca_workers=args.pca_workers,
            seed=args.seed,
        )

    if args.save_path is not None:
        save_path = args.save_path
    elif args.model_path is not None:
        save_path = args.model_path.parent / "intrinsic_dims.pt"
    elif args.assignments_path is not None:
        save_path = args.assignments_path.parent / "intrinsic_dims.pt"
    else:
        raise SystemExit("--assignments-path is required when --model-path is omitted.")

    torch.save(results, save_path)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
