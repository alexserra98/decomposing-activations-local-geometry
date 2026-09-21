"""Resolve activation rows for KMeans/PCA with the training split contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dalg.data.shard_activations import load_meta_index, stratified_split
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec


def resolve_initialization_rows(
    shard_dir: str | Path,
    *,
    layer: int,
    val_frac: float = 0.0,
    split_seed: int = 42,
    drop_prefix: int | None = None,
    default_drop_prefix: int = 0,
) -> tuple[Path, dict, list[int], dict]:
    """Return the clean root, shard config, training positions, and provenance."""
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be in [0, 1)")
    root, subset_spec = split_shard_dir_spec(shard_dir)
    config = json.loads((root / "config.json").read_text())
    window = int(config["window"])
    prefix = int(config.get("drop_prefix", default_drop_prefix) if drop_prefix is None else drop_prefix)
    if not 0 <= prefix < window:
        raise ValueError(f"drop_prefix={prefix} must be in [0, {window})")
    meta = load_meta_index(root, layer=layer)
    selected = resolve_spec_positions(meta, subset_spec, window=window, drop_prefix=prefix)
    train, _val = stratified_split(meta, val_frac=val_frac, seed=split_seed, positions=selected)
    if not train:
        raise ValueError("initialization training split is empty")
    selection = {
        "subset_spec": subset_spec,
        "val_frac": val_frac,
        "split_seed": split_seed,
        "drop_prefix": prefix,
        "selected_rows": len(selected),
        "train_rows": len(train),
        "selected_activations": len(selected) * (window - prefix),
        "train_activations": len(train) * (window - prefix),
        "train_rows_sha256": hashlib.sha256(
            json.dumps(train, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return root, config, train, selection
