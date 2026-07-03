"""Throwaway slice selector for exploratory subset runs.

A ``--shard-dir`` value may carry a ``#<spec>`` suffix, e.g.::

    dalg-cache/pile_gemma2b_activations#pile_wikipedia_100K

meaning: keep only the Wikipedia subset rows and randomly subsample them to a
~100K-token budget. The selection is deterministic (fixed seed) so every
pipeline step (training, assignments, intrinsic-dim, labeling) picks the same
rows.

To delete the feature: remove this file and revert the ``split_shard_dir_spec``
call sites back to ``Path(args.shard_dir)``.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# spec short name -> subset label baked into shard metadata
_SUBSET_ALIASES = {"wikipedia": "pile-wikipedia_en"}
_SPEC_RE = re.compile(r"^pile_(?P<subset>.+?)_(?P<num>\d+)(?P<suffix>[KkMm]?)$")
_SEED = 0  # fixed so every pipeline step selects the same rows


def split_shard_dir_spec(shard_dir_arg) -> Tuple[Path, Optional[str]]:
    """Split a ``<path>#<spec>`` value into ``(clean_path, spec_or_None)``."""
    s = str(shard_dir_arg)
    if "#" in s:
        path, spec = s.split("#", 1)
        return Path(path), spec
    return Path(s), None


def resolve_spec_positions(
    meta_index: Sequence[dict],
    spec: Optional[str],
    *,
    window: int,
    drop_prefix: int,
) -> List[int]:
    """Sorted positions into ``meta_index`` selected by ``spec``.

    Returns all positions when ``spec`` is ``None``/empty. Otherwise filters to
    the spec's subset and randomly subsamples to ``ceil(N_tokens / tokens_per_row)``
    rows, deterministically. The result is sorted so it matches the canonical
    streaming order of ``ActivationBatchDataset``.
    """
    if not spec:
        return list(range(len(meta_index)))
    m = _SPEC_RE.match(spec)
    if not m:
        raise ValueError(f"unrecognized subset spec: {spec!r}")
    subset = _SUBSET_ALIASES.get(m["subset"], f"pile-{m['subset']}")
    mult = {"k": 1_000, "m": 1_000_000}.get(m["suffix"].lower(), 1)
    n_tokens = int(m["num"]) * mult
    tokens_per_row = int(window) - int(drop_prefix)
    if tokens_per_row <= 0:
        raise ValueError(f"drop_prefix={drop_prefix} must be smaller than window={window}")
    n_rows = max(1, -(-n_tokens // tokens_per_row))  # ceil
    pool = [i for i, r in enumerate(meta_index) if r.get("subset", "all") == subset]
    if not pool:
        raise ValueError(f"no rows for subset {subset!r} in shard metadata")
    if n_rows >= len(pool):
        return pool  # already ascending
    return sorted(random.Random(_SEED).sample(pool, n_rows))
