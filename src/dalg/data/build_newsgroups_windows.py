"""Build a tokenized windows dataset from the 20 Newsgroups corpus.

Loads SetFit/20_newsgroups from HuggingFace (train + test splits), tokenizes
each post with the Gemma-2B tokenizer, and extracts one random contiguous
window of WINDOW_SIZE tokens per document. Documents shorter than WINDOW_SIZE
are skipped. The category label becomes the ``subset`` column, which the
downstream stratified-split and interpretation code uses.

Output: an HF ``datasets`` Arrow directory with columns:
    text          : str  (raw post text)
    subset        : str  (newsgroup category, e.g. "comp.graphics")
    token_ids     : list[int]  (length WINDOW_SIZE)
    window_start  : int
    window_end    : int
    doc_len       : int  (# tokens in the full post)

This output is schema-compatible with the Pile windows dataset, so
``dalg-run-extraction`` works unchanged.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer


WINDOW_SIZE = 128
MODEL_NAME = "google/gemma-2b"
DATASET_NAME = "SetFit/20_newsgroups"
OUT_DIR = Path("/orfeo/scratch/dssc/zenocosini/newsgroups_gemma2b_windows")
SEED = 0


def _category_names(ds) -> list[str]:
    """Return the 20 newsgroup category name strings from the dataset."""
    if "label_text" in ds.column_names:
        return sorted(set(ds["label_text"]))
    # Fall back: decode integer label via ClassLabel feature metadata.
    return ds.features["label"].names


def _get_category_name(row, label_names: list[str]) -> str:
    if "label_text" in row:
        return row["label_text"]
    return label_names[row["label"]]


def build_category(
    category: str,
    docs: list[str],
    tokenizer,
    window_size: int,
    limit: int | None,
    rng: random.Random,
) -> list[dict]:
    """Tokenize and window-sample posts from one newsgroup category.

    Each post that has at least ``window_size`` tokens after special-token
    filtering contributes exactly one row (one randomly positioned window).
    """
    special_ids: set[int] = set()
    for tid in [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ]:
        if tid is not None:
            special_ids.add(tid)
    special_ids.update(tokenizer.all_special_ids or [])

    rows = []
    for text in docs:
        if limit is not None and len(rows) >= limit:
            break
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = [t for t in ids if t not in special_ids]
        if len(ids) < window_size:
            continue
        start = rng.randint(0, len(ids) - window_size)
        end = start + window_size
        capped_text = tokenizer.decode(ids[:end], skip_special_tokens=False)
        rows.append({
            "text": capped_text,
            "subset": category,
            "token_ids": ids[start:end],
            "window_start": start,
            "window_end": end,
            "doc_len": len(ids),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a 20-Newsgroups windows dataset for MFA training.",
    )
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Max rows per category (for smoke tests)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print config and exit without writing anything")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    print(f"dataset:     {DATASET_NAME}  splits={splits}")
    print(f"window_size: {args.window_size}")
    print(f"model:       {args.model}")
    print(f"limit:       {args.limit or 'none (all docs)'}")
    print(f"output:      {args.out}")
    if args.dry_run:
        return

    print(f"loading {DATASET_NAME}...")
    raw_splits = [load_dataset(DATASET_NAME, split=s) for s in splits]
    combined = concatenate_datasets(raw_splits)

    label_names = _category_names(combined)
    print(f"categories ({len(label_names)}): {label_names}")

    # Group docs by category.
    category_docs: dict[str, list[str]] = {c: [] for c in label_names}
    for row in combined:
        cat = _get_category_name(row, label_names)
        category_docs[cat].append(row["text"])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    shards_dir = args.out / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = []
    for category in label_names:
        shard_path = shards_dir / category
        if shard_path.exists():
            print(f"[{category}] already on disk, skipping")
            shard_paths.append(shard_path)
            continue
        docs = category_docs[category]
        rows = build_category(category, docs, tokenizer, args.window_size, args.limit, rng)
        shard = Dataset.from_list(rows)
        shard.save_to_disk(str(shard_path))
        print(f"[{category}] {len(rows)}/{len(docs)} docs kept -> {shard_path}")
        shard_paths.append(shard_path)
        del rows, shard

    print("merging shards...")
    final = concatenate_datasets([load_from_disk(str(p)) for p in shard_paths])
    final_path = args.out / "merged"
    final.save_to_disk(str(final_path))
    print(f"saved {len(final)} total rows -> {final_path}")


if __name__ == "__main__":
    main()
