"""Build a balanced Pile subset (~100M tokens) from Timaeus's 17 Pile subsets.

For each subset, stream documents from HuggingFace, tokenize with Gemma-2B's
tokenizer (via TransformerLens HookedTransformer), and extract one random
contiguous window of WINDOW_SIZE tokens per document. Documents shorter than
WINDOW_SIZE are skipped. Rows per subset are oversampled by OVERSAMPLE to
absorb skips.

Output: an HF `datasets` Arrow directory at
  
  
with columns:
    text          : str  (raw document text)
    subset        : str  (e.g. "pile-wikipedia_en")
    token_ids     : list[int]  (length WINDOW_SIZE)
    window_start  : int
    window_end    : int
    doc_len       : int  (# tokens in the full document)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


# 17 Timaeus Pile subsets (dsir-pile-* excluded per user request).
SUBSETS = [
    "pile-arxiv",
    "pile-dm_mathematics",
    "pile-enron_emails",
    "pile-europarl",
    "pile-freelaw",
    "pile-github",
    "pile-gutenberg_pg-19",
    "pile-hackernews",
    "pile-nih_exporter",
    "pile-philpapers",
    "pile-pile-cc",
    "pile-pubmed_abstracts",
    "pile-pubmed_central",
    "pile-stackexchange",
    "pile-ubuntu_irc-broken",
    "pile-uspto_backgrounds",
    "pile-wikipedia_en",
]

TOTAL_TOKENS = 100_000_000
WINDOW_SIZE = 256
MODEL_NAME = "google/gemma-2b"
OUT_DIR = Path("/orfeo/scratch/dssc/zenocosini/pile_gemma2b_100M_windows")
OVERSAMPLE = 1.3
SEED = 0


def rows_per_subset() -> int:
    return (TOTAL_TOKENS // len(SUBSETS)) // WINDOW_SIZE


def build_subset(subset: str, tokenizer, target_rows: int, rng: random.Random):
    """Stream `subset`, tokenize, sample one window per document."""
    repo = f"timaeus/{subset}"
    ds = load_dataset(repo, split="train", streaming=True)
    budget = int(target_rows * OVERSAMPLE)
    # Filter out BOS / PAD (and any other specials defensively) from windows.
    special_ids = set()
    for tid in [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ]:
        if tid is not None:
            special_ids.add(tid)
    specials = tokenizer.all_special_ids or []
    special_ids.update(specials)
    rows = []
    seen = 0
    for ex in ds:
        if len(rows) >= target_rows or seen >= budget:
            break
        seen += 1
        text = ex["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = [t for t in ids if t not in special_ids]
        if len(ids) < WINDOW_SIZE:
            continue
        start = rng.randint(0, len(ids) - WINDOW_SIZE)
        end = start + WINDOW_SIZE
        # Store document capped at window end so the text up to and including
        # the selected window is preserved, but nothing beyond it.
        capped_text = tokenizer.decode(ids[:end], skip_special_tokens=False)
        rows.append({
            "text": capped_text,
            "subset": subset,
            "token_ids": ids[start:end],
            "window_start": start,
            "window_end": end,
            "doc_len": len(ids),
        })
    print(f"[{subset}] collected {len(rows)}/{target_rows} windows from {seen} docs")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--subsets", nargs="*", default=SUBSETS)
    ap.add_argument("--dry-run", action="store_true", help="just print config")
    args = ap.parse_args()

    target = rows_per_subset()
    print(f"subsets: {len(args.subsets)}")
    print(f"target rows/subset: {target}  (window={WINDOW_SIZE})")
    print(f"total target tokens: {target * WINDOW_SIZE * len(args.subsets):,}")
    print(f"output: {args.out}")
    if args.dry_run:
        return

    # Load only the tokenizer — matches the one TransformerLens would use
    # for gemma-2b, without loading model weights.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    args.out.mkdir(parents=True, exist_ok=True)
    shards_dir = args.out / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    shard_paths = []
    for subset in args.subsets:
        shard_path = shards_dir / subset
        if shard_path.exists():
            print(f"[{subset}] already saved at {shard_path}, skipping")
            shard_paths.append(shard_path)
            continue
        rows = build_subset(subset, tokenizer, target, rng)
        shard = Dataset.from_list(rows)
        shard.save_to_disk(str(shard_path))
        print(f"[{subset}] saved {len(rows)} rows -> {shard_path}")
        shard_paths.append(shard_path)
        del rows, shard  # free memory before next subset

    # Merge all shards into one final dataset (loaded lazily from disk).
    from datasets import concatenate_datasets, load_from_disk
    print("merging shards...")
    final = concatenate_datasets([load_from_disk(str(p)) for p in shard_paths])
    final_path = args.out / "merged"
    final.save_to_disk(str(final_path))
    print(f"saved {len(final)} total rows -> {final_path}")


if __name__ == "__main__":
    main()
