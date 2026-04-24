"""
Interpret MFA components by finding top-responding tokens and asking an LLM
to label each cluster.

Inputs
------
  * mfa_model.pt        — trained MFA for one layer (from dalg-run-layer train)
  * shard_dir           — sharded activations for that same layer (from
                          dalg-run-layer extract-windows). Used to compute
                          responsibilities for every (row, token) pair.
  * windows dataset     — HF dataset from dalg-build-pile-windows. Used to
                          recover ±pad context around each top token.

Pipeline
--------
  1. Single pass over the layer's shards: compute MFA responsibilities for
     every token and keep the top-K by responsibility per component.
     Output: `topk_index.pt` with tensors (K, topk) for resp / global_row /
     tok_pos / token_id.

  2. Build textual context snippets (±pad tokens around each target) from
     the windows dataset.

  3. Send each cluster's snippets to an OpenAI chat model and ask for a
     label. Output: `labels.json`.

Phases 1 and 2+3 can run independently; if `topk_index.pt` already exists it
is reused (unless --overwrite).

Example
-------
  dalg-interpret-mfa \
      --mfa-path   /.../pile_gemma2b_activations/layer05_mfa/mfa_model.pt \
      --shard-dir  /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations \
      --layer 5 \
      --windows-dataset /orfeo/scratch/dssc/zenocosini/pile_gemma2b_100M_windows/merged \
      --tokenizer google/gemma-2b \
      --out-dir   /.../pile_gemma2b_activations/layer05_mfa/interpretation \
      --topk 100 --pad 10 \
      --llm-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dalg.models.mfa import load_mfa


# ── Phase 1: top-K responsibilities over the full shard set ─────────────

def build_topk_index(
    mfa_path: str,
    shard_dir: str,
    layer: int,
    *,
    topk: int = 100,
    batch_size: int = 8192,
    device: str = "cuda",
    max_shards: Optional[int] = None,
) -> Dict[str, Any]:
    """One streaming pass over all shards. Returns per-component top-K by
    responsibility, with (global_row, tok_pos_in_window, token_id).
    """
    shard_dir = Path(shard_dir)
    cfg = json.loads((shard_dir / "config.json").read_text())
    drop_prefix = int(cfg.get("drop_prefix", 32))
    window = int(cfg["window"])
    per_row = window - drop_prefix

    mfa = load_mfa(mfa_path, map_location="cpu").to(device).eval()
    K, D = mfa.K, mfa.D
    print(f"MFA: K={K} D={D} q={mfa.q}")

    neg_inf = torch.finfo(torch.float32).min
    g_resp = torch.full((K, topk), neg_inf, device=device)
    g_row  = torch.zeros((K, topk), dtype=torch.long, device=device)
    g_pos  = torch.zeros((K, topk), dtype=torch.long, device=device)
    g_tok  = torch.zeros((K, topk), dtype=torch.long, device=device)

    shard_paths = sorted((shard_dir / f"layer{layer:02d}").glob("shard_*.pt"))
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]
    print(f"Scanning {len(shard_paths)} shards for layer {layer}")

    meta_dir = shard_dir / "meta"
    tok_dir  = shard_dir / "tokens"

    for shard_path in tqdm(shard_paths, desc="shards"):
        shard_i = int(shard_path.stem.split("_")[1])
        meta = json.loads((meta_dir / f"shard_{shard_i:05d}.json").read_text())
        row_indices = torch.tensor(meta["row_indices"], dtype=torch.long, device=device)

        acts = torch.load(shard_path, mmap=True, weights_only=True)             # (R, W, D) fp16
        toks = torch.load(tok_dir / f"shard_{shard_i:05d}.pt",
                          mmap=True, weights_only=True)                          # (R, W) int32
        R = acts.shape[0]

        X = acts[:, drop_prefix:, :].reshape(-1, D)        # (N, D)
        T = toks[:, drop_prefix:].reshape(-1).long()        # (N,)
        N = X.shape[0]

        # Shard-local running top-K. Keeping this per-shard caps peak memory
        # to O(K * (topk + batch_size)) regardless of shard size.
        s_resp = torch.full((K, topk), neg_inf, device=device)
        s_idx  = torch.zeros((K, topk), dtype=torch.long, device=device)

        with torch.no_grad():
            for off in range(0, N, batch_size):
                xb = X[off:off + batch_size].to(device, dtype=torch.float32,
                                                non_blocking=True)
                r = mfa.responsibilities(xb)                # (B, K)
                B = xb.shape[0]
                rT = r.T.contiguous()                        # (K, B)
                ib = torch.arange(off, off + B, device=device)\
                          .unsqueeze(0).expand(K, -1)
                cat_resp = torch.cat([s_resp, rT], dim=1)    # (K, topk+B)
                cat_idx  = torch.cat([s_idx,  ib], dim=1)
                s_resp, sel = cat_resp.topk(min(topk, cat_resp.shape[1]), dim=1)
                s_idx = cat_idx.gather(1, sel)

        # Decode flat shard-local idx → (row_in_shard, tok_pos_in_window, token_id)
        row_in_shard = s_idx // per_row                      # (K, topk)
        tok_in_slice = s_idx %  per_row
        tok_pos      = tok_in_slice + drop_prefix
        global_row   = row_indices.gather(0, row_in_shard.reshape(-1)).reshape(K, topk)
        token_id     = T.to(device).gather(0, s_idx.reshape(-1)).reshape(K, topk)

        # Merge shard top-K into global top-K.
        cat_resp = torch.cat([g_resp, s_resp],     dim=1)
        cat_row  = torch.cat([g_row,  global_row], dim=1)
        cat_pos  = torch.cat([g_pos,  tok_pos],    dim=1)
        cat_tok  = torch.cat([g_tok,  token_id],   dim=1)
        g_resp, sel = cat_resp.topk(topk, dim=1)
        g_row = cat_row.gather(1, sel)
        g_pos = cat_pos.gather(1, sel)
        g_tok = cat_tok.gather(1, sel)

        del acts, toks, X, T

    return {
        "K": K, "topk": topk, "layer": layer,
        "drop_prefix": drop_prefix, "window": window,
        "resp":       g_resp.cpu(),
        "global_row": g_row.cpu(),
        "tok_pos":    g_pos.cpu(),
        "token_id":   g_tok.cpu(),
    }


# ── Phase 2: build context snippets from the windows dataset ────────────

def build_cluster_snippets(
    index: Dict[str, Any],
    windows_dataset_path: str,
    tokenizer_name: str,
    *,
    pad: int = 10,
    max_examples: Optional[int] = None,
) -> Dict[int, List[str]]:
    """For every cluster, return a list of text snippets with the target
    token wrapped in ⟨⟨ … ⟩⟩ and ±pad context on each side.
    """
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    ds = load_from_disk(windows_dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    K = int(index["K"])
    topk = int(index["topk"])
    n = min(topk, max_examples) if max_examples else topk
    window = int(index["window"])

    rows = index["global_row"][:, :n]     # (K, n)
    poses = index["tok_pos"][:, :n]

    # Bulk fetch unique rows once instead of K*n random accesses.
    unique_rows = sorted(set(rows.reshape(-1).tolist()))
    print(f"Fetching {len(unique_rows):,} unique rows from windows dataset...")
    sub = ds.select(unique_rows)
    row_cache = {r: sub[i]["token_ids"] for i, r in enumerate(unique_rows)}

    out: Dict[int, List[str]] = {}
    for k in tqdm(range(K), desc="snippets"):
        snippets: List[str] = []
        for gr, p in zip(rows[k].tolist(), poses[k].tolist()):
            toks = row_cache[int(gr)]
            p = int(p)
            lo, hi = max(0, p - pad), min(window, p + pad + 1)
            left  = tokenizer.decode(toks[lo:p],   skip_special_tokens=False)
            mid   = tokenizer.decode([toks[p]],    skip_special_tokens=False)
            right = tokenizer.decode(toks[p+1:hi], skip_special_tokens=False)
            snippets.append(f"{left}⟨⟨{mid}⟩⟩{right}")
        out[k] = snippets
    return out


# ── Phase 3: LLM labeling via OpenAI ────────────────────────────────────

SYSTEM_PROMPT = """You are an AI interpretability researcher. You will see excerpts of text where a latent feature of a language model fires strongly. In each excerpt the target token is wrapped in ⟨⟨ and ⟩⟩. Determine what the feature represents — a concise concept, syntactic role, topic, or pattern common to the target tokens.

Respond with JSON only, in this exact schema:
{"label": "<5-8 word description>", "notes": "<1-2 sentence reasoning>"}
"""


def _user_prompt(snippets: List[str]) -> str:
    lines = [f"{i+1}. {s.strip()}" for i, s in enumerate(snippets)]
    return "Excerpts:\n" + "\n".join(lines) + "\n\nReply with JSON only."


def label_clusters(
    cluster_snippets: Dict[int, List[str]],
    *,
    llm_model: str = "gpt-4o-mini",
    max_workers: int = 8,
    api_key: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def _one(k: int, snippets: List[str]):
        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _user_prompt(snippets)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = resp.choices[0].message.content
            try:
                return k, json.loads(content)
            except Exception:
                return k, {"label": None, "notes": content}
        except Exception as e:
            return k, {"label": None, "notes": f"[error] {e}"}

    labels: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one, k, s) for k, s in cluster_snippets.items()]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="llm-label"):
            k, res = fut.result()
            labels[int(k)] = res
    return labels


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfa-path", required=True)
    ap.add_argument("--shard-dir", required=True,
                    help="Extraction output dir (contains layer{L:02d}/, tokens/, meta/)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--windows-dataset", default=None,
                    help="HF dataset dir from dalg-build-pile-windows (for context snippets)")
    ap.add_argument("--tokenizer", default="google/gemma-2b")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--pad", type=int, default=10,
                    help="±pad tokens of context around each target")
    ap.add_argument("--max-examples-per-cluster", type=int, default=None,
                    help="Use fewer than --topk in the LLM prompt to save tokens")

    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="Debug: cap the number of shards scanned")

    ap.add_argument("--skip-topk", action="store_true",
                    help="Reuse existing topk_index.pt (fails if missing)")
    ap.add_argument("--skip-label", action="store_true",
                    help="Only build the index/snippets; don't call the LLM")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute topk_index.pt even if it exists")

    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--llm-workers", type=int, default=8)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "topk_index.pt"

    # Phase 1.
    if args.skip_topk or (index_path.exists() and not args.overwrite):
        print(f"Loading cached {index_path}")
        index = torch.load(index_path, weights_only=False)
    else:
        index = build_topk_index(
            args.mfa_path, args.shard_dir, args.layer,
            topk=args.topk, batch_size=args.batch_size,
            device=args.device, max_shards=args.max_shards,
        )
        torch.save(index, index_path)
        print(f"Saved index → {index_path}")

    if args.skip_label or args.windows_dataset is None:
        print("Skipping LLM labeling (no --windows-dataset or --skip-label).")
        return

    # Phase 2.
    snippets = build_cluster_snippets(
        index, args.windows_dataset, args.tokenizer,
        pad=args.pad, max_examples=args.max_examples_per_cluster,
    )
    snippets_path = out_dir / "snippets.json"
    snippets_path.write_text(json.dumps(
        {str(k): v for k, v in snippets.items()}, ensure_ascii=False, indent=2))
    print(f"Saved snippets → {snippets_path}")

    # Phase 3.
    labels = label_clusters(
        snippets, llm_model=args.llm_model, max_workers=args.llm_workers,
    )
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(
        {str(k): v for k, v in labels.items()}, ensure_ascii=False, indent=2))
    print(f"Saved labels   → {labels_path}")


if __name__ == "__main__":
    main()
