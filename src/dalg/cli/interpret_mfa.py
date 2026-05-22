"""
Interpret MFA components by finding high-responsibility tokens and labeling them.

For modern shard runs this command prefers the precomputed
`mfa_model_assignments.pt` file next to the MFA model/run directory. That avoids
loading a large model just to recover examples. If assignments are missing, it
falls back to a streaming pass with `load_mfa`, which also supports component-
sharded final saves via `mfa_model_shards.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv
from tqdm import tqdm

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dalg.analysis.cluster_labeling import ORFEO_MODEL, label_mfa_clusters
from dalg.models.mfa import load_mfa

load_dotenv()


def _resolve_scratch_alias(path: str | Path | None) -> Path | None:
    """Map older direct scratch paths to the current dalg-cache location."""
    if path is None:
        return None
    path = Path(path)
    if path.exists():
        return path

    prefix = Path("/orfeo/scratch/dssc/zenocosini")
    try:
        rel = path.relative_to(prefix)
    except ValueError:
        return path
    if rel.parts and rel.parts[0] == "dalg-cache":
        return path

    candidate = prefix / "dalg-cache" / rel
    if candidate.exists() or candidate.parent.exists():
        print(f"Resolved missing scratch path {path} -> {candidate}")
        return candidate
    return path


def _model_dir(mfa_path: str | Path) -> Path:
    path = Path(mfa_path)
    if path.is_dir():
        return path
    return path.parent


def _default_assignments_path(mfa_path: str | Path) -> Path:
    return _model_dir(mfa_path) / "mfa_model_assignments.pt"


def _write_legacy_label_outputs(output: dict[str, Any], out_dir: Path) -> None:
    """Keep the old `snippets.json` / `labels.json` files for this CLI."""
    snippets = {
        cluster_id: [ex["snippet"] for ex in cluster.get("examples", [])]
        for cluster_id, cluster in output.get("clusters", {}).items()
    }
    labels = {
        cluster_id: {
            "label": cluster.get("label"),
            "notes": cluster.get("description") or cluster.get("evidence"),
            "description": cluster.get("description"),
            "evidence": cluster.get("evidence"),
        }
        for cluster_id, cluster in output.get("clusters", {}).items()
    }
    (out_dir / "snippets.json").write_text(
        json.dumps(snippets, ensure_ascii=False, indent=2)
    )
    (out_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2)
    )


def build_topk_index(
    mfa_path: str | Path,
    shard_dir: str | Path,
    layer: int,
    *,
    topk: int = 100,
    batch_size: int = 8192,
    device: str = "cuda",
    max_shards: Optional[int] = None,
) -> Dict[str, Any]:
    """Stream shards and keep per-component top-k responsibilities.

    This fallback is useful before assignments have been computed. For large
    trained runs, prefer the assignment-based path in `label_mfa_clusters`.
    """
    shard_dir = Path(shard_dir)
    cfg = json.loads((shard_dir / "config.json").read_text())
    drop_prefix = int(cfg.get("drop_prefix", 32))
    window = int(cfg["window"])
    per_row = window - drop_prefix

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"Requested device={requested}, but CUDA is not available; falling back to CPU.")
        requested = torch.device("cpu")

    mfa = load_mfa(mfa_path, map_location="cpu").to(requested).eval()
    K, D = mfa.K, mfa.D
    print(f"MFA: K={K} D={D} q={mfa.q}")

    neg_inf = torch.finfo(torch.float32).min
    g_resp = torch.full((K, topk), neg_inf, device=requested)
    g_row = torch.full((K, topk), -1, dtype=torch.long, device=requested)
    g_pos = torch.full((K, topk), -1, dtype=torch.long, device=requested)
    g_tok = torch.full((K, topk), -1, dtype=torch.long, device=requested)

    shard_paths = sorted((shard_dir / f"layer{layer:02d}").glob("shard_*.pt"))
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]
    print(f"Scanning {len(shard_paths)} shards for layer {layer}")

    meta_dir = shard_dir / "meta"
    tok_dir = shard_dir / "tokens"

    with torch.no_grad(), mfa.inference_cache():
        for shard_path in tqdm(shard_paths, desc="shards"):
            shard_i = int(shard_path.stem.split("_")[1])
            meta = json.loads((meta_dir / f"shard_{shard_i:05d}.json").read_text())
            row_indices = torch.tensor(meta["row_indices"], dtype=torch.long, device=requested)

            acts = torch.load(shard_path, mmap=True, weights_only=True)
            toks = torch.load(tok_dir / f"shard_{shard_i:05d}.pt", mmap=True, weights_only=True)

            X = acts[:, drop_prefix:, :].reshape(-1, D)
            T = toks[:, drop_prefix:].reshape(-1).long()
            N = X.shape[0]

            s_resp = torch.full((K, topk), neg_inf, device=requested)
            s_idx = torch.zeros((K, topk), dtype=torch.long, device=requested)

            for off in range(0, N, batch_size):
                xb = X[off:off + batch_size].to(
                    requested,
                    dtype=torch.float32,
                    non_blocking=(requested.type == "cuda"),
                )
                r = mfa.responsibilities(xb)
                B = xb.shape[0]
                rT = r.T.contiguous()
                ib = torch.arange(off, off + B, device=requested).unsqueeze(0).expand(K, -1)
                cat_resp = torch.cat([s_resp, rT], dim=1)
                cat_idx = torch.cat([s_idx, ib], dim=1)
                s_resp, sel = cat_resp.topk(min(topk, cat_resp.shape[1]), dim=1)
                s_idx = cat_idx.gather(1, sel)

            row_in_shard = s_idx // per_row
            tok_pos = (s_idx % per_row) + drop_prefix
            global_row = row_indices.gather(0, row_in_shard.reshape(-1)).reshape(K, topk)
            token_id = T.to(requested).gather(0, s_idx.reshape(-1)).reshape(K, topk)

            cat_resp = torch.cat([g_resp, s_resp], dim=1)
            cat_row = torch.cat([g_row, global_row], dim=1)
            cat_pos = torch.cat([g_pos, tok_pos], dim=1)
            cat_tok = torch.cat([g_tok, token_id], dim=1)
            g_resp, sel = cat_resp.topk(topk, dim=1)
            g_row = cat_row.gather(1, sel)
            g_pos = cat_pos.gather(1, sel)
            g_tok = cat_tok.gather(1, sel)

            del acts, toks, X, T

    return {
        "K": K,
        "topk": topk,
        "layer": layer,
        "drop_prefix": drop_prefix,
        "window": window,
        "resp": g_resp.cpu(),
        "global_row": g_row.cpu(),
        "tok_pos": g_pos.cpu(),
        "token_id": g_tok.cpu(),
    }


def build_cluster_snippets(
    index: Dict[str, Any],
    windows_dataset_path: str | Path,
    tokenizer_name: str,
    *,
    pad: int = 10,
    max_examples: Optional[int] = None,
    max_clusters: Optional[int] = None,
) -> Dict[int, List[str]]:
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    ds = load_from_disk(str(windows_dataset_path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    K_total = int(index["K"])
    K = min(K_total, max_clusters) if max_clusters else K_total
    n = min(int(index["topk"]), max_examples) if max_examples else int(index["topk"])
    if K < K_total:
        print(f"[debug] limiting interpretation to first {K}/{K_total} clusters")

    rows = index["global_row"][:K, :n]
    poses = index["tok_pos"][:K, :n]
    valid_rows = [int(r) for r in rows.reshape(-1).tolist() if int(r) >= 0]
    unique_rows = sorted(set(valid_rows))
    print(f"Fetching {len(unique_rows):,} unique rows from windows dataset...")
    sub = ds.select(unique_rows)
    row_cache = {r: sub[i]["token_ids"] for i, r in enumerate(unique_rows)}

    out: Dict[int, List[str]] = {}
    for k in tqdm(range(K), desc="snippets"):
        snippets: List[str] = []
        for gr, p in zip(rows[k].tolist(), poses[k].tolist()):
            gr = int(gr)
            p = int(p)
            if gr < 0 or p < 0:
                continue
            toks = row_cache[gr]
            lo, hi = max(0, p - pad), min(len(toks), p + pad + 1)
            left = tokenizer.decode(toks[lo:p], skip_special_tokens=False)
            mid = tokenizer.decode([toks[p]], skip_special_tokens=False)
            right = tokenizer.decode(toks[p + 1:hi], skip_special_tokens=False)
            snippets.append(f"{left}<target>{mid}</target>{right}")
        out[k] = snippets
    return out


SYSTEM_PROMPT = """You are an AI interpretability researcher. You will see excerpts of text where a latent feature of a language model fires strongly. In each excerpt the target token is wrapped in <target> and </target>. Determine what the feature represents: a concise concept, syntactic role, topic, or pattern common to the target tokens.

Respond with JSON only, in this exact schema:
{"label": "<5-8 word description>", "notes": "<1-2 sentence reasoning>"}
"""


def _user_prompt(snippets: List[str]) -> str:
    lines = [f"{i + 1}. {s.strip()}" for i, s in enumerate(snippets)]
    return "Excerpts:\n" + "\n".join(lines) + "\n\nReply with JSON only."


def label_clusters(
    cluster_snippets: Dict[int, List[str]],
    *,
    llm_model: str = ORFEO_MODEL,
    max_workers: int = 8,
    api_key: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def _one(k: int, snippets: List[str]):
        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(snippets)},
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfa-path", required=True)
    ap.add_argument(
        "--assignments-path",
        default=None,
        help="Optional path to mfa_model_assignments.pt. Defaults next to --mfa-path.",
    )
    ap.add_argument(
        "--shard-dir",
        required=True,
        help="Extraction output dir (contains layer{L:02d}/, tokens/, meta/)",
    )
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--windows-dataset", default=None)
    ap.add_argument("--tokenizer", default="google/gemma-2b")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--max-examples-per-cluster", type=int, default=None)
    ap.add_argument("--max-clusters", type=int, default=None)

    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--skip-topk", action="store_true")
    ap.add_argument("--skip-label", action="store_true")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--llm-model", default=ORFEO_MODEL)
    ap.add_argument("--llm-workers", type=int, default=8)
    args = ap.parse_args()

    if args.debug:
        if args.max_shards is None:
            args.max_shards = 2
        if args.max_clusters is None:
            args.max_clusters = 4
        if args.max_examples_per_cluster is None:
            args.max_examples_per_cluster = 5
        print(
            f"[debug] max_shards={args.max_shards} "
            f"max_clusters={args.max_clusters} "
            f"max_examples_per_cluster={args.max_examples_per_cluster}"
        )

    mfa_path = _resolve_scratch_alias(args.mfa_path)
    shard_dir = _resolve_scratch_alias(args.shard_dir)
    windows_dataset = _resolve_scratch_alias(args.windows_dataset)
    out_dir = _resolve_scratch_alias(args.out_dir)
    assert mfa_path is not None
    assert shard_dir is not None
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "topk_index.pt"
    assignments_path = (
        _resolve_scratch_alias(args.assignments_path)
        if args.assignments_path
        else _default_assignments_path(mfa_path)
    )
    assert assignments_path is not None

    if assignments_path.exists():
        if args.skip_topk and index_path.exists():
            print("Note: assignment-based interpretation currently recomputes topk_index.pt.")
        output = label_mfa_clusters(
            assignments_path,
            out_dir=out_dir,
            shard_dir=shard_dir,
            layer=args.layer,
            windows_dataset=windows_dataset,
            tokenizer_name=args.tokenizer,
            top_n=args.topk,
            max_examples_per_cluster=args.max_examples_per_cluster,
            pad=args.pad,
            max_clusters=args.max_clusters,
            chunk_size=args.chunk_size,
            skip_llm=args.skip_label,
            llm_model=args.llm_model,
            llm_workers=args.llm_workers,
            top_index_path=index_path,
        )
        _write_legacy_label_outputs(output, out_dir)
        print(f"Saved interpretation to {out_dir}")
        return

    print(f"No assignments file found at {assignments_path}; falling back to model scan.")
    if args.skip_topk or (index_path.exists() and not args.overwrite):
        print(f"Loading cached {index_path}")
        index = torch.load(index_path, weights_only=False)
    else:
        index = build_topk_index(
            mfa_path,
            shard_dir,
            args.layer,
            topk=args.topk,
            batch_size=args.batch_size,
            device=args.device,
            max_shards=args.max_shards,
        )
        torch.save(index, index_path)
        print(f"Saved index -> {index_path}")

    if args.skip_label or args.windows_dataset is None:
        print("Skipping LLM labeling (no --windows-dataset or --skip-label).")
        return

    snippets = build_cluster_snippets(
        index,
        windows_dataset,
        args.tokenizer,
        pad=args.pad,
        max_examples=args.max_examples_per_cluster,
        max_clusters=args.max_clusters,
    )
    snippets_path = out_dir / "snippets.json"
    snippets_path.write_text(json.dumps({str(k): v for k, v in snippets.items()}, ensure_ascii=False, indent=2))
    print(f"Saved snippets -> {snippets_path}")

    labels = label_clusters(
        snippets,
        llm_model=args.llm_model,
        max_workers=args.llm_workers,
        api_key=os.getenv("ORFEO_API_KEY"),
    )
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False, indent=2))
    print(f"Saved labels -> {labels_path}")


if __name__ == "__main__":
    main()
