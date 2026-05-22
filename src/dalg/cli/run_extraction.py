"""
CLI for extracting activations into on-disk shards.

Reads a pre-tokenized HF windows dataset (produced by
``dalg-build-pile-windows``) and runs one TransformerLens forward pass per
batch, caching the requested layers. Output is sharded and resume-safe:
re-running skips shards that already exist on disk.

Output layout::

    <out_dir>/config.json
    <out_dir>/progress.json
    <out_dir>/layer{L:02d}/shard_{i:05d}.pt   # (rows, window, d_model)
    <out_dir>/tokens/shard_{i:05d}.pt         # int32, (rows, window)
    <out_dir>/meta/shard_{i:05d}.json

Example::

    dalg-run-extraction \
        --dataset /path/to/windows --out-dir /path/to/activations \
        --model google/gemma-2b --layers 5 17 --dtype float16 \
        --extract-batch-size 16 --shard-size 512 --device cuda
"""
import os
import json
import time
import argparse
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch


HOOK_FN = {
    "residual":     lambda L: ("resid_post", L),
    "residual_pre": lambda L: ("resid_pre", L),
    "mlp":          lambda L: f"blocks.{L}.mlp.hook_post",
    "mlp_out":      lambda L: f"blocks.{L}.hook_mlp_out",
    "attn_out":     lambda L: f"blocks.{L}.hook_attn_out",
}


def _hook_name(mode: str, L: int) -> str:
    from transformer_lens import utils as tl_utils
    spec = HOOK_FN[mode](L)
    if isinstance(spec, tuple):
        return tl_utils.get_act_name(*spec)
    return spec


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fmt_eta(sec: float) -> str:
    s = int(sec)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def cmd_extract_windows(args) -> None:
    """Run the multi-layer activation extraction loop.

    Iterates the input dataset in shards of ``shard_size`` rows. For each
    shard the model is run on every batch, the requested layer activations
    are written to ``layer{L:02d}/shard_{i:05d}.pt``, the truncated token
    ids to ``tokens/shard_{i:05d}.pt``, and the per-row metadata to
    ``meta/shard_{i:05d}.json``. ``progress.json`` is updated after each
    shard so the run is resume-safe.
    """
    from torch.utils.data import DataLoader
    from datasets import load_from_disk
    from transformer_lens import HookedTransformer

    out_dir = Path(args.out_dir)
    if args.debug:
        args.shard_size = min(args.shard_size, 16)
        args.extract_batch_size = min(args.extract_batch_size, 4)
        _log(f"DEBUG: shard_size={args.shard_size} batch={args.extract_batch_size} limit={args.limit}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokens").mkdir(exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)
    for L in args.layers:
        (out_dir / f"layer{L:02d}").mkdir(exist_ok=True)

    torch_dtype = {"float16": torch.float16,
                   "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[args.dtype]

    _log(f"loading dataset: {args.dataset}")
    ds = load_from_disk(args.dataset)
    if args.debug:
        ds = ds.select(range(min(args.limit, len(ds))))
    N = len(ds)
    window = len(ds[0]["token_ids"])
    num_shards = (N + args.shard_size - 1) // args.shard_size
    _log(f"rows={N}  window={window}  shards={num_shards}  layers={args.layers}  drop_prefix={args.drop_prefix}")

    def shard_done(i):
        if not (out_dir / "tokens" / f"shard_{i:05d}.pt").exists():
            return False
        return all(
            (out_dir / f"layer{L:02d}" / f"shard_{i:05d}.pt").exists()
            for L in args.layers
        )

    todo = [i for i in range(num_shards) if not shard_done(i)]
    _log(f"resume: {num_shards - len(todo)}/{num_shards} shards on disk, {len(todo)} to do")
    if not todo:
        _log("nothing to do — exiting")
        return

    _log(f"loading model {args.model} on {args.device} dtype={args.dtype}")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=torch_dtype,
    )
    model.eval()
    bos_id = model.tokenizer.bos_token_id
    d_model = model.cfg.d_model
    _log(f"model loaded in {time.time()-t0:.1f}s  d_model={d_model}  bos={bos_id}")

    hook_names = {L: _hook_name(args.mode, L) for L in args.layers}
    hook_set = set(hook_names.values())
    _log(f"hook names: {hook_names}")

    config = {
        "model": args.model, "mode": args.mode, "layers": list(args.layers),
        "window": window, "d_model": d_model, "dtype": args.dtype,
        "prepend_bos": args.prepend_bos, "shard_size": args.shard_size,
        "drop_prefix": args.drop_prefix,
        "dataset": args.dataset, "num_rows": N, "num_shards": num_shards,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    def collate(batch):
        ids = torch.tensor([row["token_ids"] for row in batch], dtype=torch.long)
        if args.prepend_bos:
            bos = torch.full((ids.shape[0], 1), bos_id, dtype=torch.long)
            ids = torch.cat([bos, ids], dim=1)
        meta_rows = [
            {"subset": r["subset"],
             "window_start": r["window_start"],
             "window_end":   r["window_end"],
             "doc_len":      r["doc_len"]}
            for r in batch
        ]
        return ids, meta_rows

    progress_path = out_dir / "progress.json"
    progress = {"completed": [], "timings": []}
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
        except Exception:
            pass

    total_t0 = time.time()
    toks_done = 0
    for k, shard_i in enumerate(todo):
        start = shard_i * args.shard_size
        end = min(start + args.shard_size, N)
        sub = ds.select(range(start, end))
        loader = DataLoader(
            sub, batch_size=args.extract_batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=(args.device == "cuda"),
        )

        shard_rows = end - start
        layer_bufs = {L: torch.empty((shard_rows, window, d_model), dtype=torch_dtype)
                      for L in args.layers}
        token_buf = torch.empty((shard_rows, window), dtype=torch.int32)
        row_meta = []
        cursor = 0
        sl = 1 if args.prepend_bos else 0

        shard_t0 = time.time()
        with torch.no_grad():
            for ids, meta_rows in loader:
                ids = ids.to(args.device, non_blocking=True)
                _, cache = model.run_with_cache(
                    ids, names_filter=lambda n, keep=hook_set: n in keep,
                )
                for L in args.layers:
                    acts = cache[hook_names[L]][:, sl:sl + window, :]
                    layer_bufs[L][cursor:cursor + acts.shape[0]] = acts.to(torch_dtype).cpu()
                token_buf[cursor:cursor + ids.shape[0]] = ids[:, sl:sl + window].to(torch.int32).cpu()
                row_meta.extend(meta_rows)
                cursor += ids.shape[0]
                del cache

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for L in args.layers:
            tgt = out_dir / f"layer{L:02d}" / f"shard_{shard_i:05d}.pt"
            tmp = tgt.with_suffix(".pt.tmp")
            torch.save(layer_bufs[L], tmp)
            tmp.rename(tgt)
        tgt = out_dir / "tokens" / f"shard_{shard_i:05d}.pt"
        tmp = tgt.with_suffix(".pt.tmp")
        torch.save(token_buf, tmp)
        tmp.rename(tgt)
        with open(out_dir / "meta" / f"shard_{shard_i:05d}.json", "w") as f:
            json.dump({
                "start": start, "end": end,
                "row_indices": list(range(start, end)),
                "rows": row_meta,
            }, f)

        shard_dt = time.time() - shard_t0
        toks_done += shard_rows * window
        progress["completed"].append(shard_i)
        progress["timings"].append({"shard": shard_i, "rows": shard_rows, "sec": shard_dt})
        progress_path.write_text(json.dumps(progress, indent=2))

        elapsed = time.time() - total_t0
        rate = toks_done / max(elapsed, 1e-6)
        remaining_shards = len(todo) - (k + 1)
        remaining = remaining_shards * args.shard_size * window / max(rate, 1e-6)
        _log(f"shard {shard_i+1}/{num_shards} rows={shard_rows} "
             f"sec={shard_dt:.1f} tok/s={rate:,.0f} ETA={_fmt_eta(remaining)}")

    _log(f"done in {_fmt_eta(time.time() - total_t0)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract activations from a pre-tokenized HF windows dataset (multi-layer)",
    )
    p.add_argument("--dataset", required=True,
                   help="HF dataset dir saved by dalg-build-pile-windows")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="google/gemma-2b")
    p.add_argument("--layers", type=int, nargs="+", default=[8, 22])
    p.add_argument("--mode", default="residual",
                   choices=["residual", "residual_pre", "mlp", "mlp_out", "attn_out"])
    p.add_argument("--extract-batch-size", type=int, default=16,
                   help="Sequences per forward pass")
    p.add_argument("--shard-size", type=int, default=512,
                   help="Rows per saved shard (per layer)")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prepend-bos", action="store_true", default=True)
    p.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    p.add_argument("--drop-prefix", type=int, default=32,
                   help="Recommended # of early-window tokens to drop downstream "
                        "(stored in config.json; extraction keeps the full window).")
    p.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Smoke test: tiny shards/batches, only --limit rows")
    p.add_argument("--limit", type=int, default=64)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cmd_extract_windows(args)


if __name__ == "__main__":
    main()
