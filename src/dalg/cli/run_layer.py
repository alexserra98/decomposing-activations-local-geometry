"""
CLI experiment runner for MFA pipeline.

Subcommands:
    extract       Extract activations for a single layer and save to disk.
    intrinsic-dim Compute intrinsic dimensionality per cluster.
    overlap       Compute pairwise overlap metrics between components.
    all           Run extract + train + intrinsic-dim + overlap in sequence.

Example usage:
    # Step 1: extract (expensive LLM forward pass)
    dalg-run-layer extract \
        --model gpt2-small --dataset ./data/supervised.json \
        --layer 4 --out-dir results/gpt2-small/layer_04 \
        --max-tokens 250000 --device cuda

    # Step 2: train (can re-run with different K without re-extracting)
    dalg-run-training \
        --data-dir results/gpt2-small/layer_04 \
        --K 500 --rank 10 --epochs 10 --device cuda

    # Step 3: analysis
    dalg-run-layer overlap \
        --data-dir results/gpt2-small/layer_04 --device cuda
    dalg-run-layer intrinsic-dim \
        --data-dir results/gpt2-small/layer_04 --device cuda
"""
import os
import json
import argparse
from pathlib import Path
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch


# ── Subcommand implementations ──────────────────────────────────────────


def cmd_extract(args):
    """Extract activations for one layer and save to disk."""
    from dalg.llm.activation_generator import ActivationGenerator, extract_token_ids
    from dalg.data.concept_dataset import SupervisedConceptDataset, ConceptDataset

    os.makedirs(args.out_dir, exist_ok=True)

    # Try supervised first, fall back to unsupervised
    try:
        dataset = SupervisedConceptDataset(args.dataset)
    except Exception:
        dataset = ConceptDataset(args.dataset)
    print(f"Dataset: {len(dataset)} samples")

    act_gen = ActivationGenerator(
        args.model,
        model_device=args.device,
        data_device=args.device,
        mode=args.mode,
    )

    layer = args.layer
    activations, freq = act_gen.generate_activations(
        dataset, [layer], batch_size=args.extract_batch_size,
    )
    tokens, _, _ = extract_token_ids(dataset, act_gen, batch_size=args.extract_batch_size)

    X = activations[0]
    if args.max_tokens > 0 and X.shape[0] > args.max_tokens:
        X = X[: args.max_tokens]
        tokens = tokens[: args.max_tokens]
        freq = freq[: args.max_tokens]

    # Save
    torch.save(X.cpu(), os.path.join(args.out_dir, "activations.pt"))
    torch.save(tokens.cpu(), os.path.join(args.out_dir, "tokens.pt"))
    torch.save(freq.cpu(), os.path.join(args.out_dir, "freq.pt"))

    config = {
        "model": args.model, "dataset": args.dataset,
        "layer": layer, "mode": args.mode,
        "N": int(X.shape[0]), "D": int(X.shape[1]),
        "max_tokens": args.max_tokens,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved activations {X.shape} to {args.out_dir}")


def cmd_extract_windows(args):
    """Extract activations from a pre-tokenized HF windows dataset.

    Works on the output of dalg-build-pile-windows (rows have a
    `token_ids` column of length WINDOW_SIZE). One forward pass per batch,
    caches only the requested layers — so layers 8+22 come out in a single
    sweep. Resume-safe: existing shards are skipped.

    Output layout:
        <out_dir>/config.json
        <out_dir>/progress.json
        <out_dir>/layer{L:02d}/shard_{i:05d}.pt   # (rows, window, d_model)
        <out_dir>/tokens/shard_{i:05d}.pt         # int32, (rows, window)
        <out_dir>/meta/shard_{i:05d}.json
    """
    import time
    from pathlib import Path
    from torch.utils.data import DataLoader
    from datasets import load_from_disk
    from transformer_lens import HookedTransformer, utils as tl_utils

    HOOK_FN = {
        "residual":     lambda L: tl_utils.get_act_name("resid_post", L),
        "residual_pre": lambda L: tl_utils.get_act_name("resid_pre", L),
        "mlp":          lambda L: f"blocks.{L}.mlp.hook_post",
        "mlp_out":      lambda L: f"blocks.{L}.hook_mlp_out",
        "attn_out":     lambda L: f"blocks.{L}.hook_attn_out",
    }

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def fmt_eta(sec):
        s = int(sec)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    out_dir = Path(args.out_dir)
    if args.debug:
        args.shard_size = min(args.shard_size, 16)
        args.extract_batch_size = min(args.extract_batch_size, 4)
        log(f"DEBUG: shard_size={args.shard_size} batch={args.extract_batch_size} limit={args.limit}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokens").mkdir(exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)
    for L in args.layers:
        (out_dir / f"layer{L:02d}").mkdir(exist_ok=True)

    torch_dtype = {"float16": torch.float16,
                   "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[args.dtype]

    log(f"loading dataset: {args.dataset}")
    ds = load_from_disk(args.dataset)
    if args.debug:
        ds = ds.select(range(min(args.limit, len(ds))))
    N = len(ds)
    window = len(ds[0]["token_ids"])
    num_shards = (N + args.shard_size - 1) // args.shard_size
    log(f"rows={N}  window={window}  shards={num_shards}  layers={args.layers}  drop_prefix={args.drop_prefix}")

    def shard_done(i):
        if not (out_dir / "tokens" / f"shard_{i:05d}.pt").exists():
            return False
        return all(
            (out_dir / f"layer{L:02d}" / f"shard_{i:05d}.pt").exists()
            for L in args.layers
        )

    todo = [i for i in range(num_shards) if not shard_done(i)]
    log(f"resume: {num_shards - len(todo)}/{num_shards} shards on disk, {len(todo)} to do")
    if not todo:
        log("nothing to do — exiting")
        return

    log(f"loading model {args.model} on {args.device} dtype={args.dtype}")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=torch_dtype,
    )
    model.eval()
    bos_id = model.tokenizer.bos_token_id
    d_model = model.cfg.d_model
    log(f"model loaded in {time.time()-t0:.1f}s  d_model={d_model}  bos={bos_id}")

    hook_names = {L: HOOK_FN[args.mode](L) for L in args.layers}
    hook_set = set(hook_names.values())
    log(f"hook names: {hook_names}")

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
        log(f"shard {shard_i+1}/{num_shards} rows={shard_rows} "
            f"sec={shard_dt:.1f} tok/s={rate:,.0f} ETA={fmt_eta(remaining)}")

    log(f"done in {fmt_eta(time.time() - total_t0)}")


def validate_args(args) -> None:
    """Validate cross-argument constraints once, before running a command."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.command == "all":
        if world_size > 1:
            raise SystemExit("all: distributed training is not supported by this combined command")
        args.training_mode = "vanilla"

    elif args.command == "intrinsic-dim":
        if args.shard_dir is not None and args.act_dir is not None:
            raise SystemExit("intrinsic-dim: --shard-dir and --act-dir are mutually exclusive")
        if args.shard_dir is not None and args.layer is None:
            raise SystemExit("intrinsic-dim: --layer is required with --shard-dir")


def cmd_overlap(args):
    """Compute pairwise overlap metrics between MFA components."""
    from dalg.analysis.cluster_overlap import compute_overlap

    data_dir = args.data_dir
    # Allow --data-dir to point directly to a .pt file or to a directory
    if os.path.isfile(data_dir):
        model_path = data_dir
        data_dir = Path(os.path.dirname(data_dir))
    else:
        model_path = Path(os.path.join(data_dir, "mfa_model.pt"))

    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)

    results = compute_overlap(model_path, device=args.device, batch_pairs=args.batch_pairs)

    save_path = os.path.join(out_dir, "overlap.pt")
    torch.save(results, save_path)
    print(f"Overlap saved to {save_path}")


def cmd_intrinsic_dim(args):
    """Compute intrinsic dimensionality per cluster.

    Two input layouts supported:
      (A) monolithic --act-dir with activations.pt/tokens.pt,
      (B) sharded    --shard-dir from `extract-windows`, with --layer.
    """
    from dalg.analysis.cluster_intrinsic_dim import (
        compute_intrinsic_dims, compute_intrinsic_dims_from_shards,
    )
    from pathlib import Path # why do we need to reimport??
    data_dir = args.data_dir
    if os.path.isfile(data_dir):
        model_path = data_dir
        data_dir = Path(os.path.dirname(data_dir))
    else:
        model_path = Path(os.path.join(data_dir, "mfa_model.pt"))

    out_dir = args.out_dir or data_dir
    out_dir = Path(out_dir) 
    os.makedirs(out_dir, exist_ok=True)

    if args.shard_dir is not None:
        results = compute_intrinsic_dims_from_shards(
            model_path, Path(args.shard_dir),
            layer=args.layer,
            assignments_path=args.assignments_path,
            device=args.device,
            variance_threshold=args.variance_threshold,
            min_population=args.min_population,
            max_samples=args.max_samples_per_cluster,
            pca_device=args.pca_device,
            pca_workers=args.pca_workers,
            seed=(args.seed or 0),
        )
    else:
        act_dir = args.act_dir or data_dir
        act_path = Path(os.path.join(act_dir, "activations.pt"))
        tok_path = Path(os.path.join(act_dir, "tokens.pt"))
        results = compute_intrinsic_dims(
            model_path, act_path, tok_path,
            assignments_path=args.assignments_path,
            device=args.device,
            batch_size=args.batch_size,
            variance_threshold=args.variance_threshold,
            min_population=args.min_population,
            max_samples=args.max_samples_per_cluster,
            pca_device=args.pca_device,
            pca_workers=args.pca_workers,
            seed=(args.seed or 0),
        )

    save_path = os.path.join(out_dir, "intrinsic_dims.pt")
    torch.save(results, save_path)
    print(f"Intrinsic dims saved to {save_path}")


def cmd_all(args):
    """Run the full pipeline: extract → train → overlap → intrinsic-dim."""
    # For 'all', --out-dir is required and becomes --data-dir for subsequent steps
    args.data_dir = args.out_dir

    print("=" * 60)
    print("STEP 1: Extract activations")
    print("=" * 60)
    cmd_extract(args)

    print("\n" + "=" * 60)
    print("STEP 2: Train MFA")
    print("=" * 60)
    from dalg.cli.run_training import cmd_train
    cmd_train(args)

    print("\n" + "=" * 60)
    print("STEP 3: Compute overlap")
    print("=" * 60)
    cmd_overlap(args)

    print("\n" + "=" * 60)
    print("STEP 4: Compute intrinsic dimensions")
    print("=" * 60)
    cmd_intrinsic_dim(args)

    print("\n" + "=" * 60)
    print("DONE — all results in", args.out_dir)
    print("=" * 60)


# ── Argument parsing ────────────────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(description="MFA experiment runner (per-layer)")
    sub = p.add_subparsers(dest="command", required=True)

    # -- shared args --
    def add_common(sp):
        sp.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--batch-size", type=int, default=128)

    # -- extract --
    sp = sub.add_parser("extract", help="Extract activations for one layer")
    add_common(sp)
    sp.add_argument("--model", required=True, help="TransformerLens model name")
    sp.add_argument("--dataset", required=True, help="Path to dataset (JSON/JSONL/CSV)")
    sp.add_argument("--layer", type=int, required=True)
    sp.add_argument("--out-dir", required=True, help="Output directory")
    sp.add_argument("--max-tokens", type=int, default=0, help="Cap on tokens (0=no cap)")
    sp.add_argument("--mode", default="residual", help="Activation stream")
    sp.add_argument("--extract-batch-size", type=int, default=5, help="Batch size for LLM forward pass")
    sp.set_defaults(func=cmd_extract)

    # -- extract-windows --
    sp = sub.add_parser("extract-windows",
                        help="Extract activations from pre-tokenized HF windows dataset (multi-layer)")
    add_common(sp)
    sp.add_argument("--dataset", required=True,
                    help="HF dataset dir saved by dalg-build-pile-windows")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--model", default="google/gemma-2b")
    sp.add_argument("--layers", type=int, nargs="+", default=[8, 22])
    sp.add_argument("--mode", default="residual",
                    choices=["residual", "residual_pre", "mlp", "mlp_out", "attn_out"])
    sp.add_argument("--extract-batch-size", type=int, default=16,
                    help="Sequences per forward pass")
    sp.add_argument("--shard-size", type=int, default=512,
                    help="Rows per saved shard (per layer)")
    sp.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    sp.add_argument("--num-workers", type=int, default=2)
    sp.add_argument("--prepend-bos", action="store_true", default=True)
    sp.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    sp.add_argument("--drop-prefix", type=int, default=32,
                    help="Recommended # of early-window tokens to drop downstream "
                         "(stored in config.json; extraction keeps the full window).")
    sp.add_argument("--debug", action="store_true",
                    help="Smoke test: tiny shards/batches, only --limit rows")
    sp.add_argument("--limit", type=int, default=64)
    sp.set_defaults(func=cmd_extract_windows)

    # -- overlap --
    sp = sub.add_parser("overlap", help="Compute pairwise overlap metrics")
    add_common(sp)
    sp.add_argument("--data-dir", required=True,
                    help="Directory with mfa_model.pt, or direct path to a .pt model file")
    sp.add_argument("--out-dir", default=None,
                    help="Where to save overlap.pt (default: same as --data-dir)")
    sp.add_argument("--batch-pairs", type=int, default=4096, help="Pairs per batch for vectorized overlap")
    sp.set_defaults(func=cmd_overlap)

    # -- intrinsic-dim --
    sp = sub.add_parser("intrinsic-dim", help="Compute intrinsic dim per cluster")
    add_common(sp)
    sp.add_argument("--data-dir", required=True,
                    help="Directory with mfa_model.pt, or direct path to a .pt model file")
    sp.add_argument("--act-dir", default=None,
                    help="Monolithic layout: dir with activations.pt/tokens.pt "
                         "(default: same as --data-dir)")
    sp.add_argument("--shard-dir", default=None,
                    help="Shard layout: extract-windows output dir "
                         "(mutually exclusive with --act-dir)")
    sp.add_argument("--layer", type=int, default=None,
                    help="Layer to read from shards (required with --shard-dir)")
    sp.add_argument("--pca-device", default=None,
                    help="Device for the PCA phase (default: same as --device). "
                         "Set to 'cpu' to free the GPU for other jobs.")
    sp.add_argument("--pca-workers", type=int, default=1,
                    help="Parallel workers for the CPU PCA phase")
    sp.add_argument("--assignments-path", default=None,
                    help="Path to precomputed cluster assignments "
                         "(default: <data-dir>/mfa_model_assignments.pt)")
    sp.add_argument("--out-dir", default=None,
                    help="Where to save intrinsic_dims.pt (default: same as --data-dir)")
    sp.add_argument("--variance-threshold", type=float, default=0.90)
    sp.add_argument("--min-population", type=int, default=100)
    sp.add_argument("--max-samples-per-cluster", type=int, default=10000)
    sp.set_defaults(func=cmd_intrinsic_dim)

    # -- all --
    sp = sub.add_parser("all", help="Run full pipeline: extract → train → overlap → intrinsic-dim")
    add_common(sp)
    # extract args
    sp.add_argument("--model", required=True)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--layer", type=int, required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--max-tokens", type=int, default=0)
    sp.add_argument("--mode", default="residual")
    sp.add_argument("--extract-batch-size", type=int, default=5)
    # train args
    sp.add_argument("--K", type=int, required=True)
    sp.add_argument("--rank", type=int, default=10)
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--grad-clip", type=float, default=None)
    sp.add_argument("--proj-dim", type=int, default=32)
    sp.add_argument("--refine-epochs", type=int, default=25)
    sp.add_argument("--vocab-size", type=int, default=50257)
    sp.add_argument("--pool-size", type=int, default=None,
                    help="Reservoir pool size for kmeans init (default = max(K*2, N/5))")
    sp.add_argument("--max-pool-size", type=int, default=2_000_000)
    sp.add_argument("--compile", action="store_true")
    # overlap args
    sp.add_argument("--batch-pairs", type=int, default=4096)
    # intrinsic-dim args
    sp.add_argument("--variance-threshold", type=float, default=0.90)
    sp.add_argument("--min-population", type=int, default=100)
    sp.add_argument("--max-samples-per-cluster", type=int, default=10000)
    sp.add_argument("--assignments-path", default=None)
    sp.set_defaults(func=cmd_all)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
