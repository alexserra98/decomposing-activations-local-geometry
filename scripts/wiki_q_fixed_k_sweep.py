"""Throwaway K-sweep: held-out NLL vs K at fixed q on a Wikipedia slice.

Fits a vanilla MFA for each K in a grid (q/rank fixed), reading the shared
Wikipedia activation slice via the `#pile_wikipedia_<N>` subset suffix (no data
duplication; set SUBSET_TOKENS to pick the size, e.g. 100K or 1M). For every K
the train loop holds out the same validation split
(fixed split-seed + identical subset spec), evaluates NLL on it, and stores the
best value as ``best_metric`` in ``checkpoint.pt``. We collect those values and
plot held-out NLL against K.

This is exploratory analysis code, not core library API. Delete this file and
its sbatch wrapper to remove the experiment.

Training and plotting are decoupled: `run` only trains, `plot` only collects +
plots. The sbatch wrapper trains the grid as a job array (each task a K slice)
and submits a single dependency job that runs `plot` once the array finishes.

Usage:
    PYTHONPATH=src python scripts/wiki_q_fixed_k_sweep.py run                          # train full grid
    PYTHONPATH=src python scripts/wiki_q_fixed_k_sweep.py run --k-list 200,600,1000    # train a slice
    PYTHONPATH=src python scripts/wiki_q_fixed_k_sweep.py plot                         # collect + plot

`--k-list` selects which K to train this invocation; the sbatch wrapper uses it
to split the full grid across a job array. `collect()`/`plot()` glob the output
dir for every trained run, so the combined plot is correct no matter how the
grid was split across jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Experiment grid ────────────────────────────────────────────────────────
ACTS = "dalg-cache/pile_gemma2b_activations"
SUBSET_TOKENS = "1M"             # default token budget; override with --subset-tokens (e.g. 100K)
SUBSET = f"pile_wikipedia_{SUBSET_TOKENS}"
LAYER = 17
RANK = 337                       # q, fixed
K_LIST = [200, 400, 600, 800, 1000]   # default full grid; override per-run with --k-list
# Folder carries the dataset size and fixed q, e.g.
# pile_gemma2b_models/pile_wikipedia_1m/q337_k_sweep.
OUT_ROOT = Path(f"dalg-cache/pile_gemma2b_models/pile_wikipedia_{SUBSET_TOKENS.lower()}/q{RANK}_k_sweep")

# ── Training hyperparameters ───────────────────────────────────────────────
EPOCHS = 1000
REFINE_EPOCHS = 10
BATCH = 2048
NUM_WORKERS = 2
VAL_FRAC = 0.1                   # held-out "test" split, shared across K
SPLIT_SEED = 42
SEED = 42
# Delta-stop disabled (at NLL ~1e4 any small absolute delta is meaningless); we
# stop on patience instead: end a K once its held-out NLL hasn't improved on its
# best for EARLY_STOP_PATIENCE epochs. best_metric still records the minimum, so
# the reported per-K NLL is the minimum, not the overfit endpoint. EPOCHS is a
# safety cap.
EARLY_STOP_DELTA = 0.0
EARLY_STOP_PATIENCE = 20
# Only keep checkpoint.pt + the best mfa_model.pt per run; no per-epoch snapshots.
EPOCH_SNAPSHOT_EVERY = 0


def _run_dir(k: int) -> Path:
    return OUT_ROOT / f"layer{LAYER:02d}_{k}_{RANK}_mfa"


def train_one(k: int, *, device: str) -> None:
    """Train a vanilla MFA for one K.

    Skips only when a prior run completed (``.done`` marker). An interrupted run
    has no marker, so re-invoking resumes from its ``checkpoint.pt``.
    """
    out_dir = _run_dir(k)
    done = out_dir / ".done"
    if done.exists():
        print(f"[K={k}] already trained -> {out_dir}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = f"{ACTS}#{SUBSET}"
    cmd = [
        sys.executable, "-m", "dalg.cli.run_training",
        "--shard-dir", shard_dir,
        "--layer", str(LAYER),
        "--out-dir", str(out_dir),
        "--K", str(k),
        "--rank", str(RANK),
        "--epochs", str(EPOCHS),
        "--refine-epochs", str(REFINE_EPOCHS),
        "--batch-size", str(BATCH),
        "--num-workers", str(NUM_WORKERS),
        "--val-frac", str(VAL_FRAC),
        "--split-seed", str(SPLIT_SEED),
        "--early-stop-delta", str(EARLY_STOP_DELTA),
        "--early-stop-patience", str(EARLY_STOP_PATIENCE),
        "--epoch-snapshot-every", str(EPOCH_SNAPSHOT_EVERY),
        "--device", device,
        "--seed", str(SEED),
        "--training-mode", "vanilla",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[K={k}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    done.touch()


def collect() -> dict:
    """Read best held-out NLL (best_metric) from every trained run on disk.

    Globs the output dir rather than iterating an in-memory K list, so the plot
    is correct even when the grid was split across several array jobs.
    """
    results = {}
    for ckpt_path in sorted(OUT_ROOT.glob(f"layer{LAYER:02d}_*_{RANK}_mfa/checkpoint.pt")):
        try:
            k = int(ckpt_path.parent.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        nll = float(ckpt["best_metric"])
        epoch = int(ckpt.get("best_epoch", -1))
        results[k] = {
            "nll": nll,
            "best_epoch": epoch,
            "epoch": int(ckpt["epoch"]) if "epoch" in ckpt else None,
            "run_dir": ckpt_path.parent.name,
        }
        print(f"[K={k}] held-out NLL = {nll:.6f} (best epoch {epoch})", flush=True)
    return results


def plot(results: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    layer_key = f"layer{LAYER:02d}"
    results_path = OUT_ROOT / "results.json"
    payload = {
        "source": "checkpoint.pt best_metric",
        "rank": RANK,
        "layers": {},
    }
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            existing = {}
        if isinstance(existing, dict) and isinstance(existing.get("layers"), dict):
            payload.update(existing)
            payload["source"] = "checkpoint.pt best_metric"
            payload["rank"] = RANK
            payload.setdefault("layers", {})
    layer_payload = payload["layers"].get(layer_key, {})
    if not isinstance(layer_payload, dict):
        layer_payload = {}
    for k, v in sorted(results.items()):
        entry = layer_payload.get(str(k), {})
        if not isinstance(entry, dict):
            entry = {}
        if isinstance(v, dict):
            entry.update(v)
        else:
            entry["nll"] = v
        layer_payload[str(k)] = entry
    payload["layers"][layer_key] = {
        k: layer_payload[k]
        for k in sorted(layer_payload, key=int)
    }
    payload["layers"] = {
        layer: payload["layers"][layer]
        for layer in sorted(payload["layers"])
    }
    results_path.write_text(json.dumps(payload, indent=2) + "\n")
    if not results:
        print("no results to plot", flush=True)
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; wrote results.json only", flush=True)
        return
    ks = sorted(results)
    ys = [
        float(results[k]["nll"] if isinstance(results[k], dict) else results[k])
        for k in ks
    ]

    def save_plot(values: list[float], *, ylabel: str, filename: str, log_scale: bool) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ks, values, marker="o")
        if log_scale:
            # log x so K is evenly spread; symlog y keeps any negative
            # numerical-artifact points visible instead of breaking the axis.
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e3)
            ax.axhline(0.0, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("K (number of components)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{SUBSET}, layer {LAYER}, q={RANK}")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], rotation=45, ha="right")
        ax.minorticks_off()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        png = OUT_ROOT / filename
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"saved {png}", flush=True)

    layer_suffix = f"L{LAYER}"
    save_plot(
        ys,
        ylabel="Held-out NLL (symlog)",
        filename=f"nll_vs_k_{layer_suffix}_logscale.png",
        log_scale=True,
    )
    save_plot(
        ys,
        ylabel="Held-out NLL",
        filename=f"nll_vs_k_{layer_suffix}_linear.png",
        log_scale=False,
    )
    save_plot(
        [y / k for k, y in zip(ks, ys)],
        ylabel="Held-out NLL / K (symlog)",
        filename=f"nll_per_k_vs_k_{layer_suffix}_logscale_rescaled_by_K.png",
        log_scale=True,
    )


def main() -> None:
    global SUBSET_TOKENS, SUBSET, OUT_ROOT, LAYER, RANK

    p = argparse.ArgumentParser(description="Held-out NLL vs K at fixed q")
    p.add_argument("mode", choices=["run", "plot"], help="run: train all K then plot; plot: collect+plot only")
    p.add_argument("--subset-tokens", default=SUBSET_TOKENS,
                   help="Wikipedia token budget, e.g. 100K or 1M. Drives the #spec and the "
                        "output folder name (pile_wikipedia_gemma2b_mfa_<tokens>).")
    p.add_argument("--k-list", default=None,
                   help="Comma-separated K to train this invocation (default: full grid). "
                        "The sbatch wrapper uses this to split the sweep across array jobs.")
    p.add_argument("--layer", type=int, default=LAYER, help="Layer to train (default: 17)")
    p.add_argument("--rank", type=int, default=RANK, help=f"q/rank to train (default: {RANK})")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Re-point the subset spec and output folder at the requested size.
    SUBSET_TOKENS = args.subset_tokens
    SUBSET = f"pile_wikipedia_{SUBSET_TOKENS}"
    LAYER = args.layer
    RANK = args.rank
    OUT_ROOT = Path(f"dalg-cache/pile_wikipedia_gemma2b_mfa_{SUBSET_TOKENS.lower()}/q{RANK}_k_sweep")

    if args.mode == "run":
        train_ks = K_LIST if args.k_list is None else [
            int(x) for x in args.k_list.split(",") if x.strip()
        ]
        for k in train_ks:
            train_one(k, device=args.device)
    else:  # plot
        plot(collect())


if __name__ == "__main__":
    main()
