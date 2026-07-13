"""Compute Bhattacharyya distance matrices over q for a synthetic MFA sweep.

Heavy post-hoc analysis extracted from
``notebooks/synthetic_mfa_qk_sweep_results.ipynb`` so it can run outside
Jupyter. For one fitted K and seed, it computes or loads the pairwise Gaussian
overlap matrices for every available q value:

* ``<fit_dir>/gaussian_overlap.pt`` - full output from
  ``compute_gaussian_overlap`` for each fit
* ``<run_dir>/bhattacharyya_by_q_K####_seed####.csv`` - scalar summary
* ``<run_dir>/bhattacharyya_by_q_K####_seed####.pt`` - summary plus metadata

Example::

    PYTHONPATH=src python scripts/synthetic_mfa_bhattacharyya_by_q.py \
        --run-dir dalg-cache/qk_sweep_exploration/Ktrue1000_qtrue20 \
        --k-fit 1250 --seed 0 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import functools
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dalg.analysis.gaussian_overlap import compute_gaussian_overlap  # noqa: E402

DEFAULT_RUN_DIR = REPO_ROOT / "dalg-cache/qk_sweep_exploration/Ktrue1000_qtrue20"
FIT_DIR_RE = re.compile(r"^K(\d+)_q(\d+)_seed(\d+)$")
METRICS = ("kl_sym", "db", "db_mean", "db_cov", "bc")


def discover_fits(
    run_dir: Path,
    *,
    k_fit: int,
    seed: int | None,
    only_q: set[int] | None,
) -> list[tuple[Path, int, int, int]]:
    """List fitted models for one K and optional seed/q filters."""
    fits = []
    for path in sorted(run_dir.glob("K*_q*_seed*")):
        match = FIT_DIR_RE.match(path.name)
        if not match or not (path / "mfa_model.pt").exists():
            continue
        K, q, fit_seed = (int(group) for group in match.groups())
        if K != k_fit:
            continue
        if seed is not None and fit_seed != seed:
            continue
        if only_q is not None and q not in only_q:
            continue
        fits.append((path, K, q, fit_seed))
    return fits


def _parse_only_q(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(token.strip()) for token in value.split(",") if token.strip()}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _available_cpus() -> int:
    """Cores actually allocated to this process.

    ``os.cpu_count()`` reports the whole machine, which is wrong under Slurm:
    a job with ``--cpus-per-task=16`` on a 256-core node would otherwise spawn
    far too many workers, oversubscribing the 16 cores and blowing the job's
    memory cap. ``os.sched_getaffinity`` respects the cgroup/cpuset allocation;
    fall back to ``cpu_count`` only where affinity is unavailable.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _summary_paths(run_dir: Path, k_fit: int, seed: int) -> tuple[Path, Path]:
    stem = f"bhattacharyya_by_q_K{k_fit:04d}_seed{seed:04d}"
    return run_dir / f"{stem}.csv", run_dir / f"{stem}.pt"


def write_csv(rows: list[dict], path: Path) -> None:
    """Write scalar summary rows with a stable column order."""
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(
    rows_by_key: dict[tuple[int, int, int], dict],
    *,
    out_csv: Path,
    out_pt: Path,
    run_dir: Path,
    k_fit: int,
    seed: int,
    metric: str,
    batch_pairs: int,
) -> list[dict]:
    """Persist the q summary accumulated so far."""
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    rows = sorted(rows, key=lambda row: (row["K_fit"], row["q_fit"], row["seed"]))
    write_csv(rows, out_csv)
    torch.save(
        {
            "rows": rows,
            "meta": {
                "run_dir": str(run_dir),
                "K_fit": int(k_fit),
                "seed": int(seed),
                "metric": metric,
                "batch_pairs": int(batch_pairs),
            },
        },
        out_pt,
    )
    return rows


def load_or_compute_gaussian_overlap(
    fit_dir: Path,
    *,
    device: str,
    batch_pairs: int,
    force: bool,
    require_existing: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    """Return Gaussian-overlap matrices and their cache/compute status."""
    gaussian_overlap_path = fit_dir / "gaussian_overlap.pt"
    if gaussian_overlap_path.exists() and not force:
        return torch.load(
            gaussian_overlap_path, map_location="cpu", weights_only=False
        ), "cached"
    if require_existing:
        raise FileNotFoundError(
            f"{gaussian_overlap_path} not found. Re-run without --require-existing "
            "to compute it."
        )

    gaussian_overlap = compute_gaussian_overlap(
        fit_dir / "mfa_model.pt", device=device, batch_pairs=batch_pairs
    )
    torch.save(gaussian_overlap, gaussian_overlap_path)
    return gaussian_overlap, "computed"


def summarize_gaussian_overlap(
    fit_dir: Path,
    K: int,
    q: int,
    seed: int,
    *,
    metric: str,
    device: str,
    batch_pairs: int,
    force: bool,
    require_existing: bool,
) -> dict:
    """Compute/load one Gaussian-overlap matrix and summarize it."""
    t0 = time.time()
    gaussian_overlap, status = load_or_compute_gaussian_overlap(
        fit_dir,
        device=device,
        batch_pairs=batch_pairs,
        force=force,
        require_existing=require_existing,
    )
    mat = gaussian_overlap[metric].detach().cpu()
    off_diag = mat[~torch.eye(mat.shape[0], dtype=torch.bool)]
    if off_diag.numel():
        mean_offdiag = float(off_diag.mean().item())
        min_offdiag = float(off_diag.min().item())
        max_offdiag = float(off_diag.max().item())
    else:
        mean_offdiag = min_offdiag = max_offdiag = float("nan")

    return {
        "K_fit": int(K),
        "q_fit": int(q),
        "seed": int(seed),
        "metric": metric,
        "matrix_shape": tuple(mat.shape),
        "mean_offdiag": mean_offdiag,
        "min_offdiag": min_offdiag,
        "max_offdiag": max_offdiag,
        "gaussian_overlap_path": str(fit_dir / "gaussian_overlap.pt"),
        "status": status,
        "seconds": time.time() - t0,
    }


def _cap_worker_threads(n_threads: int) -> None:
    """Process-pool initializer: cap each worker to a small torch thread count.

    ``compute_gaussian_overlap`` works on small per-pair tensors (D and q are
    small), so its torch ops stop getting faster past roughly 8 threads and
    actively slow down with many more (thread-launch overhead dominates the tiny
    matmuls / cholesky). torch defaults ``num_threads`` to the full core count,
    which is the wrong regime here. Pinning each worker to a small slice both
    keeps every worker in its fast regime and leaves cores free for other
    workers, so a high-core machine can run many fits at once.
    """
    torch.set_num_threads(max(1, n_threads))


def _summarize_fit(
    fit: tuple[Path, int, int, int],
    *,
    metric: str,
    device: str,
    batch_pairs: int,
    force: bool,
    require_existing: bool,
) -> dict:
    """Module-level worker so it is picklable for ProcessPoolExecutor."""
    fit_dir, K, q, seed = fit
    return summarize_gaussian_overlap(
        fit_dir,
        K,
        q,
        seed,
        metric=metric,
        device=device,
        batch_pairs=batch_pairs,
        force=force,
        require_existing=require_existing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                        help="Run directory holding K*_q*_seed*/mfa_model.pt.")
    parser.add_argument("--k-fit", type=int, default=1250,
                        help="Fitted K to analyse across q.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed to analyse. Defaults to the first available seed for --k-fit.")
    parser.add_argument("--only-q", type=str, default=None,
                        help="Comma-separated q values to compute/update, e.g. '5,10,20'.")
    parser.add_argument("--metric", choices=METRICS, default="db",
                        help="Metric to summarize in the CSV/PT output.")
    parser.add_argument("--device", default="cpu",
                        help="Device used by compute_gaussian_overlap when "
                             "gaussian_overlap.pt is missing.")
    parser.add_argument("--batch-pairs", type=int, default=4096,
                        help="Pairs per Gaussian-overlap batch (tune for memory).")
    parser.add_argument("--parallel", action="store_true", default=_env_flag("PARALLEL"),
                        help="Compute q values concurrently (CPU only). Can also be enabled with PARALLEL=1.")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PARALLEL_WORKERS", "0")),
                        help="q values to process at once when --parallel is set. "
                             "0 (default) auto-picks cpu_count // threads-per-worker.")
    parser.add_argument("--threads-per-worker", type=int,
                        default=int(os.environ.get("PARALLEL_THREADS_PER_WORKER", "8")),
                        help="torch threads each worker may use. compute_gaussian_overlap stops "
                             "speeding up past ~8 threads (small D/q ops), so keeping this "
                             "low lets many workers run on a high-core machine.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute gaussian_overlap.pt even if it already exists.")
    parser.add_argument("--require-existing", action="store_true",
                        help="Only summarize existing gaussian_overlap.pt files; do not "
                             "compute missing ones.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        args.device = "cpu"

    only_q = _parse_only_q(args.only_q)
    fits = discover_fits(args.run_dir, k_fit=args.k_fit, seed=args.seed, only_q=only_q)
    if args.seed is None and fits:
        first_seed = min(seed for _, _, _, seed in fits)
        fits = [fit for fit in fits if fit[3] == first_seed]
        args.seed = first_seed
    if not fits:
        raise SystemExit(
            f"No matching fits under {args.run_dir} "
            f"(k_fit={args.k_fit}, seed={args.seed}, only_q={args.only_q})."
        )

    out_csv, out_pt = _summary_paths(args.run_dir, args.k_fit, int(args.seed))
    prev = torch.load(out_pt, map_location="cpu", weights_only=False) if out_pt.exists() else {}
    rows_by_key = {
        (int(row["K_fit"]), int(row["q_fit"]), int(row["seed"])): row
        for row in prev.get("rows", [])
    }

    sorted_fits = sorted(fits, key=lambda item: item[2])

    # A single GPU can't be split across processes usefully, and each worker
    # would reload a full model into GPU memory, so CUDA always runs serially.
    # It is also the fastest single device, so this is not a downgrade.
    if args.parallel and args.device == "cuda":
        print("CUDA uses one GPU; running serially (parallel only helps on CPU).", flush=True)
        args.parallel = False

    # Each worker is held to a small thread count (see _cap_worker_threads), so
    # on a high-core machine we can afford many workers. Auto-pick enough to
    # keep the cores busy without overcommitting them.
    threads_per_worker = max(1, args.threads_per_worker)
    if args.workers > 0:
        max_workers = args.workers
    else:
        max_workers = max(1, _available_cpus() // threads_per_worker)
    max_workers = max(1, min(max_workers, len(sorted_fits)))

    print(
        f"run_dir={args.run_dir}  K={args.k_fit}  seed={args.seed}  "
        f"fits={len(fits)}  device={args.device}  batch_pairs={args.batch_pairs}  "
        f"parallel={args.parallel}  "
        f"workers={max_workers if args.parallel else 1}  "
        f"threads/worker={threads_per_worker if args.parallel else 'default'}",
        flush=True,
    )

    worker = functools.partial(
        _summarize_fit,
        metric=args.metric,
        device=args.device,
        batch_pairs=args.batch_pairs,
        force=args.force,
        require_existing=args.require_existing,
    )

    if args.parallel:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_cap_worker_threads,
            initargs=(threads_per_worker,),
        ) as executor:
            result_iter = executor.map(worker, sorted_fits)
            for n, row in enumerate(result_iter, 1):
                rows_by_key[(row["K_fit"], row["q_fit"], row["seed"])] = row
                rows = save_summary(
                    rows_by_key,
                    out_csv=out_csv,
                    out_pt=out_pt,
                    run_dir=args.run_dir,
                    k_fit=args.k_fit,
                    seed=int(args.seed),
                    metric=args.metric,
                    batch_pairs=args.batch_pairs,
                )
                print(
                    f"[{n}/{len(sorted_fits)}] K={row['K_fit']:>4} q={row['q_fit']:>4} seed={row['seed']} "
                    f"{row['status']} | mean={row['mean_offdiag']:.4f} "
                    f"min={row['min_offdiag']:.4f} max={row['max_offdiag']:.4f} "
                    f"| {row['seconds']:.1f}s | saved {len(rows)} rows",
                    flush=True,
                )
    else:
        for n, fit in enumerate(sorted_fits, 1):
            row = worker(fit)
            rows_by_key[(row["K_fit"], row["q_fit"], row["seed"])] = row
            rows = save_summary(
                rows_by_key,
                out_csv=out_csv,
                out_pt=out_pt,
                run_dir=args.run_dir,
                k_fit=args.k_fit,
                seed=int(args.seed),
                metric=args.metric,
                batch_pairs=args.batch_pairs,
            )
            print(
                f"[{n}/{len(sorted_fits)}] K={row['K_fit']:>4} q={row['q_fit']:>4} seed={row['seed']} "
                f"{row['status']} | mean={row['mean_offdiag']:.4f} "
                f"min={row['min_offdiag']:.4f} max={row['max_offdiag']:.4f} "
                f"| {row['seconds']:.1f}s | saved {len(rows)} rows",
                flush=True,
            )

    rows = save_summary(
        rows_by_key,
        out_csv=out_csv,
        out_pt=out_pt,
        run_dir=args.run_dir,
        k_fit=args.k_fit,
        seed=int(args.seed),
        metric=args.metric,
        batch_pairs=args.batch_pairs,
    )
    print(f"\nsaved {len(rows)} rows -> {out_csv}\n                -> {out_pt}", flush=True)


if __name__ == "__main__":
    main()
