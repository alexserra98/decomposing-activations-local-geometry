"""Measure exact HDDC EM kernels on one million in-memory activation vectors.

This is a throughput benchmark, not a fit-quality experiment. Random inputs
stay in RAM; only timings and memory measurements are written to disk.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC
from dalg.models.adaptive_q.train_em_hddc import EMConfig, expectation_step, maximization_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--components", type=int, nargs="+", default=[500, 5000])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--component-chunk-size", type=int, default=32)
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    points = torch.randn(args.n, 128).pin_memory()
    cfg = EMConfig(component_chunk_size=args.component_chunk_size)
    results = []
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    for K in args.components:
        model = MFA_HDDC(torch.randn(K, 128) * .1, rank=32, shared_b=True).cuda()
        warmup = expectation_step(model, [points[:args.batch_size]], component_chunk_size=cfg.component_chunk_size)
        del warmup
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        statistics = expectation_step(
            model, points.split(args.batch_size), component_chunk_size=cfg.component_chunk_size,
            expected_rows=args.n,
        )
        torch.cuda.synchronize()
        e_seconds = time.perf_counter() - start
        start = time.perf_counter()
        maximization_step(model, statistics, cfg)
        torch.cuda.synchronize()
        m_seconds = time.perf_counter() - start
        before = statistics.nll
        del statistics
        start = time.perf_counter()
        after = expectation_step(
            model, points.split(args.batch_size), component_chunk_size=cfg.component_chunk_size,
            collect_statistics=False, expected_rows=args.n,
        ).nll
        torch.cuda.synchronize()
        record = {
            "N": args.n, "D": 128, "K": K, "q_max": 32,
            "batch_size": args.batch_size, "component_chunk_size": cfg.component_chunk_size,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "e_step_seconds": e_seconds, "m_step_seconds": m_seconds,
            "score_seconds": time.perf_counter() - start,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "train_nll_before": before, "train_nll_after": after,
            "rank_min": int(model.component_ranks.min()), "rank_max": int(model.component_ranks.max()),
            "input": "seeded float32 Gaussian vectors in pinned host memory; excludes shard I/O",
        }
        results.append(record)
        args.out_path.write_text(json.dumps(results, indent=2, allow_nan=False))
        print(json.dumps(record), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
