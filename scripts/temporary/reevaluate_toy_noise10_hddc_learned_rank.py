"""Refresh the nine noise-10 HDDC evaluations using their saved rank masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling
from dalg.pipeline import _mark_stage, _write_json_atomic, read_manifest


MANIFEST = Path(
    "outputs/experiments/toy_noise10_k1000_hddc_cluster_pca/manifest_6ebebfb99c.jsonl"
)
THRESHOLDS = [0.00025, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
PROTECTED = (
    "mfa_model.pt", "mfa_model_assignments.pt", "run_spec.json", "config.json",
    "val_indices.json", "checkpoint.pt",
)


def hashes(run_dir: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        for name in PROTECTED
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    runs = sorted(
        read_manifest(MANIFEST),
        key=lambda run: run["training"]["arguments"]["surgery_threshold"],
    )
    assert [r["training"]["arguments"]["surgery_threshold"] for r in runs] == THRESHOLDS
    for run in runs:
        run_dir = Path(run["run_dir"])
        saved_spec = json.loads((run_dir / "run_spec.json").read_text())
        assert saved_spec == run, run_dir
        assert run["training"]["model_kind"] == "hddc"
        assert run_dir.parent.name == "toy_noise10_k1000_hddc_cluster_pca"
        assert run["evaluation"]["max_mean_to_manifold_distance"] is None
        for name in PROTECTED + ("metrics.json",):
            assert (run_dir / name).is_file(), run_dir / name
        source = json.loads((run_dir / "metrics.json").read_text())
        assert source["identity_hash"] == run["identity_hash"]
        assert source["K"] == 1000 and source["dataset"]["selected_rows"] == 300_000
    if args.check_only:
        print(f"Validated all {len(runs)} runs and required artifacts", flush=True)
        return

    for run in runs:
        run_dir = Path(run["run_dir"])
        threshold = run["training"]["arguments"]["surgery_threshold"]
        print(f"Evaluating surgery_threshold={threshold}: {run_dir.name}", flush=True)
        before = hashes(run_dir)
        source = json.loads((run_dir / "metrics.json").read_text())
        metrics = evaluate_toy_manifold_tiling(
            run_dir,
            shard_dir=run["dataset"]["shard_dir"],
            layer=run["dataset"]["layer"],
            model_kind="hddc",
            assignments_path=run_dir / "mfa_model_assignments.pt",
            batch_size=run["evaluation"]["batch_size"],
            device=run["evaluation"]["device"],
            max_mean_to_manifold_distance=None,
        )
        metrics["run_id"] = run["run_id"]
        metrics["identity_hash"] = run["identity_hash"]
        for name in ("rank", "ambient_rank"):
            assert metrics[name]["definition"] == "hddc_rank_mask_count"
            assert "threshold" not in metrics[name]
        for name in ("dataset", "components", "clustering", "association"):
            assert metrics[name] == source[name], name
        assert metrics["tangent_alignment"]["definition"] == source["tangent_alignment"]["definition"]
        assert metrics["bic"]["parameters"] == source["bic"]["parameters"]
        for split in ("train", "validation"):
            assert math.isclose(metrics["nll"][split], source["nll"][split], rel_tol=1e-7)
        assert math.isclose(metrics["bic"]["value"], source["bic"]["value"], rel_tol=1e-7)
        assert hashes(run_dir) == before, "An upstream artifact changed"
        print(
            f"Alignment before={source['tangent_alignment']['subspace_overlap']} "
            f"after={metrics['tangent_alignment']['subspace_overlap']}", flush=True,
        )
        _write_json_atomic(run_dir / "metrics.json", metrics)
        _mark_stage(run_dir, "evaluation", run_dir / "metrics.json")
        print(
            f"Saved: rank {source['rank']['mean_learned']:.6f} -> "
            f"{metrics['rank']['mean_learned']:.6f}; containment "
            f"{source['tangent_containment']['subspace_overlap']['mean']:.6f} -> "
            f"{metrics['tangent_containment']['subspace_overlap']['mean']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
