"""Upgrade the 19 noise-10 BIC reports using saved likelihoods and assignments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from dalg.analysis.bic_improved import _training_cluster_sizes, active_bic_from_standard
from dalg.data.subset_spec import split_shard_dir_spec
from dalg.pipeline import _evaluation_artifact_valid, _write_json_atomic


ROOT = Path(
    "dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0/"
    "noise_ratio_10/models"
)
BACKUP_NAME = "metrics.before_augmented_bic.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="back up and update reports")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = ROOT.resolve()
    specs = sorted(root.rglob("run_spec.json"))
    assert len(specs) == 19, f"Expected 19 runs, found {len(specs)}"
    updates = []

    for spec_path in specs:
        run_dir = spec_path.parent
        run = json.loads(spec_path.read_text())
        path = run_dir / "metrics.json"
        original = path.read_bytes()
        source = json.loads(original)
        assert source["schema_version"] == 1, path
        assert source["evaluation"] == "toy_manifold_tiling"
        assert source["identity_hash"] == run["identity_hash"]
        assert Path(run["run_dir"]).resolve() == run_dir
        assert source["model_kind"] == run["training"]["model_kind"]
        assert not (run_dir / BACKUP_NAME).exists(), run_dir / BACKUP_NAME
        shard_dir, subset_spec = split_shard_dir_spec(run["dataset"]["shard_dir"])
        assert Path(source["dataset"]["shard_dir"]).resolve() == shard_dir.resolve()
        assert source["dataset"]["subset_spec"] == subset_spec
        assert source["dataset"]["layer"] == run["dataset"]["layer"]
        if source["model_kind"] == "hddc":
            for rank_name in ("rank", "ambient_rank"):
                assert source[rank_name]["definition"] == "hddc_rank_mask_count"
                assert "threshold" not in source[rank_name]

        bic = source["bic"]
        assert bic["convention"] == "lower_is_better" and bic["split"] == "train"
        standard_bic = float(bic["value"])
        sizes, n = _training_cluster_sizes(run_dir, run_dir / "mfa_model_assignments.pt")
        K = int(sizes.numel())
        active = int((sizes > 0).sum())
        assert K == source["K"] == 1000
        assert n == bic["n"] == source["dataset"]["train_rows"] == 270_000
        assert math.isclose(
            standard_bic,
            2 * n * source["nll"]["train"] + bic["parameters"] * math.log(n),
            rel_tol=1e-12,
        ), path
        score = active_bic_from_standard(standard_bic, n=n, active_components=active, K=K)
        updated = {
            **source,
            "schema_version": 2,
            "bic": {
                **bic,
                "value": score,
                "standard_bic": standard_bic,
                "standard_bic_per_sample_reward": -standard_bic / n,
                "activity_reward": active,
                "active_components": active,
                "inactive_components": K - active,
                "K": K,
                "assignment_rule": "hard_map_count_greater_than_zero",
                "formula": "-standard_bic / n + active_components",
                "convention": "higher_is_better",
            },
        }
        updates.append((path, run, original, updated))
        print(
            f"{run_dir.parent.name}/{run_dir.name.rsplit('__', 1)[-1]}: "
            f"active_train={active}, live_all={source['components']['live']}, "
            f"standard_bic={standard_bic:.6f}, augmented_bic={score:.6f}",
            flush=True,
        )

    print(f"Validated all {len(updates)} updates; no model inference required.", flush=True)
    if not args.apply:
        return

    for path, _, original, _ in updates:
        assert path.read_bytes() == original, f"Report changed during assessment: {path}"
    for path, run, original, updated in updates:
        with (path.parent / BACKUP_NAME).open("xb") as backup:
            backup.write(original)
        _write_json_atomic(path, updated)
        saved = json.loads(path.read_text())
        assert saved == updated
        assert _evaluation_artifact_valid(run), path
        before = json.loads(original)
        assert {k: v for k, v in saved.items() if k not in ("bic", "schema_version")} == {
            k: v for k, v in before.items() if k not in ("bic", "schema_version")
        }
    print(f"Updated and verified {len(updates)} reports; originals saved as {BACKUP_NAME}.")


if __name__ == "__main__":
    main()
