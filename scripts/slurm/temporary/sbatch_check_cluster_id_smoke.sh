#!/usr/bin/env bash
# Check the six approved smoke runs and exercise the existing aggregation code.
set -euo pipefail
: "${SLURM_JOB_ID:?Run artifact checks through Slurm}"
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the experiment checkout}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts/temporary${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
hostname
printf 'Artifact check Slurm job: %s\n' "$SLURM_JOB_ID"
"$REPO_ROOT/.venv/bin/python" - "$@" <<'PY'
import csv
import math
from pathlib import Path
import sys

import torch
import run_cluster_id_experiment as experiment

manifest, output = map(Path, sys.argv[1:])
plan = experiment.read_json(manifest)
indices = (1, 2, 14, 23, 149, 158)
selected = [plan["runs"][index] for index in indices]
summary = []
for run in selected:
    directory = output / "runs" / run["relative_run"]
    report = experiment.read_json(directory / "results.json")
    marker = experiment.read_json(directory / "COMPLETED.json")
    assert experiment.sha256(directory / "results.json") == marker["results_sha256"]
    assert report["run"] == run
    assert report["settings"] == plan["settings"] == experiment.SETTINGS
    assert report["provenance"]["commit"] == "44604bdb31069a9abf3fb1785ad0affee51e2b27"
    assert report["provenance"]["script_sha256"] == experiment.sha256(experiment.__file__)
    assert report["populations"] == {"train": 270000, "validation_excluded": 30000, "test": 30000}
    assert all(experiment.sha256(path) == digest for path, digest in report["input_sha256"].items())
    assert len(report["clusters"]) == 2 * run["K"]
    saved = torch.load(directory / "test_assignments.pt", map_location="cpu", weights_only=True)
    assert len(saved["assignments"]) == 30000
    counts = torch.bincount(saved["assignments"], minlength=run["K"])
    torch.testing.assert_close(counts, saved["cluster_sizes"])
    with (directory / "clusters.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [(row["split"], int(row["cluster_id"]), int(row["n_points"])) for row in csv_rows] == [
        (row["split"], row["cluster_id"], row["n_points"]) for row in report["clusters"]
    ]
    for split in ("train", "test"):
        rows = [row for row in report["clusters"] if row["split"] == split]
        assert [row["cluster_id"] for row in rows] == list(range(run["K"]))
        assert sum(row["n_points"] for row in rows) == report["populations"][split]
        if split == "test":
            assert [row["n_points"] for row in rows] == counts.tolist()
        eligible = [row for row in rows if row["n_points"] >= 500]
        for row in rows:
            if row["n_points"] < 500:
                assert row["two_nn_status"] == row["gride_status"] == "below_min_population"
                assert row["two_nn_id"] is None and row["gride_ids"] == []
            else:
                assert row["two_nn_status"] == row["gride_status"] == "ok"
                assert math.isfinite(row["two_nn_id"]) and row["two_nn_id"] > 0
                assert row["two_nn_error"] == 0 and row["two_nn_scale"] > 0
                assert row["gride_ranks"] == [2, 4, 8, 16, 32, 64]
                for key in ("gride_ids", "gride_errors", "gride_scales"):
                    assert len(row[key]) == 6 and all(math.isfinite(x) and x > 0 for x in row[key])
        summary.append({"index": run["index"], "model": run["model_kind"], "K": run["K"],
                        "split": split, "eligible": len(eligible),
                        "warnings": sorted({message for row in rows for message in row["warnings"]}),
                        "elapsed_seconds": report["elapsed_seconds"]})
experiment.aggregate({**plan, "runs": selected}, output)
experiment.write_json(output / "smoke_checks.json", {
    "manifest": str(manifest), "indices": indices, "checks_passed": True, "runs": summary,
    "artifact_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
})
print("index model K split eligible elapsed_seconds", flush=True)
for row in summary:
    print(row["index"], row["model"], row["K"], row["split"], row["eligible"],
          round(row["elapsed_seconds"], 2), flush=True)
print("All smoke artifact checks passed.", flush=True)
PY
