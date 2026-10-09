#!/usr/bin/env bash
# Validate completed production results without recomputing any estimates.
set -euo pipefail
: "${SLURM_JOB_ID:?Run artifact checks through Slurm}"
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the experiment checkout}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts/temporary${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
hostname
"$REPO_ROOT/.venv/bin/python" - "$@" <<'PY'
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import sys

import torch
import run_cluster_id_experiment as experiment

manifest, output = map(Path, sys.argv[1:])
plan = experiment.read_json(manifest)
submission = experiment.read_json(output / "submission.json")
expected_script = experiment.sha256(experiment.__file__)
assert len(plan["runs"]) == 740
totals, warnings, conditions = Counter(), Counter(), Counter()
statuses = {}
source_hashes = {}
run_summaries = []
allowed = {"ok", "non_finite_input", "zero_neighbour_distance", "non_finite_or_nonpositive_estimate"}

def csv_value(value):
    return json.dumps(value) if isinstance(value, list) else "" if value is None else str(value)

with (output / "summary/clusters.csv").open() as combined_handle:
    combined = csv.DictReader(combined_handle)
    for run in plan["runs"]:
        directory = output / "runs" / run["relative_run"]
        report = experiment.read_json(directory / "results.json")
        marker = experiment.read_json(directory / "COMPLETED.json")
        assert experiment.sha256(directory / "results.json") == marker["results_sha256"]
        assert report["run"] == run and report["settings"] == plan["settings"] == experiment.SETTINGS
        assert report["provenance"]["commit"] == submission["commit"]
        assert report["provenance"]["script_sha256"] == expected_script
        assert report["populations"] == {"train": 270000, "validation_excluded": 30000, "test": 30000}
        for path, digest in report["input_sha256"].items():
            if path not in source_hashes:
                source_hashes[path] = experiment.sha256(path)
            assert source_hashes[path] == digest
        saved = torch.load(directory / "test_assignments.pt", map_location="cpu", weights_only=True)
        assert saved["K"] == run["K"] and len(saved["assignments"]) == 30000
        counts = torch.bincount(saved["assignments"], minlength=run["K"])
        torch.testing.assert_close(counts, saved["cluster_sizes"])
        assert len(report["clusters"]) == 2 * run["K"]
        identity = {key: run[key] for key in ("condition", "family", "relative_run", "model_kind", "K")}
        with (directory / "clusters.csv").open() as local_handle:
            local = csv.DictReader(local_handle)
            for row in report["clusters"]:
                assert next(local) == {key: csv_value(value) for key, value in row.items()}
                assert next(combined) == {key: csv_value(value) for key, value in {**identity, **row}.items()}
            assert next(local, None) is None
        for split in ("train", "test"):
            rows = [row for row in report["clusters"] if row["split"] == split]
            assert [row["cluster_id"] for row in rows] == list(range(run["K"]))
            assert sum(row["n_points"] for row in rows) == report["populations"][split]
            if split == "test":
                assert [row["n_points"] for row in rows] == counts.tolist()
            for row in rows:
                totals[f"{split}_cluster_rows"] += 1
                for estimator in ("two_nn", "gride"):
                    statuses.setdefault(f"{split}_{estimator}", Counter())[row[f"{estimator}_status"]] += 1
                warnings.update(row["warnings"])
                if row["n_points"] < 500:
                    assert row["two_nn_status"] == row["gride_status"] == "below_min_population"
                    assert row["two_nn_id"] is None and not row["gride_ids"]
                    continue
                totals[f"{split}_eligible"] += 1
                assert row["two_nn_status"] in allowed and row["gride_status"] in allowed
                if row["two_nn_status"] == "ok":
                    assert math.isfinite(row["two_nn_id"]) and row["two_nn_id"] > 0
                    assert row["two_nn_error"] == 0 and row["two_nn_scale"] > 0
                if row["gride_status"] == "ok":
                    assert row["gride_ranks"] == [2, 4, 8, 16, 32, 64]
                    for key in ("gride_ids", "gride_errors", "gride_scales"):
                        assert len(row[key]) == 6 and all(math.isfinite(x) and x > 0 for x in row[key])
            run_summaries.append({**identity, "split": split, "clusters": len(rows),
                "points": sum(row["n_points"] for row in rows),
                "eligible_clusters": sum(row["n_points"] >= 500 for row in rows),
                "two_nn_valid": sum(row["two_nn_status"] == "ok" for row in rows),
                "gride_valid": sum(row["gride_status"] == "ok" for row in rows)})
        conditions[run["condition"]] += 1
        if (run["index"] + 1) % 100 == 0:
            print(f"Validated {run['index'] + 1}/740 runs", flush=True)
    assert next(combined, None) is None
with (output / "summary/runs.csv").open() as handle:
    assert list(csv.DictReader(handle)) == [{key: csv_value(value) for key, value in row.items()} for row in run_summaries]
log_issues = []
for run in plan["runs"]:
    log = Path("logs/jobs") / f"id_est_full_{submission['array']['job_id']}_{run['index']}.err"
    for line in log.read_text().splitlines():
        if any(word in line for word in ("Traceback", "Error", "Exception", "Warning")):
            log_issues.append({"index": run["index"], "line": line})
result = {"checks_passed": True, "slurm_job_id": os.environ["SLURM_JOB_ID"],
          "manifest_sha256": experiment.sha256(manifest), "runs": len(plan["runs"]),
          "conditions": conditions, "totals": totals, "statuses": statuses,
          "estimator_warnings": warnings, "stderr_issues": log_issues,
          "artifact_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file())}
experiment.write_json(output / "final_checks.json", result)
print(json.dumps(result, indent=2), flush=True)
PY
