"""CLI for planning, running, submitting, and inspecting DALG pipelines."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from dalg.pipeline import (
    REPO_ROOT,
    default_manifest_path,
    execute_run,
    group_by_resources,
    pipeline_status,
    read_manifest,
    resolve_experiment,
    sbatch_command,
    write_manifest,
)


def _plan(config: str, manifest: str | None, *, check_inputs: bool) -> tuple[Path, list[dict]]:
    runs = resolve_experiment(config, check_inputs=check_inputs)
    path = Path(manifest).resolve() if manifest else default_manifest_path(runs)
    path = write_manifest(runs, path)
    print(f"planned {len(runs)} run(s): {path}")
    for index, run in enumerate(runs):
        print(f"  [{index:04d}] {run['run_id']} -> {run['run_dir']}")
    return path, runs


def cmd_plan(args) -> None:
    _plan(args.config, args.manifest, check_inputs=not args.no_check_inputs)


def cmd_run(args) -> None:
    runs = read_manifest(args.manifest)
    if args.index < 0 or args.index >= len(runs):
        raise SystemExit(f"manifest index {args.index} is outside [0, {len(runs) - 1}]")
    execute_run(runs[args.index])


def cmd_status(args) -> None:
    statuses = pipeline_status(read_manifest(args.manifest))
    for row in statuses:
        state = "complete" if row["pipeline"] else "incomplete"
        print(
            f"{row['run_id']}: {state} "
            f"(train={row['training']} assignments={row['assignments']} "
            f"evaluation={row['evaluation']})"
        )
    complete = sum(int(row["pipeline"]) for row in statuses)
    print(f"complete: {complete}/{len(statuses)}")
    if args.json:
        print(json.dumps(statuses, indent=2))


def cmd_submit(args) -> None:
    manifest, runs = _plan(
        args.config,
        args.manifest,
        check_inputs=not args.no_check_inputs,
    )
    worker = REPO_ROOT / "scripts" / "slurm" / "temporary" / "sbatch_training_pipeline.sh"
    if not worker.is_file():
        raise SystemExit(f"Slurm worker not found: {worker}")

    groups = group_by_resources(runs)
    for group_index, group in enumerate(groups):
        group_manifest = manifest
        if len(groups) > 1:
            group_manifest = manifest.with_name(
                f"{manifest.stem}_resources{group_index:02d}{manifest.suffix}"
            )
            write_manifest(group, group_manifest)
        command = sbatch_command(group_manifest, group, worker_path=worker)
        print(f"$ {shlex.join(command)}")
        if args.dry_run:
            continue
        completed = subprocess.run(
            command,
            check=True,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        print(
            f"submitted resource group {group_index + 1}/{len(groups)}: "
            f"job {completed.stdout.strip()}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and execute YAML-defined DALG training pipelines"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="resolve YAML into an immutable JSONL manifest")
    plan.add_argument("config")
    plan.add_argument("--manifest", default=None)
    plan.add_argument("--no-check-inputs", action="store_true")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="execute one manifest row")
    run.add_argument("--manifest", required=True)
    run.add_argument("--index", type=int, required=True)
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="show artifact status for every manifest row")
    status.add_argument("--manifest", required=True)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    submit = sub.add_parser("submit", help="plan and submit Slurm arrays by resource group")
    submit.add_argument("config")
    submit.add_argument("--manifest", default=None)
    submit.add_argument("--no-check-inputs", action="store_true")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=cmd_submit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
