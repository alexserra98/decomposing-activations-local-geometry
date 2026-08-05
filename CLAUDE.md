# CLAUDE.md

Read and follow `AGENTS.md` before working in this repository. `AGENTS.md` is
the canonical source for project structure, data and model invariants, storage
policy, implementation guidance, and validation expectations. Do not duplicate
those instructions here.

## Shared Workflow Skills

Recurring workflows are shared with Codex from the canonical directories under
`.agents/skills/`. Claude Code discovers symlinks to the same skill directories
under `.claude/skills/`.

Use only the skill matching the requested stage:

- `/dalg-train-mfa`
- `/dalg-compute-assignments`
- `/dalg-compute-metrics`
- `/dalg-label-mfa-clusters`

Perform only the stage requested. Check and report missing prerequisites; do
not automatically execute upstream or downstream pipeline stages.

## Experimental Context

Temporary experimental workflows are kept out of always-loaded instructions.
Read the relevant document only when the user explicitly places that experiment
in scope:

- `docs/experiments/wikipedia-kmedoids.md`
- `docs/experiments/synthetic-mfa.md`
- `docs/experiments/hddc-rank-surgery.md`
