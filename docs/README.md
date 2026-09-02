# Documentation

This is the canonical map of the repository documentation. Start here rather
than scanning every document: choose the task below, then load only the pages
relevant to it.

## Start here

- [Research state](research/STATE.md) records the current research direction,
  observations, and immediate goals.
- [Research backlog](research/backlog.md) preserves an earlier backlog and the
  approaches considered at that point.

## I want to understand a model

- [MFA-ARD](models/mfa-ard.md) explains adaptive component rank through ARD
  shrinkage, including its invariants and failure modes.
- [MFA-HDDC](models/mfa-hddc.md) explains adaptive component rank through
  periodic covariance surgery and hard rank masks.

## I want to build or understand data

- [Toy-manifold dataset generator](reference/toy-manifold-dataset.md) defines
  the synthetic generator API, configuration, determinism, metadata, and
  activation-compatible shard format.
- [Recreating Pile activations](workflows/recreate-pile-activations.md) covers
  rebuilding token windows and Gemma-2B activation shards on a new cluster.

## I want to run or inspect training

- [YAML training pipeline](workflows/training-pipeline.md) is the operational
  guide for planning, submitting, resuming, and inspecting pipeline runs.
- [Training pipeline YAML reference](reference/training-pipeline-config.md)
  defines every supported configuration field and constraint.

Recurring agent procedures also live in `.agents/skills/`. Use the relevant
skill for execution details; use these docs for the durable model, workflow, and
output contracts. Use `$dalg-wiki` when asking an agent to navigate, write, or
reorganize this documentation collection.

## I want to evaluate a model

- [Toy-manifold tiling evaluation](evaluation/toy-manifold-tiling.md) defines
  association, rank-recovery, tangent-alignment, tangent-containment, and
  output-schema contracts.

## I want context for an experiment

Experiment pages are attachable context for a particular analysis. They are not
general workflow contracts and should be loaded only when that experiment is in
scope.

- [Adaptive-q technical card](experiments/adaptive-q-technical-card.md)
  summarizes the toy-manifold ARD/HDDC comparison and its limitations.
- [HDDC rank surgery](experiments/hddc-rank-surgery.md) records the experimental
  implementation and validation setup for covariance surgery.
- [Synthetic MFA analyses](experiments/synthetic-mfa.md) describes the temporary
  synthetic sweep and related analysis.
- [Wikipedia KMedoids slice](experiments/wikipedia-kmedoids.md) describes the
  temporary sliced-activation KMedoids workflow and verified artifacts.

## Document roles

| Directory | Role |
| --- | --- |
| `research/` | Current direction, open questions, and research snapshots |
| `models/` | Durable explanations of model variants and their invariants |
| `workflows/` | Task-oriented operational guides |
| `reference/` | Exact configuration, schema, and interface contracts |
| `evaluation/` | Metric definitions and evaluator output contracts |
| `experiments/` | Temporary or run-specific context |

If superseded documentation must remain available, put it under `archive/` and
mark it as historical. Add `decisions/` only when durable design decisions start
getting lost in experiment notes; empty taxonomy directories are intentionally
not kept.

## Maintaining this wiki

- Keep this page as the single canonical index; do not create a second summary.
- Give each new document a `Kind`, `Status`, and `Use when` routing block.
- Prefer links to canonical pages over duplicating their content.
- Update paths here and in `AGENTS.md` when moving documentation.
- Keep generated outputs and large artifacts out of `docs/`.
