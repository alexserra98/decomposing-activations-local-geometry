# YAML Training Pipeline

This is an experimental wrapper around the existing training and metric CLIs.
It does not change their implementations. One resolved run executes these
stages in order:

```text
training -> MFA assignments -> configured evaluation
```

Each stage validates its output and writes a completion marker. Re-running the
same manifest resumes training from the existing checkpoint or skips stages
whose artifacts are already valid.

## First smoke run

Inspect the resolved command and Slurm allocation without submitting:

```bash
uv run --locked dalg-run-pipeline submit \
  configs/experiments/adaptive_q_toy_pipeline_smoke.yaml \
  --dry-run
```

Submit the end-to-end pipeline:

```bash
uv run --locked dalg-run-pipeline submit \
  configs/experiments/adaptive_q_toy_pipeline_smoke.yaml
```

The submit command prints the immutable manifest path. Inspect it later with:

```bash
uv run --locked dalg-run-pipeline status \
  --manifest outputs/experiments/<name>/manifest_<hash>.jsonl
```

For a local or interactive allocation, plan and execute one row directly:

```bash
uv run --locked dalg-run-pipeline plan \
  configs/experiments/adaptive_q_toy_pipeline_smoke.yaml
uv run --locked dalg-run-pipeline run --manifest /path/printed/by/plan --index 0
```

## Configuration sections

For every supported YAML field, default, and model-specific constraint, see the
[complete configuration reference](training-pipeline-config-reference.md).

- `experiment`: a name and model output root.
- `dataset`: an existing activation-shard directory, optional subset suffix,
  and layer. The pipeline never starts extraction implicitly.
- `model` and `training`: arguments accepted by the selected existing trainer.
  `model.kind` selects `mfa`, `ard`, or `hddc`; HDDC accepts `q_max` as a YAML
  alias for the CLI's `rank` destination. Set `training.centroids_path` to reuse
  a precomputed initialization instead of fitting KMeans separately for every
  run. Set `training.direction_init: cluster_pca` to initialize loading
  directions from principal components stored with those centroids.
- `assignments`: full MFA responsibility assignments. Partial `max_batches`
  output is deliberately not part of the completed pipeline contract.
- `evaluation`: currently supports `adaptive_q_toy`, the numerical evaluation
  previously performed inside `notebooks/evaluate_adaptive_q.ipynb`.
- `resources`: Slurm allocation and maximum array concurrency.

Relative paths are resolved against the repository root. The shard subset can
be written either in `shard_dir` (`path#pile_wikipedia_1M`) or as a separate
`dataset.subset`, but not both.

### Reusing centroids

Point `training.centroids_path` directly at a `.pt` centroid artifact:

```yaml
training:
  centroids_path: dalg-cache/pile_gemma2b_models/centroids/k1000_L17/centroids.pt
  direction_init: random
  epochs: 20
```

The planner resolves the value to an absolute file path and verifies that its
centroid shape matches both `model.K` and the activation dimension. The resolved
path is stored in every manifest row and passed to the existing trainer. Each
run copies the artifact into its own output directory and does not run centroid
fitting. If the path is a directory or does not have the lowercase `.pt`
extension, planning fails. If the field is omitted, the trainer keeps its normal
fit-from-scratch behavior.

Legacy artifacts are bare `(K, D)` tensors. Enriched artifacts are mappings:

```text
centroids:             (K, D)
principal_components: (K, D, Q_stored)
```

Use `direction_init: random` (the default) with either format. To initialize
`W_k` from local KMeans geometry, set `direction_init: cluster_pca`; the trainer
uses `principal_components[:, :, :q]` and requires `Q_stored >= rank/q_max`.
Only loading directions change: every loading scale still starts at 1.

For the D=128, K=5000 toy experiment, upgrade the existing centroid tensor
without refitting KMeans:

```bash
.venv/bin/python scripts/temporary/build_toy_kmeans_centroids.py \
  --shard-dir dalg-cache/assets/toy_manifolds_circle_helix_D128_1M_noise1e4_shards \
  --layer 0 \
  --K 5000 \
  --out-dir dalg-cache/toy_manifold_models_1M/centroids/kmeans_k5000 \
  --device cuda \
  --pca-rank 32 \
  --pca-only
```

This reassigns all points to their saved centroids, accumulates exact float64
cluster covariances around those centroids, keeps only the first 32 eigenvectors,
and atomically replaces `centroids.pt` with the enriched bundle. It requires at
least 33 assigned points in every cluster. The operation is idempotent when the
artifact already stores at least 32 directions.

## Sweeps

The optional `sweep` mapping is a Cartesian product over fields already present
in the YAML:

```yaml
sweep:
  model.K: [50, 100, 200]
  training.seed: [0, 1, 2]
```

This produces nine manifest rows. Runs with identical `resources` are submitted
as one Slurm array. Different resource mappings are placed in separate arrays.

## Run directory

A completed run contains the normal model outputs plus:

```text
run_spec.json
TRAINING_COMPLETED.json
mfa_model_assignments.pt
ASSIGNMENTS_COMPLETED.json
metrics.json
EVALUATION_COMPLETED.json
PIPELINE_COMPLETED.json
```

The run directory name includes a short hash of the resolved dataset, model,
training, assignment, and evaluation configuration. An existing `run_spec.json`
must match before the pipeline will resume that directory.
