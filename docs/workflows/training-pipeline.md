# YAML Training Pipeline

> **Kind:** Workflow · **Status:** Current · **Use when:** Planning, submitting,
> resuming, or inspecting manifest-based training runs. **Related:**
> [YAML configuration reference](../reference/training-pipeline-config.md)

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
  configs/experiments/toy_manifold_tiling_pipeline_smoke.yaml \
  --dry-run
```

Submit the end-to-end pipeline:

```bash
uv run --locked dalg-run-pipeline submit \
  configs/experiments/toy_manifold_tiling_pipeline_smoke.yaml
```

The submit command prints the immutable manifest path. Inspect it later with:

```bash
uv run --locked dalg-run-pipeline status \
  --manifest outputs/experiments/<name>/manifest_<hash>.jsonl
```

For a local or interactive allocation, plan and execute one row directly:

```bash
uv run --locked dalg-run-pipeline plan \
  configs/experiments/toy_manifold_tiling_pipeline_smoke.yaml
uv run --locked dalg-run-pipeline run --manifest /path/printed/by/plan --index 0
```

## Configuration sections

For every supported YAML field, default, and model-specific constraint, see the
[complete configuration reference](../reference/training-pipeline-config.md).

- `experiment`: a name and model output root.
- `dataset`: an existing activation-shard directory, optional subset suffix,
  and layer. The pipeline never starts extraction implicitly.
- `model` and `training`: arguments accepted by the selected existing trainer.
  `model.kind` selects `mfa`, `ard`, or `hddc`; HDDC accepts `q_max` as a YAML
  alias for the CLI's `rank` destination. Set `training.centroids_path` to reuse
  a precomputed initialization instead of fitting KMeans separately for every
  run. Set `training.direction_init: cluster_pca` to initialize loading
  directions from principal components stored with those centroids; this
  direction-initialization path is a temporary experimental feature.
- `assignments`: full MFA responsibility assignments. Partial `max_batches`
  output is deliberately not part of the completed pipeline contract.
- `evaluation`: currently supports `toy_manifold_tiling`, which measures NLL,
  BIC, planted-manifold clustering recovery, component use, and effective local
  rank for vanilla MFA, ARD, or HDDC runs on toy-manifold shards. It associates each
  Gaussian with a nearby planted manifold by exact projection and measures how
  well its leading intrinsic-dimensional covariance subspace aligns with the
  ground-truth tangent space.
- `resources`: Slurm allocation and maximum array concurrency.

Relative paths are resolved against the repository root. The shard subset can
be written either in `shard_dir` (`path#pile_wikipedia_1M`) or as a separate
`dataset.subset`, but not both.

### Reusing centroids and experimental W initialization

> **Temporary experimental feature:** `direction_init: cluster_pca` and the
> `principal_components` payload in `centroids.pt` exist to support the current
> W-initialization experiments. Do not treat this path as a stable pipeline
> interface. Ordinary centroid reuse through `centroids_path` is separate, and
> `direction_init: random` remains the default.

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

For example:

```yaml
model:
  kind: hddc
  K: 5000
  q_max: 32

training:
  centroids_path: dalg-cache/toy_manifold_models_1M/centroids/kmeans_k5000/centroids.pt
  direction_init: cluster_pca
```

This initialization is available for `mfa`, `ard`, and `hddc`. It does not
compute PCA during training: the directions must already be present inside the
centroid artifact.

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

## HDDC shared noise and sub-epoch surgery

HDDC has two isotropic-noise modes. `isotropic_psi: true` learns one noise floor
`b_k` per component; `shared_b: true` learns one scalar `b` shared by the whole
mixture. The flags are mutually exclusive, and `shared_b` is supported only by
`single_process` HDDC training. When surgery is enabled, select exactly one of
these modes. During shared-b surgery, the new common noise floor pools the
residual covariance of all components that meet `surgery_min_count`.

The shared-b 20K toy configuration uses:

```yaml
model:
  kind: hddc
  K: 200
  q_max: 16
  shared_b: true
  surgery_every_epochs: 1
  surgery_threshold: 0.01
```

See
[`adaptive_q_toy_20k_hddc_shared_b.yaml`](../../configs/experiments/adaptive_q_toy_20k_hddc_shared_b.yaml)
for the complete pipeline configuration.

`surgery_every_epochs` also accepts a fraction strictly between 0 and 1 in
`single_process` mode. The fraction is converted to optimizer-step boundaries
using the resolved number of training batches per epoch. With `S` batches, for
example, `0.5` runs surgery after batch `ceil(S / 2)` and again at the epoch
boundary:

```yaml
model:
  kind: hddc
  shared_b: true
  surgery_every_epochs: 0.5
```

The full example is
[`adaptive_q_toy_20k_hddc_shared_b_surgery_half.yaml`](../../configs/experiments/adaptive_q_toy_20k_hddc_shared_b_surgery_half.yaml).
For other fractions, surgery runs after the first optimizer step that crosses
each cadence boundary in global epoch progress; the schedule does not reset at
each epoch. Each surgery performs an additional full E-pass over the training
split, so sub-epoch cadences can materially increase runtime. Values greater
than or equal to 1 must be integers; `0` disables surgery.

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

For toy-manifold runs, each Gaussian is associated with its unique nearest
planted manifold only when the exact mean-to-manifold distance is at most
`evaluation.max_mean_to_manifold_distance`. Distance ties are ambiguous and
remain unassociated. `metrics.json` records global association counts and one
entry per planted manifold with associated, assignment-live, and
assignment-dead counts. Rank recovery uses the proximity association;
assignments are used only for clustering and the explicit liveness diagnostic.

For a manifold of intrinsic dimension `r_i`, tangent alignment compares its
ground-truth tangent basis with the covariance subspace spanned by exactly
`PC1..PCr_i`. `subspace_overlap` is the mean squared cosine of their principal
angles, while `worst_direction_cosine` is the smallest cosine. Both scores are
sign- and basis-invariant and lie in `[0, 1]`. Tangent directions that occur
only in later PCs do not rescue the score.

Tangent containment separately compares the tangent with `PC1..PCs_k`, where
`s_k` is the component's effective rank. It gives full credit when the tangent
is contained in that possibly larger space, pads missing tangent directions
with zero when `s_k < r_i`, and assigns defined zero scores when `s_k = 0`.

The leading subspace is defined when the relative boundary eigengap between
eigenvalues `r_i` and `r_i + 1` exceeds `1e-6`; ties within the retained
subspace are valid. Containment applies the same rule at `s_k` and `s_k + 1`,
except at ranks zero and the full ambient dimension. Non-unique tangent
geometry and an undefined leading subspace are counted as undefined. Global
and per-manifold summaries are unweighted component means over
proximity-associated Gaussians, and an empty valid population has a JSON `null`
mean.

See [Toy-Manifold Tiling Evaluation](../evaluation/toy-manifold-tiling.md) for
the full association, effective-rank, alignment, and output-schema contract.
