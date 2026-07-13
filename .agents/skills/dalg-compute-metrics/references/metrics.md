# Metrics reference

## Output policy

- Save geometric model metrics in the model run directory.
- Save assignment-derived metrics beside the assignment bundle when unambiguous.
- Save comparison partitions in descriptive subdirectories such as `nearest_centroid_metrics/` or `shuffled_assignments_metrics/` to avoid filename collisions.
- Save centroid assignments and their derived metrics with the centroid collection under `dalg-cache/pile_gemma2b_models/centroids/`.
- Leave existing `dalg-cache/output/` contents untouched and do not use that legacy root for new metrics.

Do not force every metric through a model directory. Model-geometric metrics need a model; assignment-derived metrics can use an explicit assignment bundle without `--data-dir`.

## Component overlap

```bash
uv run dalg-run-metrics overlap \
  --data-dir /path/to/mfa_run \
  --device cuda \
  --batch-pairs 512
```

Output: `<model_dir>/overlap.pt` when `--out-dir` is omitted.

Overlap is pure MFA Gaussian geometry and needs no assignment bundle, shard directory, or subset filtering. It is not currently a metric over arbitrary hard partitions. For high latent rank, reduce `--batch-pairs`; large pair batches can create tensors proportional to `batch_pairs x D x q` and cause GPU OOM.

Relevant launcher: `scripts/slurm/sbatch_metrics.sh`.

## Intrinsic dimension

```bash
uv run dalg-run-metrics intrinsic-dim \
  --assignments-path /path/to/partition_assignments.pt \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --device cuda \
  --pca-device cpu \
  --pca-workers 8 \
  --variance-threshold 0.90 \
  --min-population 100 \
  --max-samples-per-cluster 2000
```

Output: `intrinsic_dims.pt` beside the assignment bundle when both `--data-dir` and `--out-dir` are omitted.

The assignment bundle may come from MFA responsibility argmax, nearest-centroid KMeans or KMedoids assignments, shuffled labels, or another source. Intrinsic dimension only requires a compatible `.pt` bundle containing at least `assignments` and `cluster_sizes`; provenance metadata is used for validation and reporting, not to select the algorithm. Confirm that the subset specification, item count, and stream order match the selected shards. PCA can run on CPU even when activation work uses CUDA.

For shuffled or nearest-centroid variants, use a model-local subdirectory rather than overwriting the responsibility-based `intrinsic_dims.pt`.

## Description fit

Run only after cluster labels and contexts exist:

```bash
uv run dalg-run-metrics description-fit \
  --labels-path /path/to/model/cluster_labels/cluster_labels.json \
  --out-dir /path/to/model/cluster_labels \
  --positive-examples 8 \
  --negative-examples 8 \
  --judge-workers 4
```

Output: `description_fit.json`.

## Description semantics

```bash
uv run dalg-run-metrics description-semantics \
  --labels-path /path/to/model/cluster_labels/cluster_labels.json \
  --out-dir /path/to/model/cluster_labels \
  --embedding-device cpu \
  --top-k 25 \
  --similarity-threshold 0.70
```

Outputs include `description_semantics.pt` and JSON summaries. Other label-coherence subcommands are defined in `src/dalg/cli/run_metrics.py`; inspect the current parser before composing commands.

## Validation

- Confirm the expected file was created in the intended model-local directory.
- Inspect `K`, matrix or vector shapes, finite values, and configuration metadata when present.
- For label JSON, verify cluster coverage and referenced example structure.
- Do not treat successful file creation as proof that assignments or subset alignment were correct.
- Report clearly that overlap is model-geometric, while intrinsic dimension is assignment-partition based.
