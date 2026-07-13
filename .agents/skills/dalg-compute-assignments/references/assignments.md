# Assignment reference

## MFA responsibility assignments

Preferred command:

```bash
uv run dalg-run-metrics assignments \
  --data-dir /path/to/mfa_run \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --batch-size 1024 \
  --device cuda
```

Equivalent direct module:

```bash
uv run python -m dalg.analysis.cluster_assignments \
  --model-path /path/to/mfa_run/mfa_model.pt \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --batch-size 1024 \
  --device cuda
```

The default output is `<run_dir>/mfa_model_assignments.pt`. Component-sharded runs can be addressed through their run directory; model loading falls back to `mfa_model_shards.json` when `mfa_model.pt` is absent.

Saved fields include:

- `cluster_sizes`: `(K,)`
- `assignments`: `(N,)`
- `max_responsibilities`: `(N,)`
- `peakedness`: per-cluster entropy, one-minus-max, and top1-minus-top2 summaries
- `K`
- `subset_spec` when a filtered shard suffix is used

Relevant launchers are `scripts/slurm/sbatch_assignments.sh` and `scripts/slurm/sbatch_epoch_assignments.sh`.

## Nearest-centroid assignments

Use this criterion for KMeans clusters, KMedoids, MFA means, or any other compatible centroid matrix. The assignment implementation only uses Euclidean distance; it does not depend on the algorithm that produced the representatives.

Pass `--medoids-path` or its `--centroids-path` alias instead of `--data-dir`:

```bash
uv run dalg-run-metrics assignments \
  --medoids-path /path/to/medoids.npy \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --batch-size 8192 \
  --device cuda
```

The default output is beside the centroid or medoid file. Current shared centroid collections live under `dalg-cache/pile_gemma2b_models/centroids/`. Use `--save-path` only when a downstream workflow expects a particular bundle name.

Saved fields include `cluster_sizes`, `assignments`, `min_distances`, `K`, `centroids_path`, `subset_spec`, and source metadata.

Downstream consumers should use the `assignments` and `cluster_sizes` fields as the partition itself. `centroids_path` and `source` record provenance but do not change the meaning of the hard assignments.

Implementation lives in `src/dalg/analysis/nearest_centroid_assignments.py`; `src/dalg/cli/run_metrics.py::cmd_assignments` selects the MFA or centroid path.

## Stream alignment and subsets

- `--drop-prefix` defaults to the extraction `config.json` value.
- The subset suffix has form `pile_<subset>_<N>[K|M]`, for example `#pile_wikipedia_100K`.
- Selection is deterministic and returned in sorted canonical stream order.
- Assignment item count and `cluster_sizes.sum()` must match the streamed item count.
- Labeling and intrinsic dimension must resolve exactly the same filtered metadata positions.
- Multi-worker interleaving can invalidate order-sensitive assignment vectors; use the current implementation's deterministic loader behavior.

## Existing KMedoids labels

For the materialized Wikipedia KMedoids workflow, `labels.npy` already contains the nearest-medoid assignment for every row of `activations.npy`. Do not recompute it through `dalg-run-metrics` unless a `.pt` assignment bundle is specifically required.

## Smoke tests and repair

- Use `--max-batches` for a bounded smoke test and keep its partial filename distinct.
- Before repairing an assignment file, inspect its source metadata, item count, ordering assumptions, and any existing backup.
- Do not move or rewrite legacy analysis outputs merely to normalize layout.
