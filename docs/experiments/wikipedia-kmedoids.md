# Wikipedia KMedoids Slice

This is an attachable context file for the temporary Wikipedia KMedoids workflow. It is experimental analysis code, not core library API.

## Overview

The workflow materializes the deterministic Wikipedia activation slice and runs KMedoids on it. The native subset-suffix path remains streaming: `#pile_wikipedia_100K` resolves row positions and `ActivationBatchDataset` emits flattened activation batches from the original shards. `activations.npy` is only a derived random-access cache for KMedoids or CLARA.

Files:

- `scripts/materialize_subset_activations.py`: stream a subset-spec shard selection to `activations.npy`.
- `scripts/run_kmedoids.py`: run CLARA-style KMedoids and save medoids, labels, medoid indices, distances, and config.
- `tests/test_nearest_centroid_assignments.py`: unit tests for nearest-centroid assignment.

`labels.npy` is the nearest-medoid assignment for each row of `activations.npy`.

## Clean 100K run

Run root:

```text
outputs/experiments/pile_wikipedia_100K_layer05_kmedoids/full_100k_clean
```

Logs:

```text
full_100k_clean/logs/materialize.log
full_100k_clean/logs/run_kmedoids.log
```

Commands that were run:

```bash
mkdir -p outputs/experiments/pile_wikipedia_100K_layer05_kmedoids/full_100k_clean/logs

PYTHONPATH=src .venv/bin/python scripts/materialize_subset_activations.py \
  --shard-dir /orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations#pile_wikipedia_100K \
  --layer 5 \
  --out-dir outputs/experiments/pile_wikipedia_100K_layer05_kmedoids/full_100k_clean/data \
  > outputs/experiments/pile_wikipedia_100K_layer05_kmedoids/full_100k_clean/logs/materialize.log 2>&1

PYTHONPATH=src .venv/bin/python scripts/run_kmedoids.py \
  --activations-path outputs/experiments/pile_wikipedia_100K_layer05_kmedoids/full_100k_clean/data/activations.npy \
  --K 12 \
  --out-dir outputs/experiments/pile_wikipedia_100K_layer05_kmedoids/full_100k_clean/k12 \
  --device cuda \
  > outputs/experiments/pile_wikipedia_100K_layer05_kmedoids/full_100k_clean/logs/run_kmedoids.log 2>&1
```

## Verified outputs

- `full_100k_clean/data/metadata.json`: `subset_spec=pile_wikipedia_100K`, `resolved_rows=447`, `resolved_items=100128`, `materialized_items=100128`, `shape=[100128, 2048]`
- `full_100k_clean/data/activations.npy`: `(100128, 2048)`, `float32`
- `full_100k_clean/k12/medoids.npy`: `(12, 2048)`, `float32`
- `full_100k_clean/k12/labels.npy`: `(100128,)`
- `full_100k_clean/k12/config.json`: backend, cluster sizes, paths, and any sklearn-extra fallback error

Do not recompute assignments with `dalg-run-metrics assignments --medoids-path` after `run_kmedoids.py` unless a later step specifically needs the `.pt` assignment-bundle format. For the materialized-array workflow, `labels.npy` is already the assignment output.

## Package note

- `pyproject.toml` includes `scikit-learn-extra>=0.3.0`.
- In this environment, `sklearn_extra.cluster.CLARA` installs but fails to import with NumPy 2.4 ABI errors.
- `scripts/run_kmedoids.py` tries `sklearn-extra` first and falls back to a small local CLARA-style implementation.
- The backend and import error are recorded in `full_100k_clean/k12/config.json`.

Treat this implementation as easy-to-remove experimental code. Do not delete or overwrite its generated artifacts unless explicitly requested.
