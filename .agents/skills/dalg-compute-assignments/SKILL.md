---
name: dalg-compute-assignments
description: Compute, inspect, or repair hard activation assignments independently of the clustering source. Use MFA responsibility argmax when requested, or the nearest-Euclidean-centroid criterion for KMeans centroids, KMedoids medoids, MFA means, or any compatible centroid array. Also use for assignment bundle formats, subset-sliced streams, epoch assignments, peakedness statistics, and assignment-related Slurm jobs.
---

# Compute DALG Assignments

Compute only the assignment variant requested by the user. Do not automatically run intrinsic dimension, labeling, or other downstream stages.

## Procedure

1. Read `references/assignments.md` before planning or executing an assignment job.
2. Inspect `src/dalg/cli/run_metrics.py`, the selected implementation module, and the relevant Slurm script when behavior or defaults matter.
3. Determine the requested assignment rule: MFA responsibility argmax or nearest Euclidean centroid. Treat the latter as agnostic to how the centroids were obtained.
4. Verify the model or centroid artifact, activation shard configuration, layer, subset suffix, and intended save path.
5. Reuse an existing valid assignment file unless recomputation or repair was requested.
6. Save MFA-derived assignments inside the model run directory. Save centroid-only assignments beside the centroid or medoid artifact unless the user specifies otherwise.
7. Validate the saved bundle fields, `K`, item count, subset specification, and cluster-size total.

## Invariants

- Preserve canonical stream order; assignment vectors must remain aligned with activation items.
- Treat the saved hard partition as algorithm-agnostic downstream data. Record provenance, but do not require consumers to know whether it came from MFA, KMeans, KMedoids, or another method.
- Use the extraction config's `drop_prefix` unless explicitly overridden.
- Keep the same resolved subset positions across assignments, intrinsic dimension, and labeling.
- Use `--max-batches` only for smoke tests and make partial-output filenames explicit.
- Treat `labels.npy` from the materialized KMedoids workflow as an existing assignment output; do not create a `.pt` bundle unless a later consumer needs it.
