---
name: dalg-compute-metrics
description: Compute or inspect DALG analysis metrics from the input each metric actually requires. Use for intrinsic dimension from any compatible hard-assignment bundle regardless of whether it came from MFA, KMeans, KMedoids, shuffled labels, or another partition; for Gaussian overlap directly from MFA distribution geometry; and for description or label-coherence metrics from label artifacts. Also use when choosing devices, editing metric Slurm jobs, or validating outputs.
---

# Compute DALG Metrics

Run only the metric or metric set explicitly requested. Check prerequisites, but do not generate missing assignments, labels, or models unless the user asks.

## Procedure

1. Read `references/metrics.md` before planning or executing a metric job.
2. Inspect `src/dalg/cli/run_metrics.py` and the relevant analysis module when exact defaults or output schemas matter.
3. Identify the metric's actual input contract: an MFA model for Gaussian overlap, a hard-assignment bundle plus activations for intrinsic dimension, or label artifacts for description metrics. Do not infer the assignment algorithm unless provenance is relevant to naming the output.
4. Validate all required inputs before starting an expensive job.
5. Save geometric model metrics in the model run directory. Save assignment-derived metrics beside the assignment bundle or in a descriptive partition subdirectory when standard filenames would collide.
6. For centroid partitions, keep assignments and derived metrics with the centroid collection under `dalg-cache/pile_gemma2b_models/centroids/`. Never place new metrics under legacy `dalg-cache/output/`.
7. Verify expected output files and inspect basic tensor shapes or JSON structure after completion.

## Invariants

- Intrinsic dimension treats any compatible hard-assignment bundle as an opaque partition; it requires `assignments` and `cluster_sizes`, not a particular clustering algorithm.
- Gaussian overlap is the exception to assignment agnosticism: it depends only on MFA Gaussian geometry and does not consume assignments or activation shards. It is distinct from neighbor overlap, which compares neighborhoods.
- Reduce Gaussian overlap pair batching for high-rank models instead of risking GPU OOM.
- Keep label-derived metrics beside the model's `cluster_labels/` artifacts.
- Do not overwrite the standard metric when computing shuffled, nearest-centroid, epoch, or other comparison variants.
