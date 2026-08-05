---
name: dalg-compare-metrics
description: Inspect a user-supplied DALG experiment or sweep folder and compare the metrics of the runs it contains in a concise Markdown table. Use when the user passes a folder and asks to compare, summarize, or tabulate completed run metrics.
---

# Compare DALG Metrics

1. Resolve the folder supplied by the user and check that it is a directory.
2. Inspect its contents recursively for run folders containing `metrics.json`.
3. If the folder is empty, say that it is empty. If it contains no
   `metrics.json` run artifacts, say that it contains no completed runs with
   metrics. Do not invent rows or start training/evaluation.
4. Read each `metrics.json`. When present, also read the sibling `run_spec.json`
   to identify the configuration values that differ between runs.
5. Return one compact Markdown table with one row per run. Include the run name,
   differing configuration values, and scalar metrics. Always include these
   clustering columns when they exist, even when their values are identical
   across runs:
   - `clustering.homogeneity`
   - `clustering.completeness`
   - `clustering.adjusted_rand_index`
   - `clustering.normalized_mutual_information`

   Also include train/validation NLL and rank metrics when present. Flatten
   nested names into clear labels such as `nll.validation`. Omit paths, hashes,
   repeated configuration constants, and large arrays; do not omit a metric
   merely because it is constant across runs.
6. Sort rows by the differing configuration values when possible; otherwise
   sort by run name. Briefly list malformed `metrics.json` files after the table
   instead of silently dropping them.

Do not modify the folder or its artifacts.
