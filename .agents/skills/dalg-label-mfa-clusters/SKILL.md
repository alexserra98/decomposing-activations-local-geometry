---
name: dalg-label-mfa-clusters
description: Build top activation contexts and produce natural-language labels for DALG MFA clusters. Use when recovering token-window examples from assignments, labeling selected clusters with the hosted LLM, running without LLM calls, reusing top-activation indexes, or editing and debugging the cluster-labeling Slurm job.
---

# Label DALG MFA Clusters

Perform only the requested labeling stage. Do not recompute assignments automatically, and do not call the hosted LLM when the user requests examples only or `--skip-llm` behavior.

## Procedure

1. Read `references/labeling.md` before planning or executing labeling work.
2. Inspect `src/dalg/cli/label_mfa_clusters.py` and `scripts/slurm/sbatch_label_mfa_clusters.sh` when current arguments or defaults matter.
3. Identify the assignment bundle, matching activation shards and subset selection, layer, windows dataset, tokenizer, target clusters, and requested LLM behavior.
4. Verify that assignments and recovered metadata refer to the same canonical filtered stream.
5. Save top activations, examples, and labels under `<model_dir>/cluster_labels/` unless the user specifies another model-local location.
6. Reuse cached top-activation indexes and existing examples when valid.
7. Validate requested cluster coverage and output JSON. Leave description metrics to the `dalg-compute-metrics` skill unless the user separately requests them.

## Invariants

- Prefer the assignment-based labeling workflow when assignments already exist.
- Keep example generation separable from LLM labeling.
- Limit or select clusters for debugging instead of relabeling the entire model.
- Treat LLM calls as an external side effect; honor explicit skip and scope requests.
- Do not run description metrics until `cluster_labels.json` exists.
