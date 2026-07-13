# Cluster labeling reference

## Preferred assignment-based workflow

```bash
uv run dalg-label-mfa-clusters \
  --assignments-path /path/to/mfa_run/mfa_model_assignments.pt \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --windows-dataset /path/to/windows_dataset/merged \
  --tokenizer google/gemma-2b \
  --out-dir /path/to/mfa_run/cluster_labels \
  --top-n 50 \
  --max-examples-per-cluster 25 \
  --pad 10 \
  --chunk-size 2000000 \
  --llm-workers 4 \
  --llm-temperature 0.0 \
  --llm-max-tokens 512
```

Outputs:

- `top_activations.pt`
- `cluster_examples.json`
- `cluster_labels.json` when LLM labeling runs

Use `scripts/slurm/sbatch_label_mfa_clusters.sh` for the cluster.

## Select only the requested stage

- Use `--skip-llm` to build top activations and context examples without external LLM calls.
- Use `--clusters 1 2 3` to label specific cluster IDs.
- Use `--max-clusters N` for a bounded debug run.
- Use `--top-index-path` to reuse or control the cached top-activation index.
- Reuse existing valid `top_activations.pt` or `cluster_examples.json` rather than rescanning by default.

Do not compute missing assignments automatically. Report the missing prerequisite and, if useful, point to the `dalg-compute-assignments` skill.

## Alignment requirements

- The assignment vector and shard metadata must describe the same filtered stream.
- When `--shard-dir` has no suffix, labeling can fall back to `subset_spec` recorded in the assignment bundle.
- Invert assignment positions through the same filtered metadata used during assignment generation.
- Recover token-window contexts from the matching Hugging Face windows dataset and tokenizer.

## Older interpretation path

`dalg-interpret-mfa` is an older integrated workflow. It can use assignments when available or scan the model for top responsibilities. Prefer `dalg-label-mfa-clusters` when assignments already exist.

## Description metrics

After `cluster_labels.json` exists, description-fit and semantic metrics can be run with `dalg-run-metrics`. Keep their outputs in the same `cluster_labels/` directory. Use the `dalg-compute-metrics` skill for exact metric commands and run only the requested metric.

## Validation

- Verify the requested cluster IDs appear in the examples and labels.
- Inspect a sample of recovered contexts for token-position correctness.
- Confirm `--skip-llm` did not produce or replace labels.
- Preserve existing labels unless relabeling was explicitly requested.
