# PCA sample-size validation

This experiment checks whether per-cluster intrinsic dimension, spectral
participation ratio, sample-corrected isotropy, and the top-100 empirical PCA
subspace have converged as the sample cap increases.

The default run compares the layer-5, K=1000 initial KMeans partition with the
trained q=10 MFA responsibility partition. It selects 64 same-ID clusters,
stratified across their joint population ranking and large enough to supply
20,000 samples under both assignments. Samples are nested: the 2,000 examples
are a prefix of the 5,000, 10,000, and 20,000 examples.

Submit the durable GPU job with:

```bash
sbatch scripts/pca_size_validation/sbatch_validation.sh
```

The main overrides are environment variables:

```bash
NUM_CLUSTERS=80 SAMPLE_SIZES="2000 5000 10000 20000" \
RUN_NAME=layer05_custom_seed1 SEED=1 \
sbatch scripts/pca_size_validation/sbatch_validation.sh
```

Set `OVERWRITE=1` to replace an existing run directory in place. The manifest
is marked `running` before PCA outputs are replaced and `complete` only after
all comparison tables have been regenerated.

For a targeted run, invoke the Python script directly and provide cluster IDs:

```bash
uv run python scripts/pca_size_validation/run_validation.py \
  --cluster-ids 12 84 310 \
  --sample-sizes 2000 5000 10000 20000 \
  --output-dir dalg-cache/pile_gemma2b_models/pca_size_validation/targeted
```

The output directory contains:

- `selected_clusters.csv`: selected IDs and their population in both partitions.
- `per_cluster_metrics.csv`: intrinsic dimension, participation ratio, and
  sample-corrected isotropy for every cluster and cap.
- `convergence_comparisons.csv`: scalar errors and top-PC principal-angle
  agreement against the 20k reference and against the next larger cap.
- `convergence_summary.csv`: median and 10th/90th percentiles across clusters.
- `<partition>/n<cap>/intrinsic_dims.pt`: the standard intrinsic-dimension
  result schema, restricted to the selected cluster IDs.
- `<partition>/n<cap>/cluster_top_pcs.pt`: the standard top-PC sidecar with up
  to 100 `(100, D)` bases for the selected clusters.
- `manifest.json`: exact input paths, fingerprints, settings, selected IDs, and
  completion status.

The runner refuses to write into a non-empty output directory so that a partial
or differently configured run cannot be mistaken for a fresh validation.

Open `visualize_results.ipynb` to display the latest completed run. Set
`PCA_VALIDATION_RUN_DIR=/path/to/a/run` before starting the kernel to select a
specific run instead.
