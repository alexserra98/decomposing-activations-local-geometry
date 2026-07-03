# K/NLL Relationship Experiment

This folder contains a self-contained synthetic MFA experiment for checking how
the optimal NLL changes as the true number of components `K` increases.

The runner creates five ground-truth MFA models with:

- `D = 500`
- `q = 20`
- `K in [10, 50, 100, 500, 1000]`

For each `K`, it samples a separate train/test dataset, then fits a fresh MFA
with the same `K` and `q` but independently initialized parameters. The summary
reports fitted train/test NLL, ground-truth train/test NLL, excess fitted test
NLL, and a linear fit of NLL versus `K`.

Run from this directory:

```bash
PYTHONPATH=../src uv run python run_k_nll_relationship.py
```

Main outputs are written under `runs/`:

- `K0010/`, `K0050/`, ...: per-K dataset, ground-truth model, fitted model,
  centroids, and metrics
- `summary.csv`: scalar metrics for every K
- `summary.json`: metrics plus linear-fit slope/intercept/R2
- `test_nll_vs_K.png`: fitted NLL, true NLL, and excess NLL versus K

Useful quick smoke test:

```bash
PYTHONPATH=../src uv run python run_k_nll_relationship.py \
  --K-grid 3,5 \
  --D 12 \
  --q 2 \
  --n-train 120 \
  --n-test 60 \
  --epochs 1 \
  --batch-size 32 \
  --device cpu \
  --no-plots \
  --out-dir /tmp/k_nll_smoke
```

The default full run can be expensive because the `K=1000` fit uses
`D=500, q=20` and KMeans initialization. Reduce `--n-train`, `--epochs`, or
`--kmeans-n-init` for exploratory passes.
