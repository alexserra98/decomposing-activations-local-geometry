# Synthetic MFA Analyses

> **Kind:** Experiment context · **Status:** Experimental · **Use when:** Running
> or interpreting the temporary synthetic MFA sweep.

This is an attachable context file for the temporary synthetic MFA research workflow under `scripts/` and `notebooks/`. Treat it as analysis code, not core library API.

## Main sweep

Main script:

- `scripts/synthetic_dataset/synthetic_mfa_qk_sweep.py`

Purpose:

- Generate data from a known MFA.
- Fit MFA models over a grid of fitted `K` and `q`.
- Record responsibility peakiness and label-recovery metrics.
- Collect results and plots.

Typical commands:

```bash
PYTHONPATH=src python scripts/synthetic_dataset/synthetic_mfa_qk_sweep.py generate-dataset \
  --dataset-path /orfeo/scratch/dssc/zenocosini/dalg-cache/assets/synthetic_mfa_Ktrue1000_qtrue20_D500_seed0.pt \
  --D 500 --K-true 1000 --q-true 20 \
  --n-train 500000 --n-test 10000 --seed 0

PYTHONPATH=src python scripts/synthetic_dataset/synthetic_mfa_qk_sweep.py fit-one \
  --dataset-path /path/to/synthetic_dataset.pt \
  --model-root dalg-cache/qk_sweep_exploration \
  --run-name Ktrue1000_qtrue20 \
  --K-fit 1250 --q-fit 20 \
  --device cuda

PYTHONPATH=src python scripts/synthetic_dataset/synthetic_mfa_qk_sweep.py collect-results \
  --model-root dalg-cache/qk_sweep_exploration \
  --run-name Ktrue1000_qtrue20
```

## Related analysis

- `scripts/slurm/sbatch_synthetic_qk_sweep.sh`: Slurm array over fitted `K`, with an inner loop over `q` values.
- `scripts/synthetic_dataset/synthetic_mfa_bhattacharyya_by_q.py`: post-hoc
  Gaussian overlap and Bhattacharyya summaries across `q` for a fixed fitted
  `K`.
- `scripts/synthetic_dataset/synthetic_mfa_feature_splitting.py`: feature-splitting and covariance-reconstruction analysis over fitted sweep models.
- `scripts/slurm/sbatch_feature_splitting.sh`: Slurm wrapper for feature splitting.
- `notebooks/archived/synthetic_mfa_qk_sweep_results.ipynb`: exploratory result notebook.
- `outputs/experiments/synthetic_qk_sweep_report/`: offline report artifacts.

Generated synthetic models can be very large and live under `dalg-cache/`. Do not delete or overwrite them unless explicitly requested.

Keep new experimental modules direct, easy to add, and easy to remove. Avoid turning this workflow into core library API without an explicit request.
