# Synthetic MFA Q/K Sweep Offline Report

Open `synthetic_mfa_qk_sweep_results_executed.ipynb` to read the executed
notebook with plots.

Supporting files:
- `notebook_plots/`: PNG plots and standalone HTML Plotly figures.
- `support/results.csv`: scalar sweep metrics.
- `support/results.pt`: full collected sweep metrics.
- `support/feature_splitting.csv` and `support/feature_splitting.pt`: feature
  splitting summaries used by the notebook.
- `support/bhattacharyya_by_q_*.csv` and `support/bhattacharyya_by_q_*.pt`:
  Bhattacharyya summaries used by the interactive heatmaps.
- `support/config.json`: run configuration.

The trained `mfa_model.pt` files are intentionally excluded because they are
about 72 GiB and are not needed for offline plot review.
