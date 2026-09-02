# MFA-HDDC

> **Kind:** Model explanation · **Status:** Current · **Use when:** Working on
> HDDC covariance surgery, isotropic noise, rank masks, or HDDC checkpoints.

`MFA_HDDC` learns a **per-component rank** `d_k` by periodically re-estimating
each component's covariance in closed form and reading its rank off the
eigenspectrum. Unlike the ARD path it is a self-contained fork of `mfa.py`,
because it changes parameter shapes.

Code: `src/dalg/models/adaptive_q/mfa_hddc.py`, `hddc_surgery.py`,
`train_hddc.py`, `cli/adaptive_q/run_training_hddc.py`
(`dalg-run-training-hddc`). Method: the HDDC models `[a_ij b_i Q_i d_i]` and
`[a_ij b Q_i d_i]` of Bouveyron, Girard & Schmid (arXiv:math/0604064).

## The baseline it modifies

Vanilla MFA fits `C_k = W_k W_k^T + Psi` by Adam on the mean NLL, with `W_k` of
shape `(D, q)` and **one `q` fixed by hand for every component**, and a `Psi`
that is diagonal and (by default) shared across components.

## What HDDC changes

### 1. Isotropic noise: `Psi_k = b_k I` or `b I`

With `--isotropic-psi`, `psi_rho` has shape `(K, 1)` — one scalar per component,
broadcast over `D` — instead of `(D,)` or `(K, D)`. The single-process-only
`--shared-b` mode instead stores one scalar with shape `(1,)`, shared across
components and dimensions. The flags select different models and are mutually
exclusive.

This is not a convenience, it is what makes the whole method exact. For
isotropic noise the spectrum of `Sigma_k = W_k W_k^T + b_k I` is

```text
lam_j = s_j^2 + b_*   for j <= d_k        (signal directions)
lam_j = b_*           for j >  d_k        (noise floor)
```

where `b_*` is either `b_k` or the shared `b`. Its eigenvectors are the columns
of `W_k` plus an arbitrary orthonormal completion. So an eigendecomposition of
an empirical covariance converts *exactly* into MFA parameters: eigenvectors
become directions, `sqrt(lam_j - b_*)` becomes scales, and the rank is wherever
the spectrum flattens onto its floor. With an anisotropic diagonal `Psi` there
is no such correspondence, and the CLI requires one of the isotropic modes.

### 2. A hard rank mask

A non-trainable buffer `rank_mask` of shape `(K, q_max)` gates the loading
columns. It is folded into the scale inside `_W()`:

```python
s = self._scale() * self.rank_mask      # (K, q)
return d_hat * s[:, None, :]            # (K, D, q)
```

A masked column is therefore *exactly* zero in `W`, drops out of
`C_k = W W^T + Psi`, and both `dir_raw` and `scale_rho` receive exactly zero
gradient through it — no stop-gradient machinery needed. `component_ranks`
reads `d_k = rank_mask.sum(-1)` straight off the buffer. The mask is part of the
`state_dict`, so it round-trips through save/load and shards like the other
per-component tensors.

### 3. Periodic covariance surgery instead of pure SGD

Training is unmodified `train_nll` between surgeries — `train_nll_hddc` differs
from `train_nll` only by the `surgery=` argument and the block it gates. Every
`--surgery-every-epochs` epochs a closed-form **partial M-step** runs:

- **A — statistics.** One E-pass over the train loader accumulating, in float64,
  `N_k = sum_n r_nk` and the responsibility-weighted scatter
  `S_k = sum_n r_nk (x_n - mu_k)(x_n - mu_k)^T` about the **current model mean**
  `mu_k`. Centering on `mu_k` rather than on the empirical
  responsibility-weighted mean is deliberate: `mu_k` is retained, so this is the
  ML covariance given a fixed mean. Pairing a covariance centered at `mu_hat_k`
  with a retained `mu_k` would leak the mean shift into the spectrum and inflate
  the apparent rank whenever the SGD means lag the data.
- **B — rank selection and rewrite.** Per component, `eigh(S_k / N_k)`, then a
  scale-free Cattell scree test on consecutive eigenvalue differences,

  ```text
  r_k = max{ j <= q_max : (lam_j - lam_{j+1}) / lam_1 > threshold }
  b_k = (Tr(S_k) - sum_{j<=r_k} lam_j) / (D - r_k)        # mean discarded eigenvalue
  ```

  In shared-b mode, equation 5 of the HDDC paper pools eligible components:

  ```text
  b = sum_k N_k (Tr(S_k) - sum_{j<=r_k} lam_kj)
      / sum_k N_k (D - r_k)
  ```

  For component-specific `b_k`, the Cattell proposal is the final rank
  `d_k = r_k`; surgery raises if an imposed numerical floor nevertheless
  overtakes a retained eigenvalue. For shared `b`, the independently proposed
  ranks can be inconsistent with the common absolute floor: reconstruction
  requires every retained eigenvalue to satisfy `lam_kj > b`, because its
  loading variance is `lam_kj - b`. Shared-b surgery therefore treats `r_k` as
  a rank cap and runs an active-set solve. It starts with all `j > r_k` in the
  pooled noise estimate, then considers optional directions `2 <= j <= r_k`
  globally from smallest to largest eigenvalue. A candidate enters the noise
  pool when `lam_kj <= b`; the weighted pooled floor is updated before
  considering the next candidate. Once the next candidate is above `b`, all
  larger candidates remain active. This ordering avoids the over-pruning that
  would result from discarding every violation against the initial, higher
  floor in one batch.

  Direction one remains mandatory, preserving `d_k >= 1`. Surgery still fails
  explicitly if the final floor reaches `lam_k1`, because that component would
  require an unsupported rank-zero covariance. The active set is a feasibility
  update conditional on Cattell's caps, not a replacement intrinsic-dimension
  criterion. Surgery reports the initial `b_shared_at_cattell` and the numbers
  of pruned directions and affected components for diagnosis. Before validating
  or reconstructing loadings, it round-trips the selected floor through the
  model dtype and `softplus(psi_rho) + model._eps`. The reported `b`, the strict
  retained-eigenvalue check, and `sqrt(lam_j - b)` therefore all use the value
  actually stored by the model rather than an idealized float64 target.

  The reconstruction uses `Sigma_k = W_k W_k^T + b_* I` with
  `scale_j = sqrt(lam_j - b_*)`. All `q_max` columns are written from the
  eigendecomposition and only the mask records `d_k`, so a later surgery can
  *raise* a component's rank with no revival logic. Components with
  `N_k < n_min` do not contribute to the pooled estimate and keep their
  directions and mask. The default `n_min = 0` disables this cutoff: every
  component with positive soft responsibility mass is rewritten, including a
  component with zero hard assignments. An exactly zero `N_k` cannot define an
  empirical covariance and raises explicitly. Skipped components' covariance
  floor still changes because `b` is global.
- **C — optimizer hygiene.** Adam state for the rewritten tensors (`dir_raw`,
  `scale_rho`, `psi_rho`) is dropped, optionally followed by a short LR warmup.
  State for `mu` and `pi_logits` is preserved.

Surgery touches covariances only. `mu` and `pi_logits` keep whatever SGD made
them. It runs *after* each epoch's best-model bookkeeping, so the selected
metric and the state it selected describe the same model — but it competes on
the same validation metric, otherwise a surgery landing on the final epoch would
be thrown away by the end-of-run rollback.

## At a glance

| | MFA | MFA-HDDC |
| --- | --- | --- |
| Rank | one global `q` | per-component `d_k <= q_max`, explicit |
| Psi | diagonal, shared `(D,)` or `(K, D)` | `b_k I` `(K, 1)`, or single-process `b I` `(1,)` |
| Extra state | — | `rank_mask` buffer `(K, q_max)` |
| Optimization | Adam on mean NLL | Adam on mean NLL + closed-form M-step every `T` epochs |
| Rank mechanism | — | Cattell scree test on the covariance eigenspectrum |
| Checkpoint | — | **not** readable by `mfa.load_mfa` |
| Relation to `mfa.py` | — | fork, not subclass |

Set `--surgery-every-epochs 0` for a fixed-`q` baseline on the identical stack —
that is the control an adaptive-rank claim needs.

## Implementation organization

HDDC is a self-contained research fork so its changed parameter shapes and
periodic surgery do not modify `mfa.py`, `train.py`, or `run_training.py`:

- `src/dalg/models/adaptive_q/mfa_hddc.py`: full and component-sharded models,
  masks, and checkpoint helpers
- `src/dalg/models/adaptive_q/hddc_surgery.py`: statistics, reconstruction,
  parameter selection, and optimizer-state reset
- `src/dalg/models/adaptive_q/train_hddc.py`: the training-loop fork gated by a
  `surgery` configuration
- `src/dalg/cli/adaptive_q/run_training_hddc.py`: the
  `dalg-run-training-hddc` entrypoint
- `scripts/slurm/adaptive_q/sbatch_train_hddc.sh`: cluster launcher
- `tests/test_hddc_surgery.py`: model, surgery, training, sharding, and
  checkpoint coverage

The CLI supports `single_process`, where one process owns the full model, and
`component_shard`, which partitions components across CUDA processes. Reserve
`vanilla` for the original fixed-rank MFA implementation. Shared `b`, warm-start
from a full model, and fractional-epoch surgery are single-process-only.

The `adaptive_q/` directories deliberately have no `__init__.py`; they are
implicit namespace packages. The console script resolves through the full
`dalg.cli.adaptive_q.run_training_hddc:main` path declared in `pyproject.toml`.

ARD and HDDC currently remain redundant experimental implementations. The
intention is to converge on one adaptive-rank route, delete the other, and fold
the survivor back into the main model and CLI directories. The core HDDC stack
is isolated, but the YAML pipeline, toy-manifold evaluator, assignment loader,
configs, and temporary experiment scripts now integrate it. Before removing the
variant, search the full repository for `model.kind: hddc`, `MFA_HDDC`,
`hddc_surgery`, and `adaptive_q.run_training_hddc`; removal is no longer only a
five-file deletion.

## Costs and failure modes

- **Checkpoint incompatibility.** The mask and isotropic `psi_rho` shapes make an
  `MFA_HDDC` `state_dict` unreadable by `mfa.load_mfa`, so downstream analyses
  do not consume HDDC runs. This is deliberate: a model worth analysing gets
  retrained on the production stack. (`MFAEncoderDecoder` is the exception — it
  calls public methods only and accepts an `MFA_HDDC` unchanged.)
- **D ≈ 128 scale only.** Phase A accumulates an explicit `(K, D, D)` scatter
  (65 KB per component at `D=128`). Gemma-scale `D ≈ 2304` needs the sketching
  route, which is a TODO.
- **Reading `d_k` needs care.** Skipped low-count components keep a full mask and
  report `d_k = q_max`, which looks like false saturation; count saturation only
  over components surgery actually touched. Expect `d_k` slightly *above* a
  planted dimension where a component covers a curved patch — that thickness is
  real variance.
- **Shared-b is deliberately single-process.** It is rejected with
  `training_mode=component_shard`; component-sharded checkpoints retain their
  existing noise modes and formats.
- **The shared floor must remain below every retained eigenvalue.** The active
  set removes optional Cattell directions that do not clear the jointly updated
  floor. Surgery still fails with the offending component if even its mandatory
  first eigenvalue is not above `b`; clamping would report a rank-one component
  whose only loading variance was actually zero.
- On noiseless data, `b_k` is driven to the numerical floor and the NLL goes
  strongly negative; do not compare it naively with noisy-data likelihoods.

Related: [MFA-ARD](mfa-ard.md) (the soft-shrinkage route),
[HDDC rank surgery](../experiments/hddc-rank-surgery.md) (how to run and read a
run), [adaptive-q technical card](../experiments/adaptive-q-technical-card.md)
(measured results), and the
[toy-manifold dataset reference](../reference/toy-manifold-dataset.md)
(validation data).
