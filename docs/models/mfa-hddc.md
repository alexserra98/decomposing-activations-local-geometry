# MFA-HDDC

> **Kind:** Model explanation · **Status:** Current · **Use when:** Working on
> HDDC EM, covariance surgery, isotropic noise, rank masks, or HDDC checkpoints.

`MFA_HDDC` learns a **per-component rank** `d_k` by periodically re-estimating
each component's covariance in closed form and reading its rank off the
eigenspectrum. Unlike the ARD path it is a self-contained fork of `mfa.py`,
because it changes parameter shapes.

Code: `src/dalg/models/adaptive_q/mfa_hddc.py`, `hddc_surgery.py`,
`train_hddc.py`, `train_em_hddc.py`, `cli/adaptive_q/run_training_hddc.py`
(`dalg-run-training-hddc`). Method: the HDDC models `[a_ij b_i Q_i d_i]` and
`[a_ij b Q_i d_i]` of Bouveyron, Girard & Schmid (arXiv:math/0604064).

## The baseline it modifies

Vanilla MFA fits `C_k = W_k W_k^T + Psi` by Adam on the mean NLL, with `W_k` of
shape `(D, q)` and **one `q` fixed by hand for every component**, and a `Psi`
that is diagonal and (by default) shared across components.

## Streamed full-data EM

Select `--fit-method em --shared-b`, or `training.fit_method: em` with
`model.shared_b: true` in YAML. The default `fit_method: adam` retains the
Adam/surgery trainer. EM supports one CPU or CUDA process and
`1 <= q_max < D`; the intended scale is D=128 and K=500–5000.

The EM trainer freezes parameters for a complete pass and computes soft
responsibilities over **all K components**. It accumulates the same float64
counts, residual sums and scatter described below, then calls the shared HDDC
M-step to update means, mixture weights, orientations, individual signal
variances, ranks and the single noise scalar. Every component must have positive
effective mass; EM rejects unsupported components instead of skipping them.
There are no Adam steps, learning-rate schedules, subsampled E-passes, or online
averages. Cattell ranks and the shared-noise feasibility solve run every iteration.

Fresh fits load or fit KMeans centroids through the existing initialization
path, then make one full nearest-centroid pass over training activations to
initialize **all** parameters. This hard pass uses indexed moments rather than
dense soft memberships. Stored PCA directions are not required. Alternatively,
`init_model_path` uses a compatible shared-b HDDC model directly as iteration
zero, without refitting its parameters; K, D, q and noise mode must match.

Float64 cached likelihoods explicitly center observations before evaluating
quadratic forms, including for nonorthogonal warm-start loadings. Both
likelihoods and centered moments use component chunks. Covariance eigensolvers
are batched and only the leading q eigenvectors are retained. Model parameters
and exported checkpoints stay float32. The main persistent scatter costs
`8*K*D*D` bytes: **625 MiB at K=5000, D=128**. Working memory does not grow
with the total number of activations. Covariance accumulation costs O(N K D²),
while the eigensolver costs O(K D³) per iteration.

`epochs` counts M-steps; iteration zero is initialization. The next E-pass scores
each updated model while collecting the moments for its next M-step. The final
pass only scores. `em_history.json` records train/validation NLL, ranks, rank
changes, shared noise, effective memberships, timing and peak GPU allocation.
Checkpoint scores always refer to their actual parameter state. The trainer
selects the best validation NLL (or train NLL without validation), retaining
that model as `mfa_model.pt`; `checkpoint.pt` holds the current iteration and
best state for resume. Adam training checkpoints cannot be resumed as EM;
use a new output directory and `init_model_path` to switch fitting methods.

Convergence requires three consecutive iterations with unchanged ranks and
`abs(new_nll-old_nll)/max(1,abs(old_nll)) < em_tol`. Set `em_tol: 0` to disable
this criterion. Validation-patience controls also apply. The Adam
`early_stop_delta` criterion does not apply to EM. Adaptive Cattell ranks can
decrease likelihood; monotonic likelihood is only expected when ranks stay
fixed and the numerical constraints are inactive.

See the [YAML field reference](../reference/training-pipeline-config.md#full-data-em)
and [one-million-activation example](../../configs/experiments/hddc_em_D128_1M.yaml).
The temporary `scripts/temporary/benchmark_hddc_em.py` measures E/M and scoring
times on random vectors held in host memory; it measures throughput rather than
fit quality and excludes shard I/O, KMeans initialization and validation.

Measured on one H100 80GB with N=1,000,000, D=128, q_max=32, activation
batches of 2048 and component chunks of 32 (Slurm benchmark 1560498):

| K | E-pass | M-step | Final scoring pass | Peak GPU allocation |
| --- | --- | --- | --- | --- |
| 500 | 3.37 s | 0.89 s | 2.01 s | 0.48 GB |
| 5000 | 33.00 s | 7.83 s | 19.36 s | 2.80 GB |

These measure one update on seeded random inputs, not convergence time on
activation shards. All responsibilities are retained; no sparse approximation
is used. Output: `outputs/experiments/hddc_em_benchmark_1560498.json`.

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

Toy-manifold evaluation uses this saved mask count directly for rank recovery
and tangent containment. It does not filter active HDDC directions by their
loading variance relative to noise; `evaluation.rank_threshold` is ignored for
HDDC. Matched-dimensional tangent alignment still uses the planted intrinsic
dimension.

### 3. Periodic M-step surgery

Training is unmodified `train_nll` between surgeries — `train_nll_hddc` differs
from `train_nll` only by the `surgery=` argument and the block it gates. Every
`--surgery-every-epochs` epochs a gated **M-step** runs. Following the
[HDDC paper, sections 4.1–4.2](https://arxiv.org/pdf/math/0604064#page=12), it
uses one frozen set of soft responsibilities for means, weights, and covariance:

- **A — statistics.** One E-pass over the train loader accumulating, in float64,
  counts `N_k = sum_n r_nk`, residual sums
  `A_k = sum_n r_nk (x_n - mu_k_old)`, and scatter
  `B_k = sum_n r_nk (x_n - mu_k_old)(x_n - mu_k_old)^T`. Centering the
  accumulation on the old means avoids subtracting raw second moments at the
  scale of the data's absolute offset. For eligible components, compute
  `mu_k_new = mu_k_old + A_k / N_k` and the empirical covariance
  `C_hat_k = B_k / N_k - (A_k / N_k)(A_k / N_k)^T`, centered on the new mean.

  Eligibility requires both `N_k > 0` and `N_k >= n_min`. Skipped means and
  mixture probabilities are preserved. If `E` is the eligible set and
  `P_E = sum_{k in E} pi_k_old`, the constrained weight update is
  `pi_k_new = P_E * N_k / sum_{j in E} N_j` for `k in E`. When all components
  are eligible this is the paper's `pi_k_new = N_k / n`. The model preserves
  skipped logits and the eligible logits' total exponential mass, keeping the
  softmax normalizer unchanged up to floating-point rounding. Component shards
  compute eligible masses globally.
- **B — rank selection and rewrite.** Per component, `eigh(C_hat_k)`, then a
  scale-free Cattell scree test on consecutive eigenvalue differences,

  ```text
  r_k = max{ j <= q_max : (lam_j - lam_{j+1}) / lam_1 > threshold }
  b_k = (Tr(C_hat_k) - sum_{j<=r_k} lam_j) / (D - r_k)    # mean discarded eigenvalue
  ```

  In shared-b mode, equation 5 of the HDDC paper pools eligible components:

  ```text
  b = sum_k N_k (Tr(C_hat_k) - sum_{j<=r_k} lam_kj)
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
  `N_k < n_min` or `N_k == 0` do not contribute to the pooled estimate and keep
  their means, weights, directions, and mask. The default `n_min = 0` disables
  the positive-count cutoff: every component with positive soft responsibility
  mass is rewritten, including a
  component with zero hard assignments. An exactly zero `N_k` is always skipped
  because its empirical mean and covariance are undefined. Skipped components'
  covariance floor still changes because `b` is global.
- **C — optimizer hygiene.** Adam state for all five rewritten parameter tensors
  (`mu`, `pi_logits`, `dir_raw`, `scale_rho`, `psi_rho`) is dropped, optionally
  followed by a short LR warmup. If no components are eligible globally,
  surgery leaves the model, optimizer state, and warmup schedule unchanged.

Empty E-passes, negative counts, and nonfinite statistics raise explicitly.
Parameter proposals are validated on every component shard before any commit;
successful rewrites invalidate the affected model's inference cache. The
statistics API returns `(N_k, residual_sum, scatter, n_rows)` and
`reconstruct_components` consumes the three moment tensors plus its configuration.

Epoch-boundary surgery runs *after* each epoch's best-model bookkeeping, so the
selected metric and the state it selected describe the same model — but it competes on
the same validation metric, otherwise a surgery landing on the final epoch would
be thrown away by the end-of-run rollback.

## At a glance

| | MFA | MFA-HDDC |
| --- | --- | --- |
| Rank | one global `q` | per-component `d_k <= q_max`, explicit |
| Psi | diagonal, shared `(D,)` or `(K, D)` | `b_k I` `(K, 1)`, or single-process `b I` `(1,)` |
| Extra state | — | `rank_mask` buffer `(K, q_max)` |
| Optimization | Adam on mean NLL | Adam with periodic M-steps, or streamed full EM |
| Rank mechanism | — | Cattell scree test on the covariance eigenspectrum |
| Checkpoint | — | **not** readable by `mfa.load_mfa` |
| Relation to `mfa.py` | — | fork, not subclass |

With `fit_method: adam`, set `--surgery-every-epochs 0` for a fixed-`q` baseline on the identical stack —
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
- `src/dalg/models/adaptive_q/train_em_hddc.py`: full-data EM, hard initialization,
  bounded moment accumulation, convergence and resume
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

- **Checkpoint compatibility.** Use `load_mfa_hddc`, not `mfa.load_mfa`, for
  HDDC parameters. The assignment loader and toy-manifold evaluator support
  HDDC model files from either fitting method. `MFAEncoderDecoder` also accepts
  the public model interface.
- **D ≈ 128 scale only.** Phase A accumulates an explicit `(K, D, D)` scatter
  (128 KiB per component in float64 at `D=128`). Gemma-scale `D ≈ 2304` needs the sketching
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
