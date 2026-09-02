# HDDC Rank Surgery

> **Kind:** Experiment context · **Status:** Active · **Use when:** Modifying or
> reproducing the experimental HDDC rank-surgery path.

Attachable context for the periodic HDDC covariance-surgery path, which learns a
per-component rank `d_k <= q_max` instead of fixing it at `--rank`.

## The stack

A parallel stack that leaves the production files untouched, the same
arrangement as the ARD path:

- `src/dalg/models/adaptive_q/mfa_hddc.py`: `MFA_HDDC`,
  `ComponentShardedMFA_HDDC`, `save_mfa_hddc` / `load_mfa_hddc`,
  `save_component_shard_hddc` / `load_component_shards_hddc`
- `src/dalg/models/adaptive_q/hddc_surgery.py`: `SurgeryConfig`, `hddc_surgery`,
  `surgery_params`, `reset_optimizer_state`, `parameter_count`
- `src/dalg/models/adaptive_q/train_hddc.py`: `train_nll_hddc`
- `src/dalg/cli/adaptive_q/run_training_hddc.py`: `dalg-run-training-hddc`
- `scripts/slurm/adaptive_q/sbatch_train_hddc.sh`

`mfa.py`, `train.py`, and `run_training.py` are not modified. To remove the
feature, delete those files and the `dalg-run-training-hddc` entry in
`pyproject.toml`.

## What surgery does

Every `T` epochs, the closed-form covariance update of the HDDC model
`[a_ij b_i Q_i d_i]`, or its single-process `[a_ij b Q_i d_i]` variant,
(Bouveyron, Girard & Schmid, arXiv:math/0604064) re-estimates each component's
covariance at an adaptive rank and rewrites it in MFA parameters. Three phases:

- **A** — one E-pass accumulating, in float64, the responsibility-weighted second
  moment of each component about its *current* `mu_k`.
- **B** — per component: `eigh(S_k)`, a scale-free Cattell scree test on
  consecutive eigenvalue differences to propose a rank cap `r_k`, the noise
  level `b_k = (Tr(S_k) - sum_{j<=r_k} lam_j) / (D - r_k)`, then the
  reconstruction of `Sigma_k = W_k W_k^T + b_k I` with
  `scale_j = sqrt(lam_j - b_k)`. With `--shared-b`, eligible components instead
  estimate one pooled floor:

  ```text
  b = sum_k N_k (Tr(S_k) - sum_{j<=r_k} lam_kj)
      / sum_k N_k (D - r_k)
  ```

  Shared-b surgery then reconciles the independent Cattell proposals with the
  common floor. Directions `2..r_k` are considered globally from smallest to
  largest eigenvalue; any candidate with `lam_kj <= b` enters the pooled noise
  estimate before `b` is updated and the next candidate is considered. The
  remaining prefix defines `d_k`. Direction one stays mandatory, so an eligible
  component with `lam_k1 <= b` still raises instead of silently becoming rank
  zero. The implementation docstring in `_solve_shared_b_active_set` contains
  the full derivation, stopping rule, numerical-floor handling, scope, and
  rank-one policy. Reconstruction round-trips `b` through the model dtype before
  validating retained directions, so the check and loading scales use the floor
  that is actually written rather than only its float64 target.
- **C** — Adam state for the rewritten tensors is dropped, optionally followed by
  a short LR warmup.

Invariants worth preserving when editing:

- Covariances only. `mu` and `pi_logits` keep whatever SGD made them, and their
  optimizer state is preserved while the rewritten tensors' state is dropped.
- Statistics center on the current `mu_k`, never on the empirical
  responsibility-weighted mean — pairing a covariance centered at `mu_hat_k` with
  a retained `mu_k` is inconsistent and inflates apparent rank when the SGD means
  lag the data.
- `--isotropic-psi` selects component-specific `b_k`; `--shared-b` selects a
  global `b`. Exactly one is required for surgery, and shared-b is supported
  only in `single_process` mode.
- All `q_max` columns are rewritten every time and only the mask records `d_k`,
  so a rank *increase* needs no revival logic.
- Surgery is a partial M-step, so it competes for best-model selection on the
  same validation metric; otherwise a surgery landing on the final epoch would be
  discarded by the end-of-run rollback.
- Phase A accumulates an explicit `(K, D, D)` scatter, so this is a D≈128-scale
  path. There is a TODO for the large-D sketching route.

## Running it

Activation shards, same interface as `dalg-run-training`:

```bash
dalg-run-training-hddc \
  --shard-dir dalg-cache/pile_gemma2b_activations --layer 5 \
  --K 1000 --q-max 16 --isotropic-psi \
  --surgery-every-epochs 3 --surgery-threshold 0.01 --surgery-min-count 80 \
  --epochs 30 --device cuda --training-mode single_process
```

`--surgery-every-epochs 0` gives a fixed-q baseline on the same stack.

For the single-process shared-noise model, replace `--isotropic-psi` with
`--shared-b`. The two flags are mutually exclusive. The pooled `b` excludes
components below `surgery_min_count`, although their covariance floor still
changes because the scalar is global. A value of zero disables the cutoff and
includes every component with positive soft responsibility mass; exact-zero
membership raises because `S_k / N_k` is undefined. The active set lowers
Cattell rank caps when optional signal eigenvalues do not clear the common
floor. Surgery stops explicitly only when even a mandatory first eigenvalue is
not above `b`. The Slurm launcher makes the same selection with `SHARED_B=1`;
its default remains component-specific `b_k`.

To warm-start from a saved HDDC model, pass `--init-model-path mfa_model.pt` in
`single_process` mode. The source must exactly match `K`, `D`, `q_max`, and the
Psi noise mode. The trainer records this state as an epoch-0 checkpoint
with the initial validation NLL and a fresh Adam state, after which normal
checkpoint resume behavior applies.

## Toy-manifold validation data

Use the supported generator, `src/dalg/data/manifold_dataset.py`. It has no CLI;
import it and write activation-compatible shards under `dalg-cache/assets/`:

```python
from dalg.data import ToyManifoldConfig, save_toy_manifold_shards

cfg = ToyManifoldConfig(ambient_dim=128, n_samples=400_000,
                        manifolds_per_type=8, offset_radius=4.0, seed=0)
save_toy_manifold_shards(
    "dalg-cache/assets/toy_manifolds_D128_shards",
    cfg,
    shard_size=50_000,
    layer=0,
)
```

The resulting directory uses the same `config.json`, `layerNN/shard_NNNNN.pt`,
and `meta/shard_NNNNN.json` protocol as extracted Pile activations. Each toy
point is a one-position activation window, so train with
`--shard-dir dalg-cache/assets/toy_manifolds_D128_shards --layer 0`; the normal
`--val-frac` and `--split-seed` options create the split.

`manifold_metadata.pt` contains the row-aligned manifold IDs and the planted
dimension of every manifold instance for `d_k` recovery evaluation.

Two properties of this generator shape what it can measure: the planted
intrinsic dimensions are 1 and 2 only, and there is no ambient noise, so `b_k`
has no ground-truth target and instead reports the residual left by curvature and
tiling. Reading `d_k` against the planted per-manifold dimensions still works.

## Reading a run

- **`d_k` vs the planted dim.** Expect `d_k` slightly *above* the planted
  dimension where a component covers a curved patch: the patch is thick in the
  curvature direction, and that thickness is real variance, not an error. Score
  with a within-one band as well as exact match.
- **Saturation at `q_max`** is the "raise `q_max`" warning. Count it over
  components surgery actually touched — components below `n_min` are skipped and
  keep their initial full mask, so including them reports false saturation.
- **Live component count** matters as much as NLL. A mixture that collapses onto
  far fewer components than `K` is reporting the dimension of whatever each
  survivor covers, which may be a whole manifold rather than a local patch.
- **BIC** is not computed in the training loop. The toy-manifold evaluator adds
  standard training-set BIC to `metrics.json`; compare its `bic.value` across
  runs with lower values preferred.
