# MFA-ARD

> **Kind:** Model explanation · **Status:** Current · **Use when:** Working on
> ARD rank shrinkage, training behavior, checkpoint compatibility, or rank
> readout.

`MFA_ARD` learns a **per-component rank** `q_k` by letting an ARD prior shrink
whole columns of `W_k` to zero. It is a subclass of `MFA` that changes the
*objective* and nothing else.

Code: `src/dalg/models/adaptive_q/mfa_ard.py`, `train_ard.py`,
`cli/adaptive_q/run_training_ard.py` (`dalg-run-training-ard`).

## The baseline it modifies

Vanilla MFA models activations as a mixture of low-rank Gaussians

```text
p(x) = sum_k pi_k N(x | mu_k, C_k),     C_k = W_k W_k^T + Psi
```

with `W_k` of shape `(D, q)` and **one `q` fixed by hand for every component**.
Loadings are parameterized as direction times scale, `W_k[:, j] = d_hat_kj *
s_kj`, where `d_hat` is unit-norm over `D` and `s = softplus(scale_rho)`.
Training is Adam on the mean NLL.

The problem: activation regions are not all equally complex. A single global `q`
either starves the rich regions or gives the simple ones spurious directions.

## What ARD changes

### 1. A Gamma–Gaussian prior on each loading column

Every column of every `W_k` gets its own precision `nu_kj`:

```text
p(w_j^k | nu_j^k) = N(0, (nu_j^k)^-1 I_D)      nu_j^k ~ Gamma(alpha0, b0)
```

The MAP objective adds to the NLL

```text
sum_{k,j} [ 1/2 ||w_j^k||^2 nu_j^k + b0 nu_j^k - (D/2 + alpha0 - 1) log nu_j^k ]
```

This is a **group** penalty — one term per column, not per matrix entry. That is
exactly what makes it a rank prior: it removes whole directions, not individual
weights.

### 2. `nu` is eliminated in closed form, so no parameters are added

The penalty is convex in `nu` with minimizer `nu* = c / (1/2 s^2 + b0)`,
`c = D/2 + alpha0 - 1`. Recomputing and detaching `nu` each forward pass gives
exactly the gradient of the profiled penalty `c * log(1/2 s^2 + b0)` — this is
not an approximation, it is gradient descent on the `nu`-eliminated objective.

Two consequences follow directly from the parameterization:

- Because `_dir_hat()` normalizes over `D`, `||w_j^k|| == s_kj` **exactly**, so
  the penalty is a function of `scale_rho` alone and `W` is never materialized.
- No new parameters exist, so an `MFA_ARD` `state_dict` is byte-identical in
  structure to a plain `MFA` one. `mfa.load_mfa` reads ARD checkpoints, and
  every downstream analysis (assignments, Gaussian overlap, intrinsic dimension,
  labeling) works on ARD runs with no code changes.

### 3. Why this shrinks rank where L2 would not

The gradient of the profiled penalty w.r.t. a column scale is

```text
d/ds [ c log(1/2 s^2 + b0) ] = c * s / (1/2 s^2 + b0)
```

- large `s`: pull ≈ `2c / s` — the shrinkage *weakens* as a column grows, so
  genuinely useful directions are barely taxed;
- small `s`: pull ≈ `(c / b0) * s` — a stiff linear well with stiffness
  `lambda * c / b0`, so a column that dips near zero is pinned there.

Plain weight decay has uniform stiffness at every scale and therefore shrinks
everything a little instead of killing a few things completely. `weight_decay`
stays at 0 in this codebase; all shrinkage on `W` comes from this term.

### 4. Training-loop consequences

- **Penalty weight.** The prior applies once per dataset, the loss is a
  per-sample mean, so the applied weight is `--ard-lambda / n_train`.
- **Beta warmup (required).** Full pressure from a cold start drops every column
  into the stiff `s -> 0` well before the loadings mean anything, and the run
  ends up worse *on the ARD objective itself* than the same lambda reached via a
  ramp. `ard_beta_schedule` holds beta at 0 for the first `--ard-warmup-frac`
  of the epoch horizon, ramps linearly over the next `--ard-ramp-frac`, then
  holds at 1. The horizon is stored in the checkpoint and a resume that changes
  it is rejected, since beta is a function of `(epoch, horizon)`.
- **Selection ignores the penalty.** Best-model tracking and early stopping use
  validation NLL alone, so ARD runs stay directly comparable to baseline MFA.

### 5. Rank is read out, not stored

`q_k` is not a parameter. `effective_ranks()` counts columns carrying more
variance than the noise they sit on:

```text
q_k = #{ j : s_kj^2 > rank_threshold * mean_d Psi_kd }
```

The reference is `Psi`, not the component's largest column, on purpose: a
peak-relative cutoff is blind to over-collapse — when ARD kills *every* column
the ratios stay near 1 and the count would wrongly report full rank. Against
`Psi`, a collapsed component correctly reports `q_k = 0`.

`prune_columns()` makes it permanent by zeroing both halves of the
factorization, and is a **post-training step only** — applying it mid-training
would freeze decisions the loadings can never undo. The training loop never
calls it.

## At a glance

| | MFA | MFA-ARD |
| --- | --- | --- |
| Rank | one global `q` | per-component `q_k <= q_max`, emergent |
| Parameters | `mu, dir_raw, scale_rho, psi_rho, pi_logits` | identical |
| Psi | shared `(D,)` or per-component `(K, D)` | unchanged |
| Objective | mean NLL | mean NLL + `beta * lambda/N * ARD penalty` |
| Rank mechanism | — | continuous column shrinkage, thresholded post hoc |
| Checkpoint | — | readable by `mfa.load_mfa` |

## Implementation organization

ARD is an intentionally separate research stack so it can evolve without
changing the fixed-rank implementation in `mfa.py`, `train.py`, or
`run_training.py`:

- `src/dalg/models/adaptive_q/mfa_ard.py`: `MFA_ARD` and checkpoint helpers
- `src/dalg/models/adaptive_q/train_ard.py`: `train_nll_ard` and
  `ard_beta_schedule`
- `src/dalg/cli/adaptive_q/run_training_ard.py`: the single-process
  `dalg-run-training-ard` entrypoint
- `scripts/slurm/adaptive_q/sbatch_train_ard.sh`: cluster launcher
- `tests/test_mfa_ard.py`: model, schedule, training, and checkpoint coverage

The `adaptive_q/` directories deliberately have no `__init__.py`; they are
implicit namespace packages. The console script therefore resolves through the
full `dalg.cli.adaptive_q.run_training_ard:main` path declared in
`pyproject.toml`.

ARD and HDDC currently remain redundant experimental implementations. The
intention is to converge on one adaptive-rank route, delete the other, and fold
the survivor back into the main model and CLI directories. The core ARD model
is isolated, but the YAML pipeline and toy-manifold evaluator now select and
load it explicitly. Before removing the variant, search the full repository for
`model.kind: ard`, `MFA_ARD`, and `adaptive_q.run_training_ard` references in
code, tests, configs, scripts, and entrypoints.

## Costs and failure modes

- Rank is **soft**: the reported `q_k` depends on `--rank-threshold`, and a
  column near the noise floor is a judgement call, not a decision the model made.
- Needs a `lambda` / warmup sweep. The single measured run at `lambda=1` is a
  column-collapse result (95 of 100 components pruned to rank 0), not evidence
  of rank recovery — see `docs/experiments/adaptive-q-technical-card.md`.
- Set `--rank` (i.e. `q_max`) generously; ARD can only remove columns.

Related: [MFA-HDDC](mfa-hddc.md) (the alternative, hard-rank route) and the
[adaptive-q technical card](../experiments/adaptive-q-technical-card.md)
(measured results).
