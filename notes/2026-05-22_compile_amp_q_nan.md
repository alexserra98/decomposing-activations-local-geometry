# Component-sharded MFA training: NaN loss with `--compile` + `--use-amp` + large `q`

**Date:** 2026-05-22
**Status:** root cause identified; no fix applied yet.

## TL;DR

Component-sharded MFA training produces `nll=nan` within ~30 optimizer steps
when **all three** of the following are simultaneously true:

1. `--use-amp` is on (bf16 autocast inside `MFA._core`)
2. `--compile` is on (`torch.compile` wraps the model)
3. The MFA rank `q` is large (failure observed at `q=337`; works at `q=10`)

Dropping any one of the three fixes the run. Sharding by itself is not
required to *trigger* the bug — the failure mode is fundamentally about
`torch.compile` lowering bf16 graphs in a way that perturbs a
catastrophic-cancellation step in the MFA likelihood.

## Symptom

Reference failing log: [logs/jobs/mfa_train_component_shards_1219670_5.out](../logs/jobs/mfa_train_component_shards_1219670_5.out)

```
  ep 01 step    100/8924 (  1.1%) | nll=nan |  1.72 it/s | eta 1h25m
  ep 01 step    200/8924 (  2.2%) | nll=nan |  1.96 it/s | eta 1h14m
[epoch 01/01] done in 1m48s | train_nll=nan | val_nll=n/a | best_nll=inf @ ep00
```

Loss is finite at step 1, NaN by step 100, never recovers.

## Why it happens (mechanism)

The MFA likelihood in [src/dalg/models/mfa.py:_core](../src/dalg/models/mfa.py)
computes the quadratic form via the Woodbury identity:

```
quad = quad_Psi - low_rank          # low_rank = vᵀ M⁻¹ v
ll   = -0.5 · (D·log(2π) + log|C| + quad)
```

`quad` is the Mahalanobis distance `(x−μ)ᵀ C⁻¹ (x−μ)`, which is
mathematically `≥ 0`, but it's computed as a difference of two large
positive numbers. Inside the autocast block, `v = Wᵀ Ψ⁻¹ (x − μ)` is held
in bf16, and `low_rank` is a sum of `q` products of bf16 values. Per-entry
bf16 relative error is ~2⁻⁷ ≈ 1/128; the absolute error on `low_rank`
scales roughly with `q · ‖v‖²`.

* At `q=10` the absolute error is small enough that `quad` stays positive.
* At `q=337`, the error band approaches `quad_Psi` in magnitude. In eager
  mode this is still OK in practice (see repro below). Under
  `torch.compile`, inductor's fused kernels evaluate the same expression
  with a different reduction order and accumulator strategy, and the
  numerical drift is just enough to push `quad` negative.
* Once `quad < 0`, `ll → +∞`, the distributed log-sum-exp explodes, the
  optimizer takes a catastrophic step, and every subsequent loss is NaN.

The eager AMP repro (see Tests below) confirms that the bf16 error alone
is *not* enough at `q=512` — `torch.compile` is essential.

## How we narrowed it down (bisect)

All bisect jobs ran with `MAX_STEPS=200`, fresh `OUT_DIR`, reusing
existing centroids. Each is one single-axis flip of the failing config.

| Run | mode | q | AMP | batch | compile | sharded | result | log |
|---|---|---|---|---|---|---|---|---|
| original failure | shard | 337 | on | 8192 | on | 2 GPU | ❌ NaN | mfa_train_component_shards_1219129_5.out |
| **B0** (repro) | shard | 337 | on | 8192 | on | 2 GPU | ❌ NaN | mfa_train_component_shards_1219670_5.out |
| **B1** | vanilla | 337 | on | 2048 | off | no | ✅ ~2455 | mfa_train_shards_1219402_5.out |
| **B2** | shard | **10** | on | 8192 | on | 2 GPU | ✅ ~2715 | mfa_train_component_shards_1219398_5.out |
| **B3** | shard | 337 | **off** | 8192 | on | 2 GPU | ✅ ~2737 | mfa_train_component_shards_1219389_5.out |
| **B4** | shard | 337 | on | 8192 | **off** | 2 GPU | ✅ ~2449 | mfa_train_component_shards_1219453_5.out |
| eager repro | single-proc | 10–512 | on/off | 4096 | off | no | ✅ all finite | amp_q_nan_repro_1219439.out |

Reading: each ✅ row drops one axis off the failing config and the bug
goes away. That means **all three of AMP, compile, and large q are
necessary** to trigger it. Sharding's role was not directly isolated, but
the eager repro shows the failure is fully explained by the other three.

## How to reproduce

### Trigger the bug

Component-sharded sbatch with default flags hits the failing combination:

```bash
sbatch scripts/slurm/sbatch_train_component_shards.sh
```

The defaults in that script are `RANK=337`, `USE_AMP=1`, `COMPILE=1`,
`K=1000`, `BATCH=8192`, 2 H100s.

To reproduce specifically with a fresh `OUT_DIR` and an early stop:

```bash
BASE=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
sbatch --export=ALL,RANK=337,COMPILE=1,USE_AMP=1,EPOCHS=1,REFINE_EPOCHS=1,MAX_STEPS=200,\
OUT_DIR=$BASE/bisect_B0_repro_failing,\
CENTROIDS_FROM=$BASE/layer05_1000_337_mfa/centroids.pt \
       scripts/slurm/sbatch_train_component_shards.sh
```

Expect `nll=nan` in the log by step ~100. ~2 minutes wall.

### Confirm any one workaround makes it pass

Same command, drop **any one** of the three triggers:

```bash
# drop AMP
sbatch --export=ALL,RANK=337,COMPILE=1,USE_AMP=0,EPOCHS=1,REFINE_EPOCHS=1,MAX_STEPS=200,\
OUT_DIR=$BASE/bisect_B3_compshard_q337_noamp,\
CENTROIDS_FROM=$BASE/layer05_1000_337_mfa/centroids.pt \
       scripts/slurm/sbatch_train_component_shards.sh

# drop compile
sbatch --export=ALL,RANK=337,COMPILE=0,USE_AMP=1,EPOCHS=1,REFINE_EPOCHS=1,MAX_STEPS=200,\
OUT_DIR=$BASE/bisect_B4_compshard_q337_amp_nocompile,\
CENTROIDS_FROM=$BASE/layer05_1000_337_mfa/centroids.pt \
       scripts/slurm/sbatch_train_component_shards.sh

# drop q (use RANK=10)
sbatch --export=ALL,RANK=10,COMPILE=1,USE_AMP=1,EPOCHS=1,REFINE_EPOCHS=1,MAX_STEPS=200,\
OUT_DIR=$BASE/bisect_B2_compshard_q10_amp,\
CENTROIDS_FROM=$BASE/layer05_1000_10_mfa/centroids.pt \
       scripts/slurm/sbatch_train_component_shards.sh
```

### Eager-mode standalone repro (no shards, no distributed)

[tests/amp_q_nan_repro.py](../tests/amp_q_nan_repro.py) sweeps
`q ∈ {10, 64, 128, 256, 337, 512} × use_amp ∈ {False, True}` on synthetic
data, 50 Adam steps each, single GPU. **Without compile, all
configurations are finite** — this is the experiment that proved `torch.compile`
is essential to the failure.

```bash
sbatch --partition=H100 --account=LADE --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=4 --gres=gpu:H100:1 --mem=40G --time=00:15:00 \
  --job-name=mfa_amp_q_repro \
  --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/amp_q_nan_repro_%j.out \
  --wrap='cd /u/dssc/zenocosini/decomposing-activations-local-geometry && export PYTHONPATH=src && uv run python tests/amp_q_nan_repro.py --device cuda'
```

## Workarounds (no fix committed yet)

Until a fix lands, any one of these unblocks training:

1. **Drop `--compile`** — set `COMPILE=0` in the sbatch script.
   Recommended. AMP gives most of the H100 speedup; `torch.compile` on
   top is the part with the sharp edges for this code.
2. **Drop `--use-amp`** — set `USE_AMP=0`. Slower per step but numerically
   safe.
3. **Use small `q`** — only viable if the experiment design allows it.

## Open questions for a real fix

- Narrow the autocast region inside [_core](../src/dalg/models/mfa.py) so
  that `v = Wᵀ Ψ⁻¹ (x − μ)` and the `quad = quad_Psi - low_rank`
  computation stay in fp32, while leaving the cheap reductions in bf16.
  Should fix the cancellation issue while preserving most of the AMP
  speedup, but needs verification that `torch.compile`'s lowering of the
  narrowed graph also stays numerically clean.
- Add a CI guard (extension of [tests/component_sharded_mfa_equivalence.py](../tests/component_sharded_mfa_equivalence.py))
  that runs with `use_amp=True` + `compile=True` + a large-enough `q` to
  trip the bug, and asserts the loss stays finite. Without this,
  re-enabling `--compile` later would silently regress.
