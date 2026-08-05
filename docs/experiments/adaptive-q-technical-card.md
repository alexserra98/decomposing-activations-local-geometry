# Adaptive-q MFA toy-manifold experiments

## Scope

Two completed single-seed experiments compare adaptive per-component rank in an
MFA. Both use `K=100`, ambient dimension `D=128`, and maximum rank `q_max=32`:

| Run | Rank mechanism | Artifact directory |
| --- | --- | --- |
| ARD MFA | MAP shrinkage of loading columns | `dalg-cache/toy_manifold_models/toy_manifolds_8types_2each_D128_150K_K100_q32_mfa_ard/` |
| HDDC MFA | Periodic covariance eigenspectrum surgery and a hard rank mask | `dalg-cache/toy_manifold_models/toy_manifolds_8types_2each_D128_150K_K100_q32_mfa_hddc/` |

The models have the same full loading shape `(100, 128, 32)`. ARD has 425,828
trainable scalar parameters and HDDC has 425,800; pruning or masking changes the
active covariance rank but does not shrink the stored tensors.

## Dataset

The shared dataset is
`dalg-cache/assets/toy_manifolds_8types_2each_D128_150K_shards/`.

- 150,000 float32 points in `R^128`, one point per activation-shard row.
- Eight manifold types, with two independently embedded instances per type:

  | Type | Planted intrinsic dim. | Pre-embedding dim. |
  | --- | ---: | ---: |
  | segment | 1 | 1 |
  | circle | 1 | 2 |
  | flat disk | 2 | 2 |
  | sphere | 2 | 3 |
  | torus | 2 | 3 |
  | Möbius strip | 2 | 3 |
  | Swiss roll | 2 | 3 |
  | helix | 1 | 3 |

- The 16 manifold instances are exactly balanced at 9,375 points each (18,750
  per type).
- Each type is centered and normalized to unit RMS using 50,000 calibration
  samples, then embedded by a random orthonormal map and translated by radius
  4 in a random ambient direction. No ambient observation noise is added.
- Generator seed: 0. Generator settings record 120,000 train and 30,000
  validation samples, but those sets are concatenated when shards are written.
  Training then makes a new deterministic split over all 150,000 rows:
  148,800 train / 1,200 validation (`val_frac=0.008`, split seed 42), balanced
  at 18,600 / 150 rows per manifold type.

## Common initialization and training

- Single-process float32 MFA on one NVIDIA H100 80 GB; training seed 42.
- `K=100`, `q_max=32`, uniform initial mixture weights, loading scale 1, and
  unique variance 1.
- Means initialized independently in each run by projected reservoir KMeans:
  29,760-point reservoir, projection dimension 32, followed by 10 refinement
  epochs. The saved centroid files are different, so the two runs do not share
  an identical initialization.
- Adam, learning rate `1e-3`, no weight decay, batch size 2,048, 73 optimizer
  steps per epoch, two data-loader workers, and no gradient clipping.
- Selection metric is validation NLL only. The best validation checkpoint is
  restored. Consecutive-epoch delta stopping uses `1e-3`; patience stopping is
  disabled. Both runs reached their full epoch budgets.
- Saved hard assignments are responsibility argmax over the complete canonical
  150,000-row stream.

## ARD MFA

- Shared diagonal unique variance `Psi`; rank is the number of loading columns
  whose variance exceeds `mean(Psi)` for that component.
- Gamma precision prior: `alpha0=1`, `b0=1e-4`, `ard_lambda=1`. Because the NLL
  is a per-point mean, the applied prior weight is `1 / 148800 = 6.72043e-6`.
- Twenty epochs. ARD pressure is zero for the first 15% (epochs 1--3), ramps
  linearly over the next 20% (epochs 4--7), then stays at full strength.
- Post-training pruning uses threshold `1 x mean(Psi)`, zeros both loading scale
  and direction, and retains the unpruned checkpoint.
- Best checkpoint: epoch 20, validation NLL 46.5024 before pruning; recomputed
  final validation NLL 45.9061 after pruning.

## HDDC MFA

- One isotropic unique variance per component, `Psi_k = b_k I`, plus a hard
  `(K, q_max)` loading mask.
- Every three epochs, a float64 E-pass accumulates responsibility-weighted
  covariance around the current model means. A scale-free Cattell scree rule
  selects `d_k <= 32` with threshold 0.01, and the covariance is reconstructed
  as `W_k W_k^T + b_k I`.
- Default minimum effective count is `max(5 q_max, 50) = 160`. Components below
  it are skipped and retain their previous mask; rewritten covariance tensors
  have their Adam state reset. No post-surgery LR warmup is used.
- Thirty epochs, hence ten scheduled surgeries. Best checkpoint: epoch 30 after
  surgery, validation NLL -601.8929 (notebook recomputation: -601.8993).

## Current evaluation snapshot

Metrics use all 150,000 responsibility-argmax assignments. Rank recovery pairs
each non-empty component with its dominant planted manifold instance.

| Metric | ARD | HDDC |
| --- | ---: | ---: |
| Train NLL | 45.8994 | -601.8898 |
| Validation NLL | 45.9061 | -601.8993 |
| Homogeneity | 1.0000 | 1.0000 |
| Completeness | 0.9593 | 0.9969 |
| Adjusted Rand index | 0.9585 | 0.9980 |
| Normalized mutual information | 0.9792 | 0.9985 |
| Live / dead components | 20 / 80 | 17 / 83 |
| Mean learned rank, live components | 0.85 | 2.47 |
| Exact planted-rank match, live components | 0.0% | 29.4% |
| Within-one planted-rank match, live components | 30.0% | 88.2% |
| Mean absolute rank error, live components | 2.25 | 0.82 |

“Planted-rank match, live components” compares each non-empty MFA component's
learned rank with the known intrinsic dimension of the synthetic manifold it
represents. Each live component is paired with the manifold instance that
contributes the most points assigned to that component. The exact score counts
`q_k = d_manifold`; the within-one score counts
`|q_k - d_manifold| <= 1`. Dead components are excluded because they receive no
points and therefore have no meaningful corresponding manifold rank. The
within-one score is useful because a Gaussian covering a curved manifold patch
may need one additional covariance direction.

Interpret the all-component rank summaries carefully. ARD pruned 95 of 100
components to rank zero, including 18 components that still receive points by
their mean and diagonal covariance. HDDC leaves low-count skipped components at
`q_max`: 80 components report rank 32, but none of the 17 assignment-live
components is saturated. Live-component or surgery-eligible summaries are the
meaningful rank diagnostics.

## Limits

- One dataset seed and one training seed; there is no uncertainty estimate.
- There is no fixed-q control run on this exact model stack in these artifacts.
- `K=100` is overcomplete relative to 16 planted manifolds, so the dead-component
  rate is part of the result rather than a dataset defect.
- The data have no ambient noise. HDDC drives its residual variance to the
  numerical floor, so its very negative NLL should not be read as a standalone
  rank-recovery result or compared naively with noisy-data likelihoods.
- The ARD setting is a column-collapse result at `lambda=1`, not evidence of
  successful rank recovery. A useful ARD study needs a lambda/warmup sweep.

The executable evaluation is in `notebooks/evaluate_adaptive_q.ipynb`; switch
`MODEL_DIR` between the two artifact directories above.
