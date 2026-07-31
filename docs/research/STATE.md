# Research State (updated 2026-07-30)

The goal is to extend MFA to do manifold learning. The idea is to set k reasonably high, and let the model learn per-component rank q_k adaptively so tile the manifolds. After that we will explore methods to connect the tiles into a global manifold representation. 

## Current approach
Adaptive per-component rank q_k via SGD + ARD regularizer on columns of W_k:
  L = L_MFA + Σ_j [ ½||w_j^k||² ν_j^k + b0 ν_j^k − (D/2 + α0 − 1) log ν_j^k ]
Exploits existing dir_raw/scale_rho factorization. 

## Backup method
Standard MFA training; each epoch run per-component PCA on fuzzy covariance
(Bouveyron-style), pick q_k by scree plot. Manual, but no loss changes.

## Open problems (no owner / no solution yet)
- Superposition: MFA assigns each point predominantly to one chart; features
  in superposition violate this. No immediate fix; park until adaptive-q works.
- Mean-field/ARD over-pruning risk: component collapse already observed.

## Next
- [x] Toy manifold dataset (see backlog.md#toy-manifolds)
- [ ] Implement L_ARD as opt-in regularizer in train_nll
- [ ] Scree-plot backup script under scripts/temporary/