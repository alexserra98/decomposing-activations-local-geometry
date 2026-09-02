# Research Backlog (snapshot 2026-07-30)

> **Kind:** Research backlog · **Status:** Historical snapshot · **Use when:**
> Recovering earlier open problems, fallback approaches, and completed next
> steps.

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
- [x] Implement L_ARD as opt-in regularizer in train_nll
- [x] Scree-plot backup script under scripts/temporary/

# Research State (updated 2026-08-31)

I have created a toy manifold dataset to test how much the adaptive-q mfa could recover the real rank of each local patch. Initially I have created the dataset without any noise and all of the models tended to converge to a configuration where all the gaussian components died (no dataset point is assigned to them) and only a number equal to the manifold in the dataset survived.
Unexpectedly adding a bit of noise proportional to the curvature of each manifold pushed mfa models away to the bad local minima and some mfa started to actually tile.
At the moment I have measured the quality of a tiling through 3 metrics:
 - NLL
 - Homogeneity
 - Number of alive components
According to this metrics MFA-hddc seemed to outperform MFA-ard in terms of NLL, although it tended to kill many gaussians. MFA-ard on the other tended to keep more gaussian alive and prune directions in W more aggressively. I decided to keep going with MFA-hddc and tried different strategies. Eventually I found out that to recover rank more succesfully I had to change the initial implementation so that the "noise" component of each gaussian, aka the directions with eigenvalue b, aka the set of directions the varies independentely, I needed to be constrain b to be equal for all directions and for all gaussians.
In this setup I tried different scheduling of the rank surgery and find out that the more I do it the more the NLL decreases.
I tried to build a notebook to visualise manifolds and their tiles but the viz tool is still very drafted. Nontheless inspecting the tiles and their PCs I have discovered that for that if I compute the PCA for the covariance matrix what I get is that first PC is always orthogonal to the tangent space of the manifold, while the other PCs are aligned with the tangent space. If just look at the survived components that only those to whcih have been assigned some dataset points, I get that the first PC is aligned with the tangent space only for bigger tiles while for smaller tiles the first PC is orthogonal to the tangent space.
for bigger tiles the PC aligned effectively with the tangent space of the manifold, while for the smaller tiles the PC where exactly orthogonal to that tangent space. This suggests that I need to run a second check to effectively split or merge components so to preserved the right tangent dimension.
The goal of the project is to find feature manifolds in a unsupervised way, so one might wonder why do I care to get the exact tangent space of each local patch. The reason is not entirely clear to me but generally It seems plausible that modelling the local tangent space through linear directions can be important for 2 different reasons:
- Linear features hypothesis of features is an important branch of interpretability and this local charts might be the right bridge between features manifolds and lineare features
- This information can be useful to model the final manifold faithfully.

At the moment there are many issue with my setup such for example: the dataset is too simple and my measure of goodness of tiling is sloppy.

The current goals are:
- [x] Improve and finalise the visualisation notebook
- [x] Switch to more complicated dataset with mulitple manifolds with different ID and try different noise levels
- [x] Find a better metric to assess the quality of tiling.
