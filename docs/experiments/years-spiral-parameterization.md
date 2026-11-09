# Layer-4 years spiral parameterization

> **Kind:** Experiment · **Status:** Pilot evaluated, 2026-11-04 · **Use when:**
> Fitting or inspecting an explicit year-dependent curve in layer-4 activations.

This temporary experiment compares a line, an affine one-period helix, a
generalized helix, and a cubic regression spline on the existing
`years_neutral` Llama-3.1-8B development activations. All regression objectives
use the original **4096 coordinates**, without PCA, whitening, or scaling the
activation features. The predicted year curve is supervised by the year labels;
its apparent smoothness is not an unsupervised discovery or proof of topology.

Related: [prompt extraction and development/test populations](../reference/prompt-activations.md),
[PC1 density alignment](pc1-density-alignment.md), and
[unsupervised evaluation](../evaluation/unsupervised.md).
For a two-dimensional reference sheet and Gaussian tangent-plane comparison,
see the separate [surface experiment](years-surface-parameterization.md).

## Model and interpretation

For year `y`, template `s`, and `t=y-1500`, fit

```text
x_hat(y,s) = origin + c + b_s + v*t/500
             + sum_j [u_j*cos(2*pi*t/T_j) + w_j*sin(2*pi*t/T_j)]
```

All coefficients are vectors in 4096D. Template offsets sum to zero over the
templates used to fit a model. An unseen template receives offset zero, the
mean of the fitted template offsets; no evaluation data calibrate its offset.
The constant baseline omits all year terms. The line omits periodic terms.
For the spline, cubic B-spline features replace the linear and periodic terms.

A single periodic pair produces a curve in an affine subspace of dimension at
most three. Unconstrained coefficients allow an affine deformation of a helix;
they do not require orthogonal axes or a circular cross-section. Multiple
periodic pairs can require more than three dimensions despite having one
continuous input parameter. The model describes a shared year trajectory plus
constant template shifts; interactions between year and context remain residuals.

The fitting functions live in the standalone temporary
[runner](../../scripts/temporary/fit_years_spiral.py), with no training-pipeline,
checkpoint, or installed-entrypoint changes. Exact year/template sufficient
statistics reproduce the ambient least-squares objective. The normal equations
are diagonally scaled and solved with a relative singular-value cutoff of
`1e-11`; numerically zero columns are excluded. Effective rank is recorded,
including the degenerate sine feature for period two on integer years.

## Selection and isolation

Only the root-level **16,016 development prompts**, years 1000–2000 crossed
with 20 templates before prompt holdout, are loaded. The source index and the
unshuffled `ActivationBatchDataset` stream must agree on every global row ID.
The existing external `test/` population is never read.

Within development, seed 42 permutes the 1,001 distinct year labels into
**600 fit / 200 validation / 201 evaluation years**. Seed 43 withholds four of
the 20 templates. All records for a year share its year-group identity.

1. Fit candidates on fit years and the 16 seen templates.
2. Select one candidate per family using validation-year ambient MSE on seen
   templates. Select the overall winning family using the same validation MSE.
3. Freeze all hyperparameters, refit each family on fit+validation years and
   seen templates, and evaluate three separate groups: new years / seen
   templates; seen years / new templates; new years / new templates.
4. Refit the validation-selected winner on all development rows for the final
   descriptive parameterization. No independent performance score is claimed
   for this final refit.

One-period candidates use periods 2, 5, 10, 20, 50 and 81 log-spaced periods
from 100 to 10,000 years. Five training-profile minima receive local bounded
refinement; validation chooses among the resulting candidates. Generalized
helices compare all subsets of two through four periods from 2, 5, 10, 100 and
the selected single period, adding that period only when distinct by at least
1% in log ratio. The spline compares 6, 10, 18, 34, 66, 130 and 258 uniformly
spaced knot sites, with fixed endpoints 1000 and 2000. Extrapolation is rejected.

Splines are unpenalized cubic regression splines; validation selects their
complexity through knot count. The candidate grid is a hypothesis class, not
an exhaustive search over all possible generalized helices. Periods longer
than the observed interval can be weakly identified and resemble a broad arc.

## Evaluated pilot

Slurm job **1904192** completed the comparison on 2026-11-04. All ten focused
tests passed through `srun`, including exact synthetic recovery, direct
squared-error checks, split isolation, analytic derivatives, and independent
period selection. The fit/validation populations contain 7,637/2,570 rows.
Final family comparisons use 10,207 fit+validation rows from 800 years and 16
templates; unseen-year evaluation contains **2,583 rows from 201 years**.
Withheld-template checks contain 2,557 seen-year and 669 unseen-year rows.

| Family | Selected setting | Unseen-year ambient MSE | Ambient R² | Curve dimensions for 95% variance |
| --- | --- | ---: | ---: | ---: |
| Line | Linear year trend | 0.00163324 | 0.1020 | 1 |
| One-period helix | Period 1412.54 years | 0.00149633 | 0.1773 | 3 |
| Generalized helix | Periods 5, 10, 100, 1412.54 years | 0.00130530 | 0.2823 | 8 |
| Cubic spline | 66 knot sites | 0.00116538 | 0.3592 | 48 |

The spline is selected by validation MSE and also has the lowest error in both
withheld-template groups. This comparison does not establish that the listed
Fourier periods exhaust the year representation: additional harmonics or other
functional forms were not searched. The single long period spans less than one
turn over the observed year interval, so its pitch/radius interpretation is
particularly weak.

Restricting the fitted spline's year trajectory to three curve PCs increases
unseen-year MSE from **0.00116538 to 0.00148199** and reduces ambient R² from
0.3592 to 0.1852, with the same template offsets. Its first three curve PCs retain
**40.44%** of fitted-curve variance. The 48-dimensional linear span summary is
not a claim of intrinsic dimension 48: the spline still has a single input,
year. Most ambient variation is unexplained even by this selected candidate.

Residual year-centroid MSE is 0.00112059, about 96.2% of the spline's residual
MSE; within-year MSE after fitted template offsets is 0.000044793. Substantial
year-dependent structure therefore remains, rather than all error being
within-year context variation. The final all-development refit is saved for
subsequent analysis, without an independent score for that refit.

Audit job **1904413** checked all 20 family/protocol MSEs against direct raw-row
predictions, with maximum discrepancy `1.09e-18`. An independent direct QR
least-squares solve of the selected spline agrees with the saved predictions
within `1.15e-14` per coordinate. All input/source hashes and split IDs match;
three-dimensional projections and finite analytic derivatives were also checked.
The [artifact audit](../../outputs/experiments/years_layer04_parameterization/job_1904192/artifact_audit.json)
records these checks. After refitting on all development rows and all templates,
the final spline's first three PCs retain 41.21% of its curve variance and 47
linear dimensions retain 95% (62 retain 99%); these differ slightly from the
evaluation model above.

The [completed report](../../outputs/experiments/years_layer04_parameterization/job_1904192/report.md)
links the figures. Exact metrics and selected basis specifications are in
[results.json](../../outputs/experiments/years_layer04_parameterization/job_1904192/results.json),
with split membership and input fingerprints in
[manifest.json](../../outputs/experiments/years_layer04_parameterization/job_1904192/manifest.json).

## Run and artifacts

From the repository root, with `logs/jobs/` present:

```bash
sbatch scripts/slurm/temporary/sbatch_years_spiral.sh
```

The [launcher](../../scripts/slurm/temporary/sbatch_years_spiral.sh) runs the
focused numerical tests, the experiment, and an
[independent artifact audit](../../scripts/temporary/audit_years_spiral.py)
through `srun`, using four CPU threads and 8 GiB. Outputs go to a new directory under
`outputs/experiments/years_layer04_parameterization/`; existing outputs cannot
be overwritten. The activation root can be overridden when invoking the runner
within an allocation.

Outputs include exact split membership and source row IDs, source/shard hashes,
candidate scores, frozen selection, per-family evaluation model bundles,
`results.json`, `report.md`, and residual MSEs by evaluation year. Ambient R²
uses the initial fitting population's mean as its fixed baseline. Weighted
residual-year-centroid MSE and within-year MSE sum to total ambient MSE; the
latter includes variation not represented by the shared year curve and constant
template shifts. Unseen-template scores include uncaptured context offsets.

`final_curve_all_development.npz` stores the final ambient coefficients, origin,
and basis specification without pickle. `curve_geometry.npz` contains its
sampled curve's own first three PCs, spectrum, retained variance, dimensions
needed for 95%/99% of **fitted-curve** variance, and analytic derivatives in
4096D. These dimensions do not estimate the intrinsic dimension of the full
activation cloud. Plotting subtracts fitted template offsets and projects onto
the fitted curve's PCs; captions distinguish this from raw activation PCA.
Each family's `three_dimensional_year_evaluation` additionally evaluates its
year trajectory restricted to its first three curve PCs, preserving the same
template offsets. This makes the cost of forcing the fitted curve into 3D
visible in the original evaluation coordinates.

To reuse the fitted parameterization:

```python
import runpy
api = runpy.run_path("scripts/temporary/fit_years_spiral.py")
model = api["load_model"]("outputs/experiments/years_layer04_parameterization/job_ID/final_curve_all_development.npz")
x_year = api["predict"](model, [1450, 1451])
tangent = api["predict"](model, [1450, 1451], derivative=True)
```

Coordinates between integer years describe the fitted curve; they are not
measured language-model activations for fractional-year prompts. The derivative
may support future local-PC comparisons, but this experiment does not compute
those comparisons or perform steering.

## Prior work

- [Kantamneni & Tegmark, 2025, Section 4](https://arxiv.org/html/2502.00873v1#S4):
  linear plus Fourier numerical representations, fitted in 100 PCA coordinates
  and mapped back to the residual stream. This experiment instead fits directly
  in ambient coordinates and introduces explicit template offsets.
- [Bhalla et al., 2026, Appendix B](https://arxiv.org/html/2604.28119v1#A2):
  the original years dataset is described as a helix, using different prompts
  and a different layer from this extended layer-4 dataset.
- [Hastie & Stuetzle, 1989](https://hastie.su.domains/Papers/Principal_Curves.pdf):
  nonparametric curves in arbitrary dimension. Our spline is supervised
  regression on year, not the unsupervised principal-curve algorithm.

Numerical tests are in [test_years_spiral.py](../../tests/test_years_spiral.py).
Remove the runner, audit script, launcher, focused tests, this page and its index entry to
remove the experiment; generated artifacts are separate.
