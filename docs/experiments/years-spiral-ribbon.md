# Layer-4 explicit spiral ribbon

> **Kind:** Experiment · **Status:** Pilot evaluated, 2026-11-04 · **Use when:**
> Inspecting an explicit continuous spiral surface fitted to the years cloud.

This experiment replaces the unconstrained sheet of the
[generic surface pilot](years-surface-parameterization.md) with an explicit
generalized helical ribbon. It reuses the activation reader, development splits,
Fourier features and slow-period proposal search from the
[year-curve experiment](years-spiral-parameterization.md). Gaussian axes are not
used to fit the surface, and no Gaussian alignment scores are computed here.
The subsequent [HDDC versus KMeans comparison](years-ribbon-hddc-kmeans-alignment.md)
uses this fixed reference and the wiki's alignment and containment definitions.

## Parameterization

For `z = y - 1500`, fit the vector-valued curve

```text
c(y) = origin + C0 + C1*z/500
       + sum_j [Aj*cos(2*pi*z/Tj) + Bj*sin(2*pi*z/Tj)]
d(y) = Am*cos(2*pi*z/Tm) + Bm*sin(2*pi*z/Tm)
S(y,r) = c(y) + (r-1)*d(y)
```

Every coefficient vector has **4096 coordinates**. `y` is the year coordinate
in `[1000,2000]`; positive `r` scales one fitted oscillation pair and provides
the second surface coordinate. `r=1` is the central generalized spiral. This
is a ruled ribbon winding along a prescribed Fourier trajectory, not a free
rectangular spline sheet. Multiple periods allow an oscillation around a slowly
curving trend and faster numerical structure; the result is not restricted to
an affine three-dimensional classical helix.

The [runner](../../scripts/temporary/fit_years_spiral_ribbon.py) alternates ambient
least-squares coefficient fits with exact radial projections at the supplied
years. The mean training radius is normalized to one, rescaling its coefficient
pair to preserve predictions. Radii remain positive; tiny values are clipped
at `0.001` during optimization. After fitting, the displayed radial domain is
the training radii's 0.5th–99.5th percentile interval, with a positive lower
bound. The surface between integer-year observations and across its radial
range is a modeling assumption, not observed fractional-year activations or a
uniquely identified transverse geometry.

The normal equations use column scaling and a relative singular-value cutoff
of `1e-11`; numerically zero feature columns are removed. Optimization uses up
to 60 iterations and a relative MSE tolerance of `1e-7`, followed by a final
coefficient solve consistent with the stored training radii.

## Selection and checks

Only the **16,016 root development rows** are read; the external `test/`
population remains unread. Existing internal year/template grouping is reused:
600 fit years, 200 validation years, 201 evaluation years and four withheld
templates. These are exploratory development checks following earlier analyses
of the same extraction, not a new untouched external benchmark.

The existing curve fitter proposes a slow period from its training profile and
validation choice. Ribbon candidates then compare:

- a single 100-year oscillation;
- a single slow oscillation;
- slow plus 100-year oscillations, with a radial 100-year pair;
- slow plus 100-, 10- and 5-year oscillations, with a radial 100-year pair.

Validation ambient reconstruction MSE selects the model. Year is supplied and
one radius is inferred from each query activation, clipped to the training
domain. Scores therefore measure reconstruction conditional on year, not
year-only prediction or a global nearest-point search over both parameters.
No categorical template offset is added to the fitted ribbon.

## Pilot results

Slurm fit job **1905025** selected periods **5, 10, 100 and 1412.5375 years**.
The final all-development fit converged in **20 iterations**, with radius domain
**[0.53031, 1.64768]** and no nonpositive training radius projections. This is
a continuous approximate spiral surface; most ambient variation is still not
explained by this small model.

| Evaluation group | Ambient MSE | Ambient R² | R² in training PC1–PC3 | Radius clipped at edge |
| --- | ---: | ---: | ---: | ---: |
| New years, seen templates | 0.00132926 | 0.2691 | 0.9392 | 0.97% |
| Seen years, new templates | 0.00122506 | 0.3052 | 0.9448 | 0.59% |
| New years, new templates | 0.00128835 | 0.2797 | 0.9419 | 1.05% |

The visible fit is substantially better than its ambient reconstruction:
global PC1–PC3 display only **18.79%** of activation variance. A second display
uses an empirical year-trend direction followed by the fitted 100-year cosine
and sine coefficient vectors, orthonormalized in ambient space. These learned
spiral axes display **15.11%** of activation variance. Neither view subtracts
a drift, template offsets, or residuals from the plotted raw activations.

The two-coordinate extension permits analytic tangents:
`S_y = c'(y) + (r-1)*d'(y)` and `S_r = d(y)`. Those tangents are tangents to
this estimated ribbon. Good projected reconstruction does not establish that
they recover the full activation cloud's true local geometry. In particular,
the radial direction is imposed by the model family; it is not independently
identified from a uniformly sampled two-dimensional manifold.

`tangent_space(model, years, radius)` now returns orthonormal ambient bases of
shape `(N,4096,2)`. It forms the two analytic Jacobian columns, normalizes their
lengths to remove dependence on year/radius units, and uses a thin SVD. Zero,
nonfinite or linearly dependent partials raise an error instead of inventing
a second tangent direction. The affine tangent plane is `S(y,r) + span(Q)`.

The saved fitted model has nine independent nonconstant coefficient vectors
(minimum singular value `0.4250`). Its linear-trend vector is therefore outside
the span of all oscillation vectors, and the radial cosine/sine vectors are
independent. This ensures Jacobian rank two throughout the parameter domain.
A numerical check at **303 parameter locations** returned two orthonormal
directions in all 4096 coordinates, with maximum Gram-matrix error `1.12e-15`.
The [sampled bases and partials](../../outputs/experiments/years_layer04_spiral_ribbon/job_1905025/sampled_tangent_spaces.npz)
and [validation record](../../outputs/experiments/years_layer04_spiral_ribbon/job_1905025/tangent_validation.json)
preserve those results. Five focused tests now pass, including agreement with
an independent finite-difference tangent plane and rejection of singular maps.

Independent audit job **1905037** reproduced the evaluation metrics exactly.
A direct QR solve for all 4096 response coordinates agrees with the saved
coefficients within `1.70e-15`; analytic year/radius partials agree with finite
differences within `3.52e-10` and `1.37e-11`. Source/input hashes and exact split
row IDs match. The final descriptive projected R² is 0.9532 in global PC1–PC3
and 0.9827 in the learned spiral axes; these all-development refit scores are
not independent evaluation scores.

## Artifacts and rerun

```bash
sbatch scripts/slurm/temporary/sbatch_years_spiral_ribbon.sh
```

The [launcher](../../scripts/slurm/temporary/sbatch_years_spiral_ribbon.sh) runs
five focused [tests](../../tests/test_years_spiral_ribbon.py), the fit and an
[independent audit](../../scripts/temporary/audit_years_spiral_ribbon.py) on Slurm
CPUs. The tests cover synthetic ribbon recovery, ambient projection, analytic
partials, tangent rank, monotone optimization, independent QR coefficients,
serialization and parameter-domain checks.

The [pilot directory](../../outputs/experiments/years_layer04_spiral_ribbon/job_1905025/)
contains [results](../../outputs/experiments/years_layer04_spiral_ribbon/job_1905025/results.json),
frozen selection, row IDs, source snapshots and input hashes, evaluation/final
coefficient bundles, display axes, a
[projected overlay](../../outputs/experiments/years_layer04_spiral_ribbon/job_1905025/spiral_ribbon_overlay.png),
and projected coefficients for interactive rendering. `load_model`, `predict`
and `project_radius` in the runner expose the model without pipeline changes.
`predict(..., derivative='year')` and `predict(..., derivative='radius')` expose
the ambient tangent vectors for future comparisons.
The [report](../../outputs/experiments/years_layer04_spiral_ribbon/job_1905025/report.md)
and [audit](../../outputs/experiments/years_layer04_spiral_ribbon/job_1905025/artifact_audit.json)
summarize the validation evidence.

Remove this page and its index entry, runner, audit, launcher and focused tests
to remove the experiment. Generated artifacts are separate.
