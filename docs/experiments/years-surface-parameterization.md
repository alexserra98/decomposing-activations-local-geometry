# Layer-4 years surface parameterization

> **Kind:** Experiment · **Status:** Pilot evaluated, 2026-11-04 · **Use when:**
> Fitting a smooth two-dimensional reference sheet and comparing its tangent
> planes with saved layer-4 HDDC covariance PCs.

Related: [year curves](years-spiral-parameterization.md),
[unsupervised evaluation](../evaluation/unsupervised.md),
[tangent alignment](evaluation/toy-manifold-tiling.md#tangent-subspace-geometry),
and [HDDC checkpoints](../models/mfa-hddc.md).

For an explicit surface winding around a spiral rather than this unconstrained
sheet, use the subsequent [spiral-ribbon experiment](years-spiral-ribbon.md).

## Surface and coordinates

The temporary [runner](../../scripts/temporary/fit_years_surface.py) fits
`S(u,v) = origin + sum_ab B_a(u) B_b(v) C_ab`, with each coefficient `C_ab`
in the original 4096 activation coordinates. The tensor-product B-splines are
cubic and clamped on `[0,1]^2`. Both coordinates are learned from activations;
neither year nor the arbitrary ordering of prompt-template IDs is a coordinate.
The surface is independent of the saved Gaussian axes used for comparison.

Alternating optimization regresses ambient coefficients and projects training
activations onto the sheet with bounded Gauss–Newton updates. A grid-scaled
second-difference penalty smooths the control net, with a small first-difference
term to stabilize unsupported regions. PCA initializes coordinates only. The
regression objective, projections, Jacobians and alignment use all 4096
dimensions. Projection arithmetic uses an exact QR basis of the coefficient
span; this is algebraic compression without dropping ambient residual error.

Candidate control grids are `6x6` and `10x10`, with smoothing weights `1e-4`
and `1e-3` and initial coordinates from activation PC1–PC2 or PC1–PC3. Each fit
uses at most 18 alternating iterations. Closest-point searches use separated
coarse-grid starts followed by bounded refinement; they are approximate and do
not certify a global minimum. The result can depend on initialization.

This is an estimated reference surface, not evidence that the activation set
has intrinsic dimension two or any particular spiral topology. A single square
chart may bridge branches, miss folds, or leave gaps. Tangents between observed
points depend on the smoothness model. The saved surface should be inspected
together with its residual error, support mask and projection plots.

## Development checks

Only the standalone **16,016 development rows**, layer 4, are read. The existing
external `test/` extraction is not read. Year and template grouping reuses the
[curve experiment's splits](years-spiral-parameterization.md#selection-and-isolation):
600 fit years, 200 validation years, 201 evaluation years and four withheld
templates, all inside development.

Validation reconstruction error selects the control grid, smoothing and
initialization. Selection is frozen before refitting on fit+validation rows and
scoring the three evaluation groups. A two-PC affine plane fitted on the same
refit rows is the reference reconstruction baseline. A final descriptive surface
is then refitted on all development rows for visualization and Gaussian checks.

Each evaluation activation supplies its two fitted coordinates through
projection. Consequently these are **unsupervised reconstruction scores**, not
predictions from unseen year labels. They cannot be compared directly with the
earlier supervised year-to-activation errors. No independent evaluation score is
claimed for the final all-development refit. Existing Gaussian training splits
also differ from the internal surface splits; alignment is a descriptive check
of two estimates, not a held-out ground-truth recovery metric.

## Tangent-plane comparison

For every component in the eight existing current-development layer-4 HDDC runs
(`K=50/100`, thresholds `0.0001/0.001/0.01/0.1`):

1. Approximately minimize `||mu_k - S(u,v)||` in the ambient space.
2. Orthonormalize the two analytic partial derivatives at that point.
3. Obtain exactly the leading two covariance PCs. For saved isotropic HDDC
   covariance `W W.T + b I`, a thin SVD of `W` gives these PCs exactly; a focused
   test compares this with the existing full-covariance eigensolver.
4. Use the repository's `subspace_alignment` and covariance-boundary
   identifiability function. Report both principal angles, mean squared cosine
   overlap and worst-direction cosine. Individual vector signs and rotations
   within either plane do not affect the score.

Inactive components, ranks below two, an unidentifiable PC2/PC3 eigenvalue
boundary, nearly rank-deficient surface Jacobians (singular-value ratio at most
`1e-4`), and projections within `0.005` of the chart boundary are excluded from
summary means. Unsupported projections are also excluded: their coordinates
must lie in the training-coordinate convex hull and within 1.5 times the nearest
training point's 10-neighbour radius. Per-component records retain the reason,
surface distance, support ratio, rank, mixture weight and Jacobian ratio.

The support test is a heuristic in fitted coordinates, not a calibrated
activation-density threshold. Display meshes combine sampled training coordinates
with a regular chart grid. Faces passing the support test at their centres and
edge midpoints are coloured; the rest of the fitted sheet is shown in faint gray.
Summary means report their valid-component counts; excluded components are not
silently assigned zero alignment.

## Evaluated pilot

Fit job **1904693** selected a `10x10` control net, smoothing `1e-4`, initialized
from activation PC1–PC3. Comparison/render and audit job **1904723** completed
the artifacts below. All six focused numerical tests passed, including an
end-to-end synthetic curved-sheet fit and the covariance-PC equivalence check.

| Evaluation group | Surface ambient MSE | Surface R² | Two-PC plane MSE | Two-PC plane R² | Surface edge fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| New years, seen templates | 0.00131198 | 0.2786 | 0.00155745 | 0.1437 | 27.3% |
| Seen years, new templates | 0.00117334 | 0.3346 | 0.00147381 | 0.1642 | 38.0% |
| New years, new templates | 0.00127154 | 0.2891 | 0.00152006 | 0.1502 | 28.6% |

The selected sheet reduces new-year reconstruction MSE by **15.8%** relative
to the two-PC plane. Most ambient variation remains outside this estimated
two-dimensional sheet. Both evaluation and final refits reached the 18-iteration
budget while still improving (last objective reductions about 0.15% and 0.12%);
they are finite-budget fits, not certified converged principal surfaces.

| Gaussian K | Rank threshold | Valid / total | Mean overlap | Median larger principal angle |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 0.0001 | 17 / 50 | 0.1549 | 85.7° |
| 50 | 0.001 | 17 / 50 | 0.1549 | 85.7° |
| 50 | 0.01 | 20 / 50 | 0.1547 | 84.0° |
| 50 | 0.1 | 21 / 50 | 0.1358 | 84.6° |
| 100 | 0.0001 | 52 / 100 | 0.0857 | 86.0° |
| 100 | 0.001 | 52 / 100 | 0.0858 | 86.1° |
| 100 | 0.01 | 53 / 100 | 0.0869 | 86.8° |
| 100 | 0.1 | 52 / 100 | 0.0861 | 86.4° |

Overlap is in `[0,1]`, with 1 meaning identical planes. These Gaussian PC1–PC2
planes agree poorly with this particular fitted surface. The reference fit's
large residuals, boundary projections, incomplete support, initialization
dependence and finite optimization budget prevent interpreting this as evidence
that the Gaussian local geometry is wrong or that a true two-dimensional
manifold has been recovered.

The independent artifact audit reproduced all three reconstruction scores
exactly, checked all 600 Gaussian rows and their summaries, and verified hashes
and split/source row IDs. An augmented QR fit on 24 ambient response coordinates
agrees with the saved control coefficients to `4.24e-16`; analytic partials agree
with finite differences to `7.90e-11`.

The [report](../../outputs/experiments/years_layer04_surface/job_1904693/report.md),
[results](../../outputs/experiments/years_layer04_surface/job_1904693/results.json),
[per-component CSV](../../outputs/experiments/years_layer04_surface/job_1904693/gaussian_alignment.csv)
and [audit](../../outputs/experiments/years_layer04_surface/job_1904693/artifact_audit.json)
retain the numerical evidence. The
[overlay](../../outputs/experiments/years_layer04_surface/job_1904693/surface_overlay.png)
uses the activation cloud's PCA axes; these first three display PCs retain only
18.8% of activation variance.

## Run and artifacts

```bash
sbatch scripts/slurm/temporary/sbatch_years_surface.sh
```

The [launcher](../../scripts/slurm/temporary/sbatch_years_surface.sh) runs focused
[numerical tests](../../tests/test_years_surface.py) and the fit on a Slurm CPU
node, then writes a new `outputs/experiments/years_layer04_surface/job_ID/`.
No Gaussian is retrained and no existing activation or model files are modified.
To rebuild comparisons and figures from an existing completed surface fit, set
`DALG_SURFACE_OUTPUT` with `sbatch --export=ALL,DALG_SURFACE_OUTPUT=<directory>`
and pass `--resume-comparison` after the launcher path. The runner verifies the
activation fingerprints, source row IDs and splits before reusing the bundles.

Artifacts include source snapshots, shard fingerprints, source row IDs and
split memberships, candidate and frozen-selection files, evaluation and final
surface bundles, original-dimensional sampled surface geometry, per-component
alignment CSV, summary JSON, 2D/3D overlays, and projected visualization data.
Saved bundles contain the coefficient vectors and analytic basis specification;
the runner's `load_surface`, `predict`, `project` and `tangent` functions expose
the fitted surface and tangent planes without an installed API or pipeline change.

The mathematical context is principal-surface fitting. For a different approach,
[Ozertem & Erdogmus (2011)](https://www.jmlr.org/papers/v12/ozertem11a.html)
construct principal curves and surfaces from density derivatives. This pilot
uses an explicit spline sheet and does not fit a density-derived surface to the
same Gaussians whose axes it evaluates.

Remove the runner, [audit](../../scripts/temporary/audit_years_surface.py), launcher,
focused tests, this page and its index entry to
remove the experiment. Generated artifacts remain separate.
