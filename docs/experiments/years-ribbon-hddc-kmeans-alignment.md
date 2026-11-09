# Years ribbon: HDDC versus KMeans tangent geometry

> **Kind:** Experiment · **Status:** Evaluated development pilot · **Use when:** Comparing
> saved layer-4 HDDC and KMeans PCs with the analytic spiral-ribbon tangent.

## Question and reference

Do the first two Gaussian covariance components of HDDC align more closely
with the estimated years manifold than the first two cluster PCs of KMeans?
This experiment fixes the [continuous spiral ribbon](years-spiral-ribbon.md)
before comparing either algorithm. Its coefficient vectors and tangents live
in all **4096 activation dimensions**; no display PCA enters the scores.

The analytic tangent is exact for the fitted surface, **not known ground truth
for the activation manifold**. The ribbon's internal held-out ambient R² was
approximately 0.27–0.31, despite a much closer fit in the three display axes.
Its radial coordinate is a modeling assumption. Conclusions are conditional
on this reference, and no external prompt holdout is read.

## Projection and metrics

Centres generally lie off the surface. For each centre, minimize ambient
squared distance over year `[1000,2000]` and the fixed fitted radial domain.
At any year the optimal radius is available analytically and clipped to that
domain. A 0.25-year grid locates candidate local minima; each is refined
continuously, including both endpoints. A 0.125-year grid independently checks
five centres per distinct centre array. This numerical search does not certify
the global optimum between every grid point. Equal-distance distinct minima
make the tangent undefined.

At the projected point, orthonormalize the analytic Jacobian columns
`S_y = c'(y)+(r-1)d'(y)` and `S_r = d(y)`. Unique boundary projections remain
included, as in the [canonical metric contract](evaluation/toy-manifold-tiling.md#tangent-subspace-geometry).
Interior-only averages provide a separate sensitivity check.

The implementation reuses the shared subspace functions and saved effective
ranks:

- **Alignment:** compare the two-dimensional tangent with exactly PC1–PC2.
- **Containment:** choose exactly two PCs among the first `q` that maximize
  squared overlap. This is not projection onto the complete learned span.
- **Partial containment and adjusted alignment:** follow the same rank-one
  and rank-zero rules as the canonical evaluator.

Overlap is `||TᵀP||²_F / 2`, with a companion minimum principal-angle cosine.
Alignment checks the PC2/PC3 relative boundary eigengap; containment separately
checks PCq/PC(q+1), both at `1e-6`. Undefined scores are counted and omitted
from their means. Assignment-live components define the population, excluding
invalid KMeans PCA. Means are unweighted over valid components.

HDDC uses thin SVD of its loadings: isotropic noise leaves the covariance
eigenvectors unchanged and adds its variance to squared singular values.
KMeans uses its saved ordered cluster PCs and full eigenvalue spectrum.
Both use the covariance geometry, rather than arbitrary loading columns.

## Comparison design

There are eight matched settings: `K ∈ {50,100}` and rank thresholds
`{0.0001,0.001,0.01,0.1}`. Each method uses its existing saved checkpoint,
seed 42 and split seed 42 with validation fraction 0.1. Saved validation-row
contents must match across all sixteen runs; the JSON writers format them
differently. No upstream model is retrained.
Centres differ between methods; component indices are not paired. Differences
are between run-level means at matched settings.

HDDC capacity is 16; KMeans can retain more PCs. In addition to the canonical
containment score, a supplementary score searches only the first `min(q,16)`
PCs. Interpret containment together with effective ranks and valid counts.
This cap does not force identical component ranks or populations.

## Results: job 1905982

The comparison evaluated **1,200 components across 16 checkpoints**. HDDC has
higher leading-plane alignment overlap in **8/8 matched settings**; KMeans has
higher containment overlap in **8/8**, including the cap-16 sensitivity.
For example, at threshold `0.001`:

| K | Alignment HDDC / KMeans | Containment HDDC / KMeans | Mean rank HDDC / KMeans |
| ---: | ---: | ---: | ---: |
| 50 | 0.0542 / 0.0384 | 0.1292 / 0.1374 | 15.58 / 27.40 |
| 100 | 0.0566 / 0.0532 | 0.1090 / 0.1102 | 14.25 / 17.52 |

HDDC's leading plane is closer to this reference by average squared overlap,
but neither method closely recovers the complete two-dimensional tangent.
The containment companion **worst-direction cosine favors HDDC in 8/8**:
at threshold `0.001`, it is `0.1726 / 0.0709` for K=50 and
`0.1254 / 0.0325` for K=100. Better least-covered direction and better average
squared coverage need not favor the same method.

The leading-plane advantage persists using only interior projections.
Containment on interior projections favors KMeans in seven settings and HDDC
at K=50, threshold `0.1`. Boundary incidence differs substantially: HDDC has
16–20 of 50 or 21 of 100 centres at a boundary, versus KMeans's 5 of 50 or
3 of 100. No projections were flagged ambiguous. Valid alignment/containment
counts at threshold `0.1` are 48/48 for K=50 and 99/97 for K=100 (HDDC/KMeans);
all other settings use every component. These are descriptive, single-seed
results; threshold variants are not independent replications.

All four tests passed. The aggregate audit reproduced **80 metric summaries**
and all matched deltas from the component CSV, verified identical saved split
contents and reference/source hashes, and recorded maximum finer-grid distance
discrepancy `1.13e-11` across 45 checked centres.

The [complete report](../../outputs/experiments/years_layer04_ribbon_comparison/job_1905982/report.md),
[component projections and scores](../../outputs/experiments/years_layer04_ribbon_comparison/job_1905982/components.csv),
[full results](../../outputs/experiments/years_layer04_ribbon_comparison/job_1905982/results.json),
[figure](../../outputs/experiments/years_layer04_ribbon_comparison/job_1905982/comparison.png)
and [audit](../../outputs/experiments/years_layer04_ribbon_comparison/job_1905982/aggregate_audit.json)
preserve the detailed evidence.

## Reproduce and inspect

```bash
sbatch scripts/slurm/temporary/sbatch_years_ribbon_comparison.sh
```

The [launcher](../../scripts/slurm/temporary/sbatch_years_ribbon_comparison.sh)
runs four focused [tests](../../tests/test_years_ribbon_comparison.py), the
[comparison](../../scripts/temporary/compare_years_ribbon_tangents.py) and an
[aggregate audit/report](../../scripts/temporary/report_years_ribbon_comparison.py).
Tests cover synthetic full-dimensional projection, ambiguous footpoints,
rank/eigengap semantics, and agreement of thin SVD with the existing full
covariance solver. The audit independently rebuilds aggregate scores/counts
and matched differences from component rows and checks split/reference hashes.

Outputs go under `outputs/experiments/years_layer04_ribbon_comparison/job_ID/`:
component projections and scores, full summaries with validity/interior
counts, matched differences, comparison figure, source snapshots, input hashes,
and a human-readable report. The reference is the fixed ribbon from job
`1905025`. Remove these experimental scripts, tests, launcher, page and index
entry to remove this analysis; training code is unchanged.
