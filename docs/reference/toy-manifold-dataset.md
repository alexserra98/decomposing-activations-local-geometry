# Toy-Manifold Dataset Generator

> **Kind:** Data reference · **Status:** Current · **Use when:** Generating,
> storing, or changing deterministic synthetic local-geometry datasets.
> **Related:** [Toy-manifold tiling evaluation](../experiments/evaluation/toy-manifold-tiling.md)

The public API is exported from `dalg.data`:

```python
from dalg.data import (
    ToyManifoldConfig,
    make_toy_manifold_dataset,
    save_toy_manifold_shards,
)
```

The implementation is `src/dalg/data/manifold_dataset.py`. There is no dedicated
CLI or workflow skill; import these functions directly.

## Dataset construction

The generator defines the following manifold types:

| Type | Intrinsic dimension | Native embedding dimension |
| --- | ---: | ---: |
| `segment` | 1 | 1 |
| `circle` | 1 | 2 |
| `flat_disk` | 2 | 2 |
| `sphere` | 2 | 3 |
| `torus` | 2 | 3 |
| `mobius` | 2 | 3 |
| `swiss_roll` | 2 | 3 |
| `helix` | 1 | 3 |
| `helix_4d` | 1 | 4 |
| `hypersphere_10d` | 10 | 11 |
| `product_torus_12d` | 12 | 24 |
| `cylinder` | 2 | 3 |
| `swiss_roll_10d` | 10 | 11 |
| `cylinder_10d` | 10 | 11 |
| `swiss_roll_12d` | 2 | 12 |
| `helix_12d` | 1 | 12 |
| `product_torus_6d` | 6 | 12 |
| `hypersphere_6d` | 6 | 7 |
| `product_torus_4d` | 4 | 8 |

`manifolds_per_type` independently embedded instances are created for every
selected type. Each instance is normalized by a deterministic calibration
sample, embedded in `ambient_dim` through an independently sampled orthonormal
basis, and translated by its recorded ambient offset.

The hypersphere and product torus have fixed geometry. In raw local coordinates,
the hypersphere is

\[
S^{10}=\{x\in\mathbb{R}^{11}:\lVert x\rVert_2=1\},
\]

so it is the sphere surface, not the filled unit ball. The product torus is

\[
T^{12}=(S^1)^{12}
 = \{(\cos\theta_1,\sin\theta_1,\ldots,
       \cos\theta_{12},\sin\theta_{12})\}\subset\mathbb{R}^{24}.
\]

The hypersphere is sampled by normalizing isotropic Gaussian directions. The
product torus is sampled from twelve independent uniform angles. Both have raw
maximum absolute extrinsic curvature `1.0`. Their intrinsic dimensions and
unit radii are fixed rather than configurable.

`product_torus_6d` uses the same product construction with six independent
uniform angles: `T^6 = (S^1)^6` in twelve native coordinates. Its six unit
radii and raw maximum absolute extrinsic curvature `1.0` are fixed. Select it
with `manifold_types=("product_torus_6d",)` and `ambient_dim >= 12`. It uses the
same normalization, noise, metadata, and exact projection/tangent evaluation
as `product_torus_12d`. It is appended to the registry, preserving existing
type indices and random streams for explicit selections of earlier types.

`product_torus_4d` is `T^4 = (S^1)^4`: four independent uniform angles give
four unit circles in eight native coordinates. Select it with
`manifold_types=("product_torus_4d",)` and `ambient_dim >= 8`. Its raw maximum
absolute extrinsic curvature is `1.0`; normalization, noise, metadata, and
exact projection/tangent evaluation follow the other product tori. It is
appended to the registry and included in the default selection; explicit
selections of earlier types retain their indices and random streams.

`hypersphere_6d` is the unit sphere surface `S^6` in seven native coordinates,
sampled by normalizing seven-dimensional isotropic Gaussian directions. Its
intrinsic dimension is six and raw maximum absolute curvature is `1.0`. Select
it with `manifold_types=("hypersphere_6d",)` and `ambient_dim >= 7`. It uses
the same normalization, noise, metadata, and exact projection/tangent evaluation
as `hypersphere_10d`. It is appended to the registry and included in the default
selection; explicit selections of earlier types retain their random streams.

The main configuration fields are:

| Field | Default | Meaning |
| --- | ---: | --- |
| `ambient_dim` | `128` | Ambient dimension of every generated point; must be at least 3 and at least the largest selected native embedding dimension. The default selection therefore requires at least 24. |
| `n_samples` | `400_000` | Total number of points across all manifold instances. |
| `calibration_size` | `50_000` | Per-type sample count used to compute deterministic centering and RMS normalization. |
| `manifolds_per_type` | `8` | Number of independently embedded instances of each selected type. |
| `manifold_types` | all registered types | Unique tuple of types to include. |
| `offset_radius` | `4.0` | Radius of the sphere on which instance centers are placed; `0` centers every instance at the origin. |
| `noise_ratio` | `10_000.0` | Ratio between normalized curvature radius and per-coordinate ambient Gaussian-noise standard deviation; `None` disables observation noise exactly. |
| `seed` | `0` | Seed for calibration, embeddings, offsets, sampling, noise, and final row order. |

The remaining fields control the native parameter ranges and geometry of the
low-dimensional segment, torus, Mobius strip, Swiss roll, and helices. The 4D
helix is a one-dimensional curve rotating simultaneously in the orthogonal
`xy` and `zw` planes:

\[
(x,y,z,w) = (r_{xy}\cos(\omega_{xy}t),
             r_{xy}\sin(\omega_{xy}t),
             r_{zw}\cos(\omega_{zw}t),
             r_{zw}\sin(\omega_{zw}t)).
\]

Its parameter range, two radii, and two positive frequencies are configurable
through the `helix_4d_*` fields. Read the frozen `ToyManifoldConfig` dataclass
before changing those shapes.

The Swiss roll uses `(theta * cos(theta), height, theta * sin(theta))`.
Its default outer radius is `4.5 * pi`, and its default height range is
`[0, 2 * 4.5 * pi]`, giving an axial length equal to the outer diameter
(`9 * pi`, approximately `28.27`). Height and angle bounds are independent
configuration fields; changing the angle bounds does not automatically adjust
the height. RMS normalization divides all coordinates by one scalar, so it
preserves this aspect ratio. Existing saved datasets retain their recorded
bounds; reproducing the earlier elongated default requires explicitly setting
`swiss_height_max=10 * 4.5 * pi`.

The cylinder is the lateral surface `(cos(theta), height, sin(theta))`, with
independent uniform `theta` in `[0, 2 * pi)` and `height` in `[0, 5]`. It has
fixed unit radius and height five, with no end caps or filled interior. Its
raw maximum absolute principal curvature is `1.0`. RMS normalization preserves
its height-to-radius ratio of five. Select it with `manifold_types=("cylinder",)`
or include it as part of the default dataset.

The 10D Swiss roll uses `(theta*cos(theta), theta*sin(theta), h1, ..., h9)`.
Angle and all nine heights are sampled independently and uniformly from the
`swiss_theta_min/max` and `swiss_height_min/max` bounds, so each straight
dimension also defaults to one outer diameter in length. Its raw maximum
absolute curvature equals the ordinary Swiss roll's: `(t² + 2)/(t² + 1)^(3/2)`,
where `t` is the angle closest to zero in the configured interval.

The 10D cylinder is `S^9 × [-2.5, 2.5]`: normalize a ten-dimensional isotropic
Gaussian vector to sample its unit-radius sphere, then append an independent
uniform height. It is a lateral surface, with no end caps or filled interior,
and has raw maximum absolute curvature `1.0`.

Both 10D types have eleven native coordinates and require `ambient_dim >= 11`
when selected on their own. They use the same scalar RMS normalization,
curvature-scaled noise, metadata, and shard layout as the other types.
The default selection includes all registered types. Explicit selections of the
original twelve types retain their registry indices and random streams;
restoring these samplers does not promise byte-identical reproduction of
historical 10D datasets from an unavailable generator implementation.

### Swiss roll and helix with twelve native coordinates

`swiss_roll_12d` is a custom harmonic extension with **intrinsic dimension 2**:

\[
x(t,h)=\left(t,h,
  \left[\frac{t}{k}\cos(kt),\frac{t}{k}\sin(kt)\right]_{k=1}^{5}\right)
  \in\mathbb{R}^{12}.
\]

Angle and height are independent uniform samples from `swiss_theta_min/max`
and `swiss_height_min/max`. The five frequencies and their `1/k` amplitudes
are fixed. Coordinates `(2, 1, 3)` recover the ordinary Swiss roll. The extra
angle coordinate separates successive turns, so this extension changes the
geometry rather than only rotating a three-dimensional surface.

For its spiral curve, let `A = 1 + sum(1/k², k=1..5) + 5t²` be squared speed
and `B = 20 + 55t²` be squared acceleration. Raw curvature is
`sqrt(A*B - 25t²) / A^(3/2)`, with its maximum at the allowed angle closest
to zero. The independent height direction has zero curvature.

`helix_12d` is a closed curve with **intrinsic dimension 1**:

\[
x(t)=\left[\cos(kt),\sin(kt)\right]_{k=1}^{6},
\qquad t\sim U[0,2\pi).
\]

It extends the rotating-plane construction of `helix_4d` to six unit-radius
planes with fixed frequencies 1 through 6. It has no independent axial
coordinate. Its raw curvature is constant:
`sqrt(sum(k⁴, k=1..6)) / sum(k², k=1..6)`. The existing `helix_*` and
`helix_4d_*` configuration fields do not alter this fixed curve.

For these two names, `12d` means **native embedding dimension**. In
`product_torus_12d`, it means **intrinsic dimension**: twelve independent
circles use 24 native coordinates. All three can share a 128D dataset:

```python
config = ToyManifoldConfig(
    ambient_dim=128,
    manifold_types=("swiss_roll_12d", "helix_12d", "product_torus_12d"),
)
```

The new Swiss roll and helix require `ambient_dim >= 12` when selected on
their own. They use the existing scalar RMS normalization, curvature-scaled
noise, random orthonormal embedding, offsets, and shard format. Both are
included in the default selection. Explicit selections of earlier types
preserve their registry indices and random streams; the expanded default
selection changes sample allocation across types.

Use the same seed and configuration except for `offset_radius` to generate a
paired centered and separated condition. The point geometry, embeddings,
sampling, noise, and row order remain fixed; only the recorded per-instance
offsets change.

## Return contract

`make_toy_manifold_dataset(config)` returns one balanced `TensorDataset` and a
metadata dictionary:

- the first tensor contains `float32` points with shape
  `(n_samples, ambient_dim)`;
- the second contains `int64` manifold-instance IDs with shape `(n_samples,)`;
- counts differ by at most one when `n_samples` is not divisible by the number
  of manifold instances; and
- rows are deterministically shuffled.

The metadata records the resolved config, type-name mappings, intrinsic and
embedding dimensions, calibration statistics, curvature and noise scales,
orthonormal embeddings, offset directions and offsets, and one record per
manifold instance.

The in-memory generator returns all configured points in a single dataset.
The shard writer reserves a test partition before training creates its own
deterministic train/validation split over the remaining root rows.

## Observation noise

Set `noise_ratio=None` to generate noiseless points. Gaussian noise is skipped,
all recorded `noise_stds` are exactly zero, and the saved JSON configuration
records `noise_ratio: null`. Geometry, sampling, and row order remain paired
with otherwise identical noisy configurations.

Noise is isotropic in the ambient space and is constant within each manifold
type. Its standard deviation is

```text
noise_std = normalized_curvature_radius / noise_ratio
```

The curvature definition is the maximum absolute extrinsic principal
curvature. Flat manifolds use unit normalized RMS radius as the finite scale for
adding nonzero noise.

## Activation-compatible shards

Use `save_toy_manifold_shards` when the training pipeline needs the dataset:

```python
from dalg.data import ToyManifoldConfig, save_toy_manifold_shards

config = ToyManifoldConfig(
    ambient_dim=128,
    n_samples=400_000,
    manifolds_per_type=8,
    offset_radius=4.0,
    seed=0,
)
save_toy_manifold_shards(
    "dalg-cache/assets/toy_manifolds_D128_shards",
    config,
    shard_size=50_000,
    layer=0,
    test_fraction=0.2,
    test_split_seed=42,
)
```

The destination must be absent or empty. Each point becomes a one-position
activation window, so layer tensors have shape `(rows, 1, ambient_dim)` and the
saved configuration sets `window: 1` and `drop_prefix: 0`:

```text
<root>/
  config.json
  manifold_metadata.pt
  layer00/
    shard_00000.pt
    ...
  meta/
    shard_00000.json
    ...
  test/
    config.json
    manifold_metadata.pt
    layer00/shard_00000.pt
    ...
    meta/shard_00000.json
    ...
```

Row metadata stores the manifold instance, type, and intrinsic dimension. The
larger tensors needed for exact geometry are stored in `manifold_metadata.pt`.
Token shards are intentionally absent because synthetic points have no textual
token identity.

Store large generated datasets under `dalg-cache/assets/`, not in source,
documentation, or script directories.

### Reserved test partition

`save_toy_manifold_shards` defaults to `test_fraction=0.2` and
`test_split_seed=42`. It randomly reserves `ceil(test_fraction * count)` rows
from **each manifold instance**, with reproducible membership across paired
noise and offset conditions. Every instance must retain at least one row in
both partitions; insufficient populations are rejected before writing. Use
`test_fraction=0` to save all points at the root without a test directory.

The root contains only development points (training plus validation), and
`test/` is a separate activation-compatible dataset. Existing loaders do not
recurse into it. Training still applies `val_frac` to the root population:
300,000 generated points with the default test fraction and `val_frac=0.1`
give 216,000 training, 24,000 validation, and 60,000 test points when per-instance
counts divide evenly.

Both directories use contiguous local row IDs and aligned `row_manifold_ids`.
Their `config.json` records the directory's actual `num_rows`, while
`generator_config.n_samples` retains the original generation budget.
`manifold_metadata.pt` stores `original_row_indices`, mapping each local row
back to the original in-memory population. The root and test indices are
disjoint and together cover that population; their canonical order is
`generated_subset`.

Both configuration and metadata contain a `partition` record with version 1,
`kind: reserved`, `role: development` or `test`, the fraction, split seed,
manifold-instance stratification, and source row count. The test record also
contains `source_dir: ..`, `source_config_sha256`, and `source_metadata_sha256`,
fingerprinting its parent's configuration and saved geometry/row metadata.
The same geometry is stored in both populations.

### Supplemental test points for existing runs

Datasets and runs created before test partitions were introduced have no
independent test population. The temporary migration script adds fresh points
from their saved manifolds without changing their training/validation data:

```bash
PYTHONPATH=src .venv/bin/python scripts/temporary/add_toy_manifold_test_split.py \
  dalg-cache/assets/<existing-dataset> --n-samples 30000
```

It accepts a single-layer toy dataset with saved configuration and
`manifold_metadata.pt`. Existing test destinations or declared partitions are
rejected. The default is 30,000 **additional** points, balanced across manifold
instances; this count is independent of the 20% reservation for new datasets.
Original shard files, metadata, configuration, and model artifacts remain
unchanged.

The script reuses saved calibration, embeddings, offsets, and noise scales.
It samples with the original seed and a separate stream block starting at
`10000 + 3 * num_manifolds`, beyond the original sampling streams. Using the
same sample count across paired conditions preserves sample and noise pairing.
It neither recalibrates the manifolds nor regenerates the original points.

The resulting `test/` uses the same layout and source fingerprints as a
reserved test directory. Its `partition` record has `kind: supplemental`,
`role: test`, `n_samples`, `sampling_seed`, `sampling_stream`, manifold-instance
balancing, and the original source row count. There are no
`original_row_indices`, since these are new samples. `generator_config` retains
the source generator configuration; `num_rows` describes the supplemental
population.

These artifacts provide an independent population for
[held-out coverage](../evaluation/heldout-distribution-coverage.md).
The pipeline evaluator consumes `test/` automatically when
`evaluation.heldout_distribution_coverage` is enabled (the default). Missing
test data is an error with instructions to run the migration script above.
Coverage accepts at most 100,000 test points; see the
[pipeline coverage contract](../evaluation/heldout-distribution-coverage.md#pipeline-integration).

## Downstream evaluation

The model-agnostic tiling evaluator supports vanilla MFA, ARD, HDDC, and KMeans+PCA trained
on these shards. It associates Gaussian means with exact planted manifolds and
reports rank and tangent-subspace metrics; assignments are used for clustering
and component-liveness diagnostics. Read the
[Toy-Manifold Tiling Evaluation](../experiments/evaluation/toy-manifold-tiling.md) for the
metric and artifact contract.

### Non-unique high-dimensional projections

Projection degeneracies concern MFA component means during evaluation, not
noiseless samples emitted by these generators. Here, the hypersphere's
"origin" and a product-torus "zero pair" mean raw local coordinates obtained
*after* reversing the saved ambient offset, orthonormal embedding, and
calibration. They do not generally coincide with the ambient zero vector.

- At the hypersphere origin, every point of the sphere is equally near.
- At the `helix_12d` origin, every curve point is equally near (raw squared
  distance six). Projection uses angle zero and marks its tangent non-unique.
- If any two-coordinate product-torus pair is zero, every angle on that circle
  factor is equally near.

The geometry evaluator returns a deterministic representative point so that
the exact distance stays finite, but marks these cases as non-unique because
the projected point and its tangent are not identified. See
[Exact proximity association](../experiments/evaluation/toy-manifold-tiling.md#exact-proximity-association)
for how this differs from a tie between separate planted manifolds and how it
affects rank and tangent metrics.

The generator and test-partition contracts are covered by
`tests/test_manifold_dataset.py` and `tests/test_toy_manifold_test_split.py`; exact
geometry and pipeline consumption are covered by
`tests/test_toy_manifold_geometry.py` and `tests/test_training_pipeline.py`.
