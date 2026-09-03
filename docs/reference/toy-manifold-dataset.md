# Toy-Manifold Dataset Generator

> **Kind:** Data reference · **Status:** Current · **Use when:** Generating,
> storing, or changing deterministic synthetic local-geometry datasets.
> **Related:** [Toy-manifold tiling evaluation](../evaluation/toy-manifold-tiling.md)

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

The generator defines ten manifold types:

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
| `hypersphere_10d` | 10 | 11 |
| `product_torus_12d` | 12 | 24 |

`manifolds_per_type` independently embedded instances are created for every
selected type. Each instance is normalized by a deterministic calibration
sample, embedded in `ambient_dim` through an independently sampled orthonormal
basis, and translated by its recorded ambient offset.

The two high-dimensional types have fixed geometry. In raw local coordinates,
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

The main configuration fields are:

| Field | Default | Meaning |
| --- | ---: | --- |
| `ambient_dim` | `128` | Ambient dimension of every generated point; must be at least 3 and at least the largest selected native embedding dimension. The default ten-type selection therefore requires at least 24. |
| `n_samples` | `400_000` | Total number of points across all manifold instances. |
| `calibration_size` | `50_000` | Per-type sample count used to compute deterministic centering and RMS normalization. |
| `manifolds_per_type` | `8` | Number of independently embedded instances of each selected type. |
| `manifold_types` | all ten types | Unique tuple of types to include. |
| `offset_radius` | `4.0` | Radius of the sphere on which instance centers are placed; `0` centers every instance at the origin. |
| `noise_ratio` | `10_000.0` | Ratio between normalized curvature radius and per-coordinate ambient Gaussian-noise standard deviation. |
| `seed` | `0` | Seed for calibration, embeddings, offsets, sampling, noise, and final row order. |

The remaining fields control the native parameter ranges and geometry of the
low-dimensional segment, torus, Mobius strip, Swiss roll, and helix. Read the
frozen `ToyManifoldConfig` dataclass before changing those shapes.

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

The generator returns a single dataset. Activation-shard training creates its
own deterministic train/validation split; do not introduce a second split in
the generator.

## Observation noise

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
```

Row metadata stores the manifold instance, type, and intrinsic dimension. The
larger tensors needed for exact geometry are stored in `manifold_metadata.pt`.
Token shards are intentionally absent because synthetic points have no textual
token identity.

Store large generated datasets under `dalg-cache/assets/`, not in source,
documentation, or script directories.

## Downstream evaluation

The model-agnostic tiling evaluator supports vanilla MFA, ARD, and HDDC trained
on these shards. It associates Gaussian means with exact planted manifolds and
reports rank and tangent-subspace metrics; assignments are used for clustering
and component-liveness diagnostics. Read the
[Toy-Manifold Tiling Evaluation](../evaluation/toy-manifold-tiling.md) for the
metric and artifact contract.

### Non-unique high-dimensional projections

Projection degeneracies concern MFA component means during evaluation, not
noiseless samples emitted by these generators. Here, the hypersphere's
"origin" and a product-torus "zero pair" mean raw local coordinates obtained
*after* reversing the saved ambient offset, orthonormal embedding, and
calibration. They do not generally coincide with the ambient zero vector.

- At the hypersphere origin, every point of the sphere is equally near.
- If any two-coordinate product-torus pair is zero, every angle on that circle
  factor is equally near.

The geometry evaluator returns a deterministic representative point so that
the exact distance stays finite, but marks either case as non-unique because
the projected point and its tangent are not identified. See
[Exact proximity association](../evaluation/toy-manifold-tiling.md#exact-proximity-association)
for how this differs from a tie between separate planted manifolds and how it
affects rank and tangent metrics.

The generator contract is covered by `tests/test_manifold_dataset.py`; exact
geometry and pipeline consumption are covered by
`tests/test_toy_manifold_geometry.py` and `tests/test_training_pipeline.py`.
