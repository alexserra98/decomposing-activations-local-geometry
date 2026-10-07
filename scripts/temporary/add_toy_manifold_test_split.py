"""Add an independent test set to a toy dataset created before test splits.

Test splits were introduced after many existing datasets and model runs had
already been created. Those runs have no independent test set. This migration
script generates additional test samples from their saved manifold geometry,
preserving the original training/validation data so existing trained models
can be evaluated without retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dalg.data.manifold_dataset import (
    EMBEDDING_DIMS,
    MANIFOLD_NAMES,
    ToyManifoldConfig,
    _make_dataset,
    _SAMPLERS,
    _save_toy_manifold_partition,
    _validate_config,
)


def add_test_split(dataset_dir: str | Path, *, n_samples: int = 30_000) -> Path:
    """Save fresh balanced samples under ``dataset_dir/test`` without refitting geometry."""
    root = Path(dataset_dir).resolve()
    destination = root / "test"
    if destination.exists():
        raise FileExistsError(f"test destination already exists: {destination}")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    config_path = root / "config.json"
    saved = json.loads(config_path.read_text())
    if (saved.get("source_kind") != "toy_manifolds"
            or saved.get("window") != 1 or saved.get("drop_prefix") != 0
            or len(saved.get("layers", [])) != 1):
        raise ValueError("source must be a single-layer toy-manifold shard dataset")
    if "partition" in saved:
        raise ValueError("source already declares a partition; refusing to replace its test population")
    values = dict(saved["generator_config"])
    values["manifold_types"] = tuple(values["manifold_types"])
    config = ToyManifoldConfig(**values)
    _validate_config(config)
    metadata_path = root / saved["manifold_metadata"]
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    if json.dumps(metadata["config"], sort_keys=True) != json.dumps(saved["generator_config"], sort_keys=True):
        raise ValueError("saved geometry configuration does not match source config.json")
    num_manifolds = len(config.manifold_types) * config.manifolds_per_type
    type_ids = torch.arange(len(config.manifold_types)).repeat_interleave(config.manifolds_per_type)
    if (metadata["num_manifolds"] != num_manifolds
            or tuple(metadata["manifold_types"]) != config.manifold_types
            or not torch.equal(metadata["manifold_type_ids"], type_ids)
            or saved["d_model"] != config.ambient_dim
            or saved["num_rows"] != config.n_samples
            or metadata["row_manifold_ids"].shape != (saved["num_rows"],)):
        raise ValueError("saved geometry and source population are incompatible")
    if n_samples < num_manifolds:
        raise ValueError("n_samples must include at least one point per manifold instance")
    selected_type_ids = [MANIFOLD_NAMES.index(name) for name in config.manifold_types]
    for type_id, registry_id in enumerate(selected_type_ids):
        if metadata["calibration_means"][type_id].shape != (EMBEDDING_DIMS[registry_id],):
            raise ValueError("saved calibration dimensions do not match the manifold sampler")
    if (len(metadata["embeddings"]) != num_manifolds
            or metadata["offsets"].shape != (num_manifolds, config.ambient_dim)):
        raise ValueError("saved embedding/offset dimensions are incompatible")
    for manifold_id, type_id in enumerate(type_ids.tolist()):
        if metadata["embeddings"][manifold_id].shape != (
            EMBEDDING_DIMS[selected_type_ids[type_id]], config.ambient_dim,
        ):
            raise ValueError("saved embedding dimensions do not match the manifold sampler")
    geometry_tensors = (
        *metadata["calibration_means"], *metadata["embeddings"], metadata["offsets"],
        metadata["calibration_scales"], metadata["noise_stds"],
    )
    if (any(not bool(torch.isfinite(tensor).all()) for tensor in geometry_tensors)
            or bool((metadata["calibration_scales"] <= 0).any())
            or bool((metadata["noise_stds"] < 0).any())):
        raise ValueError("saved geometry contains invalid calibration or noise values")

    # Original sampling uses streams 400+i, 1400, and 2400+i. Keep this
    # supplemental block beyond them without changing the geometry seed.
    stream = 10_000 + 3 * num_manifolds
    dataset = _make_dataset(
        n_samples,
        config=config,
        stream=stream,
        means=metadata["calibration_means"],
        scales=metadata["calibration_scales"],
        noise_stds=metadata["noise_stds"],
        samplers=tuple(_SAMPLERS[index] for index in selected_type_ids),
        embeddings=metadata["embeddings"],
        offsets=metadata["offsets"],
        manifold_type_ids=metadata["manifold_type_ids"],
    )
    provenance = {
        "version": 1,
        "kind": "supplemental",
        "role": "test",
        "source_dir": "..",
        "source_num_rows": saved["num_rows"],
        "n_samples": n_samples,
        "sampling_seed": config.seed,
        "sampling_stream": stream,
        "balancing": "manifold_instance",
    }
    for name, path in (("config", config_path), ("metadata", metadata_path)):
        with path.open("rb") as handle:
            provenance[f"source_{name}_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    return _save_toy_manifold_partition(
        destination, dataset, {**metadata, "partition": provenance},
        shard_size=saved["shard_size"], layer=saved["layers"][0],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--n-samples", type=int, default=30_000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    destination = add_test_split(args.dataset_dir, n_samples=args.n_samples)
    print(f"Saved {args.n_samples:,} supplemental test points: {destination}")


if __name__ == "__main__":
    main()
