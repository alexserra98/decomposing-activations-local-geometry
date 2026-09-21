"""Exact noiseless sampling and serialization for the three-curve experiment."""

import json
from dataclasses import replace

import torch

from dalg.data import ToyManifoldConfig, make_toy_manifold_dataset, save_toy_manifold_shards
from dalg.data.manifold_dataset import _generator, _sample_segment, _sample_circle, _sample_helix


def test_noiseless_samples_match_embedded_curves_and_roundtrip(tmp_path):
    config = ToyManifoldConfig(
        ambient_dim=8,
        n_samples=90,
        calibration_size=256,
        manifolds_per_type=1,
        manifold_types=("segment", "circle", "helix"),
        noise_ratio=None,
    )
    dataset, metadata = make_toy_manifold_dataset(config)
    points, labels = dataset.tensors
    expected = []
    for i, sampler in enumerate((_sample_segment, _sample_circle, _sample_helix)):
        raw = sampler(30, _generator(config.seed, 400 + i), config)
        normalized = (raw - metadata["calibration_means"][i]) / metadata["calibration_scales"][i]
        expected.append(normalized @ metadata["embeddings"][i] + metadata["offsets"][i])
    permutation = torch.randperm(90, generator=_generator(config.seed, 1400))
    assert torch.equal(points, torch.cat(expected)[permutation].float())
    assert labels.bincount().tolist() == [30, 30, 30]
    assert torch.count_nonzero(metadata["noise_stds"]) == 0
    assert all(item["noise_std"] == 0 for item in metadata["manifolds"])

    noisy, noisy_metadata = make_toy_manifold_dataset(replace(config, noise_ratio=10.0))
    assert torch.equal(labels, noisy.tensors[1])
    assert torch.equal(metadata["offsets"], noisy_metadata["offsets"])
    assert not torch.equal(points, noisy.tensors[0])

    root = save_toy_manifold_shards(tmp_path / "shards", config)
    saved = json.loads((root / "config.json").read_text())
    assert saved["generator_config"]["noise_ratio"] is None
    assert torch.equal(torch.load(root / "layer00/shard_00000.pt", weights_only=True)[:, 0], points)
    saved_metadata = torch.load(root / "manifold_metadata.pt", weights_only=True)
    assert torch.equal(saved_metadata["noise_stds"], metadata["noise_stds"])
