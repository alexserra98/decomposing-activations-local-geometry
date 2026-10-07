"""Reserved and supplemental test populations share an isolated shard layout."""

import hashlib
import json
import math
from pathlib import Path
import runpy
from dataclasses import replace

import pytest
import torch

from dalg.data import ToyManifoldConfig, make_toy_manifold_dataset, save_toy_manifold_shards
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index


add_test_split = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/temporary/add_toy_manifold_test_split.py")
)["add_test_split"]


def _config(**overrides):
    return replace(ToyManifoldConfig(
        ambient_dim=4, n_samples=61, calibration_size=32,
        manifold_types=("circle", "sphere"), manifolds_per_type=2, seed=7,
    ), **overrides)


def _points(root):
    return torch.cat(list(ActivationBatchDataset(
        root, layer=0, batch_size=7, shuffle_shards=False, shuffle_within_shard=False,
    )))


def _metadata(root):
    return torch.load(root / "manifold_metadata.pt", weights_only=True)


def _hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_reserved_partition_membership_streams_and_provenance(tmp_path):
    config = _config()
    original, _ = make_toy_manifold_dataset(config)
    root = save_toy_manifold_shards(tmp_path / "split", config, shard_size=11)
    positions = []
    for directory, role in ((root, "development"), (root / "test", "test")):
        metadata = _metadata(directory)
        rows = metadata["original_row_indices"]
        positions.append(rows)
        assert torch.equal(_points(directory), original.tensors[0][rows])
        assert torch.equal(metadata["row_manifold_ids"], original.tensors[1][rows])
        index = load_meta_index(directory, layer=0)
        assert [row["global_row"] for row in index] == list(range(len(rows)))
        saved = json.loads((directory / "config.json").read_text())
        assert saved["num_rows"] == len(rows)
        assert saved["generator_config"]["n_samples"] == 61
        assert saved["partition"] == metadata["partition"]
        assert saved["partition"]["kind"] == "reserved"
        assert saved["partition"]["role"] == role
    assert torch.equal(torch.cat(positions).sort().values, torch.arange(61))
    assert len(positions[0]) == 48 and len(positions[1]) == 13
    counts = torch.bincount(original.tensors[1])
    assert torch.bincount(_metadata(root / "test")["row_manifold_ids"]).tolist() == [
        math.ceil(int(count) * 0.2) for count in counts
    ]
    provenance = _metadata(root / "test")["partition"]
    assert provenance["source_config_sha256"] == _hash(root / "config.json")
    assert provenance["source_metadata_sha256"] == _hash(root / "manifold_metadata.pt")


def test_partition_seed_and_noise_pairing(tmp_path):
    roots = {}
    for name, ratio, seed in (("clean", None, 42), ("noisy", 10., 42),
                              ("quieter", 100., 42), ("repeat", None, 42),
                              ("changed", None, 43)):
        roots[name] = save_toy_manifold_shards(
            tmp_path / name, _config(noise_ratio=ratio), test_split_seed=seed,
        )
    for suffix in (Path("."), Path("test")):
        clean = roots["clean"] / suffix
        assert torch.equal(_points(clean), _points(roots["repeat"] / suffix))
        assert not torch.equal(_metadata(clean)["original_row_indices"],
                               _metadata(roots["changed"] / suffix)["original_row_indices"])
        for name in ("noisy", "quieter"):
            assert torch.equal(_metadata(clean)["original_row_indices"],
                               _metadata(roots[name] / suffix)["original_row_indices"])
        torch.testing.assert_close(
            _points(roots["noisy"] / suffix) - _points(clean),
            10 * (_points(roots["quieter"] / suffix) - _points(clean)), rtol=0, atol=5e-6,
        )


@pytest.mark.parametrize("fraction", [-0.1, 1., float("nan"), float("inf"), True])
def test_invalid_test_fraction_does_not_write(tmp_path, fraction):
    root = tmp_path / "invalid"
    with pytest.raises((TypeError, ValueError), match="test_fraction"):
        save_toy_manifold_shards(root, _config(), test_fraction=fraction)
    assert not root.exists()


def test_insufficient_per_instance_population_does_not_write(tmp_path):
    with pytest.raises(ValueError, match="every manifold instance"):
        save_toy_manifold_shards(tmp_path / "small", _config(n_samples=5))
    assert not (tmp_path / "small").exists()


def test_supplemental_samples_preserve_sources_and_pair_noise(tmp_path):
    roots = {}
    for name, ratio in (("clean", None), ("noisy", 10.), ("quieter", 100.), ("repeat", None)):
        root = save_toy_manifold_shards(tmp_path / name, _config(noise_ratio=ratio), test_fraction=0)
        assert not (root / "test").exists()
        before = {path: _hash(path) for path in root.rglob("*") if path.is_file()}
        test = add_test_split(root, n_samples=61)
        roots[name] = test
        assert all(_hash(path) == digest for path, digest in before.items())
        assert torch.isfinite(_points(test)).all()
        assert _points(test).shape == (61, 4)
        assert not bool((_points(test)[:, None, :] == _points(root)[None, :, :]).all(-1).any())
        assert torch.bincount(_metadata(test)["row_manifold_ids"]).tolist() == [16, 15, 15, 15]
        metadata = _metadata(test)
        assert torch.equal(metadata["offsets"], _metadata(root)["offsets"])
        for key in ("embeddings", "calibration_means"):
            assert all(torch.equal(a, b) for a, b in zip(metadata[key], _metadata(root)[key]))
        provenance = metadata["partition"]
        assert provenance["kind"] == "supplemental" and provenance["role"] == "test"
        assert provenance["source_config_sha256"] == before[root / "config.json"]
        assert provenance["source_metadata_sha256"] == before[root / "manifold_metadata.pt"]
        with pytest.raises(FileExistsError, match="already exists"):
            add_test_split(root)
    assert torch.equal(_points(roots["clean"]), _points(roots["repeat"]))
    assert torch.equal(_metadata(roots["clean"])["row_manifold_ids"],
                       _metadata(roots["noisy"])["row_manifold_ids"])
    torch.testing.assert_close(_points(roots["noisy"]) - _points(roots["clean"]),
                               10 * (_points(roots["quieter"]) - _points(roots["clean"])),
                               rtol=0, atol=5e-6)


def test_supplemental_uses_saved_geometry_without_recalibration(tmp_path):
    source = save_toy_manifold_shards(tmp_path / "source", _config(), test_fraction=0)
    shifted = save_toy_manifold_shards(tmp_path / "shifted", _config(), test_fraction=0)
    metadata = _metadata(shifted)
    metadata["offsets"] = metadata["offsets"] + 8
    for manifold in metadata["manifolds"]:
        manifold["position"] = manifold["position"] + 8
    torch.save(metadata, shifted / "manifold_metadata.pt")
    expected = _points(add_test_split(source, n_samples=20)) + 8
    actual = _points(add_test_split(shifted, n_samples=20))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_supplemental_rejects_incompatible_metadata(tmp_path):
    root = save_toy_manifold_shards(tmp_path / "source", _config(), test_fraction=0)
    metadata = _metadata(root)
    metadata["config"]["seed"] += 1
    torch.save(metadata, root / "manifold_metadata.pt")
    with pytest.raises(ValueError, match="configuration does not match"):
        add_test_split(root, n_samples=20)
    assert not (root / "test").exists()
