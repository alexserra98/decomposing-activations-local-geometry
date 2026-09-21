"""Check saved noise-sweep shards and their paired geometry and noise draws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    records = json.loads((args.output_root / "sweep.json").read_text())["datasets"]
    roots = [Path(record["path"]) for record in records]
    configs = [json.loads((root / "config.json").read_text()) for root in roots]
    metadata = [
        torch.load(root / "manifold_metadata.pt", weights_only=True, map_location="cpu")
        for root in roots
    ]
    reference = metadata[0]
    ratios = [record["config"]["noise_ratio"] for record in records]
    assert ratios == [10, 100, 1_000, 10_000]

    for record, config, meta in zip(records, configs, metadata, strict=True):
        assert config["generator_config"] == record["config"]
        assert config["num_rows"] == record["config"]["n_samples"]
        assert meta["manifold_types"] == reference["manifold_types"]
        for key in ("row_manifold_ids", "offsets", "calibration_scales"):
            assert torch.equal(meta[key], reference[key]), key
        for key in ("embeddings", "calibration_means"):
            assert all(torch.equal(a, b) for a, b in zip(meta[key], reference[key], strict=True))
        assert torch.allclose(
            meta["noise_stds"] * record["config"]["noise_ratio"],
            reference["curvature_radii"],
        )
        counts = torch.bincount(meta["row_manifold_ids"])
        assert len(counts) == meta["num_manifolds"]
        assert int(counts.max() - counts.min()) == 0

    total = 0
    max_pairing_error = 0.0
    for shard_id in range(configs[0]["num_shards"]):
        shard_name = f"shard_{shard_id:05d}"
        shards = [
            torch.load(root / "layer00" / f"{shard_name}.pt", weights_only=True)
            for root in roots
        ]
        rows = shards[0].shape[0]
        for root, shard in zip(roots, shards, strict=True):
            assert shard.shape == (rows, 1, records[0]["config"]["ambient_dim"])
            assert shard.dtype == torch.float32 and torch.isfinite(shard).all()
            row_meta = json.loads((root / "meta" / f"{shard_name}.json").read_text())
            assert row_meta["row_indices"] == list(range(total, total + rows))
            for row, label in zip(row_meta["rows"], reference["row_manifold_ids"][total:total + rows], strict=True):
                manifold = reference["manifolds"][int(label)]
                assert row["manifold_id"] == int(label)
                assert row["subset"] == manifold["type_name"]
                assert row["intrinsic_dim"] == manifold["intrinsic_dim"]
        # Shared clean points and Gaussian draws imply affine dependence on 1/ratio.
        for ratio, observed in zip(ratios[1:-1], shards[1:-1], strict=True):
            fraction = (1 / ratio - 1 / ratios[-1]) / (1 / ratios[0] - 1 / ratios[-1])
            expected = shards[-1].double() + fraction * (shards[0].double() - shards[-1].double())
            error = float((observed.double() - expected).abs().max())
            max_pairing_error = max(max_pairing_error, error)
            assert torch.allclose(observed.double(), expected, atol=1e-6, rtol=1e-6)
        total += rows
    assert total == configs[0]["num_rows"]
    report = {
        "datasets": len(roots),
        "points_per_dataset": total,
        "manifold_types": reference["manifold_types"],
        "noise_ratios": ratios,
        "max_pairing_error": max_pairing_error,
    }
    (args.output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
