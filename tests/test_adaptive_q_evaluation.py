from __future__ import annotations

import json
from pathlib import Path

import torch

from dalg.analysis.adaptive_q_evaluation import evaluate_adaptive_q_toy
from dalg.data.manifold_dataset import ToyManifoldConfig, save_toy_manifold_shards
from dalg.data.shard_activations import load_meta_index
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, save_mfa_hddc


def test_adaptive_q_toy_evaluation_smoke(tmp_path: Path) -> None:
    shard_dir = save_toy_manifold_shards(
        tmp_path / "toy_shards",
        ToyManifoldConfig(
            ambient_dim=8,
            n_train=64,
            n_val=32,
            calibration_size=32,
            manifolds_per_type=1,
            offset_radius=2.0,
            seed=0,
        ),
        shard_size=24,
        layer=0,
    )
    metadata = torch.load(
        shard_dir / "manifold_metadata.pt",
        map_location="cpu",
        weights_only=True,
    )
    assignments = metadata["row_manifold_ids"].long()
    K = int(metadata["num_manifolds"])

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = MFA_HDDC(
        torch.randn(K, 8),
        rank=2,
        isotropic_psi=True,
    )
    save_mfa_hddc(model, str(run_dir / "mfa_model.pt"))
    (run_dir / "config.json").write_text(json.dumps({"model": "MFA_HDDC"}))

    meta_index = load_meta_index(shard_dir, layer=0)
    val_positions = list(range(0, len(meta_index), 4))
    val_rows = [meta_index[position]["global_row"] for position in val_positions]
    (run_dir / "val_indices.json").write_text(
        json.dumps(
            {
                "train_rows": len(meta_index) - len(val_positions),
                "val_rows": len(val_positions),
                "val_global_rows": val_rows,
            }
        )
    )
    torch.save(
        {
            "K": K,
            "assignments": assignments,
            "cluster_sizes": torch.bincount(assignments, minlength=K),
            "subset_spec": None,
        },
        run_dir / "mfa_model_assignments.pt",
    )

    metrics = evaluate_adaptive_q_toy(
        run_dir,
        shard_dir=shard_dir,
        layer=0,
        model_kind="hddc",
        batch_size=16,
        device="cpu",
    )

    assert metrics["evaluation"] == "adaptive_q_toy"
    assert metrics["K"] == K
    assert metrics["dataset"]["selected_rows"] == len(meta_index)
    assert metrics["components"]["dead"] == 0
    assert metrics["clustering"]["adjusted_rand_index"] == 1.0
    assert torch.isfinite(torch.tensor(metrics["nll"]["train"]))
    assert torch.isfinite(torch.tensor(metrics["nll"]["validation"]))
