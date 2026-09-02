import json

import pytest
import torch

from dalg.analysis.bic import compute_bic
from dalg.analysis.bic_improved import (
    active_bic_from_standard,
    compute_improved_bic,
    compute_improved_bic_details,
)
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, save_mfa_hddc


def test_active_bic_rewards_each_additional_active_component():
    one_active = active_bic_from_standard(
        120.0,
        n=10,
        active_components=1,
        K=3,
    )
    three_active = active_bic_from_standard(
        120.0,
        n=10,
        active_components=3,
        K=3,
    )

    assert three_active - one_active == pytest.approx(2.0)


def _write_run(tmp_path):
    shard_dir = tmp_path / "shards"
    (shard_dir / "layer00").mkdir(parents=True)
    (shard_dir / "meta").mkdir()

    # Three two-token rows.  The middle row is validation-only, allowing the
    # test to prove that its assignments do not make component 1 train-active.
    activations = torch.tensor(
        [
            [[0.0, 0.5], [0.25, 0.0]],
            [[1.0, -0.5], [1.0, 0.25]],
            [[-1.0, 0.0], [-0.5, 0.5]],
        ]
    )
    torch.save(activations, shard_dir / "layer00" / "shard_00000.pt")
    (shard_dir / "config.json").write_text(
        json.dumps({"window": 2, "d_model": 2, "drop_prefix": 0})
    )
    (shard_dir / "meta" / "shard_00000.json").write_text(
        json.dumps(
            {
                "row_indices": [0, 1, 2],
                "rows": [{"subset": "all"}] * 3,
            }
        )
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "shard_dir": str(shard_dir),
                "layer": 0,
                "window": 2,
                "drop_prefix": 0,
                "model": "MFA_HDDC",
                "K": 2,
            }
        )
    )
    (run_dir / "val_indices.json").write_text(
        json.dumps(
            {
                "train_rows": 2,
                "val_rows": 1,
                "val_global_rows": [1],
            }
        )
    )

    model = MFA_HDDC(
        centroids=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        rank=1,
        shared_b=True,
    )
    save_mfa_hddc(model, run_dir / "mfa_model.pt")

    assignments = torch.tensor([0, 0, 1, 1, 0, 0])
    torch.save(
        {
            "assignments": assignments,
            "cluster_sizes": torch.bincount(assignments, minlength=2),
            "K": 2,
            "subset_spec": None,
        },
        run_dir / "mfa_model_assignments.pt",
    )
    return run_dir


def test_compute_improved_bic_uses_training_assignments_only(tmp_path):
    run_dir = _write_run(tmp_path)

    standard_bic = compute_bic(run_dir, batch_size=2)
    details = compute_improved_bic_details(run_dir, batch_size=2)

    assert details["n"] == 4
    assert details["active_components"] == 1
    assert details["inactive_components"] == 1
    assert details["standard_bic"] == pytest.approx(standard_bic)
    assert details["value"] == pytest.approx(-standard_bic / 4 + 1)
    assert compute_improved_bic(run_dir, batch_size=2) == pytest.approx(
        details["value"]
    )


def test_compute_improved_bic_rejects_inconsistent_assignments(tmp_path):
    run_dir = _write_run(tmp_path)
    path = run_dir / "mfa_model_assignments.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    bundle["cluster_sizes"] = torch.tensor([3, 3])
    torch.save(bundle, path)

    with pytest.raises(ValueError, match="cluster_sizes is inconsistent"):
        compute_improved_bic(run_dir)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "standard_bic": float("nan"),
                "n": 1,
                "active_components": 1,
                "K": 1,
            },
            "finite",
        ),
        (
            {"standard_bic": 1.0, "n": 0, "active_components": 0, "K": 1},
            "at least one",
        ),
        (
            {"standard_bic": 1.0, "n": 1, "active_components": 2, "K": 1},
            "\\[0, K\\]",
        ),
    ],
)
def test_active_bic_validates_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        active_bic_from_standard(**kwargs)
