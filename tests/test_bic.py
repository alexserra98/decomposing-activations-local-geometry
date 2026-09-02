import json
import math

import pytest
import torch

from dalg.analysis.bic import compute_bic, model_parameter_count
from dalg.models.adaptive_q.mfa_ard import MFA_ARD
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, save_mfa_hddc
from dalg.models.mfa import MFA


def test_model_parameter_count_for_each_model_kind():
    centroids = torch.zeros(2, 3)

    mfa = MFA(centroids, rank=2)
    ard = MFA_ARD(centroids, rank=2, rank_threshold=0.5)
    hddc = MFA_HDDC(centroids, rank=2, isotropic_psi=True)
    hddc_shared = MFA_HDDC(centroids, rank=2, shared_b=True)

    assert model_parameter_count(mfa, "mfa") == 21
    assert model_parameter_count(ard, "ard") == 22
    assert model_parameter_count(hddc, "hddc") == 21
    assert model_parameter_count(hddc_shared, "hddc") == 20


def test_compute_bic_from_run(tmp_path):
    shard_dir = tmp_path / "shards"
    (shard_dir / "layer00").mkdir(parents=True)
    (shard_dir / "meta").mkdir()

    activations = torch.tensor(
        [
            [[0.0, 0.5]],
            [[1.0, -0.5]],
            [[0.5, 0.25]],
            [[-1.0, 0.0]],
        ]
    )
    torch.save(activations, shard_dir / "layer00" / "shard_00000.pt")
    (shard_dir / "config.json").write_text(
        json.dumps({"window": 1, "d_model": 2, "drop_prefix": 0})
    )
    (shard_dir / "meta" / "shard_00000.json").write_text(
        json.dumps(
            {
                "row_indices": [0, 1, 2, 3],
                "rows": [{"subset": "all"}] * 4,
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
                "window": 1,
                "drop_prefix": 0,
                "model": "MFA_HDDC",
            }
        )
    )
    (run_dir / "val_indices.json").write_text(
        json.dumps({"train_rows": 3, "val_global_rows": [1]})
    )

    model = MFA_HDDC(
        centroids=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        rank=1,
        shared_b=True,
    )
    save_mfa_hddc(model, run_dir / "mfa_model.pt")

    train = activations[[0, 2, 3]].flatten(0, 1)
    log_likelihood = model.log_prob(train).double().sum().item()
    # rho + tau_bar + sum(d_k) + K dimension parameters + one shared b.
    parameter_count = 5 + 2 + 2 + 2 + 1
    expected = -2.0 * log_likelihood + parameter_count * math.log(3)

    assert compute_bic(run_dir, batch_size=2) == pytest.approx(expected)


def test_compute_bic_rejects_a_mismatched_split(tmp_path):
    run_dir = tmp_path / "run"
    shard_dir = tmp_path / "shards"
    run_dir.mkdir()
    (shard_dir / "layer00").mkdir(parents=True)
    (shard_dir / "meta").mkdir()
    torch.save(torch.zeros(2, 1, 1), shard_dir / "layer00" / "shard_00000.pt")
    (shard_dir / "config.json").write_text(
        json.dumps({"window": 1, "d_model": 1, "drop_prefix": 0})
    )
    (shard_dir / "meta" / "shard_00000.json").write_text(
        json.dumps({"row_indices": [0, 1], "rows": [{}, {}]})
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {"shard_dir": str(shard_dir), "layer": 0, "window": 1, "drop_prefix": 0}
        )
    )
    (run_dir / "val_indices.json").write_text(
        json.dumps({"train_rows": 0, "val_global_rows": [1]})
    )

    with pytest.raises(ValueError, match="recorded and reconstructed"):
        compute_bic(run_dir)
