"""Two-process CPU checks for gated HDDC M-steps.

Run with PYTHONPATH=src python -m torch.distributed.run --standalone \
    --nproc_per_node=2 tests/component_sharded_hddc_surgery_equivalence.py
"""

from datetime import timedelta

import torch
import torch.distributed as dist

from dalg.models.adaptive_q.hddc_surgery import (
    SurgeryConfig,
    _responsibilities,
    accumulate_statistics,
    hddc_surgery,
    reconstruct_components,
)
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, ComponentShardedMFA_HDDC


def _shard(full):
    model = ComponentShardedMFA_HDDC.from_global_centroids(
        full.mu.detach(), rank=full.q, dist_rank=dist.get_rank(), world_size=2,
        isotropic_psi=True,
    )
    start, end = model.component_start, model.component_end
    model.load_state_dict({key: value[start:end] for key, value in full.state_dict().items()})
    return model


def _check_model(full, shard):
    start, end = shard.component_start, shard.component_end
    for name in ("mu", "pi_logits", "rank_mask", "psi_rho"):
        torch.testing.assert_close(
            getattr(shard, name), getattr(full, name)[start:end], atol=3e-5, rtol=3e-5,
        )
    # Covariances avoid arbitrary eigenvector sign choices.
    covariance = full.W @ full.W.transpose(-1, -2) + torch.diag_embed(full._psi())
    local_covariance = shard.W @ shard.W.transpose(-1, -2) + torch.diag_embed(shard._psi())
    torch.testing.assert_close(local_covariance, covariance[start:end], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        shard.local_log_pi().exp(), full.pi_logits.softmax(0)[start:end],
        atol=2e-6, rtol=2e-6,
    )


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    if dist.get_world_size() != 2:
        raise ValueError("this test requires two processes")
    torch.manual_seed(37)
    full = MFA_HDDC(torch.randn(4, 4), rank=1, isotropic_psi=True, psi_init=4.0)
    with torch.no_grad():
        full.pi_logits.copy_(torch.tensor([0.4, 0.3, 0.2, 0.1]).log())
    shard = _shard(full)
    start, end = shard.component_start, shard.component_end
    x = torch.randn(150, 4) * torch.tensor([3.0, 1.5, 0.3, 0.2]) + 2
    batches = list(x.split(17))
    torch.testing.assert_close(
        _responsibilities(shard, x), full.responsibilities(x)[:, start:end],
        atol=2e-6, rtol=2e-6,
    )
    N, A, B, _ = accumulate_statistics(full, batches, device=x.device)
    cfg = SurgeryConfig(threshold=0.01)
    reconstruct_components(full, N, A, B, cfg)
    summary = hddc_surgery(shard, batches, cfg)
    assert summary["n_updated"] == 4
    _check_model(full, shard)

    # Exercise eligibility spanning shards, one empty local eligible set, and
    # the global no-op. These moments have known shifted means and covariances.
    for counts in ([50.0, 1.0, 30.0, 0.0], [50.0, 30.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]):
        full = MFA_HDDC(torch.zeros(4, 4), rank=1, isotropic_psi=True)
        with torch.no_grad():
            full.pi_logits.copy_(torch.tensor([0.4, 0.3, 0.2, 0.1]).log() + 20)
        shard = _shard(full)
        old_pi = full.pi_logits.softmax(0).detach()
        before = {key: value.clone() for key, value in shard.state_dict().items()}
        N = torch.tensor(counts, dtype=torch.float64)
        shift = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 4
        A = N[:, None] * shift
        covariance = torch.diag(torch.tensor([9.0, 1.0, 1.0, 1.0], dtype=torch.float64))
        B = N[:, None, None] * (covariance + shift[:, :, None] * shift[:, None, :])
        cfg = SurgeryConfig(min_count=2)
        reconstruct_components(full, N, A, B, cfg)
        reconstruct_components(shard, N[start:end], A[start:end], B[start:end], cfg)
        _check_model(full, shard)
        torch.testing.assert_close(full.pi_logits.softmax(0)[N < 2], old_pi[N < 2])
        if bool((N[start:end] < 2).all()):
            for key, value in shard.state_dict().items():
                assert torch.equal(value, before[key])

    # An invalid proposal on rank 1 must also stop rank 0 before it commits.
    for failure in ("statistics", "covariance"):
        full = MFA_HDDC(torch.zeros(4, 4), rank=1, isotropic_psi=True)
        shard = _shard(full)
        before = {key: value.clone() for key, value in shard.state_dict().items()}
        N = torch.ones(2, dtype=torch.float64)
        A = torch.zeros(2, 4, dtype=torch.float64)
        B = torch.diag(torch.tensor([9.0, 1.0, 1.0, 1.0], dtype=torch.float64)).repeat(2, 1, 1)
        if dist.get_rank() == 1:
            if failure == "statistics":
                N[0] = -1
            else:
                B[0].zero_()
        try:
            reconstruct_components(shard, N, A, B, SurgeryConfig())
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("both shards must reject the invalid proposal")
        for key, value in shard.state_dict().items():
            assert torch.equal(value, before[key])

    if dist.get_rank() == 0:
        print("component-sharded HDDC surgery equivalence passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
