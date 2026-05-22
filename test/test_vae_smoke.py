from __future__ import annotations

import torch

from dalg.models.train_vae import adapt_loader_batch
from dalg.models.vae import FeatureStandardizer, StandardGaussianPrior, VAE


def test_vae_forward_and_loss_finite_raw() -> None:
    torch.manual_seed(0)
    model = VAE(input_dim=2048, latent_dim=16, prior=StandardGaussianPrior(16), normalizer=None)
    x = torch.randn(32, 2048)
    out = model(x)
    losses = model.loss(x, out)

    assert out["recon"].shape == (32, 2048)
    assert out["mu"].shape == (32, 16)
    assert out["logvar"].shape == (32, 16)
    assert torch.isfinite(losses["loss"]) and losses["loss"].ndim == 0


def test_vae_forward_and_loss_finite_meanstd() -> None:
    torch.manual_seed(0)
    mean = torch.zeros(2048)
    std = torch.ones(2048)
    normalizer = FeatureStandardizer(mean, std, clip_value=5.0)
    model = VAE(input_dim=2048, latent_dim=16, prior=StandardGaussianPrior(16), normalizer=normalizer)
    x = torch.randn(32, 2048)
    out = model(x)
    losses = model.loss(x, out)

    assert "recon_input" in out
    assert torch.isfinite(losses["rec_loss"]) and torch.isfinite(losses["kl_loss"])


def test_tiny_cpu_training_step() -> None:
    torch.manual_seed(0)
    model = VAE(input_dim=2048, latent_dim=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(16, 2048)
    out = model(x)
    losses = model.loss(x, out)
    losses["loss"].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    assert torch.isfinite(losses["loss"]) and losses["loss"].item() > 0


def test_batch_adapter_current_loader_contract() -> None:
    x = torch.randn(128, 2048)
    tok = torch.randint(0, 100, (128,))
    xb, tb, mb = adapt_loader_batch((x, tok))

    assert xb.shape == (128, 2048)
    assert tb is not None and tb.shape == (128,)
    assert mb is None


def test_batch_adapter_future_loader_contract() -> None:
    x = torch.randn(4, 32, 2048)
    tok = torch.randint(0, 100, (4, 32))
    meta = [{"shard": 0}] * 4

    xb, tb, mb = adapt_loader_batch((x, tok, meta))
    assert xb.shape == (128, 2048)
    assert tb is not None and tb.shape == (128,)
    assert mb == meta

    xb2, tb2, mb2 = adapt_loader_batch(((x, tok), meta))
    assert xb2.shape == (128, 2048)
    assert tb2 is not None and tb2.shape == (128,)
    assert mb2 == meta
