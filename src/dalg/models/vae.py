from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_2PI = math.log(2.0 * math.pi)
LOGVAR_MIN = -10.0
LOGVAR_MAX = 10.0


def make_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    *,
    activation: type[nn.Module] = nn.SiLU,
    dropout: float = 0.0,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [in_dim, *hidden_dims]
    for din, dout in zip(dims, dims[1:]):
        layers.append(nn.Linear(din, dout))
        if layer_norm:
            layers.append(nn.LayerNorm(dout))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


def clamp_logvar(logvar: torch.Tensor) -> torch.Tensor:
    return logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)


def diagonal_gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = clamp_logvar(logvar)
    return -0.5 * (logvar + (z - mu).pow(2) / logvar.exp() + LOG_2PI).sum(dim=-1)


def adapt_activation_batch(batch: object, *, input_dim: int | None = None) -> torch.Tensor:
    """Extract a rank-2 activation tensor from the shard-loader batch contract."""
    x = batch
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("empty activation batch")
        first = batch[0]
        if torch.is_tensor(first):
            x = first
        elif isinstance(first, (tuple, list)) and first and torch.is_tensor(first[0]):
            x = first[0]

    if not torch.is_tensor(x):
        raise ValueError("unable to extract an activation tensor from batch")

    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim >= 3:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim != 2:
        raise ValueError(f"expected rank-2 or rank-3+ activations, got shape {tuple(x.shape)}")

    if input_dim is not None and x.shape[-1] != int(input_dim):
        raise ValueError(f"expected activation dim {input_dim}, got {x.shape[-1]}")
    return x


class Prior(nn.Module):
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logvar = clamp_logvar(logvar)
        if z is None:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        log_q = diagonal_gaussian_log_prob(z, mu, logvar)
        log_p = self.log_prob(z)
        return (log_q - log_p).mean()


class StandardGaussianPrior(Prior):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z.pow(2) + LOG_2PI).sum(dim=-1)

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logvar = clamp_logvar(logvar)
        return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1).mean()


class MoGPrior(Prior):
    def __init__(self, latent_dim: int, n_components: int = 10) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.logits = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, latent_dim) * 0.01)
        self.logvars = nn.Parameter(torch.zeros(n_components, latent_dim))

    def component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        z_expanded = z.unsqueeze(1)
        means = self.means.unsqueeze(0)
        logvars = clamp_logvar(self.logvars).unsqueeze(0)
        return -0.5 * (
            logvars + (z_expanded - means).pow(2) / logvars.exp() + LOG_2PI
        ).sum(dim=-1)

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        log_weights = F.log_softmax(self.logits, dim=0).unsqueeze(0)
        return log_weights + self.component_log_prob(z)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self.log_joint(z), dim=1)

    def responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        log_joint = self.log_joint(z)
        log_norm = torch.logsumexp(log_joint, dim=1, keepdim=True)
        return torch.exp(log_joint - log_norm)


class VampPrior(Prior):
    def __init__(
        self,
        encoder: nn.Module | None,
        latent_dim: int,
        input_dim: int,
        n_components: int = 500,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.n_components = n_components
        self.pseudo_inputs = nn.Parameter(torch.randn(n_components, input_dim) * 0.01)

    def bind_encoder(self, encoder: nn.Module) -> None:
        self.encoder = encoder

    def _component_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is None:
            raise RuntimeError("VampPrior requires an encoder; call bind_encoder(...) first.")
        mu, logvar = self.encoder.encode(self.pseudo_inputs)
        return mu, clamp_logvar(logvar)

    def component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        mu, logvar = self._component_params()
        z_expanded = z.unsqueeze(1)
        mu = mu.unsqueeze(0)
        logvar = logvar.unsqueeze(0)
        return -0.5 * (
            logvar + (z_expanded - mu).pow(2) / logvar.exp() + LOG_2PI
        ).sum(dim=-1)

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        log_weights = torch.full(
            (1, self.n_components),
            -math.log(float(self.n_components)),
            device=z.device,
            dtype=z.dtype,
        )
        return log_weights + self.component_log_prob(z)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self.log_joint(z), dim=1)

    def responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        log_joint = self.log_joint(z)
        log_norm = torch.logsumexp(log_joint, dim=1, keepdim=True)
        return torch.exp(log_joint - log_norm)


class FeatureStandardizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, clip_value: float | None = None) -> None:
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-6))
        self.clip_value = clip_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x - self.mean) / self.std
        if self.clip_value is not None:
            y = y.clamp(min=-self.clip_value, max=self.clip_value)
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


class MLPGaussianEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (1024, 512),
        dropout: float = 0.0,
        layer_norm: bool = False,
        normalizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        backbone_out = hidden_dims[-1] if hidden_dims else input_dim
        self.backbone = make_mlp(
            input_dim,
            hidden_dims,
            backbone_out,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.mu = nn.Linear(backbone_out, latent_dim)
        self.logvar = nn.Linear(backbone_out, latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalizer is not None:
            x = self.normalizer(x.float())
        h = self.backbone(x)
        return self.mu(h), clamp_logvar(self.logvar(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(x)


class MLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 1024),
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            latent_dim,
            hidden_dims,
            output_dim,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        latent_dim: int = 32,
        enc_hidden_dims: Sequence[int] = (1024, 512),
        dec_hidden_dims: Sequence[int] = (512, 1024),
        prior: Prior | None = None,
        dropout: float = 0.0,
        layer_norm: bool = False,
        beta: float = 1.0,
        normalizer: FeatureStandardizer | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = float(beta)
        self.normalizer = normalizer
        self.encoder = MLPGaussianEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=enc_hidden_dims,
            dropout=dropout,
            layer_norm=layer_norm,
            normalizer=normalizer,
        )
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=dec_hidden_dims,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.prior = prior if prior is not None else StandardGaussianPrior(latent_dim)
        if isinstance(self.prior, VampPrior):
            self.prior.bind_encoder(self.encoder)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode(x.float())

    def normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return x.float()
        return self.normalizer(x.float())

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        out: dict[str, torch.Tensor] = {"recon": recon, "mu": mu, "logvar": logvar, "z": z}
        if self.normalizer is not None:
            out["recon_input"] = self.normalizer.inverse(recon)
        return out

    def loss(self, x: torch.Tensor, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_norm = self.normalize_inputs(x)
        recon = out["recon"]
        mu = out["mu"]
        logvar = out["logvar"]
        z = out["z"]
        rec_loss = F.mse_loss(recon, x_norm, reduction="none").sum(dim=-1).mean()
        kl_loss = self.prior.kl_divergence(mu, logvar, z)
        total = rec_loss + self.beta * kl_loss
        return {
            "loss": total,
            "rec_loss": rec_loss,
            "kl_loss": kl_loss,
            "beta": torch.tensor(self.beta, device=x.device),
        }


def ActivationVAELightning(
    vae: VAE,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    beta_warmup_steps: int = 0,
):
    """Construct a LightningModule wrapper around a VAE when Lightning is needed."""
    import pytorch_lightning as pl

    class _ActivationVAELightning(pl.LightningModule):
        def __init__(
            self,
            vae: VAE,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            beta_warmup_steps: int = 0,
        ) -> None:
            super().__init__()
            self.save_hyperparameters(ignore=["vae"])
            self.vae = vae

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return self.vae(x)

        def current_beta(self) -> float:
            target_beta = float(self.vae.beta)
            warmup_steps = int(self.hparams.beta_warmup_steps)
            if warmup_steps <= 0:
                return target_beta
            progress = min(1.0, float(self.global_step + 1) / float(warmup_steps))
            return target_beta * progress

        def _shared_step(self, batch: object, stage: str) -> torch.Tensor:
            x = adapt_activation_batch(batch, input_dim=self.vae.input_dim).float()
            out = self.vae(x)
            beta = self.current_beta()
            old_beta = self.vae.beta
            self.vae.beta = beta
            losses = self.vae.loss(x, out)
            self.vae.beta = old_beta
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError(
                    f"Non-finite {stage} loss: loss={losses['loss'].item()} "
                    f"rec={losses['rec_loss'].item()} kl={losses['kl_loss'].item()}"
                )

            self.log(f"{stage}/loss", losses["loss"], prog_bar=True, on_step=stage == "train", on_epoch=True)
            self.log(f"{stage}/rec", losses["rec_loss"], prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}/kl", losses["kl_loss"], prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}/beta", beta, prog_bar=False, on_step=stage == "train", on_epoch=True)
            return losses["loss"]

        def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
            return self._shared_step(batch, "train")

        def validation_step(self, batch: object, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
            return self._shared_step(batch, "val")

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

    return _ActivationVAELightning(
        vae=vae,
        lr=lr,
        weight_decay=weight_decay,
        beta_warmup_steps=beta_warmup_steps,
    )


@dataclass
class VAEConfig:
    input_dim: int = 2048
    latent_dim: int = 32
    enc_hidden_dims: tuple[int, ...] = (1024, 512)
    dec_hidden_dims: tuple[int, ...] = (512, 1024)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    beta: float = 1.0
    beta_warmup_steps: int = 0
    input_mean: torch.Tensor | None = None
    input_std: torch.Tensor | None = None
    input_clip: float | None = None


def build_lightning_vae(config: VAEConfig, prior: Prior | None = None):
    normalizer = None
    if config.input_mean is not None and config.input_std is not None:
        normalizer = FeatureStandardizer(config.input_mean, config.input_std, clip_value=config.input_clip)
    vae = VAE(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        enc_hidden_dims=config.enc_hidden_dims,
        dec_hidden_dims=config.dec_hidden_dims,
        prior=prior,
        beta=config.beta,
        normalizer=normalizer,
    )
    return ActivationVAELightning(
        vae=vae,
        lr=config.lr,
        weight_decay=config.weight_decay,
        beta_warmup_steps=config.beta_warmup_steps,
    )


def build_prior(
    name: str,
    latent_dim: int,
    prior_components: int = 100,
    *,
    input_dim: int = 2048,
) -> Prior:
    if name == "standard":
        return StandardGaussianPrior(latent_dim)
    if name == "mog":
        return MoGPrior(latent_dim, n_components=prior_components)
    if name == "vamp":
        return VampPrior(
            encoder=None,
            latent_dim=latent_dim,
            input_dim=input_dim,
            n_components=prior_components,
        )
    raise ValueError(f"Unsupported prior {name!r}.")


def save_vae(model: VAE, path: str | Path) -> None:
    """Save a VAE checkpoint with enough config to rebuild the module."""
    prior = model.prior
    if isinstance(prior, StandardGaussianPrior):
        prior_config = {"name": "standard"}
    elif isinstance(prior, MoGPrior):
        prior_config = {"name": "mog", "n_components": prior.n_components}
    elif isinstance(prior, VampPrior):
        prior_config = {"name": "vamp", "n_components": prior.n_components}
    else:
        prior_config = {"name": type(prior).__name__}

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "beta": model.beta,
            "prior": prior_config,
        },
        path,
    )
