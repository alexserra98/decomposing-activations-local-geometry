"""MFA/VAE model definitions and training utilities."""

from .mfa import MFA, save_mfa, load_mfa
from .vae import (
    ActivationVAELightning,
    FeatureStandardizer,
    MLPGaussianEncoder,
    MLPDecoder,
    MoGPrior,
    Prior,
    StandardGaussianPrior,
    VAE,
    VAEConfig,
    VampPrior,
    adapt_activation_batch,
    build_lightning_vae,
    build_prior,
    save_vae,
)

__all__ = [
    "ActivationVAELightning",
    "FeatureStandardizer",
    "MFA",
    "MLPDecoder",
    "MLPGaussianEncoder",
    "MoGPrior",
    "Prior",
    "StandardGaussianPrior",
    "VAE",
    "VAEConfig",
    "VampPrior",
    "adapt_activation_batch",
    "build_lightning_vae",
    "build_prior",
    "load_mfa",
    "save_mfa",
    "save_vae",
]
