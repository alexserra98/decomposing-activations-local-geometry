"""Dataset loading and sharded activation utilities."""

from .manifold_dataset import ToyManifoldConfig, make_toy_manifold_datasets

__all__ = ["ToyManifoldConfig", "make_toy_manifold_datasets"]
