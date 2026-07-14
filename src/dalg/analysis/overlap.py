"""Neighbour-overlap metrics for comparing representations and labels."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from dadapy.data import Data


PrecomputedNeighbors: TypeAlias = tuple[np.ndarray, np.ndarray]
OverlapInput: TypeAlias = np.ndarray | PrecomputedNeighbors


def compute_knn_euclidean(
    reference: torch.Tensor,
    query: torch.Tensor,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` nearest reference points for every query point.

    This helper does not add a self-neighbour column. When its output is passed
    to DADApy as precomputed neighbours, the caller must request and retain the
    self match as column zero (normally by calling it with ``k + 1`` and
    ``reference is query``).
    """
    if reference.ndim != 2 or query.ndim != 2:
        raise ValueError(
            "reference and query must both be 2D tensors, got "
            f"{tuple(reference.shape)} and {tuple(query.shape)}"
        )
    if reference.shape[1] != query.shape[1]:
        raise ValueError(
            "reference and query must have the same feature dimension, got "
            f"{reference.shape[1]} and {query.shape[1]}"
        )
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= reference.shape[0]:
        raise ValueError(f"k must be in [1, {reference.shape[0]}], got {k!r}")

    distances = torch.cdist(query.float(), reference.float(), p=2)
    distances_k, indices_k = torch.topk(distances, k=k, dim=1, largest=False, sorted=True)
    return distances_k.cpu().numpy(), indices_k.cpu().numpy()


# Keep the original misspelling as a compatibility alias for the first version
# of this module.
compute_knn_euclidian = compute_knn_euclidean


def _validate_k(k: int, n_samples: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k < n_samples:
        raise ValueError(f"k must be in [1, {n_samples - 1}], got {k!r}")


def _validate_coordinates(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2:
        raise ValueError(f"{name} coordinates must be 2D, got shape {value.shape}")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} coordinates must be numeric, got dtype {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} coordinates contain non-finite values")
    return value.astype(np.float64, copy=False)


def _validate_precomputed(
    value: PrecomputedNeighbors,
    name: str,
    k: int,
) -> PrecomputedNeighbors:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a (distances, indices) tuple")
    distances, indices = map(np.asarray, value)
    if distances.ndim != 2 or indices.ndim != 2 or distances.shape != indices.shape:
        raise ValueError(
            f"{name} distances and indices must be equally shaped 2D arrays, got "
            f"{distances.shape} and {indices.shape}"
        )
    if distances.shape[1] < k + 1:
        raise ValueError(
            f"{name} needs a self column plus {k} neighbours; got only "
            f"{distances.shape[1]} columns"
        )
    if not np.isfinite(distances).all():
        raise ValueError(f"{name} distances contain non-finite values")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{name} indices must be integers, got dtype {indices.dtype}")
    expected_self = np.arange(indices.shape[0])
    if not np.array_equal(indices[:, 0], expected_self):
        raise ValueError(f"{name} column zero must contain each sample's own index")
    if np.any(indices < 0) or np.any(indices >= indices.shape[0]):
        raise ValueError(f"{name} indices are outside [0, {indices.shape[0] - 1}]")
    return distances.astype(np.float64, copy=False), indices.astype(np.int64, copy=False)


def return_data_overlap(input_i: OverlapInput, input_j: OverlapInput, k: int) -> float:
    """Return mean overlap between the ``k``-NN sets of two metric spaces.

    Rows must identify the same samples in the same order. Inputs may either be
    coordinate matrices with shape ``(n_samples, n_features)`` or DADApy-style
    ``(distances, indices)`` tuples whose first column is the self neighbour.
    """
    if isinstance(input_i, tuple) != isinstance(input_j, tuple):
        raise TypeError("input_i and input_j must use the same input representation")

    if isinstance(input_i, tuple):
        distances_i = _validate_precomputed(input_i, "input_i", k)
        distances_j = _validate_precomputed(input_j, "input_j", k)
        n_i, n_j = distances_i[0].shape[0], distances_j[0].shape[0]
        _validate_k(k, n_i)
        if n_i != n_j:
            raise ValueError(f"inputs must have the same number of samples, got {n_i} and {n_j}")
        data = Data(distances=distances_i, maxk=k)
        return float(data.return_data_overlap(distances=distances_j, k=k))

    coordinates_i = _validate_coordinates(input_i, "input_i")
    coordinates_j = _validate_coordinates(input_j, "input_j")
    if coordinates_i.shape[0] != coordinates_j.shape[0]:
        raise ValueError(
            "inputs must have the same number of samples, got "
            f"{coordinates_i.shape[0]} and {coordinates_j.shape[0]}"
        )
    _validate_k(k, coordinates_i.shape[0])
    data = Data(coordinates=coordinates_i, maxk=k)
    return float(data.return_data_overlap(coordinates=coordinates_j, k=k))


def return_label_overlap(
    tensors: OverlapInput,
    labels: np.ndarray,
    k: int,
    *,
    weighted: bool = True,
) -> float:
    """Return the fraction of metric-space neighbours sharing each point's label.

    With ``weighted=True`` (DADApy's default), clusters contribute equally even
    when their populations differ. Set it to ``False`` for a sample-weighted
    average.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    if isinstance(tensors, tuple):
        neighbor_data = _validate_precomputed(tensors, "tensors", k)
        n_samples = neighbor_data[0].shape[0]
        data = Data(distances=neighbor_data, maxk=k)
    else:
        coordinates = _validate_coordinates(tensors, "tensors")
        n_samples = coordinates.shape[0]
        data = Data(coordinates=coordinates, maxk=k)

    _validate_k(k, n_samples)
    if labels.shape[0] != n_samples:
        raise ValueError(
            f"labels and tensors must have the same number of samples, got "
            f"{labels.shape[0]} and {n_samples}"
        )
    return float(data.return_label_overlap(labels=labels, k=k, weighted=weighted))
