import numpy as np
import pytest
import torch

from dalg.analysis.overlap import (
    compute_knn_euclidean,
    return_data_overlap,
    return_label_overlap,
)


def _neighbors(indices: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    indices_array = np.asarray(indices, dtype=np.int64)
    distances = np.broadcast_to(
        np.arange(indices_array.shape[1], dtype=np.float64),
        indices_array.shape,
    ).copy()
    return distances, indices_array


def test_compute_knn_euclidean_returns_sorted_neighbors() -> None:
    reference = torch.tensor([[0.0], [3.0], [8.0]])
    query = torch.tensor([[1.0], [6.0]])

    distances, indices = compute_knn_euclidean(reference, query, k=2)

    np.testing.assert_allclose(distances, [[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_array_equal(indices, [[0, 1], [2, 1]])


def test_data_overlap_is_one_for_identical_coordinates() -> None:
    coordinates = np.array([[0.0], [1.0], [10.0], [11.0]])
    assert return_data_overlap(coordinates, coordinates.copy(), k=1) == pytest.approx(1.0)


def test_data_overlap_uses_precomputed_neighbor_sets() -> None:
    space_a = _neighbors([[0, 1], [1, 0], [2, 3], [3, 2]])
    space_b = _neighbors([[0, 2], [1, 3], [2, 0], [3, 1]])

    assert return_data_overlap(space_a, space_a, k=1) == pytest.approx(1.0)
    assert return_data_overlap(space_a, space_b, k=1) == pytest.approx(0.0)


def test_label_overlap_measures_local_label_purity() -> None:
    coordinates = np.array([[0.0], [1.0], [10.0], [11.0]])

    assert return_label_overlap(coordinates, np.array([0, 0, 1, 1]), k=1) == pytest.approx(1.0)
    assert return_label_overlap(coordinates, np.array([0, 1, 0, 1]), k=1) == pytest.approx(0.0)


def test_precomputed_neighbors_require_self_column() -> None:
    missing_self = _neighbors([[1, 2], [0, 2], [0, 1]])
    with pytest.raises(ValueError, match="column zero"):
        return_data_overlap(missing_self, missing_self, k=1)
