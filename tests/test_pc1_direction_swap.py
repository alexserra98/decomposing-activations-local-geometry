"""Controls for score-independent matching and fixed-center direction swaps."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from dalg.analysis.pc1_density import KNNLogDensity, log_average_density

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/temporary'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('direction_swap', SCRIPTS / 'run_years_pc1_direction_swap.py')
swap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swap)


def test_matching_is_bijective_global_minimum_not_greedy():
    # Greedy matching 2->1.1 and 0->100 costs 100.9; global solution costs 99.1.
    h = torch.tensor([[2.], [0.]], dtype=torch.float64)
    k = torch.tensor([[1.1], [100.]], dtype=torch.float64)
    matches = swap.match_centers(h, k)
    assert [(m['h_component'], m['k_component']) for m in matches] == [(0, 1), (1, 0)]
    assert sum(m['center_distance'] for m in matches) == pytest.approx(99.1)
    assert matches[0]['h_neighbor_rank'] == 2
    assert not any(m['mutual_nearest'] for m in matches)


def test_common_interval_uses_shorter_projected_length_for_both_directions():
    h = dict(centers=torch.tensor([[0., 1.]]), vectors=torch.tensor([[1., 0.]]),
             half_lengths=torch.tensor([3.]))
    k = dict(centers=torch.tensor([[2., 3.]]), vectors=torch.tensor([[0., 1.]]),
             half_lengths=torch.tensor([.5]))
    centers, directions, length = swap.common_geometry(h, k, 0, 0)
    assert length == .5
    torch.testing.assert_close(centers, torch.tensor([[0., 1.], [2., 3.]]))
    torch.testing.assert_close(directions.norm(dim=1) * length, torch.tensor([.5, .5]))


def test_paired_scores_match_direct_dense_query_integrals_and_cancel_center():
    generator = torch.Generator().manual_seed(91)
    reference = torch.randn(71, 2, generator=generator, dtype=torch.float64) * torch.tensor([2., .4])
    estimator = KNNLogDensity(reference, k=9)
    centers = torch.tensor([[.1, .2], [.6, -.1]], dtype=torch.float64)
    directions = torch.eye(2, dtype=torch.float64)
    result = swap.evaluate_pair(estimator, centers, directions, .8)
    t = torch.linspace(-.8, .8, 20001, dtype=torch.float64)
    expected = []
    for center in centers:
        queries = center + directions[:, None] * t[None, :, None]
        radii = torch.cdist(queries.reshape(-1, 2), reference,
                            compute_mode='donot_use_mm_for_euclid_dist').kthvalue(9).values.reshape(2, -1)
        means = log_average_density(estimator.log_constant - 2 * radii.log())
        expected.append(means[0] - means[1])
    assert result['delta_at_h_center'] == pytest.approx(float(expected[0]), abs=2e-5)
    assert result['delta_at_k_center'] == pytest.approx(float(expected[1]), abs=2e-5)
    assert result['paired_delta'] == pytest.approx(float(torch.stack(expected).mean()), abs=2e-5)
    averages = result['log_average_densities']
    torch.testing.assert_close(result['log_retentions'][:, 0] - result['log_retentions'][:, 1],
                               averages[:, 0] - averages[:, 1])


def test_direction_swap_reverses_sign_and_sign_of_eigenvector_does_not_matter():
    generator = torch.Generator().manual_seed(19)
    estimator = KNNLogDensity(torch.randn(50, 2, generator=generator, dtype=torch.float64), k=8)
    centers = torch.tensor([[0., .1], [.4, -.2]], dtype=torch.float64)
    directions = torch.eye(2, dtype=torch.float64)
    original = swap.evaluate_pair(estimator, centers, directions, .7)
    reversed_order = swap.evaluate_pair(estimator, centers, directions.flip(0), .7)
    negated = swap.evaluate_pair(estimator, centers, -directions, .7)
    assert reversed_order['paired_delta'] == pytest.approx(-original['paired_delta'], abs=1e-10)
    assert negated['paired_delta'] == pytest.approx(original['paired_delta'], abs=1e-8)
    same = swap.evaluate_pair(estimator, centers, directions[[0, 0]], .7)
    assert same['paired_delta'] == pytest.approx(0., abs=1e-12)


def test_invalid_pair_lengths_fail_explicitly():
    geometry = dict(centers=torch.zeros(1, 2), vectors=torch.tensor([[1., 0.]]),
                    half_lengths=torch.tensor([0.]))
    with pytest.raises(ValueError, match='positive finite'):
        swap.common_geometry(geometry, geometry, 0, 0)
