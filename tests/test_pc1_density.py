"""Independent numerical and covariance checks for the PC1 density metric."""
import math
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import pytest
import torch

from dalg.analysis.pc1_density import (
    KNNLogDensity, leading_components, log_average_density,
    random_unit_directions, score_component,
    fit_global_variance_pca, project_pc1_segments,
)


def test_line_density_matches_direct_distances_and_ball_volume():
    generator = torch.Generator().manual_seed(7)
    reference = torch.randn(53, 4, generator=generator, dtype=torch.float64) + 1e6
    center = reference.mean(0)
    directions = random_unit_directions(4, 5, seed=13)
    offsets = torch.linspace(-2, 2, 11, dtype=torch.float64)
    estimator = KNNLogDensity(reference, k=7)
    actual, radii = estimator.line_profiles(center, directions, offsets)
    queries = center + directions[:, None, :] * offsets[None, :, None]
    expected_radii = torch.cdist(queries.reshape(-1, 4), reference,
                                 compute_mode='donot_use_mm_for_euclid_dist').kthvalue(7).values.reshape(5, 11)
    torch.testing.assert_close(radii, expected_radii, rtol=1e-9, atol=1e-9)
    expected = math.log(7 / 53) - math.log(math.pi**2 / 2) - 4 * expected_radii.log()
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


def test_density_average_is_arithmetic_and_stable():
    logs = torch.tensor([1., 4., 1.], dtype=torch.float64).log() - 10000
    assert float(log_average_density(logs)) == pytest.approx(math.log(2.5) - 10000)
    assert float(log_average_density(torch.full((101,), -10000., dtype=torch.float64))) == pytest.approx(-10000)


def test_retention_matches_gaussian_integral_and_density_scale_cancels():
    class GaussianDensity:
        dimension = 3

        def line_profiles(self, center, directions, offsets):
            logs = (-20000 - offsets.square() / 2).expand(len(directions), -1)
            return logs, torch.ones_like(logs)

    result = score_component(GaussianDensity(), torch.zeros(3), torch.tensor([1., 0., 0.]),
                             4., n_random=8, seed=9)
    a = 2 * NormalDist().inv_cdf(.9)
    exact_average = math.sqrt(2 * math.pi) * math.erf(a / math.sqrt(2)) / (2 * a)
    assert result['half_length'] == pytest.approx(a)
    assert math.exp(result['log_retention']) == pytest.approx(exact_average, rel=2e-5)
    assert result['log_enrichment'] == pytest.approx(0., abs=1e-10)
    assert result['center_log_density'] == -20000
    assert result['relative_positions'].numel() == 201


def test_random_baseline_averages_density_scores_not_log_scores():
    class AnisotropicGaussian:
        dimension = 2

        def line_profiles(self, center, directions, offsets):
            inverse_variance = (directions / torch.tensor([.15, 3.])).square().sum(1)
            logs = -10000 - .5 * inverse_variance[:, None] * offsets[None, :].square()
            return logs, torch.ones_like(logs)

    result = score_component(AnisotropicGaussian(), torch.zeros(2), torch.tensor([0., 1.]),
                             9., n_random=32, seed=42)
    directions = random_unit_directions(2, 32, 42)
    sigma = (directions / torch.tensor([.15, 3.])).square().sum(1).rsqrt()
    a = result['half_length']
    random_scores = math.sqrt(2 * math.pi) * sigma * torch.erf(a / (math.sqrt(2) * sigma)) / (2 * a)
    pc1_score = math.sqrt(2 * math.pi) * 3 * math.erf(a / (math.sqrt(2) * 3)) / (2 * a)
    expected = math.log(pc1_score / float(random_scores.mean()))
    assert result['log_enrichment'] == pytest.approx(expected, abs=2e-5)
    assert result['log_enrichment'] > 0


def test_hddc_pc1_uses_svd_and_includes_noise_excludes_inactive():
    # Strongest loading is column 1, not column 0.
    loadings = torch.zeros(3, 3, 2)
    loadings[:, 0, 0] = 1
    loadings[:, 1, 1] = 3
    model = SimpleNamespace(mu=torch.zeros(3, 3), pi_logits=torch.tensor([0., 0., -math.inf]),
                            component_ranks=torch.tensor([2, 0, 2]),
                            _W=lambda: loadings, _psi=lambda: torch.full((3, 3), 2.))
    geometry = leading_components(model, 'hddc')
    assert geometry['eigenvalues'][0] == 11
    assert geometry['vectors'][0, 1].abs() == 1
    assert geometry['exclusion_reasons'] == ['', 'isotropic_covariance', 'zero_mixture_weight']


def test_kmeans_unit_pc_and_eigenvalue_are_separate():
    model = SimpleNamespace(mu=torch.zeros(2, 3), cluster_counts=torch.tensor([8, 1]),
                            eigenvalues=torch.tensor([[9., 2., 1.], [0., 0., 0.]]),
                            component_ranks=torch.tensor([1, 0]), pca_valid=torch.tensor([True, False]),
                            W=torch.tensor([[[1.], [0.], [0.]], [[0.], [0.], [0.]]]))
    geometry = leading_components(model, 'kmeans')
    assert geometry['eigenvalues'][0] == 9
    assert geometry['vectors'][0].norm() == 1
    assert geometry['exclusion_reasons'] == ['', 'invalid_cluster_pca']


def test_invalid_radii_and_grid_are_explicit_errors():
    estimator = KNNLogDensity(torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64), k=1)
    with pytest.raises(ValueError, match='positive'):
        estimator.line_profiles(torch.zeros(2), torch.tensor([[1., 0.]]), torch.zeros(1))
    with pytest.raises(ValueError, match='odd'):
        score_component(estimator, torch.zeros(2), torch.tensor([1., 0.]), 1., grid_points=100)


def test_adaptive_knn_integral_matches_dense_uniform_grid():
    generator = torch.Generator().manual_seed(19)
    reference = torch.randn(60, 3, generator=generator, dtype=torch.float64)
    estimator = KNNLogDensity(reference, k=8)
    center = torch.zeros(3, dtype=torch.float64)
    directions = random_unit_directions(3, 2, seed=8)
    positions = torch.linspace(-1, 1, 201, dtype=torch.float64)
    profiles, _ = estimator.line_profiles(center, directions, positions * 2)
    actual, error, _ = estimator.adaptive_log_averages(center, directions, 2., positions, profiles)
    dense, _ = estimator.line_profiles(center, directions,
                                      torch.linspace(-2, 2, 20001, dtype=torch.float64))
    torch.testing.assert_close(actual, log_average_density(dense), rtol=0, atol=2e-5)
    assert error < 1e-4


def test_low_dimensional_knn_neighbor_switches_match_direct_query_integral():
    # The kNN radius is continuous but has derivative changes at neighbour swaps.
    # This fixed 3D line triggers QUADPACK roundoff with only 16 initial bins.
    data = torch.load(Path(__file__).parent/'fixtures/pc1_density_knn3d_roundoff.pt', weights_only=True)
    estimator = KNNLogDensity(data['reference'], k=data['k'])
    center, direction, length = data['center'], data['direction'], data['half_length']
    positions = torch.linspace(-1, 1, 201, dtype=torch.float64)
    profiles, _ = estimator.line_profiles(center, direction[None], length*positions)
    actual, error, _ = estimator.adaptive_log_averages(center, direction[None], length, positions, profiles)
    offsets = torch.linspace(-length, length, 50001, dtype=torch.float64)
    dense = []
    for chunk in offsets.split(1000):
        queries = center+chunk[:, None]*direction
        radii = torch.cdist(queries, data['reference'], compute_mode='donot_use_mm_for_euclid_dist').kthvalue(data['k']).values
        dense.append(estimator.log_constant-3*radii.log())
    expected = log_average_density(torch.cat(dense))
    torch.testing.assert_close(actual[0], expected, atol=5e-5, rtol=0)
    assert error <= 1e-4


def test_global_pca_is_training_centered_and_selects_minimal_variance_prefix():
    mean = torch.tensor([7., -3., 10., 2.], dtype=torch.float64)
    variances = torch.tensor([9., 4., 1., .1], dtype=torch.float64)
    axes = torch.diag((variances * 7 / 2).sqrt())
    training = torch.cat([axes, -axes]) + mean
    pca = fit_global_variance_pca([training[:3], training[3:]], .75)
    torch.testing.assert_close(pca['mean'], mean)
    torch.testing.assert_close(pca['eigenvalues'], variances)
    assert pca['dimension'] == 2
    assert pca['previous_variance_fraction'] < .75 <= pca['retained_variance_fraction']
    torch.testing.assert_close(pca['basis'].T @ pca['basis'], torch.diag(torch.tensor([1., 1., 0., 0.], dtype=torch.float64)))
    torch.testing.assert_close(pca['basis'] @ pca['basis'].T, torch.eye(2, dtype=torch.float64))
    assert pca['standardization'] is False and pca['whitening'] is False
    assert pca['training_count'] == 8


def test_projected_segment_preserves_original_endpoints_and_alpha():
    basis = torch.tensor([[1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    mean = torch.tensor([3., -2., 1.], dtype=torch.float64)
    geometry = dict(centers=torch.tensor([[7., 5., 6.]], dtype=torch.float64),
                    vectors=torch.tensor([[math.sqrt(.75), .5, 0.]], dtype=torch.float64),
                    eigenvalues=torch.tensor([9.], dtype=torch.float64), exclusion_reasons=[''])
    projected = project_pc1_segments(geometry, mean, basis)
    a = NormalDist().inv_cdf(.9) * 3
    assert float(projected['alpha'][0]) == pytest.approx(.75)
    assert float(projected['half_lengths'][0]) == pytest.approx(a * math.sqrt(.75))
    for sign in [-1, 1]:
        original_endpoint = geometry['centers'][0] + sign * a * geometry['vectors'][0]
        projected_endpoint = projected['centers'][0] + sign * projected['half_lengths'][0] * projected['vectors'][0]
        torch.testing.assert_close(projected_endpoint, basis @ (original_endpoint - mean))
    assert float(geometry['eigenvalues'][0]) == 9


def test_null_projection_is_explicitly_excluded():
    geometry = dict(centers=torch.zeros(1, 3, dtype=torch.float64),
                    vectors=torch.tensor([[0., 1., 0.]], dtype=torch.float64),
                    eigenvalues=torch.ones(1, dtype=torch.float64), exclusion_reasons=[''])
    projected = project_pc1_segments(geometry, torch.zeros(3),
                                    torch.tensor([[1., 0., 0.], [0., 0., 1.]]))
    assert projected['exclusion_reasons'] == ['null_pc1_projection']
    assert projected['alpha'][0] == 0


def test_baseline_is_unit_isotropic_in_reduced_space_at_projected_length():
    class Capture:
        dimension = 2

        def line_profiles(self, center, directions, offsets):
            self.directions, self.offsets = directions, offsets
            return torch.zeros(len(directions), len(offsets)), torch.ones(len(directions), len(offsets))

    estimator = Capture()
    projected_length = .25
    result = score_component(estimator, torch.zeros(2), torch.tensor([1., 0.]), 100.,
                             half_length=projected_length, seed=123)
    assert result['half_length'] == projected_length
    torch.testing.assert_close(estimator.directions[1:], random_unit_directions(2, 32, 123))
    radii = (estimator.directions * estimator.offsets[-1]).norm(dim=1)
    torch.testing.assert_close(radii, torch.full((33,), projected_length, dtype=torch.float64))


def test_knn_density_uses_reduced_ball_volume_including_one_dimension():
    for dimension, volume in [(1, 2.), (2, math.pi)]:
        estimator = KNNLogDensity(torch.ones(20, dimension, dtype=torch.float64), k=20)
        actual = estimator.from_squared_radii(torch.tensor([4.], dtype=torch.float64))
        assert float(actual[0]) == pytest.approx(-math.log(volume) - dimension * math.log(2))


def test_projected_density_average_matches_original_parameter_integral():
    class ExponentialDensity:
        dimension = 2

        def line_profiles(self, center, directions, offsets):
            x = center[0] + directions[:, 0, None] * offsets[None, :]
            logs = 5 + .7 * x
            return logs, torch.ones_like(logs)

    basis = torch.tensor([[1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    geometry = dict(centers=torch.tensor([[2., 3., 4.]], dtype=torch.float64),
                    vectors=torch.tensor([[.6, .8, 0.]], dtype=torch.float64),
                    eigenvalues=torch.tensor([4.], dtype=torch.float64), exclusion_reasons=[''])
    projected = project_pc1_segments(geometry, torch.zeros(3), basis)
    result = score_component(ExponentialDensity(), projected['centers'][0], projected['vectors'][0],
                             4., half_length=float(projected['half_lengths'][0]))
    original_a = NormalDist().inv_cdf(.9) * 2
    # (1/(2a)) integral[-a,a] exp(.7 * .6 * t) dt.
    argument = .7 * .6 * original_a
    expected = math.sinh(argument) / argument
    assert math.exp(result['log_retention']) == pytest.approx(expected, rel=2e-5)
