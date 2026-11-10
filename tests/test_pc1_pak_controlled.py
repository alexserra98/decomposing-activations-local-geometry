"""Controlled PAk dimension, query independence, and integration contracts."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts/temporary'
sys.path.insert(0, str(SCRIPTS))
import run_pc1_pak_controlled as pilot


def test_fixed_dimension_and_query_batch_independence(monkeypatch):
    pytest.importorskip('dadapy')
    from dadapy.density_estimation import DensityEstimation

    def forbidden(*args, **kwargs):
        raise AssertionError('Controlled PAk must never estimate intrinsic dimension')

    monkeypatch.setattr(DensityEstimation, 'compute_id_2NN', forbidden)
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(150, 4))
    queries = rng.normal(size=(7, 4))
    estimator = pilot.PAkDensity(reference, maxk=30)
    assert estimator.data.intrinsic_dim == 4
    values, errors = estimator.query(queries)
    chunks = [estimator.query(q[None]) for q in queries]
    torch.testing.assert_close(values, torch.cat([c[0] for c in chunks]), rtol=0, atol=1e-10)
    torch.testing.assert_close(errors, torch.cat([c[1] for c in chunks]), rtol=0, atol=1e-10)
    assert np.array_equal(estimator.data.X, reference)


def test_integral_matches_gaussian_and_arithmetic_random_baseline(monkeypatch):
    class Gaussian:
        def query(self, x):
            x = torch.as_tensor(x)
            return -10000-.5*(x/torch.tensor([.3, 2.])).square().sum(1), torch.ones(len(x))

    monkeypatch.setattr(pilot, 'DENSITY', Gaussian(), raising=False)
    center = torch.zeros(2, dtype=torch.float64)
    vector = torch.tensor([0., 1.], dtype=torch.float64)
    result = pilot.score(center, vector, 1., component=7)
    directions = torch.cat([vector[None], pilot.random_unit_directions(2, 32, 49)])
    sigma = (directions/torch.tensor([.3, 2.])).square().sum(1).rsqrt()
    exact = math.sqrt(2*math.pi)*sigma*torch.erf(1/(math.sqrt(2)*sigma))/2
    expected = exact.log()
    torch.testing.assert_close(result['log_direction_retentions'], expected, atol=1e-5, rtol=0)
    assert result['log_enrichment'] == pytest.approx(float((exact[0]/exact[1:].mean()).log()), abs=1e-5)
    assert result['quadrature_max_evaluations'] >= 801
    assert result['log_relative_density_profiles'].shape == (33, 201)


def test_integration_refuses_unverified_result(monkeypatch):
    class Gaussian:
        def query(self, x):
            return -torch.as_tensor(x).square().sum(1), torch.ones(len(x))

    monkeypatch.setattr(pilot, 'DENSITY', Gaussian(), raising=False)
    with pytest.raises(RuntimeError, match='did not converge'):
        pilot.score(torch.zeros(2, dtype=torch.float64), torch.tensor([1., 0.], dtype=torch.float64),
                    1., 0, max_evaluations=201)


def test_reference_restores_original_prompt_order_from_both_partitions(tmp_path):
    import json

    root = tmp_path/'activation'
    archive = tmp_path/'pre_holdout'/'activation'
    records = [dict(row_id=10+i, prompt=f'prompt {i}', tokenization={'ids': [i]}, subset='years') for i in range(4)]
    cfg = {key: 'same' for key in ['model', 'model_revision', 'tokenizer_revision', 'tokenizer_sha256',
           'representation', 'dtype', 'token_selection']}
    cfg.update(d_model=4096, window=1, drop_prefix=0, source_records_sha256='prompts')
    for destination, indices, tensors in [(archive, list(range(4)), False),
                                           (root, [3, 0], True), (root/'test', [2, 1], True)]:
        (destination/'source').mkdir(parents=True)
        (destination/'meta').mkdir()
        (destination/'config.json').write_text(json.dumps(cfg))
        (destination/'source/records.jsonl').write_text('\n'.join(json.dumps(records[i]) for i in indices))
        (destination/'meta/shard_00000.json').write_text(json.dumps(dict(
            row_indices=list(range(len(indices))), rows=[dict(subset='years') for i in indices])))
        if tensors:
            (destination/'layer04').mkdir()
            points = torch.stack([torch.full((1, 4096), float(i)) for i in indices])
            torch.save(points, destination/'layer04/shard_00000.pt')
    manifest = dict(activation_dir='/historical/activation', layer=4, reference_global_rows=[1, 3])
    pca = dict(source={'source_records_sha256': 'prompts'}, mean=torch.zeros(4096, dtype=torch.float64),
               basis=torch.eye(4096, dtype=torch.float64)[:2])
    actual, provenance = pilot.reference_from_manifest(manifest, pca, root)
    torch.testing.assert_close(actual, torch.tensor([[1., 1.], [3., 3.]], dtype=torch.float64))
    assert provenance['prompt_and_tokenization_identity_verified']
    assert not provenance['original_tensor_identity_verified']
    altered = records[1].copy()
    altered['prompt'] = 'changed'
    (root/'test/source/records.jsonl').write_text('\n'.join(json.dumps(r) for r in [records[2], altered]))
    with pytest.raises(ValueError, match='prompt/tokenization mismatch'):
        pilot.reference_from_manifest(manifest, pca, root)


def test_neighbor_switches_match_direct_lower_envelope():
    rng = np.random.default_rng(15)
    norms, slopes = rng.uniform(0, 5, 100), rng.normal(size=100)
    switches = pilot.nearest_neighbor_switches(norms, slopes)
    edges = np.r_[-1, switches, 1]
    for left, right in zip(edges[:-1], edges[1:]):
        samples = np.linspace(left, right, 11)[1:-1]
        nearest = (norms+samples[:, None]*slopes).argmin(1)
        assert len(set(nearest)) == 1
    for switch in switches:
        values = np.sort(norms+switch*slopes)
        assert values[0] == pytest.approx(values[1], abs=1e-12)


def test_adaptive_integral_resolves_narrow_peak_at_explicit_switch():
    sigma, center = 1e-6, .123456
    def log_density(x):
        return 100-.5*((x-center)/sigma)**2
    actual, error, _ = pilot.integrate_line(log_density, [center])
    exact = 100+math.log(math.sqrt(2*math.pi)*sigma/2)
    assert actual == pytest.approx(exact, abs=1e-4)
    assert error <= 1e-4


def test_line_polynomial_agrees_with_direct_pak_and_public_api():
    rng = np.random.default_rng(91)
    reference = rng.normal(size=(140, 4))
    estimator = pilot.PAkDensity(reference, maxk=40, minimum_k=3)
    center, direction = rng.normal(size=4), rng.normal(size=4)
    direction /= np.linalg.norm(direction)
    positions = np.linspace(-1, 1, 21)
    residual = reference-center
    values = estimator.line_query(np.square(residual).sum(1), residual@direction, .7, positions)
    queries = center+.7*positions[:, None]*direction
    direct, _ = estimator.data.return_interpolated_density_PAk(queries)
    # DADApy's Newton solve stops at gradient 1e-3; the profile root is tighter.
    np.testing.assert_allclose(values, direct, atol=1e-5, rtol=0)
    estimator.minimum_k = 4
    bounded = estimator.line_query(np.square(residual).sum(1), residual@direction, .7, positions)
    bounded_direct, _ = estimator.query(queries)
    np.testing.assert_allclose(bounded, bounded_direct, atol=1e-8, rtol=0)


def _send_array_result(index):
    return pilot.transport_result({f'tensor_{i}': torch.full((3,), float(index)) for i in range(5)})


def test_pool_results_do_not_accumulate_torch_file_descriptors():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    before = len(list(Path('/proc/self/fd').iterdir()))
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(_send_array_result, range(300)))
        assert all(isinstance(value, np.ndarray) for result in results for value in result.values())
        during = len(list(Path('/proc/self/fd').iterdir()))
        assert during-before < 40
    np.testing.assert_array_equal(results[-1]['tensor_0'], [299., 299., 299.])


def test_profile_likelihood_recovers_analytic_exponential_shells():
    dimension, kstar, intercept, slope = 4, 8, 2., .1
    volume = math.pi**2/2
    shells = np.exp(-intercept-slope*np.arange(1, kstar+1))
    radii = np.r_[.2, (.2**dimension+np.cumsum(shells)/volume)**(1/dimension)]
    actual, error = pilot.pak_profile_likelihood(radii[None], dimension, np.array([kstar]))
    assert actual[0] == pytest.approx(intercept, abs=1e-11)
    assert error[0] == pytest.approx(math.sqrt((4*(kstar-1)+2)/((kstar-1)*(kstar-2))))


def test_profile_likelihood_handles_real_newton_failure():
    import json
    from scipy.optimize import minimize
    from scipy.special import gammaln

    fixture = json.loads((Path(__file__).parent/'fixtures/pc1_pak_nonfinite.json').read_text())
    radii = np.array(fixture['radii'])
    dimension, count = fixture['dimension'], fixture['kstar']
    actual, _ = pilot.pak_profile_likelihood(radii[None], dimension, np.array([count]))
    assert np.isfinite(actual).all()
    # An independent two-parameter optimization checks the profiled solution.
    indices = np.arange(1, count+1)
    log_volumes = (dimension/2*math.log(math.pi)-gammaln(dimension/2+1)
        +dimension*np.log(radii[1:])+np.log1p(-(radii[:-1]/radii[1:])**dimension))
    def objective(parameters):
        b, a = parameters
        mass = np.exp(log_volumes+b+a*indices)
        value = mass.sum()-count*b-indices.sum()*a
        gradient = np.array([mass.sum()-count, (mass*indices).sum()-indices.sum()])
        hessian = np.array([[mass.sum(), (mass*indices).sum()],
                            [(mass*indices).sum(), (mass*indices**2).sum()]])
        return value, gradient, hessian
    fitted = minimize(lambda x: objective(x)[0], [actual[0]+.2, -10.],
        jac=lambda x: objective(x)[1], hess=lambda x: objective(x)[2], method='trust-exact',
        options={'gtol': 1e-10})
    assert np.linalg.norm(objective(fitted.x)[1]) < 1e-6
    assert actual[0] == pytest.approx(fitted.x[0], abs=1e-6)


def test_open_quadrature_does_not_assign_mass_to_isolated_split_value():
    def log_density(points):
        return np.where(points == 0., 1000., -.5*points**2)
    actual, error, _ = pilot.integrate_line(log_density, [0.])
    expected = math.log(math.sqrt(2*math.pi)*math.erf(1/math.sqrt(2))/2)
    assert actual == pytest.approx(expected, abs=1e-5)
    assert error <= 1e-4
