"""Scientific and numerical checks for the temporary calibrated support metric."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
import run_pc1_support_coverage as coverage


def test_disjoint_canonical_split_and_higher_quantile():
    rng = np.random.default_rng(9)
    points = rng.normal(size=(103, 3))
    rows = rng.permutation(np.arange(103)+1000)
    ref, scores, metadata = coverage.split_and_calibrate(points, rows)
    reference_ids, query_ids = metadata['reference_global_rows'], metadata['calibration_global_rows']
    assert len(ref) == 82 and len(query_ids) == 21
    assert not set(reference_ids)&set(query_ids)
    assert set(reference_ids)|set(query_ids) == set(rows)
    mapping = {row: point for row, point in zip(rows, points)}
    direct = np.sort(cdist(np.array([mapping[row] for row in query_ids]), ref), axis=1)[:, 19]
    np.testing.assert_allclose(scores, direct)
    assert metadata['tau'] == np.quantile(direct, .95, method='higher')
    reverse = np.arange(102, -1, -1)
    ref2, scores2, metadata2 = coverage.split_and_calibrate(points[reverse], rows[reverse])
    np.testing.assert_array_equal(ref, ref2)
    np.testing.assert_array_equal(scores, scores2)
    assert metadata == metadata2


@pytest.mark.parametrize('dimension', [1, 3, 28, 141])
def test_interval_masks_equal_direct_knn_including_center_and_endpoints(dimension):
    rng = np.random.default_rng(dimension)
    ref = rng.normal(size=(70, dimension))
    center = rng.normal(size=dimension)
    direction = rng.normal(size=dimension)
    direction /= np.linalg.norm(direction)
    tau = float(np.median(np.linalg.norm(ref-center, axis=1)))
    starts, ends, intervals, exact = coverage.supported_intervals(ref, center, direction, tau, 3.)
    offsets = np.linspace(-3, 3, 2001)
    mask = coverage.grid_mask(starts, ends, offsets)
    direct = (cdist(center+offsets[:, None]*direction, ref) <= tau).sum(axis=1) >= 20
    np.testing.assert_array_equal(mask, direct)
    assert abs(mask.mean()-exact) < .003
    assert 0 <= exact <= 1
    assert intervals.shape[1] == 2


def test_closed_endpoints_tangency_and_zero_continuous_length():
    ref = np.tile([0., 1.], (20, 1))
    starts, ends, _, exact = coverage.supported_intervals(ref, np.zeros(2), np.array([1., 0.]), 1., 2.)
    np.testing.assert_array_equal(coverage.grid_mask(starts, ends, np.array([-1., 0., 1.])), [False, True, False])
    assert exact == 0
    ref = np.zeros((20, 1))
    starts, ends, _, exact = coverage.supported_intervals(ref, np.zeros(1), np.ones(1), 1., 2.)
    np.testing.assert_array_equal(coverage.grid_mask(starts, ends, np.array([-1.001, -1., 0., 1., 1.001])),
                                  [False, True, True, True, False])
    assert exact == .5


def test_empty_and_full_support_and_dense_patch_cannot_dominate():
    for x, expected in [(100., 0.), (0., 1.)]:
        reference = np.full((20, 1), x)
        _, _, _, exact = coverage.supported_intervals(reference, np.zeros(1), np.ones(1), 2., 1.)
        assert exact == expected
    for multiplicity in [20, 2000]:
        reference = np.zeros((multiplicity, 1))
        _, _, _, exact = coverage.supported_intervals(reference, np.zeros(1), np.ones(1), .05, 1.)
        assert exact == pytest.approx(.05)


def test_known_straight_manifold_tangent_rotation():
    reference = np.column_stack([np.linspace(-3, 3, 1200), np.zeros(1200)])
    values = []
    for theta in [0, .2, .5, 1., np.pi/2]:
        direction = np.array([np.cos(theta), np.sin(theta)])
        values.append(coverage.supported_intervals(reference, np.zeros(2), direction, .15, 1.)[3])
    assert values[0] == pytest.approx(1.)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[-1] < .15


def test_scores_masks_random_baseline_and_tensor_round_trip(tmp_path):
    rng = np.random.default_rng(44)
    ref = rng.normal(size=(100, 3))
    coverage.initialize_worker(ref, 1.2)
    result = coverage.score_component(np.zeros(3), np.array([1., 0., 0.]), 2., 7)
    masks = np.unpackbits(result['acceptance_mask_packed'], axis=1)[:, :result['mask_points']].astype(bool)
    values = masks.mean(axis=1)
    np.testing.assert_array_equal(result['direction_coverage'], values)
    assert result['coverage_pc1'] == values[0]
    assert result['coverage_random'] == values[1:].mean()
    assert result['delta_coverage'] == values[0]-values[1:].mean()
    assert result['max_grid_error'] <= .002
    assert result['max_refinement_change'] <= .002
    assert result['center_supported'] == masks[0, result['mask_points']//2]
    np.testing.assert_allclose(result['directions'][1:], coverage.random_unit_directions(3, 32, 49).numpy())
    path = tmp_path/'component.pt'
    torch.save(coverage.as_tensors(result), path)
    saved = torch.load(path, weights_only=True)
    np.testing.assert_array_equal(saved['acceptance_mask_packed'].numpy(), result['acceptance_mask_packed'])


def test_saved_geometry_contract_and_four_table_export(monkeypatch, tmp_path):
    """Exercise source loading, original lengths, pooling, aggregation and export."""
    from concurrent.futures import Future
    import json

    class InlinePool:
        def __init__(self, *, initializer, initargs, **kwargs):
            initializer(*initargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

    monkeypatch.setattr(coverage, 'ProcessPoolExecutor', InlinePool)
    monkeypatch.setenv('SLURM_JOB_ID', 'fixture')
    source = tmp_path/'source'
    source.mkdir()
    pca = source/'pca.pt'
    torch.save(dict(mean=torch.zeros(2), basis=torch.eye(2)), pca)
    rows = list(range(100, 160))
    baseline = dict(dimension=2, reference_global_rows=rows,
                    global_pca=dict(path=str(pca), sha256=coverage.sha256(pca), retained_variance_fraction=.8))
    (source/'manifest.json').write_text(json.dumps(baseline))
    (source/'COMPLETED.json').write_text(json.dumps(dict(completed=True, smoke=False)))
    configs, components = [], []
    for kind in ['hddc', 'kmeans']:
        for k in [50, 100]:
            for threshold in [.0001, .001, .01, .1]:
                run_id = f'{kind}_{k}_{threshold}'
                configs.append(dict(run_id=run_id, kind=kind, K=k, threshold=threshold))
                components.append(dict(run_id=run_id, component=0, exclusion_reason=''))
                geom = dict(centers=torch.zeros((1, 2), dtype=torch.float64),
                            vectors=torch.tensor([[1., 0.]], dtype=torch.float64),
                            half_lengths=torch.ones(1), original_half_lengths=torch.tensor([1.5]), alpha=torch.ones(1))
                # Source ambient geometry intentionally has no 'half_lengths'.
                torch.save(dict(projected_geometry=geom, geometry=dict(eigenvalues=torch.ones(1)), components={0: {}}),
                           source/(run_id+'.pt'))
    coverage.write_csv(source/'summary.csv', configs)
    coverage.write_csv(source/'components.csv', components)
    points = torch.tensor(np.random.default_rng(4).normal(size=(60, 2)), dtype=torch.float64)
    monkeypatch.setattr(coverage, 'reference_from_manifest', lambda *args: (points, {'fixture': True}))
    sources = [(dataset, label, layer, source, None) for dataset, label, layer in [
        ('years', 'Years', 4), ('years', 'Years', 28), ('temperature', 'Temperature', 4), ('temperature', 'Temperature', 28)]]
    monkeypatch.setattr(coverage, 'SOURCES', sources)
    root = tmp_path/'result'
    for index in range(4):
        coverage.run_case(index, root, 1)
    coverage.collect(root)
    assert json.loads((root/'COMPLETED.json').read_text())['completed']
    assert len(coverage.read_csv(root/'tabella_soglia_01.csv')) == 16
    assert len(coverage.read_csv(root/'tutte_le_64_configurazioni.csv')) == 64
    component_rows = coverage.read_csv(root/'years_l04/components.csv')
    assert all(float(row['original_half_length']) == 1.5 for row in component_rows)
    for extension in ['png', 'pdf', 'svg']:
        assert (root/f'02_pannello_tabelle.{extension}').stat().st_size > 0


def test_tied_table_scores_do_not_highlight_a_winner(monkeypatch, tmp_path):
    import matplotlib.figure
    saved = []
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', lambda fig, *args, **kwargs: saved.append(fig))
    selected = [dict(K=k, kind=kind, method=kind, retention=1., enrichment=0., eligible=k, total=k)
                for k in [50, 100] for kind in ['hddc', 'kmeans']]
    entry = dict(label='Fixture', layer=4, dimension=3, variance=.8, selected=selected)
    coverage.render_tables([entry], tmp_path, 'fixture', 'Medie')
    figure = saved[0]
    table = figure.axes[0].tables[0]
    for row in range(1, 5):
        for column in [2, 3]:
            assert table[row, column].get_text().get_fontweight() == 'normal'
    assert 'Medie' in figure._suptitle.get_text()
    assert any('medie in scala ordinaria' in text.get_text() for text in figure.texts)
