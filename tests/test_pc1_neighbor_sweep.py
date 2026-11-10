"""Per-k recalibration, PC1-only scoring and q90/k20 reproduction."""
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
import run_pc1_neighbor_sweep as sweep
import run_pc1_percentile_sweep as percentile


def test_each_neighbor_has_its_own_q90_calibration():
    rng = np.random.default_rng(14)
    reference, queries = rng.normal(size=(80, 3)), rng.normal(size=(20, 3))
    scores, thresholds = sweep.calibrate_neighbors(reference, queries)
    direct = np.sort(cdist(queries, reference), axis=1)[:, [9, 19, 39]].T
    np.testing.assert_array_equal(scores, direct)
    np.testing.assert_array_equal(thresholds, np.quantile(direct, .90, axis=1, method='higher'))
    assert np.all(np.diff(thresholds) >= 0)
    assert np.all(np.diff(scores, axis=0) >= 0)
    with pytest.raises(ValueError):
        sweep.calibrate_neighbors(reference[:39], queries)


def test_common_grid_masks_use_correct_k_for_every_axis():
    rng = np.random.default_rng(25)
    reference = rng.normal(size=(90, 3))
    _, thresholds = sweep.calibrate_neighbors(reference, rng.normal(size=(30, 3)))
    sweep.initialize_worker(reference, thresholds)
    result = sweep.score_neighbors(np.zeros(3), np.array([1., 0., 0.]), 3., 7)
    masks = np.unpackbits(result['acceptance_mask_packed'], axis=2)[:, :, :result['mask_points']].astype(bool)
    values = masks.mean(axis=2)
    np.testing.assert_array_equal(values, result['direction_coverage'])
    np.testing.assert_array_equal(result['coverage_random'], values[:, 1:].mean(axis=1))
    np.testing.assert_array_equal(result['delta_coverage'], values[:, 0]-values[:, 1:].mean(axis=1))
    indices = np.linspace(0, result['mask_points']-1, 31, dtype=int)
    offsets = np.linspace(-3, 3, result['mask_points'])[indices]
    queries = (result['directions'][:, None]*offsets[None, :, None]).reshape(-1, 3)
    sorted_distances = np.sort(cdist(queries, reference), axis=1)
    for i, k in enumerate([10, 20, 40]):
        direct = (sorted_distances[:, k-1] <= thresholds[i]).reshape(33, 31)
        np.testing.assert_array_equal(masks[i][:, indices], direct)
    assert result['max_grid_error'].max() <= .002


def test_k20_anchor_is_q90_not_q95():
    rng = np.random.default_rng(16)
    reference, queries = rng.normal(size=(80, 3)), rng.normal(size=(30, 3))
    scores, thresholds = sweep.calibrate_neighbors(reference, queries)
    percentile.initialize_worker(reference, percentile.percentile_thresholds(scores[1]))
    saved = sweep.base.as_tensors(percentile.score_percentiles(np.zeros(3), np.array([1., 0., 0.]), 2., 0))
    sweep.initialize_worker(reference, thresholds)
    result = sweep.score_neighbors(np.zeros(3), np.array([1., 0., 0.]), 2., 0)
    assert sweep.validate_k20(result, saved) == 0
    wrong = dict(result, exact_coverage=result['exact_coverage'].copy())
    wrong['exact_coverage'][1] += .01
    with pytest.raises(ValueError, match='continuous'):
        sweep.validate_k20(wrong, saved)


def test_recover_queries_preserves_full_pool_identity_and_reference_order(monkeypatch, tmp_path):
    pool = torch.arange(180, dtype=torch.float64).reshape(60, 3)
    path = tmp_path/'pca.pt'
    torch.save({}, path)
    checksum = hashlib.sha256(pool.numpy().tobytes()).hexdigest()
    metadata = dict(reference_global_rows=list(range(100, 140)), calibration_global_rows=list(range(140, 160)))
    previous = dict(baseline_manifest=dict(reference_global_rows=list(range(100, 160)), global_pca=dict(path=str(path))),
        pca_sha256=sweep.sha256(path), reference_provenance=dict(reference_sha256=checksum))
    calibration = dict(reference=pool[:40].clone(), calibration=metadata)
    monkeypatch.setattr(sweep.base, 'reference_from_manifest', lambda *args: (pool, {'reference_sha256': checksum}))
    queries, _ = sweep.recover_queries(previous, calibration)
    np.testing.assert_array_equal(queries, pool[40:].numpy())
    corrupted = dict(previous, reference_provenance=dict(reference_sha256='different'))
    with pytest.raises(ValueError, match='full evaluation pool'):
        sweep.recover_queries(corrupted, calibration)


def test_four_case_end_to_end_with_fixed_calibration_and_anchor(monkeypatch, tmp_path):
    class InlinePool:
        def __init__(self, *, initializer, initargs, **kwargs):
            initializer(*initargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, *args):
            f = Future()
            f.set_result(function(*args))
            return f
    monkeypatch.setattr(sweep, 'ProcessPoolExecutor', InlinePool)
    monkeypatch.setenv('SLURM_JOB_ID', 'fixture')
    baseline_root, anchor_root = tmp_path/'baseline', tmp_path/'anchor'
    monkeypatch.setattr(sweep, 'BASELINE', baseline_root)
    monkeypatch.setattr(sweep, 'ANCHOR', anchor_root)
    source = tmp_path/'geometry'
    source.mkdir()
    rng = np.random.default_rng(20)
    reference, queries = rng.normal(size=(80, 3)), rng.normal(size=(20, 3))
    scores, thresholds = sweep.calibrate_neighbors(reference, queries)
    old_thresholds = percentile.percentile_thresholds(scores[1])
    percentile.initialize_worker(reference, old_thresholds)
    old_result = sweep.base.as_tensors(percentile.score_percentiles(np.zeros(3), np.array([1., 0., 0.]), 2., 0))
    monkeypatch.setattr(sweep, 'recover_queries', lambda *args: (queries, {'fixture': True}))
    calibration = dict(k=20, tau=float(old_thresholds[-1]), quantile=.95, reference_global_rows=list(range(80)),
                       calibration_global_rows=list(range(80, 100)))
    configs = [dict(run_id=f'{kind}_{k}_{t}', kind=kind, K=k, threshold=t)
        for kind in ['hddc', 'kmeans'] for k in [50, 100] for t in [.0001, .001, .01, .1]]
    sweep.write_csv(source/'summary.csv', configs)
    sweep.write_csv(source/'components.csv', [dict(run_id=r['run_id'], component=0, exclusion_reason='') for r in configs])
    geom = dict(centers=torch.zeros((1, 3), dtype=torch.float64), vectors=torch.tensor([[1., 0., 0.]], dtype=torch.float64),
                half_lengths=torch.tensor([2.]), alpha=torch.ones(1))
    fingerprints = {}
    for config in configs:
        path = source/(config['run_id']+'.pt')
        torch.save(dict(projected_geometry=geom, components={0: {}}), path)
        fingerprints[config['run_id']] = sweep.sha256(path)
    sources = [(d, label, layer, source, None) for d, label, layer in [
        ('years', 'Years', 4), ('years', 'Years', 28), ('temperature', 'Temperature', 4), ('temperature', 'Temperature', 28)]]
    monkeypatch.setattr(sweep, 'SOURCES', sources)
    for dataset, _, layer, _, _ in sources:
        old, anchor = baseline_root/f'{dataset}_l{layer:02}', anchor_root/f'{dataset}_l{layer:02}'
        old.mkdir(parents=True)
        anchor.mkdir(parents=True)
        previous = dict(calibration=calibration, reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
            geometry_bundle_sha256=fingerprints, baseline_manifest=dict(dimension=3, global_pca=dict(retained_variance_fraction=.8)))
        (old/'manifest.json').write_text(json.dumps(previous))
        (old/'COMPLETED.json').write_text(json.dumps(dict(completed=True)))
        torch.save(dict(reference=torch.from_numpy(reference), calibration_scores=torch.from_numpy(scores[1]), calibration=calibration), old/'calibration.pt')
        sweep.write_csv(old/'components.csv', [dict(run_id=r['run_id'], component=0) for r in configs])
        (anchor/'manifest.json').write_text(json.dumps(dict(quantiles=[.8, .9, .95], thresholds=old_thresholds.tolist())))
        (anchor/'COMPLETED.json').write_text(json.dumps(dict(completed=True)))
        for config in configs:
            torch.save(dict(components={0: old_result}), anchor/(config['run_id']+'.pt'))
    root = tmp_path/'result'
    for index in range(4):
        sweep.run_case(index, root, 1, False)
    sweep.collect(root)
    assert json.loads((root/'COMPLETED.json').read_text())['configurations'] == 192
    assert len(sweep.read_csv(root/'tabelle_soglia_01.csv')) == 48
    for k in [10, 20, 40]:
        assert (root/f'k{k}/03_pannello_medie.png').stat().st_size > 0
    assert (root/'01_sensibilita_vicini.pdf').stat().st_size > 0
