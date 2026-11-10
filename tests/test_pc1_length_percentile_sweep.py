"""Gaussian segment mass and fixed-calibration factorial coverage."""
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
import run_pc1_length_percentile_sweep as sweep


def test_segment_mass_is_gaussian_probability_not_variance_fraction():
    assert sweep.length_factor(.8) == 1.
    for mass in [.9, .95, .99]:
        z = NormalDist().inv_cdf(.9)*sweep.length_factor(mass)
        assert 2*NormalDist().cdf(z)-1 == pytest.approx(mass)
    assert sweep.length_factor(.9) == pytest.approx(1.2834861048)
    assert sweep.length_factor(.99) == pytest.approx(2.009930285133935)
    with pytest.raises(ValueError):
        sweep.length_factor(1.)


def test_exact_ball_support_matches_each_mass_and_percentile():
    reference = np.zeros((30, 2))
    # Keep direct-audit positions away from exact floating-point ball boundaries.
    taus = np.array([.413, .617, 1.003])
    sweep.percentile.initialize_worker(reference, taus)
    result = sweep.score_lengths(np.zeros(2), np.array([1., 0.]), .8, 7)
    for index, mass in enumerate(sweep.MASSES):
        part = result['by_mass'][index]
        length = .8*sweep.length_factor(mass)
        expected = np.minimum(1., taus/length)
        np.testing.assert_allclose(part['exact_coverage'], np.broadcast_to(expected[:, None], (3, 33)), atol=1e-12)
        masks = np.unpackbits(part['acceptance_mask_packed'], axis=2)[:, :, :part['mask_points']].astype(bool)
        assert not np.any(masks[:-1] & ~masks[1:])
        np.testing.assert_array_equal(part['direction_coverage'], masks.mean(axis=2))
        assert np.all(part['center_supported'])
        assert part['max_grid_error'].max() <= .002
        np.testing.assert_array_equal(part['directions'], result['control']['directions'])


def test_mass80_control_reproduces_all_prior_percentiles():
    reference = np.random.default_rng(4).normal(size=(80, 3))
    sweep.percentile.initialize_worker(reference, [1., 1.5, 2.])
    center, vector = np.zeros(3), np.array([1., 0., 0.])
    saved = sweep.base.as_tensors(sweep.percentile.score_percentiles(center, vector, 2., 0))
    result = sweep.score_lengths(center, vector, 2., 0)
    assert sweep.validate_control(result['control'], saved) == 0
    changed = dict(result['control'], exact_coverage=result['control']['exact_coverage'].copy())
    changed['exact_coverage'][0, 0] += .01
    with pytest.raises(ValueError, match='continuous'):
        sweep.validate_control(changed, saved)


def test_four_case_factorial_export_and_original_geometry_contract(monkeypatch, tmp_path):
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
    reference = np.random.default_rng(19).normal(size=(80, 3))
    scores = np.linspace(1., 2., 20)
    thresholds = sweep.percentile.percentile_thresholds(scores)
    sweep.percentile.initialize_worker(reference, thresholds)
    old_result = sweep.base.as_tensors(sweep.percentile.score_percentiles(np.zeros(3), np.array([1., 0., 0.]), 2., 0))
    calibration = dict(k=20, tau=float(thresholds[-1]), seed=42, quantile=.95,
                       reference_global_rows=list(range(80)), calibration_global_rows=list(range(80, 100)))
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
            geometry_bundle_sha256=fingerprints, baseline_manifest=dict(probability=.80, dimension=3,
                global_pca=dict(retained_variance_fraction=.8)))
        (old/'manifest.json').write_text(json.dumps(previous))
        (old/'COMPLETED.json').write_text(json.dumps(dict(completed=True)))
        torch.save(dict(reference=torch.from_numpy(reference), calibration_scores=torch.from_numpy(scores), calibration=calibration), old/'calibration.pt')
        sweep.write_csv(old/'components.csv', [dict(run_id=r['run_id'], component=0) for r in configs])
        (anchor/'manifest.json').write_text(json.dumps(dict(thresholds=thresholds.tolist())))
        (anchor/'COMPLETED.json').write_text(json.dumps(dict(completed=True)))
        for config in configs:
            torch.save(dict(components={0: old_result}), anchor/(config['run_id']+'.pt'))
    root = tmp_path/'result'
    for index in range(4):
        sweep.run_case(index, root, 1, False)
    sweep.collect(root)
    assert json.loads((root/'COMPLETED.json').read_text())['configurations'] == 576
    assert len(sweep.read_csv(root/'tabelle_soglia_01.csv')) == 144
    rows = sweep.read_csv(root/'years_l04/components.csv')
    assert len(rows) == 16*9
    for row in rows:
        assert float(row['half_length']) == pytest.approx(2*sweep.length_factor(float(row['segment_mass'])))
    for mass in [90, 95, 99]:
        for q in [80, 90, 95]:
            assert (root/f'm{mass}_q{q}/03_pannello_medie.png').stat().st_size > 0
