"""Check a completed surface fit against raw rows and an independent QR solve."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import lstsq

import fit_years_surface as sheet
from fit_years_spiral import load_activations, sha256, split_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    out = parser.parse_args().output
    if 'SLURM_STEP_ID' not in os.environ:
        raise RuntimeError('Run through srun')
    manifest = json.loads((out / 'manifest.json').read_text())
    results = json.loads((out / 'results.json').read_text())
    assert sha256(out / 'source/fit_years_surface.py') == manifest['source_hash']
    comparison_source = out / 'source/fit_years_surface_comparison.py'
    if not comparison_source.exists():
        comparison_source = out / 'source/fit_years_surface.py'
    assert sha256(comparison_source) == results['comparison_source_hash']
    assert all(sha256(path) == digest for path, digest in manifest['input_hashes'].items())
    assert all(sha256(path) == digest for path, digest in results['gaussian_hashes'].items())
    x, years, templates, index = load_activations(Path(manifest['activation_dir']))
    x = x.astype(float)
    masks, membership = split_development(years, templates)
    assert membership == manifest['split']
    assert [r['source_row_id'] for r in index] == manifest['source_row_ids']
    evaluated, _ = sheet.load_surface(out / 'evaluation_surface.npz')
    discrepancies = []
    for name, expected in results['evaluation'].items():
        observed = sheet.score(evaluated, x[masks[name]], x[masks['fit']].mean(0))
        discrepancies.append(abs(observed['mse'] - expected['mse']))
        assert discrepancies[-1] < 1e-12
    final, uv = sheet.load_surface(out / 'final_surface_all_development.npz')
    n, smoothing = final['n'], final['smoothing']
    design = sheet.features(uv, n).toarray()
    # Independent augmented QR for 24 ambient coordinates, including the exact
    # bending regularizer; the saved solve uses positive-definite normal equations.
    values, vectors = np.linalg.eigh(sheet.penalty(n))
    regularizer = np.sqrt(np.maximum(values, 0))[:, None] * vectors.T
    augmented = np.vstack((design, np.sqrt(len(x) * smoothing) * regularizer))
    dimensions = np.random.default_rng(42).choice(x.shape[1], 24, replace=False)
    response = np.vstack((x[:, dimensions] - final['origin'][dimensions], np.zeros((n*n, 24))))
    independent, _, rank, _ = lstsq(augmented, response, lapack_driver='gelsy')
    error = float(np.max(np.abs(independent - final['coef'][:, dimensions])))
    assert error < 1e-7, error
    sample = np.random.default_rng(7).uniform(.05, .95, (30, 2))
    derivative_error = 0.
    for axis in (0, 1):
        delta = np.eye(2)[axis] * 1e-6
        finite = (sheet.predict(final, sample + delta) - sheet.predict(final, sample - delta)) / 2e-6
        derivative_error = max(derivative_error, float(np.max(np.abs(finite - sheet.predict(final, sample, axis)))))
    assert derivative_error < 1e-7
    with (out / 'gaussian_alignment.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 600
    for summary in results['gaussians']:
        valid = [r for r in rows if r['run'] == summary['run'] and r['valid'] == 'True']
        assert len(valid) == summary['valid']
        if valid:
            assert abs(np.mean([float(r['overlap']) for r in valid]) - summary['mean_overlap']) < 1e-12
    audit = dict(source_and_input_hashes_match=True, source_row_ids_and_splits_match=True,
                 reconstruction_checks=len(discrepancies), maximum_mse_discrepancy=max(discrepancies),
                 independent_augmented_qr_coordinates=24, qr_rank=int(rank), maximum_coefficient_discrepancy=error,
                 maximum_analytic_derivative_discrepancy=derivative_error,
                 gaussian_rows_checked=len(rows), slurm_job=os.environ.get('SLURM_JOB_ID'))
    (out / 'artifact_audit.json').write_text(json.dumps(audit, indent=2))
    shutil.copy2(__file__, out / 'source' / Path(__file__).name)
    shutil.copy2(Path(__file__).parents[2] / 'tests/test_years_surface.py', out / 'source/test_years_surface.py')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
