"""Audit a completed years-curve experiment using raw rows and a direct QR fit."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import lstsq

import fit_years_spiral as curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    out = parser.parse_args().output
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run the audit through srun on a compute node')
    manifest = json.loads((out / 'manifest.json').read_text())
    report = json.loads((out / 'results.json').read_text())
    assert curve.sha256(out / 'source/fit_years_spiral.py') == manifest['script_sha256']
    root = Path(manifest['activation_dir'])
    assert all(curve.sha256(root / name) == digest for name, digest in manifest['inputs'].items())
    x, years, templates, index = curve.load_activations(root)
    masks, membership = curve.split_development(years, templates, manifest['seed'])
    assert membership == manifest['membership']
    for name, mask in masks.items():
        assert [index[i]['source_row_id'] for i in np.flatnonzero(mask)] == manifest['splits'][name]['source_row_ids']
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['winning_family'] == min(
        selection['families'], key=lambda f: selection['families'][f]['validation_mse'])
    models = {family: curve.load_model(out / f'{family}_evaluation_model.npz')
              for family in report['selection']}
    discrepancies = []
    for family, model in models.items():
        assert model['spec'] == selection['families'][family]['spec']
        for protocol in ('refit', 'year_evaluation', 'template_evaluation', 'year_template_evaluation'):
            mask = masks[protocol]
            residual = x[mask] - curve.predict(model, years[mask], templates[mask])
            mse = float(np.mean(residual ** 2))
            expected = report['scores'][family][protocol]['mse']
            assert abs(mse - expected) < 1e-10 * max(mse, 1e-12)
            discrepancies.append(abs(mse - expected))
    print('Direct MSE checks:', len(discrepancies), 'max difference:', max(discrepancies), flush=True)
    model = models[report['selected_family']]
    mask, test = masks['refit'], masks['year_evaluation']
    seen = np.array(model['seen_templates'])

    def design(take):
        return np.column_stack([curve.basis(years[take], model['spec']),
                                *[(templates[take] == s).astype(float) for s in seen[:-1]]])

    coef, _, rank, _ = lstsq(design(mask), x[mask].astype(float) - model['origin'],
                            lapack_driver='gelsy')
    independent = model['origin'] + design(test) @ coef
    saved = curve.predict(model, years[test], templates[test])
    max_error = float(np.max(np.abs(independent - saved)))
    assert max_error < 1e-8
    print('Independent QR prediction discrepancy:', max_error, flush=True)
    for family in ('helix', 'generalized_helix', 'spline'):
        reduced = curve.truncate_curve(models[family], curve.curve_geometry(models[family]))
        residual = x[test] - curve.predict(reduced, years[test], templates[test])
        expected = report['scores'][family]['three_dimensional_year_evaluation']['mse']
        assert abs(float(np.mean(residual ** 2)) - expected) < 1e-12
    final = curve.load_model(out / 'final_curve_all_development.npz')
    assert final['spec'] == model['spec']
    geometry = curve.curve_geometry(final)
    assert np.isfinite(curve.predict(final, curve.YEAR_GRID, derivative=True)).all()
    # Only derived figures/geometry are re-exported; fitted models and scores stay fixed.
    curve.plots(out, x, years, templates, models, report['scores'], final)
    audit = dict(node=os.uname().nodename, job_id=os.environ['SLURM_JOB_ID'],
                 original_source_hash_matches=True, input_hashes_match=True, split_ids_match=True,
                 direct_metric_checks=len(discrepancies), maximum_mse_discrepancy=max(discrepancies),
                 independent_fit='scipy.linalg.lstsq(gelsy) on raw rows and ambient responses',
                 independent_rank=int(rank), maximum_prediction_discrepancy=max_error,
                 three_dimensional_checks=3, analytic_derivatives_finite=True,
                 plot_source_sha256=curve.sha256(curve.__file__),
                 audit_source_sha256=curve.sha256(__file__),
                 final_curve_geometry={k: geometry[k] for k in
                                       ('top3_fraction', 'dimension95', 'dimension99')})
    (out / 'artifact_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
    shutil.copyfile(__file__, out / 'source/audit_years_spiral.py')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
