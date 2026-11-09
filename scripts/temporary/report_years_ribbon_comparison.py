"""Audit saved component aggregates and write the ribbon comparison report."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


def digest(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output
    result = json.loads((output/'results.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    rows = list(csv.DictReader((output/'components.csv').open()))
    assert len(rows) == 1200 and len(result['summaries']) == 16
    checked = 0
    for summary in result['summaries']:
        eligible = [r for r in rows if r['run'] == summary['run'] and r['eligible'] == 'True']
        assert len(eligible) == summary['eligible_components']
        for metric, scores in summary.items():
            if not isinstance(scores, dict) or 'valid_components' not in scores:
                continue
            valid = [r for r in eligible if r[metric+'_overlap'] != '']
            interior = [r for r in valid if r['boundary'] == 'False']
            assert len(valid) == scores['valid_components']
            assert len(eligible)-len(valid) == scores['undefined_components']
            assert len(interior) == scores['interior_valid_components']
            for population, field in ((valid, 'mean'), (interior, 'interior_mean')):
                mean = statistics.mean(float(r[metric+'_overlap']) for r in population) if population else None
                assert (mean is None and scores[field] is None) or math.isclose(mean, scores[field], abs_tol=1e-12)
            checked += 1
    # The two writers format JSON differently; compare the actual saved split.
    splits = {json.dumps(json.loads(Path(path).read_text()), sort_keys=True)
              for path in manifest['inputs'] if path.endswith('/val_indices.json')}
    assert len(splits) == 1, 'Comparison requires identical saved validation rows'
    runner = output/'source/compare_years_ribbon_tangents.py'
    assert digest(runner) == manifest['source_hash']
    assert digest(result['reference']) == manifest['inputs'][result['reference']]
    for pair in result['matched_comparisons']:
        matched = {s['kind']: s for s in result['summaries'] if s['K'] == pair['K'] and s['threshold'] == pair['threshold']}
        for key, value in pair.items():
            if not key.endswith('_hddc_minus_kmeans'):
                continue
            metric = key.removesuffix('_hddc_minus_kmeans')
            h, k = (matched[kind][metric]['mean'] for kind in ('hddc', 'kmeans'))
            assert (value is None and (h is None or k is None)) or math.isclose(value, h-k, abs_tol=1e-12)
    audit = dict(component_rows=len(rows), checked_metric_aggregates=checked,
                 identical_validation_split=True, reference_and_runner_hashes_match=True,
                 matched_deltas_verified=True, projection_checks=result['projection_audits'])
    (output/'aggregate_audit.json').write_text(json.dumps(audit, indent=2))
    shutil.copy2(__file__, output/'source'/Path(__file__).name)
    for path in ('tests/test_years_ribbon_comparison.py', 'scripts/slurm/temporary/sbatch_years_ribbon_comparison.sh'):
        shutil.copy2(path, output/'source'/Path(path).name)
    alignment_wins = sum(p['tangent_alignment_hddc_minus_kmeans'] > 0 for p in result['matched_comparisons'])
    containment_wins = sum(p['tangent_containment_hddc_minus_kmeans'] > 0 for p in result['matched_comparisons'])
    cap_wins = sum(p['sensitivity_containment_cap16_hddc_minus_kmeans'] > 0 for p in result['matched_comparisons'])
    worst_wins = 0
    for pair in result['matched_comparisons']:
        matched = {s['kind']:s for s in result['summaries'] if s['K']==pair['K'] and s['threshold']==pair['threshold']}
        worst_wins += (matched['hddc']['tangent_containment']['worst_direction_cosine_mean'] >
                       matched['kmeans']['tangent_containment']['worst_direction_cosine_mean'])
    text = ['# HDDC versus KMeans: analytic ribbon tangents', '',
        f'HDDC has higher PC1–PC2 alignment overlap in {alignment_wins}/8 matched settings, '
        f'but higher containment overlap in {containment_wins}/8. The cap-16 containment '
        f'sensitivity gives HDDC {cap_wins}/8 wins. Thus its strongest plane is more aligned '
        'with this ribbon, while the best retained two-PC subset has lower average squared '
        'overlap than KMeans. Absolute overlaps remain low. These are descriptive results '
        'for one seed; the threshold variants are not independent replications.', '',
        f'The companion containment worst-direction cosine favors HDDC in {worst_wins}/8 '
        'settings. A larger average squared overlap and a better least-covered tangent '
        'direction need not favor the same method; consult both scores in results.json.', '',
        'The reference is the fixed layer-4 spiral ribbon fitted in 4096 dimensions. '
        'It is an estimated surface, not ground-truth activation geometry. Its internal held-out '
        'ambient reconstruction R² was approximately 0.27–0.31. No models were retrained and '
        'the external prompt holdout was not read.', '',
        'Each centre is projected onto the bounded ribbon by eliminating radius analytically '
        'and refining every year-grid local minimum. The analytic Jacobian supplies the tangent '
        'at the projected point, not at the generally off-surface centre. A 0.25-year grid is '
        'checked against 0.125 years at five centres per distinct centre array. This is a numerical '
        'nearest-point search, not a certified global optimum.', '',
        'Metrics reuse the wiki definitions and eigengap gates. Alignment compares exactly '
        'PC1–PC2; containment chooses exactly two PCs from the saved effective rank. '
        'Means are unweighted over assignment-live, PCA-valid components with defined scores. '
        'Unique boundary projections remain included; interior-only sensitivity is reported. '
        'Component indices are not paired across methods. All runs have seed 42 and identical '
        'saved validation rows.', '',
        '| K | Threshold | Mean rank H / K | Alignment H / K | Valid H / K | Containment H / K | Valid H / K |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for pair in result['matched_comparisons']:
        matched = {s['kind']:s for s in result['summaries'] if s['K']==pair['K'] and s['threshold']==pair['threshold']}
        h,k = matched['hddc'],matched['kmeans']
        def score(metric):
            return ' / '.join('undefined' if s[metric]['mean'] is None else f"{s[metric]['mean']:.4f}" for s in (h,k))
        def count(metric):return f"{h[metric]['valid_components']} / {k[metric]['valid_components']}"
        text.append(f"| {pair['K']} | {pair['threshold']:g} | {h['mean_effective_rank']:.2f} / {k['mean_effective_rank']:.2f} | {score('tangent_alignment')} | {count('tangent_alignment')} | {score('tangent_containment')} | {count('tangent_containment')} |")
    text += ['', '## Coverage and sensitivity', '',
        'Containment offers more subset choices at higher rank. The supplementary cap-16 '
        'score limits the candidate PCs to min(q,16), matching HDDC’s maximum capacity. '
        'This does not force identical ranks. Adjusted alignment includes rank-one and '
        'rank-zero cases following the wiki.', '',
        '| Method | K | Threshold | Boundary / eligible | Median distance | Interior alignment | Interior containment | Cap-16 containment | Adjusted alignment |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in sorted(result['summaries'], key=lambda s:(s['K'],s['threshold'],s['kind'])):
        def fmt(value):return 'undefined' if value is None else f'{value:.4f}'
        text.append(f"| {s['kind']} | {s['K']} | {s['threshold']:g} | {s['boundary_projections']} / {s['eligible_components']} | {s['median_surface_distance']:.4f} | {fmt(s['tangent_alignment']['interior_mean'])} | {fmt(s['tangent_containment']['interior_mean'])} | {fmt(s['sensitivity_containment_cap16']['mean'])} | {fmt(s['tangent_adjusted_alignment']['mean'])} |")
    text += ['', 'Four focused tests cover synthetic projection, ambiguous footpoints, rank/eigengap '
        'metric semantics, and thin-SVD agreement with full covariance eigendecomposition. '
        'The aggregate audit independently reconstructs means/counts/deltas from the component '
        'CSV and verifies common split and reference/source hashes.', '',
        'See [component scores](components.csv), [full results](results.json), '
        '[aggregate audit](aggregate_audit.json), and [comparison figure](comparison.png).', '']
    (output/'report.md').write_text('\n'.join(text))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
