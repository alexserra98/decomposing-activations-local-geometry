"""Collect the 12-run neutral-year validation-size study and render its comparison."""
import csv
import json
import math
from pathlib import Path

from dalg.analysis.aggregate_metrics import aggregate_metrics

ROOT = Path('dalg-cache/sae-manifold/pc1_validation_size_study')
OUTPUT = Path('outputs/experiments/pc1_validation_size_study')


def main():
    rows = []
    for name in ('hddc', 'kmeans'):
        manifest = OUTPUT / f'{name}_manifest.jsonl'
        runs = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        assert len(runs) == 6
        for run in runs:
            directory = Path(run['run_dir'])
            args = run['training']['arguments']
            report = json.loads((directory / 'metrics.json').read_text())
            split = json.loads((directory / 'val_indices.json').read_text())
            metric = report['pc1_density']
            expected = 2200 if args['val_frac'] > .1 else 1600
            assert args['seed'] == 42
            assert split['val_rows'] == metric['reference']['points'] == expected
            assert metric['reference']['global_rows'] == split['val_global_rows']
            assert split['train_rows'] == 20020 - expected
            assert metric['settings']['pc1_density_space'] == 'pca75'
            assert metric['settings']['pc1_density_neighbors'] == 20
            rows.append({
                'model': name, 'validation_points': expected, 'train_points': split['train_rows'],
                'split_seed': args['split_seed'], 'fitting_seed': args['seed'],
                'pca_dimensions': metric['density_dimension'],
                'log_retention_median': metric['log_retention']['median'],
                'log_enrichment_median': metric['log_enrichment']['median'],
                'retention': math.exp(metric['log_retention']['median']),
                'enrichment': math.exp(metric['log_enrichment']['median']),
                'eligible_components': metric['eligible_components'], 'run_dir': str(directory),
            })
    assert {(r['model'],r['validation_points'],r['split_seed']) for r in rows} == {
        (m,n,s) for m in ('hddc','kmeans') for n in (1600,2200) for s in (42,43,44)}
    rows.sort(key=lambda r:(r['model'],r['split_seed'],r['validation_points']))
    with (OUTPUT / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    general, manifolds = aggregate_metrics(ROOT)
    general.to_csv(ROOT / 'general_metrics.csv',index=False)
    manifolds.to_csv(ROOT / 'manifold_metrics.csv',index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(10,7),layout='constrained',sharex=True)
    for column,model in enumerate(('hddc','kmeans')):
        for row,metric in enumerate(('retention','enrichment')):
            ax = axes[row,column]
            for seed in (42,43,44):
                points = [r for r in rows if r['model']==model and r['split_seed']==seed]
                ax.plot([r['validation_points'] for r in points], [r[metric] for r in points],
                        marker='o',label=f'Split seed {seed}')
            ax.set_yscale('log')
            ax.set_xticks([1600,2200])
            ax.set_title(f'{model.upper()} — {metric}')
            ax.set_ylabel('exp(median component log score)')
            ax.grid(alpha=.25)
            if row == 1:
                ax.set_xlabel('Validation reference points')
            if row == 0:
                ax.legend()
    fig.suptitle('Neutral years · layer 4 · K=50 · τ=0.1 · fitting seed 42\n'
                 'Models and PCA refitted per split; PCA75 density, kNN=20')
    for extension in ('png','pdf'):
        fig.savefig(OUTPUT / f'comparison.{extension}',dpi=180)
    plt.close(fig)
    print(json.dumps(rows,indent=2),flush=True)


if __name__ == '__main__':
    main()
