"""Assignments and final checks for the temperature/years saved-run evaluation."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = [
    REPO / 'outputs/experiments/sae_manifold_temperature_02d_hddc_em/manifest_73e4d5bc8f.jsonl',
    REPO / 'outputs/experiments/sae_manifold_temperature_02d_kmeans_pca/manifest_8bd215c4f3.jsonl',
    REPO / 'outputs/experiments/sae_manifold_years_neutral_hddc_em/manifest_0cf7449e96.jsonl',
    REPO / 'outputs/experiments/sae_manifold_years_neutral_kmeans_pca/manifest_8a97901252.jsonl',
]
JOB_ROOT = REPO / 'outputs/evaluations/sae_temperature_years'
ROOTS = [REPO / 'dalg-cache' / 'sae-manifold' / name for name in (
    'sae_manifold_temperature_02d_llama31_8b_last',
    'sae_manifold_years_neutral_llama31_8b_last',
)]


def runs():
    rows = [json.loads(line) for path in MANIFESTS for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 64
    assert len({Path(row['run_dir']).resolve() for row in rows}) == 64
    return rows


def assign(index):
    from dalg.evaluation.common import load_evaluation_data, load_model
    from dalg.pipeline.assignments import _assignments_path

    run = runs()[index]
    directory = Path(run['run_dir']).resolve()
    kind = run['training']['model_kind']
    path = _assignments_path(run)
    if not path.exists():
        subprocess.run([
            sys.executable, '-m', 'dalg.cli.run_metrics', 'assignments',
            '--data-dir', str(directory), '--model-type', kind,
            '--shard-dir', run['dataset']['shard_dir'], '--layer', str(run['dataset']['layer']),
            '--drop-prefix', '0', '--batch-size', '1024', '--device', 'cuda', '--seed', '42',
        ], check=True, cwd=REPO)
    model = load_model(directory, kind)
    data = load_evaluation_data(directory, shard_dir=run['dataset']['shard_dir'],
                                layer=run['dataset']['layer'], model=model, model_kind=kind)
    print(f"Validated {index}: {directory.name}, {data.assignments.numel()} assignments", flush=True)


def launch_evaluation():
    """Run preflight after assignments, submit evaluation, then chain final validation."""
    command = [sys.executable, '-m', 'dalg.cli.run_pipeline', 'evaluate', '--manifest',
               *map(str, MANIFESTS), '--evaluation-kind', 'unsupervised',
               '--no-heldout-distribution-coverage', '--tile-alignment',
               '--tile-alignment-neighbors', '5', '--device', 'cuda', '--batch-size', '4096',
               '--resources', str(JOB_ROOT / 'dgx_resources.yaml'), '--submit']
    dry = subprocess.run([*command, '--dry-run'], cwd=REPO, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (JOB_ROOT / 'evaluation_dry_run.log').write_text(dry.stdout)
    print(dry.stdout, flush=True)
    dry.check_returncode()
    assert 'Eligible: 64; skipped: 0' in dry.stdout, 'Expected all 64 runs to pass preflight'
    submitted = subprocess.run(command, cwd=REPO, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (JOB_ROOT / 'evaluation_submission.log').write_text(submitted.stdout)
    print(submitted.stdout, flush=True)
    submitted.check_returncode()
    plan = re.search(r'^Evaluation plan: (.+)$', submitted.stdout, re.MULTILINE).group(1)
    collector = re.search(r'Submitted collection job (\d+)', submitted.stdout).group(1)
    final = subprocess.run([
        'sbatch', '--parsable', f'--dependency=afterok:{collector}',
        '--job-name=sae-eval-verify', '--partition=DGX', '--account=LADE',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=80G', '--time=02:00:00',
        f'--output={JOB_ROOT}/logs/verify_%j.out',
        'scripts/slurm/temporary/sae_saved_evaluation.sh', 'collect', '--plan', plan,
    ], cwd=REPO, check=True, text=True, capture_output=True)
    ids = {'evaluation_plan': plan, 'evaluation_jobs': re.findall(
        r'Submitted evaluation array job (\d+)', submitted.stdout),
        'collection_job': collector, 'validation_job': final.stdout.strip().split(';')[0]}
    (JOB_ROOT / 'evaluation_jobs.json').write_text(json.dumps(ids, indent=2) + '\n')
    print(json.dumps(ids, indent=2), flush=True)


def collect(plan_path):
    from dalg.analysis.aggregate_metrics import aggregate_metrics
    from dalg.evaluation.saved_runs import collect_evaluations, read_evaluation_plan

    plan = read_evaluation_plan(plan_path)
    assert len(plan['runs']) == 64 and not plan['skipped']
    collect_evaluations(plan)  # Includes the pipeline's report/completion checks.
    for root in ROOTS:
        general, manifolds = aggregate_metrics(root)
        assert len(general) == 32 and manifolds.empty
        assert general['evaluation'].eq('unsupervised').all()
        assert general['tile_alignment.version'].eq(1).all()
        assert general['tile_alignment.neighbors'].eq(5).all()
        assert not any(column.startswith('heldout_distribution_coverage.') for column in general)
        for metric in ('leading_pc_alignment', 'best_subset_pc_alignment'):
            values = general[f'tile_alignment.{metric}.mean']
            assert values.dropna().between(0, 1).all()
            assert values[general['tile_alignment.eligible_components'].ge(2)].notna().all()
        assert set(general['dataset.layer']) == {4, 28}
        assert general.groupby(['dataset.layer', 'model_kind']).size().eq(8).all()
        general.to_csv(root / 'general_metrics.csv', index=False)
        manifolds.to_csv(root / 'manifold_metrics.csv', index=False)
        print(f'Validated and aggregated {len(general)} reports: {root}', flush=True)
    # Execute the notebook in memory for both layers; leave its saved outputs clean.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    notebook = json.loads((REPO / 'notebooks/unsupervised_sweep_results.ipynb').read_text())
    for layer in (4, 28):
        namespace = {}
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source']).replace('LAYER = 4  #', f'LAYER = {layer}  #')
            source = source.replace('from IPython.display import display',
                                    'display = lambda *args, **kwargs: None')
            exec(compile(source, f'notebook-cell-{index}', 'exec'), namespace)
            plt.close('all')
        assert len(namespace['general_df']) == 32
        assert namespace['general_df']['dataset.layer'].eq(layer).all()
    (JOB_ROOT / 'validation.json').write_text(json.dumps({
        'reports': 64, 'reports_per_dataset': 32, 'notebook_layers': [4, 28],
        'tile_alignment_neighbors': 5, 'heldout_distribution_coverage': False,
        'evaluation_plan': str(plan_path), 'completed': True,
    }, indent=2) + '\n')
    print('All 64 reports and both notebook layers validated.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['assign', 'launch-evaluation', 'collect'])
    parser.add_argument('--index', type=int)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.mode == 'assign':
        if args.index is None or not 0 <= args.index < 64:
            parser.error('assign requires --index in 0..63')
        assign(args.index)
    elif args.mode == 'launch-evaluation':
        launch_evaluation()
    else:
        if args.plan is None:
            parser.error('collect requires --plan')
        collect(args.plan)
