"""PC1 density follows saved splits, survives aggregation, and validates on resume."""
import copy
import json
from pathlib import Path

import pytest
import torch

from dalg.analysis.aggregate_metrics import aggregate_metrics
from dalg.cli.run_pipeline import build_parser
from dalg.evaluation import evaluate_pipeline_run
from dalg.evaluation import pc1_density as density
from dalg.evaluation.common import load_evaluation_data, load_model
from dalg.evaluation.saved_runs import _settings
from dalg.pipeline import execute_run, map_run_to_manifest, PipelineConfigError
from dalg.pipeline.evaluation import _evaluation_metrics_valid
from tests.synthetic_shards import LAYER
from tests.test_pipeline_phases import _config


@pytest.mark.parametrize('kind', ['kmeans', 'hddc'])
@pytest.mark.parametrize('space', ['pca75', 'ambient'])
def test_pipeline_density_saved_split_and_aggregation(tmp_path, monkeypatch, kind, space):
    config = _config(tmp_path, kind)
    generator = torch.Generator().manual_seed(31)
    all_points = []
    # Distinct, nondegenerate validation points exercise real adaptive quadrature.
    for index in range(2):
        points = torch.randn(6, 3, 2, generator=generator) * torch.tensor([3., 1.])
        all_points.append(points)
        torch.save(points, Path(config['dataset']['shard_dir']) / f'layer{LAYER:02d}/shard_{index:05d}.pt')
    all_points = torch.cat(all_points)
    config['evaluation'].update(pc1_density=True, pc1_density_space=space,
                                pc1_density_neighbors=3, pc1_density_random_directions=2)
    captured = {}
    fit_pca, knn = density.fit_global_variance_pca, density.KNNLogDensity

    def capture_pca(batches, fraction):
        captured['training'] = torch.cat(list(batches))
        captured['pca'] = fit_pca([captured['training']], fraction)
        return captured['pca']

    def capture_reference(reference, **kwargs):
        captured['reference'] = reference.clone()
        return knn(reference, **kwargs)

    monkeypatch.setattr(density, 'fit_global_variance_pca', capture_pca)
    monkeypatch.setattr(density, 'KNNLogDensity', capture_reference)
    run = map_run_to_manifest(config)
    directory = execute_run(run)
    report = json.loads((directory / 'metrics.json').read_text())
    metric = report['pc1_density']
    model = load_model(directory, kind)
    data = load_evaluation_data(directory, shard_dir=run['dataset']['shard_dir'], layer=LAYER,
                                model=model, model_kind=kind)
    train = all_points[data.train_positions].reshape(-1, 2)
    val = all_points[data.val_positions].reshape(-1, 2).double()
    if space == 'pca75':
        torch.testing.assert_close(captured['training'], train)
        torch.testing.assert_close(captured['pca']['mean'], train.double().mean(0))
        expected_reference = (val - train.double().mean(0)) @ captured['pca']['basis'].T
        assert metric['projection']['training_count'] == len(train)
    else:
        assert 'training' not in captured
        assert metric['projection'] is None
        expected_reference = val
    torch.testing.assert_close(captured['reference'], expected_reference)
    assert metric['quadrature_grid_points'] == [201, 401, 801, 1601, 3201, 6401]
    assert metric['reference']['points'] == len(val)
    assert metric['eligible_components'] > 0
    assert metric['eligible_components'] + metric['excluded_components'] == 2
    assert _evaluation_metrics_valid(run, report)
    json.dumps(report, allow_nan=False)
    general, manifolds = aggregate_metrics(directory.parent)
    assert manifolds.empty
    assert general.iloc[0]['pc1_density.log_retention.median'] == metric['log_retention']['median']
    assert json.loads(general.iloc[0]['pc1_density.per_component']) == metric['per_component']

    for mutation in ('missing', 'settings', 'summary', 'split', 'duplicate_id', 'resolution', 'old_version', 'nan'):
        broken = copy.deepcopy(report)
        block = broken['pc1_density']
        if mutation == 'missing':
            del broken['pc1_density']
        elif mutation == 'settings':
            block['settings']['pc1_density_seed'] += 1
        elif mutation == 'summary':
            block['log_retention']['median'] += 1
        elif mutation == 'split':
            block['reference']['points'] += 1
        elif mutation == 'resolution':
            block['quadrature_grid_points'] = [1601, 3201, 6401]
        elif mutation == 'old_version':
            block['version'] = 1
        elif mutation == 'duplicate_id':
            block['per_component'][0]['component_id'] = 2
        else:
            block['per_component'][0]['log_retention'] = float('nan')
        assert not _evaluation_metrics_valid(run, broken), mutation

    without = copy.deepcopy(run)
    without['evaluation']['pc1_density'] = False
    assert not _evaluation_metrics_valid(without, report)
    assert 'pc1_density' not in evaluate_pipeline_run(without)
    with pytest.raises(ValueError, match='saved validation points'):
        density.evaluate_pc1_density(model, data, model_kind=kind, layer=LAYER,
                                     batch_size=8, neighbors=len(val) + 1)


@pytest.mark.parametrize('overrides', [
    {'pc1_density': 'yes'}, {'pc1_density_space': 'pca'},
    {'pc1_density_neighbors': 0}, {'pc1_density_random_directions': True},
    {'pc1_density_seed': -1}, {'pc1_density': True, 'kind': 'toy_manifold_tiling'},
])
def test_invalid_pipeline_settings(tmp_path, overrides):
    config = _config(tmp_path)
    config['evaluation'].update(overrides)
    with pytest.raises(PipelineConfigError, match='PC1|pc1'):
        map_run_to_manifest(config, check_inputs=False)


def test_saved_run_cli_inherits_and_overrides_density_settings(tmp_path, monkeypatch):
    from dalg.evaluation import saved_runs
    config = _config(tmp_path)
    run = map_run_to_manifest(config)
    args = build_parser().parse_args([
        'evaluate', '--manifest', 'source.jsonl', '--pc1-density',
        '--pc1-density-space', 'ambient', '--pc1-density-neighbors', '7',
        '--pc1-density-random-directions', '4', '--pc1-density-seed', '9', '--dry-run',
    ])
    captured = {}

    def plan(manifests, *, indices, overrides, resources):
        captured.update(overrides)
        return {'runs': [{'index': 0, 'run': run, 'evaluation': overrides, 'resources': {}}],
                'skipped': [], 'job_dir': str(tmp_path), 'summary_dirs': []}

    monkeypatch.setattr(saved_runs, 'plan_evaluation', plan)
    saved_runs.command_evaluate(args)
    resolved = _settings(run, captured)
    assert resolved['pc1_density'] is True
    assert resolved['pc1_density_space'] == 'ambient'
    assert resolved['pc1_density_neighbors'] == 7
    assert resolved['pc1_density_random_directions'] == 4
    assert resolved['pc1_density_seed'] == 9
    default = _settings(run, {'pc1_density': True})
    assert default['pc1_density_space'] == 'pca75'
