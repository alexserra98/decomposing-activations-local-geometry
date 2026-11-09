"""Prompt labels partition samples; training assignments alone label model tiles."""
import copy
import json
from pathlib import Path

import pytest
import torch

from dalg.analysis.aggregate_metrics import aggregate_metrics
from dalg.analysis.reconstruction import reconstruction_errors
from dalg.evaluation.common import EvaluationData, load_evaluation_data, load_model
from dalg.evaluation.coverage import nearest_live_centroid_distances, summarize_coverage_distances
from dalg.evaluation.unsupervised import evaluate_unsupervised
from dalg.evaluation.unsupervised_subsets import subset_populations
from dalg.pipeline import execute_run, map_run_to_manifest
from dalg.pipeline.evaluation import _evaluation_metrics_valid
from tests.synthetic_shards import LAYER, build_multi_shard
from tests.test_pipeline_phases import _config


def test_majority_ignores_validation_and_preserves_window_assignments():
    # Noncontiguous selected positions, two retained tokens per row.
    positions = [1, 3, 4, 6, 8, 9]
    meta = [{"subset": "unused"} for _ in range(10)]
    for p, name in zip(positions, ["A", "B", "A", "B", "B", "B"]):
        meta[p] = {"subset": name, "global_row": p + 100}
    assignments = torch.tensor([0, 1, 1, 2, 0, 0, 2, 2, 0, 0, 3, 3])
    data = EvaluationData(Path('.'), 'selection', {"window": 3}, meta, positions,
                          positions[:4], positions[4:], 1, assignments,
                          torch.bincount(assignments, minlength=5), torch.tensor([3, 2, 3, 0, 0]))
    groups, report = subset_populations(data)
    assert report['manifold_names'] == ['A', 'B']
    records = report['per_component']
    assert [r['manifold_id'] for r in records] == [0, None, 1, None, None]
    assert records[1]['exclusion_reason'] == 'tied_training_counts'
    assert records[1]['purity'] == .5
    assert records[3]['exclusion_reason'] == 'no_training_assignments'
    assert groups[0][1].positions == [1, 4]
    assert groups[1][1].val_positions == [8, 9]
    assert groups[1][1].assignments.tolist() == [1, 2, 2, 2, 0, 0, 3, 3]


@pytest.mark.parametrize('kind', ['kmeans', 'hddc'])
def test_partition_metrics_use_full_model_and_export(tmp_path, kind):
    config = _config(tmp_path, kind)
    config['model']['K'] = 4
    generator = torch.Generator().manual_seed(42)
    for shard in range(2):
        points = torch.randn(6, 3, 2, generator=generator)
        for row in range(6):
            points[row] += (row % 2) * 1000 + shard * 20
        torch.save(points, Path(config['dataset']['shard_dir']) / f'layer{LAYER:02d}/shard_{shard:05d}.pt')
    test_root = build_multi_shard(tmp_path / 'test', n_shards=1, rows_per_shard=4)
    test_config = json.loads((test_root / 'config.json').read_text())
    test_config['layers'] = [LAYER]
    (test_root / 'config.json').write_text(json.dumps(test_config))
    config['evaluation'].update(heldout_distribution_coverage=True, test_shard_dir=str(test_root))
    run = map_run_to_manifest(config)
    directory = execute_run(run)
    report = json.loads((directory / 'metrics.json').read_text())
    assert _evaluation_metrics_valid(run, report)
    assert [g['subset'] for g in report['per_manifold']] == ['A', 'B']
    model = load_model(directory, kind)
    data = load_evaluation_data(directory, shard_dir=config['dataset']['shard_dir'],
                                layer=LAYER, model=model, model_kind=kind)
    points = torch.cat([torch.load(p, weights_only=True) for p in sorted(
        Path(config['dataset']['shard_dir']).glob(f'layer{LAYER:02d}/*.pt'))])
    test_points = torch.load(test_root / f'layer{LAYER:02d}/shard_00000.pt', weights_only=True)
    for group, (name, sliced) in zip(report['per_manifold'], subset_populations(data)[0]):
        for split, rows in [('train', sliced.train_positions), ('validation', sliced.val_positions)]:
            with torch.no_grad():
                errors, excluded = reconstruction_errors(model, points[rows].reshape(-1, 2), model_kind=kind)
            result = group['reconstruction'][split]
            assert result['mse'] == pytest.approx(float(errors.mean()))
            assert result['standard_error'] == pytest.approx(float(errors.std() / len(errors) ** .5))
            assert result['excluded_n'] == excluded
        selected = test_points[::2] if name == 'A' else test_points[1::2]
        distances = nearest_live_centroid_distances(selected.reshape(-1, 2), model.mu,
                                                    live_components=data.train_cluster_sizes > 0)
        expected = summarize_coverage_distances(distances, thresholds=None)
        coverage = group['heldout_distribution_coverage']
        assert coverage['coverage_curve'] == expected['coverage_curve']
        assert coverage['components_live'] == report['heldout_distribution_coverage']['components_live']
        assert group['tile_alignment']['comparisons']
        for edge in group['tile_alignment']['comparisons']:
            assert edge['source'] in group['component_ids']
            assert edge['target'] in group['component_ids']
    general, manifolds = aggregate_metrics(directory.parent)
    assert len(general) == 1 and len(manifolds) == 2
    assert manifolds['subset'].tolist() == ['A', 'B']
    for mutation in ['missing', 'duplicate', 'count', 'association', 'coverage', 'reconstruction']:
        bad = copy.deepcopy(report)
        if mutation == 'missing':
            del bad['per_manifold']
        elif mutation == 'duplicate':
            bad['per_manifold'][1]['subset'] = 'A'
        elif mutation == 'count':
            bad['per_manifold'][0]['dataset']['train_points'] += 1
        elif mutation == 'association':
            bad['stratification']['per_component'][0]['purity'] = -1
        elif mutation == 'coverage':
            bad['per_manifold'][0]['heldout_distribution_coverage']['source']['manifold_subset'] = 'B'
        else:
            bad['per_manifold'][0]['reconstruction']['train']['mse'] = float('nan')
        assert not _evaluation_metrics_valid(run, bad), mutation
    # A separate test directory with incompatible labels is rejected explicitly.
    meta_path = test_root / 'meta/shard_00000.json'
    meta = json.loads(meta_path.read_text())
    meta['rows'][0]['subset'] = 'unknown'
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='test subsets must match'):
        evaluate_unsupervised(directory, shard_dir=config['dataset']['shard_dir'], layer=LAYER,
                              model_kind=kind, test_shard_dir=test_root, device='cpu')


@pytest.mark.parametrize('kind', ['kmeans', 'hddc'])
def test_empty_subgroup_split_has_null_reconstruction(kind):
    from dalg.evaluation.reconstruction import reconstruction_report_valid
    from dalg.evaluation.unsupervised_subsets import subset_reconstruction
    from dalg.models.kmeans import KMeans
    from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC

    model = (KMeans.from_centroids(torch.zeros(1, 2)) if kind == 'kmeans' else
             MFA_HDDC(torch.zeros(1, 2), rank=1, shared_b=True))
    data = EvaluationData(Path('.'), None, {'window': 1, 'd_model': 2}, [], [], [], [],
                          0, torch.empty(0, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                          torch.zeros(1, dtype=torch.long))
    report = subset_reconstruction(model, data, model_kind=kind, layer=LAYER,
                                   batch_size=8, device=torch.device('cpu'))
    assert report['validation'] == {'mse': None, 'standard_error': None, 'n': 0, 'excluded_n': 0}
    assert reconstruction_report_valid(report, model_kind=kind, dataset=data.dataset_summary(LAYER),
                                       allow_empty_splits=True)
    assert not reconstruction_report_valid(report, model_kind=kind, dataset=data.dataset_summary(LAYER))
