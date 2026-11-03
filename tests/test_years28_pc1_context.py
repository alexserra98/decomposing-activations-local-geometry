"""Guard the dataset/layer and saved-row contracts of the years/layer-28 repeat."""
import json
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
from run_years28_pc1_density import discover_runs, validate_saved_splits, validate_resume_manifest


def fixture_runs(root):
    for method, kind, checkpoint in [('hddc_em', 'hddc', 'mfa_model.pt'),
                                      ('kmeans_pca', 'kmeans', 'kmeans_model.pt')]:
        for K in [50, 100]:
            for tau in [.0001, .001, .01, .1]:
                folder = root/f'sae_manifold_years_neutral_{method}'/f'{K}_{tau}'
                folder.mkdir(parents=True)
                (folder/'run_spec.json').write_text(json.dumps(dict(dataset=dict(id='years_neutral', layer=28),
                                                                     training=dict(model_kind=kind))))
                (folder/'config.json').write_text(json.dumps(dict(layer=28, d_model=4096, window=1, drop_prefix=0,
                        shard_dir=str(root), K=K, surgery_threshold=tau, val_frac=.1, split_seed=42)))
                (folder/'TRAINING_COMPLETED.json').write_text('{"completed": true}')
                (folder/checkpoint).touch()
                (folder/'val_indices.json').write_text(json.dumps(dict(train_rows=3, val_rows=2, val_global_rows=[7, 9])))


def test_exact_years_layer28_sweep_and_saved_row_order(tmp_path):
    fixture_runs(tmp_path)
    runs = discover_runs(tmp_path)
    assert len(runs) == 16
    assert {r['config']['layer'] for r in runs} == {28}
    validate_saved_splits(runs, [0, 1, 2], [7, 9])
    with pytest.raises(ValueError, match='split mismatch'):
        validate_saved_splits(runs, [0, 1, 2], [9, 7])


def test_wrong_dataset_cannot_be_evaluated_as_years(tmp_path):
    fixture_runs(tmp_path)
    path = next(tmp_path.glob('sae_manifold_years_neutral_hddc_em/*/run_spec.json'))
    spec = json.loads(path.read_text())
    spec['dataset']['id'] = 'temperature_02d'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='Unexpected dataset/layer/model'):
        discover_runs(tmp_path)


def test_resume_requires_same_pca_metric_checkpoint_and_reference_population():
    metadata = dict(dataset='years_neutral', layer=28, reference_global_rows=[7, 9], dimension=3,
                    k=20, n_random=32, probability=.8, random_seed_rule='42 + component',
                    integration='full-pilot adaptive', sources=[dict(checkpoint_sha256='checkpoint')],
                    global_pca=dict(sha256='pca'), source_sha256={'src/dalg/analysis/pc1_density.py': 'metric'})
    validate_resume_manifest(metadata, copy.deepcopy(metadata))
    for field, value in [('dataset', 'temperature_02d'), ('layer', 4), ('dimension', 141), ('reference_global_rows', [9, 7]),
                         ('global_pca', dict(sha256='other-pca')),
                         ('sources', [dict(checkpoint_sha256='other-checkpoint')]),
                         ('source_sha256', {'src/dalg/analysis/pc1_density.py': 'other-metric'})]:
        changed = copy.deepcopy(metadata)
        changed[field] = value
        with pytest.raises(ValueError, match='Resume inputs or metric protocol'):
            validate_resume_manifest(metadata, changed)


def test_layer4_fit_cannot_replace_one_layer28_configuration(tmp_path):
    fixture_runs(tmp_path)
    path = next(tmp_path.glob('sae_manifold_years_neutral_hddc_em/*/run_spec.json'))
    spec = json.loads(path.read_text())
    spec['dataset']['layer'] = 4
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='Require all 16 saved years/layer-28'):
        discover_runs(tmp_path)
