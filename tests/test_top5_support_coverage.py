"""Five-PC eligibility, projection and equal-score aggregation contracts."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
import run_top5_support_coverage as top5


def hddc_model(ranks=(6, 4), tied=False):
    loadings = torch.zeros((2, 8, 6), dtype=torch.float64)
    values = torch.tensor([9., 8., 7., 6., 5., 4.], dtype=torch.float64)
    if tied:
        values[4] = values[5]
    for i in range(2):
        loadings[i, :6] = torch.diag(values)
    return SimpleNamespace(mu=torch.zeros((2, 8), dtype=torch.float64), component_ranks=torch.tensor(ranks),
        pi_logits=torch.zeros(2), _W=lambda: loadings, _psi=lambda: torch.ones((2, 8), dtype=torch.float64))


def test_hddc_top5_covariance_lengths_and_symmetric_rank_exclusion():
    result = top5.top5_geometry(hddc_model(), 'hddc', torch.zeros(8), torch.eye(8))
    assert result['exclusion_reasons'] == ['', 'retained_rank_below_five']
    torch.testing.assert_close(result['eigenvalues'][0], torch.tensor([82., 65., 50., 37., 26., 17.], dtype=torch.float64))
    torch.testing.assert_close(result['half_lengths'][0], result['original_half_lengths'][0])
    assert torch.all(result['half_lengths'][0, :-1] > result['half_lengths'][0, 1:])
    torch.testing.assert_close(result['alpha'][0], torch.ones(5, dtype=torch.float64))


def test_five_ambient_pcs_may_project_to_three_dimensions():
    q, _ = torch.linalg.qr(torch.randn(8, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64))
    result = top5.top5_geometry(hddc_model(), 'hddc', torch.zeros(8), q.T)
    assert not result['exclusion_reasons'][0]
    assert result['vectors'].shape == (2, 5, 3)
    torch.testing.assert_close(result['vectors'][0].norm(dim=1), torch.ones(5, dtype=torch.float64))
    torch.testing.assert_close(result['half_lengths'][0], result['original_half_lengths'][0]*result['alpha'][0].sqrt())
    assert not torch.allclose(result['vectors'][0]@result['vectors'][0].T, torch.eye(5, dtype=torch.float64))


def test_tied_fifth_boundary_and_null_projection_are_excluded():
    tied = top5.top5_geometry(hddc_model(tied=True), 'hddc', torch.zeros(8), torch.eye(8))
    assert tied['exclusion_reasons'][0] == 'nonunique_top5_axis'
    null = top5.top5_geometry(hddc_model(), 'hddc', torch.zeros(8), torch.eye(8)[:3])
    assert null['exclusion_reasons'][0] == 'null_top5_projection'


def test_kmeans_uses_same_retained_rank_rule():
    model = SimpleNamespace(mu=torch.zeros((2, 8), dtype=torch.float64), component_ranks=torch.tensor([6, 4]),
        pca_valid=torch.tensor([True, True]), eigenvalues=torch.tensor([[9., 8., 7., 6., 5., 4., 3., 2.]]*2),
        W=torch.eye(8, dtype=torch.float64)[None].expand(2, -1, -1))
    result = top5.top5_geometry(model, 'kmeans', torch.zeros(8), torch.eye(8))
    assert result['exclusion_reasons'] == ['', 'retained_rank_below_five']


def test_equal_weight_score_average_preserves_own_lengths_and_zero(monkeypatch):
    seen = []
    def score(center, vector, length, component):
        i = int(np.argmax(vector))
        seen.append((length, component))
        return dict(coverage_pc1=i/10, coverage_random=.2, delta_coverage=i/10-.2,
                    max_grid_error=0., max_refinement_change=0.)
    monkeypatch.setattr(top5.base, 'score_component', score)
    result = top5.score_five(np.zeros(6), np.eye(6)[:5], np.arange(1., 6.), 7)
    assert result['coverage_top5'] == pytest.approx(.2)
    assert result['delta_coverage'] == pytest.approx(0., abs=1e-15)
    assert result['coverage_pc1_same_subset'] == 0.
    assert seen == [(float(i), 7) for i in range(1, 6)]


def test_collect_empty_top5_population_renders_na(monkeypatch, tmp_path):
    import json
    sources = [('years', 'Years', 4, None, None), ('years', 'Years', 28, None, None),
               ('temperature', 'Temperature', 4, None, None), ('temperature', 'Temperature', 28, None, None)]
    monkeypatch.setattr(top5, 'SOURCES', sources)
    fields = ['coverage_top5', 'coverage_random', 'delta_coverage', 'coverage_pc1_same_subset', 'delta_pc1_same_subset']
    for dataset, _, layer, _, _ in sources:
        directory = tmp_path/f'{dataset}_l{layer:02}'
        directory.mkdir()
        (directory/'COMPLETED.json').write_text(json.dumps(dict(completed=True)))
        (directory/'manifest.json').write_text(json.dumps(dict(baseline_manifest=dict(dimension=3,
            global_pca=dict(retained_variance_fraction=.8)))))
        summaries = [dict(run_id=f'{kind}_{k}_{threshold}', kind=kind, K=k, threshold=threshold, eligible=0, total=k,
            **{f'{aggregate}_{field}': None for aggregate in ['mean', 'median'] for field in fields})
            for kind in ['hddc', 'kmeans'] for k in [50, 100] for threshold in [.0001, .001, .01, .1]]
        top5.write_csv(directory/'summary.csv', summaries)
        top5.write_csv(directory/'components.csv', [])
    top5.collect(tmp_path)
    assert len(top5.read_csv(tmp_path/'tabella_soglia_01.csv')) == 16
    assert (tmp_path/'03_pannello_medie.pdf').stat().st_size > 0


def test_float32_saved_axes_are_normalized_at_checkpoint_precision():
    q, _ = torch.linalg.qr(torch.randn(8, 6, dtype=torch.float32, generator=torch.Generator().manual_seed(11)))
    q[:, 0] *= 1+torch.finfo(torch.float32).eps
    model = SimpleNamespace(mu=torch.zeros((1, 8)), component_ranks=torch.tensor([6]), pca_valid=torch.tensor([True]),
        eigenvalues=torch.tensor([[9., 8., 7., 6., 5., 4., 3., 2.]]), W=q[None])
    result = top5.top5_geometry(model, 'kmeans', torch.zeros(8), torch.eye(8))
    assert not result['exclusion_reasons'][0]
    torch.testing.assert_close(result['vectors'][0].norm(dim=1), torch.ones(5, dtype=torch.float64), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(result['alpha'][0], torch.ones(5, dtype=torch.float64), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(result['half_lengths'][0], result['original_half_lengths'][0], atol=1e-14, rtol=1e-14)
