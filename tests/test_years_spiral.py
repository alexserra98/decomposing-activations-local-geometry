"""Independent numerical checks for the temporary years parameterization."""
from pathlib import Path
import runpy

import numpy as np
import pytest

curve = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                          'scripts/temporary/fit_years_spiral.py'))
basis, statistics = curve['basis'], curve['statistics']
fit_curve, predict, evaluate = curve['fit_curve'], curve['predict'], curve['evaluate']


def fixture(spec, seed=5, noise=0.):
    rng = np.random.default_rng(seed)
    years = np.repeat(np.arange(1000, 2001, 4), 20)
    templates = np.tile(np.arange(20), 251)
    f = basis(years, spec)
    coef = rng.normal(size=(f.shape[1], 9))
    offsets = rng.normal(size=(20, 9))
    offsets -= offsets.mean(0)
    origin = rng.normal(size=9) + 4
    x = origin + f @ coef + offsets[templates] + noise * rng.normal(size=(len(years), 9))
    return x, years, templates, origin


def test_exact_ambient_helix_and_template_offsets_with_uneven_counts():
    spec = dict(kind='fourier', periods=[137.3])
    x, years, templates, origin = fixture(spec)
    mask = np.random.default_rng(8).uniform(size=len(x)) > .25
    model = fit_curve(statistics(x, years, templates, mask, origin), spec)
    np.testing.assert_allclose(predict(model, years, templates), x, atol=1e-10)
    np.testing.assert_allclose(model['coef'][model['year_features']:].mean(0), 0, atol=1e-12)
    g = curve['curve_geometry'](model)
    assert g['top3_fraction'] == pytest.approx(1.)
    np.testing.assert_allclose(g['basis'].T @ g['basis'], np.eye(3), atol=1e-12)


def test_statistics_match_direct_errors_and_year_decomposition():
    spec = dict(kind='fourier', periods=[50., 201.])
    x, years, templates, origin = fixture(spec, noise=.04)
    train = (years % 12 != 0) & (templates < 16)
    validation = ~train
    model = fit_curve(statistics(x, years, templates, train, origin), spec)
    stats = statistics(x, years, templates, validation, origin)
    report = evaluate(stats, model)
    residual = x[validation] - predict(model, years[validation], templates[validation])
    expected = np.mean(residual ** 2)
    assert report['mse'] == pytest.approx(expected, abs=1e-11)
    assert report['r2'] == pytest.approx(1 - np.sum(residual ** 2) /
                                       np.sum((x[validation] - origin) ** 2), abs=1e-11)
    between = sum(np.sum(residual[years[validation] == y].mean(0) ** 2) *
                  np.sum(years[validation] == y) for y in np.unique(years[validation])) / residual.size
    assert report['year_centroid_mse'] == pytest.approx(between, abs=1e-11)
    assert report['within_year_mse'] + between == pytest.approx(expected, abs=1e-11)
    # No calibration on held-out contexts: they receive the mean (zero) offset.
    p = model['year_features']
    np.testing.assert_array_equal(model['coef'][p + np.arange(16, 20)], 0)


def test_years_and_templates_are_disjoint_and_every_row_has_one_role():
    _, years, templates, _ = fixture(dict(kind='line'))
    masks, membership = curve['split_development'](years, templates)
    roles = ('fit', 'validation', 'year_evaluation', 'template_evaluation',
             'year_template_evaluation')
    np.testing.assert_array_equal(sum(masks[name].astype(int) for name in roles), 1)
    assert set(membership['fit_years']).isdisjoint(membership['validation_years'])
    assert set(membership['evaluation_years']).isdisjoint(membership['fit_years'] +
                                                        membership['validation_years'])
    assert set(membership['seen_templates']).isdisjoint(membership['heldout_templates'])
    again, _ = curve['split_development'](years, templates)
    for name in masks:
        np.testing.assert_array_equal(masks[name], again[name])


@pytest.mark.parametrize('spec', [dict(kind='fourier', periods=[137.3, 10.]),
                                 dict(kind='spline', knots=10)])
def test_analytic_tangent_matches_central_difference(spec):
    x, years, templates, origin = fixture(spec)
    model = fit_curve(statistics(x, years, templates, np.ones(len(x), bool), origin), spec)
    query = np.array([1032.1, 1433.5, 1766.2])
    h = 1e-3
    finite = (predict(model, query + h) - predict(model, query - h)) / (2 * h)
    np.testing.assert_allclose(predict(model, query, derivative=True), finite, atol=1e-8, rtol=1e-6)


def test_spline_recovers_cubic_trend_and_rejects_extrapolation():
    years = np.repeat(np.arange(1000, 2001, 4), 20)
    templates = np.tile(np.arange(20), 251)
    t = (years - 1500) / 500
    x = np.column_stack((t, t ** 2, t ** 3)) + templates[:, None]
    origin = x.mean(0)
    model = fit_curve(statistics(x, years, templates, np.ones(len(x), bool), origin),
                      dict(kind='spline', knots=10))
    np.testing.assert_allclose(predict(model, years, templates), x, atol=1e-9)
    with pytest.raises(ValueError, match='domain'):
        predict(model, [999])


def test_integer_period_two_rank_deficiency_is_explicit_and_harmless():
    years = np.repeat(np.arange(1000, 1100), 20)
    templates = np.tile(np.arange(20), 100)
    x = np.column_stack((np.cos(np.pi * years), years / 1000))
    model = fit_curve(statistics(x, years, templates, np.ones(len(x), bool), x.mean(0)),
                      dict(kind='fourier', periods=[2.]))
    assert model['rank'] < model['year_features'] + 19
    np.testing.assert_allclose(predict(model, years, templates), x, atol=1e-10)


def test_generalized_curve_is_not_forced_into_three_dimensions(tmp_path):
    spec = dict(kind='fourier', periods=[50., 100.])
    years = np.repeat(np.arange(1000, 2001), 2)
    templates = np.tile([0, 1], 1001)
    x = basis(years, spec)[:, 1:]  # Four independent periodic axes plus the line.
    model = fit_curve(statistics(x, years, templates, np.ones(len(x), bool), x.mean(0)), spec)
    g = curve['curve_geometry'](model)
    assert g['dimension95'] > 3
    assert g['top3_fraction'] < .9
    reduced = curve['truncate_curve'](model, g)
    expected = g['center'] + (predict(model, years) - g['center']) @ g['basis'] @ g['basis'].T
    np.testing.assert_allclose(predict(reduced, years), expected, atol=1e-10)
    np.testing.assert_array_equal(reduced['coef'][model['year_features']:],
                                  model['coef'][model['year_features']:])
    path = tmp_path / 'curve.npz'
    curve['save_model'](path, model)
    loaded = curve['load_model'](path)
    np.testing.assert_array_equal(predict(loaded, years), predict(model, years))
    np.testing.assert_array_equal(predict(loaded, years, derivative=True),
                                  predict(model, years, derivative=True))


def test_constant_has_no_curve_dimensions():
    x, years, templates, origin = fixture(dict(kind='constant'))
    model = fit_curve(statistics(x, years, templates, np.ones(len(x), bool), origin),
                      dict(kind='constant'))
    g = curve['curve_geometry'](model)
    assert g['dimension95'] == g['dimension99'] == 0
    assert g['top3_fraction'] is None


def test_period_selection_on_synthetic_helix():
    spec = dict(kind='fourier', periods=[137.3])
    x, years, templates, origin = fixture(spec)
    masks, _ = curve['split_development'](years, templates)
    candidates, selected = curve['select_candidates'](
        statistics(x, years, templates, masks['fit'], origin),
        statistics(x, years, templates, masks['validation'], origin),
        periods=[137.3], spline_knots=[6])
    assert len(candidates) > 10
    assert selected['helix']['spec']['periods'][0] == pytest.approx(137.3, rel=1e-5)
    assert selected['helix']['validation_mse'] < selected['line']['validation_mse'] * 1e-8
