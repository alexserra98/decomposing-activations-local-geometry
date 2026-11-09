"""Numerical contracts for the temporary ambient principal-sheet experiment."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'temporary'))
from fit_years_surface import (compressed, features, fit_surface, isotropic_pcs, load_surface,
                               optimize_coordinates, predict, project, regress,
                               save_surface, tangent)
from dalg.evaluation.subspaces import leading_covariance_eigenspaces, subspace_alignment
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC


def curved_sheet(uv):
    u, v = uv.T
    return np.column_stack((u, v, .4 * u ** 2 + .2 * u * v, .3 * v ** 2))


def sheet_model():
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 12), np.linspace(0, 1, 12)), -1).reshape(-1, 2)
    return regress(curved_sheet(grid), grid, 6, 0., np.zeros(4))


def test_surface_recovers_polynomial_and_analytic_partials():
    model = sheet_model()
    uv = np.random.default_rng(4).uniform(.1, .9, (25, 2))
    np.testing.assert_allclose(predict(model, uv), curved_sheet(uv), atol=1e-12)
    np.testing.assert_allclose(features(uv, 6).sum(1), np.ones((len(uv), 1)), atol=1e-14)
    for j in (0, 1):
        delta = np.eye(2)[j] * 1e-6
        numeric = (predict(model, uv + delta) - predict(model, uv - delta)) / 2e-6
        np.testing.assert_allclose(predict(model, uv, j), numeric, atol=1e-9)


def test_ambient_projection_and_compression_preserve_geometry():
    model = sheet_model()
    rng = np.random.default_rng(5)
    rotation = np.linalg.qr(rng.normal(size=(19, 4)))[0]
    model['coef'] = model['coef'] @ rotation.T
    model['origin'] = rng.normal(size=19)
    uv = rng.uniform(.1, .9, (20, 2))
    points = predict(model, uv)
    target, coef = compressed(model, points)
    np.testing.assert_allclose(features(uv, 6) @ coef, target, atol=2e-14)
    estimated = project(model, points, steps=40)
    np.testing.assert_allclose(estimated, uv, atol=2e-6)
    wrong = np.clip(uv + .1, 0, 1)
    _, before = optimize_coordinates(target, coef, 6, wrong, steps=0)
    _, after = optimize_coordinates(target, coef, 6, wrong)
    assert np.all(after <= before)


def test_tangents_and_alignment_ignore_coordinate_basis_and_rotation():
    model = sheet_model()
    uv = np.array([[.3, .6]])
    bases, singular = tangent(model, uv)
    assert singular[0, 1] > 0
    expected = np.array([[1, 0], [0, 1], [.8 * .3 + .2 * .6, .2 * .3], [0, .6 * .6]])
    q = np.linalg.qr(expected)[0]
    overlap, worst = subspace_alignment(torch.from_numpy(bases[0]), torch.from_numpy(q))
    np.testing.assert_allclose([overlap, worst], [1, 1], atol=1e-14)


def test_isotropic_thin_svd_matches_full_covariance_pcs():
    torch.manual_seed(9)
    model = MFA_HDDC(torch.randn(3, 9), rank=4, isotropic_psi=True)
    with torch.no_grad():
        model.scale_rho.copy_(torch.randn(3, 4))
        model.rank_mask[1, 2:] = 0
    values, pcs = isotropic_pcs(model)
    full_values, full_pcs = leading_covariance_eigenspaces(model, 2)
    torch.testing.assert_close(values, full_values, atol=1e-12, rtol=1e-12)
    for k in range(3):
        torch.testing.assert_close(pcs[k] @ pcs[k].T, full_pcs[k] @ full_pcs[k].T, atol=1e-12, rtol=1e-12)


def test_surface_bundle_roundtrip(tmp_path):
    model = sheet_model()
    model.update(initialization=[0, 1], history=[])
    uv = np.array([[.2, .7]])
    path = tmp_path / 'sheet.npz'
    save_surface(path, model, uv)
    restored, latent = load_surface(path)
    np.testing.assert_array_equal(latent, uv)
    np.testing.assert_allclose(predict(restored, uv), predict(model, uv))


def test_alternating_fit_improves_on_plane_for_bent_sheet():
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(12)
    uv = rng.uniform(.05, .95, (500, 2))
    x = curved_sheet(uv)
    x[:, 2:] *= 4
    model, _ = fit_surface(x[:350], 6, 1e-6, iterations=12)
    latent = project(model, x[350:])
    residual = np.mean((x[350:] - predict(model, latent)) ** 2)
    plane = PCA(2).fit(x[:350])
    plane_error = np.mean((x[350:] - plane.inverse_transform(plane.transform(x[350:]))) ** 2)
    assert residual < .1 * plane_error
    objectives = [row['objective'] for row in model['history']]
    assert np.all(np.diff(objectives) <= 1e-12)
