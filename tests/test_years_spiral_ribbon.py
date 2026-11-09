"""Geometric and numerical contracts for the explicit spiral ribbon."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import lstsq
from scipy.optimize import minimize_scalar

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
from fit_years_spiral_ribbon import (design,fit_ribbon,load_model,predict,
                                    project_radius,save_model,tangent_space)


def example():
    rng=np.random.default_rng(12)
    directions=np.linalg.qr(rng.normal(size=(17,5)))[0].T
    return dict(periods=[100.,1300.],radial_period=100.,origin=rng.normal(size=17),
                coef=np.vstack((directions[0]*.1,directions[1]*2,
                                directions[2],directions[3]*.8,
                                directions[0]*.2,directions[4]*.3)),radius_bounds=[.6,1.4])


def test_ribbon_partials_match_finite_differences_and_have_rank_two():
    m=example();y=np.array([1012.3,1234.5,1765.4]);r=np.array([.8,1.,1.2])
    numeric=(predict(m,y+1e-4,r)-predict(m,y-1e-4,r))/2e-4
    np.testing.assert_allclose(predict(m,y,r,derivative='year'),numeric,atol=1e-9)
    numeric=(predict(m,y,r+1e-5)-predict(m,y,r-1e-5))/2e-5
    np.testing.assert_allclose(predict(m,y,r,derivative='radius'),numeric,atol=1e-9)
    jac=np.stack((predict(m,y,r,derivative='year'),predict(m,y,r,derivative='radius')),-1)
    assert np.all(np.linalg.svd(jac,compute_uv=False)[:,1]>1e-3)


def test_exact_radius_projection_matches_independent_minimization():
    m=example();year=np.array([1423.4]);point=predict(m,year,1.17)+.01
    direct=project_radius(m,point,year)[0]
    independent=minimize_scalar(lambda r:np.sum((point-predict(m,year,r))**2),bounds=(.6,1.4),method='bounded')
    assert abs(independent.x-direct)<1e-7
    np.testing.assert_allclose(project_radius(m,point,year,bounds=(.8,1.)),[1.])


def test_fit_recovers_a_true_ambient_helix_sheet_and_qr_solution():
    rng=np.random.default_rng(6);m=example();y=rng.uniform(1000,2000,700);r=rng.uniform(.8,1.2,len(y))
    x=predict(m,y,r)
    fitted,latent=fit_ribbon(x[:500],y[:500],m['periods'],100.,iterations=100,tolerance=1e-10)
    inferred=project_radius(fitted,x[500:],y[500:])
    error=np.mean((x[500:]-predict(fitted,y[500:],inferred))**2)
    assert error<1e-7,error
    assert np.all(np.diff(fitted['history'])<1e-12)
    f=design(y[:500],fitted['periods'],latent,100.)
    independent,_,_,_=lstsq(f,x[:500]-fitted['origin'],lapack_driver='gelsy')
    np.testing.assert_allclose(fitted['coef'],independent,atol=1e-10)


def test_saved_bundle_and_domain(tmp_path):
    m=example();m['history']=[];r=np.array([.8,1.1])
    path=tmp_path/'ribbon.npz';save_model(path,m,r)
    restored,latent=load_model(path)
    np.testing.assert_array_equal(latent,r)
    np.testing.assert_allclose(predict(restored,[1123.,1456.],r),predict(m,[1123.,1456.],r))
    with pytest.raises(ValueError):predict(m,[999.],1.)
    with pytest.raises(ValueError):predict(m,[1200.],0.)


def test_tangent_basis_matches_independent_finite_difference_plane_and_rejects_singularities():
    m=example();y=np.array([1134.5,1567.8]);r=np.array([.8,1.2])
    q=tangent_space(m,y,r)
    finite=np.stack(((predict(m,y+1e-4,r)-predict(m,y-1e-4,r))/2e-4,
                     (predict(m,y,r+1e-5)-predict(m,y,r-1e-5))/2e-5),axis=-1)
    for i in range(len(y)):
        independent=np.linalg.qr(finite[i])[0]
        np.testing.assert_allclose(q[i].T@q[i],np.eye(2),atol=1e-12)
        np.testing.assert_allclose(q[i]@q[i].T,independent@independent.T,atol=1e-8)
    m['coef'][:]=0
    m['coef'][2,0]=1
    with pytest.raises(ValueError):tangent_space(m,[1123.],1.)
