"""Projection and exact wiki-metric semantics for the saved-model comparison."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
from compare_years_ribbon_tangents import (project_centers,component_scores,isotropic_eigenspaces)
from fit_years_spiral_ribbon import predict
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC
from dalg.evaluation.subspaces import leading_covariance_eigenspaces


def helix():
    coef=np.zeros((4,7));coef[1,2]=3;coef[2,0]=1;coef[3,1]=1
    return dict(coef=coef,origin=np.zeros(7),periods=[100.],radial_period=100.,radius_bounds=[.6,1.4])


def test_projection_eliminates_radius_in_full_ambient_space():
    model=helix();years=np.array([1234.5,1765.25]);radii=np.array([.8,1.2])
    centers=predict(model,years,radii);centers[:,6]=.2
    rows=project_centers(model,centers,step=.5)
    np.testing.assert_allclose([p['year'] for p in rows],years,atol=5e-5)
    np.testing.assert_allclose([p['radius'] for p in rows],radii,atol=1e-7)
    np.testing.assert_allclose([p['distance'] for p in rows],[.2,.2],atol=1e-8)
    assert not any(p['ambiguous'] or p['boundary'] for p in rows)


def test_projection_boundary_and_nonunique_footpoints():
    model=helix();axis=np.zeros((1,7))
    row=project_centers(model,axis,step=.5)[0]
    assert row['boundary_radius'] and not row['ambiguous']
    np.testing.assert_allclose(row['year'],1500.,atol=1e-4)
    model['coef'][1]=0
    point=predict(model,[1234.5],1.)
    assert project_centers(model,point,step=.5)[0]['ambiguous']


def test_wiki_alignment_containment_rank_and_eigengap_rules():
    eye=torch.eye(4,dtype=torch.float64);tangent=eye[:,:2]
    pcs=eye[:,[2,0,1,3]];values=torch.tensor([5.,4.,3.,1.],dtype=torch.float64)
    scores=component_scores(tangent,pcs,values,3,4)
    np.testing.assert_allclose(scores['tangent_alignment'],[.5,0.])
    np.testing.assert_allclose(scores['tangent_containment'],[1.,1.])
    rank_one=component_scores(tangent,eye,values,1,4)
    assert rank_one['tangent_alignment'] is None and rank_one['tangent_containment'] is None
    np.testing.assert_allclose(rank_one['tangent_partial_containment'],[1.,1.])
    assert rank_one['tangent_adjusted_alignment']==(.5,None)
    assert component_scores(tangent,eye,values,0,4)['tangent_adjusted_alignment']==(0.,None)
    tied=values.clone();tied[2]=tied[1]
    scores=component_scores(tangent,pcs,tied,3,4)
    assert scores['tangent_alignment'] is None and scores['tangent_containment'] is not None
    tied=values.clone();tied[3]=tied[2]
    scores=component_scores(tangent,pcs,tied,3,4)
    assert scores['tangent_alignment'] is not None and scores['tangent_containment'] is None


def test_isotropic_full_rank_pcs_match_existing_covariance_solver():
    torch.manual_seed(25)
    model=MFA_HDDC(torch.randn(3,9),rank=4,isotropic_psi=True)
    with torch.no_grad():
        model.scale_rho.copy_(torch.randn(3,4));model.rank_mask[1,2:]=0
    values,pcs=isotropic_eigenspaces(model)
    full_values,full_pcs=leading_covariance_eigenspaces(model,4)
    torch.testing.assert_close(values,full_values,rtol=1e-12,atol=1e-12)
    for k,rank in enumerate([4,2,4]):
        torch.testing.assert_close(pcs[k,:,:rank]@pcs[k,:,:rank].T,
                                  full_pcs[k,:,:rank]@full_pcs[k,:,:rank].T,rtol=1e-12,atol=1e-12)
