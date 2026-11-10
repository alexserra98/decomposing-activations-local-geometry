"""Neighbor dispatch, direct masks and calibration for the fixed 4x4 experiment."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.distance import cdist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
import run_pc1_neighbor_mass_percentile as sweep


@pytest.mark.parametrize('k',[5,10,20,40])
def test_calibration_uses_requested_neighbor_and_same_quantile_rule(k):
    rng=np.random.default_rng(99)
    reference,queries=rng.normal(size=(80,3)),rng.normal(size=(30,3))
    scores,thresholds=sweep.calibrate(reference,queries,k)
    expected=np.sort(cdist(queries,reference),axis=1)[:,k-1]
    np.testing.assert_array_equal(scores,expected)
    np.testing.assert_array_equal(thresholds,np.quantile(expected,sweep.QUANTILES,method='higher'))


@pytest.mark.parametrize('k',[5,10,20,40])
def test_all_four_mass_masks_match_direct_requested_knn(k):
    rng=np.random.default_rng(109)
    reference,queries=rng.normal(size=(80,3)),rng.normal(size=(30,3))
    _,thresholds=sweep.calibrate(reference,queries,k)
    scorer=sweep.extension.previous.percentile
    scorer.initialize_worker(reference,thresholds,k)
    center,vector=np.zeros(3),np.array([1.,0.,0.])
    parts=sweep.extension.score_extension(center,vector,2.,7)
    for mass,part in zip(sweep.MASSES,parts):
        offsets=np.linspace(-2*sweep.extension.previous.length_factor(mass),2*sweep.extension.previous.length_factor(mass),part['mask_points'])
        points=center+offsets[:,None]*vector
        kth=np.sort(cdist(points,reference),axis=1)[:,k-1]
        direct=kth[None,:]<=thresholds[:,None]
        packed=part['acceptance_mask_packed'][:,0,:]
        masks=np.unpackbits(packed,axis=1)[:,:part['mask_points']].astype(bool)
        np.testing.assert_array_equal(masks,direct)
        assert part['max_grid_error'].max()<=.002
        assert not np.any(masks[:-1]&~masks[1:])
