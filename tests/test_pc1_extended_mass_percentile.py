"""Analytic coverage and reproduction checks for the q75 extension."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
import run_pc1_extended_mass_percentile as extension


def test_four_masses_two_calibrations_match_analytic_ball():
    # Twenty coincident reference points make the acceptance set a known ball.
    taus=np.array([.413,.617])
    extension.previous.percentile.initialize_worker(np.zeros((30,3)),taus)
    parts=extension.score_extension(np.zeros(3),np.array([1.,0.,0.]),.8,7)
    assert len(parts)==4
    for mass,part in zip(extension.MASSES,parts):
        half_length=.8*extension.previous.length_factor(mass)
        expected=np.minimum(1.,taus/half_length)
        np.testing.assert_allclose(part['exact_coverage'],np.broadcast_to(expected[:,None],(2,33)),atol=1e-12)
        masks=np.unpackbits(part['acceptance_mask_packed'],axis=2)[:,:,:part['mask_points']].astype(bool)
        assert not np.any(masks[0]&~masks[1])
        np.testing.assert_array_equal(part['direction_coverage'],masks.mean(axis=2))
        assert part['max_grid_error'].max()<=.002
        np.testing.assert_array_equal(part['directions'],parts[0]['directions'])


def test_q80_reproduction_checks_scores_and_masks():
    reference=np.random.default_rng(43).normal(size=(80,3))
    center,vector=np.zeros(3),np.array([1.,0.,0.])
    extension.previous.percentile.initialize_worker(reference,[1.25,1.5,2.])
    saved=extension.previous.base.as_tensors(extension.previous.percentile.score_percentiles(center,vector,2.,0))
    extension.previous.percentile.initialize_worker(reference,[1.,1.25])
    part=extension.previous.percentile.score_percentiles(center,vector,2.,0)
    assert extension.validate_q80(part,saved)==0
    bad=dict(part,exact_coverage=part['exact_coverage'].copy())
    bad['exact_coverage'][1,0]+=.01
    with pytest.raises(AssertionError):extension.validate_q80(bad,saved)
    bad=dict(part,acceptance_mask_packed=part['acceptance_mask_packed'].copy())
    bad['acceptance_mask_packed'][1,0,0]^=1
    with pytest.raises(AssertionError):extension.validate_q80(bad,saved)
