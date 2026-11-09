"""Raw-row and independent QR audit of an explicit spiral-ribbon fit."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import lstsq
from sklearn.decomposition import PCA

import fit_years_spiral as curves
import fit_years_spiral_ribbon as ribbon


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    output=parser.parse_args().output
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run through srun')
    manifest=json.loads((output/'manifest.json').read_text())
    results=json.loads((output/'results.json').read_text())
    assert curves.sha256(output/'source/fit_years_spiral_ribbon.py')==manifest['script_hash']
    assert all(curves.sha256(p)==h for p,h in manifest['inputs'].items())
    x,years,templates,index=curves.load_activations(Path(manifest['activation_dir']))
    x=x.astype(float)
    masks,membership=curves.split_development(years,templates)
    assert membership==manifest['membership']
    assert [r['source_row_id'] for r in index]==manifest['source_row_ids']
    model,radius=ribbon.load_model(output/'final_ribbon_all_development.npz')
    f=ribbon.design(years,model['periods'],radius,model['radial_period'])
    independent,_,rank,_=lstsq(f,x-model['origin'],lapack_driver='gelsy')
    coefficient_error=float(np.max(np.abs(independent-model['coef'])))
    assert coefficient_error<1e-8,coefficient_error
    evaluated,_=ribbon.load_model(output/'evaluation_ribbon.npz')
    pca=PCA(3,svd_solver='randomized',random_state=42).fit(x[masks['fit']])
    discrepancies=[]
    for name,expected in results['evaluation'].items():
        actual=ribbon.evaluate(evaluated,x[masks[name]],years[masks[name]],x[masks['fit']].mean(0),pca.components_.T)
        discrepancies.extend(abs(actual[k]-expected[k]) for k in ('mse','r2','projected_r2'))
    assert max(discrepancies)<1e-10
    ys=np.random.default_rng(12).uniform(1001,1999,32)
    rs=np.random.default_rng(13).uniform(*model['radius_bounds'],32)
    errors={}
    for name,step in [('year',1e-4),('radius',1e-5)]:
        finite=((ribbon.predict(model,ys+step,rs)-ribbon.predict(model,ys-step,rs)) if name=='year'
                else (ribbon.predict(model,ys,rs+step)-ribbon.predict(model,ys,rs-step)))/(2*step)
        errors[name]=float(np.max(np.abs(finite-ribbon.predict(model,ys,rs,derivative=name))))
    assert max(errors.values())<1e-7
    with np.load(output/'display_axes.npz') as axes:
        frames=json.loads((output/'visualization.json').read_text())
        view_scores={name:ribbon.evaluate(model,x,years,x.mean(0),axes[name])['projected_r2'] for name in ('pca','spiral')}
        for name,value in view_scores.items():
            frames['frames'][name]['descriptive_projected_r2']=value
    (output/'visualization.json').write_text(json.dumps(frames,separators=(',',':'),allow_nan=False))
    audit=dict(source_and_input_hashes_match=True,split_source_ids_match=True,
        independent_qr_ambient_dimensions=x.shape[1],independent_qr_rank=int(rank),
        maximum_coefficient_discrepancy=coefficient_error,
        maximum_evaluation_discrepancy=max(discrepancies),derivative_discrepancies=errors,
        descriptive_view_r2=view_scores,job_id=os.environ.get('SLURM_JOB_ID'))
    (output/'artifact_audit.json').write_text(json.dumps(audit,indent=2))
    shutil.copy2(__file__,output/'source'/Path(__file__).name)
    shutil.copy2(Path(__file__).parents[2]/'tests/test_years_spiral_ribbon.py',output/'source/test_years_spiral_ribbon.py')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    main()
