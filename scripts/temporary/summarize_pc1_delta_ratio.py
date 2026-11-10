"""Ratio of mean Delta across equally weighted layer-4 settings, not mean ratios."""
import csv,json,os
from pathlib import Path
from statistics import mean
ROOT=Path('outputs/experiments/pc1_extended_mass_percentile/job_1934466')
with (ROOT/'tabelle_soglia_01.csv').open() as f:
    rows=[r for r in csv.DictReader(f) if int(r['layer'])==4]
assert len(rows)==128
keys={(r['dataset'],int(r['K']),float(r['segment_mass']),float(r['quantile'])) for r in rows}
assert len(keys)==64
for key in keys:
    pair=[r for r in rows if (r['dataset'],int(r['K']),float(r['segment_mass']),float(r['quantile']))==key]
    assert {r['kind'] for r in pair}=={'hddc','kmeans'} and len(pair)==2

def summarize(subset):
    values={kind:[100*float(r['mean_delta_coverage']) for r in subset if r['kind']==kind] for kind in ['hddc','kmeans']}
    averages={kind:mean(v) for kind,v in values.items()}
    return dict(settings_per_model=len(values['hddc']),mean_delta_pp=averages,
                ratio_of_means=averages['hddc']/averages['kmeans'],
                difference_pp=averages['hddc']-averages['kmeans'])
result=dict(job_id=os.environ.get('SLURM_JOB_ID'),scope='layer4; both datasets; K50 and K100; all 4 masses x 4 calibration percentiles',
            weighting='equal weight per dataset/K/mass/percentile setting; ordinary within-setting component mean',
            overall=summarize(rows),by_K={k:summarize([r for r in rows if int(r['K'])==k]) for k in [50,100]},
            by_dataset={d:summarize([r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})})
(ROOT/'layer4_delta_ratio.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
