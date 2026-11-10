"""Plot validated aggregate scores without recomputing experiments or fitting curves."""
import csv
import json
import os
from pathlib import Path
from statistics import NormalDist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ROOT = Path('outputs/experiments/pc1_length_percentile_sweep/job_1932529')
BASELINE = Path('outputs/experiments/pc1_percentile_sweep/job_1932231/tabelle_soglia_01.csv')
OUT = ROOT / 'curves'
CASES = [('years_neutral',4,'Years · layer 4','141D'),('years_neutral',28,'Years · layer 28','28D'),
         ('temperature_02d',4,'Temperature · layer 4','4D'),('temperature_02d',28,'Temperature · layer 28','3D')]
COLORS = {'hddc':'#007F86', 'kmeans':'#D67D24'}
NAMES = {'hddc':'HDDC-EM', 'kmeans':'KMeans+PCA'}

def read(path):
    with path.open() as f: return list(csv.DictReader(f))

def main():
    OUT.mkdir(exist_ok=True)
    current = read(ROOT/'tabelle_soglia_01.csv')
    baseline = read(BASELINE)
    assert len(current)==144 and len(baseline)==48
    rows=current+[dict(r, segment_mass='.8') for r in baseline]
    keys=[(r['dataset'],int(r['layer']),r['kind'],int(r['K']),float(r['segment_mass']),float(r['quantile'])) for r in rows]
    assert len(set(keys))==192
    normal=NormalDist()
    masses=[.8,.9,.95,.99]
    x=[normal.inv_cdf((1+m)/2)/normal.inv_cdf(.9) for m in masses]
    files=[]
    with PdfPages(OUT/'PC1_support_curves_all_settings.pdf') as pdf:
        for k in [50,100]:
            for q in [.8,.9,.95]:
                fig,axs=plt.subplots(2,4,figsize=(18,8.3),sharex=True,sharey='row')
                for col,(dataset,layer,title,dim) in enumerate(CASES):
                    for kind in ['hddc','kmeans']:
                        group=sorted([r for r in rows if r['dataset']==dataset and int(r['layer'])==layer and r['kind']==kind and int(r['K'])==k and float(r['quantile'])==q],key=lambda r:float(r['segment_mass']))
                        assert len(group)==4 and [float(r['segment_mass']) for r in group]==masses
                        assert len({(r['eligible'],r['total']) for r in group})==1
                        for ax,field in [(axs[0,col],'mean_coverage_pc1'),(axs[1,col],'mean_delta_coverage')]:
                            ax.plot(x,[100*float(r[field]) for r in group],color=COLORS[kind],marker='o',lw=2)
                        axs[0,col].plot(x,[100*float(r['mean_coverage_random']) for r in group],color=COLORS[kind],marker='.',ls='--',lw=1.5)
                    axs[0,col].set_title(f'{title}\nPCA75: {dim}',fontsize=13,color='#23364A',pad=16)
                    axs[0,col].set_ylim(40,102)
                    axs[1,col].set_ylim(-2,21)
                    axs[1,col].axhline(0,color='#6A7785',lw=.8)
                    axs[1,col].set_xlabel('Segment length / original 80%-mass length',fontsize=10)
                    for ax in axs[:,col]:
                        ax.set_xticks(x,[f'{z:.2f}×' for z in x],fontsize=10)
                        ax.grid(alpha=.18)
                        ax.spines[['top','right']].set_visible(False)
                    sec=axs[0,col].secondary_xaxis('top')
                    sec.set_xticks(x,['80%','90%','95%','99%'])
                    sec.tick_params(labelsize=10)
                axs[0,0].set_ylabel('Mean coverage (%)',fontsize=12)
                axs[1,0].set_ylabel('Mean PC1 − random (pp)',fontsize=12)
                fig.suptitle(f'PC1 support versus segment length · k = 20 · K = {k} · calibration percentile = {100*q:.0f}',fontsize=18,color='#23364A',weight='bold',y=.975)
                legend=[Line2D([0],[0],color=c,lw=2,label=NAMES[kind]) for kind,c in COLORS.items()]
                legend.extend([Line2D([0],[0],color='#526173',lw=2,label='PC1'),Line2D([0],[0],color='#526173',ls='--',lw=1.5,label='32-direction random mean')])
                fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.928),ncol=4,frameon=False,fontsize=11)
                fig.text(.5,.045,'Top axis: central Gaussian marginal mass. Each point is a measured setting; connecting lines are guides, not fitted curves.',ha='center',fontsize=10,color='#526173')
                fig.text(.5,.018,'Ordinary component means · rank threshold 0.1 · 80% mass is the audited reproduction control · shared axes across all figures · no inferential bands',ha='center',fontsize=10,color='#526173')
                fig.subplots_adjust(left=.065,right=.985,bottom=.14,top=.79,wspace=.10,hspace=.25)
                stem=f'PC1_curves_K{k}_q{round(100*q)}'
                for ext in ['png','pdf','svg']:
                    path=OUT/f'{stem}.{ext}'
                    fig.savefig(path,dpi=170,facecolor='white')
                    files.append(str(path))
                pdf.savefig(fig)
                plt.close(fig)
    (OUT/'manifest.json').write_text(json.dumps({'job_id':os.environ.get('SLURM_JOB_ID'),'sources':[str(ROOT/'tabelle_soglia_01.csv'),str(BASELINE)],'measured_settings':192,'files':files,'aggregation':'ordinary component means; no uncertainty estimate','length_factors':x},indent=2)+'\n')
    print(f'Exported six curve panels and combined six-page PDF to {OUT}',flush=True)

if __name__=='__main__':main()
