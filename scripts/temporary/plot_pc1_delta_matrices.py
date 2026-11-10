"""Render existing mean Delta values as split-cell mass/calibration matrices."""
import csv
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path('outputs/experiments/pc1_length_percentile_sweep/job_1932529')
OUT=ROOT/'delta_matrices'
COLORS={'hddc':'#007F86','kmeans':'#D67D24'}
NAMES={'hddc':'HDDC-EM','kmeans':'KMeans+PCA'}
CASES=[('years_neutral',4,'Years · layer 4'),('years_neutral',28,'Years · layer 28'),
       ('temperature_02d',4,'Temperature · layer 4'),('temperature_02d',28,'Temperature · layer 28')]
MASSES=[.9,.95,.99]
QUANTILES=[.8,.9,.95]

def main(root=ROOT, masses=MASSES, quantiles=QUANTILES, layer4_only=False, neighbors=20):
    output=root/'delta_matrices'
    n=len(masses)
    assert len(quantiles)==n
    expected=2*sum(layer==4 or not layer4_only for _,layer,_ in CASES)*2*n*n
    with (root/'tabelle_soglia_01.csv').open() as f: rows=[r for r in csv.DictReader(f) if not layer4_only or int(r['layer'])==4]
    assert len(rows)==expected
    lookup={(r['dataset'],int(r['layer']),int(r['K']),float(r['segment_mass']),float(r['quantile']),r['kind']):r for r in rows}
    assert len(lookup)==expected
    output.mkdir(exist_ok=True)
    audit=[]
    if layer4_only:
        panels=[('PC1_delta_layer4_K50_K100',[(dataset,layer,f'{title} · K = {k}',k) for dataset,layer,title in CASES if layer==4 for k in [50,100]])]
        pdf_name='PC1_delta_layer4_K50_K100.pdf'
    else:
        panels=[(f'PC1_delta_matrix_K{k}',[(d,l,t,k) for d,l,t in CASES]) for k in [50,100]]
        pdf_name='PC1_delta_matrices_K50_K100.pdf'
    with PdfPages(output/pdf_name) as pdf:
        for stem,cases in panels:
            fig,axes=plt.subplots(2,2,figsize=(20,12) if n==4 else (17,11))
            for ax,(dataset,layer,title,k) in zip(axes.flat,cases):
                ax.set_xlim(0,n);ax.set_ylim(n,0)
                ax.set_xticks([j+.5 for j in range(n)],[str(round(100*q)) for q in quantiles],fontsize=13)
                ax.set_yticks([i+.5 for i in range(n)],[f'{100*m:.0f}%' for m in masses],fontsize=13)
                ax.xaxis.tick_top();ax.xaxis.set_label_position('top')
                ax.set_xlabel('Percentile di calibrazione',fontsize=13,labelpad=12)
                ax.set_ylabel('Massa gaussiana centrale',fontsize=13,labelpad=12)
                ax.set_title(title,fontsize=19,color='#23364A',weight='bold',pad=62)
                ax.tick_params(length=0,pad=8)
                for spine in ax.spines.values():spine.set_visible(False)
                for i,mass in enumerate(masses):
                    for j,q in enumerate(quantiles):
                        values={kind:100*float(lookup[(dataset,layer,k,mass,q,kind)]['mean_delta_coverage']) for kind in NAMES}
                        winner=max(values,key=values.get)
                        tied=abs(values['hddc']-values['kmeans'])<1e-10
                        for half,kind in enumerate(['hddc','kmeans']):
                            best=not tied and kind==winner
                            x=j+half*.5
                            ax.add_patch(Rectangle((x,i),.5,1,facecolor=COLORS[kind] if best else '#F5F7FA',alpha=.13 if best else 1,edgecolor='none'))
                            ax.text(x+.25,i+.30,NAMES[kind],ha='center',va='center',fontsize=9 if n==4 else 10.5,color=COLORS[kind])
                            ax.text(x+.25,i+.65,f"{values[kind]:+.2f}".replace('.',','),ha='center',va='center',fontsize=17 if n==4 else 19,color=COLORS[kind] if best else '#23364A',weight='bold' if best else 'normal')
                        ax.add_patch(Rectangle((j,i),1,1,fill=False,edgecolor='#CCD7E0',lw=1))
                        ax.plot([j+.5,j+.5],[i,i+1],color='#DDE4EA',lw=.7)
                        audit.append({'dataset':dataset,'layer':layer,'K':k,'mass':mass,'quantile':q,'delta_pp':values,'winner':None if tied else winner})
            fig.suptitle(f'Δ di copertura PC1 · layer 4 · K = 50 e K = 100 · k = {neighbors} vicini' if layer4_only else f'Δ di copertura PC1 · K = {k} · k = {neighbors} vicini',fontsize=23,color='#23364A',weight='bold',y=.965)
            fig.text(.5,.917,'Ogni cella: HDDC-EM a sinistra, KMeans+PCA a destra · valori medi in punti percentuali (pp)',ha='center',fontsize=13,color='#526173')
            fig.text(.5,.045,'Grassetto e sfondo colorato: Δ maggiore nella cella (confronto descrittivo, nessun test di significatività).',ha='center',fontsize=12,color='#526173')
            fig.text(.5,.020,f'k = {neighbors} vicini · PCA75 fissa · soglia di rango 0,1 · massa gaussiana = probabilità marginale centrale, non varianza spiegata.',ha='center',fontsize=11,color='#526173')
            fig.subplots_adjust(left=.10,right=.98,top=.79,bottom=.10,wspace=.23,hspace=.62)
            for ext in ['png','svg']:fig.savefig(output/f'{stem}.{ext}',dpi=190,facecolor='white')
            if not layer4_only:fig.savefig(output/f'{stem}.pdf',facecolor='white')
            pdf.savefig(fig)
            plt.close(fig)
    assert len(audit)==expected//2
    (output/('layer4_manifest.json' if layer4_only else 'manifest.json')).write_text(json.dumps({'job_id':os.environ.get('SLURM_JOB_ID'),'source':str(root/'tabelle_soglia_01.csv'),'field':'mean_delta_coverage * 100','rows':'segment_mass','columns':'quantile','cells':audit},indent=2)+'\n')
    print(f'Exported {len(panels)} panel(s), {len(audit)} split cells:',output,flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--layer4-only',action='store_true')
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args()
    main(root=args.root,masses=[.8,.9,.95,.99] if args.extended else MASSES,
         quantiles=[.75,.8,.9,.95] if args.extended else QUANTILES,layer4_only=args.layer4_only)
