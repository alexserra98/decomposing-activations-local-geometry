"""Illustrative step-by-step PC1 support metric diagram (not experimental data)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

OUT=Path('outputs/experiments/pc1_length_percentile_sweep/job_1932529/slides')
NAVY='#23364A'; TEAL='#007F86'; ORANGE='#D67D24'; RED='#C55D69'; GRAY='#647488'
rng=np.random.default_rng(42)
fig=plt.figure(figsize=(20,13),facecolor='white')
fig.text(.035,.956,'How is PC1 support coverage computed?',fontsize=27,weight='bold',color=NAVY)
fig.text(.035,.918,'A calibrated kNN test along a component’s PC1, compared with matched random directions',fontsize=15,color=GRAY)
positions=[(.035,.515),(.355,.515),(.675,.515),(.035,.105),(.355,.105),(.675,.105)]
titles=['Fix the evaluation space','Calibrate the support threshold','Build the PC1 segment','Test positions along PC1','Repeat for random directions','Report coverage and excess']
for i,((x,y),title) in enumerate(zip(positions,titles)):
    fig.add_artist(FancyBboxPatch((x,y),.29,.365,boxstyle='round,pad=0.012',transform=fig.transFigure,facecolor='#F5F8FA',edgecolor='#DCE5EC',zorder=-1))
    fig.text(x+.005,y+.328,f'{i+1}  {title}',fontsize=16,weight='bold',color=NAVY)

def axes(i):
    x,y=positions[i]; ax=fig.add_axes([x+.012,y+.115,.264,.19]);ax.set_xlim(-3,3);ax.set_ylim(-1.7,1.7);ax.axis('off');return ax

def caption(i,text):
    x,y=positions[i];fig.text(x+.005,y+.09,text,fontsize=11.5,color=NAVY,va='top',linespacing=1.45)

# Shared schematic cloud; geometry is intentionally illustrative.
t=np.linspace(-2.5,2.5,110)
cloud=np.c_[t,.22*t*t-.55]+rng.normal(0,.18,(110,2))
cloud=cloud[rng.permutation(len(cloud))]
ax=axes(0);ax.scatter(*cloud[:88].T,s=13,c=TEAL,alpha=.65,label='Reference R');ax.scatter(*cloud[88:].T,s=22,c=ORANGE,marker='^',label='Calibration Q');ax.legend(loc='upper left',fontsize=9,frameon=False)
caption(0,'PCA75: training PCA retaining ≥75% variance;\nunwhitened scores and Euclidean distances.\nSplit evaluation points: R (80%), Q (20%).\nHere: reconstructed validation, reused for selection.')
ax=axes(1); vals=np.sort(rng.gamma(6,.12,300));ax.hist(vals,bins=22,color='#A9C9D0',edgecolor='white'); tau=np.quantile(vals,.9,method='higher');ax.axvline(tau,color=ORANGE,lw=2);ax.set_xlim(0,1.8);ax.set_ylim(0,45);ax.text(tau+.03,34,r'$\tau_q$',fontsize=17,color=ORANGE);ax.text(.06,40,'Calibration distances',fontsize=11,color=GRAY)
caption(1,'For every z in Q, measure its Euclidean distance\nto the 20th nearest point in R. Set τq to the\nq-th percentile of these distances (q = 80, 90, 95).')
ax=axes(2); xx=np.linspace(-2.8,2.8,300); yy=np.exp(-xx**2/2);ax.plot(xx,yy,color=NAVY,lw=2);ax.fill_between(xx,0,yy,where=np.abs(xx)<=1.96,color=TEAL,alpha=.24);ax.plot([-1.96,1.96],[-.25,-.25],color=TEAL,lw=3);ax.scatter([0],[-.25],c=NAVY,s=35);ax.text(-.2,-.65,r'$\mu$',fontsize=15);ax.text(-2.1,1.2,'Central mass m of the 1D PC1 marginal',fontsize=12,color=GRAY)
caption(2,r'$\mu+t v_1,\quad |t|\leq a_m=\Phi^{-1}((1+m)/2)\sqrt{\lambda_1}$'+'\nμ: component center; v₁: unit PC1 eigenvector.\nλ₁: its original covariance eigenvalue; Φ: normal CDF.\nm: Gaussian probability mass (90%, 95%, 99%).\nProject this entire segment into shared PCA space.')
ax=axes(3)
ax.set_aspect('equal', adjustable='box')
reference=cloud[:88]
x=np.linspace(-2.7,2.7,21)
queries=np.c_[x,np.zeros_like(x)]
threshold=1.05
counts=(np.linalg.norm(queries[:,None,:]-reference[None,:,:],axis=2)<=threshold).sum(axis=1)
good=counts>=20
ax.scatter(*reference.T,s=10,c='#9BAEB9',alpha=.75)
ax.plot(x,np.zeros_like(x),c=GRAY,lw=1)
ax.scatter(x[good],np.zeros(good.sum()),c=TEAL,s=40,zorder=3)
ax.scatter(x[~good],np.zeros((~good).sum()),c=RED,marker='x',s=45,zorder=3)
j=10
ax.add_patch(Circle(queries[j],threshold,fill=False,ec=ORANGE,ls='--',lw=1.7))
ax.plot([0,threshold],[0,0],color=ORANGE,lw=2,zorder=4)
ax.text(.35,.12,r'$\tau_q$',fontsize=13,color=ORANGE)
ax.text(-2.85,1.43,'Teal = accepted; red × = rejected',fontsize=10,color=GRAY)
ax.text(-2.85,-1.55,f'Circle at one query: {counts[j]} reference points inside',fontsize=9,color=GRAY)
caption(3,r'Accept if ≥20 reference points are within $\tau_q$.'+'\nDashed circle: ball centered on one query (2D sketch).\n'+r'$C_{\mathrm{PC1}}=\mathrm{accepted\ positions}/\mathrm{all\ positions}$'+'\nUse equally spaced positions, including endpoints.')
ax=axes(4);ax.scatter(*cloud.T,s=10,c='#BCCBD2',alpha=.6)
for angle in np.linspace(.2,np.pi-.15,10):
    v=2.4*np.array([np.cos(angle),np.sin(angle)]);ax.plot([-v[0],v[0]],[-v[1],v[1]],color=ORANGE,alpha=.45,lw=1)
ax.plot([-2.4,2.4],[0,0],c=TEAL,lw=3);ax.scatter([0],[0],c=NAVY,s=40,zorder=4)
caption(4,'Draw 32 uniform random unit directions in PCA space.\nMatch PC1’s projected center and end-to-end length.\nApply the same position test to each direction.\nOnly a subset of directions is sketched above.')
ax=axes(5);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.barh([.73,.35],[.8,.6],height=.19,color=[TEAL,ORANGE]);ax.text(.02,.88,'PC1: 80%',fontsize=13,color=NAVY);ax.text(.02,.50,'Random mean: 60%',fontsize=13,color=NAVY);ax.text(.02,.05,'Illustrative example: Δ = +20 pp',fontsize=13,weight='bold',color=TEAL)
caption(5,r'$\Delta C=C_{\mathrm{PC1}}-\frac{1}{32}\sum_r C_r$'+'\nC is a fraction; report 100C in %. Report 100Δ in pp.\nAverage per-component C and Δ with equal weights.\nInclude defined PC1s with nonzero PCA projection.\nExample bars illustrate arithmetic only.')
fig.text(.035,.047,'Reading the result: positive Δ means PC1 is more supported than matched random directions.',fontsize=14,weight='bold',color=NAVY)
fig.text(.035,.018,'Schematic only • Actual runs: k = 20, 4001 positions per direction • This measures support, not a known-tangent angle.',fontsize=11,color=GRAY)
OUT.mkdir(exist_ok=True)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'PC1_metric_step_by_step.{ext}',dpi=190,facecolor='white')
print('Saved English step-by-step cartoon:',OUT,flush=True)
