"""Export final tables and scientific figures from the completed PCA75 pilots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PCA = REPO / 'outputs/experiments/years_layer04_pc1_density_pca75/job_1857111'
SWAP = REPO / 'outputs/experiments/years_layer04_pc1_direction_swap/job_1857158'
COLORS = {'hddc': '#007F86', 'kmeans': '#D67D24'}
NAMES = {'hddc': 'HDDC', 'kmeans': 'KMeans'}
TAUS = [.0001, .001, .01, .1]
TAU_LABELS = [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$']


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group(rows, K, tau, kind=None):
    return [r for r in rows if int(r['K']) == K and float(r['threshold']) == tau
            and (kind is None or r['kind'] == kind)]


def values(rows, field):
    return np.array([float(r[field]) for r in rows])


def save_figure(fig, output, name):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'{name}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def plot_original(summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), sharex=True, layout='constrained')
    for row, K in enumerate([50, 100]):
        for col, (field, title) in enumerate([('median_retention', 'Retention normalizzata A'),
                                             ('median_enrichment', 'Enrichment E')]):
            ax = axes[row, col]
            for kind in ['hddc', 'kmeans']:
                ys = [float(group(summary, K, tau, kind)[0][field]) for tau in TAUS]
                ax.plot(range(4), ys, marker='o' if kind == 'hddc' else 's', ms=6,
                        color=COLORS[kind], label=NAMES[kind], lw=2)
                ax.annotate(f'{ys[-1]:.2f}', (3, ys[-1]), xytext=(-8, 8),
                            textcoords='offset points', ha='right', color=COLORS[kind])
            ax.set(yscale='log', title=f'K = {K} · {title}', ylabel='Mediana in log-space, esponenziata')
            ax.axhline(1, color='#888888', lw=.9, ls='--')
            ax.set_xticks(range(4), TAU_LABELS)
            ax.grid(axis='y', alpha=.16, which='both')
            if row == 1:
                ax.set_xlabel(r'Soglia di rango $\tau$')
            ax.legend(loc='upper left', frameon=False)
    axes[0, 0].set_ylim(.8, 140)
    axes[1, 0].set_ylim(.8, 140)
    axes[0, 1].set_ylim(.8, 800)
    axes[1, 1].set_ylim(.8, 800)
    fig.suptitle('Score sui segmenti propri dei modelli\nYears · layer 4 · PCA 141D (75,096% varianza) · kNN k=20', fontsize=14)
    fig.supxlabel('Ogni punto riassume i componenti del modello; le soglie sono dipendenti.', fontsize=10)
    save_figure(fig, output, '01_score_originali')


def plot_paired(pairs, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharey=True, sharex=True, layout='constrained')
    for ax, K in zip(axes, [50, 100]):
        for index, tau in enumerate(TAUS):
            selected = group(pairs, K, tau)
            for mutual, offset, color, marker in [(False, -.12, '#334E68', 'o'),
                                                 (True, .12, '#8B4F96', 'D')]:
                subset = [r for r in selected if r['mutual_nearest'] == 'True'] if mutual else selected
                q1, med, q3 = np.quantile(values(subset, 'paired_delta'), [.25, .5, .75])
                label = ('Reciprocamente più vicine' if mutual else 'Tutte le coppie') if index == 0 else None
                ax.errorbar(med, index + offset, xerr=np.array([[med-q1], [q3-med]]),
                            fmt=marker, color=color, capsize=3, ms=6, lw=1.6, label=label)
        ax.axvline(0, color='#555555', ls='--', lw=1)
        ax.set(title=f'K = {K}', xlabel=r'Differenza appaiata $\Delta$ (log naturale)'
               '\nΔ < 0 favorisce KMeans; Δ > 0 favorisce HDDC')
        ax.set_yticks(range(4), TAU_LABELS)
        ax.grid(axis='x', alpha=.15)
    axes[0].set_ylabel(r'Soglia di rango $\tau$')
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False, fontsize=10)
    fig.suptitle('Direzioni a parità di centro e lunghezza\nMediane e quartili 25–75% fra coppie (dispersione, non intervalli di confidenza)', fontsize=13)
    save_figure(fig, output, '02_confronto_appaiato')


def plot_interaction(pairs, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), layout='constrained')
    for ax, K in zip(axes, [50, 100]):
        selected = group(pairs, K, .1)
        x, y = values(selected, 'delta_at_h_center'), values(selected, 'delta_at_k_center')
        mutual = np.array([r['mutual_nearest'] == 'True' for r in selected])
        limit = 1.13 * max(abs(x).max(), abs(y).max())
        ax.fill_between([0, limit], -limit, 0, color='#EAF2FA', zorder=0)
        ax.scatter(x[~mutual], y[~mutual], s=32, facecolors='none', edgecolors='#8996A2', lw=1.2, alpha=.8)
        ax.scatter(x[mutual], y[mutual], s=32, color='#007F86', alpha=.85, edgecolors='white', lw=.35)
        ax.plot([-limit, limit], [limit, -limit], ls=':', color='#7F8C8D', lw=.9)
        ax.axhline(0, color='#555555', lw=.8)
        ax.axvline(0, color='#555555', lw=.8)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', title=f'K = {K}',
               xlabel=r'$\delta_H$: log rapporto al centro HDDC',
               ylabel=r'$\delta_K$: log rapporto al centro KMeans')
        labels = [(.03, .97, f'Ognuna all’altro centro\n{int(((x<0)&(y>0)).sum())}/{K}', 'left', 'top'),
                  (.97, .97, f'HDDC ai due centri\n{int(((x>0)&(y>0)).sum())}/{K}', 'right', 'top'),
                  (.03, .03, f'KMeans ai due centri\n{int(((x<0)&(y<0)).sum())}/{K}', 'left', 'bottom'),
                  (.97, .03, f'Ognuna al proprio centro\n{int(((x>0)&(y<0)).sum())}/{K}', 'right', 'bottom')]
        for xx, yy, text, ha, va in labels:
            ax.text(xx, yy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.82, pad=2))
    fig.suptitle('La direzione dipende dal centro · soglia τ = 0,1\nOgni punto è una coppia HDDC–KMeans', fontsize=14)
    handles = [Line2D([], [], marker='o', ls='', color='#007F86', label='Coppie reciprocamente più vicine'),
               Line2D([], [], marker='o', ls='', markerfacecolor='none', color='#8996A2', label='Altre coppie'),
               Line2D([], [], ls=':', color='#7F8C8D', label='Δ = 0')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=9)
    save_figure(fig, output, '03_interazione_centro')


def plot_diagnostics(components, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), layout='constrained')
    fields = [('center_log_density', 'Densità al centro', r'$\log\hat p(z)$'),
              ('absolute_log_average', 'Densità media assoluta sul segmento', r'$\log\bar p$'),
              ('projected_half_length', 'Semilunghezza del segmento originale proiettato', r'$L_i$'),
              ('alpha', 'Frazione della PC1 originale conservata', r'$\alpha_i=\|Pv_i\|^2$')]
    rng = np.random.default_rng(2026)
    for ax, (field, title, ylabel) in zip(axes.flat, fields):
        for K, center in [(50, 0), (100, 1)]:
            for kind, shift in [('hddc', -.17), ('kmeans', .17)]:
                selected = group(components, K, .1, kind)
                y = (values(selected, 'log_retention') + values(selected, 'center_log_density')
                     if field == 'absolute_log_average' else values(selected, field))
                x = center + shift
                color = COLORS[kind]
                ax.scatter(x + rng.uniform(-.055, .055, len(y)), y, s=9, alpha=.28, color=color, zorder=1)
                ax.boxplot([y], positions=[x], widths=.22, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=color+'25', edgecolor=color),
                           medianprops=dict(color=color, lw=2), whiskerprops=dict(color=color),
                           capprops=dict(color=color), zorder=2)
        ax.set_xticks([0, 1], ['K = 50', 'K = 100'])
        ax.set(title=title, ylabel=ylabel, xlim=(-.48, 1.48))
        ax.grid(axis='y', alpha=.13)
        if field == 'alpha':
            ax.set_ylim(0, 1.03)
    fig.suptitle('Diagnostiche dei segmenti propri · soglia τ = 0,1\nScatole: mediana e quartili · punti: singoli componenti', fontsize=14)
    handles = [Line2D([], [], marker='s', ls='', color=COLORS[k], label=NAMES[k]) for k in ['hddc', 'kmeans']]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, frameon=False)
    save_figure(fig, output, '04_diagnostiche')


def build_tables(summary, components, swap_summary, pairs):
    combined = []
    for K in [50, 100]:
        for tau in TAUS:
            h, k = (group(summary, K, tau, kind)[0] for kind in ['hddc', 'kmeans'])
            s = group(swap_summary, K, tau)[0]
            selected = group(pairs, K, tau)
            mutual = [r for r in selected if r['mutual_nearest'] == 'True']
            for kind, entry in [('hddc', h), ('kmeans', k)]:
                original = group(components, K, tau, kind)
                assert len(original) == K and all(not r['exclusion_reason'] for r in original)
                for field in ['retention', 'enrichment']:
                    np.testing.assert_allclose(np.median(values(original, 'log_'+field)),
                                               float(entry['median_log_'+field]), atol=1e-12, rtol=0)
            assert len(selected) == K
            np.testing.assert_allclose(np.median(values(selected, 'paired_delta')),
                                       float(s['median_paired_log_ratio']), atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.median(values(mutual, 'paired_delta')),
                                       float(s['mutual_median_paired_log_ratio']), atol=1e-12, rtol=0)
            combined.append(dict(K=K, threshold=tau,
                                 retention_hddc=float(h['median_retention']), retention_kmeans=float(k['median_retention']),
                                 enrichment_hddc=float(h['median_enrichment']), enrichment_kmeans=float(k['median_enrichment']),
                                 paired_median_log_ratio=float(s['median_paired_log_ratio']),
                                 paired_ratio=float(s['median_paired_ratio']),
                                 mutual_paired_ratio=math.exp(float(s['mutual_median_paired_log_ratio'])),
                                 mutual_pairs=len(mutual), pairs=K, h_wins=int(s['h_wins_over_tolerance']),
                                 numerical_close=int(s['numerical_tie_pairs']), k_wins=int(s['k_wins_over_tolerance']),
                                 own_center_wins=int(s['own_center_wins'])))
    diagnostics = []
    for K in [50, 100]:
        for kind in ['hddc', 'kmeans']:
            selected = group(components, K, .1, kind)
            diagnostics.append(dict(K=K, kind=kind,
                median_center_log_density=float(np.median(values(selected, 'center_log_density'))),
                median_absolute_log_average=float(np.median(values(selected, 'center_log_density') + values(selected, 'log_retention'))),
                median_projected_half_length=float(np.median(values(selected, 'projected_half_length'))),
                median_center_knn_radius=float(np.median(values(selected, 'center_knn_radius'))),
                median_alpha=float(np.median(values(selected, 'alpha'))),
                retention_over_one_fraction=float((values(selected, 'log_retention') > 0).mean()),
                enrichment_over_one_fraction=float((values(selected, 'log_enrichment') > 0).mean())))
    return combined, diagnostics


def make_report(output, combined, diagnostics):
    lines = [r'''# PC1 e supporto di densità: risultati finali

Years_neutral, layer 4 di Llama-3.1-8B. 16 configurazioni salvate: HDDC-EM e
KMeans+PCA, K=50/100 e soglia di rango τ=0.0001/0.001/0.01/0.1.
Sono stati riutilizzati attivazioni, split e checkpoint, senza riaddestramento.
La variante finale usa una PCA globale di 141 dimensioni, con il 75.09623332%
della varianza del training. La versione ambientale in 4096D resta conservata
nei suoi output originali e usa uno stimatore di densità diverso.

## Dati, proiezione e densità

La PCA è stata stimata una sola volta sulle 18.018 attivazioni del training,
centrate senza standardizzazione o whitening. Se η sono gli autovalori della
covarianza globale ordinati in modo decrescente:

$$d=\min\left\{j:\frac{\sum_{r=1}^{j}\eta_r}{\sum_{r=1}^{4096}\eta_r}\geq0.75\right\}=141.$$

Con la media training m e P di forma (141,4096), con righe ortonormali, si
proietta come z=P(x-m). 140 dimensioni spiegano il 74.98057038% della varianza.
Questa è una soglia di varianza globale, non una stima di intrinsic dimension.

Il riferimento di densità è comune a tutti i modelli: le 2.002 attivazioni di
validation proiettate con la stessa m e P. Gli ID e l'ordine delle righe sono
verificati contro gli split salvati. Con r20(z) distanza euclidea dal ventesimo
vicino, n=2002 e d=141:

$$\hat p(z)=\frac{20}{nV_dr_{20}(z)^d},\qquad V_d=\frac{\pi^{d/2}}{\Gamma(d/2+1)}.$$

$$\log\hat p(z)=\log(20/n)-\log V_d-d\log r_{20}(z).$$

Il volume della palla e l'esponente usano 141 dimensioni. I calcoli delle
densità e delle medie restano in log-space.

## PC1 originale e intervallo centrale dell'80%

Per ogni componente si conserva l'eigenpair originale (v_i,λ_i), con v_i
unitario. KMeans usa la covarianza empirica ML del cluster; HDDC usa la
covarianza completa, rumore incluso:

$$\Sigma_i^{K}=\frac{1}{n_i}\sum_{x\in C_i}(x-\mu_i)(x-\mu_i)^T,\qquad
\Sigma_i^{H}=W_iW_i^T+\sigma_i^2I.$$

$$a_i=\Phi^{-1}(0.9)\sqrt{\lambda_i}\simeq1.28155\sqrt{\lambda_i},\quad
z_i=P(\mu_i-m),\quad b_i=Pv_i,\quad\alpha_i=\|b_i\|^2.$$

Il segmento proiettato è z_i+t b_i per t∈[-a_i,a_i]. Equivalentemente:

$$u_i=\frac{b_i}{\|b_i\|},\qquad L_i=a_i\sqrt{\alpha_i},\qquad
z_i+s u_i,\quad s\in[-L_i,L_i].$$

L'80% riguarda la marginale unidimensionale della PC1 originale. Non si
ricalcola una PC1 sulle covarianze proiettate. Proiezioni nulle sarebbero
escluse esplicitamente; tutti i 1.200 componenti del pilot sono valutabili.
Alpha misura quanta norma della direzione originale sopravvive alla PCA,
mentre il 75.096% riguarda la varianza globale: sono quantità diverse.

## Retention ed enrichment

Per un centro z, una direzione unitaria u e una semilunghezza L:

$$\bar p(u;z,L)=\frac{1}{2L}\int_{-L}^{L}\hat p(z+s u)\,ds,\qquad
A(u;z,L)=\frac{\bar p(u;z,L)}{\hat p(z)}.$$

La divisione per 2L rende l'integrale una densità media; la divisione per la
densità al centro produce la retention normalizzata, score primario.
Se A>1 la densità media lungo il segmento supera quella al centro. A può
superare uno e non è una probabilità né una frazione di manifold trattenuta.
La normalizzazione per lunghezza non elimina la dipendenza dal dominio L.

Si campionano 32 direzioni unitarie isotrope q_r direttamente nello spazio
ridotto, allo stesso centro e con la stessa L della PC1. Seed=42+indice del
componente, condiviso fra configurazioni. L'enrichment secondario è:

$$E_i=\frac{A(u_i;z_i,L_i)}{\frac1{32}\sum_{r=1}^{32}A(q_r;z_i,L_i)}.$$

Il denominatore è una media aritmetica di densità/retention, calcolata con
log-sum-exp; non è la media dei logaritmi. E>1 indica vantaggio sulla baseline
casuale. Il fattore densità al centro si cancella da E, ma la posizione
geometrica del centro continua a influenzare quali regioni attraversa la linea.

Le tabelle riportano exp(mediana(log A)) ed exp(mediana(log E)), con peso
uguale per componente. Per un numero pari di componenti questo implica
interpolazione geometrica fra i due valori centrali.

## Risultati sui segmenti propri

| K | τ | Retention HDDC | Retention KMeans | Enrichment HDDC | Enrichment KMeans |
| ---: | ---: | ---: | ---: | ---: | ---: |''']
    for r in combined:
        lines.append(f"| {r['K']} | {r['threshold']:g} | {r['retention_hddc']:.3f} | {r['retention_kmeans']:.3f} | {r['enrichment_hddc']:.3f} | {r['enrichment_kmeans']:.3f} |")
    lines += [r'''
![Score sui segmenti propri](01_score_originali.png)

HDDC ha mediane maggiori di retention ed enrichment in tutte le configurazioni.
Questo confronto combina centro, direzione e scala specifici di ciascun
modello. KMeans mantiene la stessa geometria PC1 fra soglie a K fissato:
le righe ripetute non sono repliche indipendenti.

## Controllo delle direzioni a centro e lunghezza fissati

Separatamente per ogni K e τ, i centri proiettati sono associati uno a uno
minimizzando la somma delle distanze euclidee con l'algoritmo Hungarian.
L'abbinamento non usa densità o direzioni. Per una coppia HDDC/KMeans:

$$L_j^*=\min(L_{H,j},L_{K,j}).$$

Entrambe le direzioni sono valutate a entrambi i centri sulla stessa
semilunghezza L*. Questa scala comune può troncare diversamente i due
intervalli originali, quindi non è più l'80% marginale per entrambi.

$$\delta_{H,j}=\log A(u_{H,j};z_{H,j},L_j^*)-\log A(u_{K,j};z_{H,j},L_j^*),$$

$$\delta_{K,j}=\log A(u_{H,j};z_{K,j},L_j^*)-\log A(u_{K,j};z_{K,j},L_j^*),$$

$$\Delta_j=\frac{\delta_{H,j}+\delta_{K,j}}{2},\qquad
Q=\exp\!\left[\operatorname{mediana}_j(\Delta_j)\right].$$

Q>1 favorisce HDDC; Q<1 favorisce KMeans. Per ogni coppia, exp(Δ_j) è
la media geometrica dei due rapporti fra densità medie ai due centri.
A centro e L fissati, densità centrale e normalizzazione della lunghezza
si cancellano esattamente nel rapporto. Un'ipotetica baseline casuale comune
si cancellerebbe anche dalla differenza dei log-enrichment: non servono nuove
random e non si sottraggono gli enrichment originali, con scale diverse.

Si riportano tutte le coppie e il sottoinsieme prespecificato degli abbinamenti
Hungarian che sono anche reciprocamente più vicini. È una sensibilità alla
località del matching, non una selezione in base agli score. Le vittorie sotto
usano ±0.001, con i valori intermedi classificati come numericamente prossimi.

| K | τ | Q tutte le coppie | Q reciproche | N reciproche | Vittorie H / K / prossime |
| ---: | ---: | ---: | ---: | ---: | --- |''']
    for r in combined:
        lines.append(f"| {r['K']} | {r['threshold']:g} | {r['paired_ratio']:.3f} | {r['mutual_paired_ratio']:.3f} | {r['mutual_pairs']} | {r['h_wins']} / {r['k_wins']} / {r['numerical_close']} |")
    lines += [r'''
![Confronto appaiato](02_confronto_appaiato.png)

I punti sono mediane; le barre sono intervalli interquartili fra le coppie,
non intervalli di confidenza. Le distribuzioni si sovrappongono ampiamente
allo zero. A K100 tutte le mediane complete sono negative e i sottoinsiemi
reciproci sono vicini alla parità. A K50 il vantaggio aggregato si inverte
nel sottoinsieme reciproco.

![Interazione con il centro](03_interazione_centro.png)

Il grafico usa τ=0.1, come i riepiloghi del pilot precedente. Nel quadrante
in basso a destra δ_H>0 e δ_K<0: ciascuna direzione prevale al proprio centro.
A K50 questo accade in 34/50 coppie. HDDC prevale ai due centri in 6/50,
KMeans in 8/50; nelle altre 2/50 ciascuna prevale al centro dell'altro.
A K100 i quattro conteggi sono rispettivamente 24,22,24,30.
La retta tratteggiata δ_K=-δ_H indica Δ=0: gli effetti ai due centri si
compensano. Valori vicinissimi a zero vanno letti con la tolleranza numerica.

## Diagnostiche a τ=0.1

| K | Modello | Mediana log densità centro | Mediana log densità media assoluta | Mediana L | Mediana α |
| ---: | --- | ---: | ---: | ---: | ---: |''']
    for r in diagnostics:
        lines.append(f"| {r['K']} | {NAMES[r['kind']]} | {r['median_center_log_density']:.3f} | {r['median_absolute_log_average']:.3f} | {r['median_projected_half_length']:.3f} | {r['median_alpha']:.3f} |")
    lines += [r'''
![Diagnostiche](04_diagnostiche.png)

La densità media assoluta è ricostruita per componente come
log p̄_i=log A_i+log p̂(z_i), poi riassunta con la mediana. Non si sommano
mediane. Essa favorisce KMeans, mentre la retention relativa favorisce HDDC:
le due quantità rispondono a domande diverse. Nessuna delle due misura da sola
un angolo con il manifold. Le densità assolute dipendono dalle coordinate e
non sono probabilità dei componenti.

La mediana di alpha è maggiore per HDDC. Questa diagnostica descrive la
conservazione delle direzioni nella PCA, senza rendere visibili le dimensioni
eliminate. Le scatole mostrano mediana e quartili, i baffi la convenzione
1.5×IQR e i punti i singoli componenti; non sono intervalli inferenziali.

## Accuratezza numerica e interpretazione

Le medie usano quadratura adattiva QUADPACK con picchi localizzati da un pilot
di 201 posizioni, errore relativo stimato ≤1e-4. I grid uniformi sono diagnostici.
I 19 test mirati sono passati. Per PCA75 quattro controlli con pilot a 401
posizioni hanno variazione massima nei log-score 1.56e-6; per direction-swap
16 controlli hanno variazione massima 4.69e-6. Questo export legge solo gli
artefatti completati e controlla nuovamente le mediane contro i CSV per
componente/coppia; non ricalcola densità o integrali.

La grandezza numerica della retention va letta considerando l'esponente d.
Definendo un raggio efficace lungo il segmento:

$$r_{\rm eff}=\left[\frac{1}{2L}\int_{-L}^{L}r_{20}(z+s u)^{-d}\,ds\right]^{-1/d},
\qquad A=\left(\frac{r_{20}(z)}{r_{\rm eff}}\right)^d.$$

In 141 dimensioni, A=60.33 e A=7.12 corrispondono a riduzioni del raggio
efficace rispetto al centro di circa 2.87% e 1.38%. Piccole differenze di
distanza possono quindi produrre grandi rapporti di densità. L'integrale può
essere sostenuto da picchi interni: A alto non garantisce supporto uniforme
sull'intervallo né permanenza sul manifold.

I due reviewer, inizialmente favorevole e scettico, convergono su questa
lettura: il pilot sostiene un vantaggio HDDC nello score sui propri segmenti,
ma il controllo non sostiene una superiorità direzionale generale a parità
di centro e scala. A K50 emerge una forte interazione fra centro e direzione.
Il controllo non permette di attribuire causalmente il gap iniziale al solo
centro, alla sola scala o alla sola direzione, né dimostra tangenti migliori
per KMeans.

Restano i limiti del pilot: un dataset/layer/split/seed, soglie dipendenti,
reference validation usata nella selezione dei checkpoint, matching talvolta
non locale, intervalli troncati e direzioni trasportate, densità proiettata con
variazioni dei prompt/template e coordinate eliminate invisibili.

## File

- [Tabella completa CSV](tabella_finale.csv)
- [Diagnostiche CSV](diagnostiche.csv)
- [Provenienza dell'export](manifest.json)
- Figure esportate in PNG, PDF e SVG con gli stessi nomi base.
''']
    lines.append(f"\nExport eseguito tramite srun sul nodo `{socket.gethostname()}`, "
                 f"job `{os.environ['SLURM_JOB_ID']}`, step `{os.environ['SLURM_STEP_ID']}`.\n")
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in [None, 'batch', 'extern']:
        raise RuntimeError('Run this export on a compute node via srun')
    print(f"srun execution: node={socket.gethostname()}, job={os.environ['SLURM_JOB_ID']}, "
          f"step={os.environ['SLURM_STEP_ID']}", flush=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    assert json.loads((PCA/'COMPLETED.json').read_text()) == dict(completed=True, smoke=False, configurations=16, components=1200)
    assert json.loads((SWAP/'COMPLETED.json').read_text()) == dict(completed=True, smoke=False, configurations=8, pairs=600)
    pca_manifest = json.loads((PCA/'manifest.json').read_text())
    assert pca_manifest['dimension'] == 141 and pca_manifest['k'] == 20
    for root in [PCA, SWAP]:
        assert json.loads((root/'integration_validation.json').read_text())['passed']
    summary, components = read_csv(PCA/'summary.csv'), read_csv(PCA/'components.csv')
    swap_summary, pairs = read_csv(SWAP/'summary.csv'), read_csv(SWAP/'pairs.csv')
    assert len(summary) == 16 and len(components) == 1200 and len(swap_summary) == 8 and len(pairs) == 600
    combined, diagnostics = build_tables(summary, components, swap_summary, pairs)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    output.mkdir(parents=True)
    write_csv(output/'tabella_finale.csv', combined)
    write_csv(output/'diagnostiche.csv', diagnostics)
    plot_original(summary, output)
    plot_paired(pairs, output)
    plot_interaction(pairs, output)
    plot_diagnostics(components, output)
    make_report(output, combined, diagnostics)
    inputs = [Path(__file__)] + [root/name for root in [PCA, SWAP] for name in
                                ['manifest.json', 'COMPLETED.json', 'summary.csv', 'integration_validation.json']]
    inputs += [PCA/'components.csv', SWAP/'pairs.csv']
    manifest = dict(format='dalg_years_pc1_final_export_v1', dimension=141, k=20,
                    retained_variance_fraction=pca_manifest['global_pca']['retained_variance_fraction'],
                    component_rows=1200, matched_pairs=600, comparisons=8,
                    scope='years_neutral, layer 4, PCA75 and paired direction swap',
                    statistics='equal-component/pair summaries; log medians exponentiated; bars are IQR, no inference',
                    original_distribution_plot_threshold=.1, diagnostic_jitter_seed=2026,
                    matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                    execution=dict(hostname=socket.gethostname(), slurm_job_id=os.environ['SLURM_JOB_ID'],
                                   slurm_step_id=os.environ['SLURM_STEP_ID'],
                                   cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK')),
                    input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Final tables, report and four figures (PNG/PDF/SVG) exported to {output}', flush=True)


if __name__ == '__main__':
    main()
