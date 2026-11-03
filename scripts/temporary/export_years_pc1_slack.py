"""Export a concise PCA-density Slack draft, mean profiles and score table."""
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
import torch

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'outputs/experiments/years_layer04_pc1_density_pca75/job_1857111'
COLORS = {'hddc': '#007F86', 'kmeans': '#D67D24'}
NAMES = {'hddc': 'HDDC', 'kmeans': 'KMeans'}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, output, name):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'{name}.{extension}', dpi=190, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in [None, 'batch', 'extern']:
        raise RuntimeError('Run on a compute node through srun')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    assert manifest['dimension'] == 141 and manifest['k'] == 20 and not manifest['smoke']
    assert json.loads((SOURCE/'COMPLETED.json').read_text()) == dict(completed=True, smoke=False, configurations=16, components=1200)
    with (SOURCE/'summary.csv').open() as stream:
        selected = [r for r in csv.DictReader(stream) if float(r['threshold']) == .1]
    assert len(selected) == 4
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5), sharey=True, layout='constrained')
    profiles, table, sources = [], [], [Path(__file__), SOURCE/'manifest.json', SOURCE/'summary.csv']
    for ax, K in zip(axes, [50, 100]):
        for kind in ['hddc', 'kmeans']:
            row = next(r for r in selected if int(r['K']) == K and r['kind'] == kind)
            path = SOURCE/(row['run_id']+'.pt')
            sources.append(path)
            bundle = torch.load(path, map_location='cpu', weights_only=True)
            results = [bundle['components'][i] for i in range(K)]
            positions = results[0]['relative_positions']
            assert positions.shape == (201,)
            assert all(torch.equal(r['relative_positions'], positions) for r in results)
            logs = torch.stack([r['log_relative_density_profiles'] for r in results])
            assert logs.shape == (K, 33, 201) and torch.isfinite(logs).all()
            assert logs[:, :, 100].abs().max() < 1e-10
            # A principal component is an unoriented axis. Average both signs.
            logs = .5 * (logs + logs.flip(-1))
            pc1 = logs[:, 0].mean(0)
            random = logs[:, 1:].mean((0, 1))
            for curve, label, style in [(pc1, 'PC1', '-'), (random, 'Random', '--')]:
                ax.plot(positions.numpy(), curve.numpy(), ls=style, lw=2.4 if label == 'PC1' else 1.7,
                        color=COLORS[kind], alpha=1 if label == 'PC1' else .75)
                profiles.extend(dict(K=K, kind=kind, direction=label, position=float(x),
                                     mean_log_relative_density=float(y)) for x, y in zip(positions, curve))
            table.append(dict(K=K, method=NAMES[kind], threshold=.1,
                              retention=float(row['median_retention']), enrichment=float(row['median_enrichment'])))
        ax.axhline(0, color='#888888', lw=.8)
        ax.axvline(0, color='#BBBBBB', lw=.7)
        ax.set(title=f'K = {K}', xlabel='Posizione normalizzata sul segmento, s/L', xlim=(-1, 1))
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.grid(axis='y', alpha=.13)
    axes[0].set_ylabel(r'Media di $\log[\hat p(z+s\,u)/\hat p(z)]$')
    fig.suptitle('Densità lungo PC1 e direzioni casuali\nYears · layer 4 · PCA 141D (75,10% varianza) · kNN k=20 · τ=0,1'
                 '\nProfili medi simmetrizzati · 32 direzioni casuali per componente', fontsize=13)
    handles = [Line2D([], [], color=COLORS[k], lw=2.4, ls=style, label=f'{NAMES[k]} · {label}')
               for label, style in [('PC1', '-'), ('random', '--')] for k in ['hddc', 'kmeans']]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=10)
    output.mkdir(parents=True)
    save(fig, output, 'curva_media_log_densita')
    write_csv(output/'profili_medi.csv', profiles)
    write_csv(output/'tabella.csv', table)

    fig, ax = plt.subplots(figsize=(8, 2.8), layout='constrained')
    ax.axis('off')
    cells = [[r['K'], r['method'], f"{r['retention']:.2f}", f"{r['enrichment']:.2f}"] for r in table]
    artist = ax.table(cellText=cells, colLabels=['K', 'Modello', 'Retention ↑', 'Enrichment ↑'],
                      cellLoc='center', colLoc='center', bbox=[0, .13, 1, .72], colWidths=[.12, .28, .3, .3])
    artist.auto_set_font_size(False)
    artist.set_fontsize(12)
    for (row, col), cell in artist.get_celld().items():
        cell.set_edgecolor('#DCE3E8')
        if row == 0:
            cell.set_facecolor('#EDF1F5')
            cell.set_text_props(weight='bold')
        elif row in [1, 3]:
            cell.set_facecolor('#E8F4F3')
            if col >= 2:
                cell.set_text_props(weight='bold', color=COLORS['hddc'])
    ax.set_title('Years · layer 4 · PCA 141D · soglia di rango τ=0,1', fontsize=14)
    ax.text(.5, .02, 'Mediane sui componenti, calcolate in log-space ed esponenziate.',
            transform=ax.transAxes, ha='center', fontsize=10, color='#555555')
    save(fig, output, 'tabella_risultati')

    draft = '''Ho confrontato *HDDC-EM e KMeans+PCA* su *years_neutral, layer 4*, riusando modelli e attivazioni salvati.

Ho stimato una PCA globale sul training, centrata e senza whitening: *141 dimensioni, 75,10% della varianza*. Nello spazio PCA ho usato una *densità kNN, k=20*, con le 2.002 attivazioni di validation come riferimento comune.

Per ogni componente ho proiettato la *PC1 originale* e il suo *intervallo centrale dell’80%*, calcolando:
• *Retention*: densità media sul segmento / densità al centro.
• *Enrichment*: retention PC1 / media delle retention di 32 direzioni casuali, allo stesso centro e con la stessa lunghezza.

Alla soglia di rango 0,1, *HDDC ottiene score maggiori sia a K=50 sia a K=100*; la tabella riporta le mediane sui componenti.

Il plot mostra i profili medi della log-densità relativa al centro; gli score sono calcolati integrando la densità.
'''
    (output/'messaggio_slack.md').write_text(draft)
    metadata = dict(scope='years_neutral layer 4; PCA75 original density-support pilot only',
        dimension=141, k=20, threshold=.1, centers_and_lengths='own model PC1 intervals',
        curve='equal-component arithmetic mean of log density relative to own center',
        random_curve='mean over 32 log profiles within each component, then equal-component mean',
        symmetrization='average profile(s) and profile(-s); PC1 is an unoriented axis',
        scores='existing adaptive density integrals; exp(median(log score)); no new density evaluation',
        execution=dict(hostname=socket.gethostname(), job=os.environ['SLURM_JOB_ID'], step=os.environ['SLURM_STEP_ID']),
        input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Exported Slack draft, mean profiles and result table: {output}', flush=True)
    print(metadata['execution'], flush=True)


if __name__ == '__main__':
    main()
