"""Export two summary panels from the four completed PCA-density experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import torch

from run_years_pc1_density import REPO, sha256, write_csv

ROOT = REPO/'outputs/experiments'
SOURCES = [
    ('years_neutral', 'Years', 4, ROOT/'years_layer04_pc1_density_pca75/job_1857111',
     ROOT/'years_layer04_pc1_density_slack/20261028_finale_srun/profili_medi.csv'),
    ('years_neutral', 'Years', 28, ROOT/'years_layer28_pc1_density_pca75/job_1886802', None),
    ('temperature_02d', 'Temperature', 4, ROOT/'temperature_layer04_pc1_density_pca75/job_1886682', None),
    ('temperature_02d', 'Temperature', 28, ROOT/'temperature_layer28_pc1_density_pca75/job_1884175', None),
]
COLORS = {'hddc': '#007F86', 'kmeans': '#D67D24'}
NAMES = {'hddc': 'HDDC-EM', 'kmeans': 'KMeans+PCA'}
THRESHOLDS = [.0001, .001, .01, .1]


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def number(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def score_text(value):
    return number(value, 4 if abs(value) < 10 else 2)


def title(entry):
    return (f'{entry["label"]} · layer {entry["layer"]}\n'
            f'PCA {entry["dimension"]}D · {number(100*entry["variance"])}% della varianza')


def load_results():
    entries, all_rows, profile_rows, inputs = [], [], [], [Path(__file__)]
    populations = {}
    for dataset, label, layer, source, cached_profiles in SOURCES:
        manifest = json.loads((source/'manifest.json').read_text())
        completed = json.loads((source/'COMPLETED.json').read_text())
        validation = json.loads((source/'integration_validation.json').read_text())
        assert completed['completed'] and not completed['smoke']
        assert (completed['configurations'], completed['components']) == (16, 1200)
        assert (manifest['dataset'], manifest['layer']) == (dataset, layer)
        assert manifest['evaluation_space'] == 'global_pca75' and not manifest['smoke']
        assert (manifest['k'], manifest['n_random'], manifest['probability']) == (20, 32, .8)
        assert validation['passed'] and len(validation['checks']) == 4
        reference_ids = manifest['reference_global_rows']
        if dataset in populations:
            assert populations[dataset] == reference_ids
        populations[dataset] = reference_ids
        assert len(reference_ids) == (2002 if dataset == 'years_neutral' else 1500)
        info = manifest['global_pca']
        assert info['dimension'] == manifest['dimension']
        assert sha256(info['path']) == info['sha256']
        entry = dict(dataset=dataset, label=label, layer=layer, source=source,
                     dimension=manifest['dimension'], variance=info['retained_variance_fraction'],
                     reference_count=len(reference_ids), selected=[], curves={})
        summary, components = read_csv(source/'summary.csv'), read_csv(source/'components.csv')
        assert len(summary) == 16 and len(components) == 1200
        expected = {(kind, K, tau) for kind in ['hddc', 'kmeans'] for K in [50, 100] for tau in THRESHOLDS}
        assert {(s['kind'], int(s['K']), float(s['threshold'])) for s in summary} == expected
        for s in summary:
            run_rows = [r for r in components if r['run_id'] == s['run_id']]
            valid = [r for r in run_rows if not r['exclusion_reason']]
            K = int(s['K'])
            assert len(run_rows) == K and {int(r['component']) for r in run_rows} == set(range(K))
            assert len(valid) == int(s['eligible']) and int(s['total']) == K
            assert valid
            for field in ['retention', 'enrichment']:
                median = float(np.median([float(r['log_'+field]) for r in valid]))
                np.testing.assert_allclose(median, float(s['median_log_'+field]), atol=1e-12, rtol=0)
                np.testing.assert_allclose(math.exp(median), float(s['median_'+field]), atol=1e-12, rtol=1e-12)
            row = dict(dataset=dataset, layer=layer, dimension=entry['dimension'],
                       retained_variance_fraction=entry['variance'], K=K, kind=s['kind'],
                       method=NAMES[s['kind']], threshold=float(s['threshold']), eligible=len(valid), total=K,
                       retention=float(s['median_retention']), enrichment=float(s['median_enrichment']),
                       log_retention=float(s['median_log_retention']), log_enrichment=float(s['median_log_enrichment']),
                       median_alpha=float(s['median_alpha']),
                       median_center_log_density=float(s['median_center_log_density']), run_id=s['run_id'])
            all_rows.append(row)
            if row['threshold'] != .1:
                continue
            entry['selected'].append(row)
            path = source/(s['run_id']+'.pt')
            bundle = torch.load(path, map_location='cpu', weights_only=True)
            ids = sorted(int(r['component']) for r in valid)
            assert set(bundle['components']) == set(ids)
            results = [bundle['components'][i] for i in ids]
            positions = results[0]['relative_positions']
            torch.testing.assert_close(positions, torch.linspace(-1, 1, 201, dtype=torch.float64), atol=1e-12, rtol=0)
            for component, result in zip(ids, results):
                assert torch.equal(result['relative_positions'], positions)
                direction_scores = result['log_direction_retentions']
                assert direction_scores.shape == (33,) and torch.isfinite(direction_scores).all()
                expected_enrichment = float(direction_scores[0]-torch.logsumexp(direction_scores[1:], 0)+math.log(32))
                component_row = next(r for r in valid if int(r['component']) == component)
                np.testing.assert_allclose([result['log_retention'], result['log_enrichment']],
                    [float(component_row['log_retention']), float(component_row['log_enrichment'])], atol=1e-12, rtol=0)
                assert abs(expected_enrichment-result['log_enrichment']) < 1e-12
            logs = torch.stack([r['log_relative_density_profiles'] for r in results])
            assert logs.shape == (len(valid), 33, 201) and torch.isfinite(logs).all()
            assert float(logs[:, :, 100].abs().max()) < 1e-10
            logs = .5*(logs+logs.flip(-1))
            for direction, curve in [('PC1', logs[:, 0].mean(0)), ('random', logs[:, 1:].mean((0, 1)))]:
                entry['curves'][K, s['kind'], direction] = positions.numpy(), curve.numpy()
                profile_rows.extend(dict(dataset=dataset, layer=layer, dimension=entry['dimension'], K=K,
                    kind=s['kind'], direction=direction, position=float(x), mean_log_relative_density=float(y))
                    for x, y in zip(positions, curve))
            inputs.append(path)
        cached_profiles = cached_profiles or source/'mean_profiles.csv'
        cached = read_csv(cached_profiles)
        for K in [50, 100]:
            for kind in ['hddc', 'kmeans']:
                for direction in ['PC1', 'random']:
                    rows = [r for r in cached if int(r['K']) == K and r['kind'] == kind
                            and r['direction'].lower() == direction.lower()]
                    assert len(rows) == 201
                    x, y = entry['curves'][K, kind, direction]
                    np.testing.assert_allclose([float(r['position']) for r in rows], x, atol=1e-12, rtol=0)
                    np.testing.assert_allclose([float(r['mean_log_relative_density']) for r in rows], y, atol=1e-10, rtol=0)
        inputs.extend([source/name for name in ['manifest.json', 'COMPLETED.json', 'summary.csv',
                                               'components.csv', 'integration_validation.json']])
        inputs.extend([cached_profiles, Path(info['path'])])
        entries.append(entry)
    assert len(all_rows) == 64 and len(profile_rows) == 6432
    return entries, all_rows, profile_rows, inputs


def panel_plots(entries):
    fig = plt.figure(figsize=(17, 10), facecolor='white')
    outer = fig.add_gridspec(2, 2, left=.065, right=.985, top=.84, bottom=.17, hspace=.46, wspace=.18)
    for cell, entry in zip(outer, entries):
        inner = cell.subgridspec(1, 2, wspace=.10)
        left = fig.add_subplot(inner[0])
        right = fig.add_subplot(inner[1], sharey=left)
        extrema = np.concatenate([curve[1] for curve in entry['curves'].values()])
        bottom, top = min(0., float(extrema.min())), max(0., float(extrema.max()))
        pad = .06*max(top-bottom, .01)
        for ax, K in zip([left, right], [50, 100]):
            for kind in ['hddc', 'kmeans']:
                for direction, style in [('PC1', '-'), ('random', '--')]:
                    x, y = entry['curves'][K, kind, direction]
                    ax.plot(x, y, color=COLORS[kind], ls=style, lw=2.2 if direction == 'PC1' else 1.8)
            ax.axhline(0, color='#8B949E', lw=.7, zorder=0)
            ax.set(title=f'K = {K}', xlim=(-1, 1), ylim=(bottom-pad, top+pad))
            ax.set_xticks([-1, -.5, 0, .5, 1])
            ax.grid(axis='y', alpha=.15)
            ax.tick_params(labelsize=10)
        right.tick_params(labelleft=False)
        left.set_ylabel(r'Media $\log[\hat p(z+su)/\hat p(z)]$', fontsize=11)
        box = cell.get_position(fig)
        fig.text((box.x0+box.x1)/2, box.y1+.045, title(entry), ha='center', va='bottom',
                 fontsize=13, weight='bold', color='#23364A', linespacing=1.4)
    fig.suptitle('A · Profili medi di densità lungo PC1 e direzioni casuali', y=.985, fontsize=18, weight='bold')
    fig.text(.5, .942, 'Intervallo originale centrale dell’80%, proiettato · kNN k=20 · 32 direzioni casuali · soglia di rango locale 0,1',
             ha='center', fontsize=11, color='#4B5563')
    fig.text(.5, .115, 'Posizione normalizzata sul segmento, s/L  (centro = 0)', ha='center', fontsize=12)
    handles = [Line2D([], [], color=COLORS[kind], ls=style, lw=2.2,
                      label=f'{"HDDC" if kind == "hddc" else "KMeans"} · {label}')
               for label, style in [('PC1', '-'), ('random', '--')] for kind in ['hddc', 'kmeans']]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=4, frameon=False, fontsize=11)
    fig.text(.5, .022, 'Medie sui cluster valutabili, simmetrizzate rispetto al segno. Assi y comuni entro ogni esperimento; scale diverse fra esperimenti.',
             ha='center', fontsize=10, color='#4B5563')
    return fig


def panel_tables(entries):
    fig = plt.figure(figsize=(17, 8.8), facecolor='white')
    outer = fig.add_gridspec(2, 2, left=.035, right=.965, top=.83, bottom=.18, hspace=.52, wspace=.12)
    for cell, entry in zip(outer, entries):
        ax = fig.add_subplot(cell)
        ax.axis('off')
        selected = [next(r for r in entry['selected'] if r['K'] == K and r['kind'] == kind)
                    for K in [50, 100] for kind in ['hddc', 'kmeans']]
        cells = [[r['K'], r['method'], score_text(r['retention']), score_text(r['enrichment']),
                  f'{r["eligible"]}/{r["total"]}'] for r in selected]
        table = ax.table(cellText=cells,
            colLabels=['K', 'Modello', 'Retention ↑', 'Enrichment ↑', 'Valutabili'],
            cellLoc='center', colLoc='center', bbox=[0, .02, 1, .90], colWidths=[.08, .25, .23, .23, .21])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        for (row, col), item in table.get_celld().items():
            item.set_edgecolor('#DCE3E8')
            item.set_linewidth(.65)
            if row == 0:
                item.set_facecolor('#E7EDF2' if col != 2 else '#DCE7EF')
                item.set_text_props(weight='bold', color='#23364A')
            else:
                item.set_facecolor('#F5F7F9' if row <= 2 else 'white')
                if col == 1:
                    item.set_text_props(color=COLORS[selected[row-1]['kind']])
        for K, start in [(50, 1), (100, 3)]:
            rows = selected[start-1:start+1]
            for column, field in [(2, 'retention'), (3, 'enrichment')]:
                winner = max(range(2), key=lambda i: rows[i][field])
                table[start+winner, column].set_text_props(weight='bold', color=COLORS[rows[winner]['kind']])
        box = cell.get_position(fig)
        fig.text((box.x0+box.x1)/2, box.y1+.025, title(entry), ha='center', va='bottom',
                 fontsize=14, weight='bold', color='#23364A', linespacing=1.35)
    fig.suptitle('B · Retention ed enrichment: confronto tra modelli', y=.985, fontsize=18, weight='bold')
    fig.text(.5, .936, 'Soglia 0,1 per selezionare il rango locale · mediana dei log-score, esponenziata · retention = score primario',
             ha='center', fontsize=11, color='#4B5563')
    fig.text(.5, .112, 'Retention = densità media sul segmento / densità al centro.  Enrichment = retention PC1 / media delle retention casuali.',
             ha='center', fontsize=11)
    fig.text(.5, .065, 'Grassetto: valore maggiore entro lo stesso esperimento e K; non indica significatività statistica. Valutabili: cluster inclusi / totali.',
             ha='center', fontsize=10, color='#4B5563')
    fig.text(.5, .025, 'I punteggi descrivono supporto nella proiezione PCA: dimensioni, centri e lunghezze differiscono; le magnitudini non ordinano la qualità tra esperimenti.',
             ha='center', fontsize=10, color='#4B5563')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in [None, 'batch', 'extern']:
        raise RuntimeError('Run numerical work and plotting through srun on a compute node')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    entries, all_rows, profiles, inputs = load_results()
    selected = [r for entry in entries for r in sorted(entry['selected'], key=lambda r: (r['K'], r['kind']))]
    assert len(selected) == 16
    output.mkdir(parents=True)
    write_csv(output/'tabella_soglia_01.csv', selected)
    write_csv(output/'tutte_le_64_configurazioni.csv', all_rows)
    write_csv(output/'profili_medi.csv', profiles)
    figures = [panel_plots(entries), panel_tables(entries)]
    for fig, name in zip(figures, ['01_pannello_curve', '02_pannello_tabelle']):
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(output/f'{name}.{extension}', dpi=200, facecolor='white')
    with PdfPages(output/'riepilogo_due_pannelli.pdf') as pdf:
        for fig in figures:
            pdf.savefig(fig, facecolor='white')
    for fig in figures:
        plt.close(fig)
    comparisons = []
    for entry in entries:
        for K in [50, 100]:
            h, k = [next(r for r in entry['selected'] if r['K'] == K and r['kind'] == kind) for kind in ['hddc', 'kmeans']]
            comparisons.append(dict(dataset=entry['dataset'], layer=entry['layer'], K=K,
                higher_retention='hddc' if h['retention'] > k['retention'] else 'kmeans',
                higher_enrichment='hddc' if h['enrichment'] > k['enrichment'] else 'kmeans'))
    metadata = dict(scope='four completed global-PCA75 experiments; plots/tables at threshold 0.1',
        sources=[dict(dataset=e['dataset'], layer=e['layer'], path=str(e['source']), dimension=e['dimension'],
                      retained_variance_fraction=e['variance'], reference_count=e['reference_count']) for e in entries],
        verification=dict(passed=True, configuration_medians_checked=64, input_component_entries_checked=4800,
            selected_rows=16, selected_eligible_cluster_entries=sum(r['eligible'] for r in selected),
            profile_rows=len(profiles), cached_mean_curves_checked=32, same_reference_ids_within_each_dataset=True),
        curves='equal eligible-cluster mean of log density relative to center; sign-symmetrized; random mean over 32 directions',
        scores='saved adaptive density integrals; exp(median(log score)); no new density estimation',
        y_scales='common for K50/K100 within each experiment; independent across experiments',
        bold='strictly higher descriptive score within experiment/K; no significance claim',
        comparisons_at_threshold_01=comparisons,
        integration_provenance='years/layer4 uses the archived original integrator; later repeats refine subdivision at all pilot positions; same score/tolerance',
        input_sha256={str(path): sha256(path) for path in dict.fromkeys(inputs)},
        execution=dict(hostname=socket.gethostname(), job=os.environ['SLURM_JOB_ID'], step=os.environ['SLURM_STEP_ID']))
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = ['# PC1 density support: due pannelli riepilogativi', '',
        'Quattro esperimenti PCA75: years_neutral e temperature_02d, layer 4 e 28.',
        'I due pannelli usano la soglia 0,1 per selezionare il rango locale, come nei confronti principali precedenti.',
        'Le curve sono medie sui cluster valutabili dei log-profili relativi al centro, simmetrizzate rispetto al segno.',
        'Gli score derivano dagli integrali adattivi della densità e sono riassunti esponenziando la mediana dei log-score.',
        'Le scale y sono comuni ai due K di uno stesso esperimento e diverse fra esperimenti.', '',
        '![Pannello curve](01_pannello_curve.png)', '', '![Pannello tabelle](02_pannello_tabelle.png)', '',
        'HDDC ha retention maggiore in 6/8 confronti alla soglia 0,1 ed enrichment maggiore in 8/8.',
        'KMeans ha retention maggiore su temperature/layer 28 a K50 e su years/layer 28 a K100.',
        'Questi sono confronti descrittivi dipendenti; i valori maggiori in tabella non indicano significatività statistica.',
        'I punteggi misurano supporto nella proiezione ai centri e alle scale propri dei modelli. Le dimensioni scartate restano invisibili.',
        'Le magnitudini tra spazi PCA diversi non costituiscono una graduatoria di qualità.', '',
        'Verifiche: 64 mediane di configurazione contro 4.800 righe cluster; 32 curve confrontate con gli export precedenti;',
        'bundle, profili e formule di enrichment verificati per i cluster selezionati. Tutto eseguito tramite srun.', '',
        '[PDF con i due pannelli](riepilogo_due_pannelli.pdf) · [Tabella principale CSV](tabella_soglia_01.csv) · '
        '[Tutte le 64 configurazioni CSV](tutte_le_64_configurazioni.csv) · [Provenienza e verifiche](manifest.json)', '']
    assert sum(c['higher_retention'] == 'hddc' for c in comparisons) == 6
    assert sum(c['higher_enrichment'] == 'hddc' for c in comparisons) == 8
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(metadata['verification'], indent=2))
    print(json.dumps(metadata['execution']))
    print(f'Exported panels: {output}', flush=True)


if __name__ == '__main__':
    main()
