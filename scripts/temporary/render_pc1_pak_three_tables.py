"""Render the three validated PAk cases using the original overview layout."""
import json
from pathlib import Path

from export_pc1_density_overview import NAMES, panel_tables, read_csv, write_csv

root = Path('outputs/experiments/pc1_pak_controlled/job_1927867')
output = Path('outputs/experiments/pc1_pak_controlled/three_validated_tables')
output.mkdir(parents=True, exist_ok=True)
entries, rows = [], []
for name, label, layer in [('years_neutral_l28', 'Years', 28),
                           ('temperature_02d_l04', 'Temperature', 4),
                           ('temperature_02d_l28', 'Temperature', 28)]:
    source = root/name
    assert json.loads((source/'COMPLETED.json').read_text())['completed']
    assert json.loads((source/'integration_validation.json').read_text())['passed']
    manifest = json.loads((source/'manifest.json').read_text())['baseline_manifest']
    selected = []
    for summary in read_csv(source/'summary.csv'):
        if float(summary['threshold']) != .1:
            continue
        row = dict(K=int(summary['K']), kind=summary['kind'], method=NAMES[summary['kind']],
                   retention=float(summary['median_retention']), enrichment=float(summary['median_enrichment']),
                   eligible=int(summary['eligible']), total=int(summary['total']))
        selected.append(row)
        rows.append(dict(experiment=name, **row))
    assert len(selected) == 4
    entries.append(dict(label=label, layer=layer, dimension=manifest['dimension'],
                        variance=manifest['global_pca']['retained_variance_fraction'], selected=selected))
fig = panel_tables(entries)
fig._suptitle.set_text('PAk controllato (k* ≥ 4) · Retention ed enrichment')
for extension in ['png', 'pdf', 'svg']:
    fig.savefig(output/f'02_pannello_tabelle.{extension}', dpi=200, facecolor='white')
write_csv(output/'tabella_soglia_01.csv', rows)
print(output.resolve())
