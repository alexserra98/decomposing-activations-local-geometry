"""Check neighbor-only provenance, source-to-cell values, and k20 table reproduction."""
import argparse,csv,json,math,re
from pathlib import Path


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def key(row):
    return row['dataset'],int(row['layer']),row['kind'],int(row['K']),float(row['segment_mass']),float(row['quantile'])


def main(root):
    original={key(r):r for r in read(Path('outputs/experiments/pc1_extended_mass_percentile/job_1934466/tabelle_soglia_01.csv')) if int(r['layer'])==4}
    fingerprints={};result={};previous=None
    for k in [5,10,20,40]:
        target=root/f'k{k}'
        assert json.loads((target/'COMPLETED.json').read_text())['completed']
        rows=read(target/'tabelle_soglia_01.csv');lookup={key(r):r for r in rows}
        assert set(lookup)==set(original) and len(rows)==128
        manifest=json.loads((target/'delta_matrices/layer4_manifest.json').read_text())
        assert len(manifest['cells'])==64
        for cell in manifest['cells']:
            for kind,value in cell['delta_pp'].items():
                row=lookup[(cell['dataset'],4,kind,cell['K'],cell['mass'],cell['quantile'])]
                assert value==100*float(row['mean_delta_coverage'])
            assert cell['winner']==max(cell['delta_pp'],key=cell['delta_pp'].get)
        for old_key,row in lookup.items():
            old=original[old_key]
            assert row['eligible']==old['eligible'] and row['total']==old['total']
            if k==20:
                for field in ['mean_coverage_pc1','mean_coverage_random','mean_delta_coverage','median_coverage_pc1','median_delta_coverage','q25_delta_coverage','q75_delta_coverage']:
                    assert float(row[field])==float(old[field]),(old_key,field)
        case_fingerprints={}
        validations=[]
        for case in ['years_neutral_l04','temperature_02d_l04']:
            protocol=json.loads((target/case/'manifest.json').read_text())
            assert protocol['neighbors']==k
            fixed={name:protocol[name] for name in ['masses','quantiles','calibration_source','calibration_sha256','reference_sha256','query_sha256','old_manifest_sha256','random_directions','random_seed','grid_sizes','grid_tolerance','geometry_bundle_sha256']}
            case_fingerprints[case]=fixed
            validations.append(json.loads((target/case/'validation.json').read_text()))
        if previous is not None:assert case_fingerprints==previous
        previous=case_fingerprints
        pdf=(target/'delta_matrices/PC1_delta_layer4_K50_K100.pdf').read_bytes()
        assert len(re.findall(rb'/Type /Page\b',pdf))==1
        means={kind:sum(float(r['mean_delta_coverage']) for r in rows if r['kind']==kind)/64*100 for kind in ['hddc','kmeans']}
        result[k]=dict(passed=True,selected_values=128,matrix_cells=64,mean_delta_pp=means,
            ratio_of_means=means['hddc']/means['kmeans'],max_grid_error=max(v['max_grid_error'] for v in validations),
            max_refinement_change=max(v['max_refinement_change'] for v in validations),
            max_k20_reproduction_difference=max(v['max_k20_reproduction_difference'] for v in validations) if k==20 else None)
    report=dict(passed=True,all_fixed_fingerprints_identical=True,k20_selected_values_exactly_reproduced=True,neighbors=result)
    (root/'independent_table_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);main(p.parse_args().root)
