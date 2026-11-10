"""Equal-weight PC1..PC5 support coverage with the fixed PC1 calibration."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc
from dalg.models.kmeans import load_kmeans
from export_pc1_density_overview import SOURCES, NAMES, panel_tables, read_csv
import run_pc1_support_coverage as base
from run_years_pc1_density import sha256, write_csv

BASELINE = Path('outputs/experiments/pc1_support_coverage/job_1928981')


def checkpoint_path(source, manifest):
    path = Path(source['checkpoint'])
    if not path.exists():
        archive = Path('/orfeo/cephfs/scratch/dssc/zenocosini/dalg-cache/sae-manifold/pre_holdout')
        path = archive/Path(manifest['activation_dir']).name/path.relative_to(manifest['activation_dir'])
    if sha256(path) != source['checkpoint_sha256']:
        raise ValueError('Saved checkpoint fingerprint differs')
    return path


@torch.no_grad()
def top5_geometry(model, kind, mean, basis):
    """Five unique retained covariance axes, projected without reorthogonalizing."""
    centers = model.mu.detach().cpu().double()
    ranks = model.component_ranks.detach().cpu()
    count, dimension = centers.shape
    vectors = torch.zeros((count, 5, dimension), dtype=torch.float64)
    eigenvalues = torch.zeros((count, 6), dtype=torch.float64)
    reasons = ['']*count
    if kind == 'hddc':
        loadings = model._W().detach().cpu().double()
        noise = model._psi().detach().cpu().double()
        if not torch.allclose(noise, noise[:, :1].expand_as(noise)):
            raise ValueError('Top5 experiment requires isotropic HDDC noise')
    for component in range(count):
        if kind == 'kmeans' and not bool(model.pca_valid[component]):
            reasons[component] = 'invalid_cluster_pca'
        elif kind == 'hddc' and not bool(torch.isfinite(model.pi_logits[component])):
            reasons[component] = 'zero_mixture_weight'
        elif int(ranks[component]) < 5:
            reasons[component] = 'retained_rank_below_five'
        if reasons[component]:
            continue
        if kind == 'kmeans':
            eigenvalues[component] = model.eigenvalues[component, :6].detach().cpu().double()
            vectors[component] = model.W[component, :, :5].detach().cpu().double().T
        elif kind == 'hddc':
            u, singular, _ = torch.linalg.svd(loadings[component], full_matrices=False)
            eigenvalues[component] = noise[component, 0]
            length = min(6, len(singular))
            eigenvalues[component, :length] += singular[:length].square()
            vectors[component] = u[:, :5].T
        else:
            raise ValueError('Only HDDC and KMeans supported')
        values = eigenvalues[component]
        gaps = values[:5]-values[1:6]
        if not torch.isfinite(values).all() or values[4] <= 0:
            reasons[component] = 'invalid_top5_variance'
        elif bool((gaps <= 1e-10*values[0]).any()):
            reasons[component] = 'nonunique_top5_axis'
        else:
            storage_epsilon = torch.finfo(model.W.dtype).eps if kind == 'kmeans' else torch.finfo(torch.float64).eps
            tolerance = max(1e-10, 10*storage_epsilon)
            if not torch.allclose(vectors[component]@vectors[component].T, torch.eye(5, dtype=torch.float64),
                                  atol=tolerance, rtol=tolerance):
                raise ValueError('Saved covariance PCs are not orthonormal at checkpoint precision')
            # Match the original PC1 extraction: normalize stored float32 axes
            # individually, without rotating or reorthogonalizing their span.
            vectors[component] /= vectors[component].norm(dim=1, keepdim=True)
    projected = vectors@basis.double().T
    norms = projected.norm(dim=2)
    for component in range(count):
        if not reasons[component] and bool((norms[component] <= 1e-12).any()):
            reasons[component] = 'null_top5_projection'
    eligible = torch.tensor([not reason for reason in reasons])
    projected[eligible] /= norms[eligible, :, None]
    original_lengths = NormalDist().inv_cdf(.9)*eigenvalues[:, :5].clamp_min(0).sqrt()
    return dict(centers=(centers-mean.double())@basis.double().T, vectors=projected,
        eigenvalues=eigenvalues, alpha=norms.square(), original_half_lengths=original_lengths,
        half_lengths=original_lengths*norms, ranks=ranks, exclusion_reasons=reasons)


def score_five(center, vectors, lengths, component):
    pcs = [base.score_component(center, vector, float(length), component)
           for vector, length in zip(vectors, lengths)]
    return dict(coverage_top5=float(np.mean([pc['coverage_pc1'] for pc in pcs])),
        coverage_random=float(np.mean([pc['coverage_random'] for pc in pcs])),
        delta_coverage=float(np.mean([pc['delta_coverage'] for pc in pcs])),
        coverage_pc1_same_subset=pcs[0]['coverage_pc1'],
        delta_pc1_same_subset=pcs[0]['delta_coverage'],
        max_grid_error=max(pc['max_grid_error'] for pc in pcs),
        max_refinement_change=max(pc['max_refinement_change'] for pc in pcs), pcs=pcs)


def run_case(index, root, workers, smoke):
    dataset, _, layer, source, _ = SOURCES[index]
    old = BASELINE/f'{dataset}_l{layer:02}'
    previous = json.loads((old/'manifest.json').read_text())
    if not json.loads((old/'COMPLETED.json').read_text())['completed']:
        raise ValueError('PC1 source must be complete')
    manifest = previous['baseline_manifest']
    pca_path = Path(manifest['global_pca']['path'])
    if sha256(pca_path) != previous['pca_sha256']:
        raise ValueError('PCA fingerprint differs')
    pca = torch.load(pca_path, weights_only=True, map_location='cpu')
    calibration = torch.load(old/'calibration.pt', weights_only=True, map_location='cpu')
    if calibration['calibration'] != previous['calibration']:
        raise ValueError('Calibration metadata differs from the completed PC1 protocol')
    reference, tau = calibration['reference'].numpy(), calibration['calibration']['tau']
    if hashlib.sha256(reference.tobytes()).hexdigest() != previous['reference_sha256']:
        raise ValueError('Reference fingerprint differs')
    output = root/f'{dataset}_l{layer:02}'
    output.mkdir(parents=True)
    protocol = dict(metric='arithmetic mean of PC1..PC5 calibrated support coverage',
        calibration_source=str(old), calibration_sha256=sha256(old/'calibration.pt'),
        baseline_manifest=manifest, pc1_protocol_sha256=sha256(old/'manifest.json'),
        source_sha256=sha256(Path(__file__)), scorer_sha256=sha256(Path(base.__file__)),
        calibration=calibration['calibration'], eligibility='all five unique retained axes; rank>=5; all projected norms>1e-12',
        random_seed='42+component; same base controls across PCs; own length for each PC',
        pc_aggregation='equal arithmetic mean within component, including zero scores',
        component_aggregation='ordinary median and mean; no variance weighting',
        grid_sizes=base.GRID_SIZES, grid_tolerance=base.GRID_TOLERANCE,
        job_id=os.environ['SLURM_JOB_ID'], smoke=smoke, checkpoints={})
    (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
    summaries, all_rows, exclusions, directions, cache = [], [], [], [], {}
    configs = read_csv(source/'summary.csv')
    saved_sources = {row['run_id']: row for row in manifest['sources']}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=base.initialize_worker, initargs=(reference, tau)) as executor:
        for config in configs[:1] if smoke else configs:
            run_id = config['run_id']
            path = checkpoint_path(saved_sources[run_id], manifest)
            loader = load_mfa_hddc if config['kind'] == 'hddc' else load_kmeans
            model = loader(path, map_location='cpu')
            geometry = top5_geometry(model, config['kind'], pca['mean'], pca['basis'])
            del model
            old_bundle = torch.load(source/(run_id+'.pt'), weights_only=True, map_location='cpu')['projected_geometry']
            eligible = [i for i, reason in enumerate(geometry['exclusion_reasons']) if not reason]
            for i in eligible:
                if (not torch.allclose(geometry['centers'][i], old_bundle['centers'][i], atol=1e-10, rtol=1e-10)
                    or not math.isclose(float(geometry['half_lengths'][i, 0]), float(old_bundle['half_lengths'][i]), abs_tol=1e-8, rel_tol=1e-8)
                    or not math.isclose(abs(float(geometry['vectors'][i, 0]@old_bundle['vectors'][i])), 1., abs_tol=1e-8)):
                    raise ValueError('Reconstructed PC1 differs from the historical segment')
            for i, reason in enumerate(geometry['exclusion_reasons']):
                if reason:
                    exclusions.append(dict(run_id=run_id, component=i, reason=reason, rank=int(geometry['ranks'][i])))
            if smoke:
                eligible = eligible[:2]
            signature = hashlib.sha256(b''.join(geometry[key].contiguous().numpy().tobytes()
                for key in ['centers', 'vectors', 'half_lengths'])+str(eligible).encode()).hexdigest()
            results = cache.get(signature)
            if results is None:
                futures = {i: executor.submit(score_five, geometry['centers'][i].numpy(), geometry['vectors'][i].numpy(),
                    geometry['half_lengths'][i].numpy(), i) for i in eligible}
                results = {}
                for i in eligible:
                    result = futures[i].result()
                    result['pcs'] = [base.as_tensors(pc) for pc in result['pcs']]
                    results[i] = result
                cache[signature] = results
            for i in eligible:
                result = results[i]
                row = dict(run_id=run_id, component=i, kind=config['kind'], K=int(config['K']),
                    threshold=float(config['threshold']), rank=int(geometry['ranks'][i]),
                    **{key: value for key, value in result.items() if key != 'pcs'})
                all_rows.append(row)
                for pc_index, pc in enumerate(result['pcs']):
                    directions.append(dict(run_id=run_id, component=i, pc=pc_index+1,
                        coverage=pc['coverage_pc1'], random_coverage=pc['coverage_random'], delta=pc['delta_coverage'],
                        half_length=float(geometry['half_lengths'][i, pc_index]), alpha=float(geometry['alpha'][i, pc_index])))
            torch.save(dict(run_id=run_id, geometry=geometry, components=results), output/(run_id+'.pt'))
            protocol['checkpoints'][run_id] = dict(path=str(path), sha256=saved_sources[run_id]['checkpoint_sha256'])
            (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
            valid = [row for row in all_rows if row['run_id'] == run_id]
            summary = dict(run_id=run_id, kind=config['kind'], K=int(config['K']), threshold=float(config['threshold']),
                           eligible=len(valid), total=int(config['K']))
            for field in ['coverage_top5', 'coverage_random', 'delta_coverage', 'coverage_pc1_same_subset', 'delta_pc1_same_subset']:
                values = np.array([row[field] for row in valid])
                for name, value in [('median', np.median(values) if len(values) else None),
                                    ('mean', np.mean(values) if len(values) else None)]:
                    summary[name+'_'+field] = float(value) if value is not None else None
            summaries.append(summary)
            write_csv(output/'summary.csv', summaries)
            write_csv(output/'components.csv', all_rows)
            write_csv(output/'per_pc.csv', directions)
            write_csv(output/'exclusions.csv', exclusions)
            print(f'{dataset} l{layer} {run_id}: {len(valid)}/{config["K"]}, '
                  f'median={summary["median_coverage_top5"]}', flush=True)
    if not smoke and (len(summaries) != 16 or len(all_rows)+len(exclusions) != sum(int(row['K']) for row in configs)):
        raise ValueError('Incomplete five-PC sweep')
    (output/'COMPLETED.json').write_text(json.dumps(dict(completed=not smoke, smoke=smoke,
        configurations=len(summaries), components=len(all_rows), exclusions=len(exclusions),
        max_grid_error=max((r['max_grid_error'] for r in all_rows), default=0.),
        max_refinement_change=max((r['max_refinement_change'] for r in all_rows), default=0.))))


def collect(root):
    import matplotlib.pyplot as plt
    summaries, entries = [], {'median': [], 'mean': []}
    for dataset, label, layer, _, _ in SOURCES:
        source = root/f'{dataset}_l{layer:02}'
        if not json.loads((source/'COMPLETED.json').read_text())['completed']:
            raise ValueError('Incomplete five-PC case')
        protocol = json.loads((source/'manifest.json').read_text())
        manifest = protocol['baseline_manifest']
        components = read_csv(source/'components.csv')
        cases = read_csv(source/'summary.csv')
        if len(cases) != 16:
            raise ValueError('Incomplete five-PC sweep')
        for summary in cases:
            rows = [r for r in components if r['run_id'] == summary['run_id']]
            if len(rows) != int(summary['eligible']):
                raise ValueError('Summary count mismatch')
            for field in ['coverage_top5', 'coverage_random', 'delta_coverage', 'coverage_pc1_same_subset', 'delta_pc1_same_subset']:
                for aggregate, function in [('median', np.median), ('mean', np.mean)]:
                    value = summary[aggregate+'_'+field]
                    if rows and not math.isclose(float(value), float(function([float(r[field]) for r in rows])), abs_tol=1e-12):
                        raise ValueError('Aggregate mismatch')
                    if not rows and value:
                        raise ValueError('Empty population must have undefined scores')
            summaries.append(dict(dataset=dataset, layer=layer, **summary))
        for aggregate in entries:
            selected = [dict(K=int(s['K']), kind=s['kind'], method=NAMES[s['kind']],
                retention=float(s[aggregate+'_coverage_top5']) if s[aggregate+'_coverage_top5'] else float('nan'),
                enrichment=float(s[aggregate+'_delta_coverage']) if s[aggregate+'_delta_coverage'] else float('nan'),
                eligible=int(s['eligible']), total=int(s['total'])) for s in cases if float(s['threshold']) == .1]
            entries[aggregate].append(dict(label=label, layer=layer, dimension=manifest['dimension'],
                variance=manifest['global_pca']['retained_variance_fraction'], selected=selected))
    for aggregate, table_entries in entries.items():
        fig = panel_tables(table_entries, score_formatter=lambda x: f'{100*x:.2f}'.replace('.', ',') if math.isfinite(x) else 'N/D')
        fig._suptitle.set_text(f'Supporto kNN · Media PC1–PC5 · {"Mediane" if aggregate == "median" else "Medie"} tra componenti')
        for ax, entry in zip(fig.axes, table_entries):
            table = ax.tables[0]
            table.set_fontsize(11)
            table[0, 2].get_text().set_text('Copertura PC1–5 (%) ↑')
            table[0, 3].get_text().set_text('Δ copertura (pp) ↑')
            ordered = sorted(entry['selected'], key=lambda r: (r['K'], r['kind']))
            for start in [1, 3]:
                for col, field in [(2, 'retention'), (3, 'enrichment')]:
                    pair = [ordered[start-1][field], ordered[start][field]]
                    if not all(math.isfinite(x) for x in pair) or pair[0] == pair[1]:
                        for i in [start, start+1]:
                            table[i, col].set_text_props(weight='normal', color='#23364A')
        fig.texts[-4].set_text('k=20 · stessa calibrazione PC1 · soglia di rango 0,1 · media aritmetica dei cinque score per componente')
        fig.texts[-3].set_text('Ogni PC usa il proprio segmento 80% e 32 controlli casuali della stessa lunghezza. Δ = media delle cinque differenze.')
        fig.texts[-2].set_text('Valutabili: tutte e cinque le PC definite e trattenute; rango ≥5. N/D: nessuna componente valida; nessuna media su meno PC.')
        fig.texts[-1].set_text('Popolazioni ristrette rispetto a PC1; confrontare anche PC1 sullo stesso sottoinsieme. Nessun test di significatività.')
        name = '02_pannello_tabelle' if aggregate == 'median' else '03_pannello_medie'
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(root/f'{name}.{extension}', dpi=200, facecolor='white')
        plt.close(fig)
    write_csv(root/'tutte_le_64_configurazioni.csv', summaries)
    write_csv(root/'tabella_soglia_01.csv', [r for r in summaries if float(r['threshold']) == .1])
    (root/'COMPLETED.json').write_text(json.dumps(dict(completed=True, experiments=4, configurations=64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment-index', type=int, choices=range(4))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in [None, 'batch', 'extern']:
        raise RuntimeError('Run numerical work via srun on a compute node')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.collect_only:
        collect(args.output)
    elif args.experiment_index is not None:
        run_case(args.experiment_index, args.output, args.workers, args.smoke)
    else:
        parser.error('Specify case or collect-only')


if __name__ == '__main__':
    main()
