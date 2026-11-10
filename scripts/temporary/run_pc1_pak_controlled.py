"""Controlled PAk repeat of the four saved PCA75 PC1-density experiments."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import socket
import warnings
import hashlib
from scipy.special import gammaln, logsumexp

import numpy as np
import torch

from dalg.analysis.pc1_density import KNNLogDensity, random_unit_directions, score_component
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from export_pc1_density_overview import SOURCES, NAMES, panel_tables, read_csv, score_text
from run_years_pc1_density import sha256, write_csv


def pak_profile_likelihood(distances, dimension, kstar):
    """Solve DADApy's two-parameter PAk shell likelihood in log space.

    For shells j=1..k, L=k*b+a*sum(j)-sum(v_j*exp(b+a*j)).
    Eliminate b analytically. The remaining slope root is the monotone
    equation E_a[j]=(k+1)/2, so bisection avoids singular Newton Hessians.
    Shell volumes and interpolation error retain DADApy's conventions.
    """
    count = int(kstar.max())
    radii = distances[:, :count+1]
    ratio = radii[:, :-1]/radii[:, 1:]
    tied = np.abs(ratio-1) < np.finfo(ratio.dtype).resolution
    ratio[tied] -= 10*np.finfo(ratio.dtype).resolution
    exponent = dimension*np.log(radii[:, 1:])+np.log1p(-ratio**dimension)
    log_volume = np.minimum(exponent, 300.) + dimension/2*math.log(math.pi)-gammaln(dimension/2+1)
    indices = np.arange(1, count+1, dtype=np.float64)[None]
    valid = indices <= kstar[:, None]
    if not np.isfinite(log_volume[valid]).all():
        raise ValueError('Invalid PAk shell volume')
    log_volume[~valid] = -np.inf
    target = (kstar+1)/2

    def expected_index(slope):
        log_weights = log_volume+slope[:, None]*indices
        weights = np.exp(log_weights-log_weights.max(1, keepdims=True))
        return (weights*indices).sum(1)/weights.sum(1)

    left, right = -np.ones(len(kstar)), np.ones(len(kstar))
    for _ in range(60):
        expand_left = expected_index(left) > target
        expand_right = expected_index(right) < target
        if not (expand_left.any() or expand_right.any()):
            break
        left[expand_left] *= 2
        right[expand_right] *= 2
    else:
        raise RuntimeError('Cannot bracket PAk profile-likelihood root')
    for _ in range(80):
        midpoint = (left+right)/2
        below = expected_index(midpoint) < target
        left = np.where(below, midpoint, left)
        right = np.where(below, right, midpoint)
    slope = (left+right)/2
    if np.max(np.abs(expected_index(slope)-target)) > 1e-9:
        raise RuntimeError('PAk profile-likelihood root did not converge')
    density = np.log(kstar)-logsumexp(log_volume+slope[:, None]*indices, axis=1)
    m = kstar-1
    error = np.sqrt((4*m+2)/(m*(m-1)))
    return density, error


class PAkDensity:
    """DADApy PAk interpolation with fixed dimension and at least four shells.

    Three-shell interpolation produces near-poles at first-neighbor switches.
    A minimum of four keeps the two-parameter likelihood fit away from that
    boundary. DADApy selects k; a log-space profile solve evaluates its shell likelihood.
    """

    def __init__(self, reference, maxk=100, minimum_k=4):
        from dadapy.density_estimation import DensityEstimation
        self.reference = np.asarray(reference, dtype=np.float64)
        if self.reference.ndim != 2 or not np.isfinite(self.reference).all():
            raise ValueError('Reference must be a finite matrix')
        if not 3 <= minimum_k < maxk-1:
            raise ValueError('Invalid minimum PAk neighborhood')
        self.dimension = self.reference.shape[1]
        self.minimum_k = minimum_k
        self.data = DensityEstimation(coordinates=self.reference, maxk=maxk, n_jobs=1)
        self.data.set_id(self.dimension)
        self.data.compute_distances()

    def from_neighbors(self, distances, indices):
        from dadapy._cython import cython_density as cd
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            # DADApy applies its own deterministic tie perturbation at exact switches.
            warnings.filterwarnings('ignore', message='Found nearest neighbours at identical distance.*')
            kstar = cd._compute_kstar_interp(self.dimension, len(distances), self.data.maxk,
                1e-6, indices, distances, self.data.distances, False, False)
            density, error = pak_profile_likelihood(
                distances, self.dimension, np.maximum(kstar, self.minimum_k))
            density -= math.log(len(self.reference))
        if not np.isfinite(density).all() or not np.isfinite(error).all():
            raise ValueError('Nonfinite PAk output')
        return torch.from_numpy(density), torch.from_numpy(error)

    def query(self, points):
        from dadapy._utils.utils import compute_cross_nn_distances
        distances, indices = compute_cross_nn_distances(np.asarray(points, dtype=np.float64),
                                                       self.reference, self.data.maxk)
        return self.from_neighbors(distances, indices)

    def line_query(self, norms, dots, length, positions):
        """Use exact line-distance polynomials and DADApy's unchanged likelihood."""
        logs = []
        for chunk in np.array_split(positions, max(1, math.ceil(len(positions)/256))):
            offsets = length*chunk
            distance2 = norms[None]-2*offsets[:, None]*dots[None]+offsets[:, None]**2
            indices = np.argpartition(distance2, self.data.maxk-1, axis=1)[:, :self.data.maxk]
            values = np.take_along_axis(distance2, indices, axis=1)
            order = np.argsort(values, axis=1)
            values = np.take_along_axis(values, order, axis=1)
            indices = np.ascontiguousarray(np.take_along_axis(indices, order, axis=1), dtype=np.int64)
            if (values <= 0).any():
                raise ValueError('PAk encountered a zero or negative query radius')
            logs.append(self.from_neighbors(np.sqrt(values), indices)[0].numpy())
        return np.concatenate(logs)


def nearest_neighbor_switches(norms, slopes):
    """Breakpoints of the lower envelope of squared-distance affine terms."""
    hull, starts = [], []
    for index in np.argsort(-slopes, kind='stable'):
        if hull and slopes[index] == slopes[hull[-1]]:
            if norms[index] >= norms[hull[-1]]:
                continue
            hull.pop()
            starts.pop()
        start = -np.inf
        while hull:
            start = (norms[index]-norms[hull[-1]])/(slopes[hull[-1]]-slopes[index])
            if start > starts[-1]:
                break
            hull.pop()
            starts.pop()
        hull.append(index)
        starts.append(start if len(hull) > 1 else -np.inf)
    return np.asarray([x for x in starts if -1 < x < 1])


def integrate_line_simpson(log_density, breaks, *, tolerance=1e-4, max_evaluations=200000):
    """Batched adaptive Simpson integration with explicit neighbor-switch splits.

    Endpoint samples retain narrow peaks at neighbor switches. Compare coarse
    and refined Simpson integrals without the usual /15 error reduction, and
    force one global refinement before accepting the sum of absolute errors.
    """
    nodes = np.unique(np.r_[-1., breaks, 1.])
    intervals = np.column_stack([nodes[:-1], nodes[1:]])
    evaluations = 0

    def evaluate(bounds):
        nonlocal evaluations
        positions = bounds[:, :1] + (bounds[:, 1:]-bounds[:, :1])*np.linspace(0, 1, 5)
        unique, inverse = np.unique(positions, return_inverse=True)
        values = log_density(unique)[inverse].reshape(positions.shape)
        evaluations += len(unique)
        widths = bounds[:, 1]-bounds[:, 0]
        coarse = np.log(widths/6)+logsumexp(values[:, [0, 2, 4]]+np.log([1, 4, 1]), axis=1)
        fine = np.log(widths/12)+logsumexp(values+np.log([1, 4, 2, 4, 1]), axis=1)
        return coarse, fine

    coarse, fine = evaluate(intervals)
    for iteration in range(60):
        total = logsumexp(fine)
        errors = np.abs(np.exp(coarse-total)-np.exp(fine-total))
        relative_error = float(errors.sum())
        if iteration > 0 and relative_error <= tolerance:
            return float(total-math.log(2)), relative_error, evaluations
        if evaluations > max_evaluations:
            break
        if iteration == 0:
            selected = np.arange(len(intervals))
        else:
            order = np.argsort(-errors)
            count = np.searchsorted(np.cumsum(errors[order]), .75*relative_error)+1
            selected = order[:count]
        left, right = intervals[selected].T
        midpoint = (left+right)/2
        children = np.concatenate([np.column_stack([left, midpoint]), np.column_stack([midpoint, right])])
        child_coarse, child_fine = evaluate(children)
        keep = np.ones(len(intervals), dtype=bool)
        keep[selected] = False
        intervals = np.concatenate([intervals[keep], children])
        coarse, fine = np.r_[coarse[keep], child_coarse], np.r_[fine[keep], child_fine]
    raise RuntimeError(f'PAk line integral did not converge: relative_error={relative_error}, queries={evaluations}')


def integrate_line(log_density, breaks, *, tolerance=1e-4, max_evaluations=200000, peak_positions=None):
    """Adaptive open Gauss quadrature; isolated split-point values carry no mass.

    Compare 8- and 16-node rules and sum absolute differences in log space.
    One global refinement and the unchanged relative tolerance are required.
    """
    peaks = np.asarray(breaks if peak_positions is None else peak_positions)
    offsets = np.r_[-10.**np.arange(-2, -13, -1), 10.**np.arange(-2, -13, -1)]
    padding = (peaks[:, None]+offsets).ravel() if len(peaks) else np.array([])
    padding = padding[(padding > -1) & (padding < 1)]
    nodes = np.unique(np.r_[-1., breaks, padding, 1.])
    intervals = np.column_stack([nodes[:-1], nodes[1:]])
    rules = [np.polynomial.legendre.leggauss(order) for order in [8, 16]]
    evaluations = 0

    def evaluate(bounds):
        nonlocal evaluations
        half_width = (bounds[:, 1]-bounds[:, 0])/2
        midpoint = bounds[:, 0]+half_width
        estimates = []
        for positions, weights in rules:
            queries = midpoint[:, None]+half_width[:, None]*positions
            unique, inverse = np.unique(queries, return_inverse=True)
            values = log_density(unique)[inverse].reshape(queries.shape)
            evaluations += len(unique)
            estimates.append(np.log(half_width)+logsumexp(values+np.log(weights), axis=1))
        return estimates

    coarse, fine = evaluate(intervals)
    for iteration in range(10000):
        total = logsumexp(fine)
        difference = np.abs(coarse-fine)
        with np.errstate(divide='ignore'):
            log_errors = np.maximum(coarse, fine)+np.log(-np.expm1(-difference))
        log_relative_error = float(logsumexp(log_errors)-total)
        if iteration > 0 and log_relative_error <= math.log(tolerance):
            return float(total-math.log(2)), math.exp(log_relative_error), evaluations
        if evaluations > max_evaluations:
            break
        if iteration == 0:
            selected = np.arange(len(intervals))
        else:
            order = np.argsort(-log_errors)
            weights = np.exp(log_errors[order]-log_errors[order[0]])
            count = np.searchsorted(np.cumsum(weights), .75*weights.sum())+1
            selected = order[:count]
        left, right = intervals[selected].T
        midpoint = left+(right-left)/2
        if np.any((midpoint <= left) | (midpoint >= right)):
            raise RuntimeError('PAk quadrature reached floating-point interval resolution')
        children = np.concatenate([np.column_stack([left, midpoint]), np.column_stack([midpoint, right])])
        child_coarse, child_fine = evaluate(children)
        keep = np.ones(len(intervals), dtype=bool)
        keep[selected] = False
        intervals = np.concatenate([intervals[keep], children])
        coarse, fine = np.r_[coarse[keep], child_coarse], np.r_[fine[keep], child_fine]
    raise RuntimeError(f'PAk open quadrature did not converge: log_relative_error={log_relative_error}, queries={evaluations}')


def initialize_worker(reference, maxk):
    global DENSITY, KNN_DENSITY
    torch.set_num_threads(1)
    DENSITY = PAkDensity(reference, maxk)
    KNN_DENSITY = KNNLogDensity(torch.from_numpy(reference), k=20)


def score(center, vector, length, component, *, include_knn=False, max_evaluations=200000, pilot_points=201):
    directions = torch.cat([vector[None], random_unit_directions(len(vector), 32, 42+component)])
    center_density, center_error = DENSITY.query(center[None].numpy())
    positions = np.linspace(-1, 1, pilot_points)
    profiles, averages, convergence, switch_counts = [], [], [], []
    for direction_index, direction in enumerate(directions):
        switches = []
        if isinstance(DENSITY, PAkDensity):
            residual = DENSITY.reference-center.numpy()
            norms, dots = np.square(residual).sum(1), residual@direction.numpy()
            switches = nearest_neighbor_switches(norms, -2*length*dots)
            def line_query(s):
                return DENSITY.line_query(norms, dots, length, s)-float(center_density[0])
        else:
            def line_query(s):
                return DENSITY.query(center.numpy()+length*s[:, None]*direction.numpy())[0].numpy()-float(center_density[0])
        profiles.append(torch.from_numpy(line_query(positions)))
        try:
            average, error, count = integrate_line(line_query, np.r_[positions[1:-1], switches],
                                                   max_evaluations=max_evaluations, peak_positions=switches)
        except RuntimeError as exc:
            raise RuntimeError(f'PAk component {component}, direction {direction_index}: {exc}') from exc
        averages.append(average)
        convergence.append(dict(relative_error=error, evaluations=count))
        switch_counts.append(len(switches))
    retention = torch.tensor(averages, dtype=torch.float64)
    result = dict(log_retention=float(retention[0]),
        log_enrichment=float(retention[0]-torch.logsumexp(retention[1:], 0)+math.log(32)),
        log_direction_retentions=retention, center_log_density=float(center_density[0]),
        center_log_density_error=float(center_error[0]), convergence=convergence,
        nearest_switch_counts=switch_counts,
        quadrature_max_relative_error=max(c['relative_error'] for c in convergence),
        quadrature_max_evaluations=max(c['evaluations'] for c in convergence),
        log_relative_density_profiles=torch.stack(profiles),
        relative_positions=torch.from_numpy(positions))
    if include_knn:
        baseline = score_component(KNN_DENSITY, center, vector, 1., half_length=length,
                                   seed=42+component, adaptive=True)
        result.update(knn_log_retention=baseline['log_retention'], knn_log_enrichment=baseline['log_enrichment'])
    return result


def transport_result(result):
    """Send ordinary arrays rather than shared Torch storages through the pool."""
    return {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in result.items()}


def score_in_worker(center, vector, length, component, knn_baseline=None, max_evaluations=200000):
    result = score(torch.from_numpy(center), torch.from_numpy(vector),
                   length, component, include_knn=knn_baseline is None, max_evaluations=max_evaluations)
    if knn_baseline is not None:
        result.update(knn_baseline)
    return transport_result(result)


def reference_from_manifest(manifest, pca, activation_dir=None):
    """Recover original validation prompt order from the documented split shards."""
    base = Path('/orfeo/cephfs/scratch/dssc/zenocosini/dalg-cache/sae-manifold')
    name = Path(manifest['activation_dir']).name
    root = Path(activation_dir) if activation_dir else base/name
    base = root.parent
    archive = base/'pre_holdout'/name
    archived_cfg = json.loads((archive/'config.json').read_text())
    original = [json.loads(line) for line in (archive/'source/records.jsonl').read_text().splitlines()]
    if archived_cfg['source_records_sha256'] != pca['source']['source_records_sha256']:
        raise ValueError('Archived prompt population differs from the saved PCA')
    expected = manifest['reference_global_rows']
    # Archive global rows index the original snapshot, before the holdout split.
    original_rows = [row for path in sorted((archive/'meta').glob('shard_*.json'))
                     for row in json.loads(path.read_text())['row_indices']]
    if original_rows != list(range(len(original))):
        raise ValueError('Unexpected original prompt indexing')
    wanted = {original[row]['row_id']: index for index, row in enumerate(expected)}
    if len(wanted) != len(expected):
        raise ValueError('Repeated reference prompt identity')
    reference = torch.empty((len(expected), 4096), dtype=torch.float64)
    seen, fingerprints = set(), {}
    for partition in [root, root/'test']:
        cfg = json.loads((partition/'config.json').read_text())
        for field in ['model', 'model_revision', 'tokenizer_revision', 'tokenizer_sha256',
                      'representation', 'd_model', 'window', 'drop_prefix', 'dtype', 'token_selection']:
            if cfg[field] != archived_cfg[field]:
                raise ValueError(f'Re-extraction setting differs: {field}')
        records = [json.loads(line) for line in (partition/'source/records.jsonl').read_text().splitlines()]
        meta = load_meta_index(partition, layer=manifest['layer'])
        positions = [i for i, row in enumerate(meta) if records[row['global_row']]['row_id'] in wanted]
        dataset = ActivationBatchDataset(partition, layer=manifest['layer'], row_subset=positions,
            batch_size=512, drop_prefix=0, dtype=torch.float64, return_metadata=True,
            shuffle_shards=False, shuffle_within_shard=False)
        for x, rows, tok in dataset:
            if tok.any():
                raise ValueError('Expected last-token slot zero')
            for point, row in zip(x, rows.tolist()):
                record = records[row]
                index = wanted[record['row_id']]
                if index in seen or record != original[expected[index]]:
                    raise ValueError('Reference prompt/tokenization mismatch or duplication')
                seen.add(index)
                reference[index] = point
        for path in [partition/'config.json', partition/'source/records.jsonl',
                     *sorted((partition/f'layer{manifest["layer"]:02}').glob('shard_*.pt'))]:
            fingerprints[str(path)] = sha256(path)
    if seen != set(range(len(expected))):
        raise ValueError('Missing original validation prompts')
    projected = (reference-pca['mean'])@pca['basis'].T
    provenance = dict(activation_dir=str(root), original_tensor_identity_verified=False,
        prompt_and_tokenization_identity_verified=True, reference_global_rows=expected,
        reference_sha256=hashlib.sha256(projected.numpy().tobytes()).hexdigest(),
        source_sha256=fingerprints,
        reconstruction='original validation prompt order from re-extracted development and test shards')
    return projected, provenance


def run_experiment(source, output, workers, smoke, maxk, activation_dir=None, resume_from=None, max_evaluations=200000, reuse_pak=False):
    manifest = json.loads((source/'manifest.json').read_text())
    completed = json.loads((source/'COMPLETED.json').read_text())
    if not completed['completed'] or completed['smoke']:
        raise ValueError('Require completed baseline')
    pca_path = Path(manifest['global_pca']['path'])
    if sha256(pca_path) != manifest['global_pca']['sha256']:
        raise ValueError('PCA fingerprint mismatch')
    pca = torch.load(pca_path, weights_only=True, map_location='cpu')
    if pca['dimension'] != manifest['dimension']:
        raise ValueError('Density dimension differs from saved PCA')
    reference, reference_provenance = reference_from_manifest(manifest, pca, activation_dir)
    output.mkdir(parents=True)
    print(f'Loaded {source}: reference={tuple(reference.shape)}; fixed PAk dimension={reference.shape[1]}', flush=True)
    protocol = dict(baseline_path=str(source), baseline_manifest=manifest,
                    baseline_manifest_sha256=sha256(source/'manifest.json'))
    protocol.update(estimator='PAk interpolation, DADApy neighborhoods and profiled likelihood', dadapy_version=importlib.metadata.version('dadapy'),
        reference_provenance=reference_provenance,
        density_dimension=reference.shape[1], dimension_rule='fixed coordinate dimension; no ID estimation',
        alpha=1e-6, maxk=maxk, minimum_kstar=4, likelihood_solver='log-space profile likelihood v1', bonferroni_deloc=False, bonferroni_loc=False,
        integration='adaptive open Gauss 8/16; 201 pilot positions, exact nearest switches and geometric padding 1e-2..1e-12; relative error <=1e-4',
        max_evaluations=max_evaluations, smoke=smoke, execution=dict(hostname=socket.gethostname(), job=os.environ['SLURM_JOB_ID'], step=os.environ['SLURM_STEP_ID']),
        source_sha256=sha256(Path(__file__)), pca_sha256=sha256(pca_path))
    (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
    baseline_rows = read_csv(source/'components.csv')
    resumed = {}
    if resume_from is not None:
        previous = json.loads((resume_from/'manifest.json').read_text())
        if (previous['pca_sha256'] != protocol['pca_sha256']
                or previous['reference_provenance']['reference_sha256'] != reference_provenance['reference_sha256']
                or any(previous[key] != protocol[key] for key in ['alpha', 'maxk', 'minimum_kstar', 'density_dimension'])):
            raise ValueError('Resume data or numerical protocol differs')
        if reuse_pak and any(previous[key] != protocol[key] for key in ['integration', 'likelihood_solver']):
            raise ValueError('Cannot reuse PAk with a different numerical protocol')
        for config in read_csv(resume_from/'summary.csv'):
            path = resume_from/(config['run_id']+'.pt')
            resumed[config['run_id']] = torch.load(path, weights_only=True)
        protocol['resumed_from'] = dict(path=str(resume_from), manifest_sha256=sha256(resume_from/'manifest.json'),
            reused='completed PAk and kNN configurations' if reuse_pak else 'kNN baseline only; all PAk scores recomputed', previous_integration=previous['integration'])
        (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
    summaries, all_rows, cache = [], [], {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(reference.numpy(), maxk)) as pool:
        for config in read_csv(source/'summary.csv')[:1] if smoke else read_csv(source/'summary.csv'):
            run_id = config['run_id']
            bundle = torch.load(source/(run_id+'.pt'), weights_only=True, map_location='cpu')
            geometry = bundle['projected_geometry']
            eligible = sorted(bundle['components'])[:2] if smoke else sorted(bundle['components'])
            signature = hashlib.sha256(b''.join(geometry[key].contiguous().numpy().tobytes()
                for key in ['centers', 'vectors', 'half_lengths']) + str(eligible).encode()).hexdigest()
            restored = resumed.get(run_id)
            cached = restored['components'] if reuse_pak and restored is not None else cache.get(signature)
            if cached is not None and set(cached) != set(eligible):
                raise ValueError('Incomplete cached configuration')
            futures = {} if cached is not None else {i: pool.submit(score_in_worker,
                       geometry['centers'][i].numpy(), geometry['vectors'][i].numpy(),
                       float(geometry['half_lengths'][i]), i,
                       {key: restored['components'][i][key] for key in ['knn_log_retention', 'knn_log_enrichment']}
                       if restored is not None else None, max_evaluations) for i in eligible}
            results = {}
            for i in eligible:
                result = cached[i] if cached is not None else {
                    key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                    for key, value in futures[i].result().items()}
                baseline = bundle['components'][i]
                results[i] = result
                row = dict(run_id=run_id, component=i, kind=config['kind'], K=int(config['K']),
                    threshold=float(config['threshold']), alpha=float(geometry['alpha'][i]),
                    **{key: value for key, value in result.items() if not isinstance(value, (torch.Tensor, list))},
                    historical_knn_log_retention=baseline['log_retention'], historical_knn_log_enrichment=baseline['log_enrichment'])
                all_rows.append(row)
                print(f'{source.name} {run_id} {i}: {row["log_retention"]:.5f}, evaluations={result["quadrature_max_evaluations"]}', flush=True)
            cache[signature] = results
            protocol.setdefault('baseline_bundle_sha256', {})[run_id+'.pt'] = sha256(source/(run_id+'.pt'))
            (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
            torch.save(dict(run_id=run_id, components=results), output/(run_id+'.pt'))
            valid = [r for r in all_rows if r['run_id'] == run_id]
            summary = dict(run_id=run_id, kind=config['kind'], K=int(config['K']), threshold=float(config['threshold']),
                eligible=len(valid), total=int(config['K']))
            for field in ['retention', 'enrichment', 'knn_retention', 'knn_enrichment']:
                key = 'knn_log_'+field[4:] if field.startswith('knn_') else 'log_'+field
                summary['median_log_'+field] = float(np.median([r[key] for r in valid]))
                summary['median_'+field] = math.exp(summary['median_log_'+field])
            summaries.append(summary)
            write_csv(output/'components.csv', all_rows)
            write_csv(output/'summary.csv', summaries)
    expected = sum(not r['exclusion_reason'] for r in baseline_rows)
    if not smoke and (len(summaries) != 16 or len(all_rows) != expected):
        raise ValueError('Incomplete controlled repeat')
    if not smoke:
        initialize_worker(reference.numpy(), maxk)
        checks = []
        for kind in ['hddc', 'kmeans']:
            for K in [50, 100]:
                candidates = [r for r in all_rows if r['kind'] == kind and r['K'] == K]
                row = max(candidates, key=lambda r: r['quadrature_max_relative_error'])
                source_bundle = torch.load(source/(row['run_id']+'.pt'), weights_only=True)
                saved = torch.load(output/(row['run_id']+'.pt'), weights_only=True)['components'][row['component']]
                geom, i = source_bundle['projected_geometry'], row['component']
                dense = score(geom['centers'][i], geom['vectors'][i], float(geom['half_lengths'][i]), i, pilot_points=401, max_evaluations=max_evaluations)
                change = float((dense['log_direction_retentions']-saved['log_direction_retentions']).abs().max())
                enrichment_change = abs(dense['log_enrichment']-saved['log_enrichment'])
                checks.append(dict(run_id=row['run_id'], component=i, max_direction_log_change=change,
                    enrichment_log_change=enrichment_change, passed=max(change, enrichment_change) < .001))
        validation = dict(passed=all(c['passed'] for c in checks), checks=checks, tolerance_log=.001, pilot_positions=[201, 401])
        (output/'integration_validation.json').write_text(json.dumps(validation, indent=2))
        if not validation['passed']:
            raise RuntimeError('Denser-pilot PAk validation failed')
    (output/'COMPLETED.json').write_text(json.dumps(dict(completed=not smoke, smoke=smoke,
        configurations=len(summaries), eligible=len(all_rows))))
    return manifest, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--maxk', type=int, default=100)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-evaluations', type=int, default=200000)
    parser.add_argument('--reuse-pak', action='store_true', help='Reuse completed configurations only with identical PAk integration and solver')
    parser.add_argument('--experiment-index', type=int, choices=range(4))
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--resume-from', type=Path, help='Completed configurations from an earlier result root')
    parser.add_argument('--years-activation-dir', type=Path, help='Current development extraction; archive is read from its parent/pre_holdout')
    parser.add_argument('--temperature-activation-dir', type=Path, help='Current development extraction; archive is read from its parent/pre_holdout')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in [None, 'batch', 'extern']:
        raise RuntimeError('Run numerical work through srun on a compute node')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=args.collect_only or args.experiment_index is not None)
    entries, selected, all_configurations = [], [], []
    for experiment_index, (dataset, label, layer, source, _) in enumerate(SOURCES):
        if args.experiment_index is not None and args.experiment_index != experiment_index:
            continue
        activation_dir = args.years_activation_dir if dataset == 'years_neutral' else args.temperature_activation_dir
        experiment_output = args.output/f'{dataset}_l{layer:02}'
        if args.collect_only:
            completed = json.loads((experiment_output/'COMPLETED.json').read_text())
            if not completed['completed'] or completed['smoke']:
                raise ValueError('Cannot export incomplete experiments')
            if not json.loads((experiment_output/'integration_validation.json').read_text())['passed']:
                raise ValueError('Cannot export unverified quadrature')
            manifest = json.loads((experiment_output/'manifest.json').read_text())['baseline_manifest']
            summaries = read_csv(experiment_output/'summary.csv')
            for row in summaries:
                for key in ['K', 'eligible', 'total']:
                    row[key] = int(row[key])
                for key in ['threshold', 'median_retention', 'median_enrichment', 'median_knn_retention', 'median_knn_enrichment']:
                    row[key] = float(row[key])
        else:
            manifest, summaries = run_experiment(source, experiment_output,
                args.workers, args.smoke, args.maxk, activation_dir,
                args.resume_from/f'{dataset}_l{layer:02}' if args.resume_from else None,
                args.max_evaluations, args.reuse_pak)
        entry = dict(label=label, layer=layer, dimension=manifest['dimension'],
            variance=manifest['global_pca']['retained_variance_fraction'], selected=[])
        for summary in summaries:
            all_configurations.append(dict(dataset=dataset, layer=layer, **summary))
            if summary['threshold'] != .1:
                continue
            row = dict(dataset=dataset, layer=layer, dimension=manifest['dimension'],
                K=summary['K'], kind=summary['kind'], method=NAMES[summary['kind']],
                retention=summary['median_retention'], enrichment=summary['median_enrichment'],
                knn_retention=summary['median_knn_retention'], knn_enrichment=summary['median_knn_enrichment'],
                eligible=summary['eligible'], total=summary['total'])
            entry['selected'].append(row)
            selected.append(row)
        entries.append(entry)
    if not args.smoke and args.experiment_index is None:
        import matplotlib.pyplot as plt
        # Recompute displayed medians from component rows before rendering.
        checked = 0
        for dataset, _, layer, _, _ in SOURCES:
            rows = read_csv(args.output/f'{dataset}_l{layer:02}'/'components.csv')
            for summary in [r for r in all_configurations if r['dataset'] == dataset and r['layer'] == layer]:
                components = [r for r in rows if r['run_id'] == summary['run_id']]
                if len(components) != summary['eligible']:
                    raise ValueError('Component count differs from summary')
                for field in ['retention', 'enrichment', 'knn_retention', 'knn_enrichment']:
                    key = 'knn_log_'+field[4:] if field.startswith('knn_') else 'log_'+field
                    expected = math.exp(float(np.median([float(r[key]) for r in components])))
                    if not math.isclose(expected, float(summary['median_'+field]), abs_tol=1e-10, rel_tol=1e-10):
                        raise ValueError('Table median differs from component scores')
                checked += len(components)
        if len(selected) != 16 or len(all_configurations) != 64:
            raise ValueError('Expected all four sweeps before table export')
        baseline_entries = copy.deepcopy(entries)
        for entry in baseline_entries:
            for row in entry['selected']:
                row['retention'], row['enrichment'] = row['knn_retention'], row['knn_enrichment']
        for table_entries, name, title in [
            (entries, '02_pannello_tabelle', 'PAk controllato (k* ≥ 4) · Retention ed enrichment'),
            (baseline_entries, 'knn_baseline_tabelle', 'kNN k=20 · Baseline sugli stessi riferimenti sostitutivi')]:
            def format_score(value):
                return f'{value:.3e}' if value >= 10000 or 0 < value < .001 else score_text(value)
            fig = panel_tables(table_entries, score_formatter=format_score)
            fig._suptitle.set_text(title)
            for extension in ['png', 'pdf', 'svg']:
                fig.savefig(args.output/f'{name}.{extension}', dpi=200, facecolor='white')
            plt.close(fig)
        write_csv(args.output/'tabella_soglia_01.csv', selected)
        write_csv(args.output/'tutte_le_64_configurazioni.csv', all_configurations)
        (args.output/'report.md').write_text(
            '# Controlled PAk comparison\n\n'
            'PAk and kNN k=20 use the same replacement reference activations, saved PCA75, '
            'projected original PC1 intervals and 32 random controls. Density dimension equals '
            'coordinate dimension; no intrinsic dimension is estimated.\n\n'
            'PAk uses DADApy point-adaptive neighborhoods with a minimum k*=4 (maximum 100, '
            'alpha=1e-6). The minimum excludes the near-poles observed with k*=3 at nearest-neighbor '
            'switches. Numerical quadrature splits explicitly at those switches and requires '
            'relative integration error at most 1e-4.\n\n'
            'The four PAk tables preserve the original layout and rank cutoff 0.1. '
            'Scores exponentiate medians of natural-log component scores. Bold values are '
            'descriptive comparisons, not significance tests. All 64 summaries were checked '
            'against component rows for both estimators.\n\n'
            'Reference prompt membership and tokenization match the historical validation split, '
            'but original tensor identity is not verified: activations were re-extracted. '
            'Compare PAk with the recomputed kNN baseline, rather than attributing changes '
            'from historical numbers solely to the estimator. No direction-swap or intrinsic-dimension '
            'PAk experiment was run.\n\n'
            '![PAk tables](02_pannello_tabelle.png)\n\n'
            '![Recomputed kNN baseline](knn_baseline_tabelle.png)\n')
        (args.output/'export_validation.json').write_text(json.dumps(dict(passed=True,
            checked_component_entries=checked, configuration_summaries=64, displayed_rows=16,
            estimators=['PAk minimum k*=4', 'kNN k=20']), indent=2))
        (args.output/'COMPLETED.json').write_text(json.dumps(dict(completed=True, experiments=4, tables=4)))


if __name__ == '__main__':
    main()
