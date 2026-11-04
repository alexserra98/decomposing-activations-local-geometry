"""Capture failing SAE PC1 quadratures without changing checkpoints or metrics."""
from pathlib import Path
import inspect
import json
import math
import warnings

import numpy as np
from scipy import integrate
import torch

from dalg.evaluation.common import load_model, load_evaluation_data
import dalg.evaluation.pc1_density as evaluator

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/experiments/sae_pc1_integration_diagnosis'
torch.set_num_threads(4)
original_quad = integrate.quad
original_score = evaluator.score_component
context = {}


def tracked_score(*args, **kwargs):
    context['component'] = kwargs['seed'] - 42
    context['direction'] = 0
    return original_score(*args, **kwargs)


def tracked_quad(function, a, b, **kwargs):
    result = original_quad(function, a, b, **kwargs)
    if kwargs.get('full_output'):
        direction = context['direction']
        context['direction'] += 1
        if len(result) > 3:
            outer = inspect.getclosurevars(function).nonlocals
            inner = inspect.getclosurevars(outer['log_density']).nonlocals
            value, error, info, message = result[:4]
            estimator = inner['self']
            details = {
                'run_id': context['run_id'], 'component': context['component'],
                'direction': direction, 'direction_kind': 'pc1' if direction == 0 else 'random',
                'dimension': estimator.dimension, 'reference_count': len(estimator.reference),
                'k': estimator.k, 'half_length': inner['half_length'], 'shift': outer['shift'],
                'log_constant': estimator.log_constant,
                'integral': value, 'reported_absolute_error': error,
                'reported_relative_error': error / value,
                'neval': info['neval'], 'intervals': info['last'],
                'breakpoints': len(kwargs['points']), 'message': message,
                'epsabs': kwargs['epsabs'], 'epsrel': kwargs['epsrel'],
            }
            np.savez(OUTPUT / (context['tag'] + '.npz'), norm2=inner['norm2'],
                     dot=inner['dot'], breaks=kwargs['points'])
            (OUTPUT / (context['tag'] + '.json')).write_text(json.dumps(details, indent=2) + '\n')
            print('CAPTURED', json.dumps(details), flush=True)
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((ROOT / 'outputs/experiments/sae_manifold_temperature_years_submission/full_unsupervised_submission.json').read_text())['jobs']
    samples = [('temperature_02d_years_neutral_kmeans_pca', 0),
               ('temperature_02d_years_neutral_kmeans_pca', 18),
               ('temperature_02d_hddc_em', 3)]
    integrate.quad = tracked_quad
    evaluator.score_component = tracked_score
    for name, index in samples:
        job = next(job for job in jobs if job['name'] == name)
        run = json.loads(Path(job['manifest']).read_text().splitlines()[index])
        context.update(tag=f'{name}_{index}', run_id=run['run_id'])
        print('REPRODUCE', context['tag'], flush=True)
        kind = run['training']['model_kind']
        model = load_model(Path(run['run_dir']), kind)
        data = load_evaluation_data(run['run_dir'], shard_dir=run['dataset']['shard_dir'],
                                    layer=run['dataset']['layer'], model=model, model_kind=kind)
        try:
            evaluator.evaluate_pc1_density(model, data, model_kind=kind,
                                           layer=run['dataset']['layer'], batch_size=4096)
        except RuntimeError as exc:
            if 'Adaptive density integration failed' not in str(exc):
                raise
            print('REPRODUCED', context['tag'], flush=True)
        else:
            print('DID NOT REPRODUCE', context['tag'], flush=True)


if __name__ == '__main__':
    main()
