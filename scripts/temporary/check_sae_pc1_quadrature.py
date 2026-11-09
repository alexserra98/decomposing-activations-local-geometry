"""Compare independent numerical integrations of captured failing density lines."""
from pathlib import Path
import json
import math
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[2] / 'outputs/experiments/sae_pc1_integration_diagnosis'


def check(path):
    info = json.loads(path.read_text())
    arrays = np.load(path.with_suffix('.npz'))
    n2, dot, breaks = arrays['norm2'], arrays['dot'], arrays['breaks']
    length, k, dim = info['half_length'], info['k'], info['dimension']
    shift, constant = info['shift'], info['log_constant']

    def f(s):
        d2 = n2 - 2 * length * s * dot + (length * s)**2
        r2 = np.partition(d2, k - 1)[k - 1]
        return math.exp(constant - dim / 2 * math.log(r2) - shift)

    trials = []
    for label, points, tolerance, limit in [
        ('original_breaks_higher_limit', breaks, 1e-4, 10000),
        ('401_pilot_points', np.unique(np.r_[breaks, np.linspace(-1, 1, 401)[1:-1]]), 1e-4, 10000),
        ('1601_pilot_points', np.unique(np.r_[breaks, np.linspace(-1, 1, 1601)[1:-1]]), 1e-4, 10000),
        ('1601_pilot_points_strict', np.unique(np.r_[breaks, np.linspace(-1, 1, 1601)[1:-1]]), 1e-7, 20000),
    ]:
        result = quad(f, -1, 1, points=points, epsabs=info['integral'] * tolerance / 100,
                      epsrel=tolerance, limit=limit, full_output=1)
        value, error, details = result[:3]
        trial = dict(method=label, value=value, reported_relative_error=error/value,
                     warning=result[3] if len(result)>3 else None, evaluations=details['neval'])
        trials.append(trial)
        print(path.stem, json.dumps(trial), flush=True)

    for count in [20001, 80001]:
        positions = np.linspace(-1, 1, count)
        values, nearest_ids, radii, ratios = [], [], [], []
        for chunk in np.array_split(positions, math.ceil(count / 256)):
            offsets = length * chunk[:, None]
            d2 = n2 - 2 * offsets * dot + offsets**2
            indices = np.argpartition(d2, k - 1, axis=1)[:, k-1]
            r2 = d2[np.arange(len(chunk)), indices]
            stable = (offsets - dot)**2 + (n2-dot**2)
            stable_r2 = np.partition(stable, k - 1, axis=1)[:, k-1]
            ratios.append(np.max(np.abs(stable_r2-r2)/r2))
            values.append(np.exp(constant - dim/2 * np.log(r2) - shift))
            nearest_ids.append(indices)
            radii.append(r2)
        values = np.concatenate(values)
        nearest_ids = np.concatenate(nearest_ids)
        radii = np.concatenate(radii)
        trial = dict(method=f'trapezoid_{count}', value=float(np.trapezoid(values, positions)),
                     neighbor_switches=int(np.sum(nearest_ids[1:] != nearest_ids[:-1])),
                     max_radius_formula_relative_difference=float(max(ratios)),
                     min_squared_radius=float(radii.min()), max_squared_radius=float(radii.max()),
                     scaled_density_min=float(values.min()), scaled_density_max=float(values.max()))
        trials.append(trial)
        print(path.stem, json.dumps(trial), flush=True)
    reference = trials[-1]['value']
    for trial in trials:
        trial['relative_difference_from_dense_grid'] = abs(trial['value']/reference-1)
    result = dict(original=info, comparisons=trials,
                  original_relative_difference_from_dense_grid=abs(info['integral']/reference-1))
    path.with_name(path.stem+'_comparison.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        check(ROOT / (name+'.json'))
