"""Raw control-step force/reward distributions, before PPO normalization.

One summary per rollout, including episodes that have not finished. Exact
quantiles are cheap at rollout size; no samples are retained across rollouts.
"""
import numpy as np


METRICS = ('peak_force_n', 'sum_bad_force_n', 'capped_bad_force_n',
           'r_collision', 'r_severe', 'r_collision_total', 'r_task', 'severe_event')


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {'count': 0}
    if not np.isfinite(values).all():
        raise ValueError('Calibration samples must be finite.')
    percentiles = np.percentile(values, [0, 5, 50, 90, 95, 99, 100])
    out = dict(zip(('min', 'p05', 'p50', 'p90', 'p95', 'p99', 'max'),
                   map(float, percentiles)))
    nonzero = values[values != 0]
    out.update(count=int(values.size), mean=float(values.mean()), std=float(values.std()),
               nonzero_rate=float(nonzero.size / values.size))
    # Rare impacts must not disappear into an otherwise all-zero percentile.
    out['nonzero'] = ({'count': int(nonzero.size), 'mean': float(nonzero.mean()),
                       'p50': float(np.percentile(nonzero, 50)),
                       'p95': float(np.percentile(nonzero, 95))}
                      if nonzero.size else {'count': 0})
    return out


def summarise_calibration(samples):
    """Summarize (task mode, raw metrics) pairs over every control step."""
    def summarise(rows):
        return {name: distribution([row[name] for row in rows]) for name in METRICS}
    return dict(overall=summarise([row for _, row in samples]),
                by_mode={mode: summarise([row for m, row in samples if m == mode])
                         for mode in sorted({m for m, _ in samples})})
