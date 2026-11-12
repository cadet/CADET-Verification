# -*- coding: utf-8 -*-
"""Qualitative metrics of a chromatogram against a reference chromatogram.

Error norms do not tell whether an approximation oscillates, clips a peak or
smears a front. The metrics below quantify these features of a signal c(t)
against a reference c_ref(t) given on the same time grid:

    Oscillation
        Excess TV            (TV(c) - TV(c_ref)) / TV(c_ref)
        Spurious extrema     turning points of c minus those of c_ref, where
                             only reversals larger than ``noise`` count
        Local overshoot      max_t (c(t) - max of c_ref over [t-w, t+w])^+
        Local undershoot     max_t (min of c_ref over [t-w, t+w] - c(t))^+
                             Both relative to max|c_ref|. A front that is
                             merely shifted or smeared by less than w stays
                             inside the local range of the reference, an
                             over- or undershoot does not.
        Negative mass        int min(c, 0) dt / int |c_ref| dt   (<= 0)
    Peak
        Peak height err.     (max c - max c_ref) / max c_ref
        Peak time err.       t_peak - t_peak_ref (parabolic interpolation)
        Peak width err.      relative error of the width at half height
    Elution signal
        Elution start err.   first rising crossing of 5% of the peak
        Front position err.  first rising crossing of 50%
        Front width err.     error of t(95%) - t(5%) on the rise, in time units,
                             since the front of a discontinuous reference has
                             no width
        Tail position err.   last falling crossing of 50%
        Elution end err.     last falling crossing of 5%
    Moments
        Mass err.            relative error of int c dt
        Ret. time err.       mu_1 - mu_1_ref
        Variance err.        relative error of the second central moment

Positions (elution start and end, front and tail position) are crossings of
levels that refer to the peak height of the reference, so that both signals
are measured against the same levels. Widths (peak width, front width)
describe the shape and refer to the peak height of the signal itself, which
keeps them defined for a smeared peak. Crossing times are linearly
interpolated between samples. Metrics that are not defined for a signal, e.g.
a falling crossing of a breakthrough curve, are NaN.
"""

import numpy as np

from src.validation.validation_metrics import pulse_moments, trapezoid

#: Order of the metrics in tables.
METRIC_NAMES = [
    'Excess TV', 'Spurious extrema', 'Local overshoot', 'Local undershoot',
    'Negative mass',
    'Peak height err.', 'Peak time err.', 'Peak width err.',
    'Elution start err.', 'Front position err.', 'Front width err.',
    'Tail position err.', 'Elution end err.',
    'Mass err.', 'Ret. time err.', 'Variance err.',
]


def total_variation(c):
    """Total variation of a sampled signal."""
    return float(np.sum(np.abs(np.diff(np.asarray(c, dtype=float)))))


def count_turning_points(c, noise):
    """Number of local extrema of c, ignoring reversals smaller than noise.

    The signal is followed with a hysteresis of size ``noise``: a turning
    point is counted once the signal has moved back by more than ``noise``
    from its running extremum.
    """
    c = np.asarray(c, dtype=float)
    if c.size < 3:
        return 0
    count = 0
    direction = 0  # +1 rising, -1 falling, 0 not yet determined
    extremum = c[0]
    for value in c[1:]:
        if direction >= 0 and value > extremum:
            extremum = value
            direction = 1 if direction == 0 and value - c[0] > noise else direction
        elif direction <= 0 and value < extremum:
            extremum = value
            direction = -1 if direction == 0 and c[0] - value > noise else direction
        if direction == 0:
            # the first excursion may also reverse before exceeding the noise level
            if abs(value - c[0]) > noise:
                direction = 1 if value > c[0] else -1
                extremum = value
        elif direction == 1 and extremum - value > noise:
            count += 1
            direction = -1
            extremum = value
        elif direction == -1 and value - extremum > noise:
            count += 1
            direction = 1
            extremum = value
    return count


def _running_extreme(c, half_width, fn):
    """max or min of c over the index window [i - half_width, i + half_width]."""
    c = np.asarray(c, dtype=float)
    if half_width <= 0:
        return c.copy()
    n = c.size
    out = np.empty(n)
    for i in range(n):
        out[i] = fn(c[max(0, i - half_width):min(n, i + half_width + 1)])
    return out


def local_bound_violation(t, c, c_ref, window):
    """(overshoot, undershoot) of c beyond the range that c_ref attains within
    +- window around the same time, in the units of c."""
    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t)))
    half_width = int(round(window / dt)) if dt > 0 else 0
    upper = _running_extreme(c_ref, half_width, np.max)
    lower = _running_extreme(c_ref, half_width, np.min)
    c = np.asarray(c, dtype=float)
    return float(np.max(np.clip(c - upper, 0.0, None))), float(np.max(np.clip(lower - c, 0.0, None)))


def peak(t, c, resolved=0.1):
    """(time, height) of the global maximum.

    A resolved peak is refined by a parabola through the maximal sample and
    its two neighbours. The peak counts as resolved if both neighbours lie
    within the fraction ``resolved`` of the range of the signal below the
    maximum. Otherwise, e.g. at the edge of a rectangular pulse, the parabola
    would overshoot and the maximal sample is returned.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    i = int(np.argmax(c))
    if i == 0 or i == c.size - 1:
        return float(t[i]), float(c[i])
    t0, t1, t2 = t[i - 1], t[i], t[i + 1]
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    if max(y1 - y0, y1 - y2) > resolved * (y1 - float(np.min(c))):
        return float(t1), float(y1)
    # Newton form of the interpolating parabola
    d1 = (y1 - y0) / (t1 - t0)
    d2 = ((y2 - y1) / (t2 - t1) - d1) / (t2 - t0)
    if not d2 < 0.0:
        return float(t1), float(y1)
    tp = 0.5 * (t0 + t1) - 0.5 * d1 / d2
    if not t0 <= tp <= t2:
        return float(t1), float(y1)
    return float(tp), float(y0 + d1 * (tp - t0) + d2 * (tp - t0) * (tp - t1))


def crossing_time(t, c, level, rising=True, last=False):
    """Linearly interpolated time at which c crosses level.

    rising: crossing from below to at-or-above the level, otherwise from
    at-or-above to below. last: last such crossing instead of the first one.
    NaN if there is none.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    above = c >= level
    if rising:
        idx = np.nonzero(~above[:-1] & above[1:])[0]
    else:
        idx = np.nonzero(above[:-1] & ~above[1:])[0]
    if idx.size == 0:
        return float('nan')
    i = int(idx[-1] if last else idx[0])
    return float(t[i] + (level - c[i]) / (c[i + 1] - c[i]) * (t[i + 1] - t[i]))


def _rel(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0.0:
        return float('nan')
    return float((value - reference) / abs(reference))


def signal_metrics(t, c, c_ref, noise=1e-3, window=0.02):
    """Metrics of one signal against its reference, see the module docstring.

    Parameters
    ----------
    t, c, c_ref : array_like
        Time grid, signal and reference signal, all of the same length.
    noise : float
        Reversals below noise * max|c_ref| are not counted as extrema.
    window : float
        Half width of the time window of the local bounds, as a fraction of
        the time span.

    Returns
    -------
    dict
        Metric name -> float, with the keys of METRIC_NAMES.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    c_ref = np.asarray(c_ref, dtype=float)
    if not (t.shape == c.shape == c_ref.shape and t.ndim == 1):
        raise ValueError(
            f"t, c and c_ref must be one dimensional and of equal length, got {t.shape}, {c.shape}, {c_ref.shape}")

    scale = float(np.max(np.abs(c_ref)))
    if not scale > 0.0:
        return {name: float('nan') for name in METRIC_NAMES}

    m = {}

    # oscillation
    tv_ref = total_variation(c_ref)
    m['Excess TV'] = (total_variation(c) - tv_ref) / tv_ref if tv_ref > 0.0 else float('nan')
    m['Spurious extrema'] = float(
        count_turning_points(c, noise * scale) - count_turning_points(c_ref, noise * scale))
    over, under = local_bound_violation(t, c, c_ref, window * (t[-1] - t[0]))
    m['Local overshoot'] = over / scale
    m['Local undershoot'] = under / scale
    m['Negative mass'] = trapezoid(np.minimum(c, 0.0), t) / trapezoid(np.abs(c_ref), t)

    # peak
    tp, hp = peak(t, c)
    tp_ref, hp_ref = peak(t, c_ref)
    m['Peak height err.'] = _rel(hp, hp_ref)
    m['Peak time err.'] = tp - tp_ref

    def crossings(signal, height):
        levels = {q: q * height for q in (0.05, 0.5, 0.95)}
        rise = {q: crossing_time(t, signal, lv, rising=True, last=False) for q, lv in levels.items()}
        fall = {q: crossing_time(t, signal, lv, rising=False, last=True) for q, lv in levels.items()}
        return rise, fall

    # shape: levels relative to the signal's own peak
    own_rise, own_fall = crossings(c, hp)
    rise_ref, fall_ref = crossings(c_ref, hp_ref)
    m['Peak width err.'] = _rel(own_fall[0.5] - own_rise[0.5], fall_ref[0.5] - rise_ref[0.5])

    # position: levels relative to the peak of the reference
    rise, fall = crossings(c, hp_ref)

    # elution signal
    m['Elution start err.'] = rise[0.05] - rise_ref[0.05]
    m['Front position err.'] = rise[0.5] - rise_ref[0.5]
    m['Front width err.'] = (own_rise[0.95] - own_rise[0.05]) - (rise_ref[0.95] - rise_ref[0.05])
    m['Tail position err.'] = fall[0.5] - fall_ref[0.5]
    m['Elution end err.'] = fall[0.05] - fall_ref[0.05]

    # moments
    m['Mass err.'] = _rel(trapezoid(c, t), trapezoid(c_ref, t))
    _, mu1, mu2 = pulse_moments(t, c)
    _, mu1_ref, mu2_ref = pulse_moments(t, c_ref)
    m['Ret. time err.'] = mu1 - mu1_ref
    m['Variance err.'] = _rel(mu2, mu2_ref)

    return {name: float(m[name]) for name in METRIC_NAMES}


def signal_metric_columns(t, solutions, reference, **kwargs):
    """Metric columns for a refinement series.

    Parameters
    ----------
    t : array_like
        Time grid shared by all solutions and the reference.
    solutions : list of array_like
        One solution per refinement level, of shape (n_times,) or
        (n_times, n_comp).
    reference : array_like
        Reference solution of the same shape as each solution.

    Returns
    -------
    (list of str, np.ndarray)
        Column names and values of shape (n_levels, n_columns). For more than
        one component, the names carry the suffix ' (comp i)'.
    """
    reference = np.asarray(reference, dtype=float)
    reference = reference.reshape(reference.shape[0], -1)
    n_comp = reference.shape[1]

    header = []
    for comp in range(n_comp):
        suffix = f' (comp {comp})' if n_comp > 1 else ''
        header += [name + suffix for name in METRIC_NAMES]

    rows = []
    for solution in solutions:
        solution = np.asarray(solution, dtype=float)
        solution = solution.reshape(solution.shape[0], -1)
        if solution.shape != reference.shape:
            raise ValueError(
                f"solution of shape {solution.shape} does not match the reference of shape {reference.shape}")
        row = []
        for comp in range(n_comp):
            metrics = signal_metrics(t, solution[:, comp], reference[:, comp], **kwargs)
            row += [metrics[name] for name in METRIC_NAMES]
        rows.append(row)

    return header, np.array(rows, dtype=float)
