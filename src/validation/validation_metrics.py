# -*- coding: utf-8 -*-
"""Validation metrics shared by the CADET column-geometry validation studies.

Every case study under ``src/validation`` reports the same four numbers, so
that the results of all six figures go into one LaTeX table without any
per-case reinterpretation:

    Delta mu_1 [%]     relative error of the first moment,
                       mu_1 = int(t*c dt) / int(c dt)
    Delta mu_2 [%]     relative error of the second central moment,
                       mu_2 = int((t-mu_1)^2*c dt) / int(c dt).
                       Two fallbacks apply, see ``mu2_fallback`` below: where
                       the paper reports no mu_2 the peak-height error takes
                       its place, and where the dispersion coefficient was
                       calibrated against the paper's own mu_2 the entry is
                       left empty, since a fitted quantity is not a
                       validation result.
    NRMSE [%]          root mean square deviation from the digitized
                       chromatogram, normalised by max|c_ref| -- the peak
                       height for a pulse, the plateau level for a
                       breakthrough curve.
    Mass balance [%]   outlet integral against injected mass. This verifies
                       the solver rather than validating the model: it uses
                       no reference data, so it is reported apart from the
                       three comparison metrics above.

Reference source
----------------
Where the paper tabulates moments (Gritti et al. 2019, Tables 1-3), those
measured values are the reference for Delta mu_1 and Delta mu_2. Where it
does not (Gu 2015, Ch. 14, which prints no moment table), the moments of the
digitized chromatogram are used instead. Each metric records which of the two
it used in ``mu1_ref_source`` and ``mu2_ref_source``.

Moments of breakthrough curves
------------------------------
Several curves in Gu (2015) approach a nonzero plateau instead of returning
to baseline. For those, int(t*c dt)/int(c dt) does not converge; it grows
with the upper integration limit. Their moments are therefore taken of the
residence time distribution E(t) = dF/dt underlying the normalised front
F = c/c_plateau. Integrating by parts removes the derivative, so nothing has
to be differentiated numerically, which digitized data would not survive:

    int_a^b t   E dt = b*F(b)   - a*F(a)   -  int_a^b F dt
    int_a^b t^2 E dt = b^2*F(b) - a^2*F(a) - 2 int_a^b t*F dt

mu_1 is then the stoichiometric breakthrough time and mu_2 the variance of
the front, both by the same definition as in the pulse case and computed
from integrals of F alone. Where the front overshoots its plateau and comes
back down, as the competitive-Langmuir roll-up of component 1 in Gu
Fig. 14.3 does, E changes sign and mu_2 can come out negative. That is the
signed second moment of a signed distribution; the simulated and the
digitized curve are treated identically, so the relative difference the
table reports stays meaningful.

Windowing
---------
When the reference is the digitized curve, the simulated and the digitized
moments are evaluated over the same time window -- the overlap of the two
time axes, with interpolated end points -- because a truncated tail biases
mu_1 and mu_2. When the reference is a tabulated value, the simulated
moments are evaluated over the full simulated window, which is what a
measured moment represents.
"""

import json
import os

import numpy as np

PULSE = 'pulse'
FRONTAL = 'frontal'

#: ``mu2_mode`` values -- what the Delta mu_2 column holds for a curve.
MU2_FROM_PAPER = 'mu2_paper'          # mu_2 vs. the paper's tabulated mu_2
MU2_FROM_DIGITIZED = 'mu2_digitized'  # mu_2 vs. the digitized chromatogram
MU2_PEAK_HEIGHT = 'peak_height'       # fallback: peak-height error instead
MU2_CALIBRATED = 'calibrated'         # empty: the model was fitted to this mu_2


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def trapezoid(y, x):
    """Trapezoidal integration, tolerant of the numpy trapz/trapezoid rename."""
    fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return float(fn(y, x))


def clean_curve(t, c):
    """Drop NaN samples and sort by time."""
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    mask = ~(np.isnan(t) | np.isnan(c))
    t, c = t[mask], c[mask]
    order = np.argsort(t)
    return t[order], c[order]


def common_window(t_a, t_b):
    """Overlap of two time axes."""
    lo = max(float(np.min(t_a)), float(np.min(t_b)))
    hi = min(float(np.max(t_a)), float(np.max(t_b)))
    if not hi > lo:
        raise ValueError(f"time axes do not overlap: [{lo}, {hi}]")
    return lo, hi


def restrict(t, c, lo, hi):
    """Restrict a curve to [lo, hi], adding exactly interpolated end points.

    The end points matter: the frontal moments below evaluate F at both
    window edges, so they must sit exactly on lo and hi for the simulated
    and the digitized curve alike.
    """
    t, c = clean_curve(t, c)
    inside = (t > lo) & (t < hi)
    tt = np.concatenate(([lo], t[inside], [hi]))
    cc = np.concatenate(([np.interp(lo, t, c)], c[inside], [np.interp(hi, t, c)]))
    return tt, cc


def plateau_level(t, c, tail_frac=0.1):
    """Mean of c over the last ``tail_frac`` of the time window."""
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    lo, hi = float(t[0]), float(t[-1])
    tail = t >= hi - tail_frac * (hi - lo)
    return float(np.nanmean(c[tail]))


def pulse_moments(t, c):
    """(area, mu_1, mu_2) of a curve that returns to baseline.

    Negative values are clipped first: they come from digitization noise and
    from small undershoots near the baseline.
    """
    t = np.asarray(t, dtype=float)
    c = np.clip(np.asarray(c, dtype=float), 0.0, None)
    area = trapezoid(c, t)
    if area <= 0.0:
        return 0.0, np.nan, np.nan
    mu1 = trapezoid(t * c, t) / area
    mu2 = trapezoid((t - mu1) ** 2 * c, t) / area
    return area, mu1, mu2


def frontal_moments(t, c, plateau=None, tail_frac=0.1):
    """(weight, mu_1, mu_2) of the residence time distribution E = dF/dt
    underlying a frontal / breakthrough curve, via integration by parts --
    see the module docstring. ``plateau`` defaults to the curve's own tail
    mean, so that a gain error between the simulated and the digitized
    plateau does not leak into the front-position comparison.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    if plateau is None:
        plateau = plateau_level(t, c, tail_frac)
    if not plateau > 0.0:
        return 0.0, np.nan, np.nan
    F = c / plateau
    a, b = float(t[0]), float(t[-1])
    Fa, Fb = float(F[0]), float(F[-1])
    weight = Fb - Fa
    if abs(weight) < 1e-12:
        return weight, np.nan, np.nan
    m1_raw = b * Fb - a * Fa - trapezoid(F, t)
    m2_raw = b * b * Fb - a * a * Fa - 2.0 * trapezoid(t * F, t)
    mu1 = m1_raw / weight
    mu2 = m2_raw / weight - mu1 ** 2
    return weight, mu1, mu2


def curve_moments(t, c, kind, plateau=None, tail_frac=0.1):
    """Dispatch to :func:`pulse_moments` or :func:`frontal_moments`."""
    if kind == PULSE:
        return pulse_moments(t, c)
    if kind == FRONTAL:
        return frontal_moments(t, c, plateau=plateau, tail_frac=tail_frac)
    raise ValueError(f"unknown curve kind {kind!r} (expected {PULSE!r} or {FRONTAL!r})")


def lsq_amplitude_scale(c_sim_on_ref_grid, c_ref):
    """Least-squares scale factor mapping the simulation onto the reference.

    The Gritti chromatograms are recorded in arbitrary absorbance units, so
    one amplitude factor per column has to be fitted before the shapes can be
    compared. Already normalised data, such as Gu's C/C0 curves, passes
    ``amplitude=1.0`` and skips this.
    """
    denom = float(np.sum(c_sim_on_ref_grid ** 2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(c_sim_on_ref_grid * c_ref) / denom)


def _relerr(value, reference):
    if value is None or reference is None:
        return np.nan
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0.0:
        return np.nan
    return 100.0 * abs(value - reference) / abs(reference)


# ---------------------------------------------------------------------------
# The metric set
# ---------------------------------------------------------------------------
def standard_metrics(name, t_sim, c_sim, t_ref, c_ref, kind=PULSE,
                     mu1_ref=None, mu2_ref=None, ref_label='paper table',
                     mu2_calibrated=False, mu2_fallback=MU2_PEAK_HEIGHT,
                     amplitude='lsq', mass_in=None, mass_retained=0.0,
                     mass_extra_out=0.0, mass_label=None, mass_exact=True,
                     tail_frac=0.1):
    """Compute the four metrics for one simulated/reference curve pair.

    Parameters
    ----------
    name : str
        Curve label (column configuration or component name).
    t_sim, c_sim : array
        Simulated outlet in its native units, NOT amplitude-calibrated.
    t_ref, c_ref : array
        Digitized reference curve; NaNs are dropped.
    kind : {PULSE, FRONTAL}
        Whether the curve returns to baseline or approaches a plateau; see
        the module docstring for how the moments of a frontal curve are
        defined.
    mu1_ref, mu2_ref : float, optional
        The paper's own tabulated moments, in the time units of ``t_sim``.
        When given they are the reference; when ``None``, the moments of the
        digitized curve are used instead.
    ref_label : str
        How to name the tabulated source in the printout, e.g. ``'Table 1'``.
    mu2_calibrated : bool
        True when the model's dispersion coefficient was calibrated so as to
        reproduce ``mu2_ref``. Delta mu_2 is then left empty, since it
        measures a fit and not a prediction.
    mu2_fallback : {MU2_PEAK_HEIGHT, MU2_FROM_DIGITIZED}
        What the Delta mu_2 column holds when the paper tabulates no mu_2.
        The default is the peak-height error. Frontal curves pass
        MU2_FROM_DIGITIZED instead, because their "peak" is just the plateau
        and its error is degenerate, whereas the width of the front is
        digitized reliably.
    amplitude : {'lsq'} or float
        Amplitude calibration of the simulation against the reference.
    mass_in : float, optional
        Injected mass in the units of int(c_sim dt). ``None`` leaves the
        mass balance empty.
    mass_retained : float
        Mass still held on the column at the end of the run, in the same
        units -- nonzero only for frontal runs, where the column retains a
        saturated inventory that the outlet integral alone cannot see.
    mass_extra_out : float
        Further outlet integral to add before closing the balance, used for
        the reactive case study where the fed species leaves the column
        partly as a reaction product.
    mass_label : str, optional
        Description of what the balance closes, for the printout.
    mass_exact : bool
        True when every term of the balance is known exactly, so that the
        deviation is the solver's conservation error. Set it False where a
        physical residual is expected on top, as in the reactive case study,
        whose column still holds some bound protein at the end of the run, so
        that the table can flag the entry rather than present a physical
        remainder as a numerical defect.
    """
    t_sim = np.asarray(t_sim, dtype=float)
    c_sim = np.asarray(c_sim, dtype=float)
    t_ref, c_ref = clean_curve(t_ref, c_ref)

    m = {'name': name, 'kind': kind}

    lo, hi = common_window(t_sim, t_ref)
    t_sim_w, c_sim_w = restrict(t_sim, c_sim, lo, hi)
    t_ref_w, c_ref_w = restrict(t_ref, c_ref, lo, hi)
    m['window'] = (lo, hi)

    # --- amplitude fit, over the comparison window ------------------------
    c_sim_on_ref = np.interp(t_ref_w, t_sim, c_sim)
    if amplitude == 'lsq':
        scale = lsq_amplitude_scale(c_sim_on_ref, c_ref_w)
    else:
        scale = float(amplitude)
    m['amplitude_scale'] = scale

    # --- moments -----------------------------------------------------------
    # Against a tabulated reference, the moments are taken over the full
    # simulated window, which is what a measured moment represents. Against
    # the digitized curve, both sides use the shared window so that they see
    # the same tail truncation.
    _, mu1_sim_full, mu2_sim_full = curve_moments(t_sim, c_sim, kind, tail_frac=tail_frac)
    _, mu1_sim_win, mu2_sim_win = curve_moments(t_sim_w, c_sim_w, kind, tail_frac=tail_frac)
    _, mu1_ref_win, mu2_ref_win = curve_moments(t_ref_w, c_ref_w, kind, tail_frac=tail_frac)
    m['mu1_sim_full'] = mu1_sim_full
    m['mu2_sim_full'] = mu2_sim_full
    m['mu1_sim_window'] = mu1_sim_win
    m['mu2_sim_window'] = mu2_sim_win
    m['mu1_digitized'] = mu1_ref_win
    m['mu2_digitized'] = mu2_ref_win

    # 1) Delta mu_1
    if mu1_ref is not None:
        m['mu1_sim'] = mu1_sim_full
        m['mu1_ref'] = float(mu1_ref)
        m['mu1_ref_source'] = ref_label
    else:
        m['mu1_sim'] = mu1_sim_win
        m['mu1_ref'] = mu1_ref_win
        m['mu1_ref_source'] = 'digitized'
    m['delta_mu1_%'] = _relerr(m['mu1_sim'], m['mu1_ref'])

    # --- peak height (also the mu_2 fallback) ------------------------------
    i_peak_ref = int(np.argmax(c_ref_w))
    i_peak_sim = int(np.argmax(c_sim_w))
    m['peak_height_ref'] = float(c_ref_w[i_peak_ref])
    m['peak_height_sim'] = float(scale * c_sim_w[i_peak_sim])
    m['peak_time_ref'] = float(t_ref_w[i_peak_ref])
    m['peak_time_sim'] = float(t_sim_w[i_peak_sim])
    m['delta_peak_height_%'] = _relerr(m['peak_height_sim'], m['peak_height_ref'])
    m['delta_peak_time_%'] = _relerr(m['peak_time_sim'], m['peak_time_ref'])

    # 2) Delta mu_2, with the two fallbacks described above
    if mu2_calibrated:
        m['mu2_mode'] = MU2_CALIBRATED
        m['mu2_sim'] = mu2_sim_full
        m['mu2_ref'] = float(mu2_ref) if mu2_ref is not None else np.nan
        m['mu2_ref_source'] = ref_label
        m['delta_mu2_%'] = np.nan
    elif mu2_ref is not None:
        m['mu2_mode'] = MU2_FROM_PAPER
        m['mu2_sim'] = mu2_sim_full
        m['mu2_ref'] = float(mu2_ref)
        m['mu2_ref_source'] = ref_label
        m['delta_mu2_%'] = _relerr(m['mu2_sim'], m['mu2_ref'])
    elif mu2_fallback == MU2_FROM_DIGITIZED:
        m['mu2_mode'] = MU2_FROM_DIGITIZED
        m['mu2_sim'] = mu2_sim_win
        m['mu2_ref'] = mu2_ref_win
        m['mu2_ref_source'] = 'digitized'
        m['delta_mu2_%'] = _relerr(m['mu2_sim'], m['mu2_ref'])
    else:
        m['mu2_mode'] = MU2_PEAK_HEIGHT
        m['mu2_sim'] = m['peak_height_sim']
        m['mu2_ref'] = m['peak_height_ref']
        m['mu2_ref_source'] = 'digitized'
        m['delta_mu2_%'] = m['delta_peak_height_%']

    # 3) NRMSE against the digitized chromatogram
    ref_amplitude = float(np.nanmax(np.abs(c_ref_w)))
    m['ref_amplitude'] = ref_amplitude
    m['mse'] = float(np.nanmean((scale * c_sim_on_ref - c_ref_w) ** 2))
    m['nrmse_%'] = 100.0 * np.sqrt(m['mse']) / ref_amplitude if ref_amplitude > 0 else np.nan

    # 4) Mass balance: solver verification, with no reference data involved
    mass_out = trapezoid(np.clip(c_sim, 0.0, None), t_sim) + float(mass_extra_out)
    m['mass_out'] = mass_out
    m['mass_retained'] = float(mass_retained)
    m['mass_in'] = float(mass_in) if mass_in is not None else np.nan
    if mass_in is None:
        m['mass_balance_%'] = np.nan
    else:
        m['mass_balance_%'] = 100.0 * abs(mass_out + float(mass_retained)
                                          - float(mass_in)) / abs(float(mass_in))
    m['mass_balance_label'] = mass_label or 'outlet integral vs. injected mass'
    m['mass_balance_exact'] = bool(mass_exact)

    return m


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_MU2_NOTE = {
    MU2_FROM_PAPER: 'second central moment vs. {src}',
    MU2_FROM_DIGITIZED: 'second central moment vs. the digitized curve',
    MU2_PEAK_HEIGHT: 'PEAK HEIGHT error -- the paper tabulates no mu_2 for this case',
    MU2_CALIBRATED: 'empty -- the dispersion coefficient was calibrated against '
                    'this mu_2 ({src}), so the agreement is fitted, not predicted',
}


def _fmt(value, spec='.4g', dash='--'):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return format(value, spec)


def format_metrics(m, time_unit='s'):
    """Render the metrics of one curve as a fixed text block."""
    note = _MU2_NOTE[m['mu2_mode']].format(src=m.get('mu2_ref_source', ''))
    lines = [f"\n--- {m['name']} ---"]
    lines.append(
        f"  Delta mu_1      [%] : {_fmt(m['delta_mu1_%'], '.4g'):<10s}"
        f"  (sim={_fmt(m['mu1_sim'])} {time_unit}, ref={_fmt(m['mu1_ref'])} {time_unit}"
        f" [{m['mu1_ref_source']}])")
    if m['mu2_mode'] == MU2_PEAK_HEIGHT:
        detail = f"(sim={_fmt(m['mu2_sim'])}, ref={_fmt(m['mu2_ref'])})"
    else:
        detail = f"(sim={_fmt(m['mu2_sim'])} {time_unit}^2, ref={_fmt(m['mu2_ref'])} {time_unit}^2)"
    lines.append(f"  Delta mu_2      [%] : {_fmt(m['delta_mu2_%'], '.4g'):<10s}  {detail}")
    lines.append(f"                        {note}")
    lines.append(
        f"  NRMSE           [%] : {_fmt(m['nrmse_%'], '.4g'):<10s}"
        f"  (vs. digitized chromatogram, normalised by reference"
        f" amplitude {_fmt(m['ref_amplitude'])})")
    lines.append(
        f"  Mass balance    [%] : {_fmt(m['mass_balance_%'], '.4g'):<10s}"
        f"  (in={_fmt(m['mass_in'])}, out={_fmt(m['mass_out'])}"
        + (f", retained={_fmt(m['mass_retained'])}" if m['mass_retained'] else '')
        + ")")
    lines.append(f"                        solver verification: {m['mass_balance_label']}")
    return '\n'.join(lines)


def print_metrics(m, time_unit='s'):
    print(format_metrics(m, time_unit=time_unit))


def print_metrics_table(metrics, time_unit='s'):
    """Print the block for every curve of a case study, then a compact summary."""
    for m in metrics:
        print_metrics(m, time_unit=time_unit)
    print(f"\n  {'curve':<22s}{'D.mu_1 [%]':>12s}{'D.mu_2 [%]':>22s}"
          f"{'NRMSE [%]':>12s}{'Mass bal. [%]':>15s}")
    for m in metrics:
        flag = {MU2_PEAK_HEIGHT: ' (peak h.)', MU2_CALIBRATED: ' (calib.)'}.get(m['mu2_mode'], '')
        print(f"  {m['name']:<22s}{_fmt(m['delta_mu1_%'], '.4g'):>12s}"
              f"{_fmt(m['delta_mu2_%'], '.4g') + flag:>22s}"
              f"{_fmt(m['nrmse_%'], '.4g'):>12s}{_fmt(m['mass_balance_%'], '.4g'):>15s}")


# ---------------------------------------------------------------------------
# Machine-readable output, consumed by the LaTeX table generator
# ---------------------------------------------------------------------------
def dump_metrics(output_path, case_id, case_label, metrics, time_unit='s'):
    """Write one JSON file per case study next to the plots.

    ``scripts/validation_metrics_table.py`` reads these back and assembles
    the LaTeX table, so that the table never has to be transcribed by hand.
    """
    payload = {
        'case_id': case_id,
        'case_label': case_label,
        'time_unit': time_unit,
        'curves': [],
    }
    for m in metrics:
        payload['curves'].append({
            key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in m.items() if key != 'window'
        })
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, f'{case_id}_metrics.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote validation metrics to {path}")
    return path
