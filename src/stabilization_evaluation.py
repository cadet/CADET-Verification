# -*- coding: utf-8 -*-
"""Evaluation of the stabilized (limited) DGSEM bulk discretization.

Studies, selected on the command line (default: all):

    comparison   element-wise blending vs. nodal flux limiting on the pulse
                 injection (pure convection and low dispersion)
    stepcount    time step counts over the tolerance for smooth and non-smooth
                 limiter choices on the pulse injection
    langmuir     performance benchmark of WENO FV, unlimited DGSEM and limited
                 DGSEM on the two-component Langmuir LRM with D_ax = 1e-7, at
                 the efficient and at the reference solver setting
    order        order verification on the bulk domain for a smooth solution
    limiter      amount of limiting on the Langmuir case (plots and videos)
    parameters   sensitivity of the results to the limiter parameters
    sensitivity  smoke test of parameter sensitivities with the limiter
    grm          held-out robustness case: GRM with SMA binding

    python -m src.stabilization_evaluation [study ...] [--rerun] [--small]

Every study writes its simulations, a json with the tables and its figures to
output/test_cadet-core/stabilization/<study>. The outlet tables are built by
convergence.recalculate_results with the qualitative signal metrics of
src/utility/signal_metrics.py and are stored in the layout of the convergence
jsons (convergence/<method>/outlet/<column>), extended by solver statistics
and by the amount of limiting. Finished simulations are reused unless --rerun
is given.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

repository_root = Path(__file__).resolve().parents[1]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from cadet import Cadet

import src.utility.convergence as convergence
import src.bench_func as bench_func
from src.benchmark_models import setting_Col1D_lin_1comp_benchmark1 as lin_setting
from src.benchmark_models import setting_Col1D_langLRM_2comp_benchmark1 as lang_setting

CADET_PATH = Path(os.environ.get(
    'CADET_PATH', r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"))
OUTPUT_PATH = repository_root / "output" / "test_cadet-core" / "stabilization"
LANGMUIR_REFERENCE = (repository_root / "data" / "CADET-Core_reference" / "chromatography"
                      / "axial_LRM_langmuir_2comp_benchmark1_DG_P5Z2048.h5")

# cadet-python falls back to PATH when a Cadet object is created without an
# install path, which convergence.py does when it reads simulations.
os.environ['PATH'] = str(CADET_PATH / 'bin') + os.pathsep + os.environ['PATH']

#: Amount of limiting above which a node counts as limited.
LIMITER_ACTIVE = 0.01


# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------

def solver_setting(tol, modified_newton, abstol=None, tag=None):
    """Time integrator setting with relative and algebraic tolerance tol."""
    abstol = tol if abstol is None else abstol
    if tag is None:
        tag = ('mn' if modified_newton else 'fn') + f'_tol{tol:.0e}'.replace('e-0', 'e-')
    return {
        'tag': tag,
        'time_integrator': {
            'ABSTOL': abstol, 'RELTOL': tol, 'ALGTOL': tol,
            'INIT_STEP_SIZE': 1e-8, 'MAX_STEPS': 2000000,
            'USE_MODIFIED_NEWTON': int(modified_newton),
        },
    }


#: Modified Newton at the tolerance beyond which no spatial error improves.
SOLVER_EFFICIENT = solver_setting(1e-6, True, tag='efficient')
#: Absolute tolerance relaxed to 1e-5 of the feed concentration of the Langmuir case.
SOLVER_MODERATE = solver_setting(1e-6, True, abstol=1e-4, tag='moderate')
#: Setting of the reference solution and of the earlier DG publication.
SOLVER_REFERENCE = solver_setting(1e-8, False, abstol=1e-10, tag='reference')


# ---------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------

def weno():
    return {'key': 'FVWENO3', 'label': 'WENO FV', 'poly_deg': 0, 'stabilization': None}


def dg(poly_deg):
    return {'key': f'DG_P{poly_deg}', 'label': f'DG P{poly_deg}',
            'poly_deg': poly_deg, 'stabilization': None}


def flux_limited(poly_deg, limiter, **options):
    """Nodal subcell flux limiting, limiter is POSITIVITY or HYBRID.

    options are fields of the stabilization group, e.g. SMOOTH_CLIP_EXPONENT,
    BOUND_RELAXATION, CLIP_IDENTITY_FRACTION, SENSOR_TYPE, SENSOR_THRESHOLD,
    SENSOR_SHIFT, POSITIVITY_SLACK, LIMITER_JACOBIAN.
    SMOOTH_CLIP_EXPONENT = 0 selects the sharp counterpart of every switch.
    """
    short = {'POSITIVITY': 'POS', 'HYBRID': 'HYB'}[limiter]
    names = {'SMOOTH_CLIP_EXPONENT': 'p', 'SENSOR_THRESHOLD': 'tau', 'SENSOR_SHIFT': 'sigma',
             'POSITIVITY_SLACK': 'eps', 'LIMITER_JACOBIAN': 'jac', 'BOUND_RELAXATION': 'kappa',
             'CLIP_IDENTITY_FRACTION': 'theta', 'SENSOR_TYPE': 'sensor', 'BOUND_SMOOTHING': 'beta'}
    fmt = lambda v: v.lower() if isinstance(v, str) else f'{v:g}'
    tag = ''.join(f'_{names.get(k, k.lower())}{fmt(v)}' for k, v in sorted(options.items()))
    sharp = options.get('SMOOTH_CLIP_EXPONENT', 4.0) == 0.0
    label = f'{short} P{poly_deg}' + (' sharp' if sharp else '')
    extra = {k: v for k, v in options.items() if not (sharp and k == 'SMOOTH_CLIP_EXPONENT')}
    if extra:
        label += ' (' + ', '.join(f'{names.get(k, k.lower())}={fmt(v)}' for k, v in sorted(extra.items())) + ')'
    stabilization = {'STABILIZATION_METHOD': 'SUBCELL_FLUX_LIMITING', 'LIMITER': limiter}
    stabilization.update(options)
    return {'key': f'DG_P{poly_deg}_lim{short}{tag}', 'label': label,
            'poly_deg': poly_deg, 'stabilization': stabilization}


def bounded(poly_deg, **options):
    """Subcell flux limiting with the local bounds enforced everywhere (no sensor), provably non-oscillatory.

    Needs the exact limiter Jacobian (the default of CADET-Core): the bounds
    are active on the dispersive part of the flux in the whole column, where
    the Jacobian of the unlimited scheme (LIMITER_JACOBIAN = 0) makes the
    Newton iteration restrict the step size to O(h^2 / D_ax).
    """
    named = flux_limited(poly_deg, 'HYBRID', **options)
    scheme = flux_limited(poly_deg, 'HYBRID', **{'SENSOR_TYPE': 'ALWAYS', **options})
    scheme['key'] = named['key'].replace('limHYB', 'limLED')
    scheme['label'] = named['label'].replace('HYB', 'LED')
    return scheme


def log_mean(scheme, floor=None):
    """Variant of a DG scheme (dg, flux_limited, bounded) with the logarithmic mean volume flux.

    Flux differencing with the entropy conservative two-point flux u * logmean(c_i, c_j)
    (VOLUME_FLUX = LOG_MEAN) instead of the standard DGSEM volume term. floor is
    LOG_MEAN_FLOOR, the floor of the arguments of the mean relative to the reference
    concentration of the case (CADET-Core default 1e-9).
    """
    variant = copy.deepcopy(scheme)
    disc = {'VOLUME_FLUX': 'LOG_MEAN'}
    tag = '_lm'
    if floor is not None:
        disc['LOG_MEAN_FLOOR'] = floor
        tag += f'{floor:g}'
    prefix = f"DG_P{scheme['poly_deg']}"
    variant['key'] = prefix + tag + scheme['key'][len(prefix):]
    variant['label'] = 'LM ' + scheme['label'] + (f' (floor={floor:g})' if floor is not None else '')
    variant['discretization'] = disc
    return variant


def elementwise(poly_deg, fv_order=1, shift=0.0, threshold=0.0, indicator='MODAL_ENERGY_LEGENDRE'):
    """Element-wise blending of DGSEM with subcell FV by a smoothness indicator.

    shift is the nodal shift of the modal energy indicator relative to the
    reference concentration of the case, threshold the hard blending threshold.
    """
    key = f'DG_P{poly_deg}_subFV{fv_order}'
    label = f'EW FV{fv_order} P{poly_deg}'
    if indicator == 'ALL_ELEMENTS':
        key += '_all'
        label = f'subcell FV{fv_order} P{poly_deg}'
    if shift:
        key += f'_shift{shift:g}'
        label += f' (shift={shift:g})'
    if threshold:
        key += f'_thr{threshold:g}'
        label += f' (threshold={threshold:g})'
    stabilization = {
        'STABILIZATION_METHOD': 'SUBCELL_FV_LIMITING',
        'SUBCELL_FV_ORDER': fv_order,
        'SUBCELL_FV_MAX_BLENDING': 1.0,
        'BLENDING_THRESHOLD': threshold,
        'SUBCELL_FV_LIMITER': 'MINMOD',
        'SUBCELL_FV_BOUNDARY_TREATMENT': 0,
        'smoothness_indicator': {
            'SMOOTHNESS_INDICATOR': indicator,
            'NODAL_COEFFICIENT_THRESHOLD': 0.0,
            'NODAL_COEFFICIENT_SHIFT': shift,  # scaled by the reference concentration in build_model
            'TIME_RELAXATION': 0.0,
            'NUM_MODES': 2,
        },
    }
    return {'key': key, 'label': label, 'poly_deg': poly_deg, 'stabilization': stabilization}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _get_ci(group, key):
    for k in group.keys():
        if k.lower() == key.lower():
            return group[k]
    return None


def _set_ci(group, key, value):
    """Set a field, replacing an existing one that differs only in case."""
    for k in list(group.keys()):
        if k.lower() == key.lower():
            del group[k]
    group[key] = value


def pulse_case(dispersion=0.0, t_end=50.0, pulse_end=10.0, n_times=5001):
    """Rectangular pulse injection without binding (DPFR).

    Without dispersion the outlet is the inlet pulse delayed by the residence
    time, which serves as exact reference. With dispersion the reference is a
    refined high order DG solution.
    """
    def model(poly_deg):
        return lin_setting.get_model(
            spatial_method_bulk=poly_deg, particle_type=None,
            col_dispersion=dispersion, pulse_end=pulse_end, t_end=t_end)

    cfg = model(3)
    column = cfg.input.model.unit_001
    flow_rate = float(cfg.input.model.connections.switch_000.connections[4])
    velocity = flow_rate / (float(_get_ci(column, 'cross_section_area')) * float(_get_ci(column, 'col_porosity')))
    residence_time = float(_get_ci(column, 'bed_length')) / velocity
    times = np.linspace(0.0, t_end, n_times)
    if dispersion == 0.0:
        name = 'DPFR_stepPulse_1comp_noDispersion'
        reference = ((times > residence_time) & (times <= residence_time + pulse_end)).astype(float)
    else:
        name = f'DPFR_stepPulse_1comp_Dax{dispersion:g}'
        reference = None
    return {'name': name, 'model': model, 'times': times, 'reference': reference, 'c_ref': 1.0,
            'ncomp': 1, 'nbound': 0, 'fine_reference': (7, 256)}


def smooth_case(t_end=80.0, n_times=801):
    """Dispersive pulse with a C1 inlet profile (cubic rise and fall), no binding."""
    def model(poly_deg):
        cfg = lin_setting.get_model(
            spatial_method_bulk=poly_deg, particle_type=None, pulse_end=10.0, t_end=t_end)
        sections = cfg.input.solver.sections
        sections.nsec = 3
        sections.section_continuity = [0, 0]
        sections.section_times = [0.0, 5.0, 10.0, t_end]
        inlet = cfg.input.model.unit_000
        # c(t) = 3 (t/5)^2 - 2 (t/5)^3 on [0, 5], mirrored on [5, 10]
        for i, (c0, c2, c3) in enumerate([(0.0, 3 / 25, -2 / 125), (1.0, -3 / 25, 2 / 125), (0.0, 0.0, 0.0)]):
            sec = inlet[f'sec_{i:03d}']
            sec.const_coeff = [c0]
            sec.lin_coeff = [0.0]
            sec.quad_coeff = [c2]
            sec.cube_coeff = [c3]
        return cfg

    return {'name': 'DPFR_smoothPulse_1comp', 'model': model, 'times': np.linspace(0.0, t_end, n_times),
            'reference': None, 'c_ref': 1.0, 'ncomp': 1, 'nbound': 0, 'fine_reference': (7, 128)}


def langmuir_case():
    """Two-component Langmuir LRM with D_ax = 1e-7 (self-sharpening fronts)."""
    def model(poly_deg):
        return lang_setting.get_model(poly_deg, use_collocation_dg=1)

    ref = convergence.get_simulation(str(LANGMUIR_REFERENCE)).root
    times = np.squeeze(ref.output.solution.solution_times)
    reference = np.squeeze(ref.output.solution.unit_001.solution_outlet)
    return {'name': 'LRM_langmuir_2comp_benchmark1', 'model': model, 'times': times, 'reference': reference,
            'c_ref': 10.0, 'ncomp': 2, 'nbound': 2, 'fine_reference': None}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def n_cells(scheme, dof):
    return dof if scheme['poly_deg'] == 0 else dof // (scheme['poly_deg'] + 1)


def build_model(case, scheme, dof, solver, write_bulk=False, write_limiter=False):
    cfg = case['model'](scheme['poly_deg'])
    disc = cfg.input.model.unit_001.discretization
    _set_ci(disc, 'NCOL' if scheme['poly_deg'] == 0 else 'NELEM', n_cells(scheme, dof))

    # further fields of the discretization group, e.g. the volume flux (see log_mean)
    for key, value in scheme.get('discretization', {}).items():
        _set_ci(disc, key, value)
    if 'VOLUME_FLUX' in scheme.get('discretization', {}):
        _set_ci(disc, 'REFERENCE_CONCENTRATION', case['c_ref'])

    if scheme['stabilization'] is not None:
        stabilization = copy.deepcopy(scheme['stabilization'])
        if stabilization['STABILIZATION_METHOD'] == 'SUBCELL_FLUX_LIMITING':
            stabilization.setdefault('REFERENCE_CONCENTRATION', case['c_ref'])
        else:
            stabilization['smoothness_indicator']['NODAL_COEFFICIENT_SHIFT'] *= case['c_ref']
        _set_ci(disc, 'stabilization', stabilization)

    _set_ci(cfg.input.solver, 'time_integrator', copy.deepcopy(solver['time_integrator']))
    _set_ci(cfg.input.solver, 'user_solution_times', case['times'])

    ret = cfg.input['return']
    _set_ci(ret, 'write_idas_meta', 1)
    _set_ci(ret.unit_001, 'write_solution_outlet', 1)
    _set_ci(ret.unit_001, 'write_solution_bulk', int(write_bulk))
    _set_ci(ret.unit_001, 'write_coordinates', int(write_bulk))
    _set_ci(ret.unit_001, 'write_solution_smoothness_indicator', int(write_limiter))
    return cfg


def _accumulate(counter):
    """IDAS counters restart with every section, make them cumulative."""
    counter = np.atleast_1d(np.asarray(counter, dtype=float)).copy()
    for i in np.nonzero(np.diff(counter) < 0)[0]:
        counter[i + 1:] += counter[i]
    return counter


def _total_steps(idas):
    """Time steps over all sections. idas_ntimesteps only counts the last section."""
    if 'idas_kumulative_ntimesteps' in idas:
        return int(_accumulate(np.squeeze(idas.idas_kumulative_ntimesteps))[-1])
    return int(np.squeeze(idas.idas_ntimesteps))


def _statistics(sim, scheme, times):
    """Solver statistics and amount of limiting of a finished simulation."""
    idas = sim.root.meta.idas
    stats = {'steps': _total_steps(idas)}
    for key, name in (('idas_n_residual_calls', 'residual_calls'),
                      ('idas_n_linsolver_setup_calls', 'linear_setups'),
                      ('idas_n_local_error_test_failures', 'error_test_failures')):
        if key in idas:
            stats[name] = float(_accumulate(np.squeeze(idas[key]))[-1])
    if 'idas_bdf_order' in idas:
        stats['mean_bdf_order'] = float(np.mean(np.squeeze(idas.idas_bdf_order)[1:]))

    solution = sim.root.output.solution.unit_001
    if 'solution_smoothness_indicator' in solution:
        amount = np.asarray(solution.solution_smoothness_indicator)
        amount = amount.reshape(amount.shape[0], amount.shape[1], -1)
        stab = scheme['stabilization']
        if stab is not None and stab['STABILIZATION_METHOD'] == 'SUBCELL_FV_LIMITING':
            # element-wise blending weights fill the first NELEM entries only
            amount = amount[:, :amount.shape[1] // (scheme['poly_deg'] + 1), :]
        stats['limiting_mean'] = float(np.mean(amount))
        stats['limiting_active'] = float(np.mean(amount > LIMITER_ACTIVE))
    return stats


def simulate(case, scheme, dof, solver, out_dir, rerun=False, repetitions=1,
             write_bulk=False, write_limiter=False, suffix=''):
    """Run one simulation (or reuse it) and return (file name, statistics).

    The statistics are kept in a json next to the simulation. 'time' is the
    median compute time over the repetitions. Returns (None, None) if the
    simulation fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case['name']}_{scheme['key']}_{solver['tag']}{suffix}_Z{n_cells(scheme, dof)}"
    file_name = out_dir / (stem + '.h5')
    stats_name = out_dir / (stem + '.stats.json')

    if not rerun and file_name.exists() and stats_name.exists():
        with open(stats_name) as fh:
            stats = json.load(fh)
        if (stats.get('ok') and len(stats.get('times', [])) >= repetitions
                and (not write_limiter or 'limiting_mean' in stats or scheme['stabilization'] is None)):
            return str(file_name), stats
        if stats.get('ok') is False:
            return None, None

    sim = Cadet(install_path=CADET_PATH)
    sim.root.input = copy.deepcopy(build_model(case, scheme, dof, solver, write_bulk, write_limiter).input)
    sim.filename = str(file_name)
    sim.save()

    times = []
    for _ in range(repetitions):
        result = sim.run_simulation()
        if result.return_code != 0:
            print(f"  FAILED {stem}: {(result.error_message or '')[-200:]}", flush=True)
            with open(stats_name, 'w') as fh:
                json.dump({'ok': False}, fh)
            return None, None
        times.append(float(np.squeeze(sim.root.meta.time_sim)))

    stats = {'ok': True, 'times': times, 'time': float(np.median(times)), 'dof': int(dof)}
    stats.update(_statistics(sim, scheme, case['times']))
    with open(stats_name, 'w') as fh:
        json.dump(stats, fh, indent=1)
    print(f"  {stem}: {stats['time']:.2f} s, {stats['steps']} steps", flush=True)
    return str(file_name), stats


def outlet(file_name):
    return np.squeeze(convergence.get_simulation(file_name).root.output.solution.unit_001.solution_outlet)


def reference_solution(case, out_dir, rerun=False):
    """Outlet reference of a case: given, or a refined high order DG solution."""
    if case['reference'] is not None:
        return case['reference']
    poly_deg, n_elem = case['fine_reference']
    file_name, _ = simulate(case, dg(poly_deg), n_elem * (poly_deg + 1),
                            solver_setting(1e-12, False, abstol=1e-13, tag='ref'), out_dir, rerun=rerun)
    case['reference'] = outlet(file_name)
    return case['reference']


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _plain(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def refinement_series(case, scheme, dofs, solver, out_dir, json_name, rerun=False, repetitions=1,
                      write_limiter=False, suffix=''):
    """Run a refinement series and write its outlet table to the json.

    Returns the table as dictionary of columns, or None if no simulation of
    the series succeeded.
    """
    reference = reference_solution(case, out_dir, rerun)
    files, stats, cells = [], [], []
    for dof in dofs:
        file_name, stat = simulate(case, scheme, dof, solver, out_dir, rerun=rerun,
                                   repetitions=repetitions, write_limiter=write_limiter, suffix=suffix)
        if file_name is None:
            continue
        files.append(file_name)
        stats.append(stat)
        cells.append(n_cells(scheme, dof))
    if not files:
        return None

    table = convergence.recalculate_results(
        file_path=str(out_dir) + '/', model=case['name'],
        ax_method=scheme['poly_deg'], ax_cells=cells,
        exact_name=reference, unit='001', which='outlet',
        incl_min_val=True, transport_model='LRM', ncomp=case['ncomp'], nbound=case['nbound'],
        simulation_names=files, save_names=scheme['key'],
        signal_metrics=True,
    )
    table = {column: [_plain(v) for v in values] for column, values in table.to_dict(orient='list').items()}

    # median compute time instead of the time of the last run, solver statistics, amount of limiting
    table['Sim. time'] = [s['time'] for s in stats]
    table['Bulk DoF'] = [s['dof'] for s in stats]
    table['Time steps'] = [s['steps'] for s in stats]
    for column, key in (('Mean BDF order', 'mean_bdf_order'), ('Error test failures', 'error_test_failures'),
                        ('Residual calls', 'residual_calls'), ('Linear setups', 'linear_setups'),
                        ('Limiting (mean)', 'limiting_mean'), ('Limiting (active fraction)', 'limiting_active')):
        if all(key in s for s in stats):
            table[column] = [s[key] for s in stats]
    table['label'] = scheme['label']

    bench_func.write_result_json(str(json_name), scheme['key'], table, sub_group='outlet', is_dict=True)
    return table


def write_meta(json_name, case, solver, schemes, note=''):
    bench_func.write_meta_json(str(json_name), {
        'name': Path(json_name).stem,
        'case': case['name'],
        'time_integrator': solver['time_integrator'] if solver is not None else None,
        'methods': {s['key']: {'label': s['label'], 'stabilization': s['stabilization']} for s in schemes},
        'note': note,
    })


def load_tables(json_name, sub_group='outlet'):
    with open(json_name) as fh:
        data = json.load(fh)
    return {key: group[sub_group] for key, group in data['convergence'].items() if sub_group in group}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'legend.fontsize': 8,
                         'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    return plt


#: Colour by scheme family, line style by variant, marker by polynomial degree.
FAMILY_COLOR = {'FV': '#444444', 'DG': '#1f77b4', 'POS': '#2ca02c', 'HYB': '#d62728', 'EW': '#9467bd'}
DEGREE_MARKER = {0: 's', 1: 'v', 2: '^', 3: 'o', 4: 'D', 5: 'P', 6: 'X', 7: '*'}


def scheme_style(scheme, variant=0):
    key = scheme['key']
    if key.startswith('FV'):
        family = 'FV'
    elif '_limPOS' in key:
        family = 'POS'
    elif '_limHYB' in key:
        family = 'HYB'
    elif '_subFV' in key:
        family = 'EW'
    else:
        family = 'DG'
    return {'color': FAMILY_COLOR[family], 'marker': DEGREE_MARKER.get(scheme['poly_deg'], 'o'),
            'linestyle': ['-', '--', ':', '-.'][variant % 4], 'markersize': 4.5, 'linewidth': 1.2}


def _styles(schemes):
    """One style per scheme; schemes of one family and degree differ by line style."""
    seen, styles = {}, {}
    for s in schemes:
        base = scheme_style(s)
        ident = (base['color'], base['marker'])
        styles[s['key']] = scheme_style(s, seen.get(ident, 0))
        seen[ident] = seen.get(ident, 0) + 1
    return styles


def plot_columns(tables, schemes, x_key, y_key, file_name, x_label=None, y_label=None, title=None,
                 magnitude=False, floor=None, logx=True, logy=True):
    """One line per scheme of column y_key over column x_key."""
    plt = _pyplot()
    styles = _styles(schemes)
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    plotted = 0
    for s in schemes:
        table = tables.get(s['key'])
        if table is None or y_key not in table or x_key not in table:
            continue
        x = np.asarray(table[x_key], dtype=float)
        y = np.asarray(table[y_key], dtype=float)
        if magnitude:
            y = np.abs(y)
        if floor is not None:
            y = np.maximum(y, floor)
        keep = np.isfinite(x) & np.isfinite(y)
        if logy:
            keep &= y > 0.0
        if logx:
            keep &= x > 0.0
        if not np.any(keep):
            continue
        ax.plot(x[keep], y[keep], label=s['label'], **styles[s['key']])
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(x_label or x_key)
    ax.set_ylabel(y_label or y_key)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.savefig(file_name)
    plt.close(fig)


def plot_profiles(case, schemes, files, file_name, xlim=None, comp=0, title=None):
    """Outlet signals of several schemes at one resolution against the reference."""
    plt = _pyplot()
    styles = _styles(schemes)
    reference = np.asarray(case['reference']).reshape(len(case['times']), -1)[:, comp]
    windows = xlim if xlim is not None else [None]
    fig, axes = plt.subplots(1, len(windows), figsize=(4.6 * len(windows), 3.5), squeeze=False)
    for ax, window in zip(axes[0], windows):
        ax.plot(case['times'], reference, color='black', linewidth=0.8, label='reference')
        for s in schemes:
            if files.get(s['key']) is None:
                continue
            c = outlet(files[s['key']]).reshape(len(case['times']), -1)[:, comp]
            style = dict(styles[s['key']], marker=None)
            ax.plot(case['times'], c, label=s['label'], **style)
        if window is not None:
            ax.set_xlim(*window)
        ax.set_xlabel('time / s')
        ax.set_ylabel('outlet concentration')
    axes[0][-1].legend(loc='upper right')
    if title:
        fig.suptitle(title)
    fig.savefig(file_name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

def study_comparison(rerun=False, small=False):
    """Element-wise blending against nodal flux limiting on the pulse injection."""
    out_dir = OUTPUT_PATH / 'comparison'
    dofs = [64, 128, 256] if small else [64, 128, 256, 512, 1024]
    solver = SOLVER_EFFICIENT
    profile_dof = 256

    for case in (pulse_case(0.0), pulse_case(5.75e-8)):
        json_name = out_dir / f"convergence_{case['name']}.json"
        schemes = [weno()]
        for p in (3, 5):
            schemes += [dg(p),
                        elementwise(p, 1, shift=0.1), elementwise(p, 2, shift=0.1),
                        flux_limited(p, 'POSITIVITY'), flux_limited(p, 'HYBRID'), bounded(p)]
        print(f"comparison: {case['name']}", flush=True)
        tables = {}
        for s in schemes:
            table = refinement_series(case, s, dofs, solver, out_dir, json_name, rerun, write_limiter=True)
            if table is not None:
                tables[s['key']] = table
        write_meta(json_name, case, solver, schemes,
                   note='Element-wise blending (EW) uses the modal energy indicator with a nodal shift of 0.1 '
                        'relative to the feed concentration.')

        for p in (3, 5):
            sel = [s for s in schemes if s['poly_deg'] in (0, p)]
            tag = f"{case['name']}_P{p}"
            for y_key, y_label, opts in (
                    ('$L^1$ error', r'$L^1$ error', {}),
                    ('Local overshoot', 'local overshoot / feed', {'floor': 1e-8}),
                    ('Local undershoot', 'local undershoot / feed', {'floor': 1e-8}),
                    ('Excess TV', 'excess total variation', {'magnitude': True, 'floor': 1e-8}),
                    ('Front width err.', 'relative front width error', {'magnitude': True, 'floor': 1e-8}),
                    ('Min. value', 'negative minimum', {'magnitude': True, 'floor': 1e-16}),
                    ('Time steps', 'time steps', {}),
                    ('Limiting (active fraction)', 'limited fraction of space-time', {'floor': 1e-6})):
                name = y_key.replace('$', '').replace('^', '').replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
                plot_columns(tables, sel, 'Bulk DoF', y_key, out_dir / f'{tag}_{name}.png',
                             x_label='axial DoF', y_label=y_label, **opts)
            files = {s['key']: simulate(case, s, profile_dof, solver, out_dir, write_limiter=True)[0] for s in sel}
            t0 = float(case['times'][np.argmax(np.asarray(case['reference']) > 0.5)])
            plot_profiles(case, sel, files, out_dir / f'{tag}_profiles_Z{profile_dof}.png',
                          xlim=[(t0 - 3.0, t0 + 3.0), (t0 + 7.0, t0 + 13.0)],
                          title=f'P{p}, {profile_dof} axial DoF')


def study_stepcount(rerun=False, small=False):
    """Time step counts over the tolerance for smooth and non-smooth limiters."""
    out_dir = OUTPUT_PATH / 'stepcount'
    case = pulse_case(0.0)
    reference = case['reference']
    dof, p = 256, 3
    tolerances = [1e-4, 1e-6, 1e-8] if small else [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    # (scheme, group, smooth?)
    variants = [
        (weno(), 'baseline', True),
        (dg(p), 'baseline', True),
        (elementwise(p, 1), 'element-wise', False),
        (elementwise(p, 1, threshold=0.01), 'element-wise', False),
        (elementwise(p, 1, shift=0.1), 'element-wise', True),
        (flux_limited(p, 'POSITIVITY', SMOOTH_CLIP_EXPONENT=0.0), 'positivity', False),
        (flux_limited(p, 'POSITIVITY'), 'positivity', True),
        (flux_limited(p, 'HYBRID', SMOOTH_CLIP_EXPONENT=0.0), 'hybrid', False),
        (flux_limited(p, 'HYBRID'), 'hybrid', True),
    ]
    variants[2][0]['label'] += ' (no regularisation)'

    dt = case['times'][1] - case['times'][0]
    results = {}
    print('stepcount', flush=True)
    for scheme, group, smooth in variants:
        rows = {'tolerance': [], 'Time steps': [], 'Mean BDF order': [], 'Error test failures': [],
                'Residual calls': [], 'Sim. time': [], '$L^1$ error': []}
        for tol in tolerances:
            file_name, stats = simulate(case, scheme, dof, solver_setting(tol, False), out_dir, rerun=rerun)
            if file_name is None:
                continue
            rows['tolerance'].append(tol)
            rows['Time steps'].append(stats['steps'])
            rows['Mean BDF order'].append(stats.get('mean_bdf_order'))
            rows['Error test failures'].append(stats.get('error_test_failures'))
            rows['Residual calls'].append(stats.get('residual_calls'))
            rows['Sim. time'].append(stats['time'])
            rows['$L^1$ error'].append(float(np.sum(np.abs(outlet(file_name) - reference)) * dt))
        rows.update({'label': scheme['label'], 'group': group, 'smooth': smooth})
        results[scheme['key']] = rows

    json_name = out_dir / f"stepcount_{case['name']}.json"
    with open(json_name, 'w') as fh:
        json.dump({'meta': {'case': case['name'], 'axial_dof': dof, 'poly_deg': p,
                            'note': 'Full Newton, ABSTOL = RELTOL = ALGTOL = tolerance. smooth = False marks '
                                    'limiters with non-smooth switches.'},
                   'stepcount': results}, fh, indent=4)

    plt = _pyplot()
    group_color = {'baseline': '#444444', 'element-wise': FAMILY_COLOR['EW'],
                   'positivity': FAMILY_COLOR['POS'], 'hybrid': FAMILY_COLOR['HYB']}
    for y_key, y_label, logy in (('Time steps', 'time steps', True), ('Mean BDF order', 'mean BDF order', False),
                                 ('Error test failures', 'error test failures', True)):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        count = {}
        for scheme, group, smooth in variants:
            rows = results[scheme['key']]
            y = [np.nan if v is None else v for v in rows[y_key]]
            if logy:
                y = np.maximum(np.asarray(y, dtype=float), 1.0)
            n = count.get((group, smooth), 0)
            count[(group, smooth)] = n + 1
            ax.plot(rows['tolerance'], y, label=scheme['label'], color=group_color[group],
                    linestyle='-' if smooth else '--', marker=['o', 's', '^'][n % 3] if smooth else ['x', '+', '1'][n % 3],
                    markersize=5, linewidth=1.2)
        ax.set_xscale('log')
        ax.invert_xaxis()
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('tolerance')
        ax.set_ylabel(y_label)
        ax.set_title(f'pulse injection, P{p}, {dof} axial DoF (dashed: non-smooth switches)', fontsize=8)
        ax.legend(fontsize=6.5, ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.savefig(out_dir / f"stepcount_{y_key.replace(' ', '_')}.png")
        plt.close(fig)


def langmuir_schemes(with_bounded=True):
    schemes = [weno(), dg(3), dg(4), dg(5), flux_limited(5, 'POSITIVITY'), flux_limited(5, 'HYBRID')]
    return schemes + [bounded(5)] if with_bounded else schemes


def study_langmuir(rerun=False, small=False):
    """Performance benchmark on the Langmuir LRM at two solver settings."""
    out_dir = OUTPUT_PATH / 'langmuir'
    case = langmuir_case()
    dofs = [128, 256] if small else [128, 256, 512, 1024, 2048]
    repetitions = 1 if small else 3

    for solver in (SOLVER_EFFICIENT, SOLVER_MODERATE, SOLVER_REFERENCE):
        json_name = out_dir / f"convergence_{case['name']}_{solver['tag']}.json"
        # the fully bounded scheme is left out at the tight setting, where it takes too many steps
        schemes = langmuir_schemes(with_bounded=solver is not SOLVER_REFERENCE)
        print(f"langmuir: {solver['tag']}", flush=True)
        tables = {}
        for s in schemes:
            series = dofs + ([4096] if s['poly_deg'] == 0 and not small else [])
            # nearest multiples of the nodes per element
            nodes = s['poly_deg'] + 1
            series = [nodes * round(d / nodes) for d in series]
            table = refinement_series(case, s, series, solver, out_dir, json_name, rerun, repetitions=repetitions)
            if table is not None:
                tables[s['key']] = table
        write_meta(json_name, case, solver, schemes,
                   note='Collocation DGSEM. Sim. time is the median of the repetitions. Reference: '
                        + LANGMUIR_REFERENCE.name)

        tag = f"{case['name']}_{solver['tag']}"
        for y_key, name in (('Max. error', 'max'), ('$L^1$ error', 'L1')):
            plot_columns(tables, schemes, 'Sim. time', y_key, out_dir / f'{tag}_performance_{name}.png',
                         x_label='compute time / s', y_label=y_key.replace('.', '') if name == 'max' else r'$L^1$ error')
        plot_columns(tables, schemes, 'Bulk DoF', 'Min. value', out_dir / f'{tag}_negative_concentration.png',
                     x_label='axial DoF', y_label='negative minimum of the outlet concentration',
                     magnitude=True, floor=1e-16)
        for comp in range(case['ncomp']):
            for y_key in ('Peak height err.', 'Local overshoot', 'Excess TV', 'Front position err.'):
                column = f'{y_key} (comp {comp})'
                name = y_key.replace(' ', '_').replace('.', '')
                plot_columns(tables, schemes, 'Bulk DoF', column, out_dir / f'{tag}_{name}_comp{comp}.png',
                             x_label='axial DoF', y_label=column, magnitude=True, floor=1e-8)


def study_order(rerun=False, small=False):
    """Order verification on the bulk domain for a smooth solution."""
    out_dir = OUTPUT_PATH / 'order'
    case = smooth_case()
    time_point = 150  # t = 15 s: the pulse is inside the column
    solver = solver_setting(1e-10, False, abstol=1e-12, tag='tight')
    n_levels = 3 if small else 5

    ref_p, ref_elem = case['fine_reference']
    ref_file, _ = simulate(case, dg(ref_p), ref_elem * (ref_p + 1),
                           solver_setting(1e-12, False, abstol=1e-13, tag='ref'), out_dir, rerun=rerun,
                           write_bulk=True)
    json_name = out_dir / f"convergence_{case['name']}.json"
    schemes = [weno()]
    for p in (3, 5):
        schemes += [dg(p), flux_limited(p, 'POSITIVITY'), flux_limited(p, 'HYBRID')]

    print('order', flush=True)
    tables = {}
    for s in schemes:
        base = 32 if s['poly_deg'] == 0 else (4 if s['poly_deg'] == 3 else 3)
        files, cells = [], []
        for level in range(n_levels):
            n = base * 2 ** level
            dof = n if s['poly_deg'] == 0 else n * (s['poly_deg'] + 1)
            file_name, _ = simulate(case, s, dof, solver, out_dir, rerun=rerun, write_bulk=True)
            if file_name is not None:
                files.append(file_name)
                cells.append(n)
        result = convergence.recalculate_results(
            file_path=str(out_dir) + '/', model=case['name'],
            ax_method=s['poly_deg'], ax_cells=cells, exact_name=Path(ref_file).name,
            unit='001', which='bulk', transport_model='LRM', ncomp=1, nbound=0,
            simulation_names=files, time_point=time_point,
        )
        for comp_idx, df in result.items():
            table = {column: [_plain(v) for v in values] for column, values in df.to_dict(orient='list').items()}
            table['label'] = s['label']
            bench_func.write_result_json(str(json_name), s['key'], table,
                                         sub_group=f'bulk_component_{comp_idx:03d}', is_dict=True)
            tables[s['key']] = table
        print(f"  {s['label']:10s} max. EOC: " + ' '.join(f'{v:5.2f}' for v in tables[s['key']]['Max. EOC'][1:]),
              flush=True)
    write_meta(json_name, case, solver, schemes,
               note=f"Bulk errors at solution time index {time_point} (t = {case['times'][time_point]:g} s) against "
                    f"DG P{ref_p} with {ref_elem} elements.")
    for y_key, name in (('Max. error', 'max'), ('$L^1$ error', 'L1')):
        plot_columns(tables, schemes, 'Bulk DoF', y_key, out_dir / f"{case['name']}_bulk_{name}.png",
                     x_label='axial DoF', y_label='bulk ' + y_key)


def _bulk(file_name):
    root = convergence.get_simulation(file_name).root
    unit = root.output.solution.unit_001
    c = np.asarray(unit.solution_bulk)
    c = c.reshape(c.shape[0], c.shape[1], -1)
    amount = None
    if 'solution_smoothness_indicator' in unit:
        amount = np.asarray(unit.solution_smoothness_indicator)
        amount = amount.reshape(amount.shape[0], amount.shape[1], -1)
    coords = np.squeeze(root.output.coordinates.unit_001.axial_coordinates)
    return coords, c, amount


def _nodal_amount(scheme, amount, n_points):
    """Amount of limiting per node; element-wise weights are repeated over the nodes of their element."""
    stab = scheme['stabilization']
    if amount is None or stab is None:
        return None
    if stab['STABILIZATION_METHOD'] == 'SUBCELL_FV_LIMITING':
        n_nodes = scheme['poly_deg'] + 1
        return np.repeat(amount[:, :n_points // n_nodes, :], n_nodes, axis=1)
    return amount


def study_limiter(rerun=False, small=False):
    """Amount of limiting on the Langmuir case: space-time maps, activity under refinement, videos."""
    out_dir = OUTPUT_PATH / 'limiter'
    case = langmuir_case()
    solver = SOLVER_EFFICIENT
    p, dof = 5, 258 if small else 510
    schemes = [weno(), dg(p), elementwise(p, 1, shift=0.1), flux_limited(p, 'POSITIVITY'), flux_limited(p, 'HYBRID')]
    limited = [s for s in schemes if s['stabilization'] is not None]
    plt = _pyplot()
    print('limiter', flush=True)

    # activity under refinement
    json_name = out_dir / f"convergence_{case['name']}_{solver['tag']}.json"
    dofs = [126, 258] if small else [126, 258, 510, 1020, 2040]
    tables = {}
    for s in limited:
        table = refinement_series(case, s, dofs, solver, out_dir, json_name, rerun, write_limiter=True, suffix='_lim')
        if table is not None:
            tables[s['key']] = table
    write_meta(json_name, case, solver, limited,
               note=f'Limiting (active fraction): share of space-time nodes with an amount of limiting above {LIMITER_ACTIVE}.')
    plot_columns(tables, limited, 'Bulk DoF', 'Limiting (active fraction)', out_dir / 'limiting_active_fraction.png',
                 x_label='axial DoF', y_label='limited fraction of space-time')
    plot_columns(tables, limited, 'Bulk DoF', 'Limiting (mean)', out_dir / 'limiting_mean.png',
                 x_label='axial DoF', y_label='mean amount of limiting')

    # space-time maps and videos at one resolution
    data = {}
    for s in schemes:
        file_name, _ = simulate(case, s, dof, solver, out_dir, rerun=rerun, write_bulk=True,
                                write_limiter=True, suffix='_bulk')
        if file_name is None:
            continue
        coords, c, amount = _bulk(file_name)
        data[s['key']] = (coords, c, _nodal_amount(s, amount, c.shape[1]))

    times = case['times']
    fig, axes = plt.subplots(len(limited), case['ncomp'], figsize=(4.2 * case['ncomp'], 2.6 * len(limited)),
                             squeeze=False, sharex=True, sharey=True)
    for row, s in enumerate(limited):
        if s['key'] not in data:
            continue
        coords, c, amount = data[s['key']]
        for comp in range(case['ncomp']):
            ax = axes[row][comp]
            mesh = ax.pcolormesh(coords, times, amount[:, :, comp], cmap='magma_r', vmin=0.0, vmax=1.0,
                                 shading='nearest', rasterized=True)
            ax.set_title(f"{s['label']}, component {comp + 1}", fontsize=8)
            ax.grid(False)
            if comp == 0:
                ax.set_ylabel('time / s')
            if row == len(limited) - 1:
                ax.set_xlabel('axial position / m')
    fig.colorbar(mesh, ax=axes, label='amount of limiting (0: DG, 1: low order)', shrink=0.8)
    fig.savefig(out_dir / f'limiting_spacetime_Z{dof}.png')
    plt.close(fig)

    from matplotlib.animation import FuncAnimation, PillowWriter
    comp_colors = ['#1f77b4', '#d62728']
    c_max = max(float(np.max(d[1])) for d in data.values())
    frames = range(0, len(times), 4 if small else 2)
    for s in schemes:
        if s['key'] not in data:
            continue
        coords, c, amount = data[s['key']]
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=110)
        ax.set_xlim(coords[0], coords[-1])
        ax.set_ylim(-0.12 * c_max, 1.08 * c_max)
        ax.set_xlabel('axial position / m')
        ax.set_ylabel('bulk concentration')
        ax.axhline(0.0, color='black', linewidth=0.5)
        lines = [ax.plot([], [], color=comp_colors[i], linewidth=1.2, label=f'component {i + 1}')[0]
                 for i in range(case['ncomp'])]
        ax2, fills = None, []
        if amount is not None:
            ax2 = ax.twinx()
            ax2.set_ylim(0.0, 3.0)  # limiter amount occupies the lower third
            ax2.set_yticks([0.0, 0.5, 1.0])
            ax2.set_ylabel('amount of limiting', loc='bottom')
            ax2.grid(False)
        ax.legend(loc='upper right')
        title = ax.set_title('')

        def update(frame, coords=coords, c=c, amount=amount, lines=lines, ax2=ax2, fills=fills, title=title, s=s):
            for i, line in enumerate(lines):
                line.set_data(coords, c[frame, :, i])
            if ax2 is not None:
                while fills:
                    fills.pop().remove()
                for i in range(c.shape[2]):
                    fills.append(ax2.fill_between(coords, 0.0, amount[frame, :, i], color=comp_colors[i],
                                                  alpha=0.45, linewidth=0.0, step='mid'))
            title.set_text(f"{s['label']}, {c.shape[1]} axial DoF, t = {times[frame]:5.1f} s")
            return lines

        animation = FuncAnimation(fig, update, frames=frames, blit=False)
        animation.save(out_dir / f"bulk_{s['key']}_Z{dof}.gif", writer=PillowWriter(fps=20))
        plt.close(fig)
        print(f"  video {s['label']}", flush=True)


def study_parameters(rerun=False, small=False):
    """Sensitivity of the results to the limiter parameters, varied one at a time around the defaults."""
    out_dir = OUTPUT_PATH / 'parameters'
    case = langmuir_case()
    solver = SOLVER_EFFICIENT
    p = 5
    dofs = [126, 258] if small else [258, 510, 1020]
    json_name = out_dir / f"convergence_{case['name']}_{solver['tag']}.json"

    schemes = [dg(p), flux_limited(p, 'POSITIVITY'), flux_limited(p, 'HYBRID')]
    for key, low, high in (('SMOOTH_CLIP_EXPONENT', 2.0, 8.0), ('POSITIVITY_SLACK', 1e-6, 1e-4),
                           ('BOUND_RELAXATION', 0.0, 3.0), ('CLIP_IDENTITY_FRACTION', 0.0, 0.8)):
        for limiter in ('POSITIVITY', 'HYBRID'):
            schemes += [flux_limited(p, limiter, **{key: low}), flux_limited(p, limiter, **{key: high})]
    for key, low, high in (('SENSOR_THRESHOLD', 0.1, 0.4), ('SENSOR_SHIFT', 0.05, 0.2)):
        schemes += [flux_limited(p, 'HYBRID', **{key: low}), flux_limited(p, 'HYBRID', **{key: high})]
    # plain bounds without relaxation and identity fraction, and the curvature magnitude sensor
    plain = {'BOUND_RELAXATION': 0.0, 'CLIP_IDENTITY_FRACTION': 0.0}
    schemes += [flux_limited(p, 'POSITIVITY', **plain), flux_limited(p, 'HYBRID', SENSOR_TYPE='CURVATURE'),
                flux_limited(p, 'HYBRID', SENSOR_TYPE='CURVATURE', **plain)]
    # local bounds everywhere, and the width of their regularised positive part
    schemes += [flux_limited(p, 'POSITIVITY', LIMITER_JACOBIAN=0), flux_limited(p, 'HYBRID', LIMITER_JACOBIAN=0),
                bounded(p), bounded(p, LIMITER_JACOBIAN=0), bounded(p, BOUND_SMOOTHING=1.0), bounded(p, BOUND_SMOOTHING=1000.0),
                flux_limited(p, 'HYBRID', SENSOR_TYPE='CURVATURE', BOUND_SMOOTHING=1000.0)]

    print('parameters', flush=True)
    rows = []
    for s in schemes:
        table = refinement_series(case, s, dofs, solver, out_dir, json_name, rerun)
        if table is None:
            continue
        for i, d in enumerate(table['Bulk DoF']):
            rows.append((s['label'], int(d), table['$L^1$ error'][i], table['Sim. time'][i], table['Time steps'][i],
                         table['Min. value'][i], max(table[f'Local overshoot (comp {c})'][i] for c in range(2))))
    write_meta(json_name, case, solver, schemes, note='Variations of the limiter parameters around their defaults.')
    with open(out_dir / 'parameters.txt', 'w') as fh:
        fh.write(f"{'scheme':52s} {'DoF':>5s} {'L1 error':>10s} {'time/s':>8s} {'steps':>7s} {'min':>10s} {'overshoot':>10s}\n")
        for row in sorted(rows, key=lambda r: (r[1], r[0])):
            fh.write(f"{row[0]:52s} {row[1]:5d} {row[2]:10.4g} {row[3]:8.2f} {row[4]:7.0f} {row[5]:10.2e} {row[6]:10.2e}\n")
    print(open(out_dir / 'parameters.txt').read(), flush=True)


def study_sensitivity(rerun=False, small=False):
    """Smoke test: parameter sensitivity (axial dispersion) with the limiter switched on.

    Compares the outlet sensitivity of the limited schemes with the one of
    WENO FV and unlimited DG on the Langmuir case.
    """
    out_dir = OUTPUT_PATH / 'sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)
    case = langmuir_case()
    solver = SOLVER_EFFICIENT
    p, dof = 3, 256
    results = {}
    print('sensitivity', flush=True)
    for s in (weno(), dg(p), flux_limited(p, 'POSITIVITY'), flux_limited(p, 'HYBRID')):
        cfg = build_model(case, s, dof if s['poly_deg'] else 1024, solver)
        sens = cfg.input.sensitivity
        sens.nsens = 1
        sens.sens_method = 'ad1'
        sens.param_000.sens_name = 'COL_DISPERSION'
        sens.param_000.sens_unit = 1
        for field in ('sens_comp', 'sens_partype', 'sens_reaction', 'sens_boundphase', 'sens_section'):
            sens.param_000[field] = -1
        sens.param_000.sens_abstol = 1e-6
        sens.param_000.sens_factor = 1.0
        _set_ci(cfg.input['return'].unit_001, 'write_sens_outlet', 1)
        sim = Cadet(install_path=CADET_PATH)
        sim.root.input = copy.deepcopy(cfg.input)
        sim.filename = str(out_dir / f"{case['name']}_{s['key']}_sens.h5")
        sim.save()
        result = sim.run_simulation()
        if result.return_code != 0:
            print(f"  {s['label']}: FAILED {(result.error_message or '')[-300:]}", flush=True)
            results[s['label']] = None
            continue
        results[s['label']] = np.squeeze(sim.root.output.sensitivity.param_000.unit_001.sens_outlet)
        print(f"  {s['label']}: ran, max |dc/dD_ax| = {np.max(np.abs(results[s['label']])):.4g}", flush=True)

    base = results.get('WENO FV')
    summary = {}
    if base is not None:
        dt = case['times'][1] - case['times'][0]
        scale = float(np.sum(np.abs(base)) * dt)
        for label, sens in results.items():
            summary[label] = None if sens is None else float(np.sum(np.abs(sens - base)) * dt / scale)
            print(f"  {label}: relative L1 difference to WENO FV (1024 cells) = {summary[label]}", flush=True)
    with open(out_dir / 'sensitivity.json', 'w') as fh:
        json.dump({'meta': {'case': case['name'], 'parameter': 'COL_DISPERSION',
                            'note': 'Relative L1 difference of the outlet sensitivity to WENO FV with 1024 cells.'},
                   'relative_L1_difference': summary}, fh, indent=4)


def study_grm(rerun=False, small=False):
    """Held-out robustness case: GRM with SMA binding, four components, load-wash-elute.

    No limiter parameter was tuned on this case. Bulk schemes at equal axial
    DoF with a DG and an FV particle discretization, against a refined DG
    reference (bulk P5 with 64 elements, particles P5 with 8 elements).
    """
    from src.benchmark_models import setting_Col1D_SMA_4comp_LWE_benchmark1 as sma_setting
    from src.utility.signal_metrics import signal_metric_columns

    out_dir = OUTPUT_PATH / 'grm'
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, 1500.0, 1501)
    c_ref = [100.0, 1.0, 1.0, 1.0]
    dof = 48 if small else 64

    def run(name, poly_deg, n, par_method, par_n, stabilization=None, tol=1e-6, par_limiter=False):
        file_name = out_dir / (name + '.h5')
        stats_name = out_dir / (name + '.stats.json')
        if not rerun and file_name.exists() and stats_name.exists():
            with open(stats_name) as fh:
                stats = json.load(fh)
            return (str(file_name), stats) if stats.get('ok') else (None, None)
        cfg = sma_setting.get_model(spatial_method_bulk=poly_deg, particle_type='GENERAL_RATE_PARTICLE',
                                    spatial_method_particle=par_method, parZ=par_n)
        disc = cfg.input.model.unit_000.discretization
        _set_ci(disc, 'NCOL' if poly_deg == 0 else 'NELEM', n)
        if stabilization is not None:
            _set_ci(disc, 'stabilization', dict(stabilization, REFERENCE_CONCENTRATION=c_ref))
        if par_limiter:
            par_disc = cfg.input.model.unit_000.particle_type_000.discretization
            _set_ci(par_disc, 'stabilization', {'STABILIZATION_METHOD': 'SUBCELL_FLUX_LIMITING',
                                                'REFERENCE_CONCENTRATION': c_ref})
        _set_ci(cfg.input.solver, 'time_integrator', {
            'ABSTOL': tol, 'RELTOL': tol, 'ALGTOL': tol, 'INIT_STEP_SIZE': 1e-10, 'MAX_STEPS': 1000000,
            'USE_MODIFIED_NEWTON': 1})
        _set_ci(cfg.input.solver, 'user_solution_times', times)
        _set_ci(cfg.input['return'].unit_000, 'write_solution_outlet', 1)
        _set_ci(cfg.input['return'], 'write_idas_meta', 1)
        sim = Cadet(install_path=CADET_PATH)
        sim.root.input = copy.deepcopy(cfg.input)
        sim.filename = str(file_name)
        sim.save()
        result = sim.run_simulation()
        if result.return_code != 0:
            print(f"  FAILED {name}: {(result.error_message or '')[-200:]}", flush=True)
            with open(stats_name, 'w') as fh:
                json.dump({'ok': False}, fh)
            return None, None
        stats = {'ok': True, 'time': float(np.squeeze(sim.root.meta.time_sim)),
                 'steps': _total_steps(sim.root.meta.idas)}
        with open(stats_name, 'w') as fh:
            json.dump(stats, fh)
        return str(file_name), stats

    def solution(file_name):
        return np.asarray(convergence.get_simulation(file_name).root.output.solution.unit_000.solution_outlet)

    print('grm', flush=True)
    ref_file, _ = run('GRM_SMA_4comp_reference_DG_P5Z64parP5parZ8', 5, 64, 5, 8, tol=1e-9)
    reference = solution(ref_file)

    results = {}
    schemes = [weno()]
    for poly_deg in (3, 5):
        schemes += [dg(poly_deg), flux_limited(poly_deg, 'POSITIVITY'), flux_limited(poly_deg, 'HYBRID')]
    for par_label, par_method, par_n, par_limiter in (('parDG_P3Z2', 3, 2, False), ('parDG_P3Z2_lim', 3, 2, True),
                                                      ('parFV_Z16', 0, 16, False)):
        for s in schemes:
            if par_limiter and (s['stabilization'] is None):
                continue  # the particle limiter is combined with the limited bulk schemes only
            n = dof if s['poly_deg'] == 0 else max(1, round(dof / (s['poly_deg'] + 1)))
            name = f"GRM_SMA_4comp_{s['key']}_Z{n}_{par_label}"
            file_name, stats = run(name, s['poly_deg'], n, par_method, par_n, s['stabilization'],
                                   par_limiter=par_limiter)
            entry = {'label': s['label'], 'particles': par_label, 'failed': file_name is None}
            if file_name is not None:
                c = solution(file_name)
                err = np.abs(c - reference)
                header, values = signal_metric_columns(times, [c[:, 1:]], reference[:, 1:])
                entry.update({
                    'Max. error (proteins)': float(err[:, 1:].max()),
                    '$L^1$ error (proteins)': float(err[:, 1:].sum(axis=0).max()),
                    'Min. value (proteins)': float(c[:, 1:].min()),
                    'Sim. time': stats['time'], 'Time steps': stats['steps'],
                    'Bulk DoF': n * (s['poly_deg'] + 1) if s['poly_deg'] else n,
                    'signal_metrics': dict(zip(header, [float(v) for v in values[0]])),
                })
                print(f"  {par_label:15s} {s['label']:8s} max={entry['Max. error (proteins)']:.2e} "
                      f"min={entry['Min. value (proteins)']:.1e} steps={stats['steps']} t={stats['time']:.1f}s", flush=True)
            results[name] = entry

    with open(out_dir / 'robustness_GRM_SMA_4comp.json', 'w') as fh:
        json.dump({'meta': {'case': 'GRM, SMA, 4 components, load-wash-elute',
                            'time_integrator': 'modified Newton, ABSTOL = RELTOL = ALGTOL = 1e-6',
                            'reference': 'DG P5 with 64 elements, particles DG P5 with 8 elements, tolerance 1e-9',
                            'note': 'Errors and metrics of the three protein components.'},
                   'robustness': results}, fh, indent=4)


STUDIES = {
    'comparison': study_comparison,
    'stepcount': study_stepcount,
    'langmuir': study_langmuir,
    'order': study_order,
    'limiter': study_limiter,
    'parameters': study_parameters,
    'sensitivity': study_sensitivity,
    'grm': study_grm,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('studies', nargs='*', default=list(STUDIES), choices=list(STUDIES) + [[]],
                        help='studies to run (default: all)')
    parser.add_argument('--rerun', action='store_true', help='rerun finished simulations')
    parser.add_argument('--small', action='store_true', help='reduced resolutions for a quick check')
    args = parser.parse_args(argv)
    for name in args.studies:
        STUDIES[name](rerun=args.rerun, small=args.small)


if __name__ == '__main__':
    main()
