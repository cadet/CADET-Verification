#!/usr/bin/env python3
"""
Benchmark: generalized vs specialized FV unit operations (issue #155).

Compares performance of the generalized ColumnModel1D FV path (unified
convection-dispersion operator, global sparse linear solver) against the
specialized FV units (LRM/LRMP/GRM with block-structured arrowhead solver).

The toggle is discretization.FV_ARROW_HEAD_OPTIMIZATION:
  true  (default) -> specialized FV units
  false           -> generalized ColumnModel1D FV path

Test cases:
  Study 1 — Convergence: LWE 4-comp SMA for LRMP and GRM, refining NCOL
            (and NPAR for GRM). Produces error-vs-nDOF and error-vs-time plots.
  Study 2 — Particle-type scaling: GRM with npartype in {1, 2, 4} at several
            fixed NCOL values, measuring compute time vs npartype.

Notes:
  - LRM (EQUILIBRIUM_PARTICLE) is excluded: the generalized ColumnModel1D FV
    path does not support EQUILIBRIUM_PARTICLE (createParticleModel returns
    null in ColumnModel1D.cpp:198). The LRM has no arrowhead structure anyway,
    so there is no performance difference to investigate.
  - At very coarse GRM discretizations (e.g. NCOL=8, NPAR=1), the two paths
    produce slightly different solutions because the arrowhead structure
    introduces film diffusion flux as additional DOFs. They converge to the
    same solution at finer discretizations.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
from cadet import Cadet
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.benchmark_models import setting_Col1D_SMA_4comp_LWE_benchmark1 as sma_setting
import src.utility.convergence as convergence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_arrow_head_optimization(model_dict, enabled):
    """Set FV_ARROW_HEAD_OPTIMIZATION on all column units (those with discretization)."""
    for key in model_dict['input']['model']:
        if key.startswith('unit_'):
            unit = model_dict['input']['model'][key]
            if 'discretization' in unit:
                unit['discretization']['FV_ARROW_HEAD_OPTIMIZATION'] = int(enabled)


def set_cadet_path(sim, cadet_path):
    """Set the CADET executable path on a simulation object."""
    if cadet_path is not None:
        if os.path.isfile(cadet_path):
            sim.cadet_path = cadet_path
        else:
            sim.install_path = cadet_path


def create_simulation(model_dict, filename, cadet_path=None):
    """Create a Cadet object from a model dictionary and save it."""
    sim = Cadet()
    sim.root.input = copy.deepcopy(model_dict['input'])
    sim.filename = filename
    set_cadet_path(sim, cadet_path)
    sim.save()
    return sim


def run_single_sim(sim, cadet_path=None):
    """Run a single CADET simulation."""
    set_cadet_path(sim, cadet_path)
    sim.save()
    data = sim.run_simulation()
    if data.return_code != 0:
        raise RuntimeError(
            f"Simulation failed for {sim.filename}\n"
            f"Error: {data.error_message}\nLOG: {data.log}"
        )
    return sim


def generate_disc_list(start, n_levels):
    """Generate a list of refinement levels by doubling."""
    return [start * (2 ** i) for i in range(n_levels)]


def run_and_collect_times(all_sims, sim_registry, cadet_path, n_runs, n_jobs):
    """Run all simulations n_runs times, keeping the fastest time per key."""
    best_times = {}
    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
        backend = Parallel(n_jobs=n_jobs, verbose=10)
        backend(
            delayed(run_single_sim)(sim, cadet_path) for sim in all_sims
        )
        for key, fname in sim_registry.items():
            t = float(convergence.get_compute_time(fname))
            if key not in best_times or t < best_times[key]:
                best_times[key] = t
    return best_times


# ---------------------------------------------------------------------------
# Study 1 — Convergence benchmark (LRMP, GRM)
# ---------------------------------------------------------------------------

STUDY1_MODEL_CONFIGS = [
    {
        'name': 'LRMP',
        'particle_type': 'HOMOGENEOUS_PARTICLE',
        'unit_id': '000',
        'par_method_needed': False,
        'base_ncol': 8,
        'base_npar': None,
        'idas_abstol': 1e-10,
    },
    {
        'name': 'GRM',
        'particle_type': 'GENERAL_RATE_PARTICLE',
        'unit_id': '000',
        'par_method_needed': True,
        'base_ncol': 8,
        'base_npar': 1,
        'idas_abstol': 1e-8,
    },
]

VARIANTS = [('specialized', True), ('generalized', False)]


def run_study1(output_path, cadet_path, n_refinements, n_runs, n_jobs):
    """Study 1: convergence benchmark for LRMP and GRM."""
    print("\n" + "=" * 60)
    print("STUDY 1 — Convergence: LRMP and GRM")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)

    all_sims = []
    sim_registry = {}

    for mc in STUDY1_MODEL_CONFIGS:
        ncol_list = generate_disc_list(mc['base_ncol'], n_refinements)
        for variant_name, arrow_head in VARIANTS:
            for disc_idx, ncol in enumerate(ncol_list):
                kwargs = {}
                npar = None
                if mc['par_method_needed']:
                    npar = mc['base_npar'] * (2 ** disc_idx)
                    kwargs['spatial_method_particle'] = 0
                    kwargs['parZ'] = npar

                model = sma_setting.get_model(
                    spatial_method_bulk=0,
                    particle_type=mc['particle_type'],
                    axRefinement=ncol // mc['base_ncol'],
                    idas_reftol=mc['idas_abstol'],
                    **kwargs
                )
                set_arrow_head_optimization(model, arrow_head)

                par_tag = f"_parZ{npar}" if npar is not None else ""
                filename = os.path.join(
                    output_path,
                    f"{mc['name']}_SMA_4comp_{variant_name}_FV_Z{ncol}{par_tag}.h5"
                )
                sim = create_simulation(model, filename, cadet_path)
                all_sims.append(sim)

                key = (mc['name'], variant_name, ncol, npar)
                sim_registry[key] = sim.filename

    print(f"Created {len(all_sims)} simulation configurations.")
    best_times = run_and_collect_times(
        all_sims, sim_registry, cadet_path, n_runs, n_jobs
    )

    print("\n--- Computing convergence tables ---")
    results = {}
    ncomp = 4

    for mc in STUDY1_MODEL_CONFIGS:
        ncol_list = generate_disc_list(mc['base_ncol'], n_refinements)
        ref_ncol = ncol_list[-1]
        compare_ncols = ncol_list[:-1]
        unit_str = 'unit_' + mc['unit_id']

        ref_npar = None
        if mc['par_method_needed']:
            ref_npar = mc['base_npar'] * (2 ** (n_refinements - 1))

        ref_key = (mc['name'], 'specialized', ref_ncol, ref_npar)
        ref_solution = convergence.get_solution(
            sim_registry[ref_key], unit=unit_str, which='outlet'
        )

        for variant_name, _ in VARIANTS:
            errors, dofs, times, ncols_used, npars_used = [], [], [], [], []

            for disc_idx, ncol in enumerate(compare_ncols):
                npar = None
                if mc['par_method_needed']:
                    npar = mc['base_npar'] * (2 ** disc_idx)

                key = (mc['name'], variant_name, ncol, npar)
                sol = convergence.get_solution(
                    sim_registry[key], unit=unit_str, which='outlet'
                )
                errors.append(float(np.max(np.abs(sol - ref_solution))))

                if mc['name'] == 'LRMP':
                    dofs.append(ncol * ncomp * 2)
                elif mc['name'] == 'GRM':
                    dofs.append(ncol * ncomp * (1 + npar))
                else:
                    dofs.append(ncol * ncomp)

                times.append(best_times[key])
                ncols_used.append(ncol)
                if npar is not None:
                    npars_used.append(npar)

            eoc = [float('nan')]
            for i in range(1, len(errors)):
                if errors[i] > 0 and errors[i - 1] > 0:
                    eoc.append(np.log(errors[i - 1] / errors[i]) / np.log(2))
                else:
                    eoc.append(float('nan'))

            result_key = f"{mc['name']}_{variant_name}"
            results[result_key] = {
                'NCOL': ncols_used, 'nDOF': dofs,
                'L_inf_error': errors, 'EOC': eoc, 'Sim. time': times,
            }
            if npars_used:
                results[result_key]['NPAR'] = npars_used

            print(f"\n{result_key}:")
            print(f"  NCOL:     {ncols_used}")
            if npars_used:
                print(f"  NPAR:     {npars_used}")
            print(f"  nDOF:     {dofs}")
            print(f"  Error:    {[f'{e:.3e}' for e in errors]}")
            print(f"  EOC:      {[f'{e:.2f}' if not np.isnan(e) else 'nan' for e in eoc]}")
            print(f"  Time [s]: {[f'{t:.4f}' for t in times]}")

    return results


# ---------------------------------------------------------------------------
# Study 2 — Particle-type scaling (GRM with npartype = 1, 2, 4)
# ---------------------------------------------------------------------------

def replicate_particle_type(model_dict, npartype):
    """Duplicate particle_type_000 to create npartype identical particle types."""
    col = model_dict['input']['model']['unit_000']
    col['npartype'] = npartype
    col['par_type_volfrac'] = [1.0 / npartype] * npartype

    base = copy.deepcopy(col['particle_type_000'])
    for pt in range(1, npartype):
        col[f'particle_type_{pt:03d}'] = copy.deepcopy(base)


def run_study2(output_path, cadet_path, n_runs, n_jobs,
               ncol_values=None, npar=4, npartype_values=None):
    """Study 2: timing vs npartype for GRM at fixed discretizations."""
    if ncol_values is None:
        ncol_values = [16, 32, 64]
    if npartype_values is None:
        npartype_values = [1, 2, 4]

    print("\n" + "=" * 60)
    print("STUDY 2 — Particle-type scaling: GRM")
    print(f"  NCOL values:    {ncol_values}")
    print(f"  NPAR:           {npar}")
    print(f"  npartype values: {npartype_values}")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)

    all_sims = []
    sim_registry = {}

    for ncol in ncol_values:
        for npt in npartype_values:
            for variant_name, arrow_head in VARIANTS:
                model = sma_setting.get_model(
                    spatial_method_bulk=0,
                    particle_type='GENERAL_RATE_PARTICLE',
                    axRefinement=ncol // 8,
                    spatial_method_particle=0,
                    parZ=npar,
                    idas_reftol=1e-8,
                )
                replicate_particle_type(model, npt)
                set_arrow_head_optimization(model, arrow_head)

                filename = os.path.join(
                    output_path,
                    f"GRM_SMA_4comp_{variant_name}_FV_Z{ncol}_parZ{npar}"
                    f"_npt{npt}.h5"
                )
                sim = create_simulation(model, filename, cadet_path)
                all_sims.append(sim)

                key = (ncol, npt, variant_name)
                sim_registry[key] = sim.filename

    print(f"Created {len(all_sims)} simulation configurations.")
    best_times = run_and_collect_times(
        all_sims, sim_registry, cadet_path, n_runs, n_jobs
    )

    print("\n--- Timing results ---")
    results = {}
    for ncol in ncol_values:
        for variant_name, _ in VARIANTS:
            result_key = f"GRM_Z{ncol}_{variant_name}"
            npt_list, time_list = [], []
            for npt in npartype_values:
                key = (ncol, npt, variant_name)
                npt_list.append(npt)
                time_list.append(best_times[key])
            results[result_key] = {
                'npartype': npt_list,
                'Sim. time': time_list,
            }
            print(f"\n{result_key}:")
            print(f"  npartype: {npt_list}")
            print(f"  Time [s]: {[f'{t:.4f}' for t in time_list]}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_study1(results, output_path):
    """Generate error-vs-nDOF and error-vs-time plots for Study 1."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    model_names = ['LRMP', 'GRM']
    colors = {'specialized': 'tab:blue', 'generalized': 'tab:orange'}
    markers = {'specialized': 'o', 'generalized': 's'}
    labels = {
        'specialized': 'Specialized (arrowhead solver)',
        'generalized': 'Generalized (sparse solver)',
    }

    for plot_type in ['nDOF', 'time']:
        fig, axes = plt.subplots(1, len(model_names),
                                 figsize=(7 * len(model_names), 5))
        if len(model_names) == 1:
            axes = [axes]

        for ax, model_name in zip(axes, model_names):
            for variant in ['specialized', 'generalized']:
                key = f"{model_name}_{variant}"
                if key not in results:
                    continue
                data = results[key]
                x = data['nDOF'] if plot_type == 'nDOF' else data['Sim. time']
                y = data['L_inf_error']
                ax.loglog(x, y,
                          marker=markers[variant], color=colors[variant],
                          label=labels[variant], linewidth=1.5, markersize=6)

            ax.set_title(f'{model_name} (SMA 4-comp LWE)')
            ax.set_xlabel('nDOF' if plot_type == 'nDOF' else 'Compute time [s]')
            ax.set_ylabel(r'$L^\infty$ error')
            ax.legend(fontsize=8)
            ax.grid(True, which='both', alpha=0.3)

        fig.suptitle(
            'Generalized vs Specialized FV: Error vs '
            + ('nDOF' if plot_type == 'nDOF' else 'Compute Time'),
            fontsize=14)
        fig.tight_layout()
        plot_file = os.path.join(output_path, f'gen_vs_spec_FV_{plot_type}.png')
        fig.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {plot_file}")
        plt.close(fig)


def plot_study2(results, output_path):
    """Generate timing-vs-npartype bar chart for Study 2."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    ncol_keys = sorted(set(
        k.replace('GRM_', '').replace('_specialized', '').replace('_generalized', '')
        for k in results
    ))

    fig, axes = plt.subplots(1, len(ncol_keys),
                             figsize=(5 * len(ncol_keys), 5))
    if len(ncol_keys) == 1:
        axes = [axes]

    colors = {'specialized': 'tab:blue', 'generalized': 'tab:orange'}
    bar_width = 0.35

    for ax, ncol_key in zip(axes, ncol_keys):
        for v_idx, variant in enumerate(['specialized', 'generalized']):
            rkey = f"GRM_{ncol_key}_{variant}"
            if rkey not in results:
                continue
            data = results[rkey]
            x = np.arange(len(data['npartype']))
            offset = (v_idx - 0.5) * bar_width
            ax.bar(x + offset, data['Sim. time'], bar_width,
                   label=variant.capitalize(), color=colors[variant])

        ax.set_title(f'GRM {ncol_key}')
        ax.set_xlabel('Number of particle types')
        ax.set_ylabel('Compute time [s]')
        ax.set_xticks(np.arange(len(data['npartype'])))
        ax.set_xticklabels(data['npartype'])
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('GRM: Compute Time vs Number of Particle Types', fontsize=14)
    fig.tight_layout()
    plot_file = os.path.join(output_path, 'gen_vs_spec_FV_npartype.png')
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_file}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark generalized vs specialized FV units (issue #155)'
    )
    parser.add_argument(
        '--output', '-o', default='output/generalized_vs_specialized_FV',
        help='Output directory for results')
    parser.add_argument(
        '--cadet', '-c', default=None,
        help='Path to cadet-cli (default: use conda installation)')
    parser.add_argument(
        '--refinements', '-r', type=int, default=6,
        help='Number of refinement levels for Study 1 (default: 6)')
    parser.add_argument(
        '--runs', '-n', type=int, default=3,
        help='Number of repeated runs for timing (default: 3)')
    parser.add_argument(
        '--jobs', '-j', type=int, default=-1,
        help='Number of parallel jobs (default: -1 = all cores)')
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate plots after running')
    parser.add_argument(
        '--study', type=int, nargs='+', default=[1, 2],
        help='Which studies to run: 1=convergence, 2=npartype scaling '
             '(default: both)')
    args = parser.parse_args()

    all_results = {}

    if 1 in args.study:
        results1 = run_study1(
            output_path=args.output, cadet_path=args.cadet,
            n_refinements=args.refinements, n_runs=args.runs, n_jobs=args.jobs)
        all_results['study1_convergence'] = results1
        if args.plot:
            plot_study1(results1, args.output)

    if 2 in args.study:
        results2 = run_study2(
            output_path=args.output, cadet_path=args.cadet,
            n_runs=args.runs, n_jobs=args.jobs)
        all_results['study2_npartype_scaling'] = results2
        if args.plot:
            plot_study2(results2, args.output)

    result_file = os.path.join(args.output, 'generalized_vs_specialized_FV.json')
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll results written to {result_file}")
