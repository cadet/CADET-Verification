# -*- coding: utf-8 -*-
"""
Study 1 (GRM SMA): CGL vs LGL node comparison — rGRM with SMA 4-comp.

Uses the rGRM SMA 4-comp setting. Reference = FV WENO3 with 2048 cells
(must already exist on disk). Each (node_type, polyDeg, nelem) config is
run n_repeats times and the minimum wall time is kept.
"""

import os
import copy
import traceback
import numpy as np
from functools import partial

import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp as setting_GRM_SMA
import src.bench_func as bench_func
import src.bench_configs as bench_configs
import src.utility.convergence as convergence

from cadet import Cadet


def study1_GRM_SMA_tests(n_jobs, small_test, output_path, cadet_path,
                         polydegs=None, node_types=None, n_repeats=3):
    """Run Study 1 GRM SMA: CGL vs LGL convergence with min-time reruns.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers for joblib (-1 = all cores).
    small_test : bool
        If True, use reduced discretization levels.
    output_path : str
        Directory for simulation h5 files and convergence JSONs.
    cadet_path : str
        Path to CADET install (parent of bin/cadet-cli).
    polydegs : list of int, optional
        Polynomial degrees to run. Default: [1..5] (full) or [1,2] (small).
    node_types : list of str, optional
        Node types. Default: ['CGL', 'LGL'].
    n_repeats : int
        Total number of runs per config. Minimum wall time is kept.
    """

    os.makedirs(output_path, exist_ok=True)

    time_integrator = {
        'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
        'USE_MODIFIED_NEWTON': False,
        'INIT_STEP_SIZE': 1e-6,
        'MAX_STEPS': 5000000
    }

    # ---- Refinement function ----

    def refine_DG(config_data, disc_idx, setting_name,
                  polyDeg, node_type='CGL',
                  nelem_start=1, time_integrator=None,
                  unit_id='001',
                  only_return_name=False, **kwargs):

        nElem = nelem_start * 2**disc_idx
        config_name = convergence.generate_1D_name(setting_name, polyDeg, nElem)

        if only_return_name:
            if output_path is not None:
                return str(output_path) + '/' + config_name
            return config_name

        config_data = copy.deepcopy(config_data)

        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator

        unit_cfg = config_data['input']['model']['unit_' + unit_id]

        disc = unit_cfg['discretization']
        disc['SPATIAL_METHOD'] = 'DG'
        disc['POLYDEG'] = polyDeg
        disc['NELEM'] = nElem

        unit_cfg['POLYNOMIAL_INTERPOLATION_NODES'] = node_type

        # Particle discretization for GRM DG
        pt = unit_cfg['particle_type_000']
        pt['discretization']['PAR_POLYDEG'] = 3
        pt['discretization']['PAR_NELEM'] = 1
        pt['discretization']['PAR_DISC_TYPE'] = 'EQUIDISTANT_PAR'

        model = Cadet(install_path=cadet_path)
        model.root.input = config_data['input']
        if output_path is not None:
            model.filename = str(output_path) + '/' + config_name
            model.save()
        return model

    # ---- Config ----

    poly_degs = polydegs if polydegs is not None else (list(range(1, 6)) if not small_test else [1, 2])
    node_types_list = node_types if node_types is not None else ['CGL', 'LGL']

    if not small_test:
        n_disc = {1: 10, 2: 9, 3: 7, 4: 6, 5: 5}
    else:
        n_disc = {p: 4 for p in range(1, 6)}

    base_model = setting_GRM_SMA.get_model()

    # FV reference = FV WENO3 with 2048 cells (must already exist)
    fv_ncol = 2048 if not small_test else 32
    fv_ref_file = convergence.generate_1D_name(
        'radGRM_FV_WENO3_SMA_4comp', 0, fv_ncol)

    # Verify FV reference exists
    fv_ref_path = os.path.join(output_path, fv_ref_file)
    if not os.path.exists(fv_ref_path):
        print(f"ERROR: FV reference not found: {fv_ref_path}")
        print("Provide the FV WENO3 reference file before running this study.")
        return

    # ---- Build benchmark configs ----

    cadet_configs = []
    cadet_config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    par_methods = []
    par_discs = []
    disc_refinement_functions = []

    try:
        for node_type in node_types_list:
            methods = []
            for p in poly_degs:
                methods.append((f'radGRM_DG_{node_type}_SMA_4comp_P{p}', p))
            addition = {
                'cadet_config_jsons': [base_model] * len(methods),
                'cadet_config_names': [name for name, _ in methods],
                'include_sens': [False] * len(methods),
                'ref_files': [[fv_ref_file] for _ in methods],
                'unit_IDs': ['001'] * len(methods),
                'which': ['outlet'] * len(methods),
                'idas_abstol': [[None] for _ in methods],
                'ax_methods': [[p] for _, p in methods],
                'ax_discs': [[bench_func.disc_list(1, n_disc[p])] for _, p in methods],
                'par_methods': [[None] for _ in methods],
                'par_discs': [[None] for _ in methods],
                'disc_refinement_functions': [
                    [partial(refine_DG,
                             setting_name=name,
                             polyDeg=polyDeg,
                             node_type=node_type,
                             nelem_start=1,
                             time_integrator=time_integrator)]
                    for name, polyDeg in methods
                ],
            }
            bench_configs.add_benchmark(
                cadet_configs, include_sens, ref_files, unit_IDs, which,
                ax_methods, ax_discs,
                par_methods=par_methods, par_discs=par_discs,
                idas_abstol=idas_abstol,
                cadet_config_names=cadet_config_names, addition=addition,
                disc_refinement_functions=disc_refinement_functions)

        # Initial run
        bench_func.run_convergence_analysis(
            output_path=output_path,
            cadet_path=cadet_path,
            cadet_configs=cadet_configs,
            cadet_config_names=cadet_config_names,
            include_sens=include_sens,
            ref_files=ref_files,
            unit_IDs=unit_IDs,
            which=which,
            ax_methods=ax_methods,
            ax_discs=ax_discs,
            par_methods=par_methods,
            par_discs=par_discs,
            idas_abstol=idas_abstol,
            n_jobs=n_jobs,
            rerun_sims=True,
            disc_refinement_functions=disc_refinement_functions
        )

        # Rerun to get minimum wall time
        if n_repeats > 1:
            print(f"\n  Rerunning {n_repeats - 1} more times for minimum wall time...")
            expected_prefixes = []
            for nt in node_types_list:
                for p in poly_degs:
                    expected_prefixes.append(f'radGRM_DG_{nt}_SMA_4comp_P{p}_DG_P{p}Z')
            study_files = [
                f for f in os.listdir(output_path)
                if f.endswith('.h5')
                and any(f.startswith(pfx) for pfx in expected_prefixes)
            ]
            model_rerun = Cadet()
            model_rerun.install_path = cadet_path
            for fname in sorted(study_files):
                try:
                    model_rerun.filename = os.path.join(output_path, fname)
                    model_rerun.load_from_file()
                    best_time = model_rerun.root.meta.time_sim
                    for _ in range(n_repeats - 1):
                        ret = model_rerun.run_load()
                        if ret.return_code == 0:
                            best_time = min(best_time, model_rerun.root.meta.time_sim)
                    model_rerun.load_from_file()
                    model_rerun.root.meta.time_sim = best_time
                    model_rerun.save()
                    print(f"    {fname}: best time = {best_time:.6f}s")
                except Exception as e:
                    print(f"    {fname}: SKIPPED (error: {e})")

            # Recompute convergence tables with best times
            bench_func.run_convergence_analysis(
                output_path=output_path,
                cadet_path=cadet_path,
                cadet_configs=cadet_configs,
                cadet_config_names=cadet_config_names,
                include_sens=include_sens,
                ref_files=ref_files,
                unit_IDs=unit_IDs,
                which=which,
                ax_methods=ax_methods,
                ax_discs=ax_discs,
                par_methods=par_methods,
                par_discs=par_discs,
                idas_abstol=idas_abstol,
                n_jobs=n_jobs,
                rerun_sims=False,
                disc_refinement_functions=disc_refinement_functions
            )

    except Exception:
        print(f"\n*** Study 1 GRM SMA FAILED ***\n{traceback.format_exc()}")
