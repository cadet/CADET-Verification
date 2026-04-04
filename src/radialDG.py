# -*- coding: utf-8 -*-
"""
EOC tests for radial DG discretization.

Studies:
  0. Pure bulk transport — DG self-convergence + DG vs FV WENO3
  1. CGL vs LGL node comparison — rLRM 1-comp Linear + 4-comp SMA
  2. DG vs FV with variable coefficients — 6 configurations (rGRM/rLRM/rLRMP x 1/4 comp)
  3. Langmuir 2-comp oscillation study — two dispersion levels
"""

import os
import copy
import numpy as np
from functools import partial

import src.benchmark_models.setting_radCol1D_DG_transport as setting_DG_transport
import src.benchmark_models.setting_radCol1D_DG_LRM_lin_1comp as setting_DG_LRM_lin
import src.benchmark_models.setting_radCol1D_DG_LRM_SMA_4comp as setting_DG_LRM_SMA
import src.benchmark_models.setting_radCol1D_DG_GRM_lin_1comp_varCoeff as setting_DG_GRM_lin_var
import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp_varCoeff as setting_DG_GRM_SMA_var
import src.benchmark_models.setting_radCol1D_DG_LRM_lin_1comp_varCoeff as setting_DG_LRM_lin_var
import src.benchmark_models.setting_radCol1D_DG_LRM_SMA_4comp_varCoeff as setting_DG_LRM_SMA_var
import src.benchmark_models.setting_radCol1D_DG_LRMP_lin_1comp_varCoeff as setting_DG_LRMP_lin_var
import src.benchmark_models.setting_radCol1D_DG_LRMP_SMA_4comp_varCoeff as setting_DG_LRMP_SMA_var
import src.benchmark_models.setting_radCol1D_DG_LRM_lang_2comp as setting_DG_LRM_lang

import src.bench_func as bench_func
import src.bench_configs as bench_configs
import src.utility.convergence as convergence

from cadet import Cadet


def radialDG_tests(n_jobs, small_test, output_path, cadet_path):

    os.makedirs(output_path, exist_ok=True)

    # ---- Shared time integrator settings ----

    time_integrator = {
        'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
        'USE_MODIFIED_NEWTON': False,
        'INIT_STEP_SIZE': 1e-6,
        'MAX_STEPS': 1000000
    }

    # ---- Helper: equivolume radial grid ----

    def grid_radial_equivolume(r0, r1, n):
        r2_faces = np.linspace(r0**2, r1**2, n + 1)
        return np.sqrt(r2_faces)

    # ---- Refinement functions ----

    def refine_DG(config_data, disc_idx, setting_name,
                  polyDeg, node_type='CGL',
                  nelem_start=2, time_integrator=None,
                  unit_id='001', **kwargs):
        """Refinement function for radial DG: doubles nElem at each step."""

        config_data = copy.deepcopy(config_data)

        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator

        nElem = nelem_start * 2**disc_idx

        disc = config_data['input']['model']['unit_' + unit_id]['discretization']
        disc['SPATIAL_METHOD'] = 'DG'
        disc['POLYDEG'] = polyDeg
        disc['NELEM'] = nElem

        config_data['input']['model']['unit_' + unit_id]['node_type'] = node_type

        config_name = convergence.generate_1D_name(setting_name, polyDeg, nElem)

        model = Cadet(install_path=cadet_path)
        model.root.input = config_data['input']
        if output_path is not None:
            model.filename = str(output_path) + '/' + config_name
            model.save()
        return model

    def refine_FV_WENO3(config_data, disc_idx, setting_name,
                        nCol_start=8, equivolume=True,
                        time_integrator=None,
                        unit_id='001', **kwargs):
        """Refinement function for radial FV WENO3: doubles nCol at each step."""

        config_data = copy.deepcopy(config_data)

        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator

        nCol = nCol_start * 2**disc_idx

        disc = config_data['input']['model']['unit_' + unit_id]['discretization']
        disc['SPATIAL_METHOD'] = 'FV'
        disc['NCOL'] = nCol
        disc['RECONSTRUCTION'] = 'WENO'
        disc['weno'] = {
            'WENO_ORDER': 3,
            'WENO_EPS': 1e-10,
            'BOUNDARY_MODEL': 0
        }

        if equivolume:
            unit = config_data['input']['model']['unit_' + unit_id]
            r0 = unit['col_radius_inner']
            r1 = unit['col_radius_outer']
            disc['GRID_FACES'] = grid_radial_equivolume(r0, r1, nCol).tolist()

        config_name = convergence.generate_1D_name(setting_name, 0, nCol)

        model = Cadet(install_path=cadet_path)
        model.root.input = config_data['input']
        if output_path is not None:
            model.filename = str(output_path) + '/' + config_name
            model.save()
        return model

    # ===========================================================================
    #  Study 0, Part 1: DG self-convergence (spatial profile at t_final)
    # ===========================================================================

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

    base_model_transport = setting_DG_transport.get_model()

    # P1-P5, nCells 1..512, reference = P5/1024 (last disc)
    poly_degs_0p1 = list(range(1, 6)) if not small_test else [3, 4]
    # 11 levels: 1,2,4,...,1024 (last = reference)
    n_disc_DG_0p1 = 11 if not small_test else 4

    methods_0p1 = []
    for p in poly_degs_0p1:
        methods_0p1.append((f'radCol1D_DG_bulk_transport_1comp_P{p}', p))

    addition = {
        'cadet_config_jsons': [base_model_transport] * len(methods_0p1),
        'cadet_config_names': [name for name, _ in methods_0p1],
        'include_sens': [False] * len(methods_0p1),
        'ref_files': [[None] for _ in methods_0p1],
        'unit_IDs': ['001'] * len(methods_0p1),
        'which': ['bulk'] * len(methods_0p1),
        'idas_abstol': [[None] for _ in methods_0p1],
        'ax_methods': [[p] for _, p in methods_0p1],
        'ax_discs': [[bench_func.disc_list(1, n_disc_DG_0p1)] for _ in methods_0p1],
        'par_methods': [[None] for _ in methods_0p1],
        'par_discs': [[None] for _ in methods_0p1],
        'disc_refinement_functions': [
            [partial(refine_DG,
                     setting_name=name,
                     polyDeg=polyDeg,
                     node_type='CGL',
                     nelem_start=1,
                     time_integrator=time_integrator)]
            for name, polyDeg in methods_0p1
        ],
    }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        cadet_config_names=cadet_config_names, addition=addition,
        disc_refinement_functions=disc_refinement_functions)

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

    # ===========================================================================
    #  Study 0, Part 2: DG vs FV WENO3 (outlet chromatogram)
    # ===========================================================================

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

    # DG: P1-P5, nCells 2..512
    poly_degs_0p2 = list(range(1, 6)) if not small_test else [3, 4]
    n_disc_DG_0p2 = 9 if not small_test else 4  # 2,4,8,...,512

    methods_0p2 = []
    for p in poly_degs_0p2:
        methods_0p2.append((f'radCol1D_DG_transport_1comp_P{p}', p))

    addition = {
        'cadet_config_jsons': [base_model_transport] * len(methods_0p2),
        'cadet_config_names': [name for name, _ in methods_0p2],
        'include_sens': [False] * len(methods_0p2),
        'ref_files': [[None] for _ in methods_0p2],
        'unit_IDs': ['001'] * len(methods_0p2),
        'which': ['outlet'] * len(methods_0p2),
        'idas_abstol': [[None] for _ in methods_0p2],
        'ax_methods': [[p] for _, p in methods_0p2],
        'ax_discs': [[bench_func.disc_list(2, n_disc_DG_0p2)] for _ in methods_0p2],
        'par_methods': [[None] for _ in methods_0p2],
        'par_discs': [[None] for _ in methods_0p2],
        'disc_refinement_functions': [
            [partial(refine_DG,
                     setting_name=name,
                     polyDeg=polyDeg,
                     node_type='CGL',
                     nelem_start=2,
                     time_integrator=time_integrator)]
            for name, polyDeg in methods_0p2
        ],
    }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        cadet_config_names=cadet_config_names, addition=addition,
        disc_refinement_functions=disc_refinement_functions)

    # FV WENO3: nCells 1,2,4,...,32768 (16 levels from 1)
    n_disc_FV_0 = 16 if not small_test else 4

    addition_fv = {
        'cadet_config_jsons': [base_model_transport],
        'cadet_config_names': ['radCol1D_FV_WENO3_transport_1comp'],
        'include_sens': [False],
        'ref_files': [[None]],
        'unit_IDs': ['001'],
        'which': ['outlet'],
        'idas_abstol': [[None]],
        'ax_methods': [[0]],
        'ax_discs': [[bench_func.disc_list(1, n_disc_FV_0)]],
        'par_methods': [[None]],
        'par_discs': [[None]],
        'disc_refinement_functions': [
            [partial(refine_FV_WENO3,
                     setting_name='radCol1D_FV_WENO3_transport_1comp',
                     nCol_start=1,
                     equivolume=True,
                     time_integrator=time_integrator)]
        ],
    }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        cadet_config_names=cadet_config_names, addition=addition_fv,
        disc_refinement_functions=disc_refinement_functions)

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

    # ===========================================================================
    #  Study 1: CGL vs LGL node comparison
    # ===========================================================================

    # --- Benchmark 1: rLRM, 1 comp, Linear rapid-eq ---
    # Reference: CGL P6, 512 cells

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

    poly_degs_1 = list(range(1, 6)) if not small_test else [3, 4]
    n_disc_DG_1 = 10 if not small_test else 4  # 1,2,4,...,512

    base_model_LRM_lin = setting_DG_LRM_lin.get_model()

    # Generate reference filename: CGL P6, 512 cells
    ref_name_bm1 = convergence.generate_1D_name(
        'radCol1D_DG_CGL_LRM_lin_1comp_ref', 6, 512 if not small_test else 8)
    ref_file_bm1 = str(output_path) + '/' + ref_name_bm1

    # Reference run (CGL P6/512)
    addition_ref = {
        'cadet_config_jsons': [base_model_LRM_lin],
        'cadet_config_names': ['radCol1D_DG_CGL_LRM_lin_1comp_ref'],
        'include_sens': [False],
        'ref_files': [[None]],
        'unit_IDs': ['001'],
        'which': ['outlet'],
        'idas_abstol': [[None]],
        'ax_methods': [[6 if not small_test else 3]],
        'ax_discs': [[[512 if not small_test else 8]]],
        'par_methods': [[None]],
        'par_discs': [[None]],
        'disc_refinement_functions': [
            [partial(refine_DG,
                     setting_name='radCol1D_DG_CGL_LRM_lin_1comp_ref',
                     polyDeg=6 if not small_test else 3,
                     node_type='CGL',
                     nelem_start=512 if not small_test else 8,
                     time_integrator=time_integrator)]
        ],
    }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        cadet_config_names=cadet_config_names, addition=addition_ref,
        disc_refinement_functions=disc_refinement_functions)

    # CGL and LGL test runs, all referencing the CGL P6/512 file
    for node_type in ['CGL', 'LGL']:
        methods_1 = []
        for p in poly_degs_1:
            methods_1.append((f'radCol1D_DG_{node_type}_LRM_lin_1comp_P{p}', p))

        addition = {
            'cadet_config_jsons': [base_model_LRM_lin] * len(methods_1),
            'cadet_config_names': [name for name, _ in methods_1],
            'include_sens': [False] * len(methods_1),
            'ref_files': [[ref_file_bm1] for _ in methods_1],
            'unit_IDs': ['001'] * len(methods_1),
            'which': ['outlet'] * len(methods_1),
            'idas_abstol': [[None] for _ in methods_1],
            'ax_methods': [[p] for _, p in methods_1],
            'ax_discs': [[bench_func.disc_list(1, n_disc_DG_1)] for _ in methods_1],
            'par_methods': [[None] for _ in methods_1],
            'par_discs': [[None] for _ in methods_1],
            'disc_refinement_functions': [
                [partial(refine_DG,
                         setting_name=name,
                         polyDeg=polyDeg,
                         node_type=node_type,
                         nelem_start=1,
                         time_integrator=time_integrator)]
                for name, polyDeg in methods_1
            ],
        }

        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            ax_methods, ax_discs,
            par_methods=par_methods, par_discs=par_discs,
            idas_abstol=idas_abstol,
            cadet_config_names=cadet_config_names, addition=addition,
            disc_refinement_functions=disc_refinement_functions)

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

    # --- Benchmark 2: rLRM, 4 comp, SMA kinetic ---
    # Reference: CGL P5, 64 cells

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

    base_model_LRM_SMA = setting_DG_LRM_SMA.get_model()

    # Generate reference filename: CGL P5, 64 cells
    ref_name_bm2 = convergence.generate_1D_name(
        'radCol1D_DG_CGL_LRM_SMA_4comp_ref', 5, 64 if not small_test else 8)
    ref_file_bm2 = str(output_path) + '/' + ref_name_bm2

    # Reference run (CGL P5/64)
    addition_ref2 = {
        'cadet_config_jsons': [base_model_LRM_SMA],
        'cadet_config_names': ['radCol1D_DG_CGL_LRM_SMA_4comp_ref'],
        'include_sens': [False],
        'ref_files': [[None]],
        'unit_IDs': ['001'],
        'which': ['outlet'],
        'idas_abstol': [[None]],
        'ax_methods': [[5 if not small_test else 3]],
        'ax_discs': [[[64 if not small_test else 8]]],
        'par_methods': [[None]],
        'par_discs': [[None]],
        'disc_refinement_functions': [
            [partial(refine_DG,
                     setting_name='radCol1D_DG_CGL_LRM_SMA_4comp_ref',
                     polyDeg=5 if not small_test else 3,
                     node_type='CGL',
                     nelem_start=64 if not small_test else 8,
                     time_integrator=time_integrator)]
        ],
    }

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        cadet_config_names=cadet_config_names, addition=addition_ref2,
        disc_refinement_functions=disc_refinement_functions)

    # CGL and LGL test runs
    for node_type in ['CGL', 'LGL']:
        methods_2 = []
        for p in poly_degs_1:
            methods_2.append((f'radCol1D_DG_{node_type}_LRM_SMA_4comp_P{p}', p))

        addition = {
            'cadet_config_jsons': [base_model_LRM_SMA] * len(methods_2),
            'cadet_config_names': [name for name, _ in methods_2],
            'include_sens': [False] * len(methods_2),
            'ref_files': [[ref_file_bm2] for _ in methods_2],
            'unit_IDs': ['001'] * len(methods_2),
            'which': ['outlet'] * len(methods_2),
            'idas_abstol': [[None] for _ in methods_2],
            'ax_methods': [[p] for _, p in methods_2],
            'ax_discs': [[bench_func.disc_list(1, n_disc_DG_1)] for _ in methods_2],
            'par_methods': [[None] for _ in methods_2],
            'par_discs': [[None] for _ in methods_2],
            'disc_refinement_functions': [
                [partial(refine_DG,
                         setting_name=name,
                         polyDeg=polyDeg,
                         node_type=node_type,
                         nelem_start=1,
                         time_integrator=time_integrator)]
                for name, polyDeg in methods_2
            ],
        }

        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            ax_methods, ax_discs,
            par_methods=par_methods, par_discs=par_discs,
            idas_abstol=idas_abstol,
            cadet_config_names=cadet_config_names, addition=addition,
            disc_refinement_functions=disc_refinement_functions)

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

    # ===========================================================================
    #  Study 2: DG vs FV with variable coefficients — 6 configurations
    # ===========================================================================

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

    # GRM configs: P1-P4, fewer DG refinements
    poly_degs_GRM = list(range(1, 5)) if not small_test else [3, 4]
    n_disc_DG_GRM = 9 if not small_test else 4

    # LRM/LRMP configs: P1-P5
    poly_degs_LRM = list(range(1, 6)) if not small_test else [3, 4]
    n_disc_DG_LRM = 10 if not small_test else 4

    n_disc_FV_2 = 19 if not small_test else 4  # 1,2,4,...,262144

    configs_2 = [
        # (setting_module, config_prefix, poly_degs, n_disc_DG, time_integ)
        (setting_DG_GRM_lin_var, 'radGRM_DG_lin_1comp_varCoeff', poly_degs_GRM, n_disc_DG_GRM, time_integrator),
        (setting_DG_GRM_SMA_var, 'radGRM_DG_SMA_4comp_varCoeff', poly_degs_GRM, n_disc_DG_GRM, time_integrator),
        (setting_DG_LRM_lin_var, 'radLRM_DG_lin_1comp_varCoeff', poly_degs_LRM, n_disc_DG_LRM, time_integrator),
        (setting_DG_LRM_SMA_var, 'radLRM_DG_SMA_4comp_varCoeff', poly_degs_LRM, n_disc_DG_LRM, time_integrator),
        (setting_DG_LRMP_lin_var, 'radLRMP_DG_lin_1comp_varCoeff', poly_degs_LRM, n_disc_DG_LRM, time_integrator),
        (setting_DG_LRMP_SMA_var, 'radLRMP_DG_SMA_4comp_varCoeff', poly_degs_LRM, n_disc_DG_LRM, time_integrator),
    ]

    for setting_mod, prefix, poly_degs, n_disc_DG, ti in configs_2:
        base_model = setting_mod.get_model()

        # DG methods
        methods = []
        for p in poly_degs:
            methods.append((f'{prefix}_P{p}', p))

        addition = {
            'cadet_config_jsons': [base_model] * len(methods),
            'cadet_config_names': [name for name, _ in methods],
            'include_sens': [False] * len(methods),
            'ref_files': [[None] for _ in methods],
            'unit_IDs': ['001'] * len(methods),
            'which': ['outlet'] * len(methods),
            'idas_abstol': [[None] for _ in methods],
            'ax_methods': [[p] for _, p in methods],
            'ax_discs': [[bench_func.disc_list(1, n_disc_DG)] for _ in methods],
            'par_methods': [[None] for _ in methods],
            'par_discs': [[None] for _ in methods],
            'disc_refinement_functions': [
                [partial(refine_DG,
                         setting_name=name,
                         polyDeg=polyDeg,
                         node_type='CGL',
                         nelem_start=1,
                         time_integrator=ti)]
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

        # FV WENO3 equivolume reference
        fv_name = f'{prefix}_FV_WENO3'
        addition_fv = {
            'cadet_config_jsons': [base_model],
            'cadet_config_names': [fv_name],
            'include_sens': [False],
            'ref_files': [[None]],
            'unit_IDs': ['001'],
            'which': ['outlet'],
            'idas_abstol': [[None]],
            'ax_methods': [[0]],
            'ax_discs': [[bench_func.disc_list(1, n_disc_FV_2)]],
            'par_methods': [[None]],
            'par_discs': [[None]],
            'disc_refinement_functions': [
                [partial(refine_FV_WENO3,
                         setting_name=fv_name,
                         nCol_start=1,
                         equivolume=True,
                         time_integrator=ti)]
            ],
        }

        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            ax_methods, ax_discs,
            par_methods=par_methods, par_discs=par_discs,
            idas_abstol=idas_abstol,
            cadet_config_names=cadet_config_names, addition=addition_fv,
            disc_refinement_functions=disc_refinement_functions)

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

    # ===========================================================================
    #  Study 3: Langmuir 2-comp oscillation study
    # ===========================================================================

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

    poly_degs_3 = [3, 4, 5] if not small_test else [3]
    n_disc_DG_3 = 11 if not small_test else 4  # 1,2,4,...,1024
    n_disc_FV_3 = 19 if not small_test else 4  # 1,2,4,...,262144

    for D0, label in [(1e-4, 'D1e4'), (1e-5, 'D1e5')]:
        base_model_lang = setting_DG_LRM_lang.get_model(D0=D0)

        # DG methods
        methods_3 = []
        for p in poly_degs_3:
            methods_3.append((f'radLRM_DG_lang_2comp_{label}_P{p}', p))

        addition = {
            'cadet_config_jsons': [base_model_lang] * len(methods_3),
            'cadet_config_names': [name for name, _ in methods_3],
            'include_sens': [False] * len(methods_3),
            'ref_files': [[None] for _ in methods_3],
            'unit_IDs': ['001'] * len(methods_3),
            'which': ['outlet'] * len(methods_3),
            'idas_abstol': [[None] for _ in methods_3],
            'ax_methods': [[p] for _, p in methods_3],
            'ax_discs': [[bench_func.disc_list(1, n_disc_DG_3)] for _ in methods_3],
            'par_methods': [[None] for _ in methods_3],
            'par_discs': [[None] for _ in methods_3],
            'disc_refinement_functions': [
                [partial(refine_DG,
                         setting_name=name,
                         polyDeg=polyDeg,
                         node_type='CGL',
                         nelem_start=1,
                         time_integrator=time_integrator)]
                for name, polyDeg in methods_3
            ],
        }

        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            ax_methods, ax_discs,
            par_methods=par_methods, par_discs=par_discs,
            idas_abstol=idas_abstol,
            cadet_config_names=cadet_config_names, addition=addition,
            disc_refinement_functions=disc_refinement_functions)

        # FV WENO3 equivolume reference: nCells 1,2,4,...,262144
        fv_name = f'radLRM_FV_WENO3_lang_2comp_{label}'
        addition_fv = {
            'cadet_config_jsons': [base_model_lang],
            'cadet_config_names': [fv_name],
            'include_sens': [False],
            'ref_files': [[None]],
            'unit_IDs': ['001'],
            'which': ['outlet'],
            'idas_abstol': [[None]],
            'ax_methods': [[0]],
            'ax_discs': [[bench_func.disc_list(1, n_disc_FV_3)]],
            'par_methods': [[None]],
            'par_discs': [[None]],
            'disc_refinement_functions': [
                [partial(refine_FV_WENO3,
                         setting_name=fv_name,
                         nCol_start=1,
                         equivolume=True,
                         time_integrator=time_integrator)]
            ],
        }

        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            ax_methods, ax_discs,
            par_methods=par_methods, par_discs=par_discs,
            idas_abstol=idas_abstol,
            cadet_config_names=cadet_config_names, addition=addition_fv,
            disc_refinement_functions=disc_refinement_functions)

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
