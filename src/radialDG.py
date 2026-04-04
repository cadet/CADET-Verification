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
        'ABSTOL': 1e-10, 'RELTOL': 1e-8, 'ALGTOL': 1e-10,
        'USE_MODIFIED_NEWTON': False,
        'INIT_STEP_SIZE': 1e-10,
        'MAX_STEPS': 10000000
    }

    # Stricter tolerances for non-stiff models (Study 0, Study 1 BM1 linear)
    time_integrator_strict = {
        'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
        'USE_MODIFIED_NEWTON': False,
        'INIT_STEP_SIZE': 1e-6,
        'MAX_STEPS': 5000000
    }

    # ---- Helper: equivolume radial grid ----

    def grid_radial_equivolume(r0, r1, n):
        r2_faces = np.linspace(r0**2, r1**2, n + 1)
        return np.sqrt(r2_faces)

    # ---- Refinement functions ----

    # Map from particle configuration to radial unit type.
    # RADIAL_COLUMN_MODEL_1D only works for DG with npartype=0 (pure transport).
    # For DG with particles, the builder expects dedicated radial unit types
    # and selects DG/FV variant based on SPATIAL_METHOD in discretization.
    _RADIAL_UNIT_TYPE_MAP = {
        'EQUILIBRIUM_PARTICLE': 'RADIAL_LUMPED_RATE_MODEL_WITHOUT_PORES',
        'HOMOGENEOUS_PARTICLE': 'RADIAL_LUMPED_RATE_MODEL_WITH_PORES',
        'GENERAL_RATE_PARTICLE': 'RADIAL_GENERAL_RATE_MODEL',
    }

    def _get_particle_type(unit_cfg):
        """Determine particle type string from unit config flags."""
        if unit_cfg.get('npartype', 0) == 0:
            return None
        pt = unit_cfg.get('particle_type_000', {})
        film = pt.get('has_film_diffusion', 0)
        pore = pt.get('has_pore_diffusion', 0)
        surf = pt.get('has_surface_diffusion', 0)
        if film and pore:
            return 'GENERAL_RATE_PARTICLE'
        elif film:
            return 'HOMOGENEOUS_PARTICLE'
        else:
            return 'EQUILIBRIUM_PARTICLE'

    def refine_DG(config_data, disc_idx, setting_name,
                  polyDeg, node_type='CGL',
                  nelem_start=2, time_integrator=None,
                  unit_id='001',
                  refine_par=False,
                  **kwargs):
        """Refinement function for radial DG: doubles nElem at each step.

        For GRM models, set refine_par=True to scale pore discretization
        with bulk: PAR_POLYDEG = max(1, nCells // 4), PAR_NELEM = 1.
        """

        config_data = copy.deepcopy(config_data)

        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator

        nElem = nelem_start * 2**disc_idx

        unit_cfg = config_data['input']['model']['unit_' + unit_id]

        # Switch to dedicated radial unit type if the model has particles
        ptype = _get_particle_type(unit_cfg)
        if ptype is not None and ptype in _RADIAL_UNIT_TYPE_MAP:
            unit_cfg['unit_type'] = _RADIAL_UNIT_TYPE_MAP[ptype]

        disc = unit_cfg['discretization']
        disc['SPATIAL_METHOD'] = 'DG'
        disc['POLYDEG'] = polyDeg
        disc['NELEM'] = nElem

        unit_cfg['node_type'] = node_type

        # Scale pore discretization with bulk (CADET-Julia convention)
        if refine_par and 'particle_type_000' in unit_cfg:
            par_disc = unit_cfg['particle_type_000']['discretization']
            par_disc['PAR_POLYDEG'] = max(1, nElem // 4)
            par_disc['PAR_NELEM'] = 1

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

        unit_cfg = config_data['input']['model']['unit_' + unit_id]

        # Switch to dedicated radial unit type if the model has particles
        ptype = _get_particle_type(unit_cfg)
        if ptype is not None and ptype in _RADIAL_UNIT_TYPE_MAP:
            unit_cfg['unit_type'] = _RADIAL_UNIT_TYPE_MAP[ptype]

        disc = unit_cfg['discretization']
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
        'which': ['outlet'] * len(methods_0p1),
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
                     time_integrator=time_integrator_strict)]
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
                     time_integrator=time_integrator_strict)]
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

    # FV WENO3: nCells 4,8,16,...,32768 (WENO3 requires >= 4 cells)
    n_disc_FV_0 = 14 if not small_test else 4

    addition_fv = {
        'cadet_config_jsons': [base_model_transport],
        'cadet_config_names': ['radCol1D_FV_WENO3_transport_1comp'],
        'include_sens': [False],
        'ref_files': [[None]],
        'unit_IDs': ['001'],
        'which': ['outlet'],
        'idas_abstol': [[None]],
        'ax_methods': [[0]],
        'ax_discs': [[bench_func.disc_list(4, n_disc_FV_0)]],
        'par_methods': [[None]],
        'par_discs': [[None]],
        'disc_refinement_functions': [
            [partial(refine_FV_WENO3,
                     setting_name='radCol1D_FV_WENO3_transport_1comp',
                     nCol_start=4,
                     equivolume=True,
                     time_integrator=time_integrator_strict)]
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

    # Generate reference simulation directly (CGL P6/512 or P3/8 for small_test)
    ref_polyDeg_bm1 = 6 if not small_test else 3
    ref_nElem_bm1 = 512 if not small_test else 8
    ref_model_bm1 = refine_DG(
        base_model_LRM_lin, 0,
        setting_name='radCol1D_DG_CGL_LRM_lin_1comp_ref',
        polyDeg=ref_polyDeg_bm1,
        node_type='CGL',
        nelem_start=ref_nElem_bm1,
        time_integrator=time_integrator_strict)
    ref_model_bm1.run()
    # ref_files entries are just the filename (convergence prepends output_path)
    ref_file_bm1 = convergence.generate_1D_name(
        'radCol1D_DG_CGL_LRM_lin_1comp_ref', ref_polyDeg_bm1, ref_nElem_bm1)

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
                         time_integrator=time_integrator_strict)]
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

    # Reference: CGL P6/512 or P3/8 for small_test
    ref_polyDeg_bm2 = 6 if not small_test else 3
    ref_nElem_bm2 = 512 if not small_test else 8
    ref_model_bm2 = refine_DG(
        base_model_LRM_SMA, 0,
        setting_name='radCol1D_DG_CGL_LRM_SMA_4comp_ref',
        polyDeg=ref_polyDeg_bm2,
        node_type='CGL',
        nelem_start=ref_nElem_bm2,
        time_integrator=time_integrator)
    ref_model_bm2.run()
    ref_file_bm2 = convergence.generate_1D_name(
        'radCol1D_DG_CGL_LRM_SMA_4comp_ref', ref_polyDeg_bm2, ref_nElem_bm2)

    for node_type in ['CGL', 'LGL']:
        methods_bm2 = []
        for p in poly_degs_1:
            methods_bm2.append((f'radCol1D_DG_{node_type}_LRM_SMA_4comp_P{p}', p))

        addition = {
            'cadet_config_jsons': [base_model_LRM_SMA] * len(methods_bm2),
            'cadet_config_names': [name for name, _ in methods_bm2],
            'include_sens': [False] * len(methods_bm2),
            'ref_files': [[ref_file_bm2] for _ in methods_bm2],
            'unit_IDs': ['001'] * len(methods_bm2),
            'which': ['outlet'] * len(methods_bm2),
            'idas_abstol': [[None] for _ in methods_bm2],
            'ax_methods': [[p] for _, p in methods_bm2],
            'ax_discs': [[bench_func.disc_list(1, n_disc_DG_1)] for _ in methods_bm2],
            'par_methods': [[None] for _ in methods_bm2],
            'par_discs': [[None] for _ in methods_bm2],
            'disc_refinement_functions': [
                [partial(refine_DG,
                         setting_name=name,
                         polyDeg=polyDeg,
                         node_type=node_type,
                         nelem_start=1,
                         time_integrator=time_integrator)]
                for name, polyDeg in methods_bm2
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

    n_disc_FV_2 = 17 if not small_test else 4  # 4,8,...,262144

    configs_2 = [
        # (setting_module, config_prefix, poly_degs, n_disc_DG, time_integ, is_grm)
        (setting_DG_GRM_lin_var, 'radGRM_DG_lin_1comp_varCoeff', poly_degs_GRM, n_disc_DG_GRM, time_integrator, True),
        (setting_DG_GRM_SMA_var, 'radGRM_DG_SMA_4comp_varCoeff', poly_degs_GRM, n_disc_DG_GRM, time_integrator, True),
        (setting_DG_LRM_lin_var, 'radLRM_DG_lin_1comp_varCoeff', poly_degs_LRM, n_disc_DG_LRM, time_integrator, False),
        (setting_DG_LRM_SMA_var, 'radLRM_DG_SMA_4comp_varCoeff', poly_degs_LRM, n_disc_DG_LRM, time_integrator, False),
        (setting_DG_LRMP_lin_var, 'radLRMP_DG_lin_1comp_varCoeff', poly_degs_LRM, n_disc_DG_LRM, time_integrator, False),
        (setting_DG_LRMP_SMA_var, 'radLRMP_DG_SMA_4comp_varCoeff', poly_degs_LRM, n_disc_DG_LRM, time_integrator, False),
    ]

    for setting_mod, prefix, poly_degs, n_disc_DG, ti, is_grm in configs_2:
        base_model = setting_mod.get_model()

        # GRM starts at nElem=8 to avoid stiffness issues at very coarse grids
        nelem_start_2 = 8 if is_grm else 4

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
            'ax_discs': [[bench_func.disc_list(nelem_start_2, n_disc_DG)] for _ in methods],
            'par_methods': [[None] for _ in methods],
            'par_discs': [[None] for _ in methods],
            'disc_refinement_functions': [
                [partial(refine_DG,
                         setting_name=name,
                         polyDeg=polyDeg,
                         node_type='CGL',
                         nelem_start=nelem_start_2,
                         time_integrator=ti,
                         refine_par=is_grm)]
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
            'ax_discs': [[bench_func.disc_list(4, n_disc_FV_2)]],
            'par_methods': [[None]],
            'par_discs': [[None]],
            'disc_refinement_functions': [
                [partial(refine_FV_WENO3,
                         setting_name=fv_name,
                         nCol_start=4,
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
    n_disc_FV_3 = 17 if not small_test else 4  # 4,8,...,262144

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
            'ax_discs': [[bench_func.disc_list(4, n_disc_FV_3)]],
            'par_methods': [[None]],
            'par_discs': [[None]],
            'disc_refinement_functions': [
                [partial(refine_FV_WENO3,
                         setting_name=fv_name,
                         nCol_start=4,
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
