# -*- coding: utf-8 -*-
"""
EOC tests for radial DG discretization.

Studies:
  0. Pure bulk transport — DG self-convergence + DG vs FV WENO3
  1. CGL vs LGL node comparison — rLRM 1-comp Linear + 4-comp SMA
  2. DG vs FV convergence — 3 configurations (rGRM x 1/4 comp + rLRM lin)
  3. Langmuir 2-comp oscillation study — two dispersion levels
"""

import os
import copy
import traceback
import numpy as np
from functools import partial

import src.benchmark_models.setting_radCol1D_DG_transport as setting_DG_transport
import src.benchmark_models.setting_radCol1D_DG_LRM_lin_1comp as setting_DG_LRM_lin
import src.benchmark_models.setting_radCol1D_DG_LRM_SMA_4comp as setting_DG_LRM_SMA
import src.benchmark_models.setting_radCol1D_DG_GRM_lin_1comp as setting_DG_GRM_lin
import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp as setting_DG_GRM_SMA
import src.benchmark_models.setting_radCol1D_DG_LRM_lin_1comp_varCoeff as setting_DG_LRM_lin_var
import src.benchmark_models.setting_radCol1D_DG_LRM_lang_2comp as setting_DG_LRM_lang

import src.bench_func as bench_func
import src.bench_configs as bench_configs
import src.utility.convergence as convergence

from cadet import Cadet


def radialDG_tests(n_jobs, small_test, output_path, cadet_path, studies=None, study1_polydegs=None, study1_node_types=None, study1_benchmarks=None, study1_ref_only=False, study1_skip_ref=False, study1_fv_ncol_bm2=None, study2_configs=None, study2_methods=None, study2_polydegs=None, study2_skip_fv=False, study2_skip_dg_rerun=False, study3_dispersions=None, study3_polydegs=None, study3_methods=None, fv_start_ncol=None, fv_n_disc=None):

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
                  only_return_name=False,
                  **kwargs):
        """Refinement function for radial DG: doubles nElem at each step.

        For GRM models, set refine_par=True to co-refine particle
        discretization with bulk: PAR_NELEM = max(1, nElem // 4),
        PAR_POLYDEG = 3 (fixed).
        """

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

        # Switch to dedicated radial unit type if the model has particles
        ptype = _get_particle_type(unit_cfg)
        if ptype is not None and ptype in _RADIAL_UNIT_TYPE_MAP:
            unit_cfg['unit_type'] = _RADIAL_UNIT_TYPE_MAP[ptype]

        disc = unit_cfg['discretization']
        disc['SPATIAL_METHOD'] = 'DG'
        disc['POLYDEG'] = polyDeg
        disc['NELEM'] = nElem

        unit_cfg['POLYNOMIAL_INTERPOLATION_NODES'] = node_type

        # Co-refine particle discretization with bulk (Jan Breuer's strategy):
        #   PAR_POLYDEG = 3 (fixed), PAR_NELEM = max(1, nElem // 4)
        #   NCELLS mirrors PAR_NELEM for FV particle compatibility
        if refine_par and 'particle_type_000' in unit_cfg:
            par_disc = unit_cfg['particle_type_000']['discretization']
            par_nelem = max(1, nElem // 4)
            par_disc['NCELLS'] = par_nelem
            par_disc['PAR_POLYDEG'] = 3
            par_disc['PAR_NELEM'] = par_nelem

        model = Cadet(install_path=cadet_path)
        model.root.input = config_data['input']
        if output_path is not None:
            model.filename = str(output_path) + '/' + config_name
            model.save()
        return model

    def refine_FV_WENO3(config_data, disc_idx, setting_name,
                        nCol_start=8, equivolume=True,
                        time_integrator=None,
                        refine_par=False,
                        unit_id='001', only_return_name=False,
                        **kwargs):
        """Refinement function for radial FV WENO3: doubles nCol at each step."""

        nCol = nCol_start * 2**disc_idx
        config_name = convergence.generate_1D_name(setting_name, 0, nCol)

        if only_return_name:
            if output_path is not None:
                return str(output_path) + '/' + config_name
            return config_name

        config_data = copy.deepcopy(config_data)

        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator

        unit_cfg = config_data['input']['model']['unit_' + unit_id]

        # Switch to dedicated radial unit type if the model has particles
        ptype = _get_particle_type(unit_cfg)
        if ptype is not None and ptype in _RADIAL_UNIT_TYPE_MAP:
            unit_cfg['unit_type'] = _RADIAL_UNIT_TYPE_MAP[ptype]

        disc = unit_cfg['discretization']
        disc['SPATIAL_METHOD'] = 'FV'
        disc['NCOL'] = nCol
        disc['MAX_KRYLOV'] = 0
        disc['MAX_RESTARTS'] = 10
        disc['GS_TYPE'] = 1
        disc['SCHUR_SAFETY'] = 1e-8
        disc['RECONSTRUCTION'] = 'WENO'
        disc['weno'] = {
            'WENO_ORDER': 3,
            'WENO_EPS': 1e-10,
            'BOUNDARY_MODEL': 0
        }

        # Ensure particle discretization fields for GRM
        pt0 = unit_cfg.get('particle_type_000', {})
        pt_disc = pt0.get('discretization', {})
        if pt_disc:
            pt_disc.setdefault('NCELLS', 10)
            pt_disc.setdefault('PAR_DISC_TYPE', 'EQUIDISTANT_PAR')

        # Scale particle cell count with bulk (fixed PAR_POLYDEG=3)
        if refine_par and 'particle_type_000' in unit_cfg:
            par_disc = unit_cfg['particle_type_000']['discretization']
            par_disc['NCELLS'] = max(1, nCol // 4)
            par_disc['PAR_POLYDEG'] = 3
            par_disc['PAR_NELEM'] = 1

        if equivolume:
            unit = config_data['input']['model']['unit_' + unit_id]
            r0 = unit['col_radius_inner']
            r1 = unit['col_radius_outer']
            disc['GRID_FACES'] = grid_radial_equivolume(r0, r1, nCol).tolist()

        model = Cadet(install_path=cadet_path)
        model.root.input = config_data['input']
        if output_path is not None:
            model.filename = str(output_path) + '/' + config_name
            model.save()
        return model

    def _run(s):
        return studies is None or s in studies

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
    poly_degs_0p1 = list(range(1, 6)) if not small_test else [1, 2]
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

    if _run(0):
        try:
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
        except Exception:
            print(f"\n*** Study 0 Part 1 FAILED ***\n{traceback.format_exc()}")

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
    poly_degs_0p2 = list(range(1, 6)) if not small_test else [1, 2]
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

    # FV WENO3: nCells 4,8,16,...,131072 (WENO3 requires >= 4 cells)
    _fv_start_0 = fv_start_ncol if fv_start_ncol is not None else 4
    _fv_n_0 = fv_n_disc if fv_n_disc is not None else (16 if not small_test else 4)

    addition_fv = {
        'cadet_config_jsons': [base_model_transport],
        'cadet_config_names': ['radCol1D_FV_WENO3_transport_1comp'],
        'include_sens': [False],
        'ref_files': [[None]],
        'unit_IDs': ['001'],
        'which': ['outlet'],
        'idas_abstol': [[None]],
        'ax_methods': [[0]],
        'ax_discs': [[bench_func.disc_list(_fv_start_0, _fv_n_0)]],
        'par_methods': [[None]],
        'par_discs': [[None]],
        'disc_refinement_functions': [
            [partial(refine_FV_WENO3,
                     setting_name='radCol1D_FV_WENO3_transport_1comp',
                     nCol_start=_fv_start_0,
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

    if _run(0):
        try:
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
        except Exception:
            print(f"\n*** Study 0 Part 2 FAILED ***\n{traceback.format_exc()}")

    # ===========================================================================
    #  Study 0, Part 3: DG convergence against FV WENO3 reference (no new sims)
    # ===========================================================================

    # Reuses simulation files from Part 2.  Only recomputes convergence tables
    # with the finest FV WENO3 simulation as the reference instead of DG self-ref.

    fv_finest_ncol_0 = _fv_start_0 * 2**(_fv_n_0 - 1)
    fv_ref_file_0 = convergence.generate_1D_name(
        'radCol1D_FV_WENO3_transport_1comp', 0, fv_finest_ncol_0)

    cadet_configs_0p3 = []
    cadet_config_names_0p3 = []
    include_sens_0p3 = []
    ref_files_0p3 = []
    unit_IDs_0p3 = []
    which_0p3 = []
    idas_abstol_0p3 = []
    ax_methods_0p3 = []
    ax_discs_0p3 = []
    par_methods_0p3 = []
    par_discs_0p3 = []
    disc_refinement_functions_0p3 = []

    addition_0p3 = {
        'cadet_config_jsons': [base_model_transport] * len(methods_0p2),
        'cadet_config_names': [f'{name}_FVref' for name, _ in methods_0p2],
        'include_sens': [False] * len(methods_0p2),
        'ref_files': [[fv_ref_file_0] for _ in methods_0p2],
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
        cadet_configs_0p3, include_sens_0p3, ref_files_0p3, unit_IDs_0p3,
        which_0p3, ax_methods_0p3, ax_discs_0p3,
        par_methods=par_methods_0p3, par_discs=par_discs_0p3,
        idas_abstol=idas_abstol_0p3,
        cadet_config_names=cadet_config_names_0p3, addition=addition_0p3,
        disc_refinement_functions=disc_refinement_functions_0p3)

    if _run(0):
        try:
            bench_func.run_convergence_analysis(
                output_path=output_path,
                cadet_path=cadet_path,
                cadet_configs=cadet_configs_0p3,
                cadet_config_names=cadet_config_names_0p3,
                include_sens=include_sens_0p3,
                ref_files=ref_files_0p3,
                unit_IDs=unit_IDs_0p3,
                which=which_0p3,
                ax_methods=ax_methods_0p3,
                ax_discs=ax_discs_0p3,
                par_methods=par_methods_0p3,
                par_discs=par_discs_0p3,
                idas_abstol=idas_abstol_0p3,
                n_jobs=n_jobs,
                rerun_sims=False,
                disc_refinement_functions=disc_refinement_functions_0p3
            )
        except Exception:
            print(f"\n*** Study 0 Part 3 FAILED ***\n{traceback.format_exc()}")

    # ===========================================================================
    #  Study 1: CGL vs LGL node comparison
    #  Reference: single finest FV WENO3 equivolume simulation
    # ===========================================================================

    poly_degs_1 = list(range(1, 7)) if not small_test else [1, 2]
    if study1_polydegs is not None:
        poly_degs_1 = [p for p in poly_degs_1 if p in study1_polydegs]
    node_types_1 = study1_node_types if study1_node_types is not None else ['CGL', 'LGL']

    # Per-polyDeg refinement levels: avoid running past the FV reference floor.
    # DG outlet converges at rate ~2k+1.  Floor hit at nElem ~ floor^(-1/(2k+1)).
    # Include 2-3 extra levels to show the plateau clearly.
    if not small_test:
        # BM1 floor ~1e-10 (131K FV cells)
        _n_disc_bm1 = {1: 10, 2: 10, 3: 8, 4: 7, 5: 7, 6: 7}
        # BM2 floor ~1e-7  (32K FV cells)
        _n_disc_bm2 = {1: 10, 2: 9, 3: 8, 4: 6, 5: 6}
    else:
        _n_disc_bm1 = {p: 4 for p in range(1, 7)}
        _n_disc_bm2 = {p: 4 for p in range(1, 7)}

    base_model_LRM_lin = setting_DG_LRM_lin.get_model()
    base_model_LRM_SMA = setting_DG_LRM_SMA.get_model()

    # FV WENO3 reference: only the single finest simulation, not a full sweep
    _fv_ncol_bm1 = 131072 if not small_test else 32
    fv_name_bm1 = 'radCol1D_FV_WENO3_LRM_lin_1comp'
    ref_file_bm1 = convergence.generate_1D_name(fv_name_bm1, 0, _fv_ncol_bm1)

    _fv_ncol_bm2 = study1_fv_ncol_bm2 if study1_fv_ncol_bm2 is not None else (32768 if not small_test else 32)
    fv_name_bm2 = 'radCol1D_FV_WENO3_LRM_SMA_4comp'
    ref_file_bm2 = convergence.generate_1D_name(fv_name_bm2, 0, _fv_ncol_bm2)

    _run_bm1 = study1_benchmarks is None or 1 in study1_benchmarks
    _run_bm2 = study1_benchmarks is None or 2 in study1_benchmarks

    # Compute FV references (skip if study1_skip_ref=True, e.g. downloaded from artifact)
    if _run(1) and not study1_skip_ref:
      if _run_bm1:
        try:
          fv_ref_model_bm1 = refine_FV_WENO3(
              base_model_LRM_lin, 0,
              setting_name=fv_name_bm1,
              nCol_start=_fv_ncol_bm1,
              equivolume=True,
              time_integrator=time_integrator_strict)
          fv_ref_model_bm1.run()
          print(f"  Study 1 BM1 FV reference computed ({_fv_ncol_bm1} cells).")
        except Exception:
          print(f"\n*** Study 1 BM1 FV REF FAILED ***\n{traceback.format_exc()}")

      if _run_bm2:
        try:
          fv_ref_model_bm2 = refine_FV_WENO3(
              base_model_LRM_SMA, 0,
              setting_name=fv_name_bm2,
              nCol_start=_fv_ncol_bm2,
              equivolume=True,
              time_integrator=time_integrator)
          fv_ref_model_bm2.run()
          print(f"  Study 1 BM2 FV reference computed ({_fv_ncol_bm2} cells).")
        except Exception:
          print(f"\n*** Study 1 BM2 FV REF FAILED ***\n{traceback.format_exc()}")

    # --- Benchmark 1: rLRM, 1 comp, Linear rapid-eq ---

    if _run(1) and not study1_ref_only and _run_bm1:
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
        for node_type in node_types_1:
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
                'ax_discs': [[bench_func.disc_list(1, _n_disc_bm1[p])] for _, p in methods_1],
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
      except Exception:
        print(f"\n*** Study 1 BM1 FAILED ***\n{traceback.format_exc()}")

    # --- Benchmark 2: rLRM, 4 comp, SMA kinetic ---

    if _run(1) and not study1_ref_only and _run_bm2:
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
        for node_type in node_types_1:
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
                'ax_discs': [[bench_func.disc_list(1, _n_disc_bm2[p])] for _, p in methods_bm2],
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
      except Exception:
        print(f"\n*** Study 1 BM2 FAILED ***\n{traceback.format_exc()}")

    #
    # ===========================================================================
    #  Study 2: DG vs FV convergence — 3 configurations
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

    # GRM configs: P1-P4, DG from 4 to 128 (6 levels)
    poly_degs_GRM = list(range(1, 5)) if not small_test else [1, 2]
    n_disc_DG_GRM = 6 if not small_test else 4

    # LRM/LRMP configs: P1-P5
    poly_degs_LRM = list(range(1, 6)) if not small_test else [1, 2]
    n_disc_DG_LRM = 10 if not small_test else 4

    _fv_start_2 = fv_start_ncol if fv_start_ncol is not None else 4
    n_disc_FV_2 = fv_n_disc if fv_n_disc is not None else (17 if not small_test else 4)

    # GRM FV crashes at high cell counts (memory), limit FV refinements:
    #   GRM lin: 128 to 16384 (8 levels from 128)
    #   GRM SMA: 128 to 4096 (6 levels from 128)
    n_disc_FV_2_GRM_lin = fv_n_disc if fv_n_disc is not None else (8 if not small_test else 4)
    n_disc_FV_2_GRM_SMA = fv_n_disc if fv_n_disc is not None else (6 if not small_test else 4)

    configs_2 = [
        # (setting_module, config_prefix, poly_degs, n_disc_DG, time_integ, is_grm, n_disc_FV_override, fv_time_integrator_override, consistent_init_mode_fv)
        (setting_DG_GRM_lin, 'radGRM_DG_lin_1comp', poly_degs_GRM, n_disc_DG_GRM, time_integrator_strict, True, n_disc_FV_2_GRM_lin, None, None),
        (setting_DG_GRM_SMA, 'radGRM_DG_SMA_4comp', poly_degs_GRM, n_disc_DG_GRM, time_integrator_strict, True, n_disc_FV_2_GRM_SMA, None, None),
        (setting_DG_LRM_lin_var, 'radLRM_DG_lin_1comp_varCoeff', poly_degs_LRM, n_disc_DG_LRM, time_integrator_strict, False, 16, None, None),
    ]

    if study2_configs is not None:
        configs_2 = [configs_2[i] for i in study2_configs if i < len(configs_2)]

    # GRM starts at nElem=4 (to match par refinement), others at 8
    nelem_start_2 = 8
    nelem_start_2_GRM = 4

    # --- Phase 1: Run FV WENO3 sweep (self-convergence) for all configs ---

    fv_configs = []
    fv_config_names = []
    fv_include_sens = []
    fv_ref_files = []
    fv_unit_IDs = []
    fv_which = []
    fv_idas_abstol = []
    fv_ax_methods = []
    fv_ax_discs = []
    fv_par_methods = []
    fv_par_discs = []
    fv_disc_refinement_functions = []

    # Track finest FV filename per config for DG reference
    fv_finest_names = {}

    for cfg_idx, (setting_mod, prefix, poly_degs, n_disc_DG, ti, is_grm, n_disc_FV_override, fv_ti_override, consistent_init_mode_fv) in enumerate(configs_2):
        base_model = setting_mod.get_model()

        # Apply consistent_init_mode override for FV if needed
        if consistent_init_mode_fv is not None:
            base_model_fv = copy.deepcopy(base_model)
            base_model_fv['input']['solver']['consistent_init_mode'] = consistent_init_mode_fv
        else:
            base_model_fv = base_model

        _fv_start_2_cfg = fv_start_ncol if fv_start_ncol is not None else (128 if is_grm else nelem_start_2)
        _n_disc_FV_cfg = n_disc_FV_override if n_disc_FV_override is not None else n_disc_FV_2
        _fv_ti = fv_ti_override if fv_ti_override is not None else ti
        fv_name = f'{prefix.replace("_DG_", "_FV_WENO3_")}'

        # Compute finest FV cell count and generate reference filename
        finest_ncol = _fv_start_2_cfg * 2**(_n_disc_FV_cfg - 1)
        fv_finest_names[cfg_idx] = convergence.generate_1D_name(
            fv_name, 0, finest_ncol)

        addition_fv = {
            'cadet_config_jsons': [base_model_fv],
            'cadet_config_names': [fv_name],
            'include_sens': [False],
            'ref_files': [[None]],
            'unit_IDs': ['001'],
            'which': ['outlet'],
            'idas_abstol': [[None]],
            'ax_methods': [[0]],
            'ax_discs': [[bench_func.disc_list(_fv_start_2_cfg, _n_disc_FV_cfg)]],
            'par_methods': [[None]],
            'par_discs': [[None]],
            'disc_refinement_functions': [
                [partial(refine_FV_WENO3,
                         setting_name=fv_name,
                         nCol_start=_fv_start_2_cfg,
                         equivolume=True,
                         time_integrator=_fv_ti,
                         refine_par=is_grm)]
            ],
        }

        bench_configs.add_benchmark(
            fv_configs, fv_include_sens, fv_ref_files, fv_unit_IDs, fv_which,
            fv_ax_methods, fv_ax_discs,
            par_methods=fv_par_methods, par_discs=fv_par_discs,
            idas_abstol=fv_idas_abstol,
            cadet_config_names=fv_config_names, addition=addition_fv,
            disc_refinement_functions=fv_disc_refinement_functions)

    if _run(2):
      try:
        if not study2_skip_fv:
            # Run all FV sweeps in parallel
            print("\n--- Study 2: Phase 1 — FV WENO3 reference sweep ---")
            bench_func.run_convergence_analysis(
                output_path=output_path,
                cadet_path=cadet_path,
                cadet_configs=fv_configs,
                cadet_config_names=fv_config_names,
                include_sens=fv_include_sens,
                ref_files=fv_ref_files,
                unit_IDs=fv_unit_IDs,
                which=fv_which,
                ax_methods=fv_ax_methods,
                ax_discs=fv_ax_discs,
                par_methods=fv_par_methods,
                par_discs=fv_par_discs,
                idas_abstol=fv_idas_abstol,
                n_jobs=n_jobs,
                rerun_sims=True,
                disc_refinement_functions=fv_disc_refinement_functions
            )

        # --- Phase 2: Run DG sweep with finest FV as reference ---

        print("\n--- Study 2: Phase 2 — DG convergence against FV reference ---")

        for cfg_idx, (setting_mod, prefix, poly_degs, n_disc_DG, ti, is_grm, _nfv, _fvti, _cim) in enumerate(configs_2):
            base_model = setting_mod.get_model()
            # DG uses consistent_init_mode=1 (mode 5 is only for FV SMA)
            if _cim is not None:
                base_model['input']['solver']['consistent_init_mode'] = 1
            ref_file = fv_finest_names[cfg_idx]

            _poly_degs = [p for p in poly_degs if study2_polydegs is None or p in study2_polydegs]
            if not _poly_degs:
                continue

            dg_configs = []
            dg_config_names = []
            dg_include_sens = []
            dg_ref_files = []
            dg_unit_IDs = []
            dg_which = []
            dg_idas_abstol = []
            dg_ax_methods = []
            dg_ax_discs = []
            dg_par_methods = []
            dg_par_discs = []
            dg_disc_refinement_functions = []

            methods = []
            for p in _poly_degs:
                methods.append((f'{prefix}_P{p}', p))

            _nelem_start = nelem_start_2_GRM if is_grm else nelem_start_2
            addition = {
                'cadet_config_jsons': [base_model] * len(methods),
                'cadet_config_names': [name for name, _ in methods],
                'include_sens': [False] * len(methods),
                'ref_files': [[ref_file] for _ in methods],
                'unit_IDs': ['001'] * len(methods),
                'which': ['outlet'] * len(methods),
                'idas_abstol': [[None] for _ in methods],
                'ax_methods': [[p] for _, p in methods],
                'ax_discs': [[bench_func.disc_list(_nelem_start, n_disc_DG)] for _ in methods],
                'par_methods': [[None] for _ in methods],
                'par_discs': [[None] for _ in methods],
                'disc_refinement_functions': [
                    [partial(refine_DG,
                             setting_name=name,
                             polyDeg=polyDeg,
                             node_type='CGL',
                             nelem_start=_nelem_start,
                             time_integrator=ti,
                             refine_par=is_grm)]
                    for name, polyDeg in methods
                ],
            }

            bench_configs.add_benchmark(
                dg_configs, dg_include_sens, dg_ref_files, dg_unit_IDs, dg_which,
                dg_ax_methods, dg_ax_discs,
                par_methods=dg_par_methods, par_discs=dg_par_discs,
                idas_abstol=dg_idas_abstol,
                cadet_config_names=dg_config_names, addition=addition,
                disc_refinement_functions=dg_disc_refinement_functions)

            bench_func.run_convergence_analysis(
                output_path=output_path,
                cadet_path=cadet_path,
                cadet_configs=dg_configs,
                cadet_config_names=dg_config_names,
                include_sens=dg_include_sens,
                ref_files=dg_ref_files,
                unit_IDs=dg_unit_IDs,
                which=dg_which,
                ax_methods=dg_ax_methods,
                ax_discs=dg_ax_discs,
                par_methods=dg_par_methods,
                par_discs=dg_par_discs,
                idas_abstol=dg_idas_abstol,
                n_jobs=n_jobs,
                rerun_sims=not study2_skip_dg_rerun,
                disc_refinement_functions=dg_disc_refinement_functions
            )

      except Exception:
            print(f"\n*** Study 2 FAILED ***\n{traceback.format_exc()}")

    #
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

    poly_degs_3 = [3, 4, 5] if not small_test else [1, 2]
    if study3_polydegs is not None:
        poly_degs_3 = [p for p in poly_degs_3 if p in study3_polydegs]

    # Breuer's thesis ranges (Tables B.10/B.11) + one extra refinement level
    if not small_test:
        _dg_disc_3 = {
            'D1e4': {3: [8, 16, 32, 64, 128, 256, 512],
                     4: [8, 16, 32, 64, 128, 256],
                     5: [8, 16, 32, 64, 128, 256]},
            'D1e5': {3: [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                     4: [8, 16, 32, 64, 128, 256, 512, 1024],
                     5: [8, 16, 32, 64, 128, 256, 512, 1024]},
        }
        _fv_disc_3 = {
            'D1e4': [64, 128, 256, 512, 1024, 2048, 4096, 8192],
            'D1e5': [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        }
    else:
        _dg_disc_3 = {
            'D1e4': {p: [8, 16, 32, 64] for p in [1, 2, 3, 4, 5]},
            'D1e5': {p: [8, 16, 32, 64] for p in [1, 2, 3, 4, 5]},
        }
        _fv_disc_3 = {
            'D1e4': [64, 128, 256, 512],
            'D1e5': [64, 128, 256, 512],
        }

    _disp_cases = [(1e-4, 'D1e4'), (1e-5, 'D1e5')]
    if study3_dispersions is not None:
        _disp_cases = [_disp_cases[i] for i in study3_dispersions if i < len(_disp_cases)]

    _run_dg_3 = study3_methods is None or 'DG' in study3_methods
    _run_fv_3 = study3_methods is None or 'FV' in study3_methods

    for D0, label in _disp_cases:
        base_model_lang = setting_DG_LRM_lang.get_model(D0=D0)

        # DG methods — each polyDeg gets its own disc list
        if _run_dg_3 and poly_degs_3:
            for p in poly_degs_3:
                dg_discs = _dg_disc_3[label].get(p)
                if dg_discs is None:
                    continue
                name = f'radLRM_DG_lang_2comp_{label}_P{p}'

                addition = {
                    'cadet_config_jsons': [base_model_lang],
                    'cadet_config_names': [name],
                    'include_sens': [False],
                    'ref_files': [[None]],
                    'unit_IDs': ['001'],
                    'which': ['outlet'],
                    'idas_abstol': [[None]],
                    'ax_methods': [[p]],
                    'ax_discs': [[dg_discs]],
                    'par_methods': [[None]],
                    'par_discs': [[None]],
                    'disc_refinement_functions': [
                        [partial(refine_DG,
                                 setting_name=name,
                                 polyDeg=p,
                                 node_type='CGL',
                                 nelem_start=dg_discs[0],
                                 time_integrator=time_integrator_strict)]
                    ],
                }

                bench_configs.add_benchmark(
                    cadet_configs, include_sens, ref_files, unit_IDs, which,
                    ax_methods, ax_discs,
                    par_methods=par_methods, par_discs=par_discs,
                    idas_abstol=idas_abstol,
                    cadet_config_names=cadet_config_names, addition=addition,
                    disc_refinement_functions=disc_refinement_functions)

        # FV WENO3 equivolume
        if _run_fv_3:
            fv_discs = _fv_disc_3[label]
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
                'ax_discs': [[fv_discs]],
                'par_methods': [[None]],
                'par_discs': [[None]],
                'disc_refinement_functions': [
                    [partial(refine_FV_WENO3,
                             setting_name=fv_name,
                             nCol_start=fv_discs[0],
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

    if _run(3):
        try:
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
        except Exception:
            print(f"\n*** Study 3 FAILED ***\n{traceback.format_exc()}")
