# -*- coding: utf-8 -*-
"""

This scirpt defines chromatography system tests

"""


import os
from pathlib import Path

import src.utility.convergence as convergence
import src.bench_configs as bench_configs
import src.bench_func as bench_func


# %% Reference data paths
_reference_data_path_ = str(
    Path(__file__).resolve().parent.parent / 'data' / 'CASEMA_reference'
)


# %% Define chromatography system tests


def chromatography_systems_tests(n_jobs, small_test,
                                 output_path, cadet_path,
                                 use_analytical_reference=True
                                 ):

    os.makedirs(output_path, exist_ok=True)

    if use_analytical_reference and _reference_data_path_ is None:
        raise ValueError(
            "Reference data path must be provided to test convergence towards analytical solution!")

    # %% create cyclic system benchmark configuration

    cadet_configs = []
    config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    par_methods = []
    par_discs = []

    addition = bench_configs.cyclic_systems_tests(
        n_jobs, output_path, cadet_path, small_test=small_test,
        use_analytical_reference=use_analytical_reference)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs,
        par_methods=par_methods, par_discs=par_discs, idas_abstol=idas_abstol,
        addition=addition)
    
    if use_analytical_reference:
        ref = convergence.get_solution(
            _reference_data_path_+'/cyclicSystem1_LRMP_linBnd_1comp.h5', unit='unit_'+unit_IDs[0])
        ref_files = [[ref]]

    config_names = ['cyclicSystem1_LRMP_linBnd_1comp']

    # %% run convergence analysis

    bench_func.run_convergence_analysis(
        output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rerun_sims=True,
        system_refinement_IDs=['001', '002'],
        use_analytical_reference=use_analytical_reference
    )

    # %% create acyclic system benchmark configuration

    cadet_configs = []
    config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    par_methods = []
    par_discs = []

    addition = bench_configs.acyclic_systems_tests(
        n_jobs, output_path, cadet_path, small_test=small_test,
        use_analytical_reference=use_analytical_reference)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs,
        par_methods=par_methods, par_discs=par_discs, idas_abstol=idas_abstol,
        addition=addition)

    if use_analytical_reference:
     # we compare the simulated outlet of unit 006 with the analytical
     # solution of the combined outlets of unit 004 and 005
        ref = convergence.get_solution(
            _reference_data_path_+'/acyclicSystem1_LRMP_linBnd_1comp.h5', unit='unit_004')
        ref = 0.5 * ref + 0.5 * convergence.get_solution(
            _reference_data_path_+'/acyclicSystem1_LRMP_linBnd_1comp.h5', unit='unit_005')
        ref_files = [[ref]]

    config_names = ['acyclicSystem1_LRMP_linBnd_1comp']

    # %% run convergence analysis
    
    bench_func.run_convergence_analysis(
        output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rerun_sims=True,
        system_refinement_IDs=['002', '003', '004', '005'],
        use_analytical_reference=use_analytical_reference
    )

    # %% create benchmark configuration

    cadet_configs = []
    config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    par_methods = []
    par_discs = []

    addition = bench_configs.smb_systems_tests(
        n_jobs, output_path, cadet_path, small_test=small_test)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs, par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        addition=addition)

    config_names = ["SMBsystem1_LRM_linBnd_2comp_"]

    # %% Run convergence analysis

    bench_func.run_convergence_analysis(
        output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rad_inlet_profile=None,
        rerun_sims=True,
        system_refinement_IDs=['004', '005', '006',
                               '007', '008', '009', '010', '011']
    )
