# -*- coding: utf-8 -*-
"""
Created December 2024

@author: Jesper Frandsen and Jan Breuer
"""


import numpy as np
import os

from cadet import Cadet

import utility.convergence as convergence
import bench_configs
import bench_func


# %% Define chromatography system tests


def chromatography_systems_tests(n_jobs, database_path, small_test,
                                 output_path, cadet_path,
                                 analytical_reference=True,
                                 reference_data_path=None):

    os.makedirs(output_path, exist_ok=True)

    if analytical_reference and reference_data_path is None:
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
        n_jobs, database_path, output_path, cadet_path, small_test=small_test,
        analytical_reference=analytical_reference)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol,
        ax_methods, ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        addition=addition)
    
    if analytical_reference:
        ref = convergence.get_solution(
            reference_data_path+'/ref_cyclicSystem1_LRMP_linBnd_1comp.h5', unit='unit_'+unit_IDs[0])
        ref_files = [[ref]]

    config_names = ['cyclicSystem1_LRMP_linBnd_1comp']

    # %% run convergence analysis

    Cadet.cadet_path = cadet_path

    bench_func.run_convergence_analysis(
        database_path=database_path, output_path=output_path,
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
        analytical_reference=analytical_reference
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
        n_jobs, database_path, output_path, cadet_path, small_test=small_test,
        analytical_reference=analytical_reference)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol,
        ax_methods, ax_discs,
        par_methods=par_methods, par_discs=par_discs,
        addition=addition)

    if analytical_reference:
     # we compare the simulated outlet of unit 006 with the analytical
     # solution of the combined outlets of unit 004 and 005
        ref = convergence.get_solution(
            reference_data_path+'/ref_acyclicSystem1_LRMP_linBnd_1comp.h5', unit='unit_004')
        ref = 0.5 * ref + 0.5 * convergence.get_solution(
            reference_data_path+'/ref_acyclicSystem1_LRMP_linBnd_1comp.h5', unit='unit_005')
        ref_files = [[ref]]

    config_names = ['acyclicSystem1_LRMP_linBnd_1comp']

    # %% run convergence analysis

    Cadet.cadet_path = cadet_path

    bench_func.run_convergence_analysis(
        database_path=database_path, output_path=output_path,
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
        analytical_reference=analytical_reference
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
        n_jobs, database_path, output_path, cadet_path, small_test=small_test)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which, idas_abstol,
        ax_methods, ax_discs, par_methods=par_methods, par_discs=par_discs,
        addition=addition)

    config_names = ["SMBsystem1_LRM_linBnd_2comp_"]

    # %% Run convergence analysis

    Cadet.cadet_path = cadet_path

    bench_func.run_convergence_analysis(
        database_path=database_path, output_path=output_path,
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
