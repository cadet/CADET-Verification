# -*- coding: utf-8 -*-
"""
Created May 2025

This script defines the chromatography sensitivity CADET-Verification tests for CADET-Core.

@author: jmbr
"""

# %% Include packages
import os
import sys
from pathlib import Path
import re
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

from cadet import Cadet
from cadetrdm import ProjectRepo

import bench_func
import bench_configs
from utility import convergence

# %% Run with CADET-RDM

def chromatography_sensitivity_tests(
        n_jobs, database_path, small_test, output_path, cadet_path):

    os.makedirs(output_path, exist_ok=True)

    Cadet.cadet_path = cadet_path

    # Define settings and benchmarks

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

    addition = bench_configs.radial_flow_benchmark(
        database_path+r'radial/', small_test=True, sensitivities=True)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        cadet_config_names=cadet_config_names, addition=addition)

    addition = bench_configs.sensitivity_benchmark(
        database_path, spatial_method="FV", small_test=small_test
        )

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        cadet_config_names=cadet_config_names, addition=addition)
    
    addition = bench_configs.sensitivity_benchmark(
        database_path, spatial_method="DG", small_test=small_test
        )

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        cadet_config_names=cadet_config_names, addition=addition)

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
        rerun_sims=True
    )
    
    
    #%% Plot the sensitivities for easy assessment
    
    
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
    
    addition = bench_configs.radial_flow_benchmark(
        database_path+r'radial/', small_test=True, sensitivities=True)
    
    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        cadet_config_names=cadet_config_names, addition=addition)
    
    addition = bench_configs.sensitivity_benchmark(
        database_path, spatial_method="FV", small_test=small_test
        )

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        cadet_config_names=cadet_config_names, addition=addition)
    
    addition = bench_configs.sensitivity_benchmark(
        database_path, spatial_method="DG", small_test=small_test
        )

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        cadet_config_names=cadet_config_names, addition=addition)
    
    for modelIdx in range(len(cadet_configs)):
    
        if par_methods[modelIdx][-1] is None:
            
            settingName = convergence.generate_1D_name(
                cadet_config_names[modelIdx],
                ax_methods[modelIdx][-1], ax_discs[modelIdx][0][-1]
                )
            
        else:
            
            settingName = convergence.generate_GRM_name(cadet_config_names[modelIdx],
            ax_methods[modelIdx][-1], ax_discs[modelIdx][0][-1],
            par_methods[modelIdx][-1], par_discs[modelIdx][0][-1])
            
        
        name = output_path + '/' + settingName
        simDict = convergence.get_simulation(name).root
        nSens = convergence.sim_go_to(simDict,
                  ['input', 'sensitivity', 'nsens'])
        
        for sensIdx in range(nSens):
            
            sensIdxStr = str(sensIdx).zfill(3)
            sensName = convergence.sim_go_to(simDict,
                      ['input', 'sensitivity', f'param_{sensIdxStr}', 'sens_name'])
            sensUnit = str(convergence.sim_go_to(simDict,
                      ['input', 'sensitivity', f'param_{sensIdxStr}', 'sens_unit'])).zfill(3)
            
            sensitivity = convergence.get_solution(
                name, unit=f'unit_{sensUnit}', which='sens_outlet', **{'sensIdx': sensIdx})
            
            if sensitivity.ndim == 2: # happens for multicomponent sensitivities
                sensitivity = sensitivity[:, -1]
            
            plt.plot(convergence.get_solution_times(name), sensitivity, label=sensName)
            plt.legend()
            plt.title(settingName)
            plt.savefig(name + '.png')
            plt.show()
            