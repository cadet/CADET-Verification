# -*- coding: utf-8 -*-
"""
Created on Nov 2024

@author: jmbr
"""

#%%

import utility.convergence as convergence
import re
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np

from cadet import Cadet
from cadetrdm import ProjectRepo

import bench_func
import bench_configs

import settings_2Dchromatography

# database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
#     "/-/raw/core_tests/cadet_config/test_cadet-core/chromatography/"
database_path = None # TODO use database for model setup


sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "2D_chromatography"

# The get_cadet_path function searches for the cadet-cli. If you want to use a specific source build, please define the path below
cadet_path = convergence.get_cadet_path() # path to root folder of bin\cadet-cli 



commit_message = f"Benchmark 2DGRM FV 3-zone radial inlet variance convergence"


rad_inlet_profile=None
# def rad_inlet_profile(r, r_max):
#     return np.sin(r / r_max * np.pi) + 0.1 # r * 100 #
#     # return 1.0 # np.sin(r / r_max * (0.5 * np.pi) + 0.25 * np.pi)

setting = {
    # 'film_diffusion' : 0.0,
    # 'col_dispersion_radial' : 0.0,
    'analytical_reference' : True, # If set to true, solution time 0.0 is ignored since its not computed by the analytical solution (CADET-Semi-Analytic)
    'nRadialZones' : 3,
    'name' : '2DGRM3Zone_dynLin_1Comp', # 2DGRMsd_reqLin_1Comp # 2DGRM_dynLin_1Comp
    'adsorption_model' : 'LINEAR',
    'adsorption.is_kinetic' : 1,
    'surface_diffusion' : 0.0 # 1e-11 # 0.0
    }

rerun_sims=False

with project_repo.track_results(results_commit_message=commit_message, debug=True):

    os.makedirs(output_path, exist_ok=True)
    n_jobs = 1
    
    # %% Define benchmarks
    
    # small_test is set to true to define a minimal benchmark, which can be used
    # to see if the simulations still run and see first results.
    # To run the full extensive benchmarks, this needs to be set to false.
    
    small_test = True
    
    cadet_configs = []
    config_names = []
    include_sens = []
    ref_files = [['ref_2DGRM3Zone_dynLin_1Comp.h5']]
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    rad_methods = []
    rad_discs = []
    par_methods = []
    par_discs = []

    
    def fv2D_noRadFlowBenchmark(small_test=False, **kwargs):

        nDisc = 8 if not small_test else 6
        nRadialZones=kwargs.get('nRadialZones',3)
        
        benchmark_config = {
            'cadet_config_jsons': [
                settings_2Dchromatography.SamDiss_2DVerificationSetting(
                    radNElem=nRadialZones,
                    rad_inlet_profile=rad_inlet_profile,
                    USE_MODIFIED_NEWTON=0, axMethod=0, **kwargs)
            ],
            'include_sens': [
                False
            ],
            'ref_files': [
                [None]
            ],
            'unit_IDs': [
                '000'
            ],
            'which': [
                'radial_outlet' # radial_outlet # outlet_port_000
            ],
            'idas_abstol': [
                [1e-10]
            ],
            'ax_methods': [
                [0]
            ],
            'ax_discs': [
                [bench_func.disc_list(4, nDisc)]
            ],
            'rad_methods': [
                [0]
            ],
            'rad_discs': [
                [bench_func.disc_list(nRadialZones, nDisc)]
            ],
            'par_methods': [
                [0]
            ],
            'par_discs': [ # same number of particle cells as radial cells
                [bench_func.disc_list(nRadialZones, nDisc)]
            ]
        }

        return benchmark_config
    
    # %% create benchmark configuration
    
    addition = fv2D_noRadFlowBenchmark(small_test=small_test, **setting)
    
    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol,
        ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
        par_methods=par_methods, par_discs=par_discs,
        addition=addition)

    config_names.extend([setting['name']])
    
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
        rad_methods=rad_methods, rad_discs=rad_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rad_inlet_profile=rad_inlet_profile,
        rerun_sims=rerun_sims
    )
