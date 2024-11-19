# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:15:24 2024

@author: jmbr
"""

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
database_path = None # TODO


sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "2D_chromatography"

# specify a source build cadet_path
cadet_path = r"C:\Users\jmbr\Cadet_testBuild\CADET_PR2DmodelsDG\out\install\aRELEASE\bin\cadet-cli.exe"
Cadet.cadet_path = cadet_path

commit_message = f"Benchmark 2DGRM FV radial inlet variance convergence"
with project_repo.track_results(results_commit_message=commit_message, debug=False):

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
    ref_files = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    rad_methods = []
    rad_discs = []
    par_methods = []
    par_discs = []
    
    rad_inlet_profile=None
    def rad_inlet_profile(r, r_max):
        return np.sin(r / r_max * np.pi) + 0.1 # r * 100 #
        # return 1.0 # np.sin(r / r_max * (0.5 * np.pi) + 0.25 * np.pi)
    
    def fv2D_noRadFlowBenchmark(small_test=False, **kwargs):

        benchmark_config = {
            'cadet_config_jsons': [
                settings_2Dchromatography.SamDiss_2DVerificationSetting(
                    # film_diffusion=0.0,
                    # nRadialZones=3,
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
                [bench_func.disc_list(4, 8 if not small_test else 6)]
            ],
            'rad_methods': [
                [0]
            ],
            'rad_discs': [
                [bench_func.disc_list(3, 8 if not small_test else 6)]
            ],
            'par_methods': [
                [0]
            ],
            'par_discs': [
                [[1] * 8 if not small_test else [1] * 6]
            ]
        }

        return benchmark_config
    
    
    # %% Pure axial flow benchmark
    
    addition = fv2D_noRadFlowBenchmark(small_test=small_test)
    
    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol,
        ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
        par_methods=par_methods, par_discs=par_discs,
        addition=addition)

    config_names.extend(["2DLRMP_noBnd_1Comp"])
    
    # %% Run convergence analysis
    
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
        rerun_sims=True
    )
