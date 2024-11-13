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

commit_message = f"Run 2D DG radial inlet spectral convergence"
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
    # def rad_inlet_profile(r, r_max):
    #     return r * 100.0 + 0.1 # np.sin(r / r_max * np.pi) + 0.1
    #     # return 1.0 # np.sin(r / r_max * (0.5 * np.pi) + 0.25 * np.pi)
    
    def rad_init_conc(r):
        return 0.1 + r * 100.0
    
    def dg2D_noRadFlowBenchmark(small_test=False, **kwargs):

        benchmark_config = {
            'cadet_config_jsons': [
                settings_2Dchromatography.SamDiss_2DVerificationSetting(
                    film_diffusion=0.0,
                    nRadialZones=0,
                    USE_MODIFIED_NEWTON=1, axMethod=0, **kwargs)
            ],
            'include_sens': [
                False, False
            ],
            'ref_files': [
                [kwargs.get('reference', None), kwargs.get('reference', None)]
            ],
            'unit_IDs': [
                '000', '000'
            ],
            'which': [
                'radial_outlet' # radial_outlet # outlet_port_000
            ],
            'idas_abstol': [
                [1e-10, 1e-10]
            ],
            'ax_methods': [
                [0]#, 4]
            ],
            'ax_discs': [
                [bench_func.disc_list(8, 6 if not small_test else 5)]#, bench_func.disc_list(2, 6 if not small_test else 4)]
            ],
            'rad_methods': [
                [0]#, 4]
            ],
            'rad_discs': [
                [bench_func.disc_list(4, 6 if not small_test else 5)]#, bench_func.disc_list(1, 6 if not small_test else 4)]
            ],
            'par_methods': [
                [None, None]
            ],
            'par_discs': [
                [[None], [None]]
            ]
        }

        return benchmark_config

    
    # FV reference
    # ref_name = "/ref_1DLRMP_linBnd_1Comp_FV.h5"
    
    # config1D = settings_2Dchromatography.SamDiss_2DVerificationSetting(
    #     0, 1024,
    #     0, 10,
    #     0, 1,
    #     nRadialZones=1,
    #     idas_tolerance=1e-10,
    #     plot=False, run=True,
    #     save_path=str(output_path),
    #     cadet_path=cadet_path,
    #     file_name=ref_name,
    #     export_json_config=True,
    #     **{
    #         'WRITE_SOLUTION_OUTLET' : 1,
    #         # 'WRITE_SOLUTION_BULK' : 1,
    #         'SPLIT_PORTS_DATA' : 1,
    #         # 'par_diffusion' : 1.0
    #         'film_diffusion' : 0.0,
    #         'col_dispersion_radial' : 1e-6
    #     }
    # )
    
    # ref_outlet = []
    # coords = convergence.get_radial_coordinates(str(output_path)+ref_name, '000')
    # for rad in range(len(coords)):
    #     ref_outlet.append(convergence.get_solution(str(output_path)+ref_name, unit='unit_000', which='outlet_port_{:03d}'.format(rad)))
    
    # kwargs = { 'domain_end' : 0.035, 'ref_coords' : coords}
    kwargs = { 'radial_init_conc' : rad_init_conc}
    
    # %% Pure axial flow benchmark
    
    addition = dg2D_noRadFlowBenchmark(
        small_test=small_test,# reference=np.array(ref_outlet),
            col_dispersion_radial = 1e-6,
            )
    
    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol,
        ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
        par_methods=par_methods, par_discs=par_discs,
        addition=addition)

    config_names.extend(["2DLRMP_linBnd_1Comp"])
    
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
        rerun_sims=True,
        **kwargs
    )
