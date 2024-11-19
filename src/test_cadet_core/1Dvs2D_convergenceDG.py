# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:37:16 2024

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


################################ UNTERSUCHUNG ##################################################
#
# 2D LRMP == 1D GRM if particle diffusion is FAST and no radial dispersion, radial inlet profile
#
################################################################################################


# database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
#     "/-/raw/core_tests/cadet_config/test_cadet-core/chromatography/"
database_path = None # TODO


sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "2D_chromatography"

# specify a source build cadet_path
cadet_path = r"C:\Users\jmbr\Cadet_testBuild\CADET_PR2DmodelsDG\out\install\aRELEASE\bin\cadet-cli.exe"
Cadet.cadet_path = cadet_path

commit_message = f"Run 2D DG pure axial flow benchmark"
with project_repo.track_results(results_commit_message=commit_message, debug=True):

    
    os.makedirs(output_path, exist_ok=True)
    n_jobs = 1
    
    # %% Define benchmarks
    
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
    
    def dg2D_pureFlowBenchmark(small_test=False, **kwargs):

        benchmark_config = {
            'cadet_config_jsons': [
                settings_2Dchromatography.SamDiss_2DVerificationSetting(
                    # film_diffusion=0.0,
                    nRadialZones=1,
                    USE_MODIFIED_NEWTON=1, axMethod=1, **kwargs)
            ],
            'include_sens': [
                False
            ],
            'ref_files': [
                [kwargs.get('reference', None)]
            ],
            'unit_IDs': [
                '000'
            ],
            'which': [
                'outlet_port_000' # radial_outlet outlet_port_000
            ],
            'idas_abstol': [
                [1e-10]
            ],
            'ax_methods': [
                [1]
            ],
            'ax_discs': [
                [bench_func.disc_list(4, 6 if not small_test else 5)]
            ],
            'rad_methods': [
                [2]
            ],
            'rad_discs': [
                [[1] * 6] if not small_test else [[1] * 5]
            ],
            'par_methods': [
                [None]
            ],
            'par_discs': [
                [None]
            ]
        }

        return benchmark_config
    
    
    # %% Pure axial flow benchmark
    
    # Compute reference solution
    
    ref_name = "/ref_1DLRMP_linBnd_1Comp_DG_P5Z64.h5"
    
    config1D = settings_2Dchromatography.SamDiss_2DVerificationSetting(
        5, 256,
        1, 0,
        1, 1,
        nRadialZones=1,
        idas_tolerance=1e-12,
        plot=False, run=True,
        save_path=str(output_path),
        cadet_path=cadet_path,
        file_name=ref_name,
        transport_model="GENERAL_RATE_MODEL", # LUMPED_RATE_MODEL_WITH_PORES
        export_json_config=True,
        **{
            'WRITE_SOLUTION_OUTLET' : 1,
            'SPLIT_PORTS_DATA' : 1,
            'par_diffusion' : 1.0
            # 'film_diffusion' : 0.0
        }
    )
    
    ref_outlet = convergence.get_outlet(str(output_path)+ref_name, '000')
    
    
    addition = dg2D_pureFlowBenchmark(small_test=small_test, reference=np.array(ref_outlet))
    
    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol,
        ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
        par_methods=par_methods, par_discs=par_discs,
        addition=addition)

    config_names.extend(["2DLRMP_linBnd_1Comp"])
    
    # Run convergence analysis
    
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
