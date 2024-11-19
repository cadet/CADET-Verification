# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:29:55 2024

@author: jmbr
"""

import utility.convergence as convergence
import re
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

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

commit_message = f"Run 2D FV pure axial flow benchmark"
with project_repo.track_results(results_commit_message=commit_message, debug=True):

    
    os.makedirs(output_path, exist_ok=True)
    n_jobs = 1

    # %% Pure axial flow benchmark
    
    # Compute reference solution
    
    ref_name = "/ref_1DLRMP_linBnd_1Comp_FV_Z512.h5"
    
    config1D = settings_2Dchromatography.SamDiss_2DVerificationSetting(
        0, 512,
        0, 0,
        0, 0,
        nRadialZones=1,
        USE_MODIFIED_NEWTON=0,
        idas_tolerance=1e-12,
        plot=False, run=True,
        save_path=str(output_path),
        cadet_path=cadet_path,
        file_name=ref_name,
        transport_model="LUMPED_RATE_MODEL_WITH_PORES", # LUMPED_RATE_MODEL_WITH_PORES GENERAL_RATE_MODEL
        export_json_config=True,
        **{
            'WRITE_SOLUTION_OUTLET' : 1,
            'WRITE_SOLUTION_BULK' : 1,
            'WRITE_SOLUTION_PARTICLE' : 1,
            'WRITE_SOLUTION_SOLID' : 1,
            'SPLIT_PORTS_DATA' : 1,
            'WRITE_SOLUTION_FLUX' : 1,
            # 'adsorption_model' : 'NONE',
            # 'film_diffusion' : 0.0
        }
    )
    
    ref_outlet_LRMP = convergence.get_outlet(str(output_path)+ref_name, '000')
    ref_flux_LRMP = convergence.get_solution(str(output_path)+ref_name, unit='unit_000', which='flux')
    
    
    ref_name = "/ref_1DGRM_linBnd_1Comp_FV_Z512.h5"
    
    config1D = settings_2Dchromatography.SamDiss_2DVerificationSetting(
        0, 512,
        0, 0,
        0, 3,
        nRadialZones=1,
        USE_MODIFIED_NEWTON=0,
        idas_tolerance=1e-12,
        plot=False, run=True,
        save_path=str(output_path),
        cadet_path=cadet_path,
        file_name=ref_name,
        transport_model="GENERAL_RATE_MODEL", # LUMPED_RATE_MODEL_WITH_PORES GENERAL_RATE_MODEL
        export_json_config=True,
        **{
            'WRITE_SOLUTION_OUTLET' : 1,
            'WRITE_SOLUTION_BULK' : 1,
            'WRITE_SOLUTION_PARTICLE' : 1,
            'WRITE_SOLUTION_SOLID' : 1,
            'SPLIT_PORTS_DATA' : 1,
            'WRITE_SOLUTION_FLUX' : 1,
            # 'adsorption_model' : 'NONE',
            # 'par_diffusion' : 0.0,
            # 'par_surfdiffusion' : 1.0,
            # 'film_diffusion' : 1e3
        }
    )
    
    ref_outlet_GRM = convergence.get_outlet(str(output_path)+ref_name, '000')
    ref_flux_GRM = convergence.get_solution(str(output_path)+ref_name, unit='unit_000', which='flux')
    
    time = convergence.get_solution_times(str(output_path)+ref_name)
    
    
    abs_deviation = np.abs(ref_outlet_GRM - ref_outlet_LRMP)
    print(np.max(abs_deviation))
    
    plt.plot(time, ref_outlet_LRMP, label='LRMP', linestyle='dashed')
    plt.plot(time, ref_outlet_GRM, label='GRM')
    plt.legend()
    plt.show()
    
    
    plt.plot(time, ref_flux_LRMP[:, 1], label='LRMP flux', linestyle='dashed')
    plt.legend()
    plt.show()
    
    
    plt.plot(time, ref_flux_GRM[:, 1], label='GRM flux')
    plt.legend()
    plt.show()

    
    
    

    
    
