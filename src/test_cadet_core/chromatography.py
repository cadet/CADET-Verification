# -*- coding: utf-8 -*-
"""
Created May 2024

This script executes the chromatography CADET-Verification tests for CADET-Core.
Modify the input in the 'user definitions' section if needed.

@author: jmbr
""" 

#%% Include packages
import os
import sys
from pathlib import Path
import re
from joblib import Parallel, delayed
import numpy as np

from cadet import Cadet
from cadetrdm import ProjectRepo

import utility.convergence as convergence
import bench_func
import bench_configs

#%% user definitions

rdm_debug_mode = True # Run cadet-rdm in debug mode to test if the script works

small_test = True # small test set (less numerical refinement steps)

n_jobs = -1 # for parallelization on the number of simulations

database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
    "/-/raw/core_tests/cadet_config/test_cadet-core/chromatography/"

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "chromatography"
os.makedirs(output_path, exist_ok=True)

# specify a source build cadet_path and make sure the commit hash is visible
cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master7_preV5Commit_21c653\aRELEASE\bin\cadet-cli.exe"
Cadet.cadet_path = cadet_path

commit_message = f"Rerun of CADET-Core chromatography verification simulations" 

# %% Run with CADET-RDM

with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):
    
    # Define settings and benchmarks
    
    cadet_config_jsons = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    par_methods = []
    par_discs = []
    
    addition = bench_configs.radial_flow_benchmark(small_test=small_test)
    
    bench_configs.add_benchmark(
        cadet_config_jsons, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        addition=addition)
    
    addition = bench_configs.fv_benchmark(small_test=small_test)
    
    bench_configs.add_benchmark(
        cadet_config_jsons, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        addition=addition)
    
    addition = bench_configs.dg_benchmark(small_test=small_test)
    
    bench_configs.add_benchmark(
        cadet_config_jsons, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        addition=addition)
    
    # Run convergence analysis
    
    bench_func.run_convergence_analysis(
        database_path=database_path, output_path=output_path,
        cadet_path=cadet_path,
        cadet_config_jsons=cadet_config_jsons,
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
