# -*- coding: utf-8 -*-
"""
Created May 2024

This script creates numerical references for the tests in CADET-Core

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

database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
    "/-/raw/core_tests/cadet_config/test_cadet-core/chromatography/"

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "chromatography"
os.makedirs(output_path, exist_ok=True)

# specify a source build cadet_path
cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master6_Cpp23StandardCommit_e4c3373d\aRELEASE\bin\cadet-cli.exe"
Cadet.cadet_path = cadet_path

# commit_message = f"Full rerun of CADET-Core test simulations"
# with project_repo.track_results(results_commit_message=commit_message, debug=True):

n_jobs = -1

# %% Define benchmarks

# small_test is set to true to define a minimal benchmark, which can be used
# to see if the simulations still run and see first results.
# To run the full extensive benchmarks, this needs to be set to false.

small_test = True

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

# %% Run convergence analysis

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
