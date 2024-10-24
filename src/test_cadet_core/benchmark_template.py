# -*- coding: utf-8 -*-
"""
Created May 2024

This script implements template benchmarks. We recommend to also check out other
benchmark branches.

@author: jmbr
"""

import re
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed

from cadetrdm import ProjectRepo

import bench_func


database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
    "/-/raw/master/cadet_config/test_cadet-core/chromatography/"

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "chromatography"
os.makedirs(output_path, exist_ok=True)

### Setting the cadet_path is optional and only required if a source buil is used
cadet_path = "C:/Users/jmbr/Cadet_testBuild/CADET_PRaddDGtests/out/install/aRELEASE/bin/cadet-cli.exe" # None

### CADET-RDM can be used to track the results
# commit_message = f"Full rerun of CADET-Core test simulations"
# with project_repo.track_results(results_commit_message=commit_message, debug=True):

#%% Single method, Single model specification

n_jobs = -1

cadet_config_jsons = [
    'configuration_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json'
]

ax_methods = [[3]]

ax_discs = [[bench_func.disc_list(4, 4)]]

par_methods = [[3]]

par_discs = [[bench_func.disc_list(1, 4)]]

include_sens = [True]

ref_files = [[None]]

unit_IDs = ['001']
which = ['outlet']

idas_abstol = [[1e-8]]

bench_func.run_convergence_analysis(
        database_path = database_path, output_path=output_path,
        cadet_path = cadet_path,
        cadet_config_jsons = cadet_config_jsons,
        include_sens = include_sens,
        ref_files = ref_files,
        unit_IDs = unit_IDs,
        which = which,
        ax_methods = ax_methods,
        ax_discs = ax_discs,
        par_methods = par_methods,
        par_discs = par_discs,
        idas_abstol = idas_abstol,
        n_jobs = n_jobs,
        rerun_sims=True
        )

#%% Multiple methods, Single model specification

n_jobs = -1

cadet_config_jsons = [
    'configuration_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json'
]

ax_methods = [[0, 3, 4]]

ax_discs = [[bench_func.disc_list(8, 4),
             bench_func.disc_list(1, 4),
             bench_func.disc_list(1, 4)]]

par_methods = [[0, 3, 3]] # Note that FV and DG cannot be combined
# par_methods = [[None, None, None]] # For non-GRM models

par_discs = [[
    bench_func.disc_list(1, 4),
    bench_func.disc_list(1, 4),
    bench_func.disc_list(1, 4)
    ]]

include_sens = [False]

ref_files = [[None, None, None]]

unit_IDs = ['001']
which = ['outlet']

idas_abstol = [[1e-8, 1e-8, 1e-8]]

#%% Single method, Multiple model specification

### The following parameters must be set in the following manner, in order to
### use the convergence analysis function (called at the bottom of this script)

n_jobs = -1

method = 3
startDisc = 1
nDiscs = 6
parStartDisc = 1

cadet_config_jsons = [
    'configuration_LRM_dynLin_1comp_sensbenchmark1_FV_Z256.json',
    'configuration_LRMP_dynLin_1comp_sensbenchmark1_FV_Z32.json'
]

include_sens = [False, False]

ref_files = [[None], [None]]

unit_IDs = ['001', '001']
which = ['outlet', 'outlet']

idas_abstol = [[1e-8], [1e-8]]

ax_methods = [[method], [method]]

ax_discs = [
    [bench_func.disc_list(startDisc, nDiscs)],
    [bench_func.disc_list(startDisc, nDiscs)]]

par_methods = [[None], [None]]

par_discs = [[[None]*nDiscs]]


#%% Different methods and models specification.
### Note that two methods are used for the first model and only one for
### the second one

### The following parameters must be set in the following manner, in order to
### use the convergence analysis function (called at the bottom of this script)

cadet_config_jsons = [
    'configuration_LRM_dynLin_1comp_sensbenchmark1_FV_Z256.json',
    'configuration_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json'
]

include_sens = [True, True]

ref_files = [[None, None], [None]]

unit_IDs = ['001', '001']
which = ['outlet', 'outlet']

idas_abstol = [[1e-10, 1e-10], [1e-10]]

ax_methods = [
    [0, 3],
    [0]
]

ax_discs = [
    [bench_func.disc_list(8, 3), bench_func.disc_list(1, 3)],
    [bench_func.disc_list(8, 3)]
]

par_methods = [
    [None, None],
    [0]
]

par_discs = [
    [None, None],
    [bench_func.disc_list(1, 3)]
]

#%% Convergence analysis function call

bench_func.run_convergence_analysis(
        database_path = database_path, output_path=output_path,
        cadet_path = cadet_path,
        cadet_config_jsons = cadet_config_jsons,
        include_sens = include_sens,
        ref_files = ref_files,
        unit_IDs = unit_IDs,
        which = which,
        ax_methods = ax_methods,
        ax_discs = ax_discs,
        par_methods = par_methods,
        par_discs = par_discs,
        idas_abstol = idas_abstol,
        n_jobs = n_jobs,
        rerun_sims=True
        )
