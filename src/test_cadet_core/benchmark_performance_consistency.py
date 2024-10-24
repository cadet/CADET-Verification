# -*- coding: utf-8 -*-
"""
Created Juli 8 2024

Implements a template benchmark for performance consistency between two cadet
commits using CADET-RDM.
This script requires an cadet installation of the two codes.
Please adjust the variables in the next section below to modify

@author: jmbr
"""

import json
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

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()

# %% configuration

# Specifies if you want a CADET-RDM debug mode run
rdm_debug = True

# Add the first 8 digits of the commit hash of the two commits you want to compare
benchmark_cases = [
    'f1b44765',
    'a17024ae'
    ]

# Specify the paths to the cadet-cli of the respective commits
commit_name1 = f'EigenSolvers_commit_{benchmark_cases[0]}'
commit_name2 = f'master4_MMkinetics_Commit_{benchmark_cases[1]}'
_cadet_path1 = f"C:/Users/jmbr/OneDrive/Desktop/CADET_compiled/{commit_name1}/aRELEASE/bin/cadet-cli.exe"
_cadet_path2 = f"C:/Users/jmbr/OneDrive/Desktop/CADET_compiled/{commit_name2}/aRELEASE/bin/cadet-cli.exe"


# N_RUNS > 1 will safe the lowest compute time for each simulation to the convergence table
N_RUNS = [10, 10]

# Specify the benchmark settings you want to run for this benchmark.
# You can add your own in the bench_configs.py.
benchmark_settings = 'full_chromatography_benchmark'

commit_message = f"Performance consistency run with commits {benchmark_cases[0]} and {benchmark_cases[1]} for {benchmark_settings}" 

output_path = project_repo.output_path / "benchmarks" / "performance_consistency"

 # for some benchmarks, weve specified the extensiveness (wrt number of refinement steps) of the benchmarks.
 # options are mid, large, small
benchmark_size='mid'

# number of threads to be used, note that -x specifies all threads - x
n_jobs = -1

# %%
# Benchmark models must be specified in the database_path directory, we use
# the Cadet-Database github project to store our standard setups.
database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
    "/-/raw/core_tests/cadet_config/test_cadet-core/chromatography/"
os.makedirs(output_path, exist_ok=True)


# %% run simulations and convergence analysis
method = 0
with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug):

    for benchmark_case_idx in range(len(benchmark_cases)):

        linear_solver = benchmark_cases[benchmark_case_idx]
        output_path_modified = output_path / benchmark_cases[benchmark_case_idx]

        if benchmark_case_idx == 0:
            cadet_path = _cadet_path1
        else:
            cadet_path = _cadet_path2

        convergence_results_list = bench_configs.run_benchmark(
            benchmark_settings,
            [method],
            cadet_path=cadet_path, output_path=output_path_modified,
            database_path=database_path,
            benchmark_size=benchmark_size, include_sensitivity=False,
            ref_files=None,
            n_jobs=-1, 
            N_RUNS=N_RUNS[benchmark_case_idx]
        )

    # %% plots

    import matplotlib.pyplot as plt

    subDirs = benchmark_cases
    save_fig = True
    file_path_prefix = str(output_path)

    for convergence_results in convergence_results_list:

        cmap = plt.get_cmap('gist_rainbow') # viridis # gist_rainbow
        
        # Generate a color for each directory
        colors = [cmap(i / len(subDirs)) for i in range(len(subDirs))]        

        print(convergence_results)

        plot_args = {'shape': [10, 10],
                      'y_label': '$L^\infty$ error in mol $ m^{-3}$'}
        plt.rcParams["figure.figsize"] = (10, 10)

        plot_args['title'] = convergence_results

        for subDirIdx in range(len(subDirs)):

            try:
                with open(file_path_prefix+'\\'+subDirs[subDirIdx]+'\\'+convergence_results, 'r') as json_file:
                    data = json.load(json_file)['convergence']
            except FileNotFoundError:
                data = {}
    
            if method == 0:
                data = data['FV']
            elif 'DG_P'+str(method)+'parP'+str(method) in data:
                data = data['DG_P'+str(method)+'parP'+str(method)]
            else:
                data = data['DG_P'+str(method)]
    
            convergence.std_plot(data['outlet']['Sim. time'],
                                  data['outlet']['Max. error'],
                                  label=subDirs[subDirIdx],
                                  color=colors[subDirIdx])
            print(subDirs[subDirIdx])
            print(data['outlet']['Sim. time'])

        plot_args['x_label'] = 'Compute time in seconds'
        convergence.std_plot_prep(**plot_args)
        if save_fig:
            plt.savefig(file_path_prefix +
                        '\\performance_consistency' +
                        re.search(
                            r'convergence_(.*?)(?:.json)', convergence_results).group(1)
                        + '.png',
                        bbox_inches='tight')
        plt.show()
