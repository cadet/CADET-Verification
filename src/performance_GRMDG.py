# -*- coding: utf-8 -*-
"""
Created Oct 2024

This script executes all the CADET-Verification tests for CADET-Core.
Modify the input in the 'User Input' section if needed.
To test if the script works, specify rdm_debug_mode and small_test as true.

Only specify rdm_debug_mode as False if you are sure that this run shall be
saved to the output repository!

@author: jmbr
""" 
  
#%% Include packages
import os
import sys
from pathlib import Path
import re
from joblib import Parallel, delayed
import numpy as np
import json
import matplotlib.pyplot as plt

from cadet import Cadet
from cadetrdm import ProjectRepo

import utility.convergence as convergence
import bench_func
import bench_configs

import chromatography
import chromatography_sensitivities

#%% User Input

commit_message = f"Sensitivity test run to generate baseline data"

rdm_debug_mode = 1 # Run CADET-RDM in debug mode to test if the script works

small_test = 0 # Defines a smaller test set (less numerical refinement steps)

run_chromatography_tests = 0
run_chromatography_sensitivity_tests = 1

n_jobs = -1 # For parallelization on the number of simulations

delete_h5_files = True # delete h5 files (but keep convergence tables and plots)
exclude_files = None # ["file1", "file2"] # specify h5 files that should not be deleted

database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
    "/-/raw/core_tests/cadet_config/test_cadet-core/"

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()



# %% Run convergence tests

with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):
    
    cadet_path2 = r"C:\Users\jmbr\Cadet_testBuild\CADET_2DmodelsDG\out\install\aRELEASE\bin\cadet-cli.exe"
    output_path2 = project_repo.output_path / "performance/cadet-coreDev"
    
    if run_chromatography_tests:
        
        chromatography.chromatography_tests(
            n_jobs=n_jobs, database_path=database_path+"chromatography/",
            small_test=small_test, sensitivities=True,
            output_path=str(output_path2) + "/chromatography", cadet_path=cadet_path2
            )

        if delete_h5_files:
            convergence.delete_h5_files(str(output_path2) + "/chromatography", exclude_files=exclude_files)
            
    if run_chromatography_sensitivity_tests:
                
        chromatography_sensitivities.chromatography_sensitivity_tests(
            n_jobs=n_jobs, database_path=database_path+"chromatography/", small_test=small_test,
            output_path=str(output_path2) + "/chromatography/sensitivity", cadet_path=cadet_path2
            )
    
        if delete_h5_files:
            convergence.delete_h5_files(str(output_path2) + "/chromatography/sensitivity", exclude_files=exclude_files)




    cadet_path1 = convergence.get_cadet_path() # path to root folder of bin\cadet-cli 
    output_path1 = project_repo.output_path / "performance/cadet-coreV5"
    
    if run_chromatography_tests:
        
        chromatography.chromatography_tests(
            n_jobs=n_jobs, database_path=database_path+"chromatography/",
            small_test=small_test, sensitivities=True,
            output_path=str(output_path1) + "/chromatography", cadet_path=cadet_path1
            )

        if delete_h5_files:
            convergence.delete_h5_files(str(output_path1) + "/chromatography", exclude_files=exclude_files)
            
    if run_chromatography_sensitivity_tests:
                
        chromatography_sensitivities.chromatography_sensitivity_tests(
            n_jobs=n_jobs, database_path=database_path+"chromatography/", small_test=small_test,
            output_path=str(output_path1) + "/chromatography/sensitivity", cadet_path=cadet_path1
            )
    
        if delete_h5_files:
            convergence.delete_h5_files(str(output_path1) + "/chromatography/sensitivity", exclude_files=exclude_files)
            


# %% Plot
    def compare_convergence_data(filename, id1, filepath1, id2, filepath2):
        
        with open(filepath1+'/'+filename+'.json', 'r') as file:
            data1 = json.load(file)

        with open(filepath2+'/'+filename+'.json', 'r') as file:
            data2 = json.load(file)
        
        if 'convergence' not in data1.keys():
            raise Exception(f'no convergence field for {filename}, {id1}')
        if 'convergence' not in data2.keys():
            raise Exception(f'no convergence field for {filename}, {id2}')
        
        data1 = data1['convergence']
        data2 = data2['convergence']
        
        if data1.keys() != data2.keys():
            raise Exception(f'Different keys found in convergence data of {filename} for {id1} and {id2}')
        
        for key in data1.keys():
            
            plt.plot(data1[key]['outlet']['Sim. time'], data1[key]['outlet']['Max. error'],
                     linestyle='solid', marker='o', label=id1)
            plt.plot(data2[key]['outlet']['Sim. time'], data2[key]['outlet']['Max. error'],
                     linestyle='dotted', marker='o', label=id2)
            plt.legend()
            plt.ylabel('Max. error')
            plt.xlabel('Sim. time (s)')
            plt.yscale('log')
            #plt.xscale('log')
            plt.title(filename + ', ' + key)
            plt.savefig(str(project_repo.output_path)+"/performance/"+filename+'.png')
            plt.show()
            
            
            

    path1=str(output_path1)+r"\chromatography\sensitivity"
    path2=str(output_path2)+r"\chromatography\sensitivity"
    
    json_files1 = [os.path.splitext(f)[0] for f in os.listdir(path1) if f.endswith('.json')]
    json_files2 = [os.path.splitext(f)[0] for f in os.listdir(path2) if f.endswith('.json')]

    if not json_files1 == json_files2:
        raise Exception(f'different convergence json files in the two paths.')
        
    for filename in json_files1:
        
        compare_convergence_data(
            filename=filename,
            id1="v5", filepath1=path1, id2="dev", filepath2=path2
            )
