# -*- coding: utf-8 -*-
"""
Created in Jan 2025

This file contains the software verification code for the FV and DG implementation
of cyclic and acyclic systems. The results of this convergence analysis are
published in Frandsen et al.
    'High-Performance C++ and Julia solvers in CADET for weakly and strongly
    coupled continuous chromatography problems' (2025b)

@author: jmbr and Jesper Frandsen
"""
  
#%% Include packages
import os
import sys
from pathlib import Path
import json
import csv
from joblib import Parallel, delayed
import numpy as np

from cadet import Cadet
from cadetrdm import ProjectRepo

import utility.convergence as convergence
import bench_func
import bench_configs
import chrom_systems

#%% User Input

commit_message = f"Convergence test for the linear systems in Frandsen et al. (2025b)"

small_test = False # Defines a smaller test set (less numerical refinement steps)

n_jobs = -1 # For parallelization on the number of simulations

rdm_debug_mode = False # Run CADET-RDM in debug mode to test if the script works

delete_h5_files = True # Delete simulation files, i.e. only keep convergence data

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "paper" / "Frandsen_et_al_2025b"

# The get_cadet_path function searches for the cadet-cli. If you want to use a specific source build, please define the path below
cadet_path = convergence.get_cadet_path() # path to root folder of bin\cadet-cli 


# %% Run with CADET-RDM

with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):
    
    chrom_systems.chromatography_systems_tests(
        n_jobs=n_jobs, database_path=None,
        small_test=small_test,
        output_path=str(output_path), cadet_path=cadet_path,
        analytical_reference=True, reference_data_path=str(project_repo.output_path.parent / 'data/CASEMA_reference')
        )

    if delete_h5_files:
        convergence.delete_h5_files(str(output_path), exclude_files=None)
        
    # export convergence data to csv to be used in the latex project
    def json_to_csv(json_file, csv_file, subgroup_path, ignore_data):
        # Read the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)
    
        # Navigate to the specified subgroup path
        subgroup = data
        for key in subgroup_path:
            if key not in subgroup:
                raise KeyError(f"Key '{key}' not found in JSON data at path {' -> '.join(subgroup_path[:subgroup_path.index(key)+1])}.")
            subgroup = subgroup[key]
    
        # Check if the subgroup is a dictionary with lists as values
        if not isinstance(subgroup, dict) or not all(isinstance(value, list) for value in subgroup.values()):
            raise ValueError(f"Subgroup at path '{' -> '.join(subgroup_path)}' must be a dictionary with lists as values.")
    
        # Extract keys and corresponding lists
        all_keys = list(subgroup.keys())
        all_keys = [item for item in all_keys if item not in ignore_data]
    
        # Prepare rows for the CSV
        rows = []
        max_length = max(len(values) for values in subgroup.values())
        for i in range(max_length):
            row = []
            for key in all_keys:
                row.append(subgroup[key][i] if i < len(subgroup[key]) else "")  # Fill missing values with an empty string
            rows.append(row)
    
        # Write to the CSV file
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
    
            # Write header
            writer.writerow(all_keys)
    
            # Write data rows
            writer.writerows(rows)
    
    subgroup_path = ['convergence', 'DG_P3', 'outlet']  # Path to the subgroup in the JSON file
    # ignore_data not required since desired columns can be picked in latex
    ignore_data = []#['$N_d$', 'Min. value', 'DoF', 'Bulk DoF']
    
    json_file = str(output_path) + r"/convergence_acyclicSystem1_LRMP_linBnd_1comp.json"  # Input JSON file
    csv_file = str(output_path) + r"/convergence_acyclicSystem1_LRMP_linBnd_1comp.csv"  # Output CSV file
    json_to_csv(json_file, csv_file, subgroup_path, ignore_data)
    
    subgroup_path = ['convergence', 'DG_P2', 'outlet']  # Path to the subgroup in the JSON file
    json_file = str(output_path) + r"/convergence_cyclicSystem1_LRMP_linBnd_1comp.json"  # Input JSON file
    csv_file = str(output_path) + r"/convergence_cyclicSystem1_LRMP_linBnd_1comp.csv"  # Output CSV file
    json_to_csv(json_file, csv_file, subgroup_path, ignore_data)
    


