# -*- coding: utf-8 -*-
"""
Created 2024

This script creates reference data for the MCT tests in CADET-Core.

@author: jmbr
""" 

#%% Include packages
import os
import sys
from pathlib import Path

from cadet import Cadet
from cadetrdm import ProjectRepo

import bench_func as bf

#%% user definitions

rdm_debug_mode = True # Run cadet-rdm in debug mode to test if the script works

small_test = True # small test set (less numerical refinement steps)

n_jobs = -1 # for parallelization on the number of simulations

database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
    "/-/raw/core_tests/cadet_config/test_cadet-core/mct/"

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "mct"
os.makedirs(output_path, exist_ok=True)

# specify a source build cadet_path and make sure the commit hash is visible
cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master7_preV5Commit_21c653\aRELEASE\bin\cadet-cli.exe"
Cadet.cadet_path = cadet_path

commit_message = f"Recreation of MCT reference files" 

# %% Run with CADET-RDM

with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):

    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_LRM_dynLin_1comp_MCTbenchmark.json',
        output_path=str(output_path)
        )
    model.run()
    model.load()
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_LRM_noBnd_1comp_MCTbenchmark.json',
        output_path=str(output_path)
        )
    model.run()
    model.load()
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_MCT1ch_noEx_noReac_benchmark1.json',
        output_path=str(output_path)
        )
    model.run()
    model.load()
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_MCT1ch_noEx_reac_benchmark1.json',
        output_path=str(output_path)
        )
    model.run()
    model.load()
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_MCT2ch_oneWayEx_reac_benchmark1.json',
        output_path=str(output_path)
        )
    model.run()
    model.load()
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_MCT3ch_twoWayExc_reac_benchmark1.json',
        output_path=str(output_path)
        )
    model.run()
    model.load()
    model.save()
