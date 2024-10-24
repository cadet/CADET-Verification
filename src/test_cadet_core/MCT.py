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

# %% Run with CADET-RDM

def MCT_tests(n_jobs, database_path, small_test,
              output_path, cadet_path):

    os.makedirs(output_path, exist_ok=True)
    
    Cadet.cadet_path = cadet_path
    
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
