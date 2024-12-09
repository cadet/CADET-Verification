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
import utility.convergence as convergence
import matplotlib.pyplot as plt
import re

# %% Run with CADET-RDM

def MCT_tests(n_jobs, database_path, small_test,
              output_path, cadet_path):

    os.makedirs(output_path, exist_ok=True)
    
    convergence.std_plot_prep(benchmark_plot=False, linewidth=10,
                              x_label='time / s', y_label='mol / $m^{-3}$')
    
    Cadet.cadet_path = cadet_path
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_LRM_dynLin_1comp_MCTbenchmark.json',
        output_path=str(output_path)
        )
    data = model.run()
    model.load()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_LRM_noBnd_1comp_MCTbenchmark.json',
        output_path=str(output_path)
        )
    data = model.run()
    model.load()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_MCT1ch_noEx_noReac_benchmark1.json',
        output_path=str(output_path)
        )
    data = model.run()
    model.load()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    
    sol_name = model.filename
    time = convergence.get_solution_times(sol_name)
    channel1 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_000')

    plt.plot(time, channel1, label='channel 1')
    plt.legend(fontsize=20)
    plt.savefig(re.sub(".h5", ".png", sol_name), dpi=100, bbox_inches='tight')
    
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_MCT1ch_noEx_reac_benchmark1.json',
        output_path=str(output_path)
        )
    data = model.run()
    model.load()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    
    sol_name = model.filename
    time = convergence.get_solution_times(sol_name)
    channel1 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_000')

    plt.plot(time, channel1, label='channel 1')
    plt.legend(fontsize=20)
    plt.savefig(re.sub(".h5", ".png", sol_name), dpi=100, bbox_inches='tight')
    
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_MCT2ch_oneWayEx_reac_benchmark1.json',
        output_path=str(output_path)
        )
    data = model.run()
    model.load()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    
    sol_name = model.filename
    time = convergence.get_solution_times(sol_name)
    channel1 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_000')
    channel2 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_001')

    plt.plot(time, channel1, label='channel 1')
    plt.plot(time, channel2, label='channel 2')
    plt.legend(fontsize=20)
    plt.savefig(re.sub(".h5", ".png", sol_name), dpi=100, bbox_inches='tight')
    
    model.save()
    
    model = bf.create_object_from_database(
        database_path,
        cadet_config_json_name='configuration_MCT3ch_twoWayExc_reac_benchmark1.json',
        output_path=str(output_path)
        )
    data = model.run()
    model.load()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    
    sol_name = model.filename
    time = convergence.get_solution_times(sol_name)
    channel1 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_000')
    channel2 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_001')
    channel3 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_002')

    plt.plot(time, channel1, label='channel 1')
    plt.plot(time, channel2, label='channel 2')
    plt.plot(time, channel3, label='channel 3')
    plt.legend(fontsize=20)
    plt.savefig(re.sub(".h5", ".png", sol_name), dpi=100, bbox_inches='tight')
    
    model.save()
    
