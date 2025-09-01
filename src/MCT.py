# -*- coding: utf-8 -*-
"""

This script creates reference data for the MCT tests in CADET-Core.

""" 

#%% Include packages
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import re
import json

from cadet import Cadet
from cadetrdm import ProjectRepo

import src.bench_func as bf
import src.utility.convergence as convergence

# %% Run with CADET-RDM

def MCT_tests(n_jobs, small_test,
              output_path, cadet_path):

    os.makedirs(output_path, exist_ok=True)
    
    convergence.std_plot_prep(benchmark_plot=False, linewidth=10,
                              x_label='time / s', y_label='mol / $m^{-3}$')
    
    json_path = Path(__file__).resolve().parent / "benchmark_models" / "configuration_LRM_dynLin_1comp_MCTbenchmark.json"
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f) 
    
    model = bf.create_object_from_config(
        config, 'LRM_dynLin_1comp_MCTbenchmark',
        output_path=str(output_path)
        )
    
    model.install_path = cadet_path
    data = model.run_simulation()
    model.load_from_file()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    model.save()
    
    json_path = Path(__file__).resolve().parent / "benchmark_models" / "configuration_LRM_noBnd_1comp_MCTbenchmark.json"
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f) 
    
    model = bf.create_object_from_config(
        config, 'LRM_noBnd_1comp_MCTbenchmark',
        output_path=str(output_path)
        )
    data = model.run_simulation()
    model.load_from_file()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    model.save()
    
    json_path = Path(__file__).resolve().parent / "benchmark_models" / "configuration_MCT1ch_noEx_noReac_benchmark1.json"
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f) 
    
    model = bf.create_object_from_config(
        config, 'MCT1ch_noEx_noReac_benchmark1',
        output_path=str(output_path)
        )
    data = model.run_simulation()
    model.load_from_file()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    
    sol_name = model.filename
    time = convergence.get_solution_times(sol_name)
    channel1 = convergence.get_solution(sol_name, unit='unit_001', which='outlet')

    plt.figure()
    plt.plot(time, channel1, label='channel 1')
    plt.legend(fontsize=20)
    plt.savefig(re.sub(".h5", ".png", sol_name), dpi=100, bbox_inches='tight')
    plt.close()
    
    model.save()
    
    json_path = Path(__file__).resolve().parent / "benchmark_models" / "configuration_MCT1ch_noEx_reac_benchmark1.json"
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f) 
    
    model = bf.create_object_from_config(
        config, 'MCT1ch_noEx_reac_benchmark1',
        output_path=str(output_path)
        )
    data = model.run_simulation()
    model.load_from_file()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    
    sol_name = model.filename
    time = convergence.get_solution_times(sol_name)
    channel1 = convergence.get_solution(sol_name, unit='unit_001', which='outlet')

    plt.figure()
    plt.plot(time, channel1, label='channel 1')
    plt.legend(fontsize=20)
    plt.savefig(re.sub(".h5", ".png", sol_name), dpi=100, bbox_inches='tight')
    plt.close()
    
    model.save()
    
    json_path = Path(__file__).resolve().parent / "benchmark_models" / "configuration_MCT2ch_oneWayEx_reac_benchmark1.json"
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f) 
    
    model = bf.create_object_from_config(
        config, 'MCT2ch_oneWayEx_reac_benchmark1',
        output_path=str(output_path)
        )
    data = model.run_simulation()
    model.load_from_file()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    
    sol_name = model.filename
    time = convergence.get_solution_times(sol_name)
    channel1 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_000')
    channel2 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_001')

    plt.figure()
    plt.plot(time, channel1, label='channel 1')
    plt.plot(time, channel2, label='channel 2')
    plt.legend(fontsize=20)
    plt.savefig(re.sub(".h5", ".png", sol_name), dpi=100, bbox_inches='tight')
    plt.close()
    
    model.save()
    
    json_path = Path(__file__).resolve().parent / "benchmark_models" / "configuration_MCT3ch_twoWayExc_reac_benchmark1.json"
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f) 
    
    model = bf.create_object_from_config(
        config, 'MCT3ch_twoWayExc_reac_benchmark1',
        output_path=str(output_path)
        )
    data = model.run_simulation()
    model.load_from_file()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    
    sol_name = model.filename
    time = convergence.get_solution_times(sol_name)
    channel1 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_000')
    channel2 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_001')
    channel3 = convergence.get_solution(sol_name, unit='unit_001', which='outlet_port_002')

    plt.figure()
    plt.plot(time, channel1, label='channel 1')
    plt.plot(time, channel2, label='channel 2')
    plt.plot(time, channel3, label='channel 3')
    plt.legend(fontsize=20)
    plt.savefig(re.sub(".h5", ".png", sol_name), dpi=100, bbox_inches='tight')
    plt.close()
    
    model.save()
    
