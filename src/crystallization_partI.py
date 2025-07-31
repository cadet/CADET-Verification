# -*- coding: utf-8 -*-
"""
Created Juli 2024

This script implements the EOC tests to verify the population balance model (PBM),
which is implemented in CADET-Core. The tests encompass all combinations of the
PBM terms such as nucleation, growth and growth rate dispersion. Further, the
incorporation of the PBM into a DPFR transport model is tested.

@author: wendi zhang (original draft) and jmbr (incorporation to CADET-Verification)
"""

#%% Include packages
from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import UnivariateSpline
import json

from cadet import Cadet
from cadetrdm import ProjectRepo

import bench_func
from benchmark_models import settings_crystallization


#%% Helper functions

# seed function
# A: area, y0: offset, w:std, xc: center (A,w >0)
def log_normal(x, y0, A, w, xc):
    return y0 + A/(np.sqrt(2.0*np.pi) * w*x) * np.exp(-np.log(x/xc)**2 / 2.0/w**2)

# Note: n_x is the total number of component = FVM cells - 2

def calculate_relative_L1_norm(predicted, analytical, x_grid):
    if (len(predicted) != len(analytical)) or (len(predicted) != len(x_grid)-1):
        raise ValueError(f'The size of the input arrays are wrong, got {len(predicted), len(analytical), len(x_grid)-1}')
    
    x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, len(x_grid))]
    
    area = trapezoid(analytical, x_ct)

    L1_norm = 0.0
    for i in range (0, len(predicted)):
        L1_norm += np.absolute(predicted[i] - analytical[i]) * (x_grid[i+1]-x_grid[i])
        
    return L1_norm/area

def get_slope(error):
    return -np.array([np.log2(error[i] / error[i-1]) for i in range (1, len(error))])

def get_EOC_simTimes(N_x_ref, N_x_test, target_model, xmax, cadet_path, output_path): 
    
    ## get ref solution
    
    model = target_model(N_x_ref, cadet_path, output_path)
    model.save()
    data = model.run_simulation()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    model.load_from_file()

    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]

    ## interpolate the reference solution

    x_grid = np.logspace(np.log10(1e-6), np.log10(xmax), N_x_ref - 1) 
    x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, N_x_ref-1)]

    spl = UnivariateSpline(x_ct, c_x_reference)

    ## EOC
    
    n_xs = []   ## store the result nx here
    sim_times = []
    for Nx in N_x_test:
        model = target_model(Nx, cadet_path, output_path)
        model.save()
        data = model.run_simulation()
        if not data.return_code == 0:
            print(data.error_message)
            raise Exception(f"simulation failed")
        model.load_from_file() 

        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,1:-1])
        sim_times.append(model.root.meta.time_sim)

    relative_L1_norms = []  ## store the relative L1 norms here
    for nx in n_xs:
        ## interpolate the ref solution on the test case grid
        
        x_grid = np.logspace(np.log10(1e-6), np.log10(xmax), len(nx) + 1)
        x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, len(nx)+1)]

        relative_L1_norms.append(calculate_relative_L1_norm(nx, spl(x_ct), x_grid))

    slopes = get_slope(relative_L1_norms) ## calculate slopes
    
    return np.array(slopes), sim_times


def CSTR_PBM_growth_EOC_test(small_test, output_path, cadet_path):
        
    N_x_ref = 800 + 2 if small_test else 2000
    ## grid for EOC
    N_x_test_c1 = [50, 100, 200, 400] if small_test else [50, 100, 200, 400, 800, 1600, ]
    N_x_test_c1 = np.array(N_x_test_c1) + 2
    
    EOC_c1, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c1, settings_crystallization.CSTR_PBM_growth,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_growth EOC:\n", EOC_c1)
    
    data = {
        "Nx" : N_x_test_c1.tolist(),
        "EOC" : EOC_c1.tolist(),
        "time_sim" : simTimes
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_growth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
def CSTR_PBM_growthSizeDep_EOC_test(small_test, output_path, cadet_path):
    
    N_x_ref = 700 + 2 if small_test else 1000 + 2
    
    N_x_test_c2 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c2 = np.asarray(N_x_test_c2) + 2
    
    EOC_c2, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c2, settings_crystallization.CSTR_PBM_growthSizeDep,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_growthSizeDep EOC:\n", EOC_c2)
    
    data = {
        "Nx" : N_x_test_c2.tolist(),
        "EOC" : EOC_c2.tolist(),
        "time_sim" : simTimes
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_growthSizeDep.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    

def CSTR_PBM_primaryNucleationAndGrowth_EOC_test(small_test, output_path, cadet_path):
    
    N_x_ref = 700 + 2 if small_test else 1000 + 2
    
    N_x_test_c3 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c3 = np.asarray(N_x_test_c3) + 2
    
    EOC_c3, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c3,
        settings_crystallization.CSTR_PBM_primaryNucleationAndGrowth,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_primaryNucleationAndGrowth EOC:\n", EOC_c3)
    
    data = {
        "Nx" : N_x_test_c3.tolist(),
        "EOC" : EOC_c3.tolist(),
        "time_sim" : simTimes
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_primaryNucleationAndGrowth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    
def CSTR_PBM_primarySecondaryNucleationAndGrowth_EOC_test(small_test, output_path, cadet_path):
    
    N_x_ref = 700 + 2 if small_test else 1000 + 2
    
    N_x_test_c4 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c4 = np.asarray(N_x_test_c4) + 2
    
    EOC_c4, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c4,
        settings_crystallization.CSTR_PBM_primarySecondaryNucleationAndGrowth,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_primaryNucleationAndGrowth EOC:\n", EOC_c4)
    
    data = {
        "Nx" : N_x_test_c4.tolist(),
        "EOC" : EOC_c4.tolist(),
        "time_sim" : simTimes
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_primarySecondaryNucleationAndGrowth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    

def CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion_EOC_test(
        small_test, output_path, cadet_path):
    
    N_x_ref = 700 + 2 if small_test else 1000 + 2
    
    N_x_test_c5 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c5 = np.asarray(N_x_test_c5) + 2
    
    EOC_c5, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c5,
        settings_crystallization.CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion EOC:\n", EOC_c5)
    
    data = {
        "Nx" : N_x_test_c5.tolist(),
        "EOC" : EOC_c5.tolist(),
        "time_sim" : simTimes
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
def DPFR_PBM_primarySecondaryNucleationGrowth_EOC_test(
        small_test, output_path, cadet_path):
    # This is a special case, we have Nx and Ncol
    # Here we test EOC long each coordinate
    
    N_x_ref   = 120 if small_test else 200 + 2 # very fine reference: 500 + 2
    N_col_ref = 120 if small_test else 200 # very fine reference: 500
    
    x_max = 900e-6 # um
    
    ## get ref solution
        
    model = settings_crystallization.DPFR_PBM_primarySecondaryNucleationGrowth(N_x_ref, N_col_ref, cadet_path, output_path)
    model.save()
    data = model.run_simulation()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    model.load_from_file() 
    
    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]
    
    ## interpolate the reference solution at the reactor outlet
    
    x_grid = np.logspace(np.log10(1e-6), np.log10(x_max), N_x_ref - 1) 
    x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, N_x_ref-1)]
    
    spl = UnivariateSpline(x_ct, c_x_reference)
    
    ## EOC, Nx
    
    N_x_test_c6 = [20, 40, 80, ] if small_test else [20, 40, 80, 160, ] # very fine grid: [25, 50, 100, 200, 400, ]
    N_x_test_c6 = np.asarray(N_x_test_c6) + 2
    
    n_xs = []   ## store the result nx here
    simTimesIntRefinement = []
    for Nx in N_x_test_c6:
        model = settings_crystallization.DPFR_PBM_primarySecondaryNucleationGrowth(Nx, N_col_ref, cadet_path, output_path)
        model.save()
        data = model.run_simulation()
        if not data.return_code == 0:
            print(data.error_message)
            raise Exception(f"simulation failed")
        model.load_from_file() 
    
        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,1:-1])
        simTimesIntRefinement.append(model.root.meta.time_sim)
    
    relative_L1_norms = []  ## store the relative L1 norms here
    for nx in n_xs:
        ## interpolate the ref solution on the test case grid
    
        x_grid = np.logspace(np.log10(1e-6), np.log10(900e-6), len(nx) + 1)
        x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, len(nx)+1)]
    
        relative_L1_norms.append(calculate_relative_L1_norm(nx, spl(x_ct), x_grid))
    
    slopes_Nx = get_slope(relative_L1_norms) ## calculate slopes
    print("DPFR_PBM_primarySecondaryNucleationGrowth EOC in internal coordinate:\n", slopes_Nx)
    
    ## EOC, Ncol
    
    N_col_test_c6 = [20, 40, 80, ] if small_test else [20, 40, 80, 160, ] # very fine grid: [25, 50, 100, 200, 400, ]   ## grid for EOC
    N_col_test_c6 = np.asarray(N_col_test_c6)
    
    n_xs = []   ## store the result nx here
    simTimesAxRefinement = []
    for Ncol in N_col_test_c6:
        model = settings_crystallization.DPFR_PBM_primarySecondaryNucleationGrowth(N_x_ref+2, Ncol, cadet_path, output_path)
        model.save()
        data = model.run_simulation()
        if not data.return_code == 0:
            print(data.error_message)
            raise Exception(f"simulation failed")
        model.load_from_file() 
    
        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,1:-1])
    
    relative_L1_norms = []  ## store the relative L1 norms here
    for nx in n_xs:
        ## interpolate the ref solution on the test case grid
    
        x_grid = np.logspace(np.log10(1e-6), np.log10(900e-6), len(nx) + 1)
        x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, len(nx)+1)]
    
        relative_L1_norms.append(calculate_relative_L1_norm(nx, spl(x_ct), x_grid))
        simTimesAxRefinement.append(model.root.meta.time_sim)
    
    slopes_Ncol = get_slope(relative_L1_norms) ## calculate slopes
    print(slopes_Ncol)
    
    print("DPFR_PBM_primarySecondaryNucleationGrowth EOC in axial direction:\n", slopes_Ncol)
    data = {
        "Convergence in axial direction" : {
        "Ncol" : N_col_test_c6.tolist(),
        "EOC" : slopes_Ncol.tolist(),
        "time_sim" : simTimesAxRefinement
        },
        "Convergence in internal coordinate" : {
        "Nx" : N_x_test_c6.tolist(),
        "EOC" : slopes_Nx.tolist(),
        "time_sim" : simTimesIntRefinement
        }
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/DPFR_PBM_primarySecondaryNucleationGrowth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

