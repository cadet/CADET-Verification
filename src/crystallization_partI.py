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
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import UnivariateSpline
import json

from cadet import Cadet

import src.bench_func as bench_func
from src.benchmark_models import settings_crystallization


#%% Helper functions

reference_data_path = str(Path(__file__).resolve().parent.parent / 'data' / 'CADET-Core_reference' / 'crystallization')

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
    if type(N_x_ref) is int:
        model = target_model(N_x_ref, cadet_path, output_path)
        model.save()
        data = model.run_simulation()
        if not data.return_code == 0:
            print(data.error_message)
            raise Exception(f"simulation failed")
        model.load_from_file()
    
        c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]
    elif type(N_x_ref) is str:
        model = Cadet()
        model.filename = N_x_ref
        model.load_from_file()
        c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]
        N_x_ref = model.root.input.model.unit_001.ncomp
    else:
        raise Exception("N_x_ref must be filename (string) or number of discrete points (int)")

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
        
    N_x_ref = reference_data_path + '/ref_CSTR_PBM_growth.h5'
    ## grid for EOC
    N_x_test_c1 = [50, 100, 200, 400] if small_test else [50, 100, 200, 400, 800, 1600, ]
    N_x_test_c1 = np.array(N_x_test_c1) + 2
    
    EOC_c1, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c1, settings_crystallization.CSTR_PBM_growth,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_growth EOC:\n", EOC_c1)
    
    data = {
        "convergence" : {
            "Convergence in internal coordinate": {
                "Nx" : N_x_test_c1.tolist(),
                "EOC" : EOC_c1.tolist(),
                "Sim. time" : simTimes
                }
            }
        }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_growth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
def CSTR_PBM_growthSizeDep_EOC_test(small_test, output_path, cadet_path):
    
    N_x_ref = reference_data_path + '/ref_CSTR_PBM_growthSizeDep.h5'
    
    N_x_test_c2 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c2 = np.asarray(N_x_test_c2) + 2
    
    EOC_c2, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c2, settings_crystallization.CSTR_PBM_growthSizeDep,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_growthSizeDep EOC:\n", EOC_c2)
    
    data = {
        "convergence" : {
            "Convergence in internal coordinate": {
                "Nx" : N_x_test_c2.tolist(),
                "EOC" : EOC_c2.tolist(),
                "Sim. time" : simTimes
                }
            }
        }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_growthSizeDep.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    

def CSTR_PBM_primaryNucleationAndGrowth_EOC_test(small_test, output_path, cadet_path):
    
    N_x_ref = reference_data_path + '/ref_CSTR_PBM_primaryNucleationAndGrowth.h5'
    
    N_x_test_c3 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c3 = np.asarray(N_x_test_c3) + 2
    
    EOC_c3, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c3,
        settings_crystallization.CSTR_PBM_primaryNucleationAndGrowth,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_primaryNucleationAndGrowth EOC:\n", EOC_c3)
    
    data = {
        "convergence" : {
            "Convergence in internal coordinate": {
                "Nx" : N_x_test_c3.tolist(),
                "EOC" : EOC_c3.tolist(),
                "Sim. time" : simTimes
                }
            }
        }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_primaryNucleationAndGrowth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    
def CSTR_PBM_primarySecondaryNucleationAndGrowth_EOC_test(small_test, output_path, cadet_path):
    
    N_x_ref = reference_data_path + '/ref_CSTR_PBM_primarySecondaryNucleationAndGrowth.h5'
    
    N_x_test_c4 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c4 = np.asarray(N_x_test_c4) + 2
    
    EOC_c4, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c4,
        settings_crystallization.CSTR_PBM_primarySecondaryNucleationAndGrowth,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_primarySecondaryNucleationAndGrowth EOC:\n", EOC_c4)
    
    data = {
        "convergence" : {
            "Convergence in internal coordinate": {
                "Nx" : N_x_test_c4.tolist(),
                "EOC" : EOC_c4.tolist(),
                "Sim. time" : simTimes
                }
            }
        }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_primarySecondaryNucleationAndGrowth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    

def CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion_EOC_test(
        small_test, output_path, cadet_path):
    
    N_x_ref = reference_data_path + '/ref_CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion.h5'
    
    N_x_test_c5 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c5 = np.asarray(N_x_test_c5) + 2
    
    EOC_c5, simTimes = get_EOC_simTimes(
        N_x_ref, N_x_test_c5,
        settings_crystallization.CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion,
        1000e-6, cadet_path, output_path
        )
    
    print("CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion EOC:\n", EOC_c5)
    
    data = {
        "convergence" : {
            "Convergence in internal coordinate": {
                "Nx" : N_x_test_c5.tolist(),
                "EOC" : EOC_c5.tolist(),
                "Sim. time" : simTimes
                }
            }
        }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
def DPFR_PBM_primarySecondaryNucleationGrowth_EOC_test(
        small_test, output_path, cadet_path):
    # This is a special case, we have Nx and Ncol
    # Here we test EOC long each coordinate
    
    x_max = 900e-6 # um
    
    ## get ref solution
    
    model = Cadet()
    model.filename = reference_data_path + '/ref_DPFR_PBM_primarySecondaryNucleationAndGrowth.h5'
    model.load_from_file()
    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]
    N_x_ref = model.root.input.model.unit_001.ncomp
    N_col_ref = model.root.input.model.unit_001.discretization.ncol
    
    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]
    
    ## interpolate the reference solution at the reactor outlet
    
    x_grid = np.logspace(np.log10(1e-6), np.log10(x_max), N_x_ref - 1) 
    x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, N_x_ref-1)]
    
    spl = UnivariateSpline(x_ct, c_x_reference)
    
    # Compute convergence for joint refinement of internal and external coordinate
    
    N_x_test_c6 = [20, 40, 80, ] if small_test else [20, 40, 80, 160, ]
    N_col_test_c6 = [20, 40, 80, ] if small_test else [20, 40, 80, 160, ]
    N_x_test_c6 = np.asarray(N_x_test_c6) + 2
    N_col_test_c6 = np.asarray(N_col_test_c6)
    
    n_xs = []   ## store the result nx here
    simTimesIntRefinement = []
    for i in range(0, len(N_x_test_c6)):
        model = settings_crystallization.DPFR_PBM_primarySecondaryNucleationGrowth(N_x_test_c6[i], N_col_test_c6[i], cadet_path, output_path)
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
    
    slopes = get_slope(relative_L1_norms) ## calculate slopes
    print("DPFR_PBM_primarySecondaryNucleationGrowth L1 normalized error, refinement in both coordinates:\n", relative_L1_norms)
    print("DPFR_PBM_primarySecondaryNucleationGrowth EOC, refinement in both coordinates:\n", slopes)
    
    # Only for extended/long test run, we compute convergence of external and internal coordinate independent of each other
    if small_test:
        return
    
    ## EOC, Nx
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
    
    relative_L1_norms_Nx = []  ## store the relative L1 norms here
    for nx in n_xs:
        ## interpolate the ref solution on the test case grid
    
        x_grid = np.logspace(np.log10(1e-6), np.log10(900e-6), len(nx) + 1)
        x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, len(nx)+1)]
    
        relative_L1_norms_Nx.append(calculate_relative_L1_norm(nx, spl(x_ct), x_grid))
    
    slopes_Nx = get_slope(relative_L1_norms_Nx) ## calculate slopes
    print("DPFR_PBM_primarySecondaryNucleationGrowth L1 normalized error in internal coordinate:\n", relative_L1_norms_Nx)
    print("DPFR_PBM_primarySecondaryNucleationGrowth EOC in internal coordinate:\n", slopes_Nx)
    
    ## EOC, Ncol
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
    
    relative_L1_norms_Ncol = []  ## store the relative L1 norms here
    for nx in n_xs:
        ## interpolate the ref solution on the test case grid
    
        x_grid = np.logspace(np.log10(1e-6), np.log10(900e-6), len(nx) + 1)
        x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, len(nx)+1)]
    
        relative_L1_norms_Ncol.append(calculate_relative_L1_norm(nx, spl(x_ct), x_grid))
        simTimesAxRefinement.append(model.root.meta.time_sim)
    
    slopes_Ncol = get_slope(relative_L1_norms_Ncol) ## calculate slopes
    
    print("DPFR_PBM_primarySecondaryNucleationGrowth L1 normalized error in axial coordinate:\n", relative_L1_norms_Ncol)
    print("DPFR_PBM_primarySecondaryNucleationGrowth EOC in axial direction:\n", slopes_Ncol)
    data = {
        "convergence" : {
        "Convergence in axial direction":
            {
            "Ncol" : N_col_test_c6.tolist(),
            "L1 error normalized by L1 norm of reference" : relative_L1_norms_Ncol,
            "EOC" : slopes_Ncol.tolist(),
            "Sim. time" : simTimesAxRefinement
            },
        "Convergence in internal coordinate" : {
            "Nx" : N_x_test_c6.tolist(),
            "L1 error normalized by L1 norm of reference" : relative_L1_norms_Nx,
            "EOC" : slopes_Nx.tolist(),
            "Sim. time" : simTimesIntRefinement
            }
        }
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/DPFR_PBM_primarySecondaryNucleationGrowth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

