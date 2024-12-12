# -*- coding: utf-8 -*-
"""
Created Juli 2024

This script executes the crystallization CADET-Verification tests for CADET-Core.
Modify the input in the 'user definitions' section if needed.

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


#%% Helper functions

# seed function
# A: area, y0: offset, w:std, xc: center (A,w >0)
def log_normal(x, y0, A, w, xc):
    return y0 + A/(np.sqrt(2.0*np.pi) * w*x) * np.exp(-np.log(x/xc)**2 / 2.0/w**2)

# n_x is the total number of component = FVM cells - 2

def PBM_CSTR_growth(n_x, output_path):

    # general settings

    # time
    time_resolution = 100
    cycle_time = 60*60*5  # s

    # feed
    c_feed = 2.0  # mg/ml
    c_eq = 1.2   # mg/ml

    # particle space
    x_c = 1e-6  # m
    x_max = 1000e-6  # m

    # create model
    model = Cadet()

    # Spacing
    x_grid = np.logspace(np.log10(x_c), np.log10(x_max), n_x-1)  # log grid
    x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range(1, n_x-1)]

    # Boundary conditions
    boundary_c = []
    for p in range(n_x):
        if p == 0:
            boundary_c.append(c_feed)
        elif p == n_x-1:
            boundary_c.append(c_eq)
        else:
            boundary_c.append(0.0)
    boundary_c = np.asarray(boundary_c)

    # Initial conditions
    initial_c = []
    for k in range(n_x):
        if k == n_x-1:
            initial_c.append(c_eq)
        elif k == 0:
            initial_c.append(c_feed)
        else:
            # seed dist.
            initial_c.append(log_normal(x_ct[k-1]*1e6, 0, 1e15, 0.3, 40))
    initial_c = np.asarray(initial_c)

    # number of unit operations
    model.root.input.model.nunits = 3

    # inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 20000,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = boundary_c
    model.root.input.model.unit_000.sec_000.lin_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = n_x*[0.0,]

    # CSTR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.use_analytic_jacobian = 1  # jacobian enabled
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_liquid_volume = 500e-6
    model.root.input.model.unit_001.const_solid_volume = 0.0
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    # upwind used
    model.root.input.model.unit_001.reaction_bulk.cry_growth_scheme_order = 1

    # particle properties
    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    model.root.input.model.unit_001.reaction_bulk.cry_nuclei_mass_density = 1.2e3
    model.root.input.model.unit_001.reaction_bulk.cry_vol_shape_factor = 0.524

    # nucleation
    model.root.input.model.unit_001.reaction_bulk.cry_primary_nucleation_rate = 0.0
    model.root.input.model.unit_001.reaction_bulk.cry_secondary_nucleation_rate = 0.0
    model.root.input.model.unit_001.reaction_bulk.cry_b = 2.0
    model.root.input.model.unit_001.reaction_bulk.cry_k = 1.0
    model.root.input.model.unit_001.reaction_bulk.cry_u = 1.0

    # growth
    model.root.input.model.unit_001.reaction_bulk.cry_growth_rate_constant = 0.02e-6
    # size-independent
    model.root.input.model.unit_001.reaction_bulk.cry_growth_constant = 0.0
    model.root.input.model.unit_001.reaction_bulk.cry_g = 1.0
    model.root.input.model.unit_001.reaction_bulk.cry_a = 1.0
    model.root.input.model.unit_001.reaction_bulk.cry_p = 0.0

    # growth rate dispersion
    model.root.input.model.unit_001.reaction_bulk.cry_growth_dispersion_rate = 0.0

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 0.0      # volumetric flow rate

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]  # Q, volumetric flow rate

    # numerical solver configuration
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1

    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_001.write_solution_bulk = 0
    model.root.input['return'].unit_001.write_solution_inlet = 0
    model.root.input['return'].unit_001.write_sens_outlet = 0
    model.root.input['return'].unit_001.write_sens_bulk = 0
    model.root.input['return'].unit_001.write_solution_outlet = 1

    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(
        0, cycle_time, time_resolution)

    model.filename = str(output_path) + '//ref_PBM_CSTR_growth.h5'

    return model

def PBM_CSTR_growthSizeDep(n_x, output_path):
    
    model = PBM_CSTR_growth(n_x, output_path)  # copy the same settings

    model.root.input.model.unit_001.reaction_bulk.cry_growth_constant = 1e8
    model.root.input.model.unit_001.reaction_bulk.cry_p = 1.5
    model.filename = str(output_path) + '//ref_PBM_CSTR_growthSizeDep.h5'

    return model

def PBM_CSTR_primaryNucleationAndGrowth(n_x, output_path):

    # Initial conditions
    initial_c = []
    for k in range(n_x):
        if k == n_x-1:
            initial_c.append(1.2)
        elif k == 0:
            initial_c.append(2.0)
        else:
            initial_c.append(0.0)
    initial_c = np.asarray(initial_c)

    model = PBM_CSTR_growth(n_x, output_path)  # copy the same settings
    model.root.input.model.unit_001.init_c = initial_c

    # crystallization
    # primary nucleation
    model.root.input.model.unit_001.reaction_bulk.cry_primary_nucleation_rate = 1e6
    model.root.input.model.unit_001.reaction_bulk.cry_u = 5.0

    # growth
    model.root.input.model.unit_001.reaction_bulk.cry_growth_rate_constant = 0.02e-6

    model.filename = str(output_path) + '//ref_PBM_CSTR_primaryNucleationAndGrowth.h5'

    return model

# this test is different from the paper

def PBM_CSTR_primarySecondaryNucleationAndGrowth(n_x, output_path):

    model = PBM_CSTR_primaryNucleationAndGrowth(n_x, output_path)

    # crystallization
    # add secondary nucleation
    model.root.input.model.unit_001.reaction_bulk.cry_secondary_nucleation_rate = 1e5

    model.filename = str(output_path) + '//ref_PBM_CSTR_PBM_CSTR_primarySecondaryNucleationAndGrowth.h5'

    return model

# this test is different from the paper

def PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion(n_x, output_path):

    model = PBM_CSTR_primaryNucleationAndGrowth(n_x, output_path)

    # crystallization
    # add growth rate dispersion
    model.root.input.model.unit_001.reaction_bulk.cry_growth_dispersion_rate = 2e-14

    model.filename = str(output_path) + '//ref_PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion.h5'

    return model

# DPFR case
# n_x is the total number of component = FVM cells - 2, n_col is the total number of FVM cells in the axial coordinate z, 52x50 would be a good place to start

def PBM_DPFR_primarySecondaryNucleationGrowth(n_x, n_col, output_path):
    # general settings
    # feed
    c_feed = 9.0  # mg/ml
    c_eq = 0.4   # mg/ml

    # particle space
    x_c = 1e-6      # m
    x_max = 900e-6  # m

    # time
    cycle_time = 200  # s

    # create model
    model = Cadet()

    # Spacing
    x_grid = np.logspace(np.log10(x_c), np.log10(x_max), n_x-1)  # log grid
    x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range(1, n_x-1)]

    # Boundary conditions
    boundary_c = []
    for p in range(n_x):
        if p == 0:
            boundary_c.append(c_feed)
        elif p == n_x-1:
            boundary_c.append(c_eq)
        else:
            boundary_c.append(0.0)
    boundary_c = np.asarray(boundary_c)

    # Initial conditions
    initial_c = []
    for k in range(n_x):
        if k == 0:
            initial_c.append(0.0)
        elif k == n_x-1:
            initial_c.append(c_eq)
        else:
            initial_c.append(0.0)
    initial_c = np.asarray(initial_c)

    # number of unit operations
    model.root.input.model.nunits = 3

    # inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 1500,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = boundary_c
    model.root.input.model.unit_000.sec_000.lin_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = n_x*[0.0,]

    # Tubular reactor
    model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.adsorption_model = 'NONE'
    model.root.input.model.unit_001.col_length = 0.47
    model.root.input.model.unit_001.cross_section_area = 1.46e-4  # m^2
    model.root.input.model.unit_001.total_porosity = 0.21
    model.root.input.model.unit_001.col_dispersion = 4.2e-05     # m^2/s
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_q = n_x*[0.0]

    # column discretization
    model.root.input.model.unit_001.discretization.ncol = n_col
    model.root.input.model.unit_001.discretization.nbound = n_x*[0]
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1  # jacobian enabled
    model.root.input.model.unit_001.discretization.gs_type = 1
    model.root.input.model.unit_001.discretization.max_krylov = 0
    model.root.input.model.unit_001.discretization.max_restarts = 10
    model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8

    # WENO23 is used here
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = 2

    # crystallization
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    # particle properties
    model.root.input.model.unit_001.reaction.cry_bins = x_grid
    model.root.input.model.unit_001.reaction.cry_nuclei_mass_density = 1.2e3
    model.root.input.model.unit_001.reaction.cry_vol_shape_factor = 0.524

    # primary nucleation
    model.root.input.model.unit_001.reaction.cry_primary_nucleation_rate = 5
    model.root.input.model.unit_001.reaction.cry_u = 10.0

    # secondary nucleation
    model.root.input.model.unit_001.reaction.cry_secondary_nucleation_rate = 4e8
    model.root.input.model.unit_001.reaction.cry_b = 2.0
    model.root.input.model.unit_001.reaction.cry_k = 1.0

    # size-independent growth
    model.root.input.model.unit_001.reaction.cry_growth_scheme_order = 1        # upwind is used
    model.root.input.model.unit_001.reaction.cry_growth_rate_constant = 7e-6
    model.root.input.model.unit_001.reaction.cry_growth_constant = 0
    model.root.input.model.unit_001.reaction.cry_a = 1.0
    model.root.input.model.unit_001.reaction.cry_g = 1.0
    model.root.input.model.unit_001.reaction.cry_p = 0

    # growth rate dispersion
    model.root.input.model.unit_001.reaction.cry_growth_dispersion_rate = 0.0

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 10.0*1e-6/60  # volumetric flow rate 10 ml/min

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]  # Q, volumetric flow rate

    # numerical solver configuration
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1

    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_001.write_solution_bulk = 0
    model.root.input['return'].unit_001.write_solution_inlet = 0
    model.root.input['return'].unit_001.write_solution_particle = 0
    model.root.input['return'].unit_001.write_solution_solid = 0
    model.root.input['return'].unit_001.write_solution_volume = 0
    model.root.input['return'].unit_001.write_solution_outlet = 1

    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(
        0, cycle_time, 200)

    # file name
    model.filename = str(output_path) + '//ref_PBM_DPFR_primarySecondaryNucleationGrowth.h5'

    return model

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

def get_EOC(N_x_ref, N_x_test, target_model, xmax, output_path): 
    
    ## get ref solution
    
    model = target_model(N_x_ref, output_path)
    model.save()
    data = model.run()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    model.load()

    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]

    ## interpolate the reference solution

    x_grid = np.logspace(np.log10(1e-6), np.log10(xmax), N_x_ref - 1) 
    x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, N_x_ref-1)]

    spl = UnivariateSpline(x_ct, c_x_reference)

    ## EOC
    
    n_xs = []   ## store the result nx here
    for Nx in N_x_test:
        model = target_model(Nx, output_path)
        model.save()
        data = model.run()
        if not data.return_code == 0:
            print(data.error_message)
            raise Exception(f"simulation failed")
        model.load() 

        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,1:-1])

    relative_L1_norms = []  ## store the relative L1 norms here
    for nx in n_xs:
        ## interpolate the ref solution on the test case grid
        
        x_grid = np.logspace(np.log10(1e-6), np.log10(xmax), len(nx) + 1)
        x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, len(nx)+1)]

        relative_L1_norms.append(calculate_relative_L1_norm(nx, spl(x_ct), x_grid))

    slopes = get_slope(relative_L1_norms) ## calculate slopes
    
    return np.array(slopes)


# %% Define crystallization tests

def crystallization_tests(n_jobs, database_path, small_test,
                          output_path, cadet_path):

    os.makedirs(output_path, exist_ok=True)
    
    Cadet.cadet_path = cadet_path    

    # %% Verify PBM_CSTR_growth
    
    N_x_ref = 800 + 2 if small_test else 2000
    ## grid for EOC
    N_x_test_c1 = [50, 100, 200, 400] if small_test else [50, 100, 200, 400, 800, 1600, ]
    N_x_test_c1 = np.array(N_x_test_c1) + 2
    
    EOC_c1 = get_EOC(N_x_ref, N_x_test_c1, PBM_CSTR_growth, 1000e-6, output_path)
    print("PBM_CSTR_growth EOC:\n", EOC_c1)
    data = {
        "Nx" : N_x_test_c1.tolist(),
        "EOC" : EOC_c1.tolist()
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/PBM_CSTR_growth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    # %% Verify PBM_CSTR_growthSizeDep
    
    N_x_ref = 700 + 2 if small_test else 1000 + 2
    
    N_x_test_c2 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c2 = np.asarray(N_x_test_c2) + 2
    
    EOC_c2 = get_EOC(N_x_ref, N_x_test_c2, PBM_CSTR_growthSizeDep, 1000e-6, output_path)
    print("PBM_CSTR_growthSizeDep EOC:\n", EOC_c2)
    data = {
        "Nx" : N_x_test_c2.tolist(),
        "EOC" : EOC_c2.tolist()
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/PBM_CSTR_growthSizeDep.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    # %% Verify PBM_CSTR_primaryNucleationAndGrowth
    
    N_x_ref = 700 + 2 if small_test else 1000 + 2
    
    N_x_test_c3 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c3 = np.asarray(N_x_test_c3) + 2
    
    EOC_c3 = get_EOC(N_x_ref, N_x_test_c3, PBM_CSTR_primaryNucleationAndGrowth, 1000e-6, output_path)
    print("PBM_CSTR_primaryNucleationAndGrowth EOC:\n", EOC_c3)
    data = {
        "Nx" : N_x_test_c3.tolist(),
        "EOC" : EOC_c3.tolist()
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/PBM_CSTR_primaryNucleationAndGrowth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    # %% Verify PBM_CSTR_primarySecondaryNucleationAndGrowth
    
    N_x_ref = 700 + 2 if small_test else 1000 + 2
    
    N_x_test_c4 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c4 = np.asarray(N_x_test_c4) + 2
    
    EOC_c4 = get_EOC(N_x_ref, N_x_test_c4, PBM_CSTR_primarySecondaryNucleationAndGrowth, 1000e-6, output_path)
    print("PBM_CSTR_primaryNucleationAndGrowth EOC:\n", EOC_c4)
    data = {
        "Nx" : N_x_test_c4.tolist(),
        "EOC" : EOC_c4.tolist()
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/PBM_CSTR_primarySecondaryNucleationAndGrowth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    # %% Verify PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion
    
    N_x_ref = 700 + 2 if small_test else 1000 + 2
    
    N_x_test_c5 = [50, 100, 200, 400, ] if small_test else [50, 100, 200, 400, 800, ]
    N_x_test_c5 = np.asarray(N_x_test_c5) + 2
    
    EOC_c5 = get_EOC(N_x_ref, N_x_test_c5, PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion, 1000e-6, output_path)
    print("PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion EOC:\n", EOC_c5)
    data = {
        "Nx" : N_x_test_c5.tolist(),
        "EOC" : EOC_c5.tolist()
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    # %% Verify PBM_DPFR_primarySecondaryNucleationGrowth
    # This is a special case, we have Nx and Ncol
    # Here we test EOC long each coordinate
    
    N_x_ref   = 120 if small_test else 200 + 2 # very fine reference: 500 + 2
    N_col_ref = 120 if small_test else 200 # very fine reference: 500
    
    x_max = 900e-6 # um
    
    ## get ref solution
        
    model = PBM_DPFR_primarySecondaryNucleationGrowth(N_x_ref, N_col_ref, output_path)
    model.save()
    data = model.run()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")
    model.load() 
    
    c_x_reference = model.root.output.solution.unit_001.solution_outlet[-1,1:-1]
    
    ## interpolate the reference solution at the reactor outlet
    
    x_grid = np.logspace(np.log10(1e-6), np.log10(x_max), N_x_ref - 1) 
    x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, N_x_ref-1)]
    
    spl = UnivariateSpline(x_ct, c_x_reference)
    
    ## EOC, Nx
    
    N_x_test_c6 = [20, 40, 80, ] if small_test else [20, 40, 80, 160, ] # very fine grid: [25, 50, 100, 200, 400, ]
    N_x_test_c6 = np.asarray(N_x_test_c6) + 2
    
    n_xs = []   ## store the result nx here
    for Nx in N_x_test_c6:
        model = PBM_DPFR_primarySecondaryNucleationGrowth(Nx, N_col_ref, output_path)
        model.save()
        data = model.run()
        if not data.return_code == 0:
            print(data.error_message)
            raise Exception(f"simulation failed")
        model.load() 
    
        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,1:-1])
    
    relative_L1_norms = []  ## store the relative L1 norms here
    for nx in n_xs:
        ## interpolate the ref solution on the test case grid
    
        x_grid = np.logspace(np.log10(1e-6), np.log10(900e-6), len(nx) + 1)
        x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, len(nx)+1)]
    
        relative_L1_norms.append(calculate_relative_L1_norm(nx, spl(x_ct), x_grid))
    
    slopes_Nx = get_slope(relative_L1_norms) ## calculate slopes
    print("PBM_DPFR_primarySecondaryNucleationGrowth EOC in internal coordinate:\n", slopes_Nx)
    
    ## EOC, Ncol
    
    N_col_test_c6 = [20, 40, 80, ] if small_test else [20, 40, 80, 160, ] # very fine grid: [25, 50, 100, 200, 400, ]   ## grid for EOC
    N_col_test_c6 = np.asarray(N_col_test_c6)
    
    n_xs = []   ## store the result nx here
    for Ncol in N_col_test_c6:
        model = PBM_DPFR_primarySecondaryNucleationGrowth(N_x_ref+2, Ncol, output_path)
        model.save()
        data = model.run()
        if not data.return_code == 0:
            print(data.error_message)
            raise Exception(f"simulation failed")
        model.load() 
    
        n_xs.append(model.root.output.solution.unit_001.solution_outlet[-1,1:-1])
    
    relative_L1_norms = []  ## store the relative L1 norms here
    for nx in n_xs:
        ## interpolate the ref solution on the test case grid
    
        x_grid = np.logspace(np.log10(1e-6), np.log10(900e-6), len(nx) + 1)
        x_ct = [0.5*x_grid[p] + 0.5*x_grid[p-1] for p in range (1, len(nx)+1)]
    
        relative_L1_norms.append(calculate_relative_L1_norm(nx, spl(x_ct), x_grid))
    
    slopes_Ncol = get_slope(relative_L1_norms) ## calculate slopes
    print(slopes_Ncol)
    
    print("PBM_DPFR_primarySecondaryNucleationGrowth EOC in axial direction:\n", slopes_Ncol)
    data = {
        "Convergence in axial direction" : {
        "Ncol" : N_col_test_c6.tolist(),
        "EOC" : slopes_Ncol.tolist()
        },
        "Convergence in internal coordinate" : {
        "Nx" : N_x_test_c6.tolist(),
        "EOC" : slopes_Nx.tolist()
        }
    }
    
    # Write the dictionary to a JSON file
    with open(str(output_path) + '/PBM_DPFR_primarySecondaryNucleationGrowth.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

