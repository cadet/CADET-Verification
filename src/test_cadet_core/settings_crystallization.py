# -*- coding: utf-8 -*-
'''
Created January 2025

This script implements the settings used for verification of the crystallization
code, including PBM, aggregation, fragmentation, and all combinations, as well
as the incorporation into both a CSTR and DPFR.

@author: Wendi Zhang and jmbr
'''


import numpy as np

from cadet import Cadet


# %% Auxiliary functions


def get_log_space(n_x, x_c, x_max):
    x_grid = np.logspace(np.log10(x_c), np.log10(x_max), n_x+1)  # log space
    x_ct = np.asarray([0.5 * x_grid[p+1] + 0.5 * x_grid[p]
                      for p in range(0, n_x)])
    return x_grid, x_ct


def log_normal(x, y0, A, w, xc):
    return y0 + A/(np.sqrt(2.0*np.pi) * w*x) * np.exp(-np.log(x/xc)**2 / 2.0/w**2)


# %% Crystallization settings


def CSTR_PBM_growth(n_x, output_path):

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
    model.root.input.model.unit_001.reaction_bulk.cry_mode = 1
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

    model.filename = str(output_path) + '//ref_CSTR_PBM_growth.h5'

    return model

def CSTR_PBM_growthSizeDep(n_x, output_path):
    
    model = CSTR_PBM_growth(n_x, output_path)  # copy the same settings

    model.root.input.model.unit_001.reaction_bulk.cry_growth_constant = 1e8
    model.root.input.model.unit_001.reaction_bulk.cry_p = 1.5
    model.filename = str(output_path) + '//ref_CSTR_PBM_growthSizeDep.h5'

    return model

def CSTR_PBM_primaryNucleationAndGrowth(n_x, output_path):

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

    model = CSTR_PBM_growth(n_x, output_path)  # copy the same settings
    model.root.input.model.unit_001.init_c = initial_c

    # crystallization
    # primary nucleation
    model.root.input.model.unit_001.reaction_bulk.cry_primary_nucleation_rate = 1e6
    model.root.input.model.unit_001.reaction_bulk.cry_u = 5.0

    # growth
    model.root.input.model.unit_001.reaction_bulk.cry_growth_rate_constant = 0.02e-6

    model.filename = str(output_path) + '//ref_CSTR_PBM_primaryNucleationAndGrowth.h5'

    return model

# this test is different from the paper

def CSTR_PBM_primarySecondaryNucleationAndGrowth(n_x, output_path):

    model = CSTR_PBM_primaryNucleationAndGrowth(n_x, output_path)

    # crystallization
    # add secondary nucleation
    model.root.input.model.unit_001.reaction_bulk.cry_secondary_nucleation_rate = 1e5

    model.filename = str(output_path) + '//ref_CSTR_PBM_CSTR_PBM_primarySecondaryNucleationAndGrowth.h5'

    return model

# this test is different from the paper

def CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion(n_x, output_path):

    model = CSTR_PBM_primaryNucleationAndGrowth(n_x, output_path)

    # crystallization
    # add growth rate dispersion
    model.root.input.model.unit_001.reaction_bulk.cry_growth_dispersion_rate = 2e-14

    model.filename = str(output_path) + '//ref_CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion.h5'

    return model

# DPFR case
# n_x is the total number of component = FVM cells - 2, n_col is the total number of FVM cells in the axial coordinate z, 52x50 would be a good place to start

def DPFR_PBM_primarySecondaryNucleationGrowth(n_x, n_col, output_path):
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
    model.root.input.model.unit_001.reaction_bulk.cry_mode = 1
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
    model.filename = str(output_path) + '//ref_DPFR_PBM_primarySecondaryNucleationGrowth.h5'

    return model


def PureAgg_Golovin(n_x: 'int, number of bins', x_c, x_max, v_0, N_0, beta_0, t, output_path):

    model = Cadet()

    # crystal space
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = n_x*[0.0, ]

    # Initial conditions
    initial_c = np.asarray([3.0*x_ct[k]**2 * np.exp(-x_ct[k]**3/v_0) *
                           # see our paper for the equation
                            N_0/v_0 for k in range(0, n_x)])

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

    # CSTR/MSMPR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.use_analytic_jacobian = 1
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_volume = 500e-6
    model.root.input.model.unit_001.const_solid_volume = 0.0
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reactions
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction_bulk.cry_mode = 2
    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_index = 3
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_rate_constant = beta_0

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 0                   # volumetric flow rate

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

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
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t
    model.filename = output_path + '/crystallization_aggregation_Z' + str(x_grid.size) +'.h5'

    return model, x_grid, x_ct


def PureFrag_LinBi(n_x: 'int, number of bins', x_c, x_max, S_0, t, output_path):
    model = Cadet()

    # crystal space
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = n_x*[0.0, ]

    # Initial conditions
    initial_c = np.asarray([3.0*x_ct[k]**2 * np.exp(-x_ct[k]**3)
                           # see our paper for the equation
                            for k in range(0, n_x)])

    # number of unit operations
    model.root.input.model.nunits = 4

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

    # CSTR/MSMPR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.use_analytic_jacobian = 1
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_volume = 500e-6
    model.root.input.model.unit_001.const_solid_volume = 0.0
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reactions
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction_bulk.cry_mode = 4

    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    model.root.input.model.unit_001.reaction_bulk.cry_fragmentation_kernel_gamma = 2.0
    model.root.input.model.unit_001.reaction_bulk.cry_fragmentation_rate_constant = S_0
    model.root.input.model.unit_001.reaction_bulk.cry_fragmentation_selection_function_alpha = 1.0

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 0                   # volumetric flow rate

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

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
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t
    model.filename = output_path + '/crystallization_fragmentation_Z' + str(x_grid.size) +'.h5'

    return model, x_grid, x_ct


def Agg_frag(n_x: 'int, number of bins', x_c, x_max, beta_0, S_0, t, output_path):
    model = Cadet()

    # crystal space
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = n_x*[0.0, ]

    # Initial conditions
    initial_c = np.asarray([3.0*x_ct[k]**2 * 4.0*x_ct[k]**3 * np.exp(-2.0*x_ct[k]**3)
                           # see our paper for the equation
                            for k in range(0, n_x)])

    # number of unit operations
    model.root.input.model.nunits = 6

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

    # CSTR/MSMPR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.use_analytic_jacobian = 1
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_volume = 500e-6
    model.root.input.model.unit_001.const_solid_volume = 0.0
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reactions
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction_bulk.cry_mode = 6

    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_index = 0
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_rate_constant = beta_0

    model.root.input.model.unit_001.reaction_bulk.cry_fragmentation_rate_constant = S_0
    model.root.input.model.unit_001.reaction_bulk.cry_fragmentation_kernel_gamma = 2
    model.root.input.model.unit_001.reaction_bulk.cry_fragmentation_selection_function_alpha = 1

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 0                   # volumetric flow rate

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

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
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t
    model.filename = output_path +  '/crystallization_aggFrag_Z' + str(x_grid.size) +'.h5'

    return model, x_grid, x_ct


def Agg_DPFR(n_x: 'int, number of x bins', n_col: 'int, number of z bins', x_c, x_max, axial_order, t, output_path):
    model = Cadet()

    # Spacing
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = log_normal(x_ct*1e6, 0, 1e16, 0.4, 20)

    # Initial conditions
    initial_c = n_x*[0.0, ]

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
    model.root.input.model.unit_001.cross_section_area = 1.46e-4*0.21  # m^2
    model.root.input.model.unit_001.total_porosity = 1.0
    model.root.input.model.unit_001.col_dispersion = 4.2e-05           # m^2/s
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_q = n_x*[0.0]

    # column discretization
    model.root.input.model.unit_001.discretization.ncol = n_col
    model.root.input.model.unit_001.discretization.nbound = n_x*[0]
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.gs_type = 1
    model.root.input.model.unit_001.discretization.max_krylov = 0
    model.root.input.model.unit_001.discretization.max_restarts = 10
    model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8

    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = axial_order

    # crystallization reaction
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction.cry_mode = 2
    model.root.input.model.unit_001.reaction.cry_bins = x_grid
    # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction.cry_aggregation_index = 0
    model.root.input.model.unit_001.reaction.cry_aggregation_rate_constant = 3e-11

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 10.0*1e-6 / 60         # volumetric flow rate, m^3/s

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

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
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1
    model.root.input['return'].unit_000.write_coordinates = 0

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t

    model.filename = output_path + '/crystallization_DPFR_Z' + str(n_col) + '_aggregation_Z' + str(x_grid.size) +'.h5'

    return model, x_grid, x_ct


def Frag_DPFR(n_x: 'int, number of x bins', n_col: 'int, number of z bins', x_c, x_max, axial_order, t, output_path):
    model = Cadet()

    # Spacing
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = log_normal(x_ct*1e6, 0, 1e16, 0.4,
                            150)  # moved to larger sizes

    # Initial conditions
    initial_c = n_x*[0.0, ]

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
    model.root.input.model.unit_001.cross_section_area = 1.46e-4*0.21  # m^2
    model.root.input.model.unit_001.total_porosity = 1.0
    model.root.input.model.unit_001.col_dispersion = 4.2e-05           # m^2/s
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_q = n_x*[0.0]

    # column discretization
    model.root.input.model.unit_001.discretization.ncol = n_col
    model.root.input.model.unit_001.discretization.nbound = n_x*[0]
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.gs_type = 1
    model.root.input.model.unit_001.discretization.max_krylov = 0
    model.root.input.model.unit_001.discretization.max_restarts = 10
    model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8

    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = axial_order

    # crystallization reaction
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction.cry_mode = 4

    model.root.input.model.unit_001.reaction.cry_bins = x_grid
    model.root.input.model.unit_001.reaction.cry_fragmentation_rate_constant = 0.5e12
    model.root.input.model.unit_001.reaction.cry_fragmentation_kernel_gamma = 2
    model.root.input.model.unit_001.reaction.cry_fragmentation_selection_function_alpha = 1

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 10.0*1e-6/60  # m^3/s

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

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
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1
    model.root.input['return'].unit_000.write_coordinates = 0

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t

    model.filename = output_path + '/crystallization_DPFR_Z' + str(n_col) + '_fragmentation_Z' + str(x_grid.size) +'.h5'

    return model, x_grid, x_ct


def DPFR_PBM_NGGR_aggregation(n_x: 'int, number of x bins', n_col: 'int, number of z bins', x_c, x_max, axial_order: 'for weno schemes', growth_order, t, output_path):
    model = Cadet()

    nComp = n_x + 2

    # Spacing
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # c_feed
    c_feed = 9.0
    c_eq = 0.4

    # Boundary conditions
    boundary_c = []
    for i in range(0, nComp):
        if i == 0:
            boundary_c.append(c_feed)
        elif i == nComp - 1:
            boundary_c.append(c_eq)
        else:
            boundary_c.append(0.0)

    # Initial conditions
    initial_c = []
    for k in range(nComp):
        if k == 0:
            initial_c.append(0)
        elif k == nComp-1:
            initial_c.append(c_eq)
        else:
            initial_c.append(0)

    # number of unit operations
    model.root.input.model.nunits = 3

    # inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = nComp
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 1500,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = boundary_c
    model.root.input.model.unit_000.sec_000.lin_coeff = nComp*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = nComp*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = nComp*[0.0,]

    # Tubular reactor
    model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    model.root.input.model.unit_001.ncomp = nComp
    model.root.input.model.unit_001.adsorption_model = 'NONE'
    model.root.input.model.unit_001.col_length = 0.47
    model.root.input.model.unit_001.cross_section_area = 3.066e-05
    model.root.input.model.unit_001.total_porosity = 1.0
    model.root.input.model.unit_001.col_dispersion = 4.2e-05
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_q = nComp*[0.0]

    # column discretization
    model.root.input.model.unit_001.discretization.ncol = n_col
    model.root.input.model.unit_001.discretization.nbound = nComp*[0]
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.gs_type = 1
    model.root.input.model.unit_001.discretization.max_krylov = 0
    model.root.input.model.unit_001.discretization.max_restarts = 10
    model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8

    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = axial_order

    # crystallization reaction
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction.cry_mode = 3
    model.root.input.model.unit_001.reaction.cry_bins = x_grid

    # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction.cry_aggregation_index = 0
    model.root.input.model.unit_001.reaction.cry_aggregation_rate_constant = 5e-13

    model.root.input.model.unit_001.reaction.cry_nuclei_mass_density = 1.2e3
    model.root.input.model.unit_001.reaction.cry_vol_shape_factor = 0.524
    model.root.input.model.unit_001.reaction.cry_primary_nucleation_rate = 5.0
    model.root.input.model.unit_001.reaction.cry_secondary_nucleation_rate = 4e8

    model.root.input.model.unit_001.reaction.cry_growth_rate_constant = 5e-6
    model.root.input.model.unit_001.reaction.cry_g = 1.0

    model.root.input.model.unit_001.reaction.cry_a = 1.0
    model.root.input.model.unit_001.reaction.cry_growth_constant = 0.0
    model.root.input.model.unit_001.reaction.cry_p = 0.0

    model.root.input.model.unit_001.reaction.cry_k = 1.0
    model.root.input.model.unit_001.reaction.cry_u = 10.0
    model.root.input.model.unit_001.reaction.cry_b = 2.0

    model.root.input.model.unit_001.reaction.cry_growth_dispersion_rate = 2.5e-15
    model.root.input.model.unit_001.reaction.cry_growth_scheme_order = growth_order

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = nComp

    # Connections
    Q = 10.0*1e-6/60     # Q, volumetric flow rate

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

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
    model.root.input.solver.time_integrator.init_step_size = 1e-8
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_coordinates = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t

    model.filename = output_path + '/crystallization_DPFR_Z' + str(n_col) + '_NGGR_Z' + str(x_grid.size) +'.h5'

    return model, x_grid, x_ct


def Agg_Frag_DPFR(n_x : 'int, number of x bins', n_col : 'int, number of z bins', x_c, x_max, axial_order, t, output_path):
    
    model = Cadet()

    # Spacing
    x_grid, x_ct = get_log_space(n_x, x_c, x_max)

    # Boundary conditions
    boundary_c = log_normal(x_ct*1e6,0,1e16,0.4,80)

    # Initial conditions
    initial_c = n_x*[0.0, ]

    # number of unit operations
    model.root.input.model.nunits = 3

    #inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    #time sections
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
    model.root.input.model.unit_001.cross_section_area = 1.46e-4*0.21  # m^2
    model.root.input.model.unit_001.total_porosity = 1.0
    model.root.input.model.unit_001.col_dispersion = 4.2e-05           # m^2/s
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_q = n_x*[0.0]

    # column discretization
    model.root.input.model.unit_001.discretization.ncol = n_col
    model.root.input.model.unit_001.discretization.nbound = n_x*[0]
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.gs_type = 1
    model.root.input.model.unit_001.discretization.max_krylov = 0
    model.root.input.model.unit_001.discretization.max_restarts = 10
    model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8
    
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = axial_order

    # crystallization reaction
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction.cry_mode = 7
    
    model.root.input.model.unit_001.reaction.cry_bins = x_grid
    
    model.root.input.model.unit_001.reaction.cry_aggregation_index = 0 # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction.cry_aggregation_rate_constant = 2.4e-12
    
    model.root.input.model.unit_001.reaction.cry_fragmentation_rate_constant = 6.0e10
    model.root.input.model.unit_001.reaction.cry_fragmentation_kernel_gamma = 2
    model.root.input.model.unit_001.reaction.cry_fragmentation_selection_function_alpha = 1

    ## Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 10.0*1e-6/60          ## m^3/s

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

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
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1
    model.root.input['return'].unit_000.write_coordinates = 0

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = t

    model.filename = output_path + '/crystallization_DPFR_Z' + str(n_col) + '_aggFrag_Z' + str(x_grid.size) +'.h5'
    
    return model, x_grid, x_ct
