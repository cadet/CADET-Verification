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


def PureAgg_Golovin(n_x: 'int, number of bins', x_c, x_max, v_0, N_0, beta_0, t):

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
    model.root.input.model.unit_001.porosity = 1
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reactions
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
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
    model.filename = 'practice1.h5'  # change as needed

    return model


def PureFrag_LinBi(n_x: 'int, number of bins', x_c, x_max, S_0, t):
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
    model.root.input.model.unit_001.porosity = 1
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reactions
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'

    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_kernel_gamma = 2.0
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_rate_constant = S_0
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_selection_function_alpha = 1.0

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
    model.filename = 'practice1.h5'  # change as needed

    return model


def Agg_frag(n_x: 'int, number of bins', x_c, x_max, beta_0, S_0, t):
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
    model.root.input.model.unit_001.porosity = 1
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reactions
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'

    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_index = 0
    model.root.input.model.unit_001.reaction_bulk.cry_aggregation_rate_constant = beta_0

    model.root.input.model.unit_001.reaction_bulk.cry_breakage_rate_constant = S_0
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_kernel_gamma = 2
    model.root.input.model.unit_001.reaction_bulk.cry_breakage_selection_function_alpha = 1

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
    model.filename = 'practice1.h5'  # change as needed

    return model


def Agg_DPFR(n_x: 'int, number of x bins', n_col: 'int, number of z bins', x_c, x_max, axial_order, t):
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

    model.filename = 'practice1.h5'  # change as needed

    return model


def Frag_DPFR(n_x: 'int, number of x bins', n_col: 'int, number of z bins', x_c, x_max, axial_order, t):
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

    model.root.input.model.unit_001.reaction.cry_bins = x_grid
    model.root.input.model.unit_001.reaction.cry_breakage_rate_constant = 0.5e12
    model.root.input.model.unit_001.reaction.cry_breakage_kernel_gamma = 2
    model.root.input.model.unit_001.reaction.cry_breakage_selection_function_alpha = 1

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

    model.filename = 'practice1.h5'  # change as needed

    return model


def NGRA(n_x: 'int, number of x bins + 2', n_col: 'int, number of z bins', x_c, x_max, axial_order: 'for weno schemes', growth_order, t):
    model = Cadet()

    # Spacing
    x_grid, x_ct = get_log_space(n_x - 2, x_c, x_max)

    # c_feed
    c_feed = 9.0
    c_eq = 0.4

    # Boundary conditions
    boundary_c = []
    for i in range(0, n_x):
        if i == 0:
            boundary_c.append(c_feed)
        elif i == n_x - 1:
            boundary_c.append(c_eq)
        else:
            boundary_c.append(0.0)

    # Initial conditions
    initial_c = []
    for k in range(n_x):
        if k == 0:
            initial_c.append(0)
        elif k == n_x-1:
            initial_c.append(c_eq)
        else:
            initial_c.append(0)

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
    model.root.input.model.unit_001.cross_section_area = 3.066e-05
    model.root.input.model.unit_001.total_porosity = 1.0
    model.root.input.model.unit_001.col_dispersion = 4.2e-05
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
    model.root.input.model.unit_002.ncomp = n_x

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

    model.filename = 'practice1.h5'  # change as needed

    return model


def Agg_Frag_DPFR(n_x : 'int, number of x bins', n_col : 'int, number of z bins', x_c, x_max, axial_order, t):
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
    
    model.root.input.model.unit_001.reaction.cry_bins = x_grid
    
    model.root.input.model.unit_001.reaction.cry_aggregation_index = 0 # constant kernel 0, brownian kernel 1, smoluchowski kernel 2, golovin kernel 3, differential force kernel 4
    model.root.input.model.unit_001.reaction.cry_aggregation_rate_constant = 2.4e-12
    
    model.root.input.model.unit_001.reaction.cry_breakage_rate_constant = 6.0e10
    model.root.input.model.unit_001.reaction.cry_breakage_kernel_gamma = 2 
    model.root.input.model.unit_001.reaction.cry_breakage_selection_function_alpha = 1

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

    model.filename = 'practice1.h5'                    ## change as needed
    
    return model
