# -*- coding: utf-8 -*-
"""
Created in Jan 2025

This file contains the system settings for a linear SMB, a cyclic and acyclic
system. These settings are used to verify the implementation of systems
published in Frandsen et al.
    'High-Performance C++ and Julia solvers in CADET for weakly and strongly
    coupled continuous chromatography problems' (2025b)

@author: Jesper Frandsen
"""

from addict import Dict
import numpy as np
import copy
from cadet import Cadet


# %% Define system settings in hierarchical format


def SMB_model1(nelem, polydeg, exactInt):

    ts = 1552
    QD = 4.14e-8
    QE = 3.48e-8
    QF = 2.00e-8
    QR = 2.66e-8
    Q2 = 1.05e-7
    Q3 = Q2 + QF
    Q4 = Q3 - QR
    Q1 = Q4 + QD

    # Setting up the model
    smb_model = Dict()

    # Speciy number of unit operations: input, column and output, 3
    smb_model.model.nunits = 12

    # Specify # of components (salt,proteins)
    n_comp = 2

    # First unit operation: inlet
    # Feed
    smb_model.model.unit_000.unit_type = 'INLET'
    smb_model.model.unit_000.ncomp = n_comp
    smb_model.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # Eluent
    smb_model.model.unit_001.unit_type = 'INLET'
    smb_model.model.unit_001.ncomp = n_comp
    smb_model.model.unit_001.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # Extract
    smb_model.model.unit_002.ncomp = n_comp
    smb_model.model.unit_002.unit_type = 'OUTLET'

    # Raffinate
    smb_model.model.unit_003.ncomp = n_comp
    smb_model.model.unit_003.unit_type = 'OUTLET'

    # Columns
    smb_model.model.unit_004.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    smb_model.model.unit_004.ncomp = n_comp

    # Geometry
    smb_model.model.unit_004.total_porosity = 0.38
    smb_model.model.unit_004.col_dispersion = 3.81e-6
    smb_model.model.unit_004.col_length = 5.36e-1
    # From Lubke2007, is not important
    smb_model.model.unit_004.cross_section_area = 5.31e-4

    # Isotherm specification
    smb_model.model.unit_004.adsorption_model = 'LINEAR'
    smb_model.model.unit_004.adsorption.is_kinetic = False    # Kinetic binding
    smb_model.model.unit_004.adsorption.LIN_KA = [
        0.54, 0.28]  # m^3 / (mol * s)   (mobile phase)
    smb_model.model.unit_004.adsorption.LIN_KD = [
        1, 1]      # 1 / s (desorption)
    # Initial conditions
    smb_model.model.unit_004.init_c = [0, 0]
    smb_model.model.unit_004.init_q = [0, 0]  # salt starts at max capacity

    # Grid cells in column and particle: the most important ones - ensure grid-independent solutions
    smb_model.model.unit_004.discretization.SPATIAL_METHOD = "DG"
    smb_model.model.unit_004.discretization.NELEM = nelem

    # Polynomial order
    smb_model.model.unit_004.discretization.POLYDEG = polydeg
    smb_model.model.unit_004.discretization.EXACT_INTEGRATION = exactInt

    # Bound states - for zero the compound does not bind, >1 = multiple binding sites
    smb_model.model.unit_004.discretization.nbound = np.ones(n_comp, dtype=int)

    smb_model.model.unit_004.discretization.par_disc_type = 'EQUIDISTANT_PAR'
    smb_model.model.unit_004.discretization.use_analytic_jacobian = 1
    smb_model.model.unit_004.discretization.reconstruction = 'WENO'
    smb_model.model.unit_004.discretization.gs_type = 1
    smb_model.model.unit_004.discretization.max_krylov = 0
    smb_model.model.unit_004.discretization.max_restarts = 10
    smb_model.model.unit_004.discretization.schur_safety = 1.0e-8

    smb_model.model.unit_004.discretization.weno.boundary_model = 0
    smb_model.model.unit_004.discretization.weno.weno_eps = 1e-10
    smb_model.model.unit_004.discretization.weno.weno_order = 3

    # Copy column models
    smb_model.model.unit_005 = smb_model.model.unit_004
    smb_model.model.unit_006 = smb_model.model.unit_004
    smb_model.model.unit_007 = smb_model.model.unit_004
    smb_model.model.unit_008 = smb_model.model.unit_004
    smb_model.model.unit_009 = smb_model.model.unit_004
    smb_model.model.unit_010 = smb_model.model.unit_004
    smb_model.model.unit_011 = smb_model.model.unit_004

    # To write out last output to check for steady state
    smb_model['return'].WRITE_SOLUTION_LAST = True

    # % Input and connections
    n_cycles = 10
    switch_time = ts  # s

    # Sections
    smb_model.solver.sections.nsec = 8*n_cycles
    smb_model.solver.sections.section_times = [0]
    for i in range(n_cycles):
        smb_model.solver.sections.section_times.append((8*i+1)*switch_time)
        smb_model.solver.sections.section_times.append((8*i+2)*switch_time)
        smb_model.solver.sections.section_times.append((8*i+3)*switch_time)
        smb_model.solver.sections.section_times.append((8*i+4)*switch_time)
        smb_model.solver.sections.section_times.append((8*i+5)*switch_time)
        smb_model.solver.sections.section_times.append((8*i+6)*switch_time)
        smb_model.solver.sections.section_times.append((8*i+7)*switch_time)
        smb_model.solver.sections.section_times.append((8*i+8)*switch_time)

    # Feed and Eluent concentration
    smb_model.model.unit_000.sec_000.const_coeff = [
        2.78, 2.78]  # Inlet flowrate concentration
    smb_model.model.unit_001.sec_000.const_coeff = [0, 0]  # Desorbent stream

    # Connections
    smb_model.model.connections.nswitches = 8

    smb_model.model.connections.switch_000.section = 0
    smb_model.model.connections.switch_000.connections = [
        4, 5, -1, -1, Q3,  # flowrates, Q, m3/s
        5, 6, -1, -1, Q4,
        6, 7, -1, -1, Q4,
        7, 8, -1, -1, Q4,
        8, 9, -1, -1, Q1,
        9, 10, -1, -1, Q2,
        10, 11, -1, -1, Q2,
        11, 4, -1, -1, Q2,
        0, 4, -1, -1, QF,
        1, 8, -1, -1, QD,
        5, 3, -1, -1, QR,
        9, 2, -1, -1, QE
    ]

    smb_model.model.connections.switch_001.section = 1
    smb_model.model.connections.switch_001.connections = [
        4, 5, -1, -1, Q2,  # flowrates, Q, m3/s
        5, 6, -1, -1, Q3,
        6, 7, -1, -1, Q4,
        7, 8, -1, -1, Q4,
        8, 9, -1, -1, Q4,
        9, 10, -1, -1, Q1,
        10, 11, -1, -1, Q2,
        11, 4, -1, -1, Q2,
        0, 5, -1, -1, QF,
        1, 9, -1, -1, QD,
        6, 3, -1, -1, QR,
        10, 2, -1, -1, QE
    ]

    smb_model.model.connections.switch_002.section = 2
    smb_model.model.connections.switch_002.connections = [
        4, 5, -1, -1, Q2,  # flowrates, Q, m3/s
        5, 6, -1, -1, Q2,
        6, 7, -1, -1, Q3,
        7, 8, -1, -1, Q4,
        8, 9, -1, -1, Q4,
        9, 10, -1, -1, Q4,
        10, 11, -1, -1, Q1,
        11, 4, -1, -1, Q2,
        0, 6, -1, -1, QF,
        1, 10, -1, -1, QD,
        7, 3, -1, -1, QR,
        11, 2, -1, -1, QE
    ]

    smb_model.model.connections.switch_003.section = 3
    smb_model.model.connections.switch_003.connections = [
        4, 5, -1, -1, Q2,  # flowrates, Q, m3/s
        5, 6, -1, -1, Q2,
        6, 7, -1, -1, Q2,
        7, 8, -1, -1, Q3,
        8, 9, -1, -1, Q4,
        9, 10, -1, -1, Q4,
        10, 11, -1, -1, Q4,
        11, 4, -1, -1, Q1,
        0, 7, -1, -1, QF,
        1, 11, -1, -1, QD,
        8, 3, -1, -1, QR,
        4, 2, -1, -1, QE
    ]

    smb_model.model.connections.switch_004.section = 4
    smb_model.model.connections.switch_004.connections = [
        4, 5, -1, -1, Q1,  # flowrates, Q, m3/s
        5, 6, -1, -1, Q2,
        6, 7, -1, -1, Q2,
        7, 8, -1, -1, Q2,
        8, 9, -1, -1, Q3,
        9, 10, -1, -1, Q4,
        10, 11, -1, -1, Q4,
        11, 4, -1, -1, Q4,
        0, 8, -1, -1, QF,
        1, 4, -1, -1, QD,
        9, 3, -1, -1, QR,
        5, 2, -1, -1, QE
    ]

    smb_model.model.connections.switch_005.section = 5
    smb_model.model.connections.switch_005.connections = [
        4, 5, -1, -1, Q4,  # flowrates, Q, m3/s
        5, 6, -1, -1, Q1,
        6, 7, -1, -1, Q2,
        7, 8, -1, -1, Q2,
        8, 9, -1, -1, Q2,
        9, 10, -1, -1, Q3,
        10, 11, -1, -1, Q4,
        11, 4, -1, -1, Q4,
        0, 9, -1, -1, QF,
        1, 5, -1, -1, QD,
        10, 3, -1, -1, QR,
        6, 2, -1, -1, QE
    ]

    smb_model.model.connections.switch_006.section = 6
    smb_model.model.connections.switch_006.connections = [
        4, 5, -1, -1, Q4,  # flowrates, Q, m3/s
        5, 6, -1, -1, Q4,
        6, 7, -1, -1, Q1,
        7, 8, -1, -1, Q2,
        8, 9, -1, -1, Q2,
        9, 10, -1, -1, Q2,
        10, 11, -1, -1, Q3,
        11, 4, -1, -1, Q4,
        0, 10, -1, -1, QF,
        1, 6, -1, -1, QD,
        11, 3, -1, -1, QR,
        7, 2, -1, -1, QE
    ]

    smb_model.model.connections.switch_007.section = 7
    smb_model.model.connections.switch_007.connections = [
        4, 5, -1, -1, Q4,  # flowrates, Q, m3/s
        5, 6, -1, -1, Q4,
        6, 7, -1, -1, Q4,
        7, 8, -1, -1, Q1,
        8, 9, -1, -1, Q2,
        9, 10, -1, -1, Q2,
        10, 11, -1, -1, Q2,
        11, 4, -1, -1, Q3,
        0, 11, -1, -1, QF,
        1, 7, -1, -1, QD,
        4, 3, -1, -1, QR,
        8, 2, -1, -1, QE
    ]

    # solution times
    smb_model.solver.user_solution_times = np.linspace(
        0, n_cycles*8*switch_time, int(n_cycles*8*switch_time)+1)

    # Tolerances for the time integrator
    smb_model.solver.time_integrator.ABSTOL = 1e-12  # absolute tolerance
    smb_model.solver.time_integrator.ALGTOL = 1e-10
    smb_model.solver.time_integrator.RELTOL = 1e-10  # Relative tolerance
    smb_model.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    smb_model.solver.time_integrator.MAX_STEPS = 1000000

    # Solver options in general (not only for column although the same)
    smb_model.model.solver.gs_type = 1
    smb_model.model.solver.max_krylov = 0
    smb_model.model.solver.max_restarts = 10
    smb_model.model.solver.schur_safety = 1e-8
    smb_model.solver.consistent_init_mode = 5  # necessary specifically for this sim
    smb_model.solver.time_integrator.USE_MODIFIED_NEWTON = 1

    # Number of cores for parallel simulation
    smb_model.solver.nthreads = 1

    # Specify which results we want to return
    # Return data
    smb_model['return'].split_components_data = 0
    smb_model['return'].split_ports_data = 0
    smb_model['return'].unit_000.write_solution_bulk = 0
    smb_model['return'].unit_000.write_solution_inlet = 0
    smb_model['return'].unit_000.write_solution_outlet = 0
    smb_model['return'].unit_002.write_solution_bulk = 0
    smb_model['return'].unit_002.write_solution_inlet = 0
    smb_model['return'].unit_002.write_solution_outlet = 1

    # Copy settings to the other unit operations
    smb_model['return'].unit_001 = smb_model['return'].unit_000
    smb_model['return'].unit_003 = smb_model['return'].unit_002
    smb_model['return'].unit_004 = smb_model['return'].unit_000
    smb_model['return'].unit_005 = smb_model['return'].unit_000
    smb_model['return'].unit_006 = smb_model['return'].unit_000
    smb_model['return'].unit_007 = smb_model['return'].unit_000
    smb_model['return'].unit_008 = smb_model['return'].unit_000
    smb_model['return'].unit_009 = smb_model['return'].unit_000
    smb_model['return'].unit_010 = smb_model['return'].unit_000
    smb_model['return'].unit_011 = smb_model['return'].unit_000

    return {'input': smb_model}


def Cyclic_model1(nelem, polydeg, exactInt, analytical_reference=False):

    # Setting up the model
    Cyclic_model = Cadet()

    # Speciy number of unit operations: input, column and output, 3
    Cyclic_model.root.input.model.nunits = 4

    # Specify # of components (salt,proteins)
    n_comp = 1

    # First unit operation: inlet
    # Source 1
    Cyclic_model.root.input.model.unit_000.unit_type = 'INLET'
    Cyclic_model.root.input.model.unit_000.ncomp = n_comp
    Cyclic_model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # Sink
    Cyclic_model.root.input.model.unit_003.ncomp = n_comp
    Cyclic_model.root.input.model.unit_003.unit_type = 'OUTLET'

    # Unit LRMP2
    Cyclic_model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES'
    Cyclic_model.root.input.model.unit_001.ncomp = n_comp

    # Geometry
    Cyclic_model.root.input.model.unit_001.col_porosity = 0.37
    Cyclic_model.root.input.model.unit_001.par_porosity = 0.75
    Cyclic_model.root.input.model.unit_001.col_dispersion = 2e-7
    Cyclic_model.root.input.model.unit_001.col_length = 1.4e-2
    Cyclic_model.root.input.model.unit_001.cross_section_area = 1
    Cyclic_model.root.input.model.unit_001.film_diffusion = 6.9e-6
    Cyclic_model.root.input.model.unit_001.par_radius = 45e-6
    LRMP_Q3 = 3.45*1e-2 / 60 * 0.37

    # Isotherm specification
    Cyclic_model.root.input.model.unit_001.adsorption_model = 'LINEAR'
    Cyclic_model.root.input.model.unit_001.adsorption.is_kinetic = True    # Kinetic binding
    Cyclic_model.root.input.model.unit_001.adsorption.LIN_KA = [
        3.55]  # m^3 / (mol * s)   (mobile phase)
    Cyclic_model.root.input.model.unit_001.adsorption.LIN_KD = [
        0.1]      # 1 / s (desorption)
    # Initial conditions
    Cyclic_model.root.input.model.unit_001.init_c = [0]
    Cyclic_model.root.input.model.unit_001.init_q = [
        0]  # salt starts at max capacity

    # Grid cells in column and particle: the most important ones - ensure grid-independent solutions
    Cyclic_model.root.input.model.unit_001.discretization.SPATIAL_METHOD = "DG"
    Cyclic_model.root.input.model.unit_001.discretization.NELEM = nelem

    # Polynomial order
    Cyclic_model.root.input.model.unit_001.discretization.POLYDEG = polydeg
    Cyclic_model.root.input.model.unit_001.discretization.EXACT_INTEGRATION = exactInt

    # Bound states - for zero the compound does not bind, >1 = multiple binding sites
    Cyclic_model.root.input.model.unit_001.discretization.nbound = np.ones(
        n_comp, dtype=int)

    Cyclic_model.root.input.model.unit_001.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
    Cyclic_model.root.input.model.unit_001.discretization.USE_ANALYTIC_JACOBIAN = 1
    Cyclic_model.root.input.model.unit_001.discretization.RECONSTRUCTION = 'WENO'
    Cyclic_model.root.input.model.unit_001.discretization.GS_TYPE = 1
    Cyclic_model.root.input.model.unit_001.discretization.MAX_KRYLOV = 0
    Cyclic_model.root.input.model.unit_001.discretization.MAX_RESTARTS = 10
    Cyclic_model.root.input.model.unit_001.discretization.SCHUR_SAFETY = 1.0e-8

    Cyclic_model.root.input.model.unit_001.discretization.weno.BOUNDARY_MODEL = 0
    Cyclic_model.root.input.model.unit_001.discretization.weno.WENO_EPS = 1e-10
    Cyclic_model.root.input.model.unit_001.discretization.weno.WENO_ORDER = 3

    # Copy column models
    Cyclic_model.root.input.model.unit_002 = copy.deepcopy(
        Cyclic_model.root.input.model.unit_001)

    # Unit LRMP2
    Cyclic_model.root.input.model.unit_002.adsorption.is_kinetic = False    # Kinetic binding
    Cyclic_model.root.input.model.unit_002.adsorption.LIN_KA = [
        35.5]  # m^3 / (mol * s)   (mobile phase)
    Cyclic_model.root.input.model.unit_002.adsorption.LIN_KD = [
        1]      # 1 / s (desorption)

    # To write out last output to check for steady state
    Cyclic_model.root.input['return'].WRITE_SOLUTION_LAST = True

    # % Input and connections
    # Sections
    Cyclic_model.root.input.solver.sections.nsec = 2
    Cyclic_model.root.input.solver.sections.section_times = [0, 100, 6000]

    # Feed and Eluent concentration
    Cyclic_model.root.input.model.unit_000.sec_000.const_coeff = [
        1]  # Inlet flowrate concentration

    Cyclic_model.root.input.model.unit_000.sec_001.const_coeff = [
        0]  # Inlet flowrate concentration

    # Connections
    Cyclic_model.root.input.model.connections.nswitches = 1

    Cyclic_model.root.input.model.connections.switch_000.section = 0
    Cyclic_model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, LRMP_Q3/2.0,  # flowrates, Q, m3/s
        1, 2, -1, -1, LRMP_Q3,
        2, 1, -1, -1, LRMP_Q3/2.0,
        2, 3, -1, -1, LRMP_Q3/2.0,
    ]

    # solution times
    Cyclic_model.root.input.solver.user_solution_times = np.linspace(
        1.0 if analytical_reference else 0.0, 6000.0, 6000 if analytical_reference else 6001)

    # Time
    # Tolerances for the time integrator
    Cyclic_model.root.input.solver.time_integrator.ABSTOL = 1e-12  # absolute tolerance
    Cyclic_model.root.input.solver.time_integrator.ALGTOL = 1e-10
    Cyclic_model.root.input.solver.time_integrator.RELTOL = 1e-10  # Relative tolerance
    Cyclic_model.root.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    Cyclic_model.root.input.solver.time_integrator.MAX_STEPS = 1000000

    # Solver options in general (not only for column although the same)
    Cyclic_model.root.input.model.solver.gs_type = 1
    Cyclic_model.root.input.model.solver.max_krylov = 0
    Cyclic_model.root.input.model.solver.max_restarts = 10
    Cyclic_model.root.input.model.solver.schur_safety = 1e-8
    # necessary specifically for this sim
    Cyclic_model.root.input.solver.consistent_init_mode = 5
    Cyclic_model.root.input.solver.time_integrator.USE_MODIFIED_NEWTON = 1

    # Number of cores for parallel simulation
    Cyclic_model.root.input.solver.nthreads = 1

    # Specify which results we want to return
    # Return data
    Cyclic_model.root.input['return'].split_components_data = 0
    Cyclic_model.root.input['return'].split_ports_data = 0
    Cyclic_model.root.input['return'].unit_000.write_solution_bulk = 0
    Cyclic_model.root.input['return'].unit_000.write_solution_inlet = 0
    Cyclic_model.root.input['return'].unit_000.write_solution_outlet = 0
    Cyclic_model.root.input['return'].unit_001.write_solution_bulk = 0
    Cyclic_model.root.input['return'].unit_001.write_solution_inlet = 0
    Cyclic_model.root.input['return'].unit_001.write_solution_outlet = 1

    # Copy settings to the other unit operations
    Cyclic_model.root.input['return'].unit_002 = Cyclic_model.root.input['return'].unit_001
    Cyclic_model.root.input['return'].unit_003 = Cyclic_model.root.input['return'].unit_001

    return Cyclic_model


def Acyclic_model1(nelem, polydeg, exactInt, analytical_reference=False):

    # Setting up the model
    Acyclic_model = Cadet()

    # Speciy number of unit operations: input, column and output, 3
    Acyclic_model.root.input.model.nunits = 7

    # Specify # of components (salt,proteins)
    n_comp = 1

    # First unit operation: inlet
    # Source 1
    Acyclic_model.root.input.model.unit_000.unit_type = 'INLET'
    Acyclic_model.root.input.model.unit_000.ncomp = n_comp
    Acyclic_model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # Source 2
    Acyclic_model.root.input.model.unit_001.unit_type = 'INLET'
    Acyclic_model.root.input.model.unit_001.ncomp = n_comp
    Acyclic_model.root.input.model.unit_001.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # Sink
    Acyclic_model.root.input.model.unit_006.ncomp = n_comp
    Acyclic_model.root.input.model.unit_006.unit_type = 'OUTLET'

    # Unit LRMP3
    Acyclic_model.root.input.model.unit_002.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES'
    Acyclic_model.root.input.model.unit_002.ncomp = n_comp

    # Geometry
    Acyclic_model.root.input.model.unit_002.col_porosity = 0.37
    Acyclic_model.root.input.model.unit_002.par_porosity = 0.75
    Acyclic_model.root.input.model.unit_002.col_dispersion = 2e-7
    Acyclic_model.root.input.model.unit_002.col_length = 1.4e-2
    Acyclic_model.root.input.model.unit_002.cross_section_area = 1
    Acyclic_model.root.input.model.unit_002.film_diffusion = 6.9e-6
    Acyclic_model.root.input.model.unit_002.par_radius = 45e-6
    LRMP_Q3 = 3.45*1e-2 / 60 * 0.37

    # Isotherm specification
    Acyclic_model.root.input.model.unit_002.adsorption_model = 'LINEAR'
    Acyclic_model.root.input.model.unit_002.adsorption.is_kinetic = True    # Kinetic binding
    Acyclic_model.root.input.model.unit_002.adsorption.LIN_KA = [
        3.55]  # m^3 / (mol * s)   (mobile phase)
    Acyclic_model.root.input.model.unit_002.adsorption.LIN_KD = [
        0.1]      # 1 / s (desorption)
    # Initial conditions
    Acyclic_model.root.input.model.unit_002.init_c = [0]
    Acyclic_model.root.input.model.unit_002.init_q = [
        0]  # salt starts at max capacity

    # Grid cells in column and particle: the most important ones - ensure grid-independent solutions
    Acyclic_model.root.input.model.unit_002.discretization.SPATIAL_METHOD = "DG"
    Acyclic_model.root.input.model.unit_002.discretization.NELEM = nelem

    # Polynomial order
    Acyclic_model.root.input.model.unit_002.discretization.POLYDEG = polydeg
    Acyclic_model.root.input.model.unit_002.discretization.EXACT_INTEGRATION = exactInt

    # Bound states - for zero the compound does not bind, >1 = multiple binding sites
    Acyclic_model.root.input.model.unit_002.discretization.nbound = np.ones(
        n_comp, dtype=int)

    Acyclic_model.root.input.model.unit_002.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
    Acyclic_model.root.input.model.unit_002.discretization.USE_ANALYTIC_JACOBIAN = 1
    Acyclic_model.root.input.model.unit_002.discretization.RECONSTRUCTION = 'WENO'
    Acyclic_model.root.input.model.unit_002.discretization.GS_TYPE = 1
    Acyclic_model.root.input.model.unit_002.discretization.MAX_KRYLOV = 0
    Acyclic_model.root.input.model.unit_002.discretization.MAX_RESTARTS = 10
    Acyclic_model.root.input.model.unit_002.discretization.SCHUR_SAFETY = 1.0e-8

    Acyclic_model.root.input.model.unit_002.discretization.weno.BOUNDARY_MODEL = 0
    Acyclic_model.root.input.model.unit_002.discretization.weno.WENO_EPS = 1e-10
    Acyclic_model.root.input.model.unit_002.discretization.weno.WENO_ORDER = 3

    # Copy column models
    Acyclic_model.root.input.model.unit_003 = copy.deepcopy(
        Acyclic_model.root.input.model.unit_002)
    Acyclic_model.root.input.model.unit_004 = copy.deepcopy(
        Acyclic_model.root.input.model.unit_002)
    Acyclic_model.root.input.model.unit_005 = copy.deepcopy(
        Acyclic_model.root.input.model.unit_002)

    # Unit LRMP4
    Acyclic_model.root.input.model.unit_003.col_length = 4.2e-2
    Acyclic_model.root.input.model.unit_003.adsorption.is_kinetic = False    # Kinetic binding
    Acyclic_model.root.input.model.unit_003.adsorption.LIN_KA = [
        35.5]  # m^3 / (mol * s)   (mobile phase)
    Acyclic_model.root.input.model.unit_003.adsorption.LIN_KD = [
        1]      # 1 / s (desorption)

    # Unit LRMP5
    Acyclic_model.root.input.model.unit_004.adsorption.is_kinetic = False    # Kinetic binding
    Acyclic_model.root.input.model.unit_004.adsorption.LIN_KA = [
        21.4286]  # m^3 / (mol * s)   (mobile phase)
    Acyclic_model.root.input.model.unit_004.adsorption.LIN_KD = [
        1]      # 1 / s (desorption)

    # Unit LRMP6
    Acyclic_model.root.input.model.unit_005.adsorption.LIN_KA = [
        4.55]  # m^3 / (mol * s)   (mobile phase)
    Acyclic_model.root.input.model.unit_005.adsorption.LIN_KD = [
        0.12]      # 1 / s (desorption)

    # To write out last output to check for steady state
    Acyclic_model.root.input['return'].WRITE_SOLUTION_LAST = True

    # % Input and connections
    # Sections
    Acyclic_model.root.input.solver.sections.nsec = 3
    Acyclic_model.root.input.solver.sections.section_times = [
        0, 250, 300, 3000]

    # Feed and Eluent concentration
    Acyclic_model.root.input.model.unit_000.sec_000.const_coeff = [
        1]  # Inlet flowrate concentration
    Acyclic_model.root.input.model.unit_001.sec_000.const_coeff = [
        1]  # Desorbent stream

    Acyclic_model.root.input.model.unit_000.sec_001.const_coeff = [
        0]  # Inlet flowrate concentration
    Acyclic_model.root.input.model.unit_001.sec_001.const_coeff = [
        5]  # Desorbent stream

    Acyclic_model.root.input.model.unit_000.sec_002.const_coeff = [
        0]  # Inlet flowrate concentration
    Acyclic_model.root.input.model.unit_001.sec_002.const_coeff = [
        0]  # Desorbent stream

    # Connections
    Acyclic_model.root.input.model.connections.nswitches = 1

    Acyclic_model.root.input.model.connections.switch_000.section = 0
    Acyclic_model.root.input.model.connections.switch_000.connections = [
        0, 2, -1, -1, LRMP_Q3,  # flowrates, Q, m3/s
        2, 4, -1, -1, LRMP_Q3/2,
        2, 5, -1, -1, LRMP_Q3/2,
        1, 3, -1, -1, LRMP_Q3,
        3, 4, -1, -1, LRMP_Q3/2,
        3, 5, -1, -1, LRMP_Q3/2,
        4, 6, -1, -1, LRMP_Q3,
        5, 6, -1, -1, LRMP_Q3,
    ]

    # solution times
    Acyclic_model.root.input.solver.user_solution_times = np.linspace(
        1.0 if analytical_reference else 0.0, 3000.0, 3000 if analytical_reference else 3001)

    # Time
    # Tolerances for the time integrator
    Acyclic_model.root.input.solver.time_integrator.ABSTOL = 1e-12  # absolute tolerance
    Acyclic_model.root.input.solver.time_integrator.ALGTOL = 1e-10
    Acyclic_model.root.input.solver.time_integrator.RELTOL = 1e-10  # Relative tolerance
    Acyclic_model.root.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    Acyclic_model.root.input.solver.time_integrator.MAX_STEPS = 1000000

    # Solver options in general (not only for column although the same)
    Acyclic_model.root.input.model.solver.gs_type = 1
    Acyclic_model.root.input.model.solver.max_krylov = 0
    Acyclic_model.root.input.model.solver.max_restarts = 10
    Acyclic_model.root.input.model.solver.schur_safety = 1e-8
    # necessary specifically for this sim
    Acyclic_model.root.input.solver.consistent_init_mode = 5
    Acyclic_model.root.input.solver.time_integrator.USE_MODIFIED_NEWTON = 1

    # Number of cores for parallel simulation
    Acyclic_model.root.input.solver.nthreads = 1

    # Specify which results we want to return
    # Return data
    Acyclic_model.root.input['return'].split_components_data = 0
    Acyclic_model.root.input['return'].split_ports_data = 0
    Acyclic_model.root.input['return'].unit_000.write_solution_bulk = 0
    Acyclic_model.root.input['return'].unit_000.write_solution_inlet = 0
    Acyclic_model.root.input['return'].unit_000.write_solution_outlet = 0
    Acyclic_model.root.input['return'].unit_002.write_solution_bulk = 0
    Acyclic_model.root.input['return'].unit_002.write_solution_inlet = 0
    Acyclic_model.root.input['return'].unit_002.write_solution_outlet = 1

    # Copy settings to the other unit operations
    Acyclic_model.root.input['return'].unit_001 = Acyclic_model.root.input['return'].unit_000
    Acyclic_model.root.input['return'].unit_003 = Acyclic_model.root.input['return'].unit_002
    Acyclic_model.root.input['return'].unit_004 = Acyclic_model.root.input['return'].unit_002
    Acyclic_model.root.input['return'].unit_005 = Acyclic_model.root.input['return'].unit_002
    Acyclic_model.root.input['return'].unit_006 = Acyclic_model.root.input['return'].unit_002

    return Acyclic_model
