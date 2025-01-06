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
    # From Lubke2007, is not important
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
    Cyclic_model.root.input.model.unit_003.adsorption.is_kinetic = False    # Kinetic binding
    Cyclic_model.root.input.model.unit_003.adsorption.LIN_KA = [
        35.5]  # m^3 / (mol * s)   (mobile phase)
    Cyclic_model.root.input.model.unit_003.adsorption.LIN_KD = [
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
    # From Lubke2007, is not important
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
    Acyclic_model.root.input.model.unit_004.adsorption.LIN_KA = [
        4.55]  # m^3 / (mol * s)   (mobile phase)
    Acyclic_model.root.input.model.unit_004.adsorption.LIN_KD = [
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
        0]  # Desorbent stream

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
