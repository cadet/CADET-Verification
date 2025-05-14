# -*- coding: utf-8 -*-
'''
Created Mai 2025

This script implements the settings used for verification of the michaelis-menten
kinetics in CADET-Core.

@author: Antonia Berger
'''

import numpy as np
from scipy.integrate import solve_ivp

from cadet import Cadet


def create_base_system(model, ncomp, init_c):
    """Create a basic CSTR system with inlet, CSTR and outlet"""
    # CSTR
    model.root.input.model.nunits = 3

    # Inlet
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = ncomp
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # CSTR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = ncomp
    model.root.input.model.unit_001.init_liquid_volume = 1.0
    model.root.input.model.unit_001.init_c = init_c
    model.root.input.model.unit_001.const_solid_volume = 1.0
    model.root.input.model.unit_001.use_analytic_jacobian = 1

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = ncomp
    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 1
    model.root.input['return'].unit_000.write_solution_inlet = 1
    model.root.input['return'].unit_000.write_solution_outlet = 1

    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

def setup_solver(model, sim_time=300.0):
    """Configure solver settings"""
    model.root.input.solver.user_solution_times = np.linspace(0, sim_time, 1000)
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, sim_time]
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1

def setup_connections(model, ncomp):
    """Connect the units together"""
    # Connections
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, 0.0,  # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1]
        1, 2, -1, -1, 0.0   # [unit_001, unit_002, all components, all components, Q/ m^3*s^-1]
    ]

    # Inlet coefficients - no inflow
    model.root.input.model.unit_000.sec_000.const_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0] * ncomp

def complex_inhibition_system_cadet_settings(model, parameter):

    """
    Creates a model with complex inhibition kinetics.
    - reaction 1: A -> B and C is uncompetitive inhibitor
    - reaction 2: B + C -> D where A and E are noncompetitive inhibitors
    - reaction 3:  C-> E where B and D are competitive inhibitors
    """

    #0: A, 1: B, 2: C, 3: D, 4: E
    ncomp = parameter['ncomp']

    init_c = parameter['init_c']
    sim_time = parameter['sim_time']

    km_a = parameter['km_a']   # Km for A in reaction 1
    km_b = parameter['km_b']  # Km for B in reaction 2
    km_c_1 = parameter['km_c_1'] # Km for C in reaction 2
    km_c_2 = parameter['km_c_2']  # Km for C in reaction 3

    ki_c_uc = parameter['ki_c_uc']  # C competative inhibitor for reaction 1 for A
    ki_a_nc = parameter['ki_a_nc'] # A noncompetative inhibitor for reaction 2 for B

    ki_e_nc = parameter['ki_e_nc'] # E noncompetative inhibitor for reaction 2 for C
    ki_b_c = parameter['ki_b_c'] # B competitive inhibitor for reaction 3 for C
    ki_d_c = parameter['ki_d_c']  # D competitive inhibitor for reaction 3 for C

    vmax_1 = parameter['vmax_1']   # Vmax for reaction 1
    vmax_2 = parameter['vmax_2']   # Vmax for reaction 2
    vmax_3 = parameter['vmax_3']   # Vmax for reaction 3

    # Create the model
    create_base_system(model, ncomp, init_c)
    setup_solver(model, sim_time)
    setup_connections(model, ncomp)

    # Set up michaelis-menten kinetics

    model.root.input.model.unit_001.reaction_model = 'MICHAELIS_MENTEN'

        #0: A, 1: B, 2: C, 3: D, 4: E
    model.root.input.model.unit_001.reaction_bulk.mm_kmm = [
        [km_a, 0.0, 0.0, 0.0, 0.0],
        [0.0, km_b, km_c_1, 0.0, 0.0],
        [0.0, 0.0, km_c_2, 0.0, 0.0]
    ]

    model.root.input.model.unit_001.reaction_bulk.mm_ki_c = [
        # Reaction 1: A -> B (no competitive inhibition)
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        # Reaction 2: B + C -> D ( noncompetitive inhibition)
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [ki_a_nc, 0.0, 0.0, 0.0, 0.0],# B is inhibited by A noncompetitively
            [0.0, 0.0, 0.0, 0.0, ki_e_nc],# C is inhibited by E noncompetitively
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        # Reaction 3: C -> E (B and D are competitive inhibitors)
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, ki_b_c, 0.0, ki_d_c, 0.0],# C is inhibited by B and D competitively
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]
    ]

    # Uncompetitive inhibition setup
    model.root.input.model.unit_001.reaction_bulk.mm_ki_uc = [
        # Reaction 1: A -> B (C is uncompetitive inhibitor)
        [
            [0.0, 0.0, ki_c_uc, 0.0, 0.0], # A is inhibited by C uncompetitively
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        # Reaction 2: B + C -> D (no uncompetitive inhibition)
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [ki_a_nc, 0.0, 0.0, 0.0, 0.0],# B is inhibited by A noncompetitively
            [0.0, 0.0, 0.0, 0.0, ki_e_nc],# C is inhibited by E noncompetitively
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        # Reaction 3: C -> E (no uncompetitive inhibition)
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]
    ]

    # Vmax values for the reactions
    model.root.input.model.unit_001.reaction_bulk.mm_vmax = [ vmax_1, vmax_2, vmax_3 ]

    # Stoichiometry matrix for bulk reactions
    model.root.input.model.unit_001.reaction_bulk.mm_stoichiometry_bulk = [
        # r1  r2 r3
        [-1, 0, 0],   # A
        [ 1,-1, 0],   # B
        [ 0,-1,-1],   # C
        [ 0, 1, 0],   # D
        [ 0, 0, 1]    # E
    ]

    return model

def complex_inhibition_system_ode(parameter):
    """
    Creates a model with complex inhibition kinetics.
    - reaction 1: A -> B and C is competitive
    - reaction 2: B + C -> D where A and E are noncompetitive inhibitors
    - reaction 3  C-> E where B and D are competitive inhibitors
    """

    #0: A, 1: B, 2: C, 3: D, 4: E
    km_a = parameter['km_a']   # Km for A in reaction 1
    km_b = parameter['km_b']  # Km for B in reaction 2
    km_c_1 = parameter['km_c_1'] # Km for C in reaction 2
    km_c_2 = parameter['km_c_2']  # Km for C in reaction 3

    ki_c_uc = parameter['ki_c_uc']  # C uncompetative inhibitor for reaction 1 for A
    ki_a_nc = parameter['ki_a_nc'] # A noncompetative inhibitor for reaction 2 for B

    ki_e_nc = parameter['ki_e_nc'] # E noncompetative inhibitor for reaction 2 for C
    ki_b_c = parameter['ki_b_c'] # B competitive inhibitor for reaction 3 for C
    ki_d_c = parameter['ki_d_c']  # D competitive inhibitor for reaction 3 for C

    vmax_1 = parameter['vmax_1']   # Vmax for reaction 1
    vmax_2 = parameter['vmax_2']   # Vmax for reaction 2
    vmax_3 = parameter['vmax_3']   # Vmax for reaction 3

    def model_ode(t, y):
        """
        ODE system for the complex inhibition kinetics.
        """
        A, B, C, D, E = y

        # Reaction rates
        # Reaction 1: A -> B with uncompetitive inhibition by C
        r1  = vmax_1 * A / (km_a + A*(1 + C / ki_c_uc))

        # Reaction 2: B + C -> D with noncompetitive inhibition by A and E
        r2 = vmax_2 * B * C / ((km_b + B) * (1 + A/ki_a_nc) * (km_c_1 + C) * (1 + E/ki_e_nc))

        # Reaction 3: C -> E with competitive inhibition by B and D
        r3 = vmax_3 * C / (km_c_2 * (1 + B/ki_b_c + D/ki_d_c) + C)

        return [ - r1 , r1 - r2, -r2 - r3, r2, r3 ]

    return model_ode
