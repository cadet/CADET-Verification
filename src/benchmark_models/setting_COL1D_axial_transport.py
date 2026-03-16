# -*- coding: utf-8 -*-
"""

This script defines an axial 1D transport model, 1 component, no particles

"""

from addict import Dict
import numpy as np

def get_model():
    
    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, 1.0,
        1.0, 2.0, -1.0, -1.0, 1.0,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 1
    m.input.model.unit_000.sec_000.const_coeff = [1.0]
    m.input.model.unit_000.sec_000.lin_coeff   = [0.0]
    m.input.model.unit_000.sec_000.quad_coeff  = [0.0]
    m.input.model.unit_000.sec_000.cube_coeff  = [0.0]
    m.input.model.unit_000.sec_001.const_coeff = [0.0]
    m.input.model.unit_000.sec_001.lin_coeff   = [0.0]
    m.input.model.unit_000.sec_001.quad_coeff  = [0.0]
    m.input.model.unit_000.sec_001.cube_coeff  = [0.0]

    column = m.input.model.unit_001
    column.unit_type = 'COLUMN_MODEL_1D'
    column.ncomp = 1
    column.npartype = 0
    column.col_length = 1.0
    column.col_porosity = 1.0
    column.total_porosity = 1.0
    column.col_dispersion = [1e-4]
    column.velocity = 0.01
    column.init_c = [0.0]

    disc = column.discretization
    disc.USE_ANALYTIC_JACOBIAN = 1
    # disc.NCOL = ncol
    # disc.SPATIAL_METHOD = 'FV'
    # disc.RECONSTRUCTION = reconstruction
    # disc.weno.WENO_ORDER = weno_order
    # disc.weno.WENO_EPS = 1e-10
    # disc.weno.BOUNDARY_MODEL = 0
    # disc.koren.KOREN_EPS = 1e-10
    # if grid_faces is not None:
    #     disc.GRID_FACES = grid_faces.tolist()

    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = 1

    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0

    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 2
    m.input.solver.sections.section_continuity = [0]
    m.input.solver.sections.section_times = [0.0, 10.0, 200.0]
    m.input.solver.time_integrator.ABSTOL = 1e-12
    m.input.solver.time_integrator.RELTOL = 1e-8
    m.input.solver.time_integrator.ALGTOL = 1e-12
    m.input.solver.time_integrator.INIT_STEP_SIZE = 1e-6
    m.input.solver.time_integrator.MAX_STEPS = 100
    m.input.solver.user_solution_times = np.linspace(0.0, 200.0, 401)

    return m

