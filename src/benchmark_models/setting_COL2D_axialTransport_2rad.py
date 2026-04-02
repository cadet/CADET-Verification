# -*- coding: utf-8 -*-
"""

This script defines a 2D GRM transport model,
1 component, minimal particles (no binding, no pore diffusion).

"""

from addict import Dict
import numpy as np

def get_model():

    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.connections_include_ports = 1
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0,
        1.0, 2.0, 0.0, 0.0, -1.0, -1.0, 1.0,
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
    column.unit_type = 'COLUMN_MODEL_2D'
    column.ncomp = 1
    column.npartype = 1
    column.col_length = 1.0
    column.col_radius = 0.01
    column.cross_section_area = 3.14159265358979e-4
    column.col_dispersion_axial = [1e-4]
    column.col_dispersion_radial = [1e-6]
    column.col_porosity = 0.4
    column.velocity = 0.01
    column.init_c = [0.0]
    column.par_type_volfrac = 1.0

    disc = column.discretization
    disc.USE_ANALYTIC_JACOBIAN = 1
    disc.NRAD = 2
    disc.RADIAL_DISC_TYPE = 'EQUIDISTANT'
    disc.GS_TYPE = 1
    disc.MAX_KRYLOV = 0
    disc.MAX_RESTARTS = 10
    disc.SCHUR_SAFETY = 1e-8
    disc.PAR_DISC_TYPE = ['EQUIDISTANT']

    par = column.particle_type_000
    par.has_film_diffusion = 1 # only grm implemented for FV
    par.has_pore_diffusion = 1
    par.has_surface_diffusion = 0
    par.film_diffusion = 0.0
    par.pore_diffusion = 0.0
    par.par_porosity = 0.5
    par.par_radius = 4.5e-5
    par.nbound = [0]
    par.init_cp = [0.0]
    par.pore_diffusion = [0.0]
    par.surface_diffusion = [0.0]
    par.adsorption_model = 'NONE'

    par_disc = par.discretization
    par_disc.SPATIAL_METHOD = 'FV'
    par_disc.NCELLS = 1

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
    m.input.solver.time_integrator.ABSTOL = 1e-10
    m.input.solver.time_integrator.RELTOL = 1e-8
    m.input.solver.time_integrator.ALGTOL = 1e-10
    m.input.solver.time_integrator.INIT_STEP_SIZE = 1e-6
    m.input.solver.time_integrator.MAX_STEPS = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, 200.0, 401)

    return m
