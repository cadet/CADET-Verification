# -*- coding: utf-8 -*-
"""
Study 2, Config 3: Radial LRM (without pores), 1 component, Linear rapid-eq, variable D.

Model: RADIAL_COLUMN_MODEL_1D (no pore phase)
Domain: r in [0.01, 1.01]
VELOCITY_COEFF = 2.381e-2, COL_DISPERSION = 1e-4, COL_POROSITY = 0.6

Binding: LINEAR rapid-eq (KA=1.0, KD=1.0)

Variable D(rho) = D0 * (r_in/rho) via RADIAL_RECIPROCAL_POWER_LAW, exponent=1

Sections: [0, 60, 130], c=1 then c=0.
"""

import numpy as np
from addict import Dict


def get_model():

    r_in = 0.01
    r_out = 1.01

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

    # Inlet
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

    # Radial column — LRM without pores
    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_COLUMN_MODEL_1D'
    col.ncomp = 1
    col.npartype = 1
    col.col_radius_inner = r_in
    col.col_radius_outer = r_out
    col.col_porosity = 0.6
    col.total_porosity = 0.6
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [1e-4]
    col.velocity_coeff = 2.381e-2
    col.init_c = [0.0]

    # Variable dispersion
    col.col_dispersion_dep = 'RADIAL_RECIPROCAL_POWER_LAW'
    col.col_dispersion_dep_base = 1.0
    col.col_dispersion_dep_exponent = 1.0
    col.col_dispersion_dep_rinner = r_in
    col.col_dispersion_dep_length = r_out - r_in

    # Binding: Linear rapid-eq
    col.particle_type_000.has_film_diffusion = 0
    col.particle_type_000.has_pore_diffusion = 0
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.nbound = [1]
    col.particle_type_000.adsorption_model = 'LINEAR'
    col.particle_type_000.adsorption.is_kinetic = 0
    col.particle_type_000.adsorption.lin_ka = [1.0]
    col.particle_type_000.adsorption.lin_kd = [1.0]
    col.particle_type_000.init_cp = [0.0]
    col.particle_type_000.init_cs = [0.0]

    col.discretization.USE_ANALYTIC_JACOBIAN = 1

    # Outlet
    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = 1

    # Return config
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0

    # Solver
    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 2
    m.input.solver.sections.section_continuity = [0]
    m.input.solver.sections.section_times = [0.0, 60.0, 130.0]
    m.input.solver.time_integrator.abstol = 1e-12
    m.input.solver.time_integrator.reltol = 1e-10
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-6
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, 130.0, 1301)

    return m
