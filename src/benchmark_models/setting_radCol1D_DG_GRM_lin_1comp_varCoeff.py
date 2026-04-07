# -*- coding: utf-8 -*-
"""
Study 2, Config 1: Radial GRM, 1 component, Linear kinetic, variable coefficients.

Model: RADIAL_COLUMN_MODEL_1D (GENERAL_RATE_PARTICLE)
Domain: r in [0.01, 0.024]
VELOCITY_COEFF = 6.13e-4, COL_DISPERSION = 5.75e-8, COL_POROSITY = 0.37

Binding: LINEAR kinetic (KA=3.55, KD=0.1)
Particle: eps_p=0.75, Rp=4.5e-5, kf=6.9e-6, Dp=6.07e-10

Variable D(rho) = D0 * (r_in/rho) via RADIAL_RECIPROCAL_POWER_LAW, exponent=1
Variable kf(rho) = kf0 * (r_in/rho)^(1/3) via RADIAL_RECIPROCAL_POWER_LAW, exponent=1/3

Sections: [0, 10, 1500], c=1 then c=0.
"""

import numpy as np
from addict import Dict


def get_model():

    r_in = 0.01
    r_out = 0.024

    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, 6e-5,
        1.0, 2.0, -1.0, -1.0, 6e-5,
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

    # Radial column — GRM
    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_COLUMN_MODEL_1D'
    col.ncomp = 1
    col.npartype = 1
    col.col_radius_inner = r_in
    col.col_radius_outer = r_out
    col.col_porosity = 0.37
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [5.75e-8]
    col.velocity_coeff = 6.13e-4
    col.init_c = [0.0]

    # Variable dispersion: D(rho) = D0 * (r_in / rho)
    col.col_dispersion_dep = 'RADIAL_RECIPROCAL_POWER_LAW'
    col.col_dispersion_dep_base = 1.0
    col.col_dispersion_dep_exponent = 1.0
    col.col_dispersion_dep_rinner = r_in
    col.col_dispersion_dep_length = r_out - r_in

    # Particle: GRM with pore diffusion
    pt = col.particle_type_000
    pt.has_film_diffusion = 1
    pt.has_pore_diffusion = 1
    pt.has_surface_diffusion = 0
    pt.par_porosity = 0.75
    pt.par_radius = 4.5e-5
    pt.par_coreradius = 0.0
    pt.par_geom = 'SPHERE'
    pt.film_diffusion = [6.9e-6]
    pt.pore_diffusion = [6.07e-10]
    pt.surface_diffusion = [0.0]
    pt.nbound = [1]
    pt.init_cp = [0.0]
    pt.init_cs = [0.0]

    # Binding: Linear kinetic
    pt.adsorption_model = 'LINEAR'
    pt.adsorption.is_kinetic = 1
    pt.adsorption.lin_ka = [3.55]
    pt.adsorption.lin_kd = [0.1]

    # Particle discretization (required for GRM DG)
    pt.discretization.PAR_POLYDEG = 3
    pt.discretization.PAR_NELEM = 1
    pt.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
    pt.discretization.NCELLS = 10  # FV particle discretization (ignored by DG)

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
    m.input.solver.sections.section_times = [0.0, 10.0, 1500.0]
    m.input.solver.time_integrator.abstol = 1e-12
    m.input.solver.time_integrator.reltol = 1e-10
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-8
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, 1500.0, 1501)

    return m
