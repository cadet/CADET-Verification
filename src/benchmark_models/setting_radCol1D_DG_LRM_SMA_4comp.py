# -*- coding: utf-8 -*-
"""
Study 1, Benchmark 2: Radial 1D LRM, 4 components, SMA kinetic binding.

Model: RADIAL_COLUMN_MODEL_1D (npartype=1, no film/pore diffusion)
Domain: r in [0.01, 1.01]
VELOCITY_COEFF = 2/60, COL_POROSITY = 0.6

Binding: SMA kinetic
  Lambda=1200, KA=[0,35.5,1.59,7.7], KD=[0,1e3,1e3,1e3]
  NU=[0,4.7,5.29,3.7], SIGMA=[0,11.83,10.6,10.0]

Variable dispersion: D(rho) = 1e-4 * (r_in / rho)
  via RADIAL_RECIPROCAL_POWER_LAW with exponent=1

3 sections:
  [0,10]: c=[50,1,1,1]
  [10,60]: c=[50,0,0,0]
  [60,130]: c=[100,0,0,0]

Init: c=[50,0,0,0], q=[1200,0,0,0]
t_final = 130, 1301 solution times.
"""

import numpy as np
from addict import Dict


def get_model():

    ncomp = 4
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

    # Inlet — 3 sections
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = ncomp
    # Section 0: [0, 10] — c = [50, 1, 1, 1]
    m.input.model.unit_000.sec_000.const_coeff = [50.0, 1.0, 1.0, 1.0]
    m.input.model.unit_000.sec_000.lin_coeff   = [0.0] * ncomp
    m.input.model.unit_000.sec_000.quad_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_000.cube_coeff  = [0.0] * ncomp
    # Section 1: [10, 60] — c = [50, 0, 0, 0]
    m.input.model.unit_000.sec_001.const_coeff = [50.0, 0.0, 0.0, 0.0]
    m.input.model.unit_000.sec_001.lin_coeff   = [0.0] * ncomp
    m.input.model.unit_000.sec_001.quad_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_001.cube_coeff  = [0.0] * ncomp
    # Section 2: [60, 130] — c = [100, 0, 0, 0]
    m.input.model.unit_000.sec_002.const_coeff = [100.0, 0.0, 0.0, 0.0]
    m.input.model.unit_000.sec_002.lin_coeff   = [0.0] * ncomp
    m.input.model.unit_000.sec_002.quad_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_002.cube_coeff  = [0.0] * ncomp

    # Radial column — LRM without pores
    r_in = 0.01
    r_out = 1.01

    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_COLUMN_MODEL_1D'
    col.ncomp = ncomp
    col.npartype = 1
    col.col_radius_inner = r_in
    col.col_radius_outer = r_out
    col.col_porosity = 0.6
    col.total_porosity = 0.6
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [1e-4] * ncomp
    col.velocity_coeff = 2.0 / 60.0
    col.init_c = [50.0, 0.0, 0.0, 0.0]

    # Variable dispersion: D(rho) = D0 * (r_in / rho)
    col.col_dispersion_dep = 'RADIAL_RECIPROCAL_POWER_LAW'
    col.col_dispersion_dep_base = 1.0
    col.col_dispersion_dep_exponent = 1.0
    col.col_dispersion_dep_rinner = r_in
    col.col_dispersion_dep_length = r_out - r_in

    # Binding: SMA kinetic
    col.particle_type_000.has_film_diffusion = 0
    col.particle_type_000.has_pore_diffusion = 0
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.nbound = [1] * ncomp
    col.particle_type_000.adsorption_model = 'STERIC_MASS_ACTION'
    col.particle_type_000.adsorption.is_kinetic = 1
    col.particle_type_000.adsorption.sma_lambda = 1200.0
    col.particle_type_000.adsorption.sma_ka = [0.0, 35.5, 1.59, 7.7]
    col.particle_type_000.adsorption.sma_kd = [0.0, 1000.0, 1000.0, 1000.0]
    col.particle_type_000.adsorption.sma_nu = [0.0, 4.7, 5.29, 3.7]
    col.particle_type_000.adsorption.sma_sigma = [0.0, 11.83, 10.6, 10.0]
    col.particle_type_000.init_cp = [50.0, 0.0, 0.0, 0.0]
    col.particle_type_000.init_cs = [1200.0, 0.0, 0.0, 0.0]

    col.discretization.USE_ANALYTIC_JACOBIAN = 1

    # Outlet
    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = ncomp

    # Return config
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0

    # Solver
    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 3
    m.input.solver.sections.section_continuity = [0, 0]
    m.input.solver.sections.section_times = [0.0, 10.0, 60.0, 130.0]
    m.input.solver.time_integrator.abstol = 1e-12
    m.input.solver.time_integrator.reltol = 1e-10
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-6
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, 130.0, 1301)

    return m
