# -*- coding: utf-8 -*-
"""
Study 3: Radial 1D LRM, 2 components, Langmuir rapid-equilibrium — oscillation study.

Model: RADIAL_COLUMN_MODEL_1D (npartype=1, no film/pore diffusion)
Domain: r in [0.01, 1.01]
Total porosity = 0.4

Binding: Multi-component Langmuir rapid-eq
  KA=[0.1, 0.05], KD=[1.0, 1.0], QMAX=[10.0, 10.0]

Constant dispersion D0 (parameterized: 1e-4 or 1e-5).
Matches Breuer's thesis Section 5.3 (axial LRM Langmuir oscillation study).

Velocity: v_code = 0.1 * rho_mid
  where rho_mid = sqrt(r_in^2 + 0.5*(r_out^2 - r_in^2))

Sections: [0, 12, 40], c=[10,10] then c=[0,0]. 4001 solution times.
"""

import numpy as np
from addict import Dict


def get_model(D0=1e-4):

    ncomp = 2
    r_in = 0.01
    r_out = 1.01

    # Velocity matching v_mid = 0.1 m/s in radial geometry
    rho_mid = np.sqrt(r_in**2 + 0.5 * (r_out**2 - r_in**2))
    v_code = 0.1 * rho_mid

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

    # Inlet — 2 sections
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = ncomp
    # Section 0: [0, 12] — c = [10, 10]
    m.input.model.unit_000.sec_000.const_coeff = [10.0, 10.0]
    m.input.model.unit_000.sec_000.lin_coeff   = [0.0] * ncomp
    m.input.model.unit_000.sec_000.quad_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_000.cube_coeff  = [0.0] * ncomp
    # Section 1: [12, 40] — c = [0, 0]
    m.input.model.unit_000.sec_001.const_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_001.lin_coeff   = [0.0] * ncomp
    m.input.model.unit_000.sec_001.quad_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_001.cube_coeff  = [0.0] * ncomp

    # Radial column — LRM without pores
    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_COLUMN_MODEL_1D'
    col.ncomp = ncomp
    col.npartype = 1
    col.col_radius_inner = r_in
    col.col_radius_outer = r_out
    col.total_porosity = 0.4
    col.col_porosity = 0.4
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [D0] * ncomp
    col.velocity_coeff = v_code
    col.init_c = [0.0] * ncomp

    # Binding: Multi-component Langmuir rapid-eq
    col.particle_type_000.has_film_diffusion = 0
    col.particle_type_000.has_pore_diffusion = 0
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.nbound = [1] * ncomp
    col.particle_type_000.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    col.particle_type_000.adsorption.is_kinetic = 0
    col.particle_type_000.adsorption.mcl_ka = [0.1, 0.05]
    col.particle_type_000.adsorption.mcl_kd = [1.0, 1.0]
    col.particle_type_000.adsorption.mcl_qmax = [10.0, 10.0]
    col.particle_type_000.init_cp = [0.0] * ncomp
    col.particle_type_000.init_cs = [0.0] * ncomp

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
    m.input.solver.sections.nsec = 2
    m.input.solver.sections.section_continuity = [0]
    m.input.solver.sections.section_times = [0.0, 12.0, 40.0]
    m.input.solver.time_integrator.abstol = 1e-12
    m.input.solver.time_integrator.reltol = 1e-10
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-6
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, 40.0, 4001)

    return m
