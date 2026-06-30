# -*- coding: utf-8 -*-
"""
Radial 1D column, 4 components, SMA kinetic binding.

Single parameterized setting for the radial SMA EOC study. The particle model is
selected via ``particle_type``:

  'EQUILIBRIUM_PARTICLE'  -> rLRM  (no film/pore diffusion; variable radial dispersion)
  'HOMOGENEOUS_PARTICLE'  -> rLRMP (film diffusion, no pore diffusion)
  'GENERAL_RATE_PARTICLE' -> rGRM  (film + pore diffusion; particle DG/FV discretization)

Each particle model keeps its own column geometry, velocity, dispersion and section
timing (see the per-type blocks below). The setting is used by both the DG and FV
branches of the radial EOC harness (src/radialDG.py), hence no method in the name.

Binding (all types): SMA kinetic
  Lambda=1200, KA=[0,35.5,1.59,7.7], KD=[0,1e3,1e3,1e3]
  NU=[0,4.7,5.29,3.7], SIGMA=[0,11.83,10.6,10.0]

Inlet (all types), 3 sections:
  sec_000: c=[50, 1, 1, 1]
  sec_001: c=[50, 0, 0, 0]
  sec_002: c=[100, 0, 0, 0]
Init: c=[50, 0, 0, 0], q=[1200, 0, 0, 0]
"""

import numpy as np
from addict import Dict


def get_model(particle_type='GENERAL_RATE_PARTICLE', col_dispersion_dep=None):

    valid = ('EQUILIBRIUM_PARTICLE', 'HOMOGENEOUS_PARTICLE', 'GENERAL_RATE_PARTICLE')
    if particle_type not in valid:
        raise ValueError(
            f"particle_type must be one of {valid}, got {particle_type!r}")

    ncomp = 4
    r_in = 0.01

    m = Dict()

    # Connection flow rate differs per setup
    flow = 6e-5 if particle_type == 'GENERAL_RATE_PARTICLE' else 1.0

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, flow,
        1.0, 2.0, -1.0, -1.0, flow,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # Inlet — 3 sections (shared)
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = ncomp
    m.input.model.unit_000.sec_000.const_coeff = [50.0, 1.0, 1.0, 1.0]
    m.input.model.unit_000.sec_000.lin_coeff   = [0.0] * ncomp
    m.input.model.unit_000.sec_000.quad_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_000.cube_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_001.const_coeff = [50.0, 0.0, 0.0, 0.0]
    m.input.model.unit_000.sec_001.lin_coeff   = [0.0] * ncomp
    m.input.model.unit_000.sec_001.quad_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_001.cube_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_002.const_coeff = [100.0, 0.0, 0.0, 0.0]
    m.input.model.unit_000.sec_002.lin_coeff   = [0.0] * ncomp
    m.input.model.unit_000.sec_002.quad_coeff  = [0.0] * ncomp
    m.input.model.unit_000.sec_002.cube_coeff  = [0.0] * ncomp

    # Radial column — geometry/transport depend on the particle model
    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_COLUMN_MODEL_1D'
    col.ncomp = ncomp
    col.npartype = 1
    col.col_radius_inner = r_in
    col.par_type_volfrac = [1.0]
    col.init_c = [50.0, 0.0, 0.0, 0.0]

    pt = col.particle_type_000

    if particle_type == 'EQUILIBRIUM_PARTICLE':
        # rLRM — no film/pore diffusion, variable radial dispersion
        r_out = 1.01
        col.col_radius_outer = r_out
        col.col_porosity = 0.6
        col.total_porosity = 0.6
        col.col_dispersion = [1e-4] * ncomp
        col.velocity_coeff = 2.0 / 60.0
        # Variable dispersion: D(rho) = D0 * (r_in / rho)
        if col_dispersion_dep is not None:
            if not col_dispersion_dep == 'RADIAL_RECIPROCAL_POWER_LAW':
                raise ValueError(
                    f"Only col_dispersion_dep implemented is 'RADIAL_RECIPROCAL_POWER_LAW', "
                    f"got {col_dispersion_dep!r}")
            col.col_dispersion_dep = 'RADIAL_RECIPROCAL_POWER_LAW'
            col.col_dispersion_dep_base = 1.0
            col.col_dispersion_dep_exponent = 1.0
            col.col_dispersion_dep_rinner = r_in
            col.col_dispersion_dep_length = r_out - r_in

        pt.has_film_diffusion = 0
        pt.has_pore_diffusion = 0
        pt.has_surface_diffusion = 0

    elif particle_type == 'HOMOGENEOUS_PARTICLE':
        # rLRMP — film diffusion, no pore diffusion
        col.col_radius_outer = 0.26
        col.col_porosity = 0.6
        col.total_porosity = 0.68
        col.col_dispersion = [1e-5] * ncomp
        col.velocity_coeff = 4.5e-3

        pt.has_film_diffusion = 1
        pt.has_pore_diffusion = 0
        pt.has_surface_diffusion = 0
        pt.par_porosity = 0.2
        pt.par_radius = 1e-4
        pt.film_diffusion = [3.3e-3] * ncomp

    else:  # GENERAL_RATE_PARTICLE
        # rGRM — film + pore diffusion
        col.col_radius_outer = 0.024
        col.col_porosity = 0.37
        col.col_dispersion = [5.75e-8] * ncomp
        col.velocity_coeff = 9.775e-6

        pt.has_film_diffusion = 1
        pt.has_pore_diffusion = 1
        pt.has_surface_diffusion = 0
        pt.par_porosity = 0.75
        pt.par_radius = 4.5e-5
        pt.par_coreradius = 0.0
        pt.par_geom = 'SPHERE'
        pt.film_diffusion = [6.9e-6] * ncomp
        pt.pore_diffusion = [7e-10, 6.07e-11, 6.07e-11, 6.07e-11]
        pt.surface_diffusion = [0.0] * ncomp

    # Particle bound state + SMA binding (shared)
    pt.nbound = [1] * ncomp
    pt.init_cp = [50.0, 0.0, 0.0, 0.0]
    pt.init_cs = [1200.0, 0.0, 0.0, 0.0]
    pt.adsorption_model = 'STERIC_MASS_ACTION'
    pt.adsorption.is_kinetic = 1
    pt.adsorption.sma_lambda = 1200.0
    pt.adsorption.sma_ka = [0.0, 35.5, 1.59, 7.7]
    pt.adsorption.sma_kd = [0.0, 1000.0, 1000.0, 1000.0]
    pt.adsorption.sma_nu = [0.0, 4.7, 5.29, 3.7]
    pt.adsorption.sma_sigma = [0.0, 11.83, 10.6, 10.0]

    # Particle discretization (GRM only — required for the DG/FV particle solve)
    if particle_type == 'GENERAL_RATE_PARTICLE':
        pt.discretization.PAR_POLYDEG = 3
        pt.discretization.PAR_NELEM = 1
        pt.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        pt.discretization.NCELLS = 10  # FV particle discretization (ignored by DG)

    col.discretization.USE_ANALYTIC_JACOBIAN = 1

    # Outlet (shared)
    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = ncomp

    # Return config (shared)
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0

    # Solver / sections — timing depends on the particle model
    if particle_type == 'EQUILIBRIUM_PARTICLE':
        section_times = [0.0, 10.0, 60.0, 130.0]
        t_final = 130.0
        n_times = 1301
        init_step = 1e-6
    elif particle_type == 'HOMOGENEOUS_PARTICLE':
        section_times = [0.0, 10.0, 90.0, 1500.0]
        t_final = 1500.0
        n_times = 1501
        init_step = 1e-6
    else:  # GENERAL_RATE_PARTICLE
        section_times = [0.0, 10.0, 90.0, 1500.0]
        t_final = 1500.0
        n_times = 1501
        init_step = 1e-8

    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 3
    m.input.solver.sections.section_continuity = [0, 0]
    m.input.solver.sections.section_times = section_times
    m.input.solver.time_integrator.abstol = 1e-12
    m.input.solver.time_integrator.reltol = 1e-10
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = init_step
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, t_final, n_times)

    return m
