# -*- coding: utf-8 -*-
"""
Validation setting: radial membrane adsorber tracer pulses.

Reproduces the two tracer experiments from:
  Ladd Effio et al. (2016), J. Chromatogr. A 1429, 142–154.
  https://doi.org/10.1016/j.chroma.2015.12.006
  Fig. 3.

Device: Sartobind Q nano membrane adsorber, 3 mL.
  r_inner = 3.25 mm, r_outer = 11.25 mm (Effio's reported value),
  H = 8 mm (membrane height).
Flow: inward, F = 3 mL/min.
Pulse injection: 100 µL at 3 mL/min => t_inj = 2 s.
  Section 0: [0, 2]   c = 1  (injection)
  Section 1: [2, 120] c = 0  (elution)

Two configurations:

  Fig. 3a — Dextran T2000 (get_model_dextran):
    Large tracer, excluded from the hydrogel. Pure radial transport,
    RADIAL_COLUMN_MODEL_1D with npartype=0. Only sees the interstitial
    (membrane pore) volume. DG P=5, 512 elements.

  Fig. 3b — NaCl (get_model_nacl):
    Small tracer, enters both interstitial volume and hydrogel pores.
    RADIAL_LUMPED_RATE_MODEL_WITH_PORES, film diffusion, no binding.
      k_eff,NaCl = 0.1829 s^{-1}  (Effio et al., Eq. 4)
      Mapped to CADET LRMP:  3*kf/R_p = k_eff  =>  kf = k_eff * R_p / 3
    DG P=5, 256 elements.

Shared parameters (SI):
  r_inner        = 3.25e-3 m
  r_outer        = 11.25e-3 m
  col_porosity   = 0.518   (interstitial / membrane porosity)
  D_ax           = 3.74e-8 m^2/s
  velocity_coeff = -1.9204e-6 m^2/s  (inward)

NaCl-only parameters:
  par_porosity   = 0.581   (hydrogel porosity)
  R_p            = 4.5e-5 m  (fictitious particle radius)
  kf             = 2.7435e-6 m/s
"""

import numpy as np
from addict import Dict


# Shared geometry / flow
R_IN = 3.25e-3      # inner radius [m]
R_OUT = 11.25e-3    # outer radius [m], Effio's reported value
F = 5.0e-8          # volumetric flow rate [m^3/s]  (3 mL/min)
VELOCITY_COEFF = -1.9204e-6  # inward radial flow [m^2/s]
D_AX = 3.74e-8      # axial (radial) dispersion [m^2/s]
COL_POROSITY = 0.518


def _base_model(npartype):
    """Common inlet / connections / solver setup for both configs."""
    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, F,
        1.0, 2.0, -1.0, -1.0, F,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # Inlet — pulse injection
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

    # Radial column — shared bulk transport parameters
    col = m.input.model.unit_001
    col.ncomp = 1
    col.npartype = npartype
    col.col_radius_inner = R_IN
    col.col_radius_outer = R_OUT
    col.col_porosity = COL_POROSITY
    col.col_dispersion = [D_AX]
    col.velocity_coeff = VELOCITY_COEFF
    col.init_c = [0.0]

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
    m.input.solver.sections.section_times = [0.0, 2.0, 120.0]
    m.input.solver.time_integrator.abstol = 1e-12
    m.input.solver.time_integrator.reltol = 1e-10
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-8
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, 120.0, 2401)

    return m


def get_model_dextran():
    """Fig. 3a — Dextran T2000, pure radial transport (npartype=0)."""
    m = _base_model(npartype=0)

    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_COLUMN_MODEL_1D'
    col.total_porosity = COL_POROSITY  # dextran excluded from hydrogel

    disc = col.discretization
    disc.USE_ANALYTIC_JACOBIAN = 1
    disc.SPATIAL_METHOD = 'DG'
    disc.POLYDEG = 5
    disc.NELEM = 512

    return m


def get_model_nacl():
    """Fig. 3b — NaCl, radial LRMP with film diffusion, no binding."""
    R_p = 4.5e-5           # fictitious particle radius [m]
    k_eff_NaCl = 0.1829    # lumped effective mass transfer rate [1/s]
    kf = k_eff_NaCl * R_p / 3.0  # = 2.7435e-6 [m/s]

    m = _base_model(npartype=1)

    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_LUMPED_RATE_MODEL_WITH_PORES'
    col.par_type_volfrac = [1.0]

    # Particle: film diffusion only, no pore diffusion, no binding
    pt = col.particle_type_000
    pt.has_film_diffusion = 1
    pt.has_pore_diffusion = 0
    pt.has_surface_diffusion = 0
    pt.par_porosity = 0.581  # hydrogel porosity
    pt.par_radius = R_p
    pt.par_coreradius = 0.0
    pt.par_geom = 'SPHERE'
    pt.film_diffusion = [kf]
    pt.nbound = [0]
    pt.init_cp = [0.0]

    disc = col.discretization
    disc.USE_ANALYTIC_JACOBIAN = 1
    disc.SPATIAL_METHOD = 'DG'
    disc.POLYDEG = 5
    disc.NELEM = 256

    return m
