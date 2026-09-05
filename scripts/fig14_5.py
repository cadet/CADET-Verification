# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 14.5 from:

    T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
    2nd ed., Springer, 2015, Chapter 14 ("Multicomponent Radial Flow
    Chromatography"), p. 201: "Binary elution with an inert mobile phase in
    inward flow RFC".

Self-contained script: model definition, run, comparison plot and validation
metrics. Only `cadet` (cadet-python), `numpy`, `matplotlib` and `addict`
are used.

===========================================================================
Step 0 -- case identification
===========================================================================
Governing model: Sections 14.1-14.2 of Gu (2015). RFC general rate model
(GRM) with a transformed radial coordinate V = (X^2-X0^2)/(X1^2-X0^2) in
[0,1] (V=1 at the inward-flow inlet X1, V=0 at the outlet X0), run in
"elution with inert mobile phase" mode (index = 2): a finite rectangular
pulse of duration tau_imp is injected at tau=0, then the feed is switched to
pure (zero-concentration) carrier for the remainder of the run (Eq. 14.12):

    Cf_i(tau)/C0_i = 1   for 0 < tau < tau_imp
                   = 0   otherwise

Target: Fig. 14.5 (PDF p. 209 / printed p. 201), a screenshot of the
"Chromulator RateRFC Model Simulator" GUI (v2.2, Academic Edition), which
prints every input parameter directly:

    No. of Components = 2      No. of Finite Elements = 13
    No. of Interior Collocation Points = 2
    timp = 0.5   tint = 0.05   tmax = 16   eps_b = 0.4   eps_p = 0.4
    in/outward = -1 (inward)   V0 = 0.04   iave = 0
    LC Operation: index = 2  "Elution with inert mobile phase"

    component   PeL    eta    Bi(V=1)   C0    Consta(a)   Constb(b)
        1       100     10      10      0.2      1.0         2.0
        2       120     12      12      0.2     10.0        20.0

    (a_i/b_i = 0.5 for both components -> thermodynamically consistent
    Langmuir saturation capacity qmax, as required.)

Reference plot (right panel of the screenshot): x-axis "Dimensionless Time"
0-16, y-axis "Dimensionless Concentration" 0-0.60. Component 1 (red) is a
narrow early peak (apex ~0.58 near tau~2.5); component 2 (black) is a
broader, later, shorter peak (apex ~0.16 near tau~5.5-6) with a long tail.

===========================================================================
Step 1 -- model mapping to CADET
===========================================================================
CADET binding/particle mapping (unambiguous): GENERAL_RATE_PARTICLE (film +
pore diffusion, spherical particles) with a rapid-equilibrium (is_kinetic=0)
MULTI_COMPONENT_LANGMUIR isotherm -- the direct CADET counterpart of Gu's
RFC-GRM with local equilibrium at the pore surface.

--- Bulk transport: CADET's NATIVE radial-flow geometry, with velocity-
    scaled dispersion via COL_DISPERSION_DEP=POWER_LAW ---

This script uses CADET-Core's native radial-flow column unit (UNIT_TYPE=
'COLUMN_MODEL_1D', GEOMETRY='RADIAL_FLOW_CYLINDER_SHELL'), which discretizes
the bulk PDE in the ACTUAL physical radial coordinate X. Getting this to
reproduce the paper required finding and fixing TWO real bugs in CADET-Core,
plus using a documented parameter-dependency mechanism (full detail,
including root causes and verification, in the sibling Fig. 14.3 script's
docstring, scripts/fig14_3.py -- summarized here):

(1) FORWARD_FLOW bug (fixed): a single/unchanging-direction section silently
    ignored the configured FORWARD_FLOW and always ran forward, because
    `*ConvectionDispersionOperatorBaseFV::notifyDiscontinuousSectionTransition()`
    (ConvectionDispersionOperatorFV.cpp) only flipped the velocity sign on
    an actual section *transition*. Fixed by applying the current section's
    direction directly; a related ordering bug in ColumnModel1D.cpp was
    fixed the same way. (Also independently present in this repo as commit
    a2ed7f69 "Fix backward flow conversion".)

(2) Radial backward-flow dispersion sign bug (fixed): even with (1) fixed,
    genuine inward flow gave a grid-NON-convergent, systematically-too-early
    breakthrough (this is what an earlier draft of THIS script diagnosed as
    a "naive x-space-time-to-tau mapping" problem -- e.g. the pure-transport
    control breaking through at "tau"~=0.24 instead of ~1.0, and comp 2
    breaking through before comp 1 -- but the true root cause, found later
    while investigating fig14_3.py, was this bug, not a mapping issue at
    all). Root cause: in `impl::residualBackwardsRadialFlow`
    (RadialConvectionDispersionKernelFV.hpp), the "left side" dispersion
    term's cell-center-distance denominator had the opposite sign
    convention from `impl::residualForwardsRadialFlow`'s corresponding term.
    Fixed by correcting the denominator (and its matching Jacobian entry).
    Verified: a non-adsorbing tracer now gives IDENTICAL, grid-convergent
    breakthrough in both directions, matching the theoretically-required
    tau=1 for a linear, mass-conserving transport problem.

(3) Velocity-scaled dispersion via COL_DISPERSION_DEP='POWER_LAW': per Eq.
    (14.15), Db_i(X) must scale with the local velocity v(X) so that Pe_i
    comes out constant in Gu's transformed V-space -- CADET's native radial
    unit holds COL_DISPERSION constant in physical space by default.
    CADET-Core's parameter-dependency mechanism (documented at
    https://cadet.github.io/master/modelling/parameter_dependencies.html
    and .../interface/parameter_dependencies_config.html) resolves this:
    `COL_DISPERSION_DEP='POWER_LAW'` with `COL_DISPERSION_DEP_EXPONENT=1.0`
    multiplies COL_DISPERSION[i] by the operator's actual local radial
    velocity at each cell face, giving Db_i(X) ~ v(X) exactly. The analogous
    FILM_DIFFUSION_DEP mechanism originally existed only for the legacy
    GeneralRateModel/LumpedRateModelWithPores classes, not for
    GeneralRateParticle.cpp/ParticleDiffusionOperatorFV.cpp (the particle
    framework COLUMN_MODEL_1D actually uses) -- this has since been
    implemented (see fig14_3.py's docstring point (6)), so Bi_i(V)'s
    position dependence IS now represented via FILM_DIFFUSION_DEP,
    superseding the iave=2 fallback previously used here.

(4) DG bulk discretization cross-validation, and a THIRD bug found+fixed:
    get_model()'s bulk_discretization='DG' option exposes this. A genuine
    bug in `VariableCrossSectionConvectionDispersionOperatorBaseDG::
    computeOperatorsRadial()`/`computeOperatorsFrustum()`
    (ConvectionDispersionOperatorDG.cpp) passed the CONFIGURED base
    dispersion value itself as the argument to the POWER_LAW dependency
    (instead of the local velocity) and never multiplied the result by that
    base value -- silently mis-scaling the dispersion by roughly 1/v(X1).
    Fixed by computing the true local velocity at each Gauss quadrature node
    and multiplying the dependency's result by the base dispersion, matching
    the FV kernel's convention. Verified: DG (NELEM=8/16, POLYDEG=4) now
    matches FV and the digitized reference closely. Full detail in the
    sibling Fig. 14.3 script's docstring, scripts/fig14_3.py.

Fixes (1), (2), and (4) were rebuilt and installed to
C:/Users/jmbr/software/CADET-Core/out/install/aRELEASE. With all of the
above in place, this script uses genuine inward flow (FORWARD_FLOW=[0],
single unchanging direction) and CADET's true radial PDE -- no axial-column
substitution, no flow-direction trick.

--- Position-dependent Bi_i(V) (Eq. 14.16, the paper's iave=0 treatment) ---
Represented via FILM_DIFFUSION_DEP='POWER_LAW' with EXPONENT=1/3 (see point
(3) above and fig14_3.py's docstring point (6)), evaluated at the true
local velocity for every bulk point -- the paper's own "iave=0" treatment,
no longer needing the "iave=2" constant-Bi-at-V=0.5 fallback used in
earlier versions of this script.

===========================================================================
Step 2 -- reparameterization (paper's dimensionless -> CADET's dimensional)
===========================================================================
Definitions (Ch. 3 / Ch. 14, identical notation):
    Pe_Li = v(X1)*(X1-X0)/Db_i,V=1        Bi_i = k_i*Rp/(eps_p*Dp_i)
    eta_i = eps_p*Dp_i*(X1-X0)/(Rp^2*v_char)   (Rp SQUARED -- dimensionally
        required for eta to be dimensionless)
    tau   = v_char*t/(X1-X0),  v_char := 2*v(X1)*X1/(X1+X0)  (transit-time-
        harmonic-mean velocity, required so a non-retained tracer's mean
        transit time equals tau=1 exactly)
    V0 = X0^2/(X1^2-X0^2)
Langmuir (dimensionless, Eq. 3.21): cp*_i = a_i*cp_i/(1+sum_j b_j*C0_j*cp_j)
    => dimensional: Cp*_i = a_i*Cp_i/(1+sum_j b_j*Cp_j)   (C0_j cancels)
    => CADET MULTI_COMPONENT_LANGMUIR (kd_i=1 convention): ka_i=b_i,
       qmax_i=a_i/b_i.

We are free to choose convenient absolute (SI) scales as long as the
dimensionless groups above are reproduced exactly. Chosen scales: X1=0.05 m
(outer, inward-flow inlet), Rp=5e-5 m, v(X1)=1e-4 m/s, eps_b=eps_p=0.4
(given), reference concentration unit = 1 mol/m^3. Inverse mapping (used in
the code below):
    X0 = X1*sqrt(V0/(1+V0));  L := X1-X0
    v_char = 2*v(X1)*X1/(X1+X0)
    Db_i|V=1 = v(X1)*L/PeL_i          (COL_DISPERSION[i] := Db_i|V=1/v(X1),
        so that COL_DISPERSION_DEP='POWER_LAW' with EXPONENT=1 gives
        Db_i(X) = COL_DISPERSION[i]*v(X) = Db_i|V=1*v(X)/v(X1))
    Dp_i = eta_i*Rp^2*v_char/(eps_p*L)
    k_i|V=1 = Bi_i|V=1 * eta_i * Rp * v_char/L
    k_i|avg = k_i|V=1 * [(1-V0)/(0.5+V0)]^(1/6)     (Eq. 14.16 at V=0.5, "iave=2")
    ka_i = b_i,  kd_i = 1,  qmax_i = a_i/b_i
    Q (inlet volumetric flow) = v(X1)*X1*2*pi*H*eps_b   (CADET's
        _curVelCoeff = Q/(2*pi*H*eps_b), currentVelocity(X)=_curVelCoeff/X,
        so v(X1) == Q/(2*pi*H*eps_b*X1) by construction)

Pulse timing: injection pulse runs from tau=0 to tau=tau_imp=0.5 (physical
t=0 to t=tau_imp*L/v_char), and the run continues to tau=tau_max=16 (the
paper's own printed x-axis range).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from addict import Dict
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))
# Locally-built CADET-Core install with two real bugs fixed (FORWARD_FLOW
# direction, and the radial backward-flow dispersion sign bug -- see Step 1
# discussion above); required for this script's native-radial-geometry model
# to work at all. NOT the original C:\...\CADET_compiled\...\aRelease build.
INSTALL_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

# ---------------------------------------------------------------------------
# Step 3: paper's parameters, exactly as printed in the GUI screenshot on
# PDF p. 209 / printed p. 201 (verified against the rendered figure page).
# ---------------------------------------------------------------------------
V0 = 0.04
TAU_IMP = 0.5     # dimensionless injection (pulse) duration
TAU_MAX_SIM = 16.0  # paper's own printed x-axis range (tmax = 16)

PAPER = {
    1: dict(PeL=100.0, eta=10.0, Bi_V1=10.0, C0=0.20, a=1.0, b=2.0),
    2: dict(PeL=120.0, eta=12.0, Bi_V1=12.0, C0=0.20, a=10.0, b=20.0),
}

# thermodynamic consistency check (a_i/b_i must be identical for all comps)
_qmax_dimless = [PAPER[i]['a'] / PAPER[i]['b'] for i in PAPER]
assert np.allclose(_qmax_dimless, _qmax_dimless[0]), \
    "Langmuir saturation capacities a_i/b_i are not thermodynamically consistent!"

# ---------------------------------------------------------------------------
# Step 2: reparameterization -- native radial geometry, physical (SI) scales.
# See Step 1 discussion above for the full derivation.
# ---------------------------------------------------------------------------
X1 = 0.05                             # outer column radius [m] (inward-flow inlet)
X0 = X1 * np.sqrt(V0 / (1.0 + V0))    # inner radius [m]; V0 = X0^2/(X1^2-X0^2)
BED_LENGTH = X1 - X0                  # radial bed thickness [m]
CYL_HEIGHT = 1.0                      # arbitrary cylinder height [m]
EPS_B = 0.40
EPS_P = 0.40
RP = 5.0e-5                           # particle radius [m]
V_REF = 1.0e-4                        # interstitial velocity at X1 (V=1) [m/s]
V_CHAR = 2.0 * V_REF * X1 / (X1 + X0)  # transit-time characteristic velocity
CONC_UNIT = 1.0

for i, p in PAPER.items():
    p['Db_V1'] = V_REF * BED_LENGTH / p['PeL']
    p['Dp'] = p['eta'] * RP ** 2 * V_CHAR / (EPS_P * BED_LENGTH)
    p['k_V1'] = p['Bi_V1'] * p['eta'] * RP * V_CHAR / BED_LENGTH
    p['C0_phys'] = p['C0'] * CONC_UNIT
    p['ka'] = p['b'] / CONC_UNIT
    p['kd'] = 1.0
    p['qmax'] = (p['a'] / p['b']) * CONC_UNIT
    # COL_DISPERSION config value for COL_DISPERSION_DEP='POWER_LAW',
    # EXPONENT=1: Db_i(X) = COL_DISPERSION[i]*v(X), so supply Db_i|V=1/v(X1).
    p['col_dispersion_value'] = p['Db_V1'] / V_REF
    # FILM_DIFFUSION config value for FILM_DIFFUSION_DEP='POWER_LAW',
    # EXPONENT=1/3 (per Eq. (14.16): k_i(V) ~ v^(1/3), expressed w.r.t. the
    # actual local velocity CADET's native POWER_LAW dependency multiplies
    # by -- see fig14_3.py's docstring point (5)/(6) for the full
    # derivation), so supply k_i|V=1/v(X1)^(1/3).
    p['film_diffusion_value'] = p['k_V1'] / V_REF ** (1.0 / 3.0)

# "iave=2" constant-Bi fallback (paper's own alternative to true
# position-dependent Bi_i(V), Eq. 14.16 at V=0.5) -- used when
# film_diffusion_velocity_dep=False in get_model() (see there).
IAVE2_FACTOR = ((1.0 - V0) / (0.5 + V0)) ** (1.0 / 6.0)
for i, p in PAPER.items():
    p['k_avg'] = p['k_V1'] * IAVE2_FACTOR

Q_FLOW = V_REF * X1 * 2.0 * np.pi * CYL_HEIGHT * EPS_B  # inlet flow [m^3/s]

TAU_IMP_PHYS = TAU_IMP * BED_LENGTH / V_CHAR
T_END = TAU_MAX_SIM * BED_LENGTH / V_CHAR  # physical end time [s]


def dimless_time(t_phys):
    """Map physical simulation time [s] to the paper's tau = v_char*t/(X1-X0)."""
    return np.asarray(t_phys) * V_CHAR / BED_LENGTH


# ---------------------------------------------------------------------------
# Step 5: CADET model definition
# ---------------------------------------------------------------------------
def get_model(ncol=120, par_ncells=4, n_points=800, bulk_discretization='FV',
              dg_polydeg=4, dg_nelem=None, col_dispersion_velocity_dep=True,
              film_diffusion_velocity_dep=True):
    """bulk_discretization: 'FV' (default, validated) or 'DG' (cross-validation,
    see fig14_3.py's Step 1 point (4) for the CADET-Core DG dispersion-
    dependence bug this required finding and fixing). dg_nelem defaults to
    max(ncol // (dg_polydeg + 1), 4) if not given.

    col_dispersion_velocity_dep: if True (default), Db_i(X) ~ v(X) via
        COL_DISPERSION_DEP='POWER_LAW'. If False, COL_DISPERSION is held
        constant at Db_i|V=1 everywhere.
    film_diffusion_velocity_dep: if True (default), k_i(X) ~ v(X)^(1/3) via
        FILM_DIFFUSION_DEP='POWER_LAW'. If False, falls back to the paper's
        own "iave=2" constant-Bi approximation (k_i evaluated once at
        V=0.5)."""
    m = Dict()
    m.input.model.nunits = 3

    # Two sections: 0 = injection pulse (0 < tau < tau_imp), feed = C0_i;
    # 1 = elution/wash (tau > tau_imp), feed = 0 (pure inert mobile phase),
    # per Eq. (14.12), index = 2. No "priming" section is needed here: V=1
    # is always the inlet by construction (see Step 1 discussion), so a
    # plain forward-flow axial column already represents inward-flow RFC.
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, Q_FLOW,
        1.0, 2.0, -1.0, -1.0, Q_FLOW,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # --- Inlet: pulse of duration tau_imp at C0_i, then pure carrier (0) ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 2
    feed_pulse = [PAPER[1]['C0_phys'], PAPER[2]['C0_phys']]
    feed_zero = [0.0, 0.0]
    for sec, feed in (('sec_000', feed_pulse), ('sec_001', feed_zero)):
        m.input.model.unit_000[sec].const_coeff = feed
        m.input.model.unit_000[sec].lin_coeff = [0.0, 0.0]
        m.input.model.unit_000[sec].quad_coeff = [0.0, 0.0]
        m.input.model.unit_000[sec].cube_coeff = [0.0, 0.0]

    # --- Column: CADET's native radial-flow geometry (see Step 1 for the
    # two bug fixes and the COL_DISPERSION_DEP mechanism this relies on) ---
    col = Dict()
    col.unit_type = 'COLUMN_MODEL_1D'
    col.geometry = 'RADIAL_FLOW_CYLINDER_SHELL'
    col.ncomp = 2
    col.npartype = 1
    col.par_type_volfrac = 1
    col.cross_section_area_outer = 2.0 * np.pi * X1 * CYL_HEIGHT
    col.cylinder_height = CYL_HEIGHT
    col.bed_length = BED_LENGTH
    col.col_porosity = EPS_B
    if col_dispersion_velocity_dep:
        col.col_dispersion = [PAPER[1]['col_dispersion_value'], PAPER[2]['col_dispersion_value']]
        col.col_dispersion_dep = 'POWER_LAW'
        col.col_dispersion_dep_exponent = 1.0
    else:
        col.col_dispersion = [PAPER[1]['Db_V1'], PAPER[2]['Db_V1']]
    col.forward_flow = [0]
    col.init_c = [0.0, 0.0]

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    if bulk_discretization == 'DG':
        col.discretization.SPATIAL_METHOD = 'DG'
        col.discretization.POLYDEG = dg_polydeg
        col.discretization.NELEM = dg_nelem if dg_nelem is not None else max(ncol // (dg_polydeg + 1), 4)
        col.discretization.USE_COLLOCATION_DG = 0
        col.dispersion_spatial_dependence_polydeg = dg_polydeg
    else:
        col.discretization.SPATIAL_METHOD = 'FV'
        col.discretization.NCOL = ncol
        col.discretization.RECONSTRUCTION = 'WENO'
        col.discretization.weno.WENO_ORDER = 3
        col.discretization.weno.WENO_EPS = 1e-10
        col.discretization.weno.BOUNDARY_MODEL = 0
        col.discretization.GS_TYPE = 1
        col.discretization.MAX_KRYLOV = 0
        col.discretization.MAX_RESTARTS = 10
        col.discretization.SCHUR_SAFETY = 1e-8

    # --- Particles: GENERAL_RATE_PARTICLE (film + pore diffusion, spherical),
    # instantaneous local equilibrium multicomponent Langmuir ---
    col.particle_type_000.nbound = [1, 1]
    col.particle_type_000.init_cp = [0.0, 0.0]
    col.particle_type_000.init_cs = [0.0, 0.0]

    col.particle_type_000.has_film_diffusion = 1
    if film_diffusion_velocity_dep:
        col.particle_type_000.film_diffusion = [PAPER[1]['film_diffusion_value'], PAPER[2]['film_diffusion_value']]
        col.particle_type_000.film_diffusion_dep = 'POWER_LAW'
        col.particle_type_000.film_diffusion_dep_exponent = 1.0 / 3.0
    else:
        col.particle_type_000.film_diffusion = [PAPER[1]['k_avg'], PAPER[2]['k_avg']]
    col.particle_type_000.has_pore_diffusion = 1
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.par_geom = 'SPHERE'
    col.particle_type_000.par_coreradius = 0.0
    col.particle_type_000.par_porosity = EPS_P
    col.particle_type_000.par_radius = RP
    col.particle_type_000.pore_diffusion = [PAPER[1]['Dp'], PAPER[2]['Dp']]
    col.particle_type_000.surface_diffusion = [0.0, 0.0]

    col.particle_type_000.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    col.particle_type_000.adsorption.is_kinetic = 0  # rapid/local equilibrium
    col.particle_type_000.adsorption.mcl_ka = [PAPER[1]['ka'], PAPER[2]['ka']]
    col.particle_type_000.adsorption.mcl_kd = [PAPER[1]['kd'], PAPER[2]['kd']]
    col.particle_type_000.adsorption.mcl_qmax = [PAPER[1]['qmax'], PAPER[2]['qmax']]

    if bulk_discretization == 'DG':
        col.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.PAR_NELEM = par_ncells
        col.particle_type_000.discretization.PAR_POLYDEG = 2
    else:
        col.particle_type_000.discretization.SPATIAL_METHOD = 'FV'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.NCELLS = par_ncells
        col.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2

    m.input.model.unit_001 = col

    m.input.model.unit_002.ncomp = 2
    m.input.model.unit_002.unit_type = 'OUTLET'

    # --- return group ---
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_000.write_solution_outlet = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0
    m.input['return'].unit_002.write_solution_outlet = 0

    # --- time integration ---
    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 2
    m.input.solver.sections.section_continuity = [0]
    m.input.solver.sections.section_times = [0.0, TAU_IMP_PHYS, T_END]
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-10
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, T_END, n_points)

    return m


def run_model(ncol=120, par_ncells=4, n_points=800, fname='fig14_5.h5', **kwargs):
    model = get_model(ncol=ncol, par_ncells=par_ncells, n_points=n_points, **kwargs)
    c = Cadet(install_path=INSTALL_PATH)
    c.root.input = model.input
    c.filename = os.path.join(HERE, fname)
    c.save()
    rc = c.run_simulation()
    if rc.return_code != 0:
        raise RuntimeError(f"CADET failed: {getattr(rc, 'error_message', rc)}")
    c.load_from_file()
    t = np.asarray(c.root.output.solution.solution_times)
    outlet = np.asarray(c.root.output.solution.unit_001.solution_outlet)  # (ntime, ncomp)
    return t, outlet


# ---------------------------------------------------------------------------
# Step 4: reference (digitized) data
# ---------------------------------------------------------------------------
def load_digitized(path=None):
    if path is None:
        path = os.path.join(HERE, 'fig14_5_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    names = data.dtype.names

    def pick(*candidates):
        for c in candidates:
            if c in names:
                return data[c]
        return None
    tau = pick('tau', 'time_dimensionless', 'x', 'f0')
    c1 = pick('c1_dimensionless', 'component_1', 'c1', 'f1')
    c2 = pick('c2_dimensionless', 'component_2', 'c2', 'f2')
    if tau is None or c1 is None or c2 is None:
        raise RuntimeError(f"Could not identify columns in digitized CSV; found: {names}")
    return tau, c1, c2


# ---------------------------------------------------------------------------
# Step 6: validation metrics (classic pulse/elution chromatogram analysis --
# both components return to baseline, so peak position, first-moment elution
# time, mass balance and MSE are all directly applicable).
# ---------------------------------------------------------------------------
def compute_metrics(tau_sim, c1_sim, c2_sim, tau_ref, c1_ref, c2_ref):
    metrics = {}

    for name, c_sim_native, c_ref, C0 in (
        ('component_1', c1_sim, c1_ref, PAPER[1]['C0']),
        ('component_2', c2_sim, c2_ref, PAPER[2]['C0']),
    ):
        m = {}

        # interpolate CADET solution onto the reference (digitized) tau grid
        c_sim_i = np.interp(tau_ref, tau_sim, c_sim_native)

        # 1) Peak position (time of maximum concentration)
        i_sim = np.argmax(c_sim_native)
        i_ref = np.nanargmax(c_ref)
        t_peak_sim = tau_sim[i_sim]
        t_peak_ref = tau_ref[i_ref]
        m['peak_time_sim'] = t_peak_sim
        m['peak_time_ref'] = t_peak_ref
        m['peak_time_relerr_%'] = 100 * abs(t_peak_sim - t_peak_ref) / t_peak_ref
        m['peak_height_sim'] = c_sim_native[i_sim]
        m['peak_height_ref'] = c_ref[i_ref]
        m['peak_height_relerr_%'] = 100 * abs(c_sim_native[i_sim] - c_ref[i_ref]) / c_ref[i_ref]

        # 2) Elution time (first moment): int(t*c dt) / int(c dt), over the
        #    full simulated time window (CADET's own dense time grid, not
        #    the sparser digitized grid, for accuracy).
        def first_moment(t, c):
            c = np.clip(c, 0.0, None)
            return np.trapz(t * c, t) / np.trapz(c, t)

        tm_sim = first_moment(tau_sim, c_sim_native)
        tm_ref = first_moment(tau_ref, np.nan_to_num(c_ref))
        m['moment_time_sim'] = tm_sim
        m['moment_time_ref'] = tm_ref
        m['moment_time_relerr_%'] = 100 * abs(tm_sim - tm_ref) / tm_ref

        # 3) Mass balance: injected mass vs. eluted mass, both expressed in
        #    C/C0-normalized dimensionless units (c1_sim, c2_sim, c1_ref,
        #    c2_ref are all already C0-normalized). The injected pulse has
        #    normalized concentration 1 for duration tau_imp, so its
        #    normalized "mass" (area) is exactly tau_imp -- C0 must NOT be
        #    multiplied in again here. Component 2's long tail is not fully
        #    captured within tau_max=16 (matching the paper's own truncated
        #    plot window), so a residual undershoot here is expected and
        #    consistent with the reference curve, not necessarily a bug.
        injected = TAU_IMP
        eluted_sim = np.trapz(np.clip(c_sim_native, 0.0, None), tau_sim)
        eluted_ref = np.trapz(np.nan_to_num(np.clip(c_ref, 0.0, None)), tau_ref)
        m['mass_injected_dimless'] = injected
        m['mass_eluted_sim_dimless'] = eluted_sim
        m['mass_eluted_ref_dimless'] = eluted_ref
        m['mass_balance_relerr_%'] = 100 * abs(eluted_sim - injected) / injected
        m['mass_sim_vs_ref_relerr_%'] = 100 * abs(eluted_sim - eluted_ref) / eluted_ref

        # 4) Chromatogram MSE over the full digitized time window
        m['mse'] = np.nanmean((c_sim_i - c_ref) ** 2)

        metrics[name] = m

    return metrics


def print_metrics(metrics):
    for comp, m in metrics.items():
        print(f"\n--- {comp} ---")
        print(f"  Peak position   : sim={m['peak_time_sim']:.4g}  ref={m['peak_time_ref']:.4g}"
              f"  rel.err={m['peak_time_relerr_%']:.3g}%")
        print(f"  Peak height     : sim={m['peak_height_sim']:.4g}  ref={m['peak_height_ref']:.4g}"
              f"  rel.err={m['peak_height_relerr_%']:.3g}%")
        print(f"  Elution time    : sim={m['moment_time_sim']:.4g}  ref={m['moment_time_ref']:.4g}"
              f"  rel.err={m['moment_time_relerr_%']:.3g}%   [first moment int(t c dt)/int(c dt)]")
        print(f"  Mass balance    : injected={m['mass_injected_dimless']:.4g}  "
              f"eluted(sim)={m['mass_eluted_sim_dimless']:.4g}  "
              f"rel.err(sim vs inj)={m['mass_balance_relerr_%']:.3g}%   "
              f"rel.err(sim vs ref)={m['mass_sim_vs_ref_relerr_%']:.3g}%")
        print(f"  Chromatogram MSE: {m['mse']:.4g}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Physical (SI) parameters derived from the paper's dimensionless groups:")
    for i, p in PAPER.items():
        print(f"  Component {i}: Db(V=1)={p['Db_V1']:.4g} m^2/s (COL_DISPERSION={p['col_dispersion_value']:.4g}), "
              f"Dp={p['Dp']:.4g} m^2/s, "
              f"k(V=1)={p['k_V1']:.4g} m/s (FILM_DIFFUSION={p['film_diffusion_value']:.4g}), "
              f"ka={p['ka']:.4g}, kd={p['kd']:.4g}, qmax={p['qmax']:.4g}")
    print(f"  X0={X0:.4g} m, X1={X1:.4g} m, bed_length={BED_LENGTH:.4g} m, "
          f"V_REF={V_REF:.4g} m/s, V_CHAR={V_CHAR:.4g} m/s, "
          f"Q={Q_FLOW:.4g} m^3/s, tau_imp={TAU_IMP}, T_END={T_END:.4g} s")

    print("\nRunning CADET simulation (native radial geometry, genuine inward flow)...")
    t_phys, outlet = run_model()
    tau_sim = dimless_time(t_phys)
    c1_sim = outlet[:, 0] / PAPER[1]['C0_phys']
    c2_sim = outlet[:, 1] / PAPER[2]['C0_phys']

    print("Loading digitized reference data...")
    tau_ref, c1_ref, c2_ref = load_digitized()

    print("Computing validation metrics...")
    metrics = compute_metrics(tau_sim, c1_sim, c2_sim, tau_ref, c1_ref, c2_ref)
    print_metrics(metrics)

    # --- comparison plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tau_sim, c1_sim, '-', color='tab:red', label='Component 1 (CADET)')
    ax.plot(tau_sim, c2_sim, '-', color='black', label='Component 2 (CADET)')
    ax.plot(tau_ref, c1_ref, 'o', color='tab:red', ms=3, mfc='none',
            label='Component 1 (digitized, Gu 2015)')
    ax.plot(tau_ref, c2_ref, 's', color='black', ms=3, mfc='none',
            label='Component 2 (digitized, Gu 2015)')
    ax.set_xlabel('Dimensionless time, ' + r'$\tau = vt/(X_1-X_0)$')
    ax.set_ylabel('Dimensionless concentration, ' + r'$C/C_0$')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 0.65)
    ax.set_title('Gu (2015), Fig. 14.5 -- binary elution with inert mobile phase, inward-flow RFC\n'
                  '(CADET native radial geometry; velocity-scaled dispersion and film\n'
                  'diffusion via COL_DISPERSION_DEP/FILM_DIFFUSION_DEP)', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(HERE, 'fig14_5_comparison.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved comparison plot to {outpath}")

    # --- DG bulk-discretization cross-validation (see fig14_3.py Step 1 point (4)) ---
    print("\nCross-validating against DG bulk discretization (bulk_discretization='DG')...")
    for dg_nelem in (8, 16):
        t_dg, outlet_dg = run_model(par_ncells=4, bulk_discretization='DG', dg_polydeg=4,
                                     dg_nelem=dg_nelem, fname=f'fig14_5_dg{dg_nelem}.h5')
        tau_dg = dimless_time(t_dg)
        c1_dg = outlet_dg[:, 0] / PAPER[1]['C0_phys']
        c2_dg = outlet_dg[:, 1] / PAPER[2]['C0_phys']
        i1_dg, i2_dg = np.argmax(c1_dg), np.argmax(c2_dg)
        print(f"  DG NELEM={dg_nelem:2d} POLYDEG=4: peak1 tau={tau_dg[i1_dg]:.4g} h1={c1_dg[i1_dg]:.4g}"
              f"  peak2 tau={tau_dg[i2_dg]:.4g} h2={c2_dg[i2_dg]:.4g}"
              f"  (FV/ref: 2.34/0.578, 5.41/0.151 -- 2.375/0.568, 5.375/0.153)")
