# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 14.3 from:

    T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
    2nd ed., Springer, 2015, Chapter 14 ("Multicomponent Radial Flow
    Chromatography"), p. 199: "Simulation of binary frontal adsorption in
    inward flow RFC".

This is a self-contained script: model definition, run, comparison plot and
validation metrics.  Only `cadet` (cadet-python), `numpy`, `matplotlib` and
`addict` are used (the latter is the standard Dict-builder convention used by
this repository's src/benchmark_models scripts).

===========================================================================
Step 0 -- case identification
===========================================================================
Governing model: Sections 14.1-14.2 of Gu (2015). Binary frontal adsorption
(breakthrough) in an INWARD-flow radial flow column (RFC): feed enters at the
outer radius X1 and flows toward the inner (hollow-core) radius X0. Multi-
component Langmuir isotherm, instantaneous local equilibrium at the pore
surface, film mass transfer + pore diffusion inside spherical particles
(general rate model, "GRM"), axial concentration gradients neglected. This
is explicitly the "same numerical strategy as the axial GRM of Chapter 3"
per the text, just written in a transformed radial coordinate
V = (X^2 - X0^2)/(X1^2 - X0^2) in [0, 1].

Target: Fig. 14.3 (PDF p. 207 / printed p. 199), with the exact Fortran
RATERFC.FOR data-file dump for that figure reproduced on the same page:

    in/outward = -1 (inward flow)        V0 = 0.04000     iave = 0
    nsp=2  nelemb=15  nc=2  index=1 (frontal/breakthrough)
    epsip (eps_p) = 0.400   epsib (eps_b) = 0.400

    component   PeL     eta     Bi(V=1)   C0      a       b
        1       100.00  10.000  10.000    0.20000 1.000   2.000
        2        80.00   8.000   8.000    0.20000 10.000  20.000

    (a_i/b_i = 0.5 for both components -> thermodynamically consistent
    Langmuir saturation capacity, as required.)

===========================================================================
Step 1 -- model mapping to CADET
===========================================================================
CADET binding/particle mapping (unambiguous): GENERAL_RATE_PARTICLE (film +
pore diffusion, spherical particles) with a rapid-equilibrium (is_kinetic=0)
MULTI_COMPONENT_LANGMUIR isotherm -- the direct CADET counterpart of Gu's
RFC-GRM with local equilibrium at the pore surface.

--- Bulk transport: CADET's NATIVE radial-flow geometry, with velocity-
    scaled dispersion via COL_DISPERSION_DEP=POWER_LAW ---

This script uses CADET's native radial-flow column unit (UNIT_TYPE=
'COLUMN_MODEL_1D', GEOMETRY='RADIAL_FLOW_CYLINDER_SHELL'), which discretizes
the bulk PDE in the ACTUAL physical radial coordinate X (velocity v(X) =
Q/(2*pi*H*eps_b*X)), i.e. genuinely solves Eq. (14.1), not merely its
V-transformed rewrite Eq. (14.7) -- getting this to actually reproduce the
paper required finding and fixing TWO real bugs in CADET-Core, plus using a
documented but previously-unused parameter-dependency mechanism:

(1) FORWARD_FLOW bug (fixed): `*ConvectionDispersionOperatorBaseFV::
    notifyDiscontinuousSectionTransition()` (ConvectionDispersionOperatorFV.cpp,
    all three geometries) only flipped the flow-direction sign on an actual
    section *transition*, never for an unchanging/first section -- silently
    running forward regardless of a configured FORWARD_FLOW=0. Fixed by
    applying the CURRENT section's direction directly instead of only
    reacting to a change relative to the previous section; a related
    ordering bug in ColumnModel1D.cpp (Jacobian pattern built before the
    direction was applied) was fixed the same way. (This exact fix is also
    independently present in this repo as commit a2ed7f69 "Fix backward
    flow conversion".) Verified via direct bulk-concentration-profile
    inspection: FORWARD_FLOW=[0] now genuinely feeds at the outer (large,
    X1) radius and propagates to the inner (small, X0) radius -- true
    inward flow, no priming section needed.

(2) Radial backward-flow dispersion sign bug (fixed): even with (1) fixed,
    a non-adsorbing tracer control (isolating this from the isotherm) showed
    genuine inward flow giving a grid-NON-convergent breakthrough time that
    diverged further from the theoretically-required tau=1 as NCOL
    increased (50->800: t50_tau 0.62->0.10), while outward flow was
    perfectly grid-convergent. Comparing `impl::residualBackwardsRadialFlow`
    line-by-line against `impl::residualForwardsRadialFlow`
    (RadialConvectionDispersionKernelFV.hpp) revealed the "right side"
    dispersion term's cell-center-distance denominator was consistent
    between the two functions, but the "left side" term's denominator was
    sign-flipped in the backward function relative to the forward one
    (`cellCenters[col-1]-cellCenters[col]` instead of
    `cellCenters[col]-cellCenters[col-1]`) -- a genuine, previously
    unexercised bug (this code path was unreachable before fix (1), since a
    single/unchanging backward section never actually ran backward). Fixed
    by correcting the denominator (and its matching Jacobian entry) to match
    the forward function's sign convention. Verified: after the fix, a
    non-adsorbing tracer gives IDENTICAL, grid-convergent t50~=0.99 in BOTH
    directions (NCOL 50-800, both with and without the dispersion
    dependency below), matching the theoretically-required tau=1 (exact for
    a linear, mass-conserving transport problem by construction of tau) to
    within ordinary numerical dispersion smoothing.

(3) Velocity-scaled dispersion via COL_DISPERSION_DEP='POWER_LAW': per Eq.
    (14.15), Gu's model requires Db_i(X) ~ v(X) (so that Pe_i comes out
    constant in the transformed V-space) -- CADET's native radial unit
    holds COL_DISPERSION constant in physical space by default, which is
    NOT the same physics. CADET-Core has a documented (but, before this
    script, unused for this geometry) parameter-dependency mechanism for
    exactly this: setting `COL_DISPERSION_DEP='POWER_LAW'` on the column
    multiplies the configured COL_DISPERSION[i] by
    `COL_DISPERSION_DEP_BASE * |v_local|^COL_DISPERSION_DEP_EXPONENT` at
    each cell FACE, where v_local is evaluated from the operator's actual
    local radial velocity at that face (confirmed in
    RadialConvectionDispersionKernelFV.hpp: `p.u / p.cellBounds[...]`,
    exactly CADET's own `currentVelocity(X) = _curVelCoeff/X`). Using
    EXPONENT=1 (BASE defaults to 1) and COL_DISPERSION[i] = Db_i|V=1/v(X1)
    gives Db_i(X) = Db_i|V=1 * v(X)/v(X1), i.e. exactly Db_i(X) ~ v(X) as
    required, with Db_i(X1)=Db_i|V=1 by construction. See docs at
    https://cadet.github.io/master/modelling/parameter_dependencies.html
    and https://cadet.github.io/master/interface/parameter_dependencies_config.html.
    NOTE: the analogous FILM_DIFFUSION_DEP mechanism originally existed in
    CADET-Core only for the legacy GeneralRateModel/LumpedRateModelWithPores
    classes (old GENERAL_RATE_MODEL/LUMPED_RATE_MODEL_WITH_PORES unit types),
    not for GeneralRateParticle.cpp/ParticleDiffusionOperatorFV.cpp (the
    particle framework used by COLUMN_MODEL_1D) -- this has since been
    implemented (see point (6) below), so Bi_i(V)'s position-dependence IS
    now represented via FILM_DIFFUSION_DEP, superseding the iave=2 fallback
    previously used here.

(4) DG bulk discretization cross-validation, and a THIRD bug found+fixed:
    the user asked whether these scripts also reproduce the paper using DG
    (not just FV) bulk discretization -- get_model()'s new
    bulk_discretization='DG' option (see below) exposes this. Naively
    switching to DG with COL_DISPERSION_DEP='POWER_LAW' gave a badly wrong,
    resolution-INDEPENDENT result (bit-identical across NELEM=8/16/32 --
    itself a red flag, since real numerical error should shrink with
    refinement). Root cause, found in
    `VariableCrossSectionConvectionDispersionOperatorBaseDG::
    computeOperatorsRadial()` (ConvectionDispersionOperatorDG.cpp): at each
    Gauss quadrature node, the code called
    `_dispersionDep->getValue(pos, comp, ..., baseDispersion)` -- passing
    the CONFIGURED (base) dispersion value itself as the argument the
    POWER_LAW dependency exponentiates, instead of the local velocity, and
    then used the bare getValue() result AS the final dispersion (never
    multiplying by baseDispersion). With EXPONENT=1 this silently evaluates
    to `dispAtQNodes = baseDispersion` (a no-op modifier applied to the
    wrong base), so the actual dispersion used was whatever raw
    COL_DISPERSION[i] value was configured for the POWER_LAW scaling
    (Db_i|V=1/v(X1), a small ratio) rather than the correctly-scaled
    physical dispersion (Db_i|V=1) -- roughly 1/v(X1) too large, badly
    over-dispersing the front. The identical bug existed in
    `computeOperatorsFrustum()`. FIXED by computing the true local velocity
    at each quadrature node (from the operator's `_QOverEps` and the local
    cross-sectional area at that position, mirroring the FV kernel's
    `p.u/p.cellBounds[...]`) and explicitly multiplying the getValue()
    result by baseDispersion, matching the FV convention exactly. Verified:
    DG with the fix now matches FV and the digitized reference closely
    (e.g. NELEM=8/16, POLYDEG=4: peak1 tau=2.897, height=1.366-1.367 vs.
    FV/reference's 2.93-2.94/1.369), with sensible grid-refinement behavior
    (NELEM=32 hit a separate solver performance/stiffness issue at this
    resolution, unrelated to correctness, not chased further).

(5) DG bulk+DG particle self-consistency check (user request): refining BOTH
    FV and DG bulk discretization was required to converge to the SAME answer
    (MSE < 1e-4 between the two, checked here with a dedicated grid-
    convergence test) -- not just each independently resembling the digitized
    literature curve. FV converges cleanly on its own (MSE between successive
    NCOL 120/240/480/960 shrinks ~100x per doubling, as expected for a 2nd-
    order scheme). DG, however, converged to FV distinctly SLOWER than
    expected until a FOURTH bug was found: `surfaceIntegralMainImpl()` and
    `DGjacobianDispBlock()` (ConvectionDispersionOperatorDG.cpp) computed the
    DG interface/surface numerical flux for the dispersion term using the
    raw, constant, position-INDEPENDENT `COL_DISPERSION[i]` value, even
    though fix (4) above had already made the VOLUME term correctly
    position-dependent -- i.e. the volume and surface terms of the same DG
    operator represented two different values of Db_i(X) at the same
    physical location, an inconsistent (if internally residual/Jacobian-
    matched, hence not itself a source of solver instability) discretization.
    FIXED by precomputing the true local (position-dependent) dispersion
    value exactly at each of the NELEM+1 element interfaces (new
    `_dispAtInterfaces` table, populated in `computeOperatorsRadial()`/
    `computeOperatorsFrustum()` alongside the existing volume-term
    computation) and using that table -- instead of the raw constant -- in
    both the residual's surface flux and its analytic Jacobian. This
    dramatically improved DG's convergence rate: NELEM=4 (POLYDEG=4) MSE
    vs. FV(NCOL=960) dropped from ~8e-6/1.3e-5 (components 1/2, pre-fix) to
    ~1.5e-7/5.2e-7 (post-fix), and NELEM=8 already reaches the apparent noise
    floor (~5e-8/1e-7) -- comfortably under the requested 1e-4 tolerance at
    even the coarsest resolutions tested. NOTE: a separate, NOT-yet-
    root-caused performance issue remains -- with COL_DISPERSION_DEP active
    (EXPONENT=1, genuine spatial variation) and DG bulk discretization, the
    IDAS time-stepper's required step count grows roughly with NELEM^2
    (isolated via direct A/B/C testing: absent entirely without the
    dependency; absent even WITH the dependency's code path active if
    EXPONENT=0, i.e. no real variation; independent of particle
    discretization choice; NOT explained by the surface-flux bug above, since
    fixing it left the step-count blowup unchanged). This makes very fine DG
    bulk grids (NELEM >~ 30-60) impractically slow, though it does not affect
    correctness at the practical resolutions used here (both the accuracy
    convergence and the MSE-vs-FV check above already pass well before this
    resolution is reached).

(6) FILM_DIFFUSION_DEP implemented for GeneralRateParticle/COLUMN_MODEL_1D
    (user request): per Eq. (14.16), Bi_i(V) (hence k_i(V)) scales with the
    local velocity as k_i(V) ~ v^(1/3) -- structurally identical to the
    dispersion dependency (3) above, just with EXPONENT=1/3 instead of 1, and
    living on the PARTICLE side of the model rather than the bulk operator.
    This did not exist for COLUMN_MODEL_1D's particle framework before this
    change (only for the legacy GeneralRateModel/LumpedRateModelWithPores
    classes) and required new communication between the bulk operator (which
    owns the position/velocity information) and the particle module (which
    owns the film-diffusion coefficient), bridged by the unit operation
    (ColumnModel1D): `columnPackingParameters` (the existing per-bulk-point
    channel from ColumnModel1D to GeneralRateParticle) gained a `velocity`
    field, populated once per bulk point via a new/existing
    `currentVelocity(pos)` accessor on the bulk operator (already present for
    FV; newly added for the DG variable-cross-section operator, mirroring
    FV's convention exactly). `ParticleDiffusionOperatorBase` gained a
    `_filmDiffusionDep` (mirroring `_dispersionDep`) and a
    `modifiedFilmDiffusion()` helper evaluating it pointwise at that
    velocity. This is a pointwise/"mass-lumped" evaluation for BOTH FV and
    DG -- for FV this is standard (matches the doc's own "midpoint rule"
    treatment of this exact term); for DG this is a deliberate simplification
    of the fully-consistent alternative (a per-DG-element weighted mass
    matrix, mirroring how dispersion's stiffness matrix is built via Gauss
    quadrature), chosen because film diffusion is a reaction/source-type term
    in the bulk PDE (not a derivative/stiffness term like dispersion): for a
    CONSTANT k_f, the area-weighted mass matrix cancels exactly against its
    own inverse, collapsing to the identical pointwise form used here, so the
    pointwise treatment introduces only a (small, standard, order-reducing)
    mass-lumping approximation once k_f varies spatially -- not a
    qualitatively different fidelity level than what FV already accepts for
    this term.

Fixes (1), (2), (4), and (5), and feature (6), were rebuilt and installed to
C:/Users/jmbr/software/CADET-Core/out/install/aRELEASE.

--- Position-dependent Bi_i(V) (Eq. 14.16, the paper's iave=0 treatment) ---
Represented via FILM_DIFFUSION_DEP='POWER_LAW' with EXPONENT=1/3 (see point
(6) above), evaluated at the true local velocity for every bulk point -- the
paper's own "iave=0" treatment, no longer needing the "iave=2"
constant-Bi-at-V=0.5 fallback used in earlier versions of this script.

===========================================================================
Step 2 -- reparameterization (paper's dimensionless -> CADET's dimensional)
===========================================================================
Definitions (Ch. 3 / Ch. 14, identical notation):
    Pe_Li = v(X1)*(X1-X0)/Db_i,V=1        Bi_i = k_i*Rp/(eps_p*Dp_i)
    eta_i = eps_p*Dp_i*(X1-X0)/(Rp*v(X1))   zeta_i = 3*Bi_i*eta_i*(1-eps_b)/eps_b
    tau   = v_char*t/(X1-X0),  v_char := 2*v(X1)*X1/(X1+X0)  (the transit-
        time-harmonic-mean velocity across the bed -- required so that a
        non-retained tracer's mean transit time equals tau=1 exactly,
        matching Gu's V-coordinate time-normalization; NOT simply v(X1))
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
        Db_i(X) = COL_DISPERSION[i]*v(X) = Db_i|V=1*v(X)/v(X1), matching
        Db_i(X1)=Db_i|V=1 exactly -- see Step 1(3))
    Dp_i = eta_i*Rp^2*v_char/(eps_p*L)   (Rp SQUARED -- dimensionally
        required for eta to be dimensionless: eta = eps_p*Dp*L/(Rp^2*v))
    k_i|V=1 = Bi_i|V=1 * eta_i * Rp * v_char/L
    k_i|avg = k_i|V=1 * [(1-V0)/(0.5+V0)]^(1/6)   (Eq. 14.16 at V=0.5, "iave=2")
    ka_i = b_i,  kd_i = 1,  qmax_i = a_i/b_i
    Q (inlet volumetric flow) = v(X1)*X1*2*pi*H*eps_b   (CADET's
        _curVelCoeff = Q/(2*pi*H*eps_b), currentVelocity(X)=_curVelCoeff/X,
        so v(X1) == Q/(2*pi*H*eps_b*X1) by construction)
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
# Step 3: paper's parameters, exactly as printed in the Fortran data dump
# on p. 199 (verified against the rendered figure page).
# ---------------------------------------------------------------------------
V0 = 0.04
TAU_MAX_SIM = 6.0  # simulate out to the figure's full x-axis range (paper's
                    # own run only went to tau_max=4.0, but the figure axis
                    # and plateau region extend to tau=6)

PAPER = {
    1: dict(PeL=100.0, eta=10.0, Bi_V1=10.0, C0=0.20, a=1.0, b=2.0),
    2: dict(PeL=80.0, eta=8.0, Bi_V1=8.0, C0=0.20, a=10.0, b=20.0),
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
CONC_UNIT = 1.0                       # reference concentration scale

for i, p in PAPER.items():
    p['Db_V1'] = V_REF * BED_LENGTH / p['PeL']
    p['Dp'] = p['eta'] * RP ** 2 * V_CHAR / (EPS_P * BED_LENGTH)
    p['k_V1'] = p['Bi_V1'] * p['eta'] * RP * V_CHAR / BED_LENGTH
    p['C0_phys'] = p['C0'] * CONC_UNIT
    p['ka'] = p['b'] / CONC_UNIT
    p['kd'] = 1.0
    p['qmax'] = (p['a'] / p['b']) * CONC_UNIT
    # COL_DISPERSION config value: with COL_DISPERSION_DEP='POWER_LAW' and
    # EXPONENT=1, CADET computes Db_i(X) = COL_DISPERSION[i] * v(X), so we
    # supply Db_i|V=1 / v(X1) here to get Db_i(X1) = Db_i|V=1 exactly.
    p['col_dispersion_value'] = p['Db_V1'] / V_REF
    # FILM_DIFFUSION config value: with FILM_DIFFUSION_DEP='POWER_LAW' and
    # EXPONENT=1/3 (per Eq. (14.16): k_i(V) ~ v^(1/3), the same relationship
    # expressed w.r.t. the actual local velocity that CADET's native
    # POWER_LAW dependency multiplies by), we supply k_i|V=1 / v(X1)^(1/3)
    # here to get k_i(X1) = k_i|V=1 exactly -- see Step 1 point (5) in the
    # module docstring for the full derivation and how this now supersedes
    # the "iave=2" constant-Bi fallback used previously.
    p['film_diffusion_value'] = p['k_V1'] / V_REF ** (1.0 / 3.0)

# "iave=2" constant-Bi fallback (paper's own alternative to true
# position-dependent Bi_i(V), Eq. 14.16 at V=0.5) -- used when
# film_diffusion_velocity_dep=False in get_model() (see there).
IAVE2_FACTOR = ((1.0 - V0) / (0.5 + V0)) ** (1.0 / 6.0)
for i, p in PAPER.items():
    p['k_avg'] = p['k_V1'] * IAVE2_FACTOR

Q_FLOW = V_REF * X1 * 2.0 * np.pi * CYL_HEIGHT * EPS_B  # inlet flow [m^3/s]

T_END = TAU_MAX_SIM * BED_LENGTH / V_CHAR  # physical end time [s]


def dimless_time(t_phys):
    """Map physical simulation time [s] to the paper's tau = v_char*t/(X1-X0)."""
    return np.asarray(t_phys) * V_CHAR / BED_LENGTH


# ---------------------------------------------------------------------------
# Step 5: CADET model definition
# ---------------------------------------------------------------------------
def get_model(ncol=120, par_ncells=4, n_points=400, spatial_method='FV',
              dg_polydeg=4, col_dispersion_velocity_dep=True,
              film_diffusion_velocity_dep=True):
    """bulk_discretization: 'FV' (default, validated) or 'DG' (cross-validation,
    see Step 1 point (4) in the module docstring -- also requires
    DISPERSION_SPATIAL_DEPENDENCE_POLYDEG for COL_DISPERSION_DEP with DG).
    dg_nelem defaults to max(ncol // (dg_polydeg + 1), 4) if not given, so the
    two discretizations can be compared at a roughly similar total DOF count.

    col_dispersion_velocity_dep: if True (default), Db_i(X) ~ v(X) via
        COL_DISPERSION_DEP='POWER_LAW' (Eq. 14.15, see Step 1 point (3)). If
        False, COL_DISPERSION is held constant at Db_i|V=1 everywhere (an
        ablation of the paper's own model, since the paper does not offer a
        non-dependent variant of the dispersion relationship).
    film_diffusion_velocity_dep: if True (default), k_i(X) ~ v(X)^(1/3) via
        FILM_DIFFUSION_DEP='POWER_LAW' (Eq. 14.16, see Step 1 point (6)). If
        False, falls back to the paper's own "iave=2" constant-Bi
        approximation (k_i evaluated once at V=0.5) used by earlier versions
        of this script."""
    m = Dict()
    m.input.model.nunits = 3

    # Single, unchanging-direction section, FORWARD_FLOW=[0] (genuine inward
    # flow -- feed at outer radius X1, exit at inner radius X0 -- now that
    # both CADET-Core bugs described in Step 1 are fixed; no priming section
    # needed).
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

    # --- Inlet: frontal / breakthrough feed, both components held at C0_i
    # from time zero (index=1 in the paper's Fortran code) ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 2
    feed = [PAPER[1]['C0_phys'], PAPER[2]['C0_phys']]
    m.input.model.unit_000.sec_000.const_coeff = feed
    m.input.model.unit_000.sec_000.lin_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_000.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_000.cube_coeff = [0.0, 0.0]

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
    if spatial_method == 'DG':
        col.discretization.SPATIAL_METHOD = 'DG'
        col.discretization.POLYDEG = dg_polydeg
        col.discretization.NELEM = ncol
        col.discretization.USE_COLLOCATION_DG = 0
        # Required whenever COL_DISPERSION_DEP is set with DG bulk
        # discretization (quadrature degree for the variable-dispersion
        # integral) -- see Step 1 point (4).
        col.dispersion_spatial_dependence_polydeg = dg_polydeg
    elif spatial_method == 'FV':
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

    if spatial_method == 'FV':
        col.particle_type_000.discretization.SPATIAL_METHOD = 'FV'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.NCELLS = par_ncells
        col.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2
    elif spatial_method == 'DG':
        col.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.PAR_NELEM = par_ncells
        col.particle_type_000.discretization.PAR_POLYDEG = 2

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
    m.input.solver.sections.nsec = 1
    m.input.solver.sections.section_continuity = []
    m.input.solver.sections.section_times = [0.0, T_END]
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-10
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, T_END, n_points)

    return m


def run_model(ncol=240, par_ncells=8, dg_polydeg=None, n_points=400, fname='fig14_3.h5', **kwargs):
# def run_model(ncol=60, par_ncells=1, dg_polydeg=3, n_points=400, fname='fig14_3.h5', **kwargs):
    model = get_model(ncol=ncol, par_ncells=par_ncells, dg_polydeg=dg_polydeg, n_points=n_points, **kwargs)
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
        path = os.path.join(HERE, 'fig14_3_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    return data['time_dimensionless'], data['c1_dimensionless'], data['c2_dimensionless']


# ---------------------------------------------------------------------------
# Step 6: validation metrics (adapted for a FRONTAL/breakthrough chromatogram
# -- classic pulse peak/first-moment analysis does not directly apply since
# both components approach a nonzero plateau (C/C0 -> 1) rather than
# returning to baseline; see per-metric notes below)
# ---------------------------------------------------------------------------
def t_at_level(t, c, level, rising_after=None):
    """First crossing time of c(t) through `level` (linear interpolation).
    If rising_after is given, only search t >= rising_after."""
    t = np.asarray(t)
    c = np.asarray(c)
    if rising_after is not None:
        mask = t >= rising_after
        t, c = t[mask], c[mask]
    idx = np.where(np.diff(np.sign(c - level)) > 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    t0, t1 = t[i], t[i + 1]
    c0, c1 = c[i], c[i + 1]
    frac = (level - c0) / (c1 - c0)
    return t0 + frac * (t1 - t0)


def compute_metrics(tau_sim, c1_sim, c2_sim, tau_ref, c1_ref, c2_ref):
    metrics = {}

    # interpolate CADET solution onto the reference (digitized) time grid
    # for MSE and plateau comparisons
    c1_sim_i = np.interp(tau_ref, tau_sim, c1_sim)
    c2_sim_i = np.interp(tau_ref, tau_sim, c2_sim)

    for name, c_sim, c_sim_i, c_ref in (
        ('component_1', c1_sim, c1_sim_i, c1_ref),
        ('component_2', c2_sim, c2_sim_i, c2_ref),
    ):
        m = {}

        # 1) Peak position (only meaningful for component 1, which shows
        #    competitive-Langmuir roll-up / overshoot above C/C0=1; component
        #    2 rises monotonically to its plateau and has no true interior
        #    peak, so this metric is reported as N/A there).
        if name == 'component_1':
            i_sim = np.argmax(c_sim)
            i_ref = np.argmax(c_ref)
            t_peak_sim = tau_sim[i_sim]
            t_peak_ref = tau_ref[i_ref]
            m['peak_time_sim'] = t_peak_sim
            m['peak_time_ref'] = t_peak_ref
            m['peak_time_relerr_%'] = 100 * abs(t_peak_sim - t_peak_ref) / t_peak_ref
            m['peak_height_sim'] = c_sim[i_sim]
            m['peak_height_ref'] = c_ref[i_ref]
            m['peak_height_relerr_%'] = 100 * abs(c_sim[i_sim] - c_ref[i_ref]) / c_ref[i_ref]
        else:
            m['peak_time_sim'] = np.nan
            m['peak_time_ref'] = np.nan
            m['peak_time_relerr_%'] = np.nan
            m['peak_height_sim'] = np.nan
            m['peak_height_ref'] = np.nan
            m['peak_height_relerr_%'] = np.nan

        # 2) "Elution time" analog: classic first-moment analysis
        #    (int t*c dt / int c dt) diverges for a step/frontal input since
        #    c(t) does not return to zero. We instead report the
        #    breakthrough time at 50% of the feed concentration (t50), the
        #    standard adapted metric for breakthrough curves.
        t50_sim = t_at_level(tau_sim, c_sim, 0.5)
        t50_ref = t_at_level(tau_ref, c_ref, 0.5)
        m['t50_sim'] = t50_sim
        m['t50_ref'] = t50_ref
        m['t50_relerr_%'] = 100 * abs(t50_sim - t50_ref) / t50_ref

        # 3) "Mass balance" analog: for a frontal input, overall mass
        #    balance is trivially satisfied once the column has reached
        #    saturation (all feed either passes through or is retained in
        #    the stationary phase, and outlet concentration must return to
        #    the feed value C/C0=1). We therefore check that both the
        #    simulated and the digitized curves converge to the same
        #    plateau value C/C0=1 at the end of the time window (deviation
        #    indicates either a units/scaling error or an under-resolved
        #    simulation).
        plateau_window = tau_ref >= (tau_ref.max() - 0.5)
        plateau_ref = np.nanmean(c_ref[plateau_window])
        plateau_sim = np.nanmean(c_sim_i[plateau_window])
        m['plateau_sim'] = plateau_sim
        m['plateau_ref'] = plateau_ref
        m['plateau_relerr_%'] = 100 * abs(plateau_sim - plateau_ref) / plateau_ref
        m['plateau_vs_feed_relerr_%'] = 100 * abs(plateau_sim - 1.0)

        # 4) Chromatogram MSE over the full digitized time window
        m['mse'] = np.nanmean((c_sim_i - c_ref) ** 2)

        metrics[name] = m

    return metrics


def print_metrics(metrics):
    for comp, m in metrics.items():
        print(f"\n--- {comp} ---")
        print(f"  Peak position   : sim={m['peak_time_sim']:.4g}  ref={m['peak_time_ref']:.4g}"
              f"  rel.err={m['peak_time_relerr_%']:.3g}%" if not np.isnan(m['peak_time_sim'])
              else "  Peak position   : N/A (monotonic breakthrough, no overshoot)")
        if not np.isnan(m['peak_height_sim']):
            print(f"  Peak height     : sim={m['peak_height_sim']:.4g}  ref={m['peak_height_ref']:.4g}"
                  f"  rel.err={m['peak_height_relerr_%']:.3g}%")
        print(f"  t50 (bt. time)  : sim={m['t50_sim']:.4g}  ref={m['t50_ref']:.4g}"
              f"  rel.err={m['t50_relerr_%']:.3g}%   [adapted 'elution time' metric]")
        print(f"  Plateau (C/C0)  : sim={m['plateau_sim']:.4g}  ref={m['plateau_ref']:.4g}"
              f"  rel.err={m['plateau_relerr_%']:.3g}%  (vs. feed=1: {m['plateau_vs_feed_relerr_%']:.3g}%)"
              "   [adapted 'mass balance' metric]")
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
          f"Q={Q_FLOW:.4g} m^3/s, T_END={T_END:.4g} s")

    print("\nRunning CADET simulation...")
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
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.plot(tau_sim, c1_sim, '-', color='tab:blue', label='Component 1 (CADET)')
    ax.plot(tau_sim, c2_sim, '-', color='tab:orange', label='Component 2 (CADET)')
    ax.plot(tau_ref, c1_ref, 'o', color='tab:blue', ms=3, mfc='none',
            label='Component 1 (digitized, Gu 2015)')
    ax.plot(tau_ref, c2_ref, 's', color='tab:orange', ms=3, mfc='none',
            label='Component 2 (digitized, Gu 2015)')
    ax.set_xlabel('Dimensionless time, ' + r'$\tau = v_{char}t/(X_1-X_0)$')
    ax.set_ylabel('Dimensionless concentration, ' + r'$C/C_0$')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.4)
    ax.set_title('Gu (2015), Fig. 14.3 -- binary frontal adsorption, inward-flow RFC\n'
                  '(CADET native radial geometry; velocity-scaled dispersion and film\n'
                  'diffusion via COL_DISPERSION_DEP/FILM_DIFFUSION_DEP; see script docstring)',
                  fontsize=9)
    ax.legend(loc='center right', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(HERE, 'fig14_3_comparison.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved comparison plot to {outpath}")

    # # --- DG bulk-discretization cross-validation (Step 1 point (4)) ---
    # print("\nCross-validating against DG bulk discretization (bulk_discretization='DG')...")
    # for dg_nelem in (8, 16):
    #     t_dg, outlet_dg = run_model(par_ncells=4, spatial_method='DG', dg_polydeg=4,
                                #   ncol=dg_nelem, fname=f'fig14_3_dg{dg_nelem}.h5')
    #     tau_dg = dimless_time(t_dg)
    #     c1_dg = outlet_dg[:, 0] / PAPER[1]['C0_phys']
    #     i1_dg = np.argmax(c1_dg)
    #     print(f"  DG NELEM={dg_nelem:2d} POLYDEG=4: component 1 peak tau={tau_dg[i1_dg]:.4g}"
    #           f"  height={c1_dg[i1_dg]:.4g}  (FV/ref: tau~2.93/2.938, height~1.367/1.369)")
