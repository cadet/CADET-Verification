# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 14.6 from:

    T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
    2nd ed., Springer, 2015, Chapter 14 ("Multicomponent Radial Flow
    Chromatography"), p. 210: "Simulation of affinity RFC with inward flow".

This is a self-contained script: model definition, run, comparison plot and
validation metrics. Only `cadet` (cadet-python), `numpy`, `matplotlib` and
`addict` are used, matching this repository's src/benchmark_models
convention.

===========================================================================
Step 0 -- case identification
===========================================================================
Target: Fig. 14.6 (PDF p. 210), a 3-pseudo-component affinity RFC
simulation with inward flow: frontal loading of a protein (component 1),
a wash stage, and an elution stage using a soluble ligand (component 2).
Component 3 is the protein-ligand complex (PL) formed by a *liquid-phase*
reversible reaction between components 1 and 2; it forms/leaves in the
mobile+pore liquid and never binds the stationary phase.

Model equations ARE given in the book (Model-based path): Chapter 10,
Secs. 10.1-10.5 (the "AFFINITY" model, axial flow) together with Chapter 14,
Sec. 14.6 ("Extensions of the General Multicomponent Rate Model for RFC"),
which states literally: "The addition of reaction terms for the interaction
between a macromolecule and the soluble ligand involves the bulk-fluid
phase, but it does not touch the characteristic terms of an RFC model" --
i.e. take the Chapter 10 AFFINITY model's reaction terms verbatim and drop
them into the Chapter 14 RFC transport operator (Eq. 14.7/14.8), using
CADET's NATIVE radial-flow geometry (see point (b) below for the two
CADET-Core bugs that had to be fixed, and the velocity-scaled-dispersion
mechanism, to make this work).

---------------------------------------------------------------------------
(a) Resolution of the "Fig. 11.14" erratum
---------------------------------------------------------------------------
p. 210 states (OCR-verified against the rendered page): "The parameters for
simulation of Fig. 14.6 are the same as those for Fig. 11.14, except inward
flow." There is NO Fig. 11.14 anywhere in this book -- Chapter 11 is about
ion-exchange chromatography and contains only an unrelated Eq. (11.14) on
p. 162. The very same paragraph on p. 210 continues: "Compared with AFC for
Fig. 10.14, Fig. 14.6 for RFC has almost no practical difference when the
two figures are superimposed because comparable physical parameters ... are
used in the two cases." Fig. 10.14 (p. 138, a "Chromulator AFFINITY
simulator" GUI screenshot) was visually compared here against the rendered
Fig. 14.6 (p. 210): both show (i) an identical frontal loading shape for
component 1 rising to C/C0=1 between tau~6-14 and dropping sharply at
tau~16-17, (ii) an identical smooth S-shaped rise of component 2 to a
plateau near 1 starting at the same tau, and (iii) an identical sharp
component-3 peak (height ~0.93) exactly at the tau~17 crossover, decaying
slowly toward 0 by tau~55-60. CONCLUSION (consistent with the book's own
text, which never mentions a Fig. 11.14 anywhere else and explicitly claims
RFC/AFC near-equivalence for this exact pair of figures): "Fig. 11.14" is a
book erratum for "Fig. 10.14". This script takes ALL Chapter-10-model
dimensionless parameters directly from the Fig. 10.14 GUI screenshot
(p. 138), applied to the RFC geometry of Chapter 14 with V0=0.04 (p. 210).

---------------------------------------------------------------------------
(b) Bulk transport: CADET's NATIVE radial-flow geometry -- two real
    CADET-Core bugs found and fixed, plus a velocity-scaled-dispersion
    mechanism, were needed to make this work.
---------------------------------------------------------------------------
This script uses CADET-Core's native radial-flow column unit (COLUMN_MODEL_1D
+ GEOMETRY='RADIAL_FLOW_CYLINDER_SHELL'), which discretizes the bulk PDE in
the ACTUAL physical radial coordinate X. Getting this to reproduce the paper
required (full detail, including root causes and verification, in the
sibling Fig. 14.3 script's docstring, scripts/fig14_3.py -- summarized here):

(1) FORWARD_FLOW bug (fixed): a single/unchanging-direction section silently
    ignored the configured FORWARD_FLOW and always ran forward, because
    `*ConvectionDispersionOperatorBaseFV::notifyDiscontinuousSectionTransition()`
    (ConvectionDispersionOperatorFV.cpp, all three geometries) only flipped
    the velocity sign on an actual section *transition*. Fixed by applying
    the current section's direction directly; a related ordering bug in
    ColumnModel1D.cpp was fixed the same way. (Also independently present in
    this repo as commit a2ed7f69 "Fix backward flow conversion".)

(2) Radial backward-flow dispersion sign bug (fixed): even with (1) fixed,
    genuine inward flow gave a grid-NON-convergent, systematically-too-early
    breakthrough (verified on a non-adsorbing tracer control, isolating this
    from the isotherm/reaction), while outward flow was fine. Root cause: in
    `impl::residualBackwardsRadialFlow` (RadialConvectionDispersionKernelFV.hpp),
    the "left side" dispersion term's cell-center-distance denominator had
    the opposite sign convention from the corresponding term in
    `impl::residualForwardsRadialFlow` -- a genuine, previously unexercised
    bug (this code path was unreachable before fix (1)). Fixed by correcting
    the denominator (and its matching Jacobian entry) to match the forward
    function's convention. Verified: a non-adsorbing tracer now gives
    IDENTICAL, grid-convergent breakthrough in both directions, matching the
    theoretically-required tau=1 for a linear, mass-conserving transport
    problem.

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
    FILM_DIFFUSION_DEP mechanism exists but is only wired into the legacy
    GeneralRateModel/LumpedRateModelWithPores classes, NOT into
    GeneralRateParticle.cpp/ParticleDiffusionOperatorFV.cpp (the particle
    framework COLUMN_MODEL_1D actually uses) -- so Bi_i(V)'s position
    dependence still cannot be represented exactly; the iave=2 fallback
    below remains necessary for that piece.

Fixes (1) and (2) were rebuilt and installed to
C:/Users/jmbr/software/CADET-Core/out/install/aRELEASE. With all of the
above in place, this script uses genuine inward flow (FORWARD_FLOW=[0,0,0],
feed at the outer radius X1, single unchanging direction across all 3
sections) and CADET's true radial PDE -- no axial-column substitution, no
flow-direction trick.

---------------------------------------------------------------------------
(c) Chapter 10 model equations (verified against the actual page images,
    not OCR alone -- OCR of sub/superscripts is unreliable in this PDF)
---------------------------------------------------------------------------
Reactions (Eqs. 10.5-10.6):
    P + I <=(ka2,kd2)=> PI            (component 1 + component 2 -> 3)
    P + L <=(ka1,kd1)=> PL            (component 1 binds immobilized ligand)
Only component 1 (protein) has a bound state; components 2 (soluble ligand)
and 3 (complex) never bind the stationary phase (they are transported by
bulk convection/dispersion and pore diffusion only).

Bulk-fluid governing equation (Eq. 10.10, dimensionless, axial form; Sec.
14.6 replaces the axial d/dz, d2/dz2 terms with the RFC operator of
Eq. 14.7 while keeping the reaction term unchanged):
    -(1/PeLi) d2cbi/dz2 + dcbi/dz + dcbi/dtau + xi_i*(cbi - cpi|r=1)
        - f(i)*[Da2a*cb1*(C02/C0i)*cb2 - Da2d*(C03/C0i)*cb3] = 0
    f(1)=f(2)=+1 (protein & ligand consumed by reaction),
    f(3)=-1 (complex produced by reaction).

Particle-phase governing equation (Eq. 10.11, dimensionless):
    g(i)*(1-eps_p)*dc*pi/dtau + eps_ap*dcpi/dtau
        - f(i)*eps_ap*[Da2a*cp1*(C02/C0i)*cp2 - Da2d*(C03/C0i)*cp3]
        - eta_i*(1/r^2) d/dr(r^2 dcpi/dr) = 0
    g(1)=1, g(2)=g(3)=0 (only component 1 has a bound-phase term).
Note both the pore-diffusion accumulation term (eps_ap*dcpi/dtau) and the
pore-liquid reaction term carry the SAME eps_ap prefactor -- so when this
equation is divided through to isolate dcpi/dtau (the form CADET's own
particle-phase residual actually solves), eps_ap cancels identically and
the *same* physical rate constants ka2, kd2 govern the reaction in both the
bulk and the pore-liquid phase. This is exactly mirrored by using the SAME
MASS_ACTION_LAW reaction (same kFwd/kBwd, same stoichiometry) on both the
column unit's bulk ("liquid") phase and the particle type's pore ("liquid")
phase in the CADET model below.

Bound-phase kinetics for component 1 only (Eq. 10.12/10.9):
    dc*p1/dtau = Da1a*cp1*(c1_inf - c*p1) - Da1d*c*p1
This is a single-component KINETIC (non-equilibrium) Langmuir isotherm with
finite capacity c1_inf (the dimensionless immobilized-ligand site density,
c1_inf = C_inf,1/C0_1) -- i.e. exactly CADET's MULTI_COMPONENT_LANGMUIR
binding model with IS_KINETIC=1, restricted to component 1 (nbound=[1,0,0]).
Verified against src/libcadet/model/binding/LangmuirBinding.cpp: the
fluxImpl() residual is `kD[i]*y - kA[i]*yCp*qMax[i]*qSum`, i.e.
dq/dt = ka*cp*(qmax - q) - kd*q exactly, with qSum skipping components that
have zero bound states -- confirming components 2 and 3 (nbound=0) simply
do not participate in the isotherm at all.

===========================================================================
Step 1 -- CADET model mapping
===========================================================================
- Bulk transport: COLUMN_MODEL_1D with GEOMETRY='RADIAL_FLOW_CYLINDER_SHELL'
  (see point (b) above for the two bug fixes and COL_DISPERSION_DEP
  mechanism this relies on), FV bulk discretization.
- Particle transport: GENERAL_RATE_PARTICLE (film + pore diffusion,
  spherical particles), identical film/pore-diffusion coefficients for all
  3 components since the paper's own PeL/eta/Bi/ExF table (Fig. 10.14 GUI,
  reproduced below) happens to be numerically IDENTICAL across all three
  components (PeL=300, eta=10, Bi=40, ExF=0.8 for i=1,2,3) -- a
  simplification of this particular case study, not a general CADET
  limitation.
- Binding: MULTI_COMPONENT_LANGMUIR, IS_KINETIC=1, active for component 1
  only (nbound=[1,0,0]); components 2 and 3 access the particle pore liquid
  (has_film_diffusion / has_pore_diffusion enabled for all 3) but never
  bind.
- Reaction: MASS_ACTION_LAW, one reaction P + I -> PI, applied identically
  (a) on the column unit's bulk "liquid" phase (unit_001.nreac_liquid=1,
  unit_001.liquid_reaction_000) and (b) on the particle type's pore
  "liquid" phase (unit_001.particle_type_000.nreac_liquid=1, ...
  .liquid_reaction_000), confirmed against
  src/libcadet/model/reaction/MassActionLawReaction.cpp (fields
  MAL_STOICHIOMETRY, MAL_KFWD, MAL_KBWD; MAL_EXPONENTS_FWD/BWD default to
  the negative/positive parts of the stoichiometry, i.e. rate =
  kFwd*c1*c2 - kBwd*c3, exactly the desired mass-action form) and against
  src/libcadet/model/particle/GeneralRateParticle.cpp (same NREAC_LIQUID /
  liquid_reaction_000 config keys, applied per particle type). Field-naming
  convention (nreac_liquid, liquid_reaction_000.type/mal_*) verified
  against this repo's own src/benchmark_models/settings_crystallization.py,
  which already uses an (unrelated) liquid-phase reaction model with
  identical scope/field names.

---------------------------------------------------------------------------
Resolution of the Daa/Dad row-to-Da1/Da2 mapping ambiguity
---------------------------------------------------------------------------
The Fig. 10.14 GUI table lists a "Daa/Dad" column PER COMPONENT ROW, but
the underlying model only has TWO independent Damkohler pairs: Da1a/Da1d
(component 1's own binding to the immobilized ligand, Eq. 10.12 -- a
property of component 1 alone) and Da2a/Da2d (the single shared bulk+pore
reaction rate constants for P+I<->PI, Eq. 10.10/10.11 -- these appear
identically in every component's governing equation, they are not
"per-component"). Reading the table:
    Component 1 (protein):        Daa=2,   Dad=0.2,  C_inf=0.00001
    Component 2 (soluble ligand):  Daa=2,   Dad=0.2,  C_inf=0.0
    Component 3 (complex):        Daa=0.0, Dad=0.0,  C_inf=0.0
The natural reading is: row 1's Daa/Dad = Da1a/Da1d (component 1 is the
only one with a nonzero C_inf, consistent with only component 1 having an
Eq. 10.12-type term), and row 2's Daa/Dad = Da2a/Da2d (tabulated under the
soluble ligand's own row since it is the reaction's second reactant); row
3 is (0,0) because the complex has no independent kinetic process of its
own beyond the shared reaction already captured via row 2.
IMPORTANT: this resolution turns out not to matter numerically here, since
row 1 and row 2 report the SAME Daa/Dad values (2, 0.2) -- so regardless of
which row is attributed to Da1 vs Da2, the simulation uses Da1a=Da2a=2,
Da1d=Da2d=0.2. This is stated explicitly rather than left implicit.

---------------------------------------------------------------------------
Size-exclusion factor ExF (=0.8 for all 3 components)
---------------------------------------------------------------------------
Per Sec. 10.3 / Eq. 8.8 (referenced on p. 125), size exclusion is modeled
in the book by replacing the particle porosity eps_p with an "accessible"
particle porosity eps_ap = ExF*eps_p in the PORE-liquid accumulation and
diffusion/reaction terms only, while the SOLID-phase accumulation term
(the (1-eps_p) prefactor multiplying dq*/dt in Eq. 10.11) keeps using the
TRUE total porosity eps_p. Standard CADET has a single scalar PAR_POROSITY
parameter that multiplies BOTH the pore-liquid accumulation term (as
eps_p) AND (via 1-PAR_POROSITY) the solid-phase accumulation term -- i.e.
CADET has no separate "accessible vs. total porosity" split. Since ExF is
IDENTICAL (0.8) for all three components here (there is no differential
size-exclusion effect between species in this particular case study), this
is treated as a reasonable, fully-determined approximation: CADET's
PAR_POROSITY is set to the accessible value eps_ap = ExF*eps_p = 0.8*0.45 =
0.36 (this is what actually enters the given PeL/eta/Bi dimensionless
numbers, so getting the pore-transport physics right was prioritized). The
resulting mismatch is that CADET's solid-fraction weight becomes
(1-0.36)=0.64 instead of the book's (1-0.45)=0.55, a secondary effect
expected to be small relative to the dominant stoichiometric-capacity and
reaction/mass-transfer kinetics (c1_inf=10, i.e. a strong/high-capacity
binder).

===========================================================================
Step 2 -- reparameterization (paper's dimensionless groups -> CADET's
          dimensional/physical parameters)
===========================================================================
Definitions (Ch. 3/10 for Pe/Bi/eta/Da, Ch. 14 Eq. 14.1/14.7/14.16 for the
RFC-specific V0/time-normalization/Bi(V) treatment; re-derived and
dimensionally checked in coordination with fig14_3.py, which found and
fixed the same formula errors independently -- see that script's own
Step-2 docstring for the full re-derivation):
Applied to the native radial geometry (physical radial coordinate X):
    Pe_Li  = v(X1)*(X1-X0)/Db_i,V=1        Bi_i = k_i*Rp/(eps_ap*Dp_i)
    eta_i  = eps_ap*Dp_i*(X1-X0)/(Rp^2*v_char)   (Rp SQUARED -- dimensionally
        required for eta to be dimensionless)
    tau    = v_char*t/(X1-X0),  v_char := 2*v(X1)*X1/(X1+X0)  (transit-time-
        harmonic-mean velocity, required so a non-retained tracer's mean
        transit time equals tau=1 exactly)
    V0 = X0^2/(X1^2-X0^2)
    Da1a = L*ka1*C0_1/v_char,  Da1d = L*kd1/v_char   (component 1's binding,
        L := X1-X0)
    Da2a = L*ka2*C0_1/v_char,  Da2d = L*kd2/v_char   (shared P+I reaction;
        this book chapter's own explicit convention is to normalize with
        C0_1, not C0_2 -- see p. 135: "For convenience, Da2a's definition
        uses C0_1 instead of C0_2 ... Note that Gu et al. [17] used C0_2
        instead of C0_1"; this script follows the BOOK CHAPTER's own
        convention, i.e. C0_1, exactly as instructed.)
    c1_inf = C_inf,1/C0_1  (dimensionless immobilized-ligand site density)
    Bi_i,V = [(1-V0)/(V+V0)]^(1/6) * Bi_i,V=1              (Eq. 14.16, same
        sign convention as the validated sibling scripts; only used for the
        V=0.5 "iave=2" average below, an O(V0)=O(4%) effect either way)

We are free to choose convenient absolute (SI-like) scales as long as the
above dimensionless groups are reproduced exactly. Chosen scales: X1=0.05 m
(outer, inward-flow inlet), Rp=5e-5 m, v(X1)=1e-4 m/s, reference
concentration unit = 1 [arbitrary consistent unit]. Inverse mapping (used in
the code below):
    X0 = X1*sqrt(V0/(1+V0));  L := X1-X0
    v_char = 2*v(X1)*X1/(X1+X0)
    Db_i|V=1 = v(X1)*L/PeL_i    (COL_DISPERSION[i] := Db_i|V=1/v(X1), so that
        COL_DISPERSION_DEP='POWER_LAW' with EXPONENT=1 gives
        Db_i(X) = COL_DISPERSION[i]*v(X) = Db_i|V=1*v(X)/v(X1) -- see point (b)(3))
    Dp_i    = eta_i*Rp^2*v_char/(eps_ap*L)
    k_i|V=1 = Bi_i|V=1 * eta_i * Rp * v_char/L
    IAVE2 (constant-Bi approximation at V=0.5, per the paper's own
        documented "iave=2" mode): k_i|avg = k_i|V=1 * [(1-V0)/(0.5+V0)]^(1/6)
    ka1 = Da1a*v_char/(L*C0_1);  kd1 = Da1d*v_char/L
    ka2 = Da2a*v_char/(L*C0_1);  kd2 = Da2d*v_char/L
    qmax1 = C_inf,1
    tau -> t_phys = tau*L/v_char
    Q (inlet volumetric flow) = v(X1)*X1*2*pi*H*eps_b   (CADET's
        _curVelCoeff = Q/(2*pi*H*eps_b), currentVelocity(X)=_curVelCoeff/X,
        so v(X1) == Q/(2*pi*H*eps_b*X1) by construction)
Reference concentration unit CONC_UNIT = 1 [arbitrary consistent unit] so
that the table's dimensionless C0_i/C_inf,i values can be used directly as
physical concentrations (C0_i,phys = C0_i,table * CONC_UNIT).

NOTE on position-dependent Bi_i(V): FILM_DIFFUSION_DEP is unavailable for
this particle framework (see point (b)(3) above), so, per the paper's own
documented "iave=2" fallback, a single constant FILM_DIFFUSION value (Bi_i
evaluated at V=0.5 via Eq. 14.16) is used; Pe_i/eta_i are genuine constants
in the paper's own model already, so no such approximation is needed there
(and Db_i(X)'s position dependence is instead handled exactly via
COL_DISPERSION_DEP, per point (b)(3)).

Component-3 (complex) note: since C0_3 is not known a priori, the book
nondimensionalizes component 3 using C0_1 (Cb3 = cb3*C0_1), consistent with
the table itself listing C0(row 3) = C0(row 1) = 0.000001 identically.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from addict import Dict
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))
# Locally-built CADET-Core install with two real bugs fixed (FORWARD_FLOW
# direction, and the radial backward-flow dispersion sign bug -- see point
# (b) above); required for this script's native-radial-geometry model to
# work at all. NOT the original C:\...\CADET_compiled\...\aRelease build.
INSTALL_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

# ---------------------------------------------------------------------------
# Step 3: paper's parameters, exactly as read off the Fig. 10.14 GUI
# screenshot (p. 138), applied per p. 210 to RFC with V0=0.04.
# ---------------------------------------------------------------------------
V0 = 0.04
TIMP = 14.0     # protein frontal-loading duration (dimensionless tau)
TSHIFT = 15.0   # switch to soluble-ligand elution feed (dimensionless tau)
TAU_MAX_SIM = 60.0

EPS_B = 0.40
EPS_P_TOTAL = 0.45
EXF = 0.8                     # size-exclusion factor, identical for i=1,2,3
EPS_AP = EXF * EPS_P_TOTAL    # accessible particle porosity used in CADET

# component  PeL   eta   Bi    C_inf     C0         Daa   Dad
PAPER = {
    1: dict(PeL=300.0, eta=10.0, Bi_V1=40.0, C_inf=1.0e-5, C0=1.0e-6, Daa=2.0, Dad=0.2),  # protein
    2: dict(PeL=300.0, eta=10.0, Bi_V1=40.0, C_inf=0.0,    C0=5.0e-6, Daa=2.0, Dad=0.2),  # soluble ligand
    3: dict(PeL=300.0, eta=10.0, Bi_V1=40.0, C_inf=0.0,    C0=1.0e-6, Daa=0.0, Dad=0.0),  # complex (PI)
}
DA1A, DA1D = PAPER[1]['Daa'], PAPER[1]['Dad']
DA2A, DA2D = PAPER[2]['Daa'], PAPER[2]['Dad']
C1_INF = PAPER[1]['C_inf'] / PAPER[1]['C0']

# ---------------------------------------------------------------------------
# Step 2: reparameterization -- native radial geometry, physical (SI) scales.
# See Step 1/docstring point (b) for the full derivation.
# ---------------------------------------------------------------------------
X1 = 0.05                             # outer column radius [m] (inward-flow inlet)
X0 = X1 * np.sqrt(V0 / (1.0 + V0))    # inner radius [m]; V0 = X0^2/(X1^2-X0^2)
BED_LENGTH = X1 - X0                  # radial bed thickness [m]
CYL_HEIGHT = 1.0                      # arbitrary cylinder height [m]
RP = 5.0e-5                           # particle radius [m]
V_REF = 1.0e-4                        # interstitial velocity at X1 (V=1) [m/s]
V_CHAR = 2.0 * V_REF * X1 / (X1 + X0)  # transit-time characteristic velocity
CONC_UNIT = 1.0                        # reference concentration scale

for i, p in PAPER.items():
    p['Db_V1'] = V_REF * BED_LENGTH / p['PeL']
    p['Dp'] = p['eta'] * RP ** 2 * V_CHAR / (EPS_AP * BED_LENGTH)
    p['k_V1'] = p['Bi_V1'] * p['eta'] * RP * V_CHAR / BED_LENGTH
    p['C0_phys'] = p['C0'] * CONC_UNIT
    p['C_inf_phys'] = p['C_inf'] * CONC_UNIT
    # COL_DISPERSION config value for COL_DISPERSION_DEP='POWER_LAW',
    # EXPONENT=1: Db_i(X) = COL_DISPERSION[i]*v(X), so supply Db_i|V=1/v(X1).
    p['col_dispersion_value'] = p['Db_V1'] / V_REF

# "iave = 2" fallback for Bi_i/k_i only (FILM_DIFFUSION_DEP unavailable for
# this particle framework, see point (b)(3)): constant k_i evaluated at
# V=0.5 via Eq. (14.16).
IAVE2_FACTOR = ((1.0 - V0) / (0.5 + V0)) ** (1.0 / 6.0)
for i, p in PAPER.items():
    p['k_avg'] = p['k_V1'] * IAVE2_FACTOR

C0_1_PHYS = PAPER[1]['C0_phys']
C0_2_PHYS = PAPER[2]['C0_phys']
C0_3_PHYS = PAPER[3]['C0_phys']  # == C0_1_PHYS by construction (book convention)
QMAX1_PHYS = PAPER[1]['C_inf_phys']

KA1 = DA1A * V_CHAR / (BED_LENGTH * C0_1_PHYS)
KD1 = DA1D * V_CHAR / BED_LENGTH
KA2 = DA2A * V_CHAR / (BED_LENGTH * C0_1_PHYS)   # book-chapter convention: C0_1, not C0_2
KD2 = DA2D * V_CHAR / BED_LENGTH

Q_FLOW = V_REF * X1 * 2.0 * np.pi * CYL_HEIGHT * EPS_B  # inlet flow [m^3/s]

T_IMP = TIMP * BED_LENGTH / V_CHAR
T_SHIFT = TSHIFT * BED_LENGTH / V_CHAR
T_END = TAU_MAX_SIM * BED_LENGTH / V_CHAR


def dimless_time(t_phys):
    """Map physical simulation time [s] to the paper's tau = v_char*t/(X1-X0)."""
    return np.asarray(t_phys) * V_CHAR / BED_LENGTH


# ---------------------------------------------------------------------------
# Step 5: CADET model definition
# ---------------------------------------------------------------------------
def get_model(ncol=200, par_ncells=4, n_points=900):
    m = Dict()
    m.input.model.nunits = 3

    # 3 sections: 0 = frontal protein loading (0 <= tau < 14); 1 = wash with
    # inert mobile phase (14 <= tau < 15); 2 = elution with soluble ligand
    # feed held constant (tau >= 15). No flow-direction trick of any kind is
    # needed: V=1 (Gu's inward-flow RFC inlet) is *always* CADET's z=0 inlet
    # via genuine inward flow (FORWARD_FLOW=[0,0,0], single unchanging
    # direction across all 3 sections; no priming section needed now that
    # both CADET-Core bugs are fixed -- see docstring point (b)).
    sec_times = [0.0, T_IMP, T_SHIFT, T_END]
    n_sections = 3

    m.input.model.connections.nswitches = n_sections
    for s in range(n_sections):
        key = f'switch_{s:03d}'
        m.input.model.connections[key].connections = [
            0.0, 1.0, -1.0, -1.0, Q_FLOW,
            1.0, 2.0, -1.0, -1.0, Q_FLOW,
        ]
        m.input.model.connections[key].section = s

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # --- Inlet: 3 components (1=protein, 2=soluble ligand, 3=complex,
    # which is never fed) ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 3

    feed_by_section = [
        [C0_1_PHYS, 0.0, 0.0],   # sec 0: frontal protein loading
        [0.0, 0.0, 0.0],         # sec 1: wash (inert mobile phase)
        [0.0, C0_2_PHYS, 0.0],   # sec 2: elution with soluble ligand
    ]
    for s, feed in enumerate(feed_by_section):
        key = f'sec_{s:03d}'
        m.input.model.unit_000[key].const_coeff = feed
        m.input.model.unit_000[key].lin_coeff = [0.0, 0.0, 0.0]
        m.input.model.unit_000[key].quad_coeff = [0.0, 0.0, 0.0]
        m.input.model.unit_000[key].cube_coeff = [0.0, 0.0, 0.0]

    # --- Column: CADET's native radial-flow geometry (see docstring point
    # (b) for the two bug fixes and the COL_DISPERSION_DEP mechanism) ---
    col = Dict()
    col.unit_type = 'COLUMN_MODEL_1D'
    col.geometry = 'RADIAL_FLOW_CYLINDER_SHELL'
    col.ncomp = 3
    col.npartype = 1
    col.par_type_volfrac = 1
    col.cross_section_area_outer = 2.0 * np.pi * X1 * CYL_HEIGHT
    col.cylinder_height = CYL_HEIGHT
    col.bed_length = BED_LENGTH
    col.col_porosity = EPS_B
    col.col_dispersion = [PAPER[1]['col_dispersion_value'], PAPER[2]['col_dispersion_value'], PAPER[3]['col_dispersion_value']]
    col.col_dispersion_dep = 'POWER_LAW'
    col.col_dispersion_dep_exponent = 1.0
    col.forward_flow = [0, 0, 0]
    col.init_c = [0.0, 0.0, 0.0]

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
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

    # --- Bulk-phase (liquid) reaction: P(1) + I(2) <-> PI(3) ---
    col.nreac_liquid = 1
    col.liquid_reaction_000.type = 'MASS_ACTION_LAW'
    col.liquid_reaction_000.mal_stoichiometry = [-1.0, -1.0, 1.0]
    col.liquid_reaction_000.mal_kfwd = [KA2]
    col.liquid_reaction_000.mal_kbwd = [KD2]

    # --- Particles: GENERAL_RATE_PARTICLE (film + pore diffusion,
    # spherical), identical transport parameters for all 3 components ---
    col.particle_type_000.nbound = [1, 0, 0]  # only component 1 (protein) binds
    col.particle_type_000.init_cp = [0.0, 0.0, 0.0]
    col.particle_type_000.init_cs = [0.0]

    col.particle_type_000.has_film_diffusion = 1
    col.particle_type_000.film_diffusion = [PAPER[1]['k_avg'], PAPER[2]['k_avg'], PAPER[3]['k_avg']]
    col.particle_type_000.has_pore_diffusion = 1
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.par_geom = 'SPHERE'
    col.particle_type_000.par_coreradius = 0.0
    col.particle_type_000.par_porosity = EPS_AP
    col.particle_type_000.par_radius = RP
    col.particle_type_000.pore_diffusion = [PAPER[1]['Dp'], PAPER[2]['Dp'], PAPER[3]['Dp']]
    col.particle_type_000.surface_diffusion = [0.0, 0.0, 0.0]

    # Pore-liquid-phase reaction (same reaction, same rate constants -- see
    # Step 1(b) discussion of why eps_ap cancels).
    col.particle_type_000.nreac_liquid = 1
    col.particle_type_000.liquid_reaction_000.type = 'MASS_ACTION_LAW'
    col.particle_type_000.liquid_reaction_000.mal_stoichiometry = [-1.0, -1.0, 1.0]
    col.particle_type_000.liquid_reaction_000.mal_kfwd = [KA2]
    col.particle_type_000.liquid_reaction_000.mal_kbwd = [KD2]

    # Kinetic (non-equilibrium) Langmuir binding, component 1 only
    # (Eq. 10.12: dq1/dt = ka1*cp1*(qmax1 - q1) - kd1*q1). Entries for
    # components 2,3 are unused since nbound=0 there.
    col.particle_type_000.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    col.particle_type_000.adsorption.is_kinetic = 1
    col.particle_type_000.adsorption.mcl_ka = [KA1, 0.0, 0.0]
    col.particle_type_000.adsorption.mcl_kd = [KD1, 0.0, 0.0]
    col.particle_type_000.adsorption.mcl_qmax = [QMAX1_PHYS, 0.0, 0.0]

    col.particle_type_000.discretization.SPATIAL_METHOD = 'FV'
    col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
    col.particle_type_000.discretization.NCELLS = par_ncells
    col.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2

    m.input.model.unit_001 = col

    m.input.model.unit_002.ncomp = 3
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
    m.input.solver.sections.nsec = n_sections
    m.input.solver.sections.section_continuity = [0, 0]
    m.input.solver.sections.section_times = sec_times
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-10
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, sec_times[-1], n_points)

    return m


def run_model(ncol=200, par_ncells=4, n_points=900, fname='fig14_6.h5'):
    model = get_model(ncol=ncol, par_ncells=par_ncells, n_points=n_points)
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
#
# Digitized from the rendered p. 210 figure using pixel colour-thresholding.
# Curves: protein = teal/green solid, soluble ligand = black dashed,
# complex = navy solid. Extraction quality was checked visually by
# overlaying the digitized points against a matplotlib reproduction next to
# a crop of the original page (fig14_6_digitized_reference_check.png,
# alongside this script) -- the digitized curves reproduce the original
# figure's shape, timing, and peak heights closely; the only expected
# artifact is a sparse/gapped sampling of the dashed "soluble ligand" curve
# (dash gaps => missing x-samples there), which is immaterial since the
# curve is smooth and slowly varying in that region.
# ---------------------------------------------------------------------------
def load_digitized(path=None):
    if path is None:
        path = os.path.join(HERE, 'fig14_6_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)

    def clean(tcol, ccol):
        t, c = data[tcol], data[ccol]
        mask = ~(np.isnan(t) | np.isnan(c))
        t, c = t[mask], c[mask]
        order = np.argsort(t)
        return t[order], c[order]

    t1, c1 = clean('time_protein', 'protein')
    t2, c2 = clean('time_soluble_ligand', 'soluble_ligand')
    t3, c3 = clean('time_complex', 'complex')
    return (t1, c1), (t2, c2), (t3, c3)


# ---------------------------------------------------------------------------
# Step 6: validation metrics
# ---------------------------------------------------------------------------
def first_moment(t, c, t_lo=None, t_hi=None):
    t = np.asarray(t)
    c = np.clip(np.asarray(c), 0.0, None)
    if t_lo is not None or t_hi is not None:
        lo = t_lo if t_lo is not None else t.min()
        hi = t_hi if t_hi is not None else t.max()
        mask = (t >= lo) & (t <= hi)
        t, c = t[mask], c[mask]
    area = np.trapz(c, t)
    if area <= 0:
        return np.nan, np.nan
    moment = np.trapz(t * c, t) / area
    return moment, area


def t_at_level(t, c, level, direction='rising'):
    t = np.asarray(t)
    c = np.asarray(c)
    d = np.diff(np.sign(c - level))
    if direction == 'rising':
        idx = np.where(d > 0)[0]
    else:
        idx = np.where(d < 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    t0, t1 = t[i], t[i + 1]
    c0, c1 = c[i], c[i + 1]
    frac = (level - c0) / (c1 - c0)
    return t0 + frac * (t1 - t0)


def compute_metrics(tau_sim, sims, tau_refs, refs):
    """sims/refs: dicts {'protein': c_arr, 'soluble_ligand': c_arr, 'complex': c_arr}"""
    metrics = {}
    for name in ('protein', 'soluble_ligand', 'complex'):
        c_sim = sims[name]
        tau_ref, c_ref = tau_refs[name], refs[name]
        c_sim_i = np.interp(tau_ref, tau_sim, c_sim)

        m = {}

        # 1) Peak position -- meaningful for protein and complex (both show
        # a genuine interior maximum). Soluble ligand rises monotonically
        # to a plateau (no interior peak), so this metric is reported N/A.
        if name in ('protein', 'complex'):
            i_sim = np.argmax(c_sim)
            i_ref = np.argmax(c_ref)
            t_peak_sim, t_peak_ref = tau_sim[i_sim], tau_ref[i_ref]
            m['peak_time_sim'] = t_peak_sim
            m['peak_time_ref'] = t_peak_ref
            m['peak_time_relerr_%'] = 100 * abs(t_peak_sim - t_peak_ref) / t_peak_ref
            m['peak_height_sim'] = c_sim[i_sim]
            m['peak_height_ref'] = c_ref[i_ref]
            m['peak_height_relerr_%'] = 100 * abs(c_sim[i_sim] - c_ref[i_ref]) / c_ref[i_ref]
        else:
            m['peak_time_sim'] = m['peak_time_ref'] = m['peak_time_relerr_%'] = np.nan
            m['peak_height_sim'] = m['peak_height_ref'] = m['peak_height_relerr_%'] = np.nan

        # 2) Elution time (first moment). For protein and complex, c(t)
        # returns close to baseline within the simulated window
        # [0, tau_max=60]. For soluble ligand, c(t) plateaus near 1 and
        # never returns to baseline, so the first moment is dominated by
        # (and diverges with) the upper integration limit; we instead
        # report the breakthrough time t50 (time to cross 50% of the FINAL
        # plateau value), the standard adapted metric for a non-eluting/
        # plateauing curve.
        if name == 'soluble_ligand':
            plateau_val_sim = np.nanmean(c_sim[tau_sim >= tau_sim.max() - 2.0])
            plateau_val_ref = np.nanmean(c_ref[tau_ref >= tau_ref.max() - 2.0])
            t50_sim = t_at_level(tau_sim, c_sim, 0.5 * plateau_val_sim)
            t50_ref = t_at_level(tau_ref, c_ref, 0.5 * plateau_val_ref)
            m['elution_metric'] = 't50 (50% of final plateau), adapted for a non-eluting curve'
            m['elution_time_sim'] = t50_sim
            m['elution_time_ref'] = t50_ref
            m['elution_time_relerr_%'] = 100 * abs(t50_sim - t50_ref) / t50_ref
        else:
            mu_sim, _ = first_moment(tau_sim, c_sim, t_lo=0.0, t_hi=tau_sim.max())
            mu_ref, _ = first_moment(tau_ref, c_ref, t_lo=0.0, t_hi=tau_ref.max())
            m['elution_metric'] = 'first moment int(t*c dt)/int(c dt) over [0, tau_max]'
            m['elution_time_sim'] = mu_sim
            m['elution_time_ref'] = mu_ref
            m['elution_time_relerr_%'] = 100 * abs(mu_sim - mu_ref) / mu_ref

        # 3) Mass balance. For protein: total fed (=C0_1 for 0<=tau<14,
        # i.e. area=14 in dimensionless (C/C0)*tau units) vs.
        # int(c1_out dtau) over the full run. Because protein reacts with
        # the soluble ligand to form the complex, most of the loaded
        # protein leaves the column AS COMPLEX, not as free protein -- so
        # a component-1-only mass balance is expected to show a large
        # "deficit" that reflects real chemistry (conversion + residual
        # binding), not a modeling error (see the printed atom-balance
        # check). For soluble ligand, the feed is a "displacer" that is
        # never turned off (per Eq. 14.12's index=4 "displacer" clause), so
        # its cumulative mass balance only makes sense over a bounded
        # window; per the task's guidance we use the window
        # [tau_shift, tau_max] = [15, 60] (a "suitably long integration
        # window" starting when the ligand feed switches on) and compare
        # int(c2_out dtau) there against the ligand fed over the same
        # window (=1.0*(tau_max-tau_shift)=45 in (C/C0)*tau units); some
        # ligand mass is expected to be "missing" here too since part of it
        # reacts to form the complex rather than exiting as free ligand.
        # For complex: it is not fed at all, so mass balance is simply
        # int(c3_out dtau) over the full run compared between simulation
        # and digitized reference.
        if name == 'protein':
            fed = 1.0 * TIMP
            out_sim = np.trapz(c_sim, tau_sim)
            out_ref = np.trapz(c_ref, tau_ref)
            m['mass_balance_metric'] = 'int(c_out dtau) vs. protein fed (=1*tau_imp); ' \
                                        'deficit reflects protein retained on-column + ' \
                                        'converted to complex (see printed atom-balance check)'
            m['mass_fed'] = fed
            m['mass_out_sim'] = out_sim
            m['mass_out_ref'] = out_ref
            m['mass_relerr_sim_vs_fed_%'] = 100 * abs(out_sim - fed) / fed
            m['mass_relerr_sim_vs_ref_%'] = 100 * abs(out_sim - out_ref) / out_ref if out_ref > 0 else np.nan
        elif name == 'soluble_ligand':
            window = (tau_sim >= TSHIFT)
            out_sim = np.trapz(c_sim[window], tau_sim[window])
            window_ref = (tau_ref >= TSHIFT)
            out_ref = np.trapz(c_ref[window_ref], tau_ref[window_ref]) if window_ref.sum() > 1 else np.nan
            fed = 1.0 * (TAU_MAX_SIM - TSHIFT)
            m['mass_balance_metric'] = f'int(c_out dtau) over [tau_shift={TSHIFT}, tau_max={TAU_MAX_SIM}] ' \
                                        'vs. ligand fed over same window; deficit reflects ligand ' \
                                        'consumed by ongoing complex formation (see atom-balance check)'
            m['mass_fed'] = fed
            m['mass_out_sim'] = out_sim
            m['mass_out_ref'] = out_ref
            m['mass_relerr_sim_vs_fed_%'] = 100 * abs(out_sim - fed) / fed
            m['mass_relerr_sim_vs_ref_%'] = 100 * abs(out_sim - out_ref) / out_ref if (out_ref and out_ref > 0) else np.nan
        else:  # complex
            out_sim = np.trapz(c_sim, tau_sim)
            out_ref = np.trapz(c_ref, tau_ref)
            m['mass_balance_metric'] = 'int(c_out dtau), sim vs. digitized reference (no independent ' \
                                        'feed-side reference for a non-fed product species)'
            m['mass_fed'] = np.nan
            m['mass_out_sim'] = out_sim
            m['mass_out_ref'] = out_ref
            m['mass_relerr_sim_vs_fed_%'] = np.nan
            m['mass_relerr_sim_vs_ref_%'] = 100 * abs(out_sim - out_ref) / out_ref if out_ref > 0 else np.nan

        # 4) Chromatogram MSE over the full digitized time window
        m['mse'] = np.nanmean((c_sim_i - c_ref) ** 2)

        metrics[name] = m

    return metrics


def print_metrics(metrics):
    for comp, m in metrics.items():
        print(f"\n--- {comp} ---")
        if not np.isnan(m['peak_time_sim']):
            print(f"  Peak position   : sim tau={m['peak_time_sim']:.4g}  ref tau={m['peak_time_ref']:.4g}"
                  f"  rel.err={m['peak_time_relerr_%']:.3g}%")
            print(f"  Peak height     : sim={m['peak_height_sim']:.4g}  ref={m['peak_height_ref']:.4g}"
                  f"  rel.err={m['peak_height_relerr_%']:.3g}%")
        else:
            print("  Peak position   : N/A (monotonic rise to plateau, no interior peak)")
        print(f"  Elution metric  : {m['elution_metric']}")
        print(f"  Elution time    : sim={m['elution_time_sim']:.4g}  ref={m['elution_time_ref']:.4g}"
              f"  rel.err={m['elution_time_relerr_%']:.3g}%")
        print(f"  Mass balance    : {m['mass_balance_metric']}")
        fed_str = f"{m['mass_fed']:.4g}" if not np.isnan(m['mass_fed']) else "N/A"
        print(f"    fed={fed_str}  out(sim)={m['mass_out_sim']:.4g}  out(ref)={m['mass_out_ref']:.4g}")
        if not np.isnan(m['mass_relerr_sim_vs_fed_%']):
            print(f"    sim vs. fed rel.err={m['mass_relerr_sim_vs_fed_%']:.3g}%")
        if not np.isnan(m['mass_relerr_sim_vs_ref_%']):
            print(f"    sim vs. ref rel.err={m['mass_relerr_sim_vs_ref_%']:.3g}%")
        print(f"  Chromatogram MSE: {m['mse']:.4g}")


def print_atom_balance(tau_sim, c1_sim, c2_sim, c3_sim):
    """Diagnostic (not one of the 4 core metrics): since P + I -> PI is a
    1:1 reaction and c3 is normalized by C0_1 (book convention), a
    "protein-equivalent" balance should approximately hold:
        protein fed (=1*tau_imp) ~= int(c1_out dtau) + int(c3_out dtau)
                                     + protein remaining bound on-column
    """
    fed = 1.0 * TIMP
    out_protein = np.trapz(c1_sim, tau_sim)
    out_complex = np.trapz(c3_sim, tau_sim)
    print("\n--- Diagnostic: protein-equivalent atom balance (not one of the 4 core metrics) ---")
    print(f"  Protein fed (1*tau_imp)              : {fed:.4g}")
    print(f"  int(c1_out dtau) [free protein out]  : {out_protein:.4g}")
    print(f"  int(c3_out dtau) [as complex out]    : {out_complex:.4g}")
    print(f"  Sum (free + complex, protein-equiv.) : {out_protein + out_complex:.4g}")
    residual = fed - (out_protein + out_complex)
    print(f"  Residual (interpreted as protein still bound on-column"
          f" at tau_max={TAU_MAX_SIM}): {residual:.4g}  ({100 * residual / fed:.3g}% of fed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Physical (SI-like) parameters derived from the paper's dimensionless groups:")
    for i, p in PAPER.items():
        print(f"  Component {i}: Db(V=1)={p['Db_V1']:.4g} m^2/s (COL_DISPERSION={p['col_dispersion_value']:.4g}), "
              f"Dp={p['Dp']:.4g} m^2/s, "
              f"k(V=1)={p['k_V1']:.4g} m/s, k(avg,V=0.5,iave2)={p['k_avg']:.4g} m/s, "
              f"C0={p['C0_phys']:.4g}")
    print(f"  KA1={KA1:.4g}, KD1={KD1:.4g}, QMAX1={QMAX1_PHYS:.4g}  (component-1 kinetic Langmuir)")
    print(f"  KA2(=KFWD)={KA2:.4g}, KD2(=KBWD)={KD2:.4g}  (shared P+I<->PI mass-action reaction)")
    print(f"  X0={X0:.4g} m, X1={X1:.4g} m, bed_length={BED_LENGTH:.4g} m, "
          f"Q={Q_FLOW:.4g} m^3/s, T_END={T_END:.4g} s")
    print(f"  eps_b={EPS_B}, eps_p(total)={EPS_P_TOTAL}, ExF={EXF}, eps_ap(CADET PAR_POROSITY)={EPS_AP}")
    print(f"  iave=2 averaging factor at V=0.5: {IAVE2_FACTOR:.5g}")

    print("\nRunning CADET simulation (native radial geometry, genuine inward flow -- see script docstring)...")
    t_phys, outlet = run_model()
    tau_sim = dimless_time(t_phys)
    c1_sim = outlet[:, 0] / C0_1_PHYS
    c2_sim = outlet[:, 1] / C0_2_PHYS
    c3_sim = outlet[:, 2] / C0_3_PHYS

    print("Loading digitized reference data...")
    (t1_ref, c1_ref), (t2_ref, c2_ref), (t3_ref, c3_ref) = load_digitized()

    print("Computing validation metrics...")
    sims = {'protein': c1_sim, 'soluble_ligand': c2_sim, 'complex': c3_sim}
    tau_refs = {'protein': t1_ref, 'soluble_ligand': t2_ref, 'complex': t3_ref}
    refs = {'protein': c1_ref, 'soluble_ligand': c2_ref, 'complex': c3_ref}
    metrics = compute_metrics(tau_sim, sims, tau_refs, refs)
    print_metrics(metrics)
    print_atom_balance(tau_sim, c1_sim, c2_sim, c3_sim)

    # --- comparison plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(tau_sim, c1_sim, '-', color='#3dc498', lw=1.8, label='Protein (CADET)')
    ax.plot(tau_sim, c2_sim, '--', color='black', lw=1.5, label='Soluble ligand (CADET)')
    ax.plot(tau_sim, c3_sim, '-', color='#39408f', lw=1.8, label='Complex (CADET)')
    ax.plot(t1_ref, c1_ref, 'o', color='#3dc498', ms=3, mfc='none', mew=1.0,
            label='Protein (digitized, Gu 2015)')
    ax.plot(t2_ref, c2_ref, 's', color='black', ms=3, mfc='none', mew=1.0,
            label='Soluble ligand (digitized, Gu 2015)')
    ax.plot(t3_ref, c3_ref, '^', color='#39408f', ms=3, mfc='none', mew=1.0,
            label='Complex (digitized, Gu 2015)')
    ax.set_xlabel('Dimensionless time, ' + r'$\tau = v_{char}t/(X_1-X_0)$')
    ax.set_ylabel('Dimensionless concentration, ' + r'$C/C_0$')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.2)
    fig.suptitle('Gu (2015), Fig. 14.6 -- affinity RFC with inward flow', y=0.985, fontsize=12)
    ax.set_title('CADET native radial geometry; velocity-scaled dispersion via\n'
                  'COL_DISPERSION_DEP; iave=2 constant-Bi approx.; see script docstring',
                  fontsize=9)
    ax.legend(loc='center right', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(HERE, 'fig14_6_comparison.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved comparison plot to {outpath}")
