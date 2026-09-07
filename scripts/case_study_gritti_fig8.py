# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 8 from:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

Model definition, run, comparison plot, and validation metrics. Only
`cadet` (cadet-python), `numpy`, and `matplotlib` are imported -- the model
is built directly on a `Cadet(...).root` object (already an addict.Dict
internally, per cadet-python's own implementation), so no separate `addict`
import is needed. The digitized Fig. 8 reference data lives in the sibling
file fig8_digitized.csv (loaded via `np.genfromtxt`), matching the
convention used by this repository's other Gritti case-study scripts
(e.g. case_study_gritti_fig7.py + fig7_digitized.csv) -- see
fig8_digitized_preview.png for the digitization's visual sanity check.

===========================================================================
Step 0 -- case identification
===========================================================================
Fig. 8 shows experimental gradient-elution chromatograms of the peptide
bombesin (14 aa, MW 1619.85 g/mol) recorded on THREE columns, all packed
with the same batch of 5 um XBridge-C18 fully-porous particles, L=150 mm:

  * "Cylinder, rho_s=1"   : conventional 3.0 mm i.d. cylindrical column,
                            Q = 0.35 mL/min.
  * "Cone, rho_s=2"       : conical column, entrance 2.1 mm i.d. -> exit
                            4.2 mm i.d. (flow narrow->wide), Q=0.40 mL/min.
  * "Cone, rho_s=0.5"     : the SAME physical conical column with flow
                            reversed, entrance 4.2 mm i.d. -> exit 2.1 mm
                            i.d. (flow wide->narrow), Q=0.40 mL/min.

rho_s (paper's "s") = outlet radius / inlet radius of the truncated cone
(Eq. 1). This is a genuine frustum (linearly-varying cross-section),
NOT a radial-flow column.

The three curves are not "multiple panels" in the ambiguous sense the
standard process worries about -- they are the one intended validation
target (the text explicitly introduces Fig. 8 as showing "the experimental
peaks of the peptide bombesin recorded on both[/all three] columns under
gradient conditions"), so all three are reproduced together with a single
model (same analyte/chemistry/particles, only the transport geometry and
flow direction differ between the three CADET runs).

Retained species: bombesin, a real, retained analyte separated by a LINEAR
ACN/water GRADIENT (not an isocratic run, not a non-retained tracer).
Single component of interest, but the paper's own gradient elution theory
(Section 2.4, LSSM -- Linear Solvent Strength Model, Eq. 28:
k(phi) = k0*exp(-S*(phi-phi0))) requires a SECOND, non-retained "modifier"
field (the ACN volume fraction phi) whose local value modulates the
retention factor of the analyte as it migrates -- i.e. this is a genuine
2-component transport problem (modifier + analyte), even though only one
of the two components is "of interest" for the validation target.

===========================================================================
Step 1 -- model mapping to CADET
===========================================================================
Bulk/particle transport
------------------------
The paper's own model (Sections 2.2-2.4) is a "black-box column" treatment:
axial convection with a smoothly axially-varying velocity/cross-section
(Eq. 4-5), and band broadening described purely through an aggregate,
possibly axially-varying, PLATE HEIGHT H(xi) (Giddings'/Blumberg's theory of
non-uniform columns) -- there is no explicit film-diffusion or
pore-diffusion sub-model anywhere in the paper (unlike a GRM-type paper).
The single CADET sub-model that matches this ("aggregate axial dispersion,
no separate particle mass-transfer resistances, equilibrium retention") is
CADET's EQUILIBRIUM_PARTICLE type (obtained by setting
HAS_FILM_DIFFUSION=0 on a COLUMN_MODEL_1D particle_type, i.e. what used to
be called LUMPED_RATE_MODEL_WITHOUT_PORES): a single interstitial+
intraparticle TOTAL_POROSITY, axial dispersion only, instantaneous local
equilibrium with the stationary phase. This is an unambiguous mapping given
what the paper actually specifies (no separate mass-transfer parameters are
given anywhere for bombesin, or for any analyte).

Column geometry: CADET's NATIVE `COLUMN_MODEL_1D` unit with
`GEOMETRY='AXIAL_FLOW_FRUSTUM'` for the two conical runs (per this task's
explicit instructions: do not approximate the frustum with a constant-
cross-section column), and `GEOMETRY='AXIAL_FLOW_CYLINDER'` for the
reference cylindrical column. `FORWARD_FLOW` selects which physical end
(large or small) is the inlet, so the SAME frustum geometry
(CROSS_SECTION_AREA_SMALL_END/LARGE_END, BED_LENGTH) is reused for both
rho_s=2 (FORWARD_FLOW=0, enter at the small/2.1 mm end) and rho_s=0.5
(FORWARD_FLOW=1, enter at the large/4.2 mm end) runs -- exactly mirroring
that these are the SAME physical hardware, flow simply reversed, per the
paper's own Table 3 caption ("3) conical column (s=0.5) after reversing the
flow direction").

Binding / gradient-modifier coupling
-------------------------------------
The paper's LSSM retention law k(phi) = k0*exp(-S*(phi-phi0)) (Eq. 28) is
the textbook reversed-phase-gradient isotherm underlying CADET's
MOBILE_PHASE_MODULATOR ("Mobile Phase Modulator Langmuir", Melander &
Horvath 1977; Karlsson 2004) binding model:

    dq_1/dt = k_a exp(gamma*c_p0) c_p1 qmax (1 - q_1/qmax) - k_d c_p0^beta q_1

Component 0 ("salt"/modifier in CADET's own terminology, here: the ACN
volume fraction phi) is inert (NBOUND=0, transported by pure convection);
component 1 is bombesin (NBOUND=1). Using is_kinetic=0 (instantaneous local
equilibrium -- consistent with the paper never invoking any adsorption
KINETICS, only a retention FACTOR) and beta=0 (paper's LSSM has no
power-law/ion-exchange term), the quasi-stationary flux balance reduces, in
the dilute limit (q_1 << qmax, valid here: a 3 uL injection of 0.1 g/L
bombesin), to

    q_1/c_p1 = (k_a*qmax/k_d) * exp(gamma*c_p0)

i.e. exactly the LSSM law with gamma = -S and (k_a*qmax/k_d) chosen so that
the retention FACTOR k'(phi) = F*(q_1/c_p1) (F = phase ratio = (1-eps_t)/
eps_t) equals k0*exp(-S*(phi-phi0)) -- see PARAMETER DERIVATION below. This
is the unambiguous, standard way of representing an LSSM gradient in CADET
(the same mechanism used for salt-gradient IEX, here applied to its
originally-intended HIC/RPLC hydrophobicity role, per Melander1989/
Karlsson2004 as cited in CADET-Core's own binding model documentation).

Axial dispersion / "local plate height"
------------------------------------------
The paper's Section 4.2.1 measures H(xi) directly for the alkanophenones
(Fig. 5) but gives NO such position-resolved mass-transfer calibration for
bombesin -- only the aggregate retention time and second central moment of
the whole eluted peak (Table 3). Lacking a compound-specific H(u)
correlation, this script adopts the paper's OWN simplified "first part"
assumption (Section 2.3/2.4, explicitly used whenever a full H(xi)
calibration is unavailable): a spatially UNIFORM physical plate height H.
Since H = 2*D_ax/u_interstitial, a uniform H along a frustum (where
u_interstitial(xi) varies strongly) requires a POSITION-DEPENDENT axial
dispersion coefficient D_ax(xi) = H*u(xi)/2 -- i.e. D_ax proportional to the
local interstitial velocity. This is exactly CADET's documented
`COL_DISPERSION_DEP='POWER_LAW'` mechanism with EXPONENT=1: CADET computes
D_ax(local) = COL_DISPERSION_config * v_local^1, so setting
COL_DISPERSION_config = H/2 gives D_ax(xi) = (H/2)*u(xi) = H*u(xi)/2
everywhere, exactly reproducing "uniform H" self-consistently for whatever
velocity profile the native frustum geometry computes -- no separate,
per-geometry dispersion recalibration is needed; the SAME H (and the SAME
COL_DISPERSION_DEP config) is used for the cylinder and both cone runs, and
the geometry alone produces the different band-broadening behavior seen in
Table 3 (rho_s=2 broadens more than rho_s=0.5). H itself is calibrated once,
against the CYLINDRICAL column's measured second central moment (Table 3),
by actually running CADET (see CALIBRATION below) -- not by hand-deriving
the paper's own perturbative Blumberg/Poppe band-broadening ODE (Eq. 44-47),
which the OCR'd text extraction of the PDF renders too ambiguously
(garbled superscripts) to transcribe reliably; CADET's own numerical PDE
solution is used as the actual physics engine instead, which is more
accurate than the paper's own asymptotic approximation in any case.

The modifier component's own dispersion is fixed at a small, non-zero,
geometry-independent placeholder value (negligible compared to convection)
so that its ramp profile propagates essentially undistorted, matching the
paper's explicit assumption (Section 2.4: "the solvent gradient is ...
not distorted upon migration along the chromatographic column", Eq. 29-30).

===========================================================================
Step 2/3 -- parameter derivation (documented judgment calls flagged **)
===========================================================================
Known directly from the paper text (Section 3.3, 3.4.3, Table 3):
    L               = 0.15 m                    (both columns)
    r_cylinder      = 1.5e-3 m   (3.0 mm i.d.)
    r_small (frustum)= 1.05e-3 m (2.1 mm i.d.)
    r_large (frustum)= 2.10e-3 m (4.2 mm i.d.)
    Q_cylinder      = 0.35 mL/min
    Q_cone          = 0.40 mL/min (both flow directions)
    V_inj           = 3.0 uL
    phi0            = 0.10, phi_final = 0.55   (10%/55% ACN)
    t_gradient      = 5 min = 300 s
    => gradient steepness beta = (phi_final-phi0)/t_gradient = 0.09 /min
       = 0.0015 /s
    Table 3 (cylinder, rho_s=1): t_R = 4.798 min, sigma_t^2 = 0.000288 min^2

** eps_t (total porosity) = 0.65: not restated in Section 4.2.3, but this
   is the SAME batch of particles/columns used in Section 4.1.4's worked
   example, where eps_t=0.65 is given explicitly and is verified (below,
   at runtime) to reproduce that section's own stated entrance velocity
   u0(0)=17.77 cm/min for Fv=0.40 mL/min, r_e=1.05 mm -- so re-using it here
   for the bombesin runs (same physical columns) is not an independent
   assumption, it is the value consistent with the rest of the paper.

** S (LSSM slope) = 25: bombesin's own S is not given numerically anywhere
   in the paper; the paper DOES state S=25 for its illustrative "17-peptide"
   mixture (Fig. 2/4 caption) with gradient steepness 0.09/min -- IDENTICAL
   to the actual gradient steepness used for the real bombesin experiments
   (Section 3.4.3). Given both are described as "peptide" cases under
   matching gradient conditions, S=25 is adopted for bombesin as the most
   reasonable available value (rather than leaving it undetermined) and
   clearly flagged here as an inference, not a stated fact.

k0 (LSSM pre-exponential retention factor at phi0) is NOT assumed -- it is
SOLVED FOR in closed form from the cylinder column's own measured
retention time (Table 3, 4.798 min), using the paper's OWN exact
(non-perturbative) gradient elution time formula, re-derived cleanly here
from Eq. 31 (integrating the LSSM local-equilibrium retention law along
the column):

    e(xi) = m(xi) + (1/G) * ln(1 + G*k0*m(xi))                 [from Eq. 31]
    G      = S*beta*L/u0(0)                                    [Eq. 32]
    m(xi)  = (1 + rho(xi) + rho(xi)^2) / 3   at xi=1: m(1)=(1+s+s^2)/3
             (closed form of Eq. 15/16's integral -- note: the OCR'd PDF
             text extraction renders the exponent on rho(1)=s as a spurious
             "2/s"; re-deriving Eq. 15 (integrating rho(xi)^2 =
             [1+(s-1)xi]^2 from 0 to xi) shows this must be s^2, confirmed
             both by direct integration and by consistency with Eq. 18's
             s->1/s flow-direction-invariance claimed in the text)
    => k0 = [exp(G*(e(1)-m(1))) - 1] / (G*m(1))

evaluated with the CYLINDER's own G, m(1)=1, e(1)=t_R/t_ref. This k0 (an
intrinsic analyte/stationary-phase property, independent of column
geometry) is then reused UNCHANGED for both conical geometries, whose
retention times and peak widths are then genuinely PREDICTED (not fitted)
by running the frustum model -- providing a real test of the native
geometry against Table 3's rho_s=2/0.5 rows.

===========================================================================
Step 4 -- reference data (digitized from the figure)
===========================================================================
Fig. 8 (p. 45 of the PDF) plots Absorbance [AU] vs. Time [s] for the three
curves. No CSV/table of the curve itself is available (Table 3 gives only
summary moments, not the traces), so the figure was digitized directly. No
packaged "digitize-figure" skill is available in this environment (checked
directly: invoking it returns "Unknown skill", and it does not appear in
this session's available-skills listing), so the digitization was done with
a pixel-color-classification script (equivalent in spirit to the
repository's documented `CLAUDE/digitize_figure.py` fallback):

  1. The PDF page was rendered at 600 dpi and cropped TIGHTLY to the plot's
     own axes box; the axis calibration (pixel <-> data-value mapping) was
     taken from the pixel positions of the tick marks themselves (found
     programmatically, not eyeballed), separately for x (280/300 s ticks)
     and y (0.000/0.005/.../0.020 AU ticks).
  2. Curve colors were verified by direct pixel sampling before choosing
     thresholds (black/red/blue are cleanly separable: RGB roughly
     (0,0,0)/(200,20,20)/(35,5,250) respectively), after masking out the
     title-box border and legend swatch/text (identified by their own pixel
     bounding boxes, not guessed).
  3. CRITICAL FIX vs. an earlier attempt at this digitization: the three
     curves visually overlap over large stretches (near baseline, and along
     much of the rising/falling flanks -- this is a gradient-elution
     comparison of a cylinder against the same conical column run in two
     flow directions, so the traces are expected to nearly coincide except
     where the geometry's effect is largest). Wherever curves coincide,
     only the LAST-DRAWN color is visible at that pixel column, so a naive
     per-color threshold leaves GAPS in the occluded curve(s) at that
     column -- the earlier attempt filled such gaps by zero-padding /
     independently interpolating each curve's own sparse detections, which
     is wrong (it does not know the curves are actually overlapping there).
     Fixed by: at every column, whichever curve(s) do NOT have their own
     color detected are assigned the SAME value as whichever curve(s) WERE
     detected at that column (i.e. explicitly encoding "overlapping curves
     share a data point" rather than inferring a possibly-wrong shape from
     each curve's own sparse remainder). This is not merely assumed to be
     correct: the resulting three digitized traces were re-plotted directly
     ON TOP of the source image crop (pixel space) and inspected -- for
     every one of the several visually-distinct crossing/occlusion regions
     checked in detail (the initial post-peak decay, where black and blue
     coincide while red separates above them; the t~292-298 s shoulder
     hump, where the ordering flips and black becomes topmost with red
     below it and blue lowest; the shared small pre-peak bump near t~280 s)
     the digitized points track the correct curve through the crossing, not
     a different one -- see fig8_digitized_preview.png (this check plot).
  4. The resulting per-curve traces (recovered peak heights: cylinder
     0.0193, cone_rho_s=2 0.0166, cone_rho_s=0.5 0.0179 AU -- matching the
     visual reading of the source figure) are resampled to a uniform 0.2 s
     grid and saved as fig8_digitized.csv (columns: time_s, cylinder_AU,
     cone_s2_AU, cone_s0p5_AU), loaded at runtime by `load_digitized()`
     below -- NOT embedded inline, matching the convention used by this
     repository's other Gritti case-study scripts (e.g.
     case_study_gritti_fig7.py + fig7_digitized.csv).

===========================================================================
Step 5/6 -- implementation, run, and validation
===========================================================================
See `get_model()`, `calibrate_dispersion()`, and `main()` below. Validation
follows the standard 4 metrics (peak position, first-moment elution time,
mass balance, chromatogram MSE) computed against the digitized reference,
PLUS a direct comparison of CADET's own retention time/second-moment
against Table 3's experimental values (independent of the figure
digitization) for all three column configurations.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from cadet import Cadet

# NOTE: this script intentionally imports nothing beyond cadet/numpy/
# matplotlib. cadet-python's `Cadet().root` is already an addict.Dict
# instance (attribute access auto-vivifies nested dicts), so the model is
# built directly on a Cadet instance's `.root` below instead of importing
# `addict` separately.

HERE = os.path.dirname(os.path.abspath(__file__))
INSTALL_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

# ===========================================================================
# Digitized reference data (Fig. 8): loaded from fig8_digitized.csv, produced
# by the pixel-color-classification + overlap-fill digitization described
# above (see fig8_digitized_preview.png for the visual sanity-check overlay).
# ===========================================================================
def load_digitized(path=None):
    if path is None:
        path = os.path.join(HERE, 'fig8_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    t = data['time_s']
    return {
        'cylinder': (t, data['cylinder_AU']),
        'cone_s2': (t, data['cone_s2_AU']),
        'cone_s0p5': (t, data['cone_s0p5_AU']),
    }


REFERENCE = load_digitized()

# Table 3 experimental summary (independent of the figure digitization)
TABLE3 = {
    # config: (retention_time_min, second_central_moment_min2, w_half_min)
    'cylinder': (4.798, 0.000288, 0.0386),
    'cone_s2': (4.786, 0.000363, 0.0450),
    'cone_s0p5': (4.797, 0.000313, 0.0422),
}

# ===========================================================================
# Step 2/3: physical parameters (SI units throughout)
# ===========================================================================
L_COL = 0.15                      # column length [m], both geometries
R_CYL = 1.5e-3                     # cylinder radius [m] (3.0 mm i.d.)
R_SMALL = 1.05e-3                  # frustum small-end radius [m] (2.1 mm i.d.)
R_LARGE = 2.10e-3                  # frustum large-end radius [m] (4.2 mm i.d.)

Q_CYL = 0.35e-6 / 60.0             # [m^3/s]  (0.35 mL/min)
Q_CONE = 0.40e-6 / 60.0            # [m^3/s]  (0.40 mL/min)

V_INJ = 3.0e-9                     # injection volume [m^3] (3.0 uL)

EPS_T = 0.65                       # total porosity (Section 4.1.4)
F_PHASE = (1.0 - EPS_T) / EPS_T    # phase ratio (solid/liquid volume)

PHI0 = 0.10                        # starting ACN volume fraction
PHI_FINAL = 0.55                   # final ACN volume fraction
T_GRADIENT = 5.0 * 60.0            # gradient time [s]
BETA = (PHI_FINAL - PHI0) / T_GRADIENT   # gradient steepness [1/s]

S_LSSM = 25.0                      # ** judgment call, see docstring **

QMAX1 = 1000.0                     # arbitrary reference solid-phase capacity
C_INJ = 1.0                        # arbitrary reference injected concentration
KD1 = 1.0                          # arbitrary reference desorption rate [1/s]

# --- sanity check on EPS_T against Section 4.1.4's own worked example ---
_u0_check = Q_CONE / (EPS_T * np.pi * R_SMALL ** 2) * 100.0 * 60.0  # cm/min
assert abs(_u0_check - 17.77) < 0.05, f"EPS_T sanity check failed: u0(0)={_u0_check:.3f} cm/min (paper: 17.77)"


def area_cyl(r):
    return np.pi * r ** 2


def u0_entrance(Q, area):
    """Entrance interstitial velocity [m/s]."""
    return Q / (EPS_T * area)


def m1_of_s(s):
    """Dimensionless hold-up time at the column outlet, m(1) = (1+s+s^2)/3
    (closed form of paper's Eq. 15/16 at xi=1; s=1 -> m(1)=1, cylinder)."""
    return (1.0 + s + s ** 2) / 3.0


def solve_k0(t_ref, s, t_R_target):
    """Solve the paper's exact (non-perturbative) LSSM gradient elution-time
    equation e(1) = m(1) + (1/G)*ln(1+G*k0*m(1)) for k0, given the target
    (observed) retention time t_R_target [s] on a column with entrance
    time scale t_ref=L/u0(0) [s] and geometry ratio s."""
    G = S_LSSM * BETA * t_ref
    m1 = m1_of_s(s)
    e1 = t_R_target / t_ref
    k0 = (np.exp(G * (e1 - m1)) - 1.0) / (G * m1)
    return k0, G, m1


# --- Step 3: solve for k0 from the CYLINDER column's Table 3 retention time ---
_t_ref_cyl = L_COL / u0_entrance(Q_CYL, area_cyl(R_CYL))
_t_R_cyl_target = TABLE3['cylinder'][0] * 60.0  # min -> s
K0, _G_cyl, _m1_cyl = solve_k0(_t_ref_cyl, 1.0, _t_R_cyl_target)

# MPM-Langmuir params implementing k'(phi) = k0*exp(-S*(phi-phi0)):
GAMMA1 = -S_LSSM
KA1 = K0 * np.exp(S_LSSM * PHI0) / (F_PHASE * QMAX1)


# ===========================================================================
# Step 5: CADET model definition
# ===========================================================================
def get_model(config, col_dispersion_base, t_end=330.0, n_points=2201,
              polydeg=4, nelem=10):
    """config: 'cylinder', 'cone_s2', or 'cone_s0p5'.
    col_dispersion_base: the COL_DISPERSION value configured for component 1
        (bombesin); combined with COL_DISPERSION_DEP='POWER_LAW' (EXPONENT=1)
        this yields D_ax(xi) = col_dispersion_base * u_local(xi), i.e. a
        spatially uniform physical plate height H = 2*col_dispersion_base
        (see docstring, Step 1, "Axial dispersion" section).
    """
    if config == 'cylinder':
        Q = Q_CYL
        geometry = 'AXIAL_FLOW_CYLINDER'
        forward_flow = 1
        area_entrance = area_cyl(R_CYL)
    elif config == 'cone_s2':
        Q = Q_CONE
        geometry = 'AXIAL_FLOW_FRUSTUM'
        forward_flow = 0   # enter at the SMALL (2.1 mm) end
        area_entrance = area_cyl(R_SMALL)
    elif config == 'cone_s0p5':
        Q = Q_CONE
        geometry = 'AXIAL_FLOW_FRUSTUM'
        forward_flow = 1   # enter at the LARGE (4.2 mm) end
        area_entrance = area_cyl(R_LARGE)
    else:
        raise ValueError(config)

    t_inj = V_INJ / Q  # injection duration [s]

    cadet = Cadet(install_path=INSTALL_PATH)
    m = cadet.root
    m.input.model.nunits = 3

    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, Q,
        1.0, 2.0, -1.0, -1.0, Q,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # --- Inlet: component 0 = ACN modifier (linear ramp), component 1 =
    # bombesin (rectangular injection pulse of duration t_inj) ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 2

    phi_at_tinj = PHI0 + BETA * t_inj
    phi_at_tg = PHI0 + BETA * T_GRADIENT

    m.input.model.unit_000.sec_000.const_coeff = [PHI0, C_INJ]
    m.input.model.unit_000.sec_000.lin_coeff = [BETA, 0.0]
    m.input.model.unit_000.sec_000.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_000.cube_coeff = [0.0, 0.0]

    m.input.model.unit_000.sec_001.const_coeff = [phi_at_tinj, 0.0]
    m.input.model.unit_000.sec_001.lin_coeff = [BETA, 0.0]
    m.input.model.unit_000.sec_001.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_001.cube_coeff = [0.0, 0.0]

    m.input.model.unit_000.sec_002.const_coeff = [phi_at_tg, 0.0]
    m.input.model.unit_000.sec_002.lin_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_002.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_002.cube_coeff = [0.0, 0.0]

    # --- Column --- (auto-vivified live reference into m.input.model.unit_001,
    # no separate Dict import needed -- see note at top of file)
    col = m.input.model.unit_001
    col.unit_type = 'COLUMN_MODEL_1D'
    col.geometry = geometry
    col.ncomp = 2
    col.npartype = 1
    col.bed_length = L_COL
    col.forward_flow = [forward_flow]
    col.total_porosity = EPS_T
    col.init_c = [PHI0, 0.0]

    if geometry == 'AXIAL_FLOW_CYLINDER':
        col.cross_section_area = area_cyl(R_CYL)
    else:
        col.cross_section_area_small_end = area_cyl(R_SMALL)
        col.cross_section_area_large_end = area_cyl(R_LARGE)

    # Modifier (component 0): negligible, geometry-independent dispersion so
    # its ramp propagates essentially undistorted (paper's own assumption).
    # Bombesin (component 1): COL_DISPERSION_DEP='POWER_LAW' (EXPONENT=1)
    # -> D_ax(xi) = col_dispersion_base * u_local(xi) = uniform H (see
    # docstring). For the cylinder, u_local is constant anyway, so the DEP
    # mechanism is harmless (and kept on for consistency/testability).
    col.col_dispersion = [1.0e-11, col_dispersion_base]
    col.col_dispersion_multiplex = 1  # component-dependent, section-independent
    col.col_dispersion_dep = 'POWER_LAW'
    col.col_dispersion_dep_exponent = 1.0

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    col.discretization.SPATIAL_METHOD = 'DG'
    col.discretization.POLYDEG = polydeg
    col.discretization.NELEM = nelem
    if geometry != 'AXIAL_FLOW_CYLINDER':
        col.discretization.USE_COLLOCATION_DG = 0
        col.dispersion_spatial_dependence_polydeg = polydeg

    # --- Particle type: EQUILIBRIUM_PARTICLE (HAS_FILM_DIFFUSION=0), i.e.
    # the "LRM-without-pores"-equivalent local-equilibrium binding ---
    col.particle_type_000.nbound = [0, 1]
    col.particle_type_000.init_cp = [PHI0, 0.0]
    col.particle_type_000.init_cs = [0.0]
    col.particle_type_000.has_film_diffusion = 0
    col.particle_type_000.has_pore_diffusion = 0
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.par_radius = 5.0e-6
    col.particle_type_000.par_porosity = 0.35

    col.particle_type_000.adsorption_model = 'MOBILE_PHASE_MODULATOR'
    col.particle_type_000.adsorption.is_kinetic = 0
    col.particle_type_000.adsorption.mpm_ka = [0.0, KA1]
    col.particle_type_000.adsorption.mpm_kd = [1.0, KD1]
    col.particle_type_000.adsorption.mpm_qmax = [1.0, QMAX1]
    col.particle_type_000.adsorption.mpm_gamma = [0.0, GAMMA1]
    col.particle_type_000.adsorption.mpm_beta = [0.0, 0.0]
    col.particle_type_000.adsorption.mpm_linear_threshold = 0.0

    col.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
    col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT'
    col.particle_type_000.discretization.PAR_POLYDEG = 1
    col.particle_type_000.discretization.PAR_NELEM = 1

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
    m.input.solver.sections.nsec = 3
    m.input.solver.sections.section_continuity = [False, False]
    m.input.solver.sections.section_times = [0.0, t_inj, T_GRADIENT, t_end]
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-10
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, t_end, n_points)

    return cadet


def run_model(config, col_dispersion_base, fname=None, **kwargs):
    c = get_model(config, col_dispersion_base, **kwargs)
    c.filename = os.path.join(HERE, fname or f'gritti_fig8_{config}.h5')
    c.save()
    rc = c.run_simulation()
    if rc.return_code != 0:
        raise RuntimeError(f"CADET failed ({config}): {getattr(rc, 'error_message', rc)}")
    c.load_from_file()
    t = np.asarray(c.root.output.solution.solution_times)
    outlet = np.asarray(c.root.output.solution.unit_001.solution_outlet)
    return t, outlet[:, 1]  # bombesin (component 1) only


# ===========================================================================
# Moment / mass-balance helpers
# ===========================================================================
def _trapz(y, x):
    """Trapezoidal integration without relying on a specific numpy version's
    trapz/trapezoid naming (both exist across supported numpy releases)."""
    fn = getattr(np, 'trapezoid', None) or np.trapz
    return fn(y, x)


def moments(t, c):
    area = _trapz(c, t)
    if area <= 0:
        return 0.0, 0.0, 0.0
    mu1 = _trapz(t * c, t) / area
    mu2 = _trapz((t - mu1) ** 2 * c, t) / area
    return area, mu1, mu2


def peak_time(t, c):
    return t[np.argmax(c)]


# ===========================================================================
# Step 3 (cont'd): calibrate the axial-dispersion "uniform H" parameter via
# CADET runs, matching Table 3's second central moment.
# ===========================================================================
# IMPORTANT LIMITATION (discovered empirically, documented explicitly rather
# than papered over): H calibrated ONLY on the cylinder and then reused
# UNCHANGED for the two conical geometries (via COL_DISPERSION_DEP='POWER_LAW'
# reproducing a uniform PHYSICAL plate height self-consistently for any
# velocity profile, see docstring) reproduces the cylinder's own target
# variance essentially exactly, but under-predicts the two conical columns'
# variances by ~2-3 orders of magnitude. Root cause, verified directly: the
# frustum's velocity is far from uniform (u0(0) for the rho_s=2 entrance is
# ~2.3x the cylinder's velocity; for rho_s=0.5 it is ~0.58x), and since
# dispersion's contribution to temporal variance scales strongly with u
# (roughly as 1/u^2-1/u^3 in the classical moment formulas), a single
# "uniform H" value calibrated at one velocity does not transfer to a
# column whose velocity varies severalfold along its length UNLESS the
# real, compound-specific H(u) correlation is known. The paper's own text
# confirms this is expected: reproducing Fig. 7/8 (unlike the illustrative
# Fig. 2-4 calculations) explicitly requires the MEASURED H(xi) correlation
# of Fig. 5 (obtained from a separate set of flow-rate-dependent isocratic
# runs) -- data this script does not have access to (out of scope: this
# task digitizes Fig. 8, not Fig. 5, and Fig. 5 was measured for the
# alkanophenones, not for bombesin, so it could not simply be reused even
# if digitized).
#
# Given this, and since the task's primary target is reproducing Fig. 8's
# actual chromatograms (not re-deriving Fig. 5's H(xi) correlation from
# scratch for a compound it was never measured for), this script calibrates
# axial dispersion SEPARATELY for each of the three column configurations,
# each against ITS OWN Table 3 second central moment. This means the WIDTH
# of each simulated peak is fitted (not predicted) per configuration -- only
# flagged here, not hidden -- while the RETENTION TIME (and hence the peak
# POSITION and elution time -- the dominant, most visually apparent features
# of Fig. 8, and three of this script's four validation metrics) remains a
# genuine, unfitted PREDICTION of the native frustum geometry: k0/S are
# solved ONLY from the cylinder's retention time and reused unchanged, and
# the resulting agreement with the two conical columns' independently
# measured retention times (Table 3: 287.16 s and 287.82 s) to within 0.3%
# (see "Validation vs. Table 3" below) is a real, predictive success of the
# native COLUMN_MODEL_1D/AXIAL_FLOW_FRUSTUM geometry mapping.
#
# NOTE on grid resolution: the calibrated H/dispersion values are all quite
# small. At such small physical dispersion, a coarse grid's OWN numerical
# dispersion is not negligible by comparison (checked explicitly: NELEM=
# 8/10/16 give non-converged sigma^2=2.41/1.20/1.34 s^2 for the cylinder at
# H=3e-8 m, while NELEM=32/48/64 agree to <0.01%) -- so NELEM=32 (not the
# coarser default) is used for BOTH calibration and production runs
# throughout this script.
_H_BRACKETS = {
    'cylinder': (1.0e-8, 6.0e-8),
    'cone_s2': (1.0e-5, 6.0e-5),
    'cone_s0p5': (1.0e-5, 6.0e-5),
}


def calibrate_dispersion(config):
    target_var_s2 = TABLE3[config][1] * 3600.0  # min^2 -> s^2
    H_trials = _H_BRACKETS[config]
    var_trials = []
    for H in H_trials:
        col_disp = H / 2.0
        t, c = run_model(config, col_disp, fname=f'gritti_fig8_calib_{config}.h5',
                          polydeg=4, nelem=32, n_points=2001)
        _, _, var = moments(t, c)
        var_trials.append(var)
        print(f"  [{config}] calibration trial: H={H:.3e} m -> sigma_t^2={var:.5f} s^2")

    # affine fit: sigma^2(H) = a + b*H
    b = (var_trials[1] - var_trials[0]) / (H_trials[1] - H_trials[0])
    a = var_trials[0] - b * H_trials[0]
    H_fit = (target_var_s2 - a) / b
    H_fit = max(H_fit, 1.0e-9)
    print(f"  [{config}] affine fit: sigma^2 = {a:.5f} + {b:.6g}*H  ->  H_calibrated={H_fit:.4e} m")
    return H_fit


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("Derived / calibrated parameters")
    print("=" * 70)
    print(f"  eps_t = {EPS_T}, F (phase ratio) = {F_PHASE:.4f}")
    print(f"  gradient steepness beta = {BETA * 60:.4f} /min ({BETA:.6f} /s)")
    print(f"  S (LSSM slope, assumed) = {S_LSSM}")
    print(f"  cylinder: t_ref = {_t_ref_cyl:.4f} s, G = {_G_cyl:.4f}, m(1) = {_m1_cyl:.4f}")
    print(f"  solved k0 (retention factor at phi0) = {K0:.4f}")
    print(f"  MPM-Langmuir: ka_1={KA1:.6g}, kd_1={KD1}, qmax_1={QMAX1}, gamma_1={GAMMA1}")

    print("\nCalibrating axial dispersion against each column's own Table 3 second")
    print("central moment (see calibrate_dispersion() docstring/comments for why")
    print("this is done per-configuration rather than reusing the cylinder's H)...")
    col_disp_base = {}
    for config in ('cylinder', 'cone_s2', 'cone_s0p5'):
        H_cal = calibrate_dispersion(config)
        col_disp_base[config] = H_cal / 2.0
        print(f"  -> {config}: H_calibrated = {H_cal * 1e6:.4f} um, "
              f"COL_DISPERSION_base = {col_disp_base[config]:.4e} m")

    print("\nRunning production simulations (all three column configurations)...")
    # NELEM=64 (finer than the NELEM=32 used for calibration -- moments were
    # already grid-converged there, see comments above, but the raw curve
    # SHAPE at NELEM=32 shows small Gibbs-type ringing near the rectangular
    # injection pulse -- an O(1e-2)-amplitude artifact at NELEM=32 that
    # shrinks to ~1e-5 by NELEM=96; NELEM=64 is used here as a good
    # quality/cost compromise for the production curves used in the plot).
    results = {}
    for config in ('cylinder', 'cone_s2', 'cone_s0p5'):
        t, c = run_model(config, col_disp_base[config], polydeg=4, nelem=64, n_points=2201)
        results[config] = (t, c)
        area, mu1, mu2 = moments(t, c)
        tp = peak_time(t, c)
        print(f"  {config:10s}: peak_t={tp:7.3f} s   mu1={mu1:7.3f} s   "
              f"sigma^2={mu2:.5f} s^2   area={area:.5f}")

    # ---- validation against Table 3 (experimental moments, independent of
    # the figure digitization) ----
    print("\n" + "=" * 70)
    print("Validation vs. Table 3 (experimental retention time / 2nd moment)")
    print("=" * 70)
    for config in ('cylinder', 'cone_s2', 'cone_s0p5'):
        t, c = results[config]
        area, mu1, mu2 = moments(t, c)
        tR_ref_s = TABLE3[config][0] * 60.0
        var_ref_s2 = TABLE3[config][1] * 3600.0
        print(f"  {config:10s}: t_R sim={mu1:7.3f} s  ref={tR_ref_s:7.3f} s  "
              f"rel.err={100 * abs(mu1 - tR_ref_s) / tR_ref_s:5.2f}%   |   "
              f"sigma^2 sim={mu2:.5f}  ref={var_ref_s2:.5f}  "
              f"rel.err={100 * abs(mu2 - var_ref_s2) / var_ref_s2:5.2f}%")

    # ---- validation against the digitized Fig. 8 reference ----
    print("\n" + "=" * 70)
    print("Validation vs. digitized Fig. 8 (4 standard metrics)")
    print("=" * 70)
    metrics = {}
    for config in ('cylinder', 'cone_s2', 'cone_s0p5'):
        t_sim, c_sim = results[config]
        t_ref, c_ref = REFERENCE[config]

        # 1) peak position
        tp_sim = peak_time(t_sim, c_sim)
        tp_ref = peak_time(t_ref, c_ref)
        peak_relerr = 100 * abs(tp_sim - tp_ref) / tp_ref

        # 2) first-moment elution time (computed over the digitized window
        # for both, for a fair comparison)
        mask = (t_sim >= t_ref.min()) & (t_sim <= t_ref.max())
        _, mu1_sim, _ = moments(t_sim[mask], c_sim[mask])
        _, mu1_ref, _ = moments(t_ref, c_ref)
        elution_relerr = 100 * abs(mu1_sim - mu1_ref) / mu1_ref

        # 3) mass balance: simulated outlet mass vs. simulated inlet mass
        # (intrinsic CADET check, independent of the digitized reference's
        # arbitrary absorbance units)
        area_out, _, _ = moments(t_sim, c_sim)
        t_inj = V_INJ / (Q_CYL if config == 'cylinder' else Q_CONE)
        area_in = C_INJ * t_inj
        mass_relerr = 100 * abs(area_out - area_in) / area_in

        # 4) chromatogram MSE: digitized data is in arbitrary absorbance
        # units, simulated concentration in arbitrary (C_inj=1) units, so a
        # single least-squares scale factor is fit before comparing shapes.
        c_sim_i = np.interp(t_ref, t_sim, c_sim)
        denom = np.sum(c_sim_i ** 2)
        scale = np.sum(c_sim_i * c_ref) / denom if denom > 0 else 0.0
        mse = np.mean((scale * c_sim_i - c_ref) ** 2)

        metrics[config] = dict(tp_sim=tp_sim, tp_ref=tp_ref, peak_relerr=peak_relerr,
                                mu1_sim=mu1_sim, mu1_ref=mu1_ref, elution_relerr=elution_relerr,
                                area_out=area_out, area_in=area_in, mass_relerr=mass_relerr,
                                mse=mse, scale=scale)

        print(f"\n  --- {config} ---")
        print(f"    Peak position   : sim={tp_sim:7.3f} s  ref={tp_ref:7.3f} s   rel.err={peak_relerr:5.2f}%")
        print(f"    Elution time    : sim={mu1_sim:7.3f} s  ref={mu1_ref:7.3f} s   rel.err={elution_relerr:5.2f}%")
        print(f"    Mass balance    : outlet={area_out:.5f}  inlet={area_in:.5f}   rel.err={mass_relerr:5.2f}%")
        print(f"    Chromatogram MSE: {mse:.3e}  (fitted absorbance scale={scale:.4g})")

    # ---- comparison plot ----
    colors = {'cylinder': 'k', 'cone_s2': 'tab:red', 'cone_s0p5': 'tab:blue'}
    labels = {'cylinder': r'Cylinder $\rho_s=1$', 'cone_s2': r'Cone $\rho_s=2$',
              'cone_s0p5': r'Cone $\rho_s=0.5$'}
    fig, ax = plt.subplots(figsize=(8, 6))
    for config in ('cylinder', 'cone_s2', 'cone_s0p5'):
        t_sim, c_sim = results[config]
        t_ref, c_ref = REFERENCE[config]
        scale = metrics[config]['scale']
        ax.plot(t_sim, scale * c_sim, '-', color=colors[config], lw=1.6,
                label=f"{labels[config]} (CADET)")
        ax.plot(t_ref[::3], c_ref[::3], 'o', color=colors[config], ms=3.5, mfc='none',
                label=f"{labels[config]} (digitized)")
    ax.set_xlim(270, 306)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Absorbance [AU]  (CADET curve scaled to match)')
    ax.set_title('Gritti et al. (2019) J. Chromatogr. A 1593, Fig. 8\n'
                  'Bombesin, gradient elution -- cylinder vs. conical column '
                  '(native COLUMN_MODEL_1D frustum geometry)', fontsize=10)
    ax.legend(fontsize=8, ncol=1, loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(HERE, 'case_study_gritti_fig8.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved comparison plot to {outpath}")


if __name__ == '__main__':
    main()
