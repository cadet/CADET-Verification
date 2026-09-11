Reproduction of Fig. 7 from:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

Self-contained script: model definition, run, comparison plot, and
validation metrics.

===========================================================================
Case identification
===========================================================================
Fig. 7 (p. 43), "Valerophenone (gradient)": experimental gradient-elution
chromatograms of n-valerophenone on three physical column configurations,
all L=15 cm, packed with the same batch of 5 um XBridge-C18 fully-porous
particles:

  1) Cylinder (rho_s=1): conventional column, i.d.=3.0 mm (re=1.5 mm),
     0.35 mL/min.
  2) Cone rho_s=2:  truncated-cone (frustum) column, i.d. 2.1->4.2 mm, flow
     entering the NARROW (2.1 mm) end, 0.40 mL/min.
  3) Cone rho_s=0.5: the same physical frustum tube as (2), flow reversed --
     entering the WIDE (4.2 mm) end.

Gradient: linear ACN/water gradient, phi: 0.60->0.95 over a gradient time of
5 min (beta=0.07/min, matching Fig. 2's caption), Sec. 3.4.2. Injection
volume 0.5 uL. Isocratic reference retention factor k(phi=0.75)=1.08 (text,
p. 44) and isocratic plate height H=9.5 um (cylindrical column, 0.35 mL/min;
text, p. 43) are stationary-phase/particle-batch properties, so both apply
to all three column configurations. Table 2 gives the experimental gradient
first moment (retention time): tR = 4.656 min (cylinder), 4.659 min (cone
rho_s=2), 4.674 min (cone rho_s=0.5), together with second central moments
(used below for dispersion calibration).

Governing equations (paper-provided): Giddings' plate-height/band-broadening
framework for isocratic elution (Sec. 2.3), and the Blumberg/Poppe spatial-
variance ODE for gradient elution in non-uniform (conical) columns (Sec.
2.4, Eqs. (28)-(47)), built on the Linear Solvent Strength Model (LSSM)
retention law k(phi) = k0*exp(-S*(phi-phi0)) (Eq. 28). The paper does not
tabulate the LSSM parameters (k0, S) for valerophenone directly; they are
DERIVED below from the paper's own equations and tabulated numbers (Table 1
isocratic k, Table 2 gradient retention time) -- fitting is reserved for the
AU-scale amplitude calibration and the per-column dispersion-variance
calibration described further below.

===========================================================================
Model selection and justification
===========================================================================
Bulk transport: CADET's native axial-flow geometries under COLUMN_MODEL_1D
(GEOMETRY='AXIAL_FLOW_CYLINDER' for the cylindrical column,
GEOMETRY='AXIAL_FLOW_FRUSTUM' for the conical column in both flow directions
-- CROSS_SECTION_AREA_SMALL_END/LARGE_END are identical for both cone
directions since it is the same physical tube; only FORWARD_FLOW differs).
This discretizes the real, physically varying cross-section/velocity along z
directly, with no cylindrical-column approximation for the conical geometry.
Frustum FORWARD_FLOW convention (verified with a dedicated non-adsorbing-
tracer control run: identical mean transit time in both directions, 1.972/
1.976 min vs. the mass-conservation-implied eps_t*V_col/Fv=1.966 min, to
<0.5%): FORWARD_FLOW=0 is flow from the small end (z=0) to the large end
(z=L); FORWARD_FLOW=1 is the reverse.

Dispersion model: the paper's theoretical treatment has two parts (Sec. 1,
Introduction). An illustrative part (Sec. 2.3/4.1.4, Figs. 2/3) assumes a
spatially uniform plate height H to explore the generic theoretical
potential of conical columns; the actual experimental comparison (Sec.
4.2.1, the target of this script) instead explicitly accounts for the
"change in plate height along the conical column (Fig. 5)". This script
therefore uses the real, flow-rate-dependent H(v) measured in Fig. 5 -- the
same digitized curve and VAN_DEEMTER fit (VD_A, VD_B, VD_C) as
Gritti2019_fig6.py -- evaluated at the true local interstitial velocity via
COL_DISPERSION_DEP='VAN_DEEMTER' (H(v)=VD_A+VD_B/v+VD_C*v), giving
Dax(z)=H(v(z))*v(z)/2 self-consistently along the frustum's varying velocity
field rather than a single column-averaged or uniform value. This maps to
CADET's Lumped Rate Model without pores within the unified interface
(NPARTYPE=1, particle_type_000.HAS_FILM_DIFFUSION=0, TOTAL_POROSITY
replacing COL_POROSITY per axial_flow_column_1D_config.rst), consistent with
the paper's own model, which reports no separate particle-scale transport
resistances.

Dispersion calibration (per column): the paper's Table 1 reports both a
tailing-blind half-height efficiency N_1/2 (Eq. 68) and a moment-based
efficiency N_moments (Eq. 69, sensitive to tailing) for valerophenone on all
three configurations:

    column        N_1/2   N_moments  N_1/2 / N_moments
    cylinder      16090    9596       1.68  (substantially tailed)
    cone rho=2    13181   13342       0.988 (essentially Gaussian)
    cone rho=0.5  13563   13635       0.995 (essentially Gaussian)

I.e. the cylindrical column's real peak is genuinely tailed -- a column-
specific packing/wall effect the paper attributes explicitly to the
cylinder itself (p. 43: peaks on the cylinder "systematically tail more
than those observed for the conical column, irrespective of flow
direction", ruling out a shared instrument/extra-column effect) -- while the
conical column's real peak is essentially Gaussian in both flow directions.
Since CADET's axial-dispersion model is symmetric, it can only reproduce a
column's true (moment-based) variance, not a tailed peak shape; calibrating
H against the half-height-derived "H=9.5 um" alone would therefore
reproduce a peak that is too narrow specifically for the tailed cylinder.
Each column's dispersion is therefore calibrated with a per-column,
dimensionless scale factor on the real, measured Fig. 5 H(v) curve
(COL_DISPERSION[1], COL_DISPERSION_DEP='VAN_DEEMTER'), chosen (via
calibrate_dispersion()) so the full gradient-elution PDE reproduces that
column's own measured second central moment (Table 2). This is a closed-
form rescaling rather than a free fit: variance scales linearly with the
dispersion scale factor (verified numerically -- doubling/tripling it
reproduces the same factor in simulated variance to <0.1%), and the
expected outcome is a scale factor near 1.0 for the (Gaussian) cones -- the
unscaled Fig. 5 curve already matches, as the paper's own Sec. 4.2.1
comparison implies -- and substantially above 1.0 for the (tailed)
cylinder, since no symmetric-dispersion model can capture tailing; only the
total (variance-matched) spread can be reproduced. The retention-time/LSSM
parameters (below), derived solely from the cylinder's isocratic k and
Table 2's cylinder retention time, are unaffected by this calibration and
remain a genuine, unfitted prediction for both conical configurations.

Binding law: CADET's MOBILE_PHASE_MODULATOR_LANGMUIR model
(ADSORPTION_MODEL='MOBILE_PHASE_MODULATOR') implements, per component i and
modulator ("salt") component 0,
    dq_i/dt = ka_i*exp(gamma_i*cp_0)*cp_i*qmax_i*(1-sum_j q_j/qmax_j)
              - kd_i*cp_0^beta_i*q_i .
Setting beta_i=0 (no power-law/ion-exchange term; not applicable to an
organic-modifier RPLC gradient) and taking qmax_i -> large (dilute, linear
limit) gives, at quasi-equilibrium,
    K_i(phi) := q_i/cp_i = (ka_i*qmax_i/kd_i)*exp(gamma_i*phi) ,
an exact exponential-in-phi law: gamma_i=-S_i reproduces the LSSM law (28)
exactly, without any polynomial/EXTFUN approximation of the exponential (as
would be needed with the generic EXT_LINEAR route). The modulator component
0 represents the local ACN volume fraction; since HAS_FILM_DIFFUSION=0, cp_0
entering the isotherm is identically the local bulk concentration, so the
gradient is genuinely transported (with its own near-negligible axial
dispersion, so its profile stays essentially undistorted, matching the
paper's own assumption in Sec. 2.4 that "the solvent gradient is linear and
not distorted upon migration") through the actual frustum velocity field --
more physically direct than CADET's generic EXTFUN/EXT_LINEAR mechanism,
which would require a separately-configured propagation velocity, awkward
for a geometry whose velocity is itself axially varying. Modulator: NBOUND=0
(non-binding, per the model's documented salt convention). Valerophenone:
NBOUND=1, qmax set to a large placeholder (1e4) so the Langmuir competition
term (1-q/qmax) stays within ~1e-4 of 1 throughout, i.e. genuinely linear/
dilute adsorption, matching the trace-level small-molecule mixture used
experimentally.

===========================================================================
Reparameterization
===========================================================================
Physical parameters (SI units): L=0.15 m (both column types); EPS_T=0.65
(total porosity, the paper's own value, Sec. 4.1.4, used to compute
u0(0)=17.77 cm/min for Fv=0.40 mL/min, re=1.05 mm); cylinder re=1.5 mm,
Fv=0.35 mL/min, V_col=1.06 cm^3; frustum small-end r=1.05 mm (2.1 mm i.d.),
large-end r=2.10 mm (4.2 mm i.d.), Fv=0.40 mL/min, V_col=1.21 cm^3 ("cone
rho_s=2": FORWARD_FLOW=0, small->large; "cone rho_s=0.5": FORWARD_FLOW=1,
large->small); gradient phi0=0.60->phi_final=0.95, tg=5 min (beta=
(phi_final-phi0)/tg=0.07/min, matching Fig. 2's caption); k(phi=0.75)=1.08
and H=9.5 um as above (applied to all three columns via COL_DISPERSION_DEP,
subject to the per-column calibration above).

LSSM parameters (k0, S) for valerophenone are derived (not fitted) from two
of the paper's own equations evaluated at the cylindrical column only
(rho_s=1):
  (i)  Eq. (28) at phi=0.75: k0 = k(0.75)*exp(S*(0.75-0.60))
  (ii) Eq. (34) at rho_s=1: tau_e(1) = 1 + (1/G)*ln(1+G*k0),
       G = S*beta*tau0, tau0 = L/u0(0) = eps_t*V_col/Fv (cylinder's own
       hold-up time, Eq. 18 at rho_s=1), tau_e(1) = tR_grad/tau0 with
       tR_grad = 4.656 min (Table 2).
Solving (i)+(ii) simultaneously (derive_lssm_parameters()) gives S=4.7756
(volume fraction)^-1, k0=2.2107 -- both within the typical literature range
for a small aromatic ketone in RPLC gradient elution (S~3-10). As an
independent (not fitted) check, evaluating Eq. (34) for both conical
orientations (rho_s=2, 0.5; each using its own u0(0)=Fv/(eps_t*pi*re^2) at
its own inlet radius) reproduces the measured conical retention times to
within 0.03%/0.35% -- the paper's own analytical theory, with zero
additional free parameters, already predicts the conical retention times
essentially exactly. The full CADET PDE simulation is a strictly harder,
independent test: it must reproduce the full digitized peak SHAPES (not
just these retention times) using the same H-derived, self-consistent axial
dispersion field.

===========================================================================
Reference (digitized) data
===========================================================================
Fig. 7 (three overlaid experimental chromatograms sharing one time axis
270-295 s and one absorbance axis 0-0.20+ AU) was digitized by pixel-colour
thresholding (axis tick marks located from the rendered page image; each
curve's colour -- black/red/blue -- isolated via RGB thresholds, with the
title box and in-plot legend masked out first). See
Gritti2019_fig7_digitized.csv (digitized points) and
Gritti2019_fig7_digitized_preview.png (overlay used to confirm extraction
quality; the red "cone rho_s=2" curve has fewer recovered points, 436 vs
~845, purely from partial occlusion by the other two traces where curves
cross, not extraction error). Digitized peak heights (0.201/0.180/0.189 AU)
match the paper's plotted values essentially exactly.

===========================================================================
Simulation, resolution, and validation
===========================================================================
get_model()/run_column() build and run the model for a given column; the
__main__ block runs all three configurations, computes validation metrics
(peak position, first-moment elution time, mass balance, chromatogram MSE)
against the digitized curves, and produces the comparison plot.

Numerical resolution: DG with NELEM=128 (POLYDEG=4) is used throughout.
Grid-convergence checks (NELEM=16..128, cross-validated against FV at
NCOL=100..1600) show the elution time is already converged by NELEM=64
(4.6561 min vs. 4.6562 min at NELEM=128, matching Table 2's 4.656 min to
<0.01%), but the mass-balance error for cone_rho_s=2 -- whose inlet is the
frustum's small (fastest) end, where the short (~0.075 s) injection pulse is
hardest to resolve -- only drops below the SOP's 1% tolerance at NELEM=128
(17.5% at NELEM=32, 2.4% at NELEM=64, 0.0015% at NELEM=128); NELEM=128 is
therefore used uniformly for all three columns. A DG-vs-FV cross-check at
this resolution shows a small (~2.5% of peak height), resolution-sensitive
pre-peak ripple specific to the DG solution for cone_rho_s=2 (absent in FV
at NCOL=800); it does not measurably affect any reported validation metric.
(These grid-convergence figures were obtained at the COL_DISP_PROBE
dispersion value -- a purely numerical/discretization check, independent of
the per-column dispersion calibration above, which simply rescales the
COL_DISPERSION input on the same, already-converged discretization.)

AU-scale amplitude: CADET's simulated valerophenone concentration is in
arbitrary units (C0=1), unrelated to the digitized reference's real,
uncalibrated detector absorbance -- the paper gives no molar-absorptivity/
detector calibration, so a per-column least-squares scale factor (scale =
argmin_s |s*c_sim-c_ref|^2) is fit independently for each column and used
only for plotting and peak-height/MSE reporting in the paper's native
Absorbance [AU] units; it does not enter peak position, elution time, or
mass balance, so this is a display convention, not a modeling choice.
Fitting the scale independently per column (rather than one shared/averaged
scale) is deliberate: the three columns' individually implied scales differ
by ~7%, specifically because the tailed cylinder's Gaussian-equivalent
model variance-matches a shorter apparent peak height than its real one
(see "Dispersion calibration" above); a shared scale would let this
cylinder-specific effect leak into the (mutually consistent to <1%) conical
columns' apparent fit quality.