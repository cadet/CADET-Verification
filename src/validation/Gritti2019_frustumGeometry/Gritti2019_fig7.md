# Gritti et al. (2019), Fig. 7 — gradient elution of valerophenone

Reference:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

`Gritti2019_fig7.py` holds the model definition, the simulation run, the
comparison plot and the validation metrics.

## The case

Fig. 7 (p. 43) shows measured gradient-elution chromatograms of
n-valerophenone on the same three configurations as Fig. 6, all 15 cm long and
packed with the same batch of 5 µm XBridge-C18 particles:

1. **Cylinder, rho_s = 1**: 3.0 mm i.d. (r_e = 1.5 mm), 0.35 mL/min.
2. **Cone, rho_s = 2**: frustum from 2.1 to 4.2 mm i.d., flow entering the
   narrow end, 0.40 mL/min.
3. **Cone, rho_s = 0.5**: the same tube with the flow reversed, entering the
   wide end.

The gradient is linear in the acetonitrile fraction, phi = 0.60 to 0.95 over
5 min, i.e. beta = 0.07/min (Sec. 3.4.2, matching the caption of Fig. 2). The
injection volume is 0.5 µL. The isocratic retention factor k(phi = 0.75) = 1.08
(p. 44) and the isocratic plate height H = 9.5 µm (p. 43) are properties of the
particle batch and stationary phase, so they apply to all three columns.

Table 2 gives the measured gradient retention times — 4.656 min for the
cylinder, 4.659 min for rho_s = 2 and 4.674 min for rho_s = 0.5 — together with
the second central moments used as the reference for Delta mu_2.

The paper's theory is Giddings' plate-height framework for isocratic elution
(Sec. 2.3) plus the Blumberg/Poppe spatial-variance ODE for gradient elution in
non-uniform columns (Sec. 2.4, Eqs. 28–47), built on the linear solvent
strength model (LSSM) k(phi) = k0*exp(-S*(phi-phi0)) of Eq. 28. The LSSM
parameters k0 and S are not tabulated for valerophenone; they are derived below
from the paper's own equations and numbers.

## Model choice

**Geometry.** `COLUMN_MODEL_1D` with `GEOMETRY='AXIAL_FLOW_CYLINDER'` for the
cylindrical column and `GEOMETRY='AXIAL_FLOW_FRUSTUM'` for both cone
directions. Since both cones are the same physical tube, they share
`CROSS_SECTION_AREA_SMALL_END` and `_LARGE_END` and differ only in
`FORWARD_FLOW`: 0 is flow from the small end at z = 0 to the large end at z = L,
1 is the reverse. The varying cross-section and velocity are resolved directly,
with no cylindrical approximation of the cone. A non-adsorbing tracer run
confirms the convention: the mean transit time is the same in both directions
(1.972 and 1.976 min) and agrees with eps_t*V_col/Fv = 1.966 min to under 0.5 %.

**Dispersion.** The paper has two separate treatments. The illustrative one
(Secs. 2.3/4.1.4) assumes a uniform plate height to explore what conical columns
can do in principle; the experimental comparison of Sec. 4.2.1, which is what
this script reproduces, accounts for the change of plate height along the column
shown in Fig. 5. This script therefore uses the measured H(v) — the same
digitized curve and van Deemter coefficients VD_A, VD_B, VD_C as
`Gritti2019_fig6.py` — through `COL_DISPERSION_DEP='VAN_DEEMTER'`, which gives
D_ax(z) = H(v(z))*v(z)/2 along the frustum's own velocity field rather than one
column-averaged value. Gritti2019_fig6.md derives that relation from the paper's
plate-height definition and describes the fit.

The `COL_DISPERSION` scale factor stays at 1.0, so the plate height enters as
measured and nothing is fitted. `Gritti2019_fig8.py` cannot do this, since the
paper reports no plate height for bombesin.

H(v) was measured under isocratic conditions and is reused unchanged here for
gradient elution, which is what the paper does in its own predictions (p. 45).

Since the paper resolves no separate particle-scale transport resistances, the
particle side is the lumped rate model without pores: `NPARTYPE=1`,
`HAS_FILM_DIFFUSION=0` and `TOTAL_POROSITY` in place of `COL_POROSITY`.

One consequence is worth knowing when reading the results. Table 1 reports both
a half-height efficiency N_1/2 and a moment-based efficiency N_moments for each
column. They differ by a factor 1.68 on the cylinder but agree to within 1 % on
both cones, i.e. the real cylindrical peak is tailed while the conical ones are
essentially Gaussian. The paper attributes the tailing to the cylindrical
column itself (p. 43), not to the instrument. An axial dispersion model is
symmetric and can match a column's variance but not a tailed shape, so the
cylinder deviates more than the two cones.

**Retention.** The gradient is modelled with
`ADSORPTION_MODEL='MOBILE_PHASE_MODULATOR'`, whose isotherm is

    dq_i/dt = ka_i*exp(gamma_i*cp_0)*cp_i*qmax_i*(1-sum_j q_j/qmax_j)
              - kd_i*cp_0^beta_i*q_i

Component 0 is the modulator, here the local acetonitrile volume fraction.
Setting beta_i = 0 removes the power-law term, which belongs to ion exchange
rather than an organic-modifier gradient, and taking qmax_i large puts the
isotherm in its linear, dilute limit. At quasi-equilibrium this leaves

    K_i(phi) = q_i/cp_i = (ka_i*qmax_i/kd_i)*exp(gamma_i*phi)

so gamma_i = -S_i reproduces the LSSM law of Eq. 28 exactly, with no polynomial
approximation of the exponential. Because `HAS_FILM_DIFFUSION=0`, the cp_0 that
enters the isotherm is the local bulk concentration, so the gradient is
transported through the actual frustum velocity field. Its own axial dispersion
is negligible, so the profile stays essentially undistorted, which is what the
paper assumes in Sec. 2.4. The modulator has `NBOUND=0`; valerophenone has
`NBOUND=1` and qmax = 1e4, large enough that the competition term stays within
1e-4 of 1.

## Parameters

Physical parameters in SI units: L = 0.15 m for all columns; total porosity
eps_t = 0.65 (the paper's own value, Sec. 4.1.4); cylinder r_e = 1.5 mm,
Fv = 0.35 mL/min, V_col = 1.06 cm^3; frustum small end r = 1.05 mm, large end
r = 2.10 mm, Fv = 0.40 mL/min, V_col = 1.21 cm^3. rho_s = 2 is
`FORWARD_FLOW=0` (small to large), rho_s = 0.5 is `FORWARD_FLOW=1`.

The LSSM parameters are derived, not fitted, from two of the paper's equations
evaluated on the cylindrical column alone:

    (i)  Eq. 28 at phi = 0.75:   k0 = k(0.75)*exp(S*(0.75-0.60))
    (ii) Eq. 34 at rho_s = 1:    tau_e(1) = 1 + (1/G)*ln(1+G*k0)
                                 G = S*beta*tau0
                                 tau0 = L/u0(0) = eps_t*V_col/Fv
                                 tau_e(1) = tR_grad/tau0, tR_grad = 4.656 min

`derive_lssm_parameters()` solves the pair simultaneously and gives
S = 4.7756 (volume fraction)^-1 and k0 = 2.2107, both in the usual range for a
small aromatic ketone in reversed-phase gradient elution. As a check that adds
no free parameters, evaluating Eq. 34 for the two conical orientations — each
with its own u0(0) = Fv/(eps_t*pi*r_e^2) at its own inlet radius — reproduces
the measured conical retention times to within 0.03 % and 0.35 %. The CADET
simulation is the harder test, since it has to reproduce the full peak shapes
with the same H-derived dispersion field.

## Reference data

Fig. 7 shows the three chromatograms overlaid on one time axis (270–295 s) and
one absorbance axis. They were digitized by pixel colour thresholding: the axis
tick marks locate the frame, each curve is isolated by its RGB range
(black/red/blue), and the title box and in-plot legend are masked out first.
The result is in `Gritti2019_fig7_digitized.csv`. The red rho_s = 2 curve
recovers fewer points than the others (436 against roughly 845), because the
other two traces occlude it where the curves cross. The digitized peak heights
(0.201/0.180/0.189 AU) match the plotted values.

## Numerical resolution

All three columns are run with DG at `POLYDEG=4` and `NELEM=128`. The elution
time is already converged at `NELEM=64`, but the mass balance for rho_s = 2 is
not: its inlet is the frustum's small, fastest end, where the 0.075 s injection
pulse is hardest to resolve, and the mass-balance error only falls below 1 % at
`NELEM=128`. The same resolution is then used for all three columns. At that
resolution the DG solution for rho_s = 2 shows a small pre-peak ripple of about
2.5 % of the peak height, which is absent in FV and does not measurably affect
any reported metric.

## The absorbance scale factor

CADET's simulated concentration is in arbitrary units (C0 = 1) and the
digitized reference is an uncalibrated detector absorbance; the paper gives no
detector calibration. A least-squares scale factor is therefore fitted per
column and used only for plotting and for the peak-height and NRMSE numbers, so
that they can be reported in the paper's own absorbance units. It does not
enter the peak position, the elution time or the mass balance.

The scale is fitted per column rather than shared because the three implied
scales differ by about 7 %: a symmetric model matched to the tailed cylinder's
variance comes out shorter than its real peak. A shared scale would carry that
cylinder-specific effect into the two conical columns, whose own scales agree
to within 1 %.
