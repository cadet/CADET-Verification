Reproduction of Fig. 8 from:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

Self-contained script: model definition, run, comparison plot, and
validation metrics.

===========================================================================
Case identification
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
(Eq. 1) -- a genuine frustum (linearly-varying cross-section), not a
radial-flow column. The text introduces Fig. 8 as showing "the experimental
peaks of the peptide bombesin recorded on ... all three columns under
gradient conditions", so all three are reproduced together with a single
model (same analyte/chemistry/particles; only the transport geometry and
flow direction differ between the three CADET runs).

Bombesin is a real, retained analyte separated by a linear ACN/water
gradient. Its retention follows the paper's gradient-elution theory
(Section 2.4, Linear Solvent Strength Model / LSSM, Eq. 28):
k(phi) = k0*exp(-S*(phi-phi0)), where phi is the ACN volume fraction. This
requires a second, non-retained "modifier" transport field (phi itself)
whose local value modulates the analyte's retention factor as it migrates
-- i.e. a genuine 2-component transport problem (modifier + analyte), even
though only the analyte is the validation target.

Reference data: the digitized chromatogram (Gritti2019_fig8_digitized.csv,
see "Reference data" below) and Table 3's summary retention time / first
and second moments / half-height width for all three configurations.

===========================================================================
Model mapping to CADET
===========================================================================
Bulk/particle transport
------------------------
The paper's own model (Sections 2.2-2.4) is a "black-box column" treatment:
axial convection with a smoothly axially-varying velocity/cross-section
(Eq. 4-5), and band broadening described purely through an aggregate,
possibly axially-varying, plate height H(xi) (Giddings'/Blumberg's theory of
non-uniform columns) -- there is no explicit film- or pore-diffusion
sub-model in the paper (unlike a GRM-type case study). CADET's matching
sub-model is EQUILIBRIUM_PARTICLE (HAS_FILM_DIFFUSION=0 on a
COLUMN_MODEL_1D particle_type, i.e. the model formerly named
LUMPED_RATE_MODEL_WITHOUT_PORES): a single interstitial+intraparticle
TOTAL_POROSITY, axial dispersion only, instantaneous local equilibrium with
the stationary phase -- an unambiguous choice given that no separate
mass-transfer parameters are reported for any analyte in this paper.

Column geometry: CADET's native `COLUMN_MODEL_1D` unit with
`GEOMETRY='AXIAL_FLOW_FRUSTUM'` for the two conical runs and
`GEOMETRY='AXIAL_FLOW_CYLINDER'` for the reference cylindrical column.
`FORWARD_FLOW` selects which physical end (large or small) is the inlet, so
the SAME frustum geometry (CROSS_SECTION_AREA_SMALL_END/LARGE_END,
BED_LENGTH) is reused for both rho_s=2 (FORWARD_FLOW=0, enter at the
small/2.1 mm end) and rho_s=0.5 (FORWARD_FLOW=1, enter at the large/4.2 mm
end) runs, mirroring that these are the same physical hardware with flow
simply reversed, per Table 3's caption ("conical column (s=0.5) after
reversing the flow direction").

Binding / gradient-modifier coupling
-------------------------------------
The paper's LSSM retention law k(phi) = k0*exp(-S*(phi-phi0)) (Eq. 28) is
the reversed-phase-gradient isotherm underlying CADET's
MOBILE_PHASE_MODULATOR ("Mobile Phase Modulator Langmuir", Melander &
Horvath 1977; Karlsson 2004) binding model:

    dq_1/dt = k_a exp(gamma*c_p0) c_p1 qmax (1 - q_1/qmax) - k_d c_p0^beta q_1

Component 0 (the modifier, CADET's "salt" role, here the ACN volume
fraction phi) is inert (NBOUND=0, pure convection); component 1 is bombesin
(NBOUND=1). With is_kinetic=0 (instantaneous local equilibrium, consistent
with the paper's retention-factor-only description) and beta=0 (no
power-law/ion-exchange term in the LSSM), the quasi-stationary flux balance
reduces in the dilute limit (q_1 << qmax, valid for the 3 uL injection of
0.1 g/L bombesin used here) to

    q_1/c_p1 = (k_a*qmax/k_d) * exp(gamma*c_p0)

i.e. exactly the LSSM law with gamma = -S and (k_a*qmax/k_d) chosen so the
retention factor k'(phi) = F*(q_1/c_p1) (F = phase ratio = (1-eps_t)/eps_t)
equals k0*exp(-S*(phi-phi0)) -- see "Parameter derivation" below for how k0
is obtained. This is the standard way of representing an LSSM gradient in
CADET (the same mechanism used for salt-gradient IEX, here applied to its
originally-intended HIC/RPLC hydrophobicity role per Melander1989/
Karlsson2004, as cited in CADET-Core's own binding-model documentation).

Axial dispersion / local plate height
------------------------------------------
The paper measures H(xi) directly only for the alkanophenones (Section
4.2.1, Fig. 5), not for bombesin, but explicitly reuses that same
alkanophenone-derived curve for its own bombesin prediction (p. 45,
discussing this exact Fig. 8/Table 3 comparison): "If the theoretical
predictions account for the actual flow rates ..., column dimensions ...,
and change in plate height along the conical column (Fig. 5), it actually
predicts a reduction of peak capacity ... of 18.2% for s=2.0 and even an
increase of 2.9% for s=0.5." This script follows the same approach, reusing
the digitized Fig. 5 curve and VAN_DEEMTER fit (VD_A, VD_B, VD_C) shared
with Gritti2019_fig6.py/fig7.py: D_ax(xi) = scale * H(v(xi))*v(xi)/2 via
COL_DISPERSION_DEP='VAN_DEEMTER', H(v) = VD_A + VD_B/v + VD_C*v. The one
remaining free parameter is a single, dimensionless SCALE FACTOR (see
"Calibration" below), calibrated once against the cylindrical column's
measured second central moment (Table 3) and reused unchanged for both
conical geometries -- mirroring the paper's own single-calibration-then-
reuse structure and testing directly whether the real, compound-independent
H(v) curve transfers correctly across the several-fold velocity swing along
the frustum. CADET's own numerical PDE solution is used as the physics
engine, rather than the paper's perturbative Blumberg/Poppe band-broadening
approximation (Eq. 44-47).

The modifier component's own dispersion is fixed at a small, non-zero,
geometry-independent placeholder value (negligible compared to convection)
so that its ramp propagates essentially undistorted, matching the paper's
explicit assumption (Section 2.4, Eq. 29-30) that the solvent gradient is
not distorted upon migration along the column.

===========================================================================
Parameter derivation
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

Two quantities are not restated for this specific case study and are
therefore documented explicitly (flagged with ** below) as values carried
over from elsewhere in the paper, rather than independent assumptions:

** eps_t (total porosity) = 0.65: taken from Section 4.1.4's worked example
   for the same batch of particles/columns, and verified at runtime to
   reproduce that section's own stated entrance velocity u0(0)=17.77 cm/min
   for Fv=0.40 mL/min, r_e=1.05 mm.

** S (LSSM slope) = 25: not given numerically for bombesin; the paper
   states S=25 for its illustrative "17-peptide" mixture (Fig. 2/4 caption)
   under a gradient steepness of 0.09/min, identical to the actual gradient
   steepness used for the bombesin experiments (Section 3.4.3). Both are
   "peptide" cases under matching gradient conditions, so S=25 is adopted
   for bombesin as the best available value.

k0 (the LSSM pre-exponential retention factor at phi0) is solved for in
closed form from the cylinder column's own measured retention time (Table
3, 4.798 min), using the paper's exact (non-perturbative) gradient-elution
time relation (Eq. 31, integrating the LSSM local-equilibrium retention law
along the column):

    e(xi) = m(xi) + (1/G) * ln(1 + G*k0*m(xi))                 [Eq. 31]
    G      = S*beta*L/u0(0)                                    [Eq. 32]
    m(xi)  = (1 + rho(xi) + rho(xi)^2) / 3   at xi=1: m(1)=(1+s+s^2)/3
             (closed form of Eq. 15/16's integral -- confirmed by direct
             integration of rho(xi)^2 = [1+(s-1)xi]^2 from 0 to xi, and by
             consistency with Eq. 18's stated s->1/s flow-direction
             invariance)
    => k0 = [exp(G*(e(1)-m(1))) - 1] / (G*m(1))

evaluated with the cylinder's own G, m(1)=1, e(1)=t_R/t_ref. k0 is an
intrinsic analyte/stationary-phase property, independent of column
geometry, and is therefore reused unchanged for both conical geometries;
their retention times and peak widths are then genuinely predicted (not
fitted) by running the frustum model, providing a direct test of the native
geometry against Table 3's rho_s=2/0.5 rows.

===========================================================================
Reference data (digitized from the figure)
===========================================================================
Fig. 8 (p. 45) plots Absorbance [AU] vs. Time [s] for the three curves;
Table 3 gives only summary moments, not the traces, so the figure was
digitized directly via pixel-color classification:

  1. The page was rendered at 600 dpi and cropped to the plot's axes box;
     the pixel<->data-value calibration used the tick marks' own pixel
     positions (x: 280/300 s ticks; y: 0.000-0.020 AU ticks in 0.005 steps).
  2. Curve colors were verified by direct pixel sampling (black/red/blue,
     RGB roughly (0,0,0)/(200,20,20)/(35,5,250)), after masking the
     title-box border and legend swatch/text.
  3. The three curves visually overlap over large stretches (near baseline
     and along much of the rising/falling flanks, since this compares a
     cylinder against the same conical column run in both flow directions).
     Where curves coincide, only the last-drawn color is visible at that
     pixel column; overlapping curves are therefore assigned the same value
     at that column rather than being independently interpolated through
     the gap. The resulting traces were re-plotted on top of the source
     image crop and checked against every visually distinct
     crossing/occlusion region (see Gritti2019_fig8_digitized_preview.png).
  4. The resulting per-curve traces (recovered peak heights: cylinder
     0.0193, cone_rho_s=2 0.0166, cone_rho_s=0.5 0.0179 AU, matching the
     source figure) are resampled to a uniform 0.2 s grid and saved as
     Gritti2019_fig8_digitized.csv (columns: time_s, cylinder_AU,
     cone_s2_AU, cone_s05_AU), loaded at runtime by `load_digitized()`.