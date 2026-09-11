Reproduction of Fig. 6 from:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

Self-contained script: model definition, run, comparison plot, and
validation metrics.

===========================================================================
Step 0 -- Case identification
===========================================================================
Fig. 6 (p. 43, Sec. 4.2.2 "Isocratic elution") is the experimental
isocratic elution peak profile of n-valerophenone on three column
configurations, all packed with the same batch of 5-micron XBridge-C18
particles, mobile phase acetonitrile/water 75/25 (v/v), 27 C:

  1) "Cylinder rho_s=1"  : conventional cylindrical column, r_e=1.50 mm
                            (3.0 mm i.d.) x 150 mm, Fv=0.35 mL/min.
  2) "Cone rho_s=2"      : conical column, r_e=1.05 mm entrance (2.1 mm
                            i.d.) widening to 4.2 mm i.d. exit, x 150 mm,
                            Fv=0.40 mL/min (flow: narrow -> wide end).
  3) "Cone rho_s=0.5"    : the SAME physical conical tube, reversed flow
                            direction (entrance r_e=2.1 mm i.d., exit
                            1.05 mm i.d., i.e. wide -> narrow end), same
                            Fv=0.40 mL/min.

Table 1 (p. 44) tabulates, for valerophenone specifically, the measured
retention time, zeroth/first/second moments, half-height width and both
efficiency estimates for these three configurations -- an unambiguous,
purely numerical validation target (first and second central moments) in
addition to the digitized Fig. 6 curve itself:

    Config      Fv[mL/min]  t_R[min]  mu1[min]  mu2'[min^2]  w1/2[min]
    Cylinder    0.35        3.865     3.869      0.00156      0.0718
    Cone s=2    0.40        3.898     3.900      0.00114      0.0800
    Cone s=0.5  0.40        3.915     3.917      0.00113      0.0792

===========================================================================
Step 1 -- Governing model and CADET model choice
===========================================================================
The paper's own theoretical treatment (Sec. 2, "Theory") is Giddings'
classical band-broadening model, expressed purely through (i) a total
column porosity epsilon_t, (ii) a retention factor k, and (iii) a *local
plate height* H(xi) that lumps all non-idealities (eddy dispersion,
longitudinal diffusion, film and pore mass-transfer resistance) into one
coefficient integrated along the column (Eqs. 11-26). No particle
porosity, film mass-transfer coefficient, or pore diffusivity is given or
needed in the paper's own model.

Since the paper resolves no separate transport resistances, CADET's
Lumped Rate Model without Pores (a single axial dispersion coefficient,
instantaneous local equilibrium) is the model that matches the paper
exactly -- not an approximation of a richer model. This is configured as
COLUMN_MODEL_1D with NPARTYPE=1, particle_type_000/HAS_FILM_DIFFUSION=0,
ADSORPTION_MODEL=LINEAR with IS_KINETIC=0 (matching the paper's constant
retention factor k). Column geometry maps onto CADET's native geometries:

    Cylinder column   -> GEOMETRY='AXIAL_FLOW_CYLINDER'
    Both cone columns -> GEOMETRY='AXIAL_FLOW_FRUSTUM' (same physical tube,
                          same CROSS_SECTION_AREA_SMALL_END/_LARGE_END for
                          both rho_s=2 and rho_s=0.5 -- only FORWARD_FLOW
                          differs between the two flow directions)

Fig. 5 (p. 42) reports the actual measured, flow-rate-dependent local
plate height H(xi) for valerophenone along the 2.1/4.2 mm i.d. conical
column (the geometry used for cone_rho_s=0.5 here): a van-Deemter-shaped
curve, decreasing from ~10.5 micron at xi=0 to a minimum ~9.5 micron
around xi=0.4-0.6, then rising to ~11.6-11.7 micron at xi=1. Reproducing
this requires the local axial dispersion coefficient to satisfy
D_ax(v) = H(v)*v/2 for the van Deemter form H(v) = A + B/v + C*v (A/2 =
longitudinal-diffusion-independent term, B = B-term prefactor, C = C-term
prefactor), i.e. D_ax(v) = (A*v + B + C*v^2)/2, a quadratic in v. This is
represented in CADET via the COL_DISPERSION_DEP='VAN_DEEMTER' parameter
dependency (coefficients VD_A, VD_B, VD_C, derived in Step 2).

===========================================================================
Step 2 -- Reparameterization (paper's notation -> CADET parameters)
===========================================================================
Paper symbols: r_e, r_s (entrance/exit radii), s=r_s/r_e, L (column
length), k (retention factor), H(v) (local plate height as a function of
local velocity), epsilon_t (total porosity, "et").

epsilon_t is not given directly for the real experimental columns (only
"assumed 65%" for the separate, purely theoretical Sec. 4.1 calculations),
but it is derived exactly from data reported for the cylindrical column:
its bed volume (V_bed=1.06 cm^3, matching pi*r_e^2*L for r_e=1.5 mm,
L=15 cm to 4 digits), flow rate (0.35 mL/min), measured valerophenone
first moment (mu_1=3.869 min, Table 1), and retention factor (k=1.08,
p. 43). From mu_1 = t_0*(1+k):

    t_0   = mu_1 / (1+k)                              [void time]
    et    = t_0 * Fv / V_bed                          [total porosity]
    K_eq  = k * et / (1 - et)                         [LINEAR ka/kd, kd=1]

H(v) = A + B/v + C*v is obtained by:
  1. Digitizing Fig. 5's solid curve (the fitted valerophenone H(xi)) at
     600 DPI -- see `Gritti2019_fig6_fig5H_digitized.csv`
     (1749 points; axis-tick calibration residuals <0.015 units; title/
     legend regions excluded; stray misclassified pixels removed by a
     rolling-median outlier filter, <0.6 micron threshold).
  2. Converting xi to local interstitial velocity v(xi) for the exact
     column Fig. 5 was measured on (r_e=2.1 mm, s=0.5, Fv=0.40 mL/min),
     divided by the total porosity derived above.
  3. Fitting A, B, C by nonlinear least squares (`scipy.optimize.curve_fit`)
     to H(v) = A + B/v + C*v: RMSE=0.071 micron, max abs. error=0.39
     micron (over a 9.5-12 micron range) -- see `VD_A, VD_B, VD_C` below.

This same (et, K_eq, H(v)) triple is applied, unchanged, to all three
configurations, per the paper's own stated assumption (Sec. 4.1: "All
columns are packed identically with the same particles... and have the
same external porosity"); H(v) is a velocity-only relationship, with no
further xi- or geometry-dependence, exactly as measured. Column geometry
(radii, length, flow rate, injection volume) is otherwise the real,
physical, reported geometry -- no other free/fitted parameters are
introduced.

===========================================================================
Step 3 -- Extracted parameters (traceable to a specific paper location)
===========================================================================
  L               = 0.15 m                 (p. 34 abstract; p. 42 Sec. 4.1.4)
  r_e (cylinder)  = 1.5 mm  -> 3.0 mm i.d. (Fig. 6 caption; Sec. 3.3)
  r_e (cone, small end) = 1.05 mm -> 2.1 mm i.d. (Fig. 6 caption)
  r_s (cone, large end) = 2.10 mm -> 4.2 mm i.d. (Fig. 6 caption)
  d_p             = 5 micron                (Sec. 3.3, "5 micron XBridge-C18")
  Fv (cylinder)   = 0.35 mL/min             (Sec. 4.2.2 / Table 1 header)
  Fv (both cones) = 0.40 mL/min             (Sec. 4.2.2 / Table 1 header)
  V_inj           = 0.5 microL              (Sec. 3.4.2)
  k (valerophenone) = 1.08                  (p. 43, efficiency-loss list)
  H(xi), valerophenone (Fig. 5, digitized)  -- see Step 2 above.
  H_bar_paper_uniform = 10.8 micron, 12.1% loss (p. 43; the paper's own
                                              "H uniform" cross-check,
                                              superseded here by the real
                                              H(v), kept for reference)
  H_bar_paper_full = 11.6 micron, 18.1% loss (p. 43; the paper's full,
                                              flow-dependent-H result --
                                              this script's primary target)
  mu_1, mu_2' (Table 1, p. 44)  -- see table above; used for et and as a
                                    primary quantitative validation target.

All conversions beyond mm/micron/mL/min -> SI (m, m^2/s, m^3/s) are
applied explicitly in the code below.