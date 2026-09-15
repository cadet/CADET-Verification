# Gritti et al. (2019), Fig. 6 — isocratic elution of valerophenone

Reference:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

`Gritti2019_fig6.py` holds the model definition, the simulation run, the
comparison plot and the validation metrics.

## The case

Fig. 6 (p. 43, Sec. 4.2.2) shows the measured isocratic elution peak of
n-valerophenone on three column configurations, all packed with the same batch
of 5 µm XBridge-C18 particles and run with acetonitrile/water 75/25 (v/v) at
27 °C:

1. **Cylinder, rho_s = 1**: a conventional cylindrical column, r_e = 1.50 mm
   (3.0 mm i.d.) × 150 mm, Fv = 0.35 mL/min.
2. **Cone, rho_s = 2**: a conical column narrowing from 2.1 mm i.d. to 4.2 mm
   i.d. over 150 mm, Fv = 0.40 mL/min, flow from the narrow to the wide end.
3. **Cone, rho_s = 0.5**: the same physical tube with the flow reversed, wide
   to narrow, same flow rate.

Table 1 (p. 44) lists the measured retention time, moments and half-height
width for valerophenone on all three, which gives a numerical validation target
alongside the digitized curve itself:

    Config      Fv[mL/min]  t_R[min]  mu1[min]  mu2'[min^2]  w1/2[min]
    Cylinder    0.35        3.865     3.869      0.00156      0.0718
    Cone s=2    0.40        3.898     3.900      0.00114      0.0800
    Cone s=0.5  0.40        3.915     3.917      0.00113      0.0792

## Model choice

The paper's own theory (Sec. 2) is Giddings' band-broadening model, written in
terms of a total column porosity epsilon_t, a retention factor k, and a local
plate height H(xi) that lumps eddy dispersion, longitudinal diffusion and the
mass-transfer resistances into a single coefficient integrated along the column
(Eqs. 11–26). It gives no particle porosity, film mass-transfer coefficient or
pore diffusivity, and needs none.

The matching CADET model is therefore the lumped rate model without pores: one
axial dispersion coefficient and local equilibrium. It is set up as
`COLUMN_MODEL_1D` with `NPARTYPE=1`, `HAS_FILM_DIFFUSION=0` and
`ADSORPTION_MODEL=LINEAR` with `IS_KINETIC=0`, the linear isotherm standing for
the paper's constant retention factor. The geometries map directly:

    Cylinder    ->  GEOMETRY='AXIAL_FLOW_CYLINDER'
    Both cones  ->  GEOMETRY='AXIAL_FLOW_FRUSTUM', same
                    CROSS_SECTION_AREA_SMALL_END / _LARGE_END for both, since
                    it is the same tube; only FORWARD_FLOW differs

Fig. 5 (p. 42) gives the local plate height along the 2.1/4.2 mm i.d. conical
column, the geometry used here for rho_s = 0.5. The paper obtains it by
measuring H at seven flow rates on the 3.0 mm i.d. cylindrical reference column
and assigning each measurement to the axial position at which the cone's local
velocity matches, assuming both columns are equally well packed. The curve drawn
is the one for valerophenone, one of the five n-alkanophenones measured.

Gritti et al. work in plate heights throughout and never introduce a dispersion
coefficient, so the measurement has to be translated. They define the local
plate height through the variance a band accumulates per unit migration
distance, dsigma_z^2 = H dz (Eqs. 19-20). In this model a pulse under a linear
isotherm spreads as sigma_z^2 = 2*D_ax*t/(1+k) while its centre migrates as
z = v*t/(1+k), so dsigma_z^2/dz = 2*D_ax/v and therefore D_ax = H(v)*v/2.

With the van Deemter form H(v) = A + B/v + C*v this is
D_ax(v) = (A*v + B + C*v^2)/2, which is CADET's
`COL_DISPERSION_DEP='VAN_DEEMTER'` with coefficients VD_A, VD_B, VD_C. CADET
evaluates it at the local interstitial velocity at every quadrature point, and
the factor 1/2 is part of the dependency, which is why `COL_DISPERSION` is set
to 1.

## From the paper's numbers to CADET parameters

The paper uses r_e and r_s for the entrance and exit radii, s = r_s/r_e, L for
the column length, k for the retention factor, H(v) for the local plate height
and epsilon_t for the total porosity.

epsilon_t is not stated for the real columns — the 65 % figure belongs to the
purely theoretical calculations of Sec. 4.1 — but it follows from data reported
for the cylindrical column: its bed volume (1.06 cm^3, which matches
pi*r_e^2*L to four digits), its flow rate (0.35 mL/min), the measured first
moment of valerophenone (3.869 min, Table 1) and the retention factor
(k = 1.08, p. 43). From mu_1 = t_0*(1+k):

    t_0   = mu_1 / (1+k)                 void time
    et    = t_0 * Fv / V_bed             total porosity
    K_eq  = k * et / (1 - et)            LINEAR ka, with kd = 1

The van Deemter coefficients come from Fig. 5 in three steps:

1. The solid valerophenone curve of Fig. 5 is digitized at 600 DPI into
   `Gritti2019_fig6_fig5H_digitized.csv` (1749 points; axis calibration
   residuals below 0.015 units; title and legend regions excluded; stray
   misclassified pixels dropped by a rolling-median filter). Its position
   column is named xi; the paper writes the same coordinate as zeta.
2. The paper's assignment is inverted, converting xi back to the local
   interstitial velocity of the column Fig. 5 was measured on (r_e = 2.1 mm,
   s = 0.5, Fv = 0.40 mL/min), divided by the total porosity above.
3. A, B and C are fitted to H(v) = A + B/v + C*v by least squares, giving an
   RMSE of 0.071 µm and a maximum error of 0.39 µm over a 9.5–12 µm range.
   `plot_fig5_verification()` in the script redraws the fit against the
   digitized points.

The van Deemter form is ours: the paper draws a smooth best curve through the
points and states no functional form. It serves as an interpolant, not as a
mechanistic decomposition, and it never extrapolates -- the seven flow rates
span 0.204 to 0.816 mL/min, a factor of four, which is exactly the velocity
ratio along a cone whose radius doubles.

The same epsilon_t, K_eq and H(v) are used unchanged for all three
configurations, which is the paper's own assumption (Sec. 4.1: all columns are
packed identically with the same particles and share the same external
porosity). H(v) depends on the velocity only, with no further dependence on
position or geometry. Everything else — radii, length, flow rates, injection
volume — is the reported physical geometry, and nothing else is fitted.

## Parameters and where they come from

    L               = 0.15 m                 (p. 34 abstract; p. 42 Sec. 4.1.4)
    r_e (cylinder)  = 1.5 mm  -> 3.0 mm i.d. (Fig. 6 caption; Sec. 3.3)
    r_e (cone, small end) = 1.05 mm -> 2.1 mm i.d. (Fig. 6 caption)
    r_s (cone, large end) = 2.10 mm -> 4.2 mm i.d. (Fig. 6 caption)
    d_p             = 5 micron               (Sec. 3.3)
    Fv (cylinder)   = 0.35 mL/min            (Sec. 4.2.2 / Table 1)
    Fv (both cones) = 0.40 mL/min            (Sec. 4.2.2 / Table 1)
    V_inj           = 0.5 microL             (Sec. 3.4.2)
    k (valerophenone) = 1.08                 (p. 43)
    H(xi)           = Fig. 5, digitized      (see above)
    mu_1, mu_2'     = Table 1, p. 44         (used for et and as validation target)

The paper also quotes two mean plate heights on p. 43: 10.8 µm (12.1 % loss)
for its own uniform-H cross-check, and 11.6 µm (18.1 % loss) for the full
flow-dependent H. The latter is what this script aims at; both are kept in the
script for reference.

All unit conversions to SI are done explicitly in the code.

## A caveat on the NRMSE

Sec. 4.2.2 notes that the peaks *printed* in Fig. 6 were slightly adjusted in
time for display. A few tenths of a percent of the NRMSE are therefore that
display artifact rather than a shape mismatch. Delta mu_1 and Delta mu_2 are
unaffected, since they are taken against Table 1 rather than against the
digitized curve. No comparable statement exists for Figs. 7 and 8.
