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
is the one for valerophenone, one of the five n-alkanophenones measured. The
seven measured values themselves are used here, not the drawn curve; see below.

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

The van Deemter coefficients come from the measurements behind Fig. 5, which
Gritti et al. provided as a table per personal correspondence and which is
hardcoded as `PLATE_HEIGHT_TABLE` in `data/Gritti2019_plate_height.py`: H_1/2 from
the half-height peak width of six n-alkanophenones on the 3.0 mm i.d.
cylindrical reference column at the seven flow rates, together with the
paper's own assignment of each flow rate to an axial position in each cone.
Two steps remain:

1. Each flow rate is converted to the interstitial velocity of the column the
   plate heights were measured on, Fv / (pi*r_e^2) / et with r_e = 1.5 mm and
   the total porosity above. The z/L columns are not used: the assignment
   Fig. 5 draws is a presentation of the same seven points, not an extra
   measurement.
2. A, B and C are fitted to H(v) = A + B/v + C*v by least squares over the
   valerophenone column, giving an RMSE of 0.019 µm and a maximum error of
   0.039 µm over the measured 9.47–11.54 µm range. `fit_van_deemter()` in
   `data/Gritti2019_plate_height.py` performs the fit, so the coefficients are
   derived from the table rather than written out; the three scripts
   import it and share one H(v). `plot_fig5_verification()` in
   `Gritti2019_fig6.py` redraws the fit against the measured points on
   Fig. 5's own axes.

As a check on the result, the fit gives H = 9.48 µm at 0.35 mL/min on the
cylindrical column, against the 9.5 µm the paper states on p. 43.

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
    H(v)            = measured, 7 flow rates (data/Gritti2019_plate_height.py)
    mu_1, mu_2'     = Table 1, p. 44         (used for et and as validation target)

The paper also quotes two mean plate heights on p. 43: 10.8 µm (12.1 % loss)
for its own uniform-H cross-check, and 11.6 µm (18.1 % loss) for the full
flow-dependent H. The latter is what this script aims at; both are kept in the
script for reference.

All unit conversions to SI are done explicitly in the code.

## Reference data

`data/Gritti2019_fig6_measured.csv` holds the three measured chromatograms that
Fig. 6 plots: time in seconds, absorbance in AU, 0.1 s sampling, over the
panel's own 221.7-247.7 s window. They are the authors' measured traces,
taken from the `CylIsoAlkano` worksheet of their Origin project for
Figs. 6-8, which holds the cylinder run together with both cone runs at the
positions the published panel uses. The project, its worksheet exports and
`make_measured_csv.py`, which rebuilds the file from them, are all in `data/`. The analyte is valerophenone, the fourth
of the five n-alkanophenones in the isocratic run.

The two cone traces in that worksheet are the standalone cone runs moved in
time, by -1.897 and -2.889 s, so that all three peaks coincide; the values are
otherwise identical to the standalone runs to the last digit, and the
s = 2 copy additionally carries a 1.00046 amplitude factor. This is the
display adjustment Sec. 4.2.2 refers to, and it is what the published figure
shows, so it is what a comparison against the figure has to use. Replacing the
digitized curves with these exact traces therefore changes the precision of
the reference, not its time base.

What the adjustment costs is that the NRMSE no longer sees the retention-time
error. The recorded, unshifted retention times are 232.1, 234.0 and 235.0 s
against the model's 232.2, 232.3 and 232.3 s, and that error is reported in
full by Delta mu_1, which is taken against Table 1 and is unaffected by any
display alignment. See the section below: it is a real discrepancy of about
1 % in cone void volume, not an artifact of the reference data.

## The retention-time discrepancy hidden by the display alignment

Because the reference traces carry the figure's alignment, the NRMSE measures
peak shape and the retention-time error lives entirely in Delta mu_1. For the
cones that error is 0.74 % and 1.17 % against Table 1, and it is worth setting
out where it comes from, since it is easy to lose sight of behind a 1 % NRMSE.

It is also worth knowing how large an effect it is. On peaks whose half-height
width is about 4.8 s, the 1.8-2.7 s offset is roughly half the full width, and
the pointwise difference along the flanks becomes comparable to the peak height
itself: compared at the recorded times instead of the figure's, the same
simulations score 21.5 % and 29.2 % rather than about 1 %. Sliding a measured
trace against *itself* by the same offset produces 21.6 %, so that figure is
retention time alone and says nothing about shape.

Extracolumn volume does not explain the offset. Working in
transported volume, mu_1*Fv, the cylinder gives 1.3542 mL and the cones 1.5600
and 1.5668 mL. Calibrating epsilon_t on the cylinder with no extracolumn
volume reproduces the cylinder exactly by construction and leaves the cones
11.8 and 18.6 µL short. A shared extracolumn volume makes that worse rather
than better: the cylinder runs at the lower flow rate, 0.35 against
0.40 mL/min, so a given dead volume costs it more time, and absorbing it into
epsilon_t lowers the cone prediction further — 25 µL moves the predicted cone
mu_1 from 232.2 s to 231.7 s, away from the measured 234-235 s. Solving for
the extracolumn volume that would reconcile cylinder and cone gives -82 and
-130 µL, which is not a volume. The two cone directions also differ from each
other by 6.8 µL, or 1.02 s, in the same tube at the same flow rate, which no
fixed volume can produce and which the model structurally cannot reproduce
either, since the first moment of a frustum does not depend on flow direction.
What remains are the cone's real internal volume and its packing: about 1 %
more void volume than the nominal frustum, which is the open question noted in
the geometry discussion.
