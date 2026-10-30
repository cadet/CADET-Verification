# Gritti et al. (2019), Fig. 8 — gradient elution of bombesin

Reference:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

`Gritti2019_fig8.py` holds the model definition, the simulation run, the
comparison plot and the validation metrics.

## The case

Fig. 8 (p. 45) shows measured gradient-elution chromatograms of the peptide
bombesin (14 residues, 1619.85 g/mol) on the same three configurations as
Figs. 6 and 7, all 150 mm long and packed with the same batch of 5 µm
XBridge-C18 particles:

* **Cylinder, rho_s = 1**: 3.0 mm i.d., 0.35 mL/min.
* **Cone, rho_s = 2**: 2.1 mm i.d. entrance widening to 4.2 mm i.d.,
  0.40 mL/min.
* **Cone, rho_s = 0.5**: the same tube with the flow reversed, 0.40 mL/min.

The paper's s = rho_s is the ratio of outlet to inlet radius of the truncated
cone (Eq. 1); this is a frustum with a linearly varying cross-section, not a
radial-flow column. All three are reproduced with one model, since only the
geometry and the flow direction differ between them.

Bombesin is separated by a linear acetonitrile/water gradient, and its
retention follows the linear solvent strength model of Sec. 2.4,
k(phi) = k0*exp(-S*(phi-phi0)), with phi the acetonitrile volume fraction.
Reproducing that needs a second, non-retained transport field for phi itself,
whose local value modulates the analyte's retention as the band migrates, so
the simulation carries two components even though only the analyte is compared
against data.

The references are the measured chromatograms in
`data/Gritti2019_fig8_measured.csv` and the retention times, moments and
half-height widths of Table 3.

## Model choice

**Bulk and particle transport.** The paper's model (Secs. 2.2–2.4) treats the
column as a black box: axial convection with a smoothly varying cross-section
and velocity (Eqs. 4–5), with all band broadening in an aggregate, axially
varying plate height H(xi). There is no film- or pore-diffusion sub-model and
no mass-transfer parameter is reported for any analyte. The matching CADET
choice is therefore the equilibrium particle, i.e. `HAS_FILM_DIFFUSION=0` on a
`COLUMN_MODEL_1D` particle type: one `TOTAL_POROSITY`, axial dispersion only
and local equilibrium.

**Geometry.** `GEOMETRY='AXIAL_FLOW_FRUSTUM'` for the two conical runs and
`GEOMETRY='AXIAL_FLOW_CYLINDER'` for the cylindrical reference.
`FORWARD_FLOW` picks which end is the inlet, so the same frustum — same
`CROSS_SECTION_AREA_SMALL_END`, `_LARGE_END` and `BED_LENGTH` — serves both
rho_s = 2 (`FORWARD_FLOW=0`, entering the 2.1 mm end) and rho_s = 0.5
(`FORWARD_FLOW=1`, entering the 4.2 mm end). That mirrors the experiment, where
Table 3's caption describes rho_s = 0.5 as the same conical column after
reversing the flow.

**Retention.** The LSSM law of Eq. 28 is the isotherm underlying CADET's
`MOBILE_PHASE_MODULATOR` binding model (Melander & Horvath 1977;
Karlsson 2004):

    dq_1/dt = k_a*exp(gamma*c_p0)*c_p1*qmax*(1 - q_1/qmax) - k_d*c_p0^beta*q_1

Component 0 is the modifier, here the acetonitrile fraction, and is inert
(`NBOUND=0`); component 1 is bombesin (`NBOUND=1`). With `is_kinetic=0`, which
matches the paper's retention-factor-only description, and beta = 0, which
drops the ion-exchange power-law term, the dilute limit (q_1 << qmax, valid for
a 3 µL injection of 0.1 g/L bombesin) gives

    q_1/c_p1 = (k_a*qmax/k_d) * exp(gamma*c_p0)

which is the LSSM law with gamma = -S, once k_a*qmax/k_d is chosen so that the
retention factor k'(phi) = F*(q_1/c_p1), with the phase ratio
F = (1-eps_t)/eps_t, equals k0*exp(-S*(phi-phi0)).

**Axial dispersion.** The paper measures the plate height only for the
alkanophenones (Sec. 4.2.1, Fig. 5), not for bombesin, but reuses that curve for
its own bombesin prediction on p. 45. This script does the same, reusing the
measured plate heights and the van Deemter fit VD_A, VD_B, VD_C shared with
`Gritti2019_fig6.py` and `Gritti2019_fig7.py` through
`data/Gritti2019_plate_height.py`, so that
D_ax(z) = scale*H(v(z))*v(z)/2 through `COL_DISPERSION_DEP='VAN_DEEMTER'`.
Gritti2019_fig6.md derives D_ax = H(v)*v/2 from the paper's plate-height
definition and describes the fit.

Because the curve belongs to a different compound, one scale factor on it is
unavoidable. It is calibrated once against the cylindrical column's second
central moment from Table 3 and reused unchanged for both cones, which follows
the paper's own structure and tests whether the compound-independent H(v)
transfers across the several-fold velocity change along the frustum. That
calibration is why the cylinder's Delta mu_2 entry is left empty in the
validation table: it is fitted there, not predicted. The curve is isocratic and
is likewise reused unchanged under gradient conditions, as the paper does.

The modifier's own dispersion is a small, geometry-independent placeholder,
negligible against convection, so its ramp travels essentially undistorted.
That is the paper's own assumption in Sec. 2.4 (Eqs. 29–30).

## Parameters

Read directly from Secs. 3.3, 3.4.3 and Table 3:

    L                = 0.15 m
    r_cylinder       = 1.5e-3 m    (3.0 mm i.d.)
    r_small (frustum)= 1.05e-3 m   (2.1 mm i.d.)
    r_large (frustum)= 2.10e-3 m   (4.2 mm i.d.)
    Q_cylinder       = 0.35 mL/min
    Q_cone           = 0.40 mL/min (both flow directions)
    V_inj            = 3.0 uL
    phi0             = 0.10, phi_final = 0.55
    t_gradient       = 5 min  ->  beta = 0.09/min = 0.0015/s
    Table 3, cylinder: t_R = 4.798 min, sigma_t^2 = 0.000288 min^2

Two values are not restated for this case study and are carried over from
elsewhere in the paper:

* **eps_t = 0.65**, the total porosity, from the worked example of Sec. 4.1.4
  for the same particles and columns. The script checks at runtime that it
  reproduces that section's entrance velocity u0(0) = 17.77 cm/min for
  Fv = 0.40 mL/min and r_e = 1.05 mm.
* **S = 25**, the LSSM slope, which the paper gives for its illustrative
  17-peptide mixture (captions of Figs. 2 and 4) at a gradient steepness of
  0.09/min — the same steepness as the bombesin experiments of Sec. 3.4.3. It
  is adopted here as the best available value for a peptide under matching
  conditions.

k0, the retention factor at phi0, is solved in closed form from the cylinder's
own measured retention time, using the paper's exact gradient-elution relation
(Eq. 31), which integrates the LSSM retention law along the column:

    e(xi) = m(xi) + (1/G)*ln(1 + G*k0*m(xi))            [Eq. 31]
    G     = S*beta*L/u0(0)                              [Eq. 32]
    m(xi) = (1 + rho(xi) + rho(xi)^2)/3,  m(1) = (1+s+s^2)/3
    =>  k0 = [exp(G*(e(1)-m(1))) - 1] / (G*m(1))

evaluated with the cylinder's own G, m(1) = 1 and e(1) = t_R/t_ref. The closed
form for m(xi) follows from integrating rho(xi)^2 = [1+(s-1)xi]^2 and is
consistent with the s -> 1/s flow-direction invariance of Eq. 18.

k0 is a property of the analyte and the stationary phase, not of the geometry,
so it is reused unchanged for both cones. Their retention times and peak widths
are then predictions, which is what makes Table 3's rho_s = 2 and 0.5 rows a
test of the frustum geometry.

## Reference data

Table 3 gives only summary moments, so the three measured traces are used as
the curve reference. `data/Gritti2019_fig8_measured.csv` holds the three
chromatograms that Fig. 8 plots: time in seconds, absorbance in AU, 0.1 s
sampling, over the panel's own 272-306 s window, columns time_s, cylinder_AU,
cone_s2_AU and cone_s05_AU, with peak heights 0.0193, 0.0167 and 0.0179 AU.
They are the authors' measured traces, taken from the `CylGrdBombesin`
worksheet of their Origin project for Figs. 6-8, which holds the cylinder run
together with both cone runs at the positions the published panel uses. The
project, its worksheet exports and `make_measured_csv.py`, which rebuilds the
file from them, are all in `data/`.
Bombesin elutes as a single peak, so no peak selection is needed.

The two cone traces in that worksheet are the standalone cone runs moved in
time by +0.529 and -0.126 s, with values otherwise identical to the last
digit, so the NRMSE measures peak shape and the retention-time error is
carried by Delta mu_1 against Table 3. The offsets are small here in any case.
Gritti2019_fig6.md works through the same point for Fig. 6, where it matters
most.
