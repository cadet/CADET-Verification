# Brandani (2021), Fig. 2 — m-xylene batch uptake in a finite bath

Reference:

    S. Brandani, "Kinetics of liquid phase batch adsorption experiments",
    Adsorption 27 (2021) 353-368, https://doi.org/10.1007/s10450-020-00258-9
    (open access, CC BY 4.0). Fig. 2, the dashed m-xylene curve
    (c0 = 250 mol/m^3, M_S = 40 g, Sh = 2).

`Brandani2021_fig2.py` holds the model definition, the simulation run, the
comparison plot and the NRMSE. `Brandani2021_fig2_extract.py` regenerates
`Brandani2021_fig2_digitized.csv` from the published PDF.

## The case

A known volume of solution is contacted with a known mass of adsorbent in a
closed, well stirred vessel, and the liquid concentration is followed until
equilibrium. This is the batch uptake (immersion) experiment, and it is the
unit operation that CADET calls `FINITE_BATH`: the 0D bulk of the CSTR
combined with the particle models of the GRM, coupled through a film
diffusion resistance.

The system is the separation of xylenes on Y zeolite beads, measured in the
finite bath experiments of

    E. Santacesaria, M. Morbidelli, P. Danise, M. Mercenari, S. Carra,
    "Separation of xylenes on Y zeolites. 1. Determination of the adsorption
    equilibrium parameters, selectivities, and mass transfer coefficients
    through finite bath experiments", Ind. Eng. Chem. Process Des. Dev. 21
    (1982) 440-445.

Brandani takes that system's parameters (his Table 1) and uses it to generate
batch uptake curves with a macropore diffusion model, in order to show that
the usual pseudo-first-order, pseudo-second-order and Elovich linearisations
fit those curves with R^2 close to unity while identifying the wrong
transport mechanism.

**This is therefore a code-to-code comparison, not a validation against
measurement.** Fig. 2 is a simulation, not data. Reproducing it checks the
CADET finite bath against an independent implementation of the same
equations; validating the model against the experiment would require the
uptake curves of Santacesaria et al. (1982), which are not reproduced here.

## Which curve

Fig. 2 holds four curves: p-xylene and m-xylene, each for Sh = inf (pure pore
diffusion, solid lines) and Sh = 2 (film resistance added, dashed lines).
This case study reproduces **m-xylene, Sh = 2**. It is the only one of the
four that exercises both the film boundary condition and the intraparticle
diffusion operator while staying mildly nonlinear: Brandani's nonlinearity
parameter (his Eq. 38) is Gamma = 0.44, against Gamma = 0.90 for p-xylene,
whose internal profiles (his Fig. 5) are nearly a shrinking core. A
discrepancy in this case is therefore attributable rather than buried in a
steep internal front.

Sh = inf is not directly configurable. A finite bath requires
`HAS_FILM_DIFFUSION = 1`; setting it to 0 reconfigures the unit as a CSTR,
whose particles are in rapid equilibrium with the bulk, which is a different
model. Brandani's own solid lines use an arbitrarily large k_F (1e3 m/s) for
the same reason.

## Model choice

Brandani's Eqs. 2-3, 10-13 are, in order: a closed well mixed bulk with a
film flux to the particles, macropore diffusion inside a spherical bead with
the adsorbed phase in local equilibrium with the pore fluid, and a Langmuir
isotherm. The case is therefore set up as CADET's `FINITE_BATH` with a
`GENERAL_RATE_PARTICLE` and `MULTI_COMPONENT_LANGMUIR` at `IS_KINETIC = 0`.

The vessel is closed, so the flow rates are set to zero. With
F_in = F_out = 0 the `LIQUID_VOLUME` term drops out of the bulk balance
entirely and only `BULK_POROSITY` affects the solution; the volume is still a
required input and is set to the paper's 200 mL for documentation.

## From Brandani's parameterisation to CADET parameters

Brandani's bead equation (his Eq. 10) is

    eps_p dc_p/dt + (1-eps_p) dq/dt
        = (eps_p/tau) D_m (1/r^2) d/dr ( r^2 dc_p/dr )

and CADET's bead equation (`general_rate_model.rst`, Eq. `ModelBead`),
without surface diffusion and with a pore accessibility factor of 1, is

    dc_p/dt + ((1-eps_p)/eps_p) dq/dt
        = D_p [ d^2/dr^2 + (2/r) d/dr ] c_p

Multiplying the second by eps_p puts it in the form of the first, term for
term, and identifies

    PORE_DIFFUSION  D_p = D_m / tau

Both equations carry the same `(1-eps_p)/eps_p` prefactor on dq/dt, so
CADET's solid phase concentration and Brandani's q are both per unit solid
(non-pore) volume and `MCL_QMAX` takes q_S unchanged.

The film boundary condition matches directly, CADET's

    k_f [ c_l - c_p(r_p) ] = eps_p D_p dc_p/dr (r_p)

being the usual equality of the film flux and the pore flux. Brandani
estimates the worst case film resistance from Sh = 2, i.e.

    FILM_DIFFUSION  k_f = D_m / r_p

which reproduces his tabulated 3.31e-6 m/s exactly and so confirms the
Sherwood convention.

CADET's `MULTI_COMPONENT_LANGMUIR` at equilibrium is
q = q_max (k_a/k_d) c_p / (1 + (k_a/k_d) c_p), so the Langmuir affinity is
b = k_a/k_d and the isotherm is set with `MCL_KD = 1`, `MCL_KA = b`.

The bulk porosity follows from Brandani's volume ratio alpha = V_S/V_F,

    BULK_POROSITY  eps_b = V_F / (V_F + V_S),   V_S = M_S / rho_S

For this to be the bead volume that eps_b needs, rho_S has to be the bead
envelope density rather than the skeletal density. Brandani does not say so
explicitly, but his Q = eps_p c_p + (1-eps_p) q is per bead volume, which
implies it, and the reading is confirmed numerically below.

## Parameters (Brandani, Table 1; m-xylene column)

| Quantity | Paper | CADET |
|---|---|---|
| eps_p | 0.20 | `PAR_POROSITY` = 0.20 |
| tau | 2.15 | folded into `PORE_DIFFUSION` |
| R_p | 0.65 mm | `PAR_RADIUS` = 6.5e-4 m |
| D_m | 2.15e-9 m^2/s | folded into `PORE_DIFFUSION`, `FILM_DIFFUSION` |
| D_m/tau | | `PORE_DIFFUSION` = 1.0e-9 m^2/s |
| k_F (Sh = 2) | 3.31e-6 m/s | `FILM_DIFFUSION` = D_m/R_p = 3.3077e-6 m/s |
| c0 | 250 mol/m^3 | `INIT_C` = 250 |
| rho_S | 1400 kg/m^3 | via eps_b |
| M_S | 40 g | via eps_b |
| V_F | 200 mL | `LIQUID_VOLUME` = 2.0e-4 m^3 (inactive, see above) |
| q_S | 2450 mol/m^3 | `MCL_QMAX` = 2450 |
| b | 0.006 m^3/mol | `MCL_KA` = 0.006, `MCL_KD` = 1 |

giving alpha = V_S/V_F = 0.142857 and eps_b = 0.875.

The particles start clean and the bulk starts loaded, so `INIT_C` = c0 while
`INIT_CP` = `INIT_CS` = 0. That non-equilibrium start is the experiment.

### Consistency check on rho_S

Brandani plots the particle phase concentration Q, not the liquid
concentration. The two are tied by his Eq. 2, Q = (c0 - c)/alpha, so the
curve has to approach the Q that solves the isotherm and the mass balance
simultaneously,

    c_inf = c0 - alpha Q_eq(c_inf),
    Q_eq(c) = eps_p c + (1-eps_p) q_S b c / (1 + b c)

With the values above this gives c_inf = 125.9 mol/m^3 and
Q_inf = 868.7 mol/m^3. The extracted Sh = 2 curve reaches 848.1 mol/m^3 at
t = 3000 s, i.e. 97.6% of that equilibrium, which is what a film-limited
curve that has not quite levelled off should do, and the Sh = inf curve in
the same figure reaches 866 mol/m^3. Had rho_S been a skeletal density the
plateau would have missed by tens of percent, so the envelope reading is
confirmed.

## Reference data

Fig. 2 is a vector graphic, not a raster image: page 9 of the PDF draws the
four curves as stroked polylines with 2782 `lineto` operators and carries no
image XObject. `Brandani2021_fig2_extract.py` therefore reads the curve
vertices straight out of the page content stream with PyMuPDF instead of
going through WebPlotDigitizer or `CLAUDE/digitize_figure.py`. The m-xylene
Sh = 2 curve is identified as the lower of the two dashed (black) paths; the
solid blue paths are the Sh = inf pair.

The axes are calibrated on the bounding boxes of the tick labels, which the
PDF also carries as text: t = 0 at x = 335.05 pt and t = 3000 s at
x = 536.45 pt, Q = 0 at y = 407.9 pt and Q = 2000 mol/m^3 at y = 285.3 pt,
both sets equally spaced to within the extraction precision. The solid curves
run past the right edge of the plot box because the chart clips them, so the
trace is cut at t = 3000 s.

This yields 868 distinct points with no pixel-quantisation error. The residual
uncertainty is Brandani's own plotting resolution, not the extraction, which
is why the file keeps the `_digitized` name of the other case studies but is
not digitized in the usual sense.

## Metric

Only the NRMSE is reported, the same definition as in
`src/validation/validation_metrics.py`: the root mean square deviation
between simulation and reference on the overlap of the two time axes,
normalised by max|Q_ref|.

The other three metrics of the shared table do not carry over. A batch uptake
curve has no peak and no elution time, so the first and second moments are
undefined; and the vessel is closed, so there is no injected mass to balance
an outlet integral against. The mass balance that does apply,
V_F c0 = V_F c(t) + V_S Q(t), is how Q is obtained from the simulated liquid
concentration in the first place and so cannot also serve as an independent
check of it.

## Result

Run with a DG particle discretization at `PAR_POLYDEG = 4`, `PAR_NELEM = 8`:

    NRMSE               0.209 %
    max |deviation|     3.77 mol/m^3   (reference amplitude 848.1 mol/m^3)
    final Q             849.2 mol/m^3  vs. 848.1 extracted, 97.8% of Q_inf

The reported run is converged well below that residual. Refining the
particle grid and switching to the other spatial method leaves the uptake
curve unchanged to within a small fraction of the deviation from the figure:

| discretization | max dev. vs. finest [mol/m^3] | NRMSE vs. Fig. 2 [%] |
|---|---|---|
| DG, polydeg 4, nelem 8 (reported) | 5.9e-2 | 0.2091 |
| DG, polydeg 4, nelem 16 | 2.6e-3 | 0.2091 |
| DG, polydeg 6, nelem 16 | reference | 0.2091 |
| FV, 128 cells | 9.7e-2 | 0.2085 |

The spread across discretizations is about 0.01% of the curve amplitude,
roughly thirty times smaller than the 0.21% NRMSE, and the finite volume and
DG code paths agree. The residual is therefore attributable to the
resolution of the published figure rather than to the discretization or to
the model mapping.
