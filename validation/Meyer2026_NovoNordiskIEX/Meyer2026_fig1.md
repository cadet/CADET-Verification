Reproduction of Fig. 1 from:

    Meyer et al., 2026, Computers and Chemical Engineering,
    "ChromOps.jl: High-order simulation and discrete forward sensitivity
     analysis for chromatography models", Section 4.2: "Ion-exchange
     chromatography with six components".

Files in this directory:

| file | purpose |
|---|---|
| `Meyer2026_fig1.py` | runs the CADET setup under both readings of Eq. (4), overlays Figure 1, prints metrics |
| `Meyer2026_fig1_paper_equations.py` | standalone solver of the paper's *own* Eqs. (1),(2),(4),(5); numpy/scipy only, no CADET |
| `Meyer2026_fig1_extract.py` | reads the Figure 1 curves out of the PDF vector graphics |
| `Meyer2026_fig1_digitized/` | the extracted curves, `A..F.csv` and `salt.csv` |

The CADET model itself is
`src/benchmark_models/setting_COL1D_NovoNordiskIEXbenchmark.py`.

===========================================================================
1. Case identification
===========================================================================
Governing model: Section 2 of the paper. Lumped-rate model with pores, axial
dispersion and linear driving-force film mass transfer (Eqs. 1-2), Danckwerts
boundary conditions (Eq. 3), and a steric mass action isotherm written in
driving-force form (Eqs. 4-5). Seven components: salt plus proteins A-F.

Parameters: Table 3 (model), Table 4 (isotherm), Eqs. (35)-(36) (rectangular
360 s protein injection), Eqs. (37)-(40) (six-phase piecewise-linear salt
gradient). Outlet concentrations are converted to optical density with
w = 1.167e4 AU L mol^-1 cm^-1.

Target: Figure 1 (PDF p. 11), the simulated chromatogram.

The paper does not state the initial condition. The only choice consistent
with the first phase of the salt program is a column pre-equilibrated at
0.04 mol/L salt with the exchanger fully in the salt form, q_s(0) = Lambda;
that is what is used here and in the standalone solver.

===========================================================================
2. Extraction of the reference curves
===========================================================================
Figure 1 is a vector graphic, so `Meyer2026_fig1_extract.py` reads the
polylines directly from the PDF content stream (PyMuPDF) rather than
digitizing a raster image. Curves are identified by stroke colour and the axes
are calibrated on the tick marks, which are themselves vector segments.

The calibration is exact rather than approximate: the recovered salt program
returns 0.2400 mol/L on the hold plateau and 1.0399 mol/L on the final step,
against the 0.24 and 1.04 of Eqs. (39)-(40), and the recovered phase-5 gradient
slope is 5.541e-5 mol/(L s) against the stated 0.4/7200 = 5.556e-5. So neither
of the two discrepancies in Section 5 is a digitization artifact.

===========================================================================
3. Mapping the paper's model onto CADET
===========================================================================
The paper's Eqs. (1)-(2) are CADET's `COLUMN_MODEL_1D` with
`HOMOGENEOUS_PARTICLE` particles (the LRMP) under an exact change of state
variable, not an approximation.

The paper's q_i is the adsorbed concentration per *pore-liquid* volume -- this
follows from Eq. (2), which carries a coefficient of 1 on both dc_p,i/dt and
dq_i/dt, and it is what makes Eqs. (1)-(2) conserve mass, since
d/dt[eps*c + (1-eps)*eps_p*(c_p + q)] then reduces to the flux terms alone.
CADET's solid-phase state c^s_i is per *solid* volume. Hence

    q_i = alpha * c^s_i,        alpha := (1 - eps_p)/eps_p = 0.5152.

Substituting this into CADET's pore balance

    dc^p_i/dt + alpha * dc^s_i/dt = 3/(eps_p*r_p) * k_f,i * (c^b_i - c^p_i)

gives the paper's Eq. (2) exactly for k_f,i = eps_p*r_p*k_MT,i/3, and the same
k_f,i turns CADET's bulk film term (1-eps_c)/eps_c * (3/r_p) * k_f,i into the
paper's (1-eps)*eps_p/eps * k_MT,i of Eq. (1). In the isotherm,

    Lambda - sum_j (nu_j+sigma_j) q_j
        = alpha * (Lambda_CADET - sum_j (nu_j+sigma_j) c^s_j),
    Lambda_CADET = Lambda/alpha.

So the porosity factor is absorbed completely and nothing is left over.

Resulting parameter mapping (input parameters only; no state is transformed,
and CADET's own c^s_i then equals q_i*eps_p/(1-eps_p) automatically):

| paper | CADET |
|---|---|
| k_MT,i | `FILM_DIFFUSION` k_f,i = eps_p * r_p * k_MT,i / 3 |
| Lambda = 324.7 mol/m^3 | `SMA_LAMBDA` = Lambda/alpha = 630.3 mol/m^3 |
| k_eq,i | `SMA_KA`/`SMA_KD` = k_eq,i * alpha^(nu_i - 1) |
| nu_i, sigma_i | `SMA_NU`, `SMA_SIGMA` unchanged |

The paper lumps r_p into k_MT and never reports it, but only the ratio k_f/r_p
enters CADET, so the assumed `PAR_RADIUS` cancels from the mapping and has no
effect on the solution. `SMA_REFQ` = `SMA_REFC0` = `SMA_LAMBDA` keeps ka, kd
numerically well-scaled (they would otherwise be O(1e-60), since nu_i ~ 20);
equal reference concentrations cancel from the equilibrium condition.

### 3.1 The one genuine difference: driving-force vs. mass-action kinetics

Factoring qbar^(-nu_i) out of the paper's Eq. (4) leaves CADET's mass-action
SMA *exactly*, times a state-dependent prefactor:

    dq_i/dt |paper = (ka_bar/alpha) * qbar_C^(-nu_i)
                     * [ qbar_C^(nu_i) c^p_i
                         - c^p_s^(nu_i) c^s_i / (k_eq,i * alpha^(nu_i - 1)) ]

Two consequences. The prefactor is strictly positive, so the two laws have the
same zero set: **the equilibrium isotherm is identical**, which is what makes
the ka/kd ratio above exact and state-independent. But the transient rates
cannot be matched by any constant ka_i, kd_i -- the required forward
coefficient (ka_bar/alpha)*qbar_C^(-nu_i) varies by ~20 orders of magnitude
over the load. There is no constant-coefficient reparameterization.

Rapid-equilibrium binding (`IS_KINETIC = 0`) is therefore used. It imposes the
exactly-mapped isotherm and is the exact common limit of both laws. That the
paper's ka_bar = 10 1/s actually sits in that limit is *verified*, not assumed:
against `Meyer2026_fig1_paper_equations.py`, which integrates the paper's own
driving-force kinetics,

| isotherm | max abs. peak-time difference | max peak-height difference |
|---|---|---|
| Eq. (4) as printed | 5 s | 0.67 % |
| bound-salt reading | 6 s | 0.70 % |

So the kinetic-form difference is real but immaterial for this benchmark, and
it is **not** the cause of either discrepancy below.

===========================================================================
4. Numerical validation
===========================================================================
* CADET is grid converged: FV/WENO3 with 256 and 1024 cells give peak times
  equal to 1 s and peak heights equal to 0.1 %.
* The standalone solver is grid converged: n = 400 and n = 800 give identical
  peak times and peak heights within 0.06 %.
* The standalone solver uses an analytic sparse Jacobian, verified against
  central differences to 8e-8 relative (`--check-jacobian`). This is a
  requirement, not an optimization: the desorption coefficient
  (ka_bar/k_eq,i)*(c_p,s/qbar)^nu_i spans more than 20 decades over the load
  and a finite-difference Jacobian fails outright.
* Integrated peak areas agree with Figure 1 to within 0.04 % for all six
  components under *both* readings of Eq. (4), confirming the feed
  concentrations of Eq. (36), the flow rate and the extinction coefficient
  independently of the two discrepancies below.

The two codes share no code and differ in discretization (FV/WENO3 vs. upwind
FV with a numerical-dispersion correction), in time integration (IDAS vs.
scipy BDF) and in binding treatment (quasi-stationary vs. the paper's
kinetics). Their agreement to 6 s and 0.7 % is therefore a genuine
cross-validation.

===========================================================================
5. Two inconsistencies between the paper's equations and its Figure 1
===========================================================================

### 5.1 The available-sites term of Eq. (4)

With Eq. (4) exactly as printed, the six peaks come out 253-569 s early and up
to 18 % low. Replacing the available-sites term

    qbar = Lambda - sum_j (nu_j + sigma_j) q_j        (Eq. 4 as printed)

by

    qbar = Lambda - sum_j nu_j q_j                    ( = q_s of Eq. 5 )

reproduces Figure 1 to within 0.5 % in every peak height, with all six peak
times collapsing onto a single uniform offset. CADET, mean NRMSE over the six
curves relative to peak height:

| Eq. (4) denominator | mean NRMSE | peak-height error | peak-time offset |
|---|---|---|---|
| Lambda - sum (nu_j+sigma_j) q_j | 20.1 % | -1.7 to -18.1 % | +38 to +356 s (non-uniform) |
| Lambda - sum nu_j q_j | **0.18 %** | +0.01 to +0.46 % | -6 to +11 s (after the uniform offset of 5.2) |

and the standalone solver of the paper's own equations, independently:

| Eq. (4) denominator | mean NRMSE | peak-height error | peak-time offset |
|---|---|---|---|
| Lambda - sum (nu_j+sigma_j) q_j | 14.4 % | -2.3 to -18.4 % | +253 to +569 s (spread 316 s) |
| Lambda - sum nu_j q_j | **0.13 %** | -0.01 to -0.44 % | +204 to +225 s (spread 21 s) |

**What identifies the steric term specifically.** The elution position of a
gradient-eluted SMA peak scales with qbar at the moment it desorbs, and qbar is
depleted only by whatever is *still bound*. Comparing the outlet salt
concentration at each peak maximum -- a quantity that is invariant under any
uniform time shift, so it isolates the isotherm from Section 5.2 entirely:

| comp | nu_i | salt at peak, Eq. (4) as printed | salt at peak, Fig. 1 | ratio | still bound in Fig. 1 |
|---|---|---|---|---|---|
| E | 10 | 0.2640 | 0.2795 | 1.059 | D C B A F |
| D | 20 | 0.3037 | 0.3234 | 1.065 | C B A F |
| C | 21 | 0.3161 | 0.3352 | 1.060 | B A F |
| B | 22 | 0.3768 | 0.3892 | 1.033 | A F |
| A | 22 | 0.3768 | 0.3898 | 1.035 | F |
| F | 23 | 0.5474 | 0.5495 | **1.004** | none |

F elutes last, with the column otherwise empty, and matches to 0.4 %. Every
component that still has others bound when it elutes is 3-6.5 % off, and the
deviation grows with the amount still bound. This is the signature of the
capacity term, and of nothing else: a change in Lambda, in v_int, in the
porosities or in the film transfer would not switch itself off for F.

The magnitude is consistent too. The injection is 1.00 column volumes, so the
column-average loading is sum_j (nu_j+sigma_j) q_j = 0.1165 mol/L against
Lambda = 0.3247 mol/L, i.e. 36 % ligand occupancy, versus
sum_j nu_j q_j = 0.1018 mol/L. The two readings of qbar then differ by
(0.3247-0.1018)/(0.3247-0.1165) = 1.071 with everything bound, falling to 1.000
once only one component is left -- which brackets the observed 1.004 to 1.065.

**Why this looks like a paper/code mismatch rather than an intended model.**
`q_s = Lambda - sum_j nu_j q_j` is exactly the bound-salt state that the paper
already integrates in Eq. (5), so using it in the denominator of Eq. (4) is a
natural substitution to make when the quantity is already at hand. And under
that reading sigma_i appears *nowhere* in the model, which would make the
sigma_i = 3 of Table 4 inert -- self-inconsistent for a reported parameter.
Note also that Eq. (4) as printed is the classical Brooks & Cramer form and
matches CADET's SMA term for term.

### 5.2 A uniform +216 s offset

Independently of the isotherm, the whole published chromatogram is later than
the stated inlet program plus the column hold-up implied by Table 3 can
produce, by a strikingly uniform amount.

Salt is the cleanest probe because it is unretained: with no protein bound,
Eq. (5) gives dq_s/dt = 0, so salt is delayed only by the mobile and pore
hold-up, independently of the isotherm and of k_MT. Table 3 gives

    (L/v_int) * (1 + (1-eps)*eps_p/eps) = 133.2 s * 2.1237 = 283 s

whereas Figure 1 shows 499 s. Measuring the offset feature by feature:

| feature | offset |
|---|---|
| phase-3 ramp, at salt = 0.06 / 0.10 / 0.15 / 0.20 mol/L | +214 / +216 / +216 / +216 s |
| end of ramp, salt = 0.235 mol/L | +224 s |
| phase-5 gradient, at salt = 0.25 / 0.30 / 0.40 / 0.50 / 0.60 mol/L | +216 / +216 / +216 / +216 / +216 s |
| phase-6 step, at salt = 0.84 / 1.00 mol/L | +210 / +241 s |
| protein peak maxima A / B / C / D / E / F | +220 / +210 / +215 / +227 / +214 / +219 s |

That is +216 s to within a few seconds across twelve salt features and all six
peaks, i.e. a pure dead time rather than any kind of model error: the shape,
the gradient slope and every phase duration are reproduced correctly, only the
clock is shifted. The gap cannot be column hold-up -- closing it would require
a non-flowing accessible volume fraction above 1.0 -- and the paper reports no
extra-column volume, hold-up volume or equilibration phase that would account
for it. Figure 1 also extends to t = 10440 s while the program of Eq. (38)
totals 10260 s, a further 180 s that is unexplained.

Because no number in the paper supports it, the 216 s is applied to the
*reference data only* (`FIGURE_TIME_OFFSET` in `Meyer2026_fig1.py`). The CADET
input runs on the paper's stated program unchanged and contains no fitted
quantity.

===========================================================================
6. Questions to the authors
===========================================================================
Reproducing Section 4.2 from Tables 3-4 and Eqs. (35)-(40) gives a
chromatogram that differs from Figure 1 in two specific ways. Both were
isolated with two independent codes (CADET, and a numpy/scipy solver of
Eqs. (1),(2),(4),(5) written directly from the paper), which agree with each
other to 6 s in peak time and 0.7 % in peak height.

1. **Which quantity is in the denominator of Eq. (4)?** As printed,
   `Lambda - sum_j (nu_j+sigma_j) q_j`, the six peaks are 253-569 s early and
   up to 18 % low. Using instead the bound-salt concentration
   `q_s = Lambda - sum_j nu_j q_j` of Eq. (5) reproduces Figure 1 to 0.13 %
   mean NRMSE and 0.5 % in every peak height. Was Figure 1 produced with q_s
   in the denominator? If so, is sigma_i = 3 (Table 4) used anywhere in the
   model, since it would otherwise not enter at all?

   To check: `python Meyer2026_fig1_paper_equations.py --denominator paper_eq4`
   versus `--denominator bound_salt`. The script needs only numpy and scipy,
   and the two runs differ in nothing but that term.

2. **What accounts for the uniform 216 s delay of Figure 1?** Every
   salt-program transition and every protein peak in Figure 1 is 216 +/- 7 s
   later than Eqs. (35)-(40) plus the 283 s outlet hold-up implied by Table 3.
   Since salt is unretained here, this is isotherm independent. Is there an
   extra-column or hold-up volume, or an equilibration phase preceding the
   program, that is not listed in Section 4.2? Relatedly, Figure 1 extends to
   10440 s while the durations of Eq. (38) sum to 10260 s.

Neither point affects the paper's numerical results: the convergence studies,
work-precision sweeps and DFSA cost scaling are all insensitive to which of
the two isotherm readings is used, and to any uniform time shift.
