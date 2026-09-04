# Reproducing the ChromOps 6-component IEX benchmark (Section 4.2) in CADET

**Context / wrap-up** (usable as an AI prompt)

We reproduced the 6-component IEX benchmark from Section 4.2 of Meyer et al. (2026),
*"ChromOps.jl: High-order simulation and discrete forward sensitivity analysis for
chromatography models"*, using CADET's lumped-rate model with pores (`COLUMN_MODEL_1D`
with homogeneous particles) and the steric mass action (SMA) isotherm. The paper's
equations differ slightly from CADET's, but they are equivalent under the following
reparameterization of **input parameters only** (no state variables are transformed):

$$
\alpha := \frac{1-\varepsilon^p}{\varepsilon^p}, \qquad
k^f_i = \frac{\varepsilon^p\, r^p}{3}\, k_{\mathrm{MT},i}, \qquad
\Lambda_{\mathrm{CADET}} = \frac{\Lambda}{\alpha}, \qquad
k^a_i = \frac{\bar k_{a,i}}{\alpha}, \qquad
k^d_i = \frac{\bar k_{a,i}}{k_{\mathrm{eq},i}\,\alpha^{\nu_i}},
$$

together with the identification $q_i = \alpha\, c^s_i$ relating the paper's adsorbed
concentration $q_i$ (per pore-liquid volume) to CADET's solid-phase state $c^s_i$
(per solid volume), and SMA reference concentrations
$c^{\mathrm{ref}}_0 = q^{\mathrm{ref}} = \Lambda/\alpha$ to keep $k^a, k^d$ numerically
well-scaled (otherwise they are $\mathcal O(10^{-60})$ due to the $\nu_i$-th powers,
$\nu_i \approx 20$).

## Recovery of the paper's equations from CADET's

**Bulk:** inserting $k^f_i = \varepsilon^p r^p k_{\mathrm{MT},i}/3$ into CADET's film term gives

$$
\frac{1-\varepsilon^b}{\varepsilon^b}\frac{3}{r^p}k^f_i\,(c^b_i - c^p_i)
= \frac{(1-\varepsilon^b)\,\varepsilon^p}{\varepsilon^b}\,k_{\mathrm{MT},i}\,(c^b_i - c^p_i),
$$

which is exactly the paper's Eq. (1).

**Pore:** with $\partial_t q_i = \alpha\,\partial_t c^s_i$, CADET's pore balance

$$
\partial_t c^p_i + \alpha\,\partial_t c^s_i = \frac{3}{\varepsilon^p r^p}k^f_i(c^b_i - c^p_i)
$$

becomes $\partial_t c^p_i + \partial_t q_i = k_{\mathrm{MT},i}(c^b_i - c^p_i)$, the
paper's Eq. (2).

**SMA:** substituting $c^s_j = q_j/\alpha$, $\Lambda_{\mathrm{CADET}} = \Lambda/\alpha$,
$k^a_i = \bar k_{a,i}/\alpha$, $k^d_i = \bar k_{a,i}/(k_{\mathrm{eq},i}\alpha^{\nu_i})$
into CADET's law and multiplying by $\alpha$ yields, at equilibrium
($\partial_t c^s_i = 0$), exactly the paper's isotherm — Eq. (4) with driving-force
form and $k_{\mathrm{eq},i}$ = Table 4 values. The *transient* kinetic laws are
structurally different: CADET's forward rate carries the factor
$\big(\Lambda_{\mathrm{CADET}} - \sum_j (\nu_j+\sigma_j)c^s_j\big)^{\nu_i}$, which is
state-dependent, while the paper's $\bar k_{a,i}$ is constant; no constant choice of
$k^a, k^d$ can match them. Since the paper's binding rate
($\bar k_a = 10\,\mathrm{s^{-1}}$) is ~70x faster than film transfer
($k_{\mathrm{MT}} = 0.139\,\mathrm{s^{-1}}$), the solid phase is effectively at local
equilibrium, so CADET is run with **rapid-equilibrium binding** (`IS_KINETIC = 0`),
which imposes the exactly-mapped isotherm.

## Validation

1. An independent finite-volume/BDF solve of the paper's exact equations
   (`paper_reference_solver.py`) matches the reparameterized CADET simulation to
   within ~1% peak height and ~30 s peak time for all six components.
2. Against curves extracted (color-based, tick-calibrated; `extract_fig1_data.py`,
   output in `chromops_fig1_extracted/`) from the paper's Figure 1, integrated peak
   areas agree to <= 6% and shift-corrected shape NRMSE is 1-8% of peak height —
   but all figure curves elute a consistent ~430-565 s *later* than both our CADET
   solve and the paper's own stated equations/inlet program (Eqs. 35-40) predict,
   indicating a systematic time offset in the published figure (e.g. an undocumented
   equilibration phase or hold-up volume) rather than a model mismatch.

## Files

- `src/benchmark_models/setting_COL1D_NovoNordiskIEXbenchmark.py` — CADET setup with
  the reparameterization, comparison plot, and error metrics
- `paper_reference_solver.py` — independent solver of the paper's exact equations
- `extract_fig1_data.py` / `chromops_fig1_extracted/` — Figure 1 data extraction
- `comparison_cadet_vs_paper_equations.png` — overlay plot
