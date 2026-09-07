# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 6 from:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

Self-contained script: model definition, run, comparison plot, and
validation metrics.

===========================================================================
Step 0 -- case identification
===========================================================================
Fig. 6 (PDF/printed p. 43, Sec. 4.2.2 "Isocratic elution") shows the
EXPERIMENTAL isocratic elution peak profile of n-valerophenone recorded on
three column configurations, all packed with the same batch of 5-micron
XBridge-C18 particles, mobile phase acetonitrile/water 75/25 (v/v), 27 C:

  1) "Cylinder rho_s=1"  : conventional cylindrical column, r_e=1.50 mm
                            (3.0 mm i.d.) x 150 mm, Fv=0.35 mL/min.
  2) "Cone rho_s=2"      : conical column, r_e=1.05 mm entrance (2.1 mm
                            i.d.) widening to 4.2 mm i.d. exit, x 150 mm,
                            Fv=0.40 mL/min (flow: narrow -> wide end).
  3) "Cone rho_s=0.5"    : the SAME physical conical tube, reversed flow
                            direction (entrance r_e=2.1 mm i.d., exit 2.1
                            mm/2=1.05 mm i.d., i.e. wide -> narrow end),
                            same Fv=0.40 mL/min.

Table 1 (p. 44) tabulates, for valerophenone specifically, the measured
retention time, zeroth/first/second moments, half-height width and both
efficiency estimates for exactly these three configurations -- this gives
an unambiguous, purely numerical validation target (first and second
central moments) IN ADDITION to the digitized Fig. 6 curve itself:

    Config      Fv[mL/min]  t_R[min]  mu1[min]  mu2'[min^2]  w1/2[min]
    Cylinder    0.35        3.865     3.869      0.00156      0.0718
    Cone s=2    0.40        3.898     3.900      0.00114      0.0800
    Cone s=0.5  0.40        3.915     3.917      0.00113      0.0792

Target for validation: Fig. 6 (isocratic), NOT Fig. 7 (the analogous
gradient-elution figure) -- Fig. 6 is the figure named in the task.

===========================================================================
Step 1 -- model mapping to CADET
===========================================================================
The paper's OWN theoretical treatment (Sec. 2, "Theory") is not a
transport PDE with separate film/pore-diffusion resistances -- it is
Giddings' classical treatment of band broadening expressed purely through
(i) a total column porosity epsilon_t, (ii) a retention factor k, and
(iii) a *local plate height* H(xi) that lumps ALL non-idealities (eddy
dispersion, longitudinal diffusion, film and pore mass-transfer
resistance) into one coefficient integrated along the column (Eqs.
11-26). No particle porosity, film mass-transfer coefficient, or pore
diffusivity is ever given or needed in the paper's own model.

This maps onto CADET's axial-dispersion + local-
equilibrium linear-isotherm model: COLUMN_MODEL_1D with NPARTYPE=1,
particle_type_000/HAS_FILM_DIFFUSION=0, ADSORPTION_MODEL=LINEAR with IS_KINETIC=0
(instantaneous local equilibrium, matching the paper's constant retention
factor k). The column geometry is the paper's own physical geometry,
mapped onto CADET's NATIVE geometries as instructed:

    Cylinder column   -> GEOMETRY='AXIAL_FLOW_CYLINDER'
    Both cone columns -> GEOMETRY='AXIAL_FLOW_FRUSTUM' (same physical tube,
                          same CROSS_SECTION_AREA_SMALL_END/_LARGE_END for
                          both rho_s=2 and rho_s=0.5 -- only FORWARD_FLOW
                          differs between the two flow directions)

--- Reproducing the paper's REAL, flow-dependent local plate height H(v)
    (Fig. 5) via a NEW CADET-Core parameter dependency, 'VAN_DEEMTER' ---

Fig. 5 (p. 42) gives the actual measured H(xi) for valerophenone along the
2.1/4.2 mm i.d. conical column (i.e. exactly the geometry used for
cone_rho_s=0.5 here) -- a genuine van-Deemter-shaped curve, decreasing
from ~10.5 micron at xi=0 to a minimum ~9.5 micron around xi=0.4-0.6, then
rising to ~11.6-11.7 micron at xi=1. Reproducing this requires the LOCAL
axial dispersion coefficient to satisfy D_ax(v) = H(v)*v/2 for a 
non-monomial H(v) = A + B/v + C*v (van Deemter form:
A/2 = longitudinal-diffusion-independent term, B = B-term prefactor,
C = C-term prefactor) -- i.e. D_ax(v) = (A*v + B + C*v^2)/2, a
quadratic in v.

===========================================================================
Step 2 -- reparameterization (paper's notation -> CADET parameters)
===========================================================================
Paper symbols: r_e, r_s (entrance/exit radii), s=r_s/r_e, L (column
length), k (retention factor), H(v) (local plate height as a function of
local velocity), epsilon_t (total porosity, "et").

epsilon_t is NOT given directly for the real experimental columns (only
"assumed 65%" for the separate, purely theoretical Sec. 4.1 calculations)
-- but it can be derived exactly from data actually reported for these
specific columns: the cylindrical column's bed volume (V_bed=1.06 cm^3,
matching pi*r_e^2*L for r_e=1.5 mm, L=15 cm to 4 digits -- confirms r_e),
its flow rate (0.35 mL/min), the measured first moment of valerophenone
on it (mu_1=3.869 min, Table 1), and its retention factor (k=1.08, p.
43). From mu_1 = t_0*(1+k):

    t_0   = mu_1 / (1+k)                              [void time]
    et    = t_0 * Fv / V_bed                          [total porosity]
    K_eq  = k * et / (1 - et)                         [LINEAR ka/kd, kd=1]

H(v) = A + B/v + C*v is obtained by:
  1. Digitizing Fig. 5's solid curve (the fitted valerophenone H(xi)) at
     600 DPI -- see `case_study_gritti_fig6_fig5H_digitized.csv`
     (1749 points; axis-tick calibration residuals <0.015 units; title/
     legend regions explicitly excluded; a handful of stray misclassified
     pixels removed by a rolling-median outlier filter, <0.6 micron
     threshold).
  2. Converting xi -> local INTERSTITIAL velocity v(xi) for the exact
     column Fig. 5 was measured on (r_e=2.1 mm, s=0.5, Fv=0.40 mL/min,
     divided by the same total porosity derived above).
  3. Fitting A, B, C by nonlinear least squares (`scipy.optimize.curve_fit`)
     to H(v) = A + B/v + C*v: RMSE=0.071 micron, max abs. error=0.39
     micron (over a 9.5-12 micron range) -- see
     `VD_A, VD_B, VD_C` below.

This same (et, K_eq, H(v)) triple is applied, UNCHANGED, to all three
configurations (paper's own stated assumption, Sec. 4.1: "All columns are
packed identically with the same particles... and have the same external
porosity"; H(v) is a velocity-only relationship, with no further xi- or
geometry-dependence, exactly as measured). Column geometry (radii,
length, flow rate, injection volume) is otherwise the real, physical,
reported geometry -- no other free/fitted parameters are introduced.

===========================================================================
Step 3 -- extracted parameters (all traceable to a specific paper location)
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
                                              superseded in THIS version
                                              by the real H(v) below, kept
                                              only for reference/context)
  H_bar_paper_full = 11.6 micron, 18.1% loss (p. 43; the paper's full,
                                              flow-dependent-H result --
                                              THIS version's primary
                                              target)
  mu_1, mu_2' (Table 1, p. 44)  -- see table above; used for et and as a
                                    primary quantitative validation target.

No parameter here required unit conversion beyond the trivial mm/micron/
mL/min -> SI (m, m^2/s, m^3/s) conversions, applied explicitly in code
below. No parameter was ambiguous, missing, or unit-less.

===========================================================================
Step 4 -- implementation, run, and validation
===========================================================================
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))
INSTALL_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
DIGITIZED_CSV = os.path.join(HERE, 'case_study_gritti_fig6_digitized.csv')
FIG5_DIGITIZED_CSV = os.path.join(HERE, 'case_study_gritti_fig6_fig5H_digitized.csv')

# ---------------------------------------------------------------------------
# Step 3: paper parameters (SI units; conversions shown explicitly)
# ---------------------------------------------------------------------------
MM = 1e-3
MICRON = 1e-6
ML_MIN = 1e-6 / 60.0   # 1 mL/min -> m^3/s
MIN = 60.0             # 1 min -> s

L_BED = 0.15                     # m, column length (all configurations)
R_CYL = 1.50 * MM                 # m, cylindrical column radius (3.0 mm i.d.)
R_SMALL = 1.05 * MM               # m, conical column small-end radius (2.1 mm i.d.)
R_LARGE = 2.10 * MM               # m, conical column large-end radius (4.2 mm i.d.)
DP = 5.0 * MICRON                 # m, particle diameter (XBridge-C18)

K_RET = 1.08                      # valerophenone retention factor (p. 43)
H_BAR_PAPER_UNIFORM = 10.8 * MICRON   # m, paper's own "H uniform" cross-check (p. 43;
                                       # superseded here, kept for reference)
H_BAR_PAPER_FULL = 11.6 * MICRON      # m, paper's full (flow-dependent H) value (p. 43) --
                                       # THIS version's primary target.

V_INJ = 0.5e-9                    # m^3 (0.5 microL), Sec. 3.4.2

# Table 1 (p. 44), valerophenone, isocratic -- ground-truth validation data
TABLE1 = {
    'cylinder': dict(s=1.0, Fv=0.35 * ML_MIN, tR=3.865 * MIN, mu1=3.869 * MIN,
                      mu2=0.00156 * MIN ** 2, w50=0.0718 * MIN, N12=16090, Nmom=9596),
    'cone_s2':  dict(s=2.0, Fv=0.40 * ML_MIN, tR=3.898 * MIN, mu1=3.900 * MIN,
                      mu2=0.00114 * MIN ** 2, w50=0.0800 * MIN, N12=13181, Nmom=13342),
    'cone_s05': dict(s=0.5, Fv=0.40 * ML_MIN, tR=3.915 * MIN, mu1=3.917 * MIN,
                      mu2=0.00113 * MIN ** 2, w50=0.0792 * MIN, N12=13563, Nmom=13635),
}

# ---------------------------------------------------------------------------
# Step 2: reparameterization -- derive total porosity & equilibrium constant
# from the cylindrical column's own reported bed volume/flow rate/moment/k
# ---------------------------------------------------------------------------
V_BED_CYL = np.pi * R_CYL ** 2 * L_BED               # m^3; matches paper's "1.06 cm^3"
T0_CYL = TABLE1['cylinder']['mu1'] / (1.0 + K_RET)    # s, void time of the cylinder column
ET = T0_CYL * TABLE1['cylinder']['Fv'] / V_BED_CYL    # total porosity (dimensionless)
KEQ = K_RET * ET / (1.0 - ET)                         # LINEAR isotherm ka/kd (kd=1)

V_BED_CONE = np.pi / 3.0 * L_BED * (R_SMALL ** 2 + R_SMALL * R_LARGE + R_LARGE ** 2)  # matches "1.21 cm^3"

# Van Deemter fit to Fig. 5 (H(v) = A + B/v + C*v), derived once (see Step 2
# docstring above and the digitization/fit procedure) and hardcoded here for
# a fully self-contained script; case_study_gritti_fig6_fig5H_digitized.csv
# is kept alongside as the supporting/traceable raw digitized data.
VD_A = 3.15452704e-06   # m
VD_B = 4.52688414e-09   # m^2/s
VD_C = 2.23971679e-03   # s

CONFIGS = {
    'cylinder': dict(
        geometry='AXIAL_FLOW_CYLINDER',
        Fv=TABLE1['cylinder']['Fv'],
        forward_flow=1,
        label='Cylinder ' + r'$\rho_s=1$', color='k',
    ),
    'cone_s2': dict(
        geometry='AXIAL_FLOW_FRUSTUM',
        Fv=TABLE1['cone_s2']['Fv'],
        forward_flow=0,   # flow enters the SMALL end -> narrow-to-wide (rho_s=2)
        label='Cone ' + r'$\rho_s=2$', color='r',
    ),
    'cone_s05': dict(
        geometry='AXIAL_FLOW_FRUSTUM',
        Fv=TABLE1['cone_s05']['Fv'],
        forward_flow=1,   # flow enters the LARGE end -> wide-to-narrow (rho_s=0.5)
        label='Cone ' + r'$\rho_s=0.5$', color='b',
    ),
}


# ---------------------------------------------------------------------------
# Step 5: CADET model definition
# ---------------------------------------------------------------------------
def get_model(config_key, spatial_method='FV', ncol=16, dg_polydeg=4,
             n_points=3000, t_end=400.0, tracer=False):
    """
    Dispersion: COL_DISPERSION_DEP='VAN_DEEMTER' (see Step 1 docstring),
        with COL_DISPERSION=[1.0] (dimensionless placeholder -- the
        dependence factor (VD_A*v + VD_B + VD_C*v^2)/2 already IS the full
        physical D_ax(v) = H(v)*v/2 by construction).
    """
    cfg = CONFIGS[config_key]
    Fv = cfg['Fv']

    c = Cadet(install_path=INSTALL_PATH)
    m = c.root.input.model
    m.nunits = 3

    # --- Inlet: narrow rectangular injection pulse of duration V_inj/Fv ---
    t_inj = V_INJ / Fv
    m.unit_000.unit_type = 'INLET'
    m.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.unit_000.ncomp = 1
    m.unit_000.sec_000.const_coeff = [1.0]
    m.unit_000.sec_000.lin_coeff = [0.0]
    m.unit_000.sec_000.quad_coeff = [0.0]
    m.unit_000.sec_000.cube_coeff = [0.0]
    m.unit_000.sec_001.const_coeff = [0.0]
    m.unit_000.sec_001.lin_coeff = [0.0]
    m.unit_000.sec_001.quad_coeff = [0.0]
    m.unit_000.sec_001.cube_coeff = [0.0]

    # --- Column ---
    col = m.unit_001
    col.unit_type = 'COLUMN_MODEL_1D'
    col.geometry = cfg['geometry']
    col.ncomp = 1
    col.bed_length = L_BED
    col.forward_flow = [cfg['forward_flow']]
    if cfg['geometry'] == 'AXIAL_FLOW_CYLINDER':
        col.cross_section_area = np.pi * R_CYL ** 2
    elif cfg['geometry'] == 'AXIAL_FLOW_FRUSTUM':
        col.cross_section_area_small_end = np.pi * R_SMALL ** 2
        col.cross_section_area_large_end = np.pi * R_LARGE ** 2
    else:
        raise ValueError(cfg['geometry'])

    col.npartype = 1
    col.total_porosity = ET
    col.col_porosity = ET  # placeholder (unused: TOTAL_POROSITY governs
                            # velocity/capacity whenever HAS_FILM_DIFFUSION=0,
                            # per axial_flow_column_1D_config.rst)
    col.col_dispersion = [1.0]
    col.col_dispersion_dep = 'VAN_DEEMTER'
    col.col_dispersion_dep_a = VD_A
    col.col_dispersion_dep_b = VD_B
    col.col_dispersion_dep_c = VD_C
    col.init_c = [0.0]

    col.discretization.use_analytic_jacobian = 1
    if spatial_method == 'DG':
        col.discretization.spatial_method = 'DG'
        col.discretization.polydeg = dg_polydeg
        col.discretization.nelem = ncol
        col.discretization.use_collocation_dg = 0
        col.dispersion_spatial_dependence_polydeg = dg_polydeg
    elif spatial_method == 'FV':
        col.discretization.spatial_method = 'FV'
        col.discretization.ncol = ncol
        col.discretization.reconstruction = 'WENO'
        col.discretization.weno.weno_order = 3
        col.discretization.weno.weno_eps = 1e-10
        col.discretization.weno.boundary_model = 0
        col.discretization.gs_type = 1
        col.discretization.max_krylov = 0
        col.discretization.max_restarts = 10
        col.discretization.schur_safety = 1e-8
    else:
        raise ValueError(spatial_method)

    # --- Particle type 000: degenerate (HAS_FILM_DIFFUSION=0,
    # HAS_PORE_DIFFUSION=0) -> Lumped Rate Model Without Pores, matching
    # the paper's own model (no separate film/pore resistance is given or
    # needed) ---
    par = col.particle_type_000
    par.par_radius = DP / 2.0
    par.par_porosity = 0.5  # placeholder (unused, see above)
    par.has_film_diffusion = 0
    par.has_pore_diffusion = 0
    par.has_surface_diffusion = 0
    par.init_cp = [0.0]
    par.init_cs = [0.0]
    par.nbound = [0 if tracer else 1]
    if tracer:
        par.adsorption_model = 'NONE'
    else:
        par.adsorption_model = 'LINEAR'
        par.adsorption.is_kinetic = 0
        par.adsorption.lin_ka = [KEQ]
        par.adsorption.lin_kd = [1.0]

    # --- Outlet ---
    m.unit_002.unit_type = 'OUTLET'
    m.unit_002.ncomp = 1

    # --- Connections (single, unchanging switch) ---
    m.connections.nswitches = 1
    m.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, Fv,
        1.0, 2.0, -1.0, -1.0, Fv,
    ]
    m.connections.switch_000.section = 0

    m.solver.gs_type = 1
    m.solver.max_krylov = 0
    m.solver.max_restarts = 10
    m.solver.schur_safety = 1e-8

    # --- return group ---
    ret = c.root.input['return']
    ret.split_components_data = 0
    ret.split_ports_data = 0
    ret.unit_000.write_solution_outlet = 0
    ret.unit_001.write_solution_outlet = 1
    ret.unit_001.write_solution_bulk = 0
    ret.unit_002.write_solution_outlet = 0

    # --- time integration ---
    slv = c.root.input.solver
    slv.consistent_init_mode = 1
    slv.nthreads = 1
    slv.sections.nsec = 2
    slv.sections.section_continuity = [0]
    slv.sections.section_times = [0.0, t_inj, t_end]
    slv.time_integrator.abstol = 1e-10
    slv.time_integrator.reltol = 1e-8
    slv.time_integrator.algtol = 1e-10
    slv.time_integrator.init_step_size = 1e-10
    slv.time_integrator.max_steps = 100000
    slv.user_solution_times = np.linspace(0.0, t_end, n_points)

    return c


def run_model(config_key, fname=None, **kwargs):

    c = get_model(config_key, **kwargs)
    c.filename = fname or os.path.join(HERE, f'case_study_gritti_fig6_{config_key}.h5')
    c.save()
    rc = c.run_simulation()
    if rc.return_code != 0:
        raise RuntimeError(f"CADET failed for {config_key}: {getattr(rc, 'error_message', rc)}")
    c.load_from_file()
    t = np.asarray(c.root.output.solution.solution_times)
    outlet = np.asarray(c.root.output.solution.unit_001.solution_outlet).reshape(-1)
    inlet = np.asarray(c.root.output.solution.unit_000.solution_outlet).reshape(-1) \
        if 'unit_000' in c.root.output.solution else None
    return t, outlet, inlet


# ---------------------------------------------------------------------------
# Step 4: reference (digitized) data
# ---------------------------------------------------------------------------
def load_digitized():
    data = np.genfromtxt(DIGITIZED_CSV, delimiter=',', names=True)
    return data


# ---------------------------------------------------------------------------
# Step 6: validation metrics
# ---------------------------------------------------------------------------
def moments(t, c):
    """Zeroth, first (mean), and second central moment of a chromatogram."""
    m0 = np.trapz(c, t)
    m1 = np.trapz(t * c, t) / m0
    m2 = np.trapz((t - m1) ** 2 * c, t) / m0
    return m0, m1, m2


def compute_metrics(config_key, t_sim, c_sim, t_inj_duration, c_inj_area, ref_t, ref_c):
    ref = TABLE1[config_key]
    m = {}

    # 1) Peak position
    i_peak = np.argmax(c_sim)
    t_peak_sim = t_sim[i_peak]
    m['peak_time_sim'] = t_peak_sim
    m['peak_time_ref'] = ref['tR']
    m['peak_time_relerr_%'] = 100 * abs(t_peak_sim - ref['tR']) / ref['tR']

    # 2) Elution time (first moment)
    m0_sim, m1_sim, m2_sim = moments(t_sim, c_sim)
    m['mu1_sim'] = m1_sim
    m['mu1_ref'] = ref['mu1']
    m['mu1_relerr_%'] = 100 * abs(m1_sim - ref['mu1']) / ref['mu1']

    # (extra, not in the standard 4-metric table, but directly checks the
    # frustum/dispersion-dependence physics against the paper's own
    # full, flow-dependent-H result)
    m['mu2_sim'] = m2_sim
    m['mu2_ref_measured'] = ref['mu2']
    H_sim = L_BED * m2_sim / m1_sim ** 2
    m['H_bar_sim_micron'] = H_sim / MICRON

    # 3) Mass balance: injected mass (area under the rectangular inlet
    # pulse, known analytically as C0*t_inj) vs. integral of outlet
    m_in = c_inj_area
    m_out = m0_sim
    m['mass_balance_relerr_%'] = 100 * abs(m_out - m_in) / m_in

    # 4) Chromatogram MSE vs digitized reference, AREA-normalized (each
    # curve divided by its own zeroth moment = 1, and time-aligned by peak
    # position (paper itself "slightly adjusted" peak positions in Fig. 6
    # for display).
    valid = ~np.isnan(ref_c)
    rt = ref_t[valid]
    rc_raw = ref_c[valid]
    rc_area = np.trapz(rc_raw, rt)
    rc = rc_raw / rc_area
    t_shift = t_sim - t_peak_sim + rt[np.argmax(rc_raw)]
    c_sim_n = c_sim / m0_sim
    c_sim_interp = np.interp(rt, t_shift, c_sim_n, left=0.0, right=0.0)
    m['mse'] = float(np.mean((c_sim_interp - rc) ** 2))

    return m


def print_metrics(config_key, m):
    print(f"\n--- {config_key} ---")
    print(f"  Peak position   : sim={m['peak_time_sim']:.4f} s  ref(tR, Table 1)={m['peak_time_ref']:.4f} s"
          f"  rel.err={m['peak_time_relerr_%']:.3g}%")
    print(f"  Elution time (mu1): sim={m['mu1_sim']:.4f} s  ref(Table 1)={m['mu1_ref']:.4f} s"
          f"  rel.err={m['mu1_relerr_%']:.3g}%")
    print(f"  Mass balance    : rel.err={m['mass_balance_relerr_%']:.3g}%")
    print(f"  Chromatogram MSE (area-normalized, peak-aligned): {m['mse']:.4g}")
    print(f"  [extra] H_bar from sim moments: {m['H_bar_sim_micron']:.3f} micron"
          f"   (mu2_sim={m['mu2_sim']:.6g} s^2 vs Table-1 measured mu2'={m['mu2_ref_measured']:.6g} s^2)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Derived parameters (Step 2):")
    print(f"  V_bed (cylinder) = {V_BED_CYL*1e6:.4f} cm^3  (paper: 1.06 cm^3)")
    print(f"  V_bed (cone)     = {V_BED_CONE*1e6:.4f} cm^3  (paper: 1.21 cm^3)")
    print(f"  t0 (cylinder)    = {T0_CYL:.4f} s = {T0_CYL/MIN:.4f} min")
    print(f"  total porosity epsilon_t = {ET:.4f}")
    print(f"  LINEAR K_eq (ka, kd=1)   = {KEQ:.4f}")
    print(f"  Van Deemter H(v)=A+B/v+C*v : A={VD_A:.4e} m, B={VD_B:.4e} m^2/s, C={VD_C:.4e} s")
    print()

    digitized = load_digitized()
    ref_time = digitized['time_s']
    ref_cols = {'cylinder': 'cylinder_black', 'cone_s2': 'cone_s2_red', 'cone_s05': 'cone_s05_blue'}

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    all_metrics = {}
    sim_results = {}

    spatial_method = 'DG'

    for i, key in enumerate(['cylinder', 'cone_s2', 'cone_s05']):

        fig, ax = plt.subplots(figsize=(8, 6))

        cfg = CONFIGS[key]
        print(f"\nRunning CADET (DG, POLYDEG=4, NELEM=128) for configuration '{key}' "
              f"({cfg['geometry']}, Fv={cfg['Fv']/ML_MIN:.2f} mL/min, "
              f"forward_flow={cfg['forward_flow']})...")
        t_sim, c_sim, c_inlet = run_model(key, spatial_method=spatial_method, dg_polydeg=4, ncol=128,
                                           t_end=400.0, n_points=4000)
        t_inj_duration = V_INJ / cfg['Fv']
        c_inj_area = 1.0 * t_inj_duration  # C0=1 * pulse duration (analytic inlet integral)
        sim_results[key] = (t_sim, c_sim)

        ref_c = digitized[ref_cols[key]]
        m = compute_metrics(key, t_sim, c_sim, t_inj_duration, c_inj_area, ref_time, ref_c)
        all_metrics[key] = m
        print_metrics(key, m)

        # plot: CADET curve AREA-normalized (own zeroth moment = 1 and time-shifted
        # to align with the digitized peak (paper itself display-shifted the peaks)
        i_peak = np.argmax(c_sim)
        valid = ~np.isnan(ref_c)
        rt = ref_time[valid]
        rc_raw = ref_c[valid]
        rc = rc_raw / np.trapz(rc_raw, rt)
        t_shift = t_sim - t_sim[i_peak] + rt[np.argmax(rc_raw)]
        m0_sim = np.trapz(c_sim, t_sim)
        ax.plot(t_shift, c_sim / m0_sim, '-', color=cfg['color'], lw=1.5,
                label=f"{cfg['label']} (CADET)")
        ax.plot(rt, rc, 'o', color=cfg['color'], ms=2.5, mfc='none', mew=0.7,
                label=f"{cfg['label']} (digitized)")

        ax.set_xlabel('Time [s] (digitized-figure axis; CADET curves peak-aligned, see Step 4)')
        ax.set_ylabel('Area-normalized signal (each curve: ' + r'$\int c\,dt=1$' + ')')
        # ax.set_title("Gritti et al. (2019), Fig. 6 -- valerophenone, isocratic elution\n"
        #               "cylindrical vs. conical (frustum) column, both flow directions\n"
        #               "(CADET native COLUMN_MODEL_1D, GEOMETRY=AXIAL_FLOW_CYLINDER/FRUSTUM,\n"
        #               "real flow-dependent H(v) from Fig. 5 via COL_DISPERSION_DEP=VAN_DEEMTER)",
        #               fontsize=12)
        ax.legend(fontsize=12, ncol=1, loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim(215.0, 250.0)
        # add MSE metric box to plot for all three configurations
        mse = all_metrics[key]['mse']
        ax.text(0.98, 0.95,# + i*0.05,
                f"{key} MSE: {mse:.4g}", transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )

        fig.tight_layout()
        outpath = os.path.join(HERE, f'case_study_gritti_fig6_comparison_{key}_{spatial_method}.png')
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"\nSaved comparison plot to {outpath}")

if __name__ == '__main__':
    main()
