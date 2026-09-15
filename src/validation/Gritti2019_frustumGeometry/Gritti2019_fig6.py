# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 6 from:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

Self-contained script: model definition, run, comparison plot, and
validation metrics. Further explanation on model and parameter selection is
provided under Gritti2019_fig6.md.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))
DIGITIZED_CSV = os.path.join(HERE, 'Gritti2019_fig6_digitized.csv')
FIG5_DIGITIZED_CSV = os.path.join(HERE, 'Gritti2019_fig6_fig5H_digitized.csv')

# ---------------------------------------------------------------------------
# Step 3 parameters (SI units; conversions shown explicitly)
# ---------------------------------------------------------------------------
MM = 1e-3
MICRON = 1e-6
ML_MIN = 1e-6 / 60.0   # 1 mL/min -> m^3/s
MIN = 60.0             # 1 min -> s

L_BED = 0.15                      # m, column length (all configurations)
R_CYL = 1.50 * MM                 # m, cylindrical column radius (3.0 mm i.d.)
R_SMALL = 1.05 * MM               # m, conical column small-end radius (2.1 mm i.d.)
R_LARGE = 2.10 * MM               # m, conical column large-end radius (4.2 mm i.d.)
DP = 5.0 * MICRON                 # m, particle diameter (XBridge-C18)

K_RET = 1.08                      # valerophenone retention factor (p. 43)
H_BAR_PAPER_UNIFORM = 10.8 * MICRON   # m, paper's own "H uniform" cross-check (p. 43;
                                       # superseded here by the real H(v), kept for reference)
H_BAR_PAPER_FULL = 11.6 * MICRON      # m, paper's full (flow-dependent H) value (p. 43),
                                       # this script's primary target

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
# Step 2 -- reparameterization: total porosity & equilibrium constant from
# the cylindrical column's own reported bed volume/flow rate/moment/k
# ---------------------------------------------------------------------------
V_BED_CYL = np.pi * R_CYL ** 2 * L_BED               # m^3; matches paper's "1.06 cm^3"
T0_CYL = TABLE1['cylinder']['mu1'] / (1.0 + K_RET)    # s, void time of the cylinder column
ET = T0_CYL * TABLE1['cylinder']['Fv'] / V_BED_CYL    # total porosity (dimensionless)
KEQ = K_RET * ET / (1.0 - ET)                         # LINEAR isotherm ka/kd (kd=1)

V_BED_CONE = np.pi / 3.0 * L_BED * (R_SMALL ** 2 + R_SMALL * R_LARGE + R_LARGE ** 2)  # matches "1.21 cm^3"

# Van Deemter fit to Gritti et al. Fig. 5, H(v) = A + B/v + C*v (see Step 2 in the
# module docstring for the digitization/fit procedure); raw digitized data in
# Gritti2019_fig6_fig5H_digitized.csv.
VD_A = 3.15452704e-06   # m
VD_B = 4.52688414e-09   # m^2/s
VD_C = 2.23971679e-03   # s

CONFIGS = {
    'cylinder': dict(
        geometry='AXIAL_FLOW_FRUSTUM',
        Fv=TABLE1['cylinder']['Fv'],
        forward_flow=1,
        label=r'$\rho_s=1$', color='k',
    ),
    'cone_s2': dict(
        geometry='AXIAL_FLOW_FRUSTUM',
        Fv=TABLE1['cone_s2']['Fv'],
        forward_flow=0,   # flow enters the SMALL end -> narrow-to-wide (rho_s=2)
        label=r'$\rho_s=2$', color='red',
    ),
    'cone_s05': dict(
        geometry='AXIAL_FLOW_FRUSTUM',
        Fv=TABLE1['cone_s05']['Fv'],
        forward_flow=1,   # flow enters the LARGE end -> wide-to-narrow (rho_s=0.5)
        label=r'$\rho_s=0.5$', color='blue',
    ),
}


# ---------------------------------------------------------------------------
# Step 4 -- CADET model definition
# ---------------------------------------------------------------------------
def get_model(cadet_path, config_key, spatial_method='FV', ncol=16, dg_polydeg=4,
              n_points=3000, t_end=400.0, tracer=False):
    """Build the CADET model for one column configuration.

    Dispersion: COL_DISPERSION_DEP='VAN_DEEMTER' (see Step 1 in the module
    docstring), with COL_DISPERSION=[1.0] (dimensionless placeholder --
    the dependence factor (VD_A*v + VD_B + VD_C*v^2)/2 already IS the full
    physical D_ax(v) = H(v)*v/2 by construction).
    """
    cfg = CONFIGS[config_key]
    Fv = cfg['Fv']

    c = Cadet(install_path=cadet_path)
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
    if cfg['label'] == r'$\rho_s=1$':
        col.cross_section_area_small_end = np.pi * R_CYL ** 2
        col.cross_section_area_large_end = np.pi * R_CYL ** 2
    elif cfg['label'] in [r'$\rho_s=2$', r'$\rho_s=0.5$']:
        col.cross_section_area_small_end = np.pi * R_SMALL ** 2
        col.cross_section_area_large_end = np.pi * R_LARGE ** 2
    else:
        raise ValueError(cfg['label'])

    col.npartype = 1
    col.total_porosity = ET
    col.col_porosity = ET   # unused (TOTAL_POROSITY governs velocity/capacity
                            # whenever HAS_FILM_DIFFUSION=0, per
                            # axial_flow_column_1D_config.rst), set for completeness
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
        raise ValueError(f"Unsupported spatial method: {spatial_method}")

    # --- Particle type 000: LRM (no film/pore diffusion -- see Step 1) ---
    par = col.particle_type_000
    par.par_radius = DP / 2.0
    par.par_porosity = 0.5  # unused (LRM does not require explicit particle porosity)
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


def run_model(cadet_path, output_path, config_key, fname=None, **kwargs):

    c = get_model(cadet_path, config_key, **kwargs)
    c.filename = fname or os.path.join(output_path, f'Gritti2019_fig6_{config_key}.h5')
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
# Reference (digitized) data
# ---------------------------------------------------------------------------
def load_digitized():
    data = np.genfromtxt(DIGITIZED_CSV, delimiter=',', names=True)
    return data


def plot_fig5_verification(output_path):
    """Verification plot for the digitized Gritti et al. (2019) Fig. 5 plate-height
    data (Gritti2019_fig6_fig5H_digitized.csv) and the VAN_DEEMTER fit
    (VD_A, VD_B, VD_C) derived from it and used throughout this script.

    Reproduces Fig. 5's own axes (H [um] vs. xi=z/L) for a direct visual
    side-by-side comparison against the paper figure: overlaying the
    digitized points on a pixel-registered re-render of Fig. 5 confirms the
    digitized curve matches the paper's solid valerophenone curve throughout
    xi in [0, 1].
    """
    fig5 = np.genfromtxt(FIG5_DIGITIZED_CSV, delimiter=',', names=True)
    xi = fig5['xi']
    H_digitized = fig5['H_um']

    # Local interstitial velocity along the SPECIFIC column Fig. 5 was measured
    # on: r_e=2.1 mm, s=0.5 (wide r_e to narrow r_s=s*r_e), Fv=0.40 mL/min,
    # divided by the total porosity ET derived in Step 2 (see module docstring).
    r_e_fig5 = 2.10 * MM
    s_fig5 = 0.5
    Fv_fig5 = 0.40 * ML_MIN
    r_xi = r_e_fig5 * (1.0 + (s_fig5 - 1.0) * xi)
    v_interstitial = Fv_fig5 / (np.pi * r_xi ** 2) / ET

    H_fit = (VD_A + VD_B / v_interstitial + VD_C * v_interstitial) / MICRON

    resid = H_fit - H_digitized
    rmse = np.sqrt(np.mean(resid ** 2))
    maxabs = np.max(np.abs(resid))
    print(f"Fig. 5 digitization vs. VAN_DEEMTER fit: RMSE={rmse:.4f} um, max|resid|={maxabs:.4f} um "
          f"(digitized H range {H_digitized.min():.2f}-{H_digitized.max():.2f} um, n={len(xi)} points)")

    fontsize = 15
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(xi, H_digitized, '.', ms=3, color='tab:blue', alpha=0.4,
            label='Digitized (this work), Fig. 5 solid curve')
    order = np.argsort(xi)
    ax.plot(xi[order], H_fit[order], '-', color='tab:orange', lw=2,
            label='VAN_DEEMTER fit used in CADET (VD_A, VD_B, VD_C)')
    ax.axhline(9.50, ls='--', color='tab:cyan',
               label='Cylinder @ 0.35 mL/min (H=9.50 um, paper text p.43)')
    ax.set_xlabel(r'Dimensionless axial position $\xi = z/L$', fontsize=fontsize)
    ax.set_ylabel(r'Local plate height $H$ [$\mu$m]', fontsize=fontsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 14)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_title('Verification: Gritti et al. (2019) Fig. 5 digitization vs. CADET VAN_DEEMTER fit')
    ax.legend(loc='lower right', fontsize=fontsize)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.02, f'RMSE={rmse:.3f} um\nmax|resid|={maxabs:.3f} um', transform=ax.transAxes,
            fontsize=fontsize, va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.tight_layout()
    outpath = os.path.join(output_path, 'Gritti2019_fig6_fig5H_verification.png')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved Fig. 5 digitization/fit verification plot to {outpath}")


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------
def moments(t, c):
    """Zeroth, first (mean), and second central moment of a chromatogram.
    Negative values (small DG undershoot/ringing near baseline) are clipped
    before integrating -- same convention as Gritti2019_fig7.py's and
    Gritti2019_fig8.py's moments()."""
    c = np.clip(np.asarray(c), 0.0, None)
    m0 = np.trapz(c, t)
    m1 = np.trapz(t * c, t) / m0
    m2 = np.trapz((t - m1) ** 2 * c, t) / m0
    return m0, m1, m2


def compute_metrics(config_key, t_sim, c_sim, t_inj_duration, c_inj_area, ref_t, ref_c):
    """Identical metric set/formulas as Gritti2019_fig7.py's/fig8.py's
    compute_metrics(): peak position and elution time are each checked
    against BOTH the tabulated (Table 1) ground truth AND the digitized
    curve; amplitude is a per-column least-squares AU-scale fit (NOT area
    normalization); chromatogram NRMSE is computed on the RAW (unshifted)
    time axis, with no peak-realignment step -- see the NOTE printed by
    print_metrics() for the one known, paper-documented caveat specific to
    this figure (Fig. 6's own printed peaks were intentionally, cosmetically
    time-shifted by the paper's authors for display -- Sec. 4.2.2: "for the
    sake of comparison, the time position of the three peaks were slightly
    adjusted"; no analogous statement exists for Fig. 7/8, confirmed by
    checking the paper text directly)."""
    ref = TABLE1[config_key]
    m = {}

    valid = ~np.isnan(ref_c)
    rt = ref_t[valid]
    rc = ref_c[valid]

    # 1) Peak position: sim (raw) vs. Table 1 (ground truth) AND vs. digitized curve
    i_peak_sim = np.argmax(c_sim)
    t_peak_sim = t_sim[i_peak_sim]
    i_peak_ref = np.argmax(rc)
    t_peak_ref_digitized = rt[i_peak_ref]
    m['peak_time_sim'] = t_peak_sim
    m['peak_time_table_ref'] = ref['tR']
    m['peak_time_table_relerr_%'] = 100 * abs(t_peak_sim - ref['tR']) / ref['tR']
    m['peak_time_digitized_ref'] = t_peak_ref_digitized
    m['peak_time_digitized_relerr_%'] = 100 * abs(t_peak_sim - t_peak_ref_digitized) / t_peak_ref_digitized

    # 2) Elution time (first moment): sim (raw) vs. Table 1 AND vs. digitized curve
    m0_sim, m1_sim, m2_sim = moments(t_sim, c_sim)
    _, m1_ref_digitized, _ = moments(rt, rc)
    m['mu1_sim'] = m1_sim
    m['mu1_table_ref'] = ref['mu1']
    m['mu1_table_relerr_%'] = 100 * abs(m1_sim - ref['mu1']) / ref['mu1']
    m['mu1_digitized_ref'] = m1_ref_digitized
    m['mu1_digitized_relerr_%'] = 100 * abs(m1_sim - m1_ref_digitized) / m1_ref_digitized

    # (extra, not in the standard 4-metric table, but directly checks the
    # frustum/dispersion-dependence physics against the paper's own
    # full, flow-dependent-H result)
    m['mu2_sim'] = m2_sim
    m['mu2_ref_measured'] = ref['mu2']
    H_sim = L_BED * m2_sim / m1_sim ** 2
    m['H_bar_sim_micron'] = H_sim / MICRON

    # 3) Mass balance: injected mass (area under the rectangular inlet
    # pulse, known analytically as C0*t_inj) vs. integral of the RAW
    # (un-amplitude-calibrated) simulated outlet.
    m_in = c_inj_area
    m_out = m0_sim
    m['mass_balance_relerr_%'] = 100 * abs(m_out - m_in) / m_in

    # 4) Per-column least-squares Absorbance-[AU] scale factor (simulated ->
    # digitized), fit independently for this column -- identical formula to
    # Gritti2019_fig7.py's/fig8.py's amplitude calibration.
    c_sim_i = np.interp(rt, t_sim, c_sim)
    denom = np.sum(c_sim_i ** 2)
    scale = np.sum(c_sim_i * rc) / denom if denom > 0 else 0.0
    m['au_scale'] = scale
    m['peak_height_sim'] = scale * c_sim[i_peak_sim]
    m['peak_height_ref'] = rc[i_peak_ref]
    m['peak_height_relerr_%'] = 100 * abs(m['peak_height_sim'] - m['peak_height_ref']) / m['peak_height_ref']

    # 5) Chromatogram MSE/NRMSE, RAW time axis, NO peak-realignment -- see
    # docstring/print_metrics() NOTE for the Fig.-6-specific caveat this
    # implies (this figure's own digitized reference has a known,
    # paper-documented cosmetic peak-position shift that Fig. 7/8 do not).
    m['mse'] = float(np.mean((scale * c_sim_i - rc) ** 2))
    m['nrmse_%'] = 100.0 * np.sqrt(m['mse']) / m['peak_height_ref']

    return m


def print_metrics(config_key, m):
    print(f"\n--- {config_key} ---")
    print(f"  Peak position    : sim={m['peak_time_sim']:.4f} s  "
          f"ref(tR, Table 1)={m['peak_time_table_ref']:.4f} s  rel.err={m['peak_time_table_relerr_%']:.3g}%  |  "
          f"ref(digitized)={m['peak_time_digitized_ref']:.4f} s  rel.err={m['peak_time_digitized_relerr_%']:.3g}%")
    print(f"  Peak height [AU] : sim={m['peak_height_sim']:.4g}  ref={m['peak_height_ref']:.4g}  "
          f"rel.err={m['peak_height_relerr_%']:.3g}%")
    print(f"  Elution time     : sim={m['mu1_sim']:.4f} s  "
          f"ref(Table 1)={m['mu1_table_ref']:.4f} s  rel.err={m['mu1_table_relerr_%']:.3g}%  |  "
          f"ref(digitized)={m['mu1_digitized_ref']:.4f} s  rel.err={m['mu1_digitized_relerr_%']:.3g}%")
    print(f"  Mass balance     : rel.err={m['mass_balance_relerr_%']:.3g}% "
          "(sim. outlet integral vs. analytically known injected mass)")
    print(f"  Chromatogram MSE [AU^2] : {m['mse']:.4g}  (NRMSE={m['nrmse_%']:.2f}% of peak height)")
    print(f"  Fitted AU scale  : {m['au_scale']:.4g} (independent least-squares fit for this column)")
    print("  NOTE: unlike Fig. 7/8, the paper's own text (Sec. 4.2.2) states this specific "
          "figure's printed peaks were 'slightly adjusted' (time-shifted) for display -- "
          "so a few tenths of a percent of the NRMSE/peak-position-vs-digitized numbers above "
          "may reflect that known display artifact rather than a genuine model/shape mismatch.")
    print(f"  [extra] H_bar from sim moments: {m['H_bar_sim_micron']:.3f} micron"
          f"   (mu2_sim={m['mu2_sim']:.6g} s^2 vs Table-1 measured mu2'={m['mu2_ref_measured']:.6g} s^2)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from pathlib import Path
CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "output" / "validation"

def main(cadet_path=CADET_PATH, output_path=OUTPUT_PATH):

    os.makedirs(output_path, exist_ok=True)

    print("Derived parameters (Step 2):")
    print(f"  V_bed (cylinder) = {V_BED_CYL*1e6:.4f} cm^3  (paper: 1.06 cm^3)")
    print(f"  V_bed (cone)     = {V_BED_CONE*1e6:.4f} cm^3  (paper: 1.21 cm^3)")
    print(f"  t0 (cylinder)    = {T0_CYL:.4f} s = {T0_CYL/MIN:.4f} min")
    print(f"  total porosity epsilon_t = {ET:.4f}")
    print(f"  LINEAR K_eq (ka, kd=1)   = {KEQ:.4f}")
    print(f"  Van Deemter H(v)=A+B/v+C*v : A={VD_A:.4e} m, B={VD_B:.4e} m^2/s, C={VD_C:.4e} s")
    print()

    plot_fig5_verification(output_path)

    digitized = load_digitized()
    ref_time = digitized['time_s']
    ref_cols = {'cylinder': 'cylinder_black', 'cone_s2': 'cone_s2_red', 'cone_s05': 'cone_s05_blue'}

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    all_metrics = {}
    sim_results = {}

    spatial_method = 'DG'
    chromatograms = {}

    for i, key in enumerate(['cylinder', 'cone_s2', 'cone_s05']):

        cfg = CONFIGS[key]
        print(f"\nRunning CADET (DG, POLYDEG=4, NELEM=128) for configuration '{key}' "
              f"({cfg['geometry']}, Fv={cfg['Fv']/ML_MIN:.2f} mL/min, "
              f"forward_flow={cfg['forward_flow']})...")
        t_sim, c_sim, c_inlet = run_model(cadet_path, output_path, key, spatial_method=spatial_method, dg_polydeg=4, ncol=128,
                                           t_end=400.0, n_points=4000)
        chromatograms[key] = c_sim
        t_inj_duration = V_INJ / cfg['Fv']
        c_inj_area = 1.0 * t_inj_duration  # C0=1 * pulse duration (analytic inlet integral)
        sim_results[key] = (t_sim, c_sim)

        ref_c = digitized[ref_cols[key]]
        m = compute_metrics(key, t_sim, c_sim, t_inj_duration, c_inj_area, ref_time, ref_c)
        all_metrics[key] = m
        print_metrics(key, m)

        # plot: CADET curve on its RAW time axis (no peak-realignment), scaled
        # by the same per-column least-squares AU factor used for the metrics
        # -- identical convention to Gritti2019_fig7.py's/fig8.py's plots.
        valid = ~np.isnan(ref_c)
        rt = ref_time[valid]
        rc = ref_c[valid]
        scale = m['au_scale']

        fig, ax = plt.subplots(figsize=(7.5, 5.8))

        ax.plot(t_sim, scale * c_sim, '-', color=cfg['color'], lw=1.5,
                label=f"{cfg['label']} (CADET)")
        ax.plot(rt, rc, 'o', color=cfg['color'], ms=2.5, mfc='none', mew=0.7,
                label=f"{cfg['label']} (Gritti 2019)")

        fontsize = 15
        ax.set_xlabel('Time [s]', fontsize=fontsize)
        ax.set_ylabel('Absorbance [AU]', fontsize=fontsize)
        # ax.set_title("Gritti et al. (2019), Fig. 6 -- valerophenone, isocratic elution\n"
        #               "cylindrical vs. conical (frustum) column, both flow directions\n",
        #               fontsize=fontsize)
        ax.legend(fontsize=fontsize, ncol=1)
        ax.grid(alpha=0.3)
        ax.set_xlim(215.0, 250.0)
        # ax.set_ylim(None, 0.2)
        # add NRMSE metric box to plot for all three configurations
        nrmse = all_metrics[key]['nrmse_%']
        ax.text(
            0.98, 0.75, f"NRMSE: {nrmse:.2f}%", transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=fontsize, bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white', alpha=0.7
            )
        )
        fig.tight_layout()
        outpath = os.path.join(output_path, f'Gritti2019_fig6_comparison_{key}_{spatial_method}.png')
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"\nSaved comparison plot to {outpath}")


    print("Cone fwd vs bwd flow max difference: ", np.max(np.abs(chromatograms['cone_s2'] - chromatograms['cone_s05'])))

if __name__ == '__main__':
    main()
