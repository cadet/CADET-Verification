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
import sys

import numpy as np
import matplotlib.pyplot as plt
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))

# The shared metric definitions live one directory up, so that all six
# validation case studies report an identical set of numbers. Adding that
# directory to sys.path keeps this script runnable both directly and as an
# imported module (scripts/verify_geometries.py imports main()).
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
import validation_metrics as vm  # noqa: E402

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
def compute_metrics(config_key, t_sim, c_sim, c_inj_area, ref_t, ref_c):
    """The four unified validation metrics -- see src/validation/validation_metrics.py
    for their definitions, which are shared verbatim by all six case studies.

    The reference for Delta mu_1 and Delta mu_2 is the paper's own Table 1,
    i.e. moments measured on the real columns. The dispersion coefficient
    used here is the paper's Fig. 5 H(v) curve and was NOT fitted to Table
    1's mu_2, so Delta mu_2 is a genuine prediction in this figure (unlike
    in Gritti2019_fig7.py, and unlike the cylinder of Gritti2019_fig8.py,
    where the dispersion was calibrated against the tabulated mu_2 and the
    Delta mu_2 entry is consequently left empty).

    NOTE specific to this figure: Sec. 4.2.2 states that the peaks PRINTED
    in Fig. 6 were "slightly adjusted" in time for display. A few tenths of
    a percent of the NRMSE therefore reflect that known display artifact
    rather than a genuine model/shape mismatch; no analogous statement
    exists for Fig. 7/8. Delta mu_1 and Delta mu_2 are unaffected, since
    they are taken against Table 1 rather than against the digitized curve.
    """
    ref = TABLE1[config_key]
    m = vm.standard_metrics(
        name=config_key,
        t_sim=t_sim, c_sim=c_sim, t_ref=ref_t, c_ref=ref_c,
        kind=vm.PULSE,
        mu1_ref=ref['mu1'], mu2_ref=ref['mu2'], ref_label='Gritti Table 1',
        amplitude='lsq',
        mass_in=c_inj_area,
        mass_label='simulated outlet integral vs. the analytically known '
                   'injected mass C0*t_inj',
    )
    # Diagnostic (not one of the four metrics): the mean plate height implied
    # by the simulated moments, directly comparable to the paper's own
    # H_bar = 11.6 um for the full, flow-dependent-H case (p. 43).
    m['H_bar_sim_micron'] = L_BED * m['mu2_sim_full'] / m['mu1_sim_full'] ** 2 / MICRON
    return m


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
        m = compute_metrics(key, t_sim, c_sim, c_inj_area, ref_time, ref_c)
        all_metrics[key] = m

        # plot: CADET curve on its RAW time axis (no peak-realignment), scaled
        # by the same per-column least-squares AU factor used for the metrics
        # -- identical convention to Gritti2019_fig7.py's/fig8.py's plots.
        valid = ~np.isnan(ref_c)
        rt = ref_time[valid]
        rc = ref_c[valid]
        scale = m['amplitude_scale']

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


    print("Cone fwd vs bwd flow max difference: ",
          np.max(np.abs(chromatograms['cone_s2'] - chromatograms['cone_s05'])))

    metrics = [all_metrics[key] for key in ['cylinder', 'cone_s2', 'cone_s05']]
    print("\n" + "=" * 70)
    print("Validation metrics -- Gritti et al. (2019), Fig. 6 "
          "(valerophenone, isocratic)")
    print("=" * 70)
    vm.print_metrics_table(metrics, time_unit='s')
    for m in metrics:
        print(f"  [diagnostic] {m['name']:10s}: H_bar from the simulated moments = "
              f"{m['H_bar_sim_micron']:.3f} micron "
              f"(paper, full flow-dependent H: {H_BAR_PAPER_FULL / MICRON:.1f} micron)")
    print("  NOTE: Sec. 4.2.2 states the peaks PRINTED in Fig. 6 were 'slightly "
          "adjusted' in time for display, so part of the NRMSE above is that "
          "known display artifact.")
    vm.dump_metrics(output_path, 'Gritti2019_fig6',
                    'Isocratic valerophenone', metrics, time_unit='s')
    return metrics


if __name__ == '__main__':
    main()
