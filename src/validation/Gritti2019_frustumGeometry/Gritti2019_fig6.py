# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 6 from:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

The model, the source of the parameters and the van Deemter fit to the
measured plate heights are explained in detail in Gritti2019_fig6.md.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))

# The metric definitions shared by all validation case studies live one
# directory up. Adding it to sys.path keeps this script runnable both
# directly and as an import (scripts/verify_geometries.py calls main()).
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
import validation_metrics as vm  # noqa: E402

# The reference chromatograms, the measured plate heights and the script that
# builds the former all live in data/, next to the raw exports they come from.
DATA = os.path.join(HERE, 'data')
if DATA not in sys.path:
    sys.path.insert(0, DATA)
import Gritti2019_plate_height as ph  # noqa: E402

MEASURED_CSV = os.path.join(DATA, 'Gritti2019_fig6_measured.csv')

# ---------------------------------------------------------------------------
# Parameters from the paper, converted to SI here
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
H_BAR_PAPER_UNIFORM = 10.8 * MICRON   # m, the paper's uniform-H cross-check (p. 43)
H_BAR_PAPER_FULL = 11.6 * MICRON      # m, the paper's flow-dependent-H value (p. 43),
                                      # which is what this script targets

V_INJ = 0.5e-9                    # m^3 (0.5 microL), Sec. 3.4.2

# Table 1 (p. 44), valerophenone, isocratic
TABLE1 = {
    'cylinder': dict(s=1.0, Fv=0.35 * ML_MIN, tR=3.865 * MIN, mu1=3.869 * MIN,
                      mu2=0.00156 * MIN ** 2, w50=0.0718 * MIN, N12=16090, Nmom=9596),
    'cone_s2':  dict(s=2.0, Fv=0.40 * ML_MIN, tR=3.898 * MIN, mu1=3.900 * MIN,
                      mu2=0.00114 * MIN ** 2, w50=0.0800 * MIN, N12=13181, Nmom=13342),
    'cone_s05': dict(s=0.5, Fv=0.40 * ML_MIN, tR=3.915 * MIN, mu1=3.917 * MIN,
                      mu2=0.00113 * MIN ** 2, w50=0.0792 * MIN, N12=13563, Nmom=13635),
}

# ---------------------------------------------------------------------------
# Total porosity and equilibrium constant, derived from the cylindrical
# column's reported bed volume, flow rate, first moment and retention factor
# ---------------------------------------------------------------------------
V_BED_CYL = np.pi * R_CYL ** 2 * L_BED               # m^3; matches paper's "1.06 cm^3"
T0_CYL = TABLE1['cylinder']['mu1'] / (1.0 + K_RET)    # s, void time of the cylinder column
ET = T0_CYL * TABLE1['cylinder']['Fv'] / V_BED_CYL    # total porosity (dimensionless)
KEQ = K_RET * ET / (1.0 - ET)                         # LINEAR isotherm ka/kd (kd=1)

V_BED_CONE = np.pi / 3.0 * L_BED * (R_SMALL ** 2 + R_SMALL * R_LARGE + R_LARGE ** 2)  # matches "1.21 cm^3"

# Van Deemter fit, H(v) = A + B/v + C*v, to the measured plate heights in
# Gritti2019_plate_height.py. ET is passed explicitly so that the
# velocity the fit is made against is the one this script's model runs on.
VD_A, VD_B, VD_C = ph.fit_van_deemter(ET)

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
# CADET model definition
# ---------------------------------------------------------------------------
def get_model(cadet_path, config_key, spatial_method='FV', ncol=16, dg_polydeg=4,
              n_points=3000, t_end=400.0, tracer=False):
    """Build the CADET model for one column configuration.

    Dispersion uses COL_DISPERSION_DEP='VAN_DEEMTER' with COL_DISPERSION=[1.0].
    The value is a placeholder because the dependency factor
    (VD_A*v + VD_B + VD_C*v^2)/2 is already the full D_ax(v) = H(v)*v/2.
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

    # --- Particle type: lumped rate model, i.e. no film or pore diffusion ---
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
# Reference (measured) data
# ---------------------------------------------------------------------------
def load_measured():
    data = np.genfromtxt(MEASURED_CSV, delimiter=',', names=True)
    return data


def plot_fig5_verification(output_path):
    """Plot the measured plate heights against the van Deemter fit.

    Drawn on Fig. 5's own axes, H [um] over xi = z/L, using the paper's own
    assignment of each flow rate to an axial position of the 4.2 -> 2.1 mm
    cone, so that the panel can be held next to the paper figure. The fit
    itself is made against velocity and does not use that assignment.
    """
    table = ph.load_table()
    Fv_mL_min = table['Fv_mL_min']
    H_measured = table['valerophenone_H50_um']
    xi = table['zL_cone_4p2_to_2p1']

    rmse, maxabs = ph.fit_residuals(ET)
    print(f"Measured plate heights vs. VAN_DEEMTER fit: RMSE={rmse/MICRON:.4f} um, "
          f"max|resid|={maxabs/MICRON:.4f} um "
          f"(measured H range {H_measured.min():.2f}-{H_measured.max():.2f} um, "
          f"n={len(Fv_mL_min)} flow rates)")

    # A dense curve over the same velocity range, for drawing only.
    xi_dense = np.linspace(xi.min(), xi.max(), 400)
    Fv_dense = np.interp(xi_dense, xi, Fv_mL_min)
    v_dense = ph.interstitial_velocity(Fv_dense, ET)
    H_dense = (VD_A + VD_B / v_dense + VD_C * v_dense) / MICRON

    fontsize = 15
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(xi, H_measured, 'o', ms=8, color='tab:blue',
            label='Measured $H_{1/2}$, valerophenone (Gritti et al.)')
    ax.plot(xi_dense, H_dense, '-', color='tab:orange', lw=2,
            label='VAN_DEEMTER fit used in CADET (VD_A, VD_B, VD_C)')
    ax.axhline(9.50, ls='--', color='tab:cyan',
               label='Cylinder @ 0.35 mL/min (H=9.50 um, paper text p.43)')
    ax.set_xlabel(r'Dimensionless axial position $\xi = z/L$', fontsize=fontsize)
    ax.set_ylabel(r'Local plate height $H$ [$\mu$m]', fontsize=fontsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 14)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_title('Verification: measured plate heights vs. CADET VAN_DEEMTER fit')
    ax.legend(loc='lower right', fontsize=fontsize)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.02,
            f'RMSE={rmse/MICRON:.3f} um\nmax|resid|={maxabs/MICRON:.3f} um',
            transform=ax.transAxes,
            fontsize=fontsize, va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.tight_layout()
    outpath = os.path.join(output_path, 'Gritti2019_fig6_fig5H_verification.png')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plate-height/fit verification plot to {outpath}")


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------
def compute_metrics(config_key, t_sim, c_sim, c_inj_area, ref_t, ref_c):
    """The validation metrics, see src/validation/validation_metrics.py.

    Delta mu_1 and Delta mu_2 are taken against the moments measured on the
    real columns in Table 1. The dispersion coefficient comes from the
    measured H(v) and was not fitted to Table 1, so Delta mu_2 is a prediction
    here, unlike in Gritti2019_fig8.py's cylinder, where the dispersion was
    calibrated against the tabulated mu_2 and the entry is left empty.

    The NRMSE is against the measured chromatograms as Fig. 6 plots them.
    Sec. 4.2.2 notes that the peaks printed there were adjusted in time for
    display, and the authors' own worksheet holds the cone traces at those
    adjusted positions, so the NRMSE measures peak shape and the
    retention-time error stays in Delta mu_1 against Table 1.
    """
    ref = TABLE1[config_key]
    m = vm.standard_metrics(
        name=config_key,
        t_sim=t_sim, c_sim=c_sim, t_ref=ref_t, c_ref=ref_c,
        kind=vm.PULSE,
        mu1_ref=ref['mu1'], mu2_ref=ref['mu2'], ref_label='Gritti Table 1',
        ref_curve_label='measured',
        amplitude='lsq',
        mass_in=c_inj_area,
        mass_label='simulated outlet integral vs. the analytically known '
                   'injected mass C0*t_inj',
    )
    # Diagnostic: the mean plate height implied by the simulated moments,
    # comparable to the paper's H_bar = 11.6 um on p. 43.
    m['H_bar_sim_micron'] = L_BED * m['mu2_sim_full'] / m['mu1_sim_full'] ** 2 / MICRON
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "output" / "validation"

def main(cadet_path=CADET_PATH, output_path=OUTPUT_PATH):

    os.makedirs(output_path, exist_ok=True)

    print("Derived parameters:")
    print(f"  V_bed (cylinder) = {V_BED_CYL*1e6:.4f} cm^3  (paper: 1.06 cm^3)")
    print(f"  V_bed (cone)     = {V_BED_CONE*1e6:.4f} cm^3  (paper: 1.21 cm^3)")
    print(f"  t0 (cylinder)    = {T0_CYL:.4f} s = {T0_CYL/MIN:.4f} min")
    print(f"  total porosity epsilon_t = {ET:.4f}")
    print(f"  LINEAR K_eq (ka, kd=1)   = {KEQ:.4f}")
    print(f"  Van Deemter H(v)=A+B/v+C*v : A={VD_A:.4e} m, B={VD_B:.4e} m^2/s, C={VD_C:.4e} s")
    print()

    plot_fig5_verification(output_path)

    measured = load_measured()
    ref_time = measured['time_s']
    ref_cols = {'cylinder': 'cylinder_AU', 'cone_s2': 'cone_s2_AU', 'cone_s05': 'cone_s05_AU'}

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

        ref_c = measured[ref_cols[key]]
        m = compute_metrics(key, t_sim, c_sim, c_inj_area, ref_time, ref_c)
        all_metrics[key] = m

        # The CADET curve is drawn on its own time axis, without any peak
        # realignment, so that the plot shows the true retention-time error;
        # the NRMSE in the box is the shape-only value, with the shift it was
        # measured at printed underneath. The absorbance factor is the same
        # least-squares scale the metrics use.
        # Gritti2019_fig7.py and fig8.py do the same.
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
        ax.legend(fontsize=fontsize, ncol=1)
        ax.grid(alpha=0.3)
        ax.set_xlim(215.0, 250.0)

        # NRMSE of this configuration, as a box inside the axes
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
        print(f"  {m['name']:10s}: H_bar from the simulated moments = "
              f"{m['H_bar_sim_micron']:.3f} micron "
              f"(paper, full flow-dependent H: {H_BAR_PAPER_FULL / MICRON:.1f} micron)")
    print("  Note: the NRMSE is against the measured chromatograms as Fig. 6 "
          "plots them, i.e. with the cone traces time-aligned onto the "
          "cylinder as Sec. 4.2.2 describes, so it measures peak shape. The "
          "retention-time error is in Delta mu_1 against Table 1: at their "
          "recorded times the cone peaks sit 1.8 and 2.7 s later than the "
          "model puts them.")
    vm.dump_metrics(output_path, 'Gritti2019_fig6',
                    'Isocratic valerophenone', metrics, time_unit='s')
    return metrics


if __name__ == '__main__':
    main()
