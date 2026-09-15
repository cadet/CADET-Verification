# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 8 from:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

The model, the source of the parameters and the dispersion calibration are
explained in Gritti2019_fig8.md.
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

# ===========================================================================
# Reference data, digitized from Fig. 8; see the markdown file
# ===========================================================================
def load_digitized(path=None):
    if path is None:
        path = os.path.join(HERE, 'Gritti2019_fig8_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    t = data['time_s']
    return {
        'cylinder': (t, data['cylinder_AU']),
        'cone_s2': (t, data['cone_s2_AU']),
        'cone_s05': (t, data['cone_s05_AU']),
    }


REFERENCE = load_digitized()

# Table 3, measured independently of the figure digitization. tR is the
# retention time and mu1 the first moment; they are distinct quantities but
# nearly equal for these near-Gaussian peaks. mu2 is the second central
# moment and w50 the half-height width, in minutes and minutes^2.
TABLE3 = {
    'cylinder': dict(tR=4.798, mu1=4.798, mu2=0.000288, w50=0.0386),
    'cone_s2': dict(tR=4.786, mu1=4.786, mu2=0.000363, w50=0.0450),
    'cone_s05': dict(tR=4.797, mu1=4.798, mu2=0.000313, w50=0.0422),
}

# ===========================================================================
# Physical parameters in SI units, see the markdown file
# ===========================================================================
L_COL = 0.15                      # column length [m], both geometries
R_CYL = 1.5e-3                     # cylinder radius [m] (3.0 mm i.d.)
R_SMALL = 1.05e-3                  # frustum small-end radius [m] (2.1 mm i.d.)
R_LARGE = 2.10e-3                  # frustum large-end radius [m] (4.2 mm i.d.)

Q_CYL = 0.35e-6 / 60.0             # [m^3/s]  (0.35 mL/min)
Q_CONE = 0.40e-6 / 60.0            # [m^3/s]  (0.40 mL/min)

V_INJ = 3.0e-9                     # injection volume [m^3] (3.0 uL)

EPS_T = 0.65                       # total porosity (Section 4.1.4)
F_PHASE = (1.0 - EPS_T) / EPS_T    # phase ratio (solid/liquid volume)

PHI0 = 0.10                        # starting ACN volume fraction
PHI_FINAL = 0.55                   # final ACN volume fraction
T_GRADIENT = 5.0 * 60.0            # gradient time [s]
BETA = (PHI_FINAL - PHI0) / T_GRADIENT   # gradient steepness [1/s]

S_LSSM = 25.0                      # carried over from the 17-peptide case,
                                   # see the markdown file

QMAX1 = 1000.0                     # arbitrary reference solid-phase capacity
C_INJ = 1.0                        # arbitrary reference injected concentration
KD1 = 1.0                          # arbitrary reference desorption rate [1/s]

# Van Deemter fit to Fig. 5, H(v) = A + B/v + C*v. Same values as in
# Gritti2019_fig6.py and fig7.py, repeated here so that this script runs on
# its own; Gritti2019_fig6.md describes the digitization and the fit.
VD_A = 3.15452704e-06   # m
VD_B = 4.52688414e-09   # m^2/s
VD_C = 2.23971679e-03   # s

# --- check EPS_T against the worked example of Sec. 4.1.4 ---
_u0_check = Q_CONE / (EPS_T * np.pi * R_SMALL ** 2) * 100.0 * 60.0  # cm/min
assert abs(_u0_check - 17.77) < 0.05, f"EPS_T does not reproduce the paper's u0(0): {_u0_check:.3f} cm/min vs. 17.77"


def area_cyl(r):
    return np.pi * r ** 2


def u0_entrance(Q, area):
    """Entrance interstitial velocity [m/s]."""
    return Q / (EPS_T * area)


def m1_of_s(s):
    """Dimensionless hold-up time at the column outlet, m(1) = (1+s+s^2)/3,
    the closed form of Eqs. 15/16 at xi=1. A cylinder (s=1) gives m(1)=1."""
    return (1.0 + s + s ** 2) / 3.0


def solve_k0(t_ref, s, t_R_target):
    """Solve the LSSM gradient elution-time equation
    e(1) = m(1) + (1/G)*ln(1+G*k0*m(1)) for k0, given the observed retention
    time t_R_target [s] on a column with entrance time scale t_ref = L/u0(0)
    [s] and geometry ratio s."""
    G = S_LSSM * BETA * t_ref
    m1 = m1_of_s(s)
    e1 = t_R_target / t_ref
    k0 = (np.exp(G * (e1 - m1)) - 1.0) / (G * m1)
    return k0, G, m1


# --- solve for k0 from the cylindrical column's Table 3 retention time ---
_t_ref_cyl = L_COL / u0_entrance(Q_CYL, area_cyl(R_CYL))
_t_R_cyl_target = TABLE3['cylinder']['tR'] * 60.0  # min -> s
K0, _G_cyl, _m1_cyl = solve_k0(_t_ref_cyl, 1.0, _t_R_cyl_target)

# MPM-Langmuir params implementing k'(phi) = k0*exp(-S*(phi-phi0)):
GAMMA1 = -S_LSSM
KA1 = K0 * np.exp(S_LSSM * PHI0) / (F_PHASE * QMAX1)


# ===========================================================================
# CADET model definition
# ===========================================================================
def get_model(cadet_path, config, col_dispersion_base, t_end=330.0, n_points=2201,
              polydeg=4, nelem=10, spatial_method='DG'):
    """config: 'cylinder', 'cone_s2', or 'cone_s05'.

    col_dispersion_base: scale factor on the Fig. 5 H(v) curve for bombesin,
        so that D_ax(xi) = col_dispersion_base*H(v(xi))*v(xi)/2 through
        COL_DISPERSION_DEP='VAN_DEEMTER'.
    """
    if config == 'cylinder':
        Q = Q_CYL
        geometry = 'AXIAL_FLOW_CYLINDER'
        forward_flow = 1
        area_entrance = area_cyl(R_CYL)
    elif config == 'cone_s2':
        Q = Q_CONE
        geometry = 'AXIAL_FLOW_FRUSTUM'
        forward_flow = 0   # enter at the SMALL (2.1 mm) end
        area_entrance = area_cyl(R_SMALL)
    elif config == 'cone_s05':
        Q = Q_CONE
        geometry = 'AXIAL_FLOW_FRUSTUM'
        forward_flow = 1   # enter at the LARGE (4.2 mm) end
        area_entrance = area_cyl(R_LARGE)
    else:
        raise ValueError(config)

    t_inj = V_INJ / Q  # injection duration [s]

    cadet = Cadet(install_path=cadet_path)
    m = cadet.root
    m.input.model.nunits = 3

    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, Q,
        1.0, 2.0, -1.0, -1.0, Q,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # --- Inlet: component 0 = ACN modifier (linear ramp), component 1 =
    # bombesin (rectangular injection pulse of duration t_inj) ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 2

    phi_at_tinj = PHI0 + BETA * t_inj
    phi_at_tg = PHI0 + BETA * T_GRADIENT

    m.input.model.unit_000.sec_000.const_coeff = [PHI0, C_INJ]
    m.input.model.unit_000.sec_000.lin_coeff = [BETA, 0.0]
    m.input.model.unit_000.sec_000.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_000.cube_coeff = [0.0, 0.0]

    m.input.model.unit_000.sec_001.const_coeff = [phi_at_tinj, 0.0]
    m.input.model.unit_000.sec_001.lin_coeff = [BETA, 0.0]
    m.input.model.unit_000.sec_001.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_001.cube_coeff = [0.0, 0.0]

    m.input.model.unit_000.sec_002.const_coeff = [phi_at_tg, 0.0]
    m.input.model.unit_000.sec_002.lin_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_002.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_002.cube_coeff = [0.0, 0.0]

    # --- Column ---
    col = m.input.model.unit_001
    col.unit_type = 'COLUMN_MODEL_1D'
    col.geometry = geometry
    col.ncomp = 2
    col.npartype = 1
    col.bed_length = L_COL
    col.forward_flow = [forward_flow]
    col.total_porosity = EPS_T
    col.init_c = [PHI0, 0.0]

    if geometry == 'AXIAL_FLOW_CYLINDER':
        col.cross_section_area = area_cyl(R_CYL)
    else:
        col.cross_section_area_small_end = area_cyl(R_SMALL)
        col.cross_section_area_large_end = area_cyl(R_LARGE)

    # The modifier (component 0) gets a negligible dispersion, a tiny
    # fraction of the analyte's own curve, so that its ramp travels
    # undistorted as the paper assumes. Bombesin (component 1) uses the
    # Fig. 5 plate-height curve. For the cylinder the velocity is constant,
    # so the dependency simply evaluates H(v) once.
    col.col_dispersion = [1.0e-6, col_dispersion_base]
    col.col_dispersion_multiplex = 1  # component-dependent, section-independent
    col.col_dispersion_dep = 'VAN_DEEMTER'
    col.col_dispersion_dep_a = VD_A
    col.col_dispersion_dep_b = VD_B
    col.col_dispersion_dep_c = VD_C

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    col.discretization.SPATIAL_METHOD = spatial_method
    if spatial_method == 'DG':
        col.discretization.POLYDEG = polydeg
        col.discretization.NELEM = nelem
        col.discretization.USE_COLLOCATION_DG = 0
        col.dispersion_spatial_dependence_polydeg = polydeg
    elif spatial_method == 'FV':
        col.discretization.SPATIAL_METHOD = 'FV'
        col.discretization.NCOL = nelem
        col.discretization.RECONSTRUCTION = 'WENO'
        col.discretization.weno.WENO_ORDER = 3
        col.discretization.weno.WENO_EPS = 1e-10
        col.discretization.weno.BOUNDARY_MODEL = 0
        col.discretization.GS_TYPE = 1
        col.discretization.MAX_KRYLOV = 0
        col.discretization.MAX_RESTARTS = 10
        col.discretization.SCHUR_SAFETY = 1e-8
    else:
        raise ValueError(f"Unsupported spatial method: {spatial_method}")

    # --- Particle type: local equilibrium, no film or pore diffusion ---
    col.particle_type_000.nbound = [0, 1]
    col.particle_type_000.init_cp = [PHI0, 0.0]
    col.particle_type_000.init_cs = [0.0]
    col.particle_type_000.has_film_diffusion = 0
    col.particle_type_000.has_pore_diffusion = 0
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.par_radius = 5.0e-6
    col.particle_type_000.par_porosity = 0.35

    col.particle_type_000.adsorption_model = 'MOBILE_PHASE_MODULATOR'
    col.particle_type_000.adsorption.is_kinetic = 0
    col.particle_type_000.adsorption.mpm_ka = [0.0, KA1]
    col.particle_type_000.adsorption.mpm_kd = [1.0, KD1]
    col.particle_type_000.adsorption.mpm_qmax = [1.0, QMAX1]
    col.particle_type_000.adsorption.mpm_gamma = [0.0, GAMMA1]
    col.particle_type_000.adsorption.mpm_beta = [0.0, 0.0]
    col.particle_type_000.adsorption.mpm_linear_threshold = 0.0

    col.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
    col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT'
    col.particle_type_000.discretization.PAR_POLYDEG = 1
    col.particle_type_000.discretization.PAR_NELEM = 1

    m.input.model.unit_002.ncomp = 2
    m.input.model.unit_002.unit_type = 'OUTLET'

    # --- return group ---
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_000.write_solution_outlet = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0
    m.input['return'].unit_002.write_solution_outlet = 0

    # --- time integration ---
    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 3
    m.input.solver.sections.section_continuity = [False, False]
    m.input.solver.sections.section_times = [0.0, t_inj, T_GRADIENT, t_end]
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-10
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, t_end, n_points)

    return cadet


def run_model(cadet_path, output_path, config, col_dispersion_base, fname=None,
              spatial_method='DG', **kwargs):
    c = get_model(cadet_path, config, col_dispersion_base,
                  spatial_method=spatial_method, **kwargs)
    c.filename = os.path.join(output_path, fname or f'Gritti2019_fig8_{config}.h5')
    c.save()
    rc = c.run_simulation()
    if rc.return_code != 0:
        raise RuntimeError(f"CADET failed ({config}): {getattr(rc, 'error_message', rc)}")
    c.load_from_file()
    t = np.asarray(c.root.output.solution.solution_times)
    outlet = np.asarray(c.root.output.solution.unit_001.solution_outlet)
    return t, outlet[:, 1]  # bombesin (component 1) only


# ===========================================================================
# Moment / mass-balance helpers
# ===========================================================================
def _trapezoid(y, x):
    """Trapezoidal integration, tolerant of the numpy trapz/trapezoid rename."""
    return vm.trapezoid(y, x)


def moments(t, c):
    area = _trapezoid(c, t)
    if area <= 0:
        return 0.0, 0.0, 0.0
    mu1 = _trapezoid(t * c, t) / area
    mu2 = _trapezoid((t - mu1) ** 2 * c, t) / area
    return area, mu1, mu2


def peak_time(t, c):
    return t[np.argmax(c)]


def compute_metrics(config, t_sim, c_sim, t_ref, c_ref):
    """The validation metrics, see src/validation/validation_metrics.py.

    ``c_sim`` is the raw simulated outlet. The reference is an uncalibrated
    absorbance, so the least-squares amplitude factor is fitted inside the
    metric routine, per column.

    Delta mu_1 is taken against Table 3's first moment. For the two cones,
    Delta mu_2 is a prediction against Table 3's second central moment: the
    dispersion scale factor is calibrated once on the cylinder and reused
    unchanged (see calibrate_dispersion()). The cylinder's own entry is left
    empty, since its mu_2 is what was calibrated against.
    """
    ref = TABLE3[config]
    t_inj = V_INJ / (Q_CYL if config == 'cylinder' else Q_CONE)
    return vm.standard_metrics(
        name=config,
        t_sim=t_sim, c_sim=c_sim, t_ref=t_ref, c_ref=c_ref,
        kind=vm.PULSE,
        mu1_ref=ref['mu1'] * 60.0,        # Table 3 [min] -> [s]
        mu2_ref=ref['mu2'] * 3600.0,      # Table 3 [min^2] -> [s^2]
        ref_label='Gritti Table 3',
        mu2_calibrated=(config == 'cylinder'),
        amplitude='lsq',
        mass_in=C_INJ * t_inj,
        mass_label='simulated outlet integral vs. the analytically known '
                   'injected mass C0*t_inj',
    )


# ===========================================================================
# Calibration of the dispersion scale factor
#
# The Fig. 5 plate height was measured for the alkanophenones, not for
# bombesin, so one scale factor on that curve is unavoidable. It is
# calibrated against the cylinder's Table 3 second central moment and then
# reused unchanged for both cones, which is what makes their mu_2 a
# prediction: the entrance velocity of the rho_s=2 cone is about 2.3 times
# the cylinder's and that of rho_s=0.5 about 0.58 times.
#
# The calibrated dispersion is small enough that a coarse grid's own
# numerical dispersion would not be negligible against it. NELEM=32 is the
# coarsest resolution at which the moments are converged (NELEM=32/48/64
# agree to better than 0.01%), so it is used for the calibration runs.
# ===========================================================================
_SCALE_BRACKET = (0.3, 3.0)   # probe bracket for the scale factor


def calibrate_dispersion(cadet_path, output_path, config='cylinder'):
    """Calibrate the scale factor against `config`'s own Table 3 second
    central moment. main() calls this once, on the cylinder, and reuses the
    result unchanged for both cones."""
    target_var_s2 = TABLE3[config]['mu2'] * 3600.0  # min^2 -> s^2
    scale_trials = _SCALE_BRACKET
    var_trials = []
    for scale in scale_trials:
        t, c = run_model(cadet_path, output_path, config, scale,
                          fname=f'Gritti2019_fig8_calib_{config}.h5',
                          polydeg=4, nelem=32, n_points=2001)
        _, _, var = moments(t, c)
        var_trials.append(var)
        print(f"  [{config}] calibration trial: scale={scale:.3f} -> sigma_t^2={var:.5f} s^2")

    # affine fit: sigma^2(scale) = a + b*scale
    b = (var_trials[1] - var_trials[0]) / (scale_trials[1] - scale_trials[0])
    a = var_trials[0] - b * scale_trials[0]
    scale_fit = (target_var_s2 - a) / b
    scale_fit = max(scale_fit, 1.0e-3)
    print(f"  [{config}] affine fit: sigma^2 = {a:.5f} + {b:.6g}*scale  ->  scale_calibrated={scale_fit:.4f}")
    return scale_fit


# ===========================================================================
# Main
# ===========================================================================
CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "output" / "validation"

def main(cadet_path=CADET_PATH, output_path=OUTPUT_PATH):

    os.makedirs(output_path, exist_ok=True)

    spatial_method = 'DG'

    print("=" * 70)
    print("Derived / calibrated parameters")
    print("=" * 70)
    print(f"  eps_t = {EPS_T}, F (phase ratio) = {F_PHASE:.4f}")
    print(f"  gradient steepness beta = {BETA * 60:.4f} /min ({BETA:.6f} /s)")
    print(f"  S (LSSM slope, assumed) = {S_LSSM}")
    print(f"  cylinder: t_ref = {_t_ref_cyl:.4f} s, G = {_G_cyl:.4f}, m(1) = {_m1_cyl:.4f}")
    print(f"  solved k0 (retention factor at phi0) = {K0:.4f}")
    print(f"  MPM-Langmuir: ka_1={KA1:.6g}, kd_1={KD1}, qmax_1={QMAX1}, gamma_1={GAMMA1}")

    print("\nCalibrating the dispersion scale factor against the cylinder's "
          "Table 3 second moment, then reusing it for both cones...")
    scale_cal = calibrate_dispersion(cadet_path, output_path, 'cylinder')
    col_disp_base = {'cylinder': scale_cal, 'cone_s2': scale_cal, 'cone_s05': scale_cal}
    print(f"  -> calibrated scale factor = {scale_cal:.4f} "
          f"(reused for cone_s2 and cone_s05)")

    print("\nRunning the three column configurations...")
    # NELEM=64 rather than the 32 used for the calibration: the moments are
    # already converged at 32, but the curve still shows a small ringing of
    # order 1e-2 near the rectangular injection pulse, which is down to
    # about 1e-5 by NELEM=96. 64 is a reasonable compromise for the plots.
    results = {}
    for config in ('cylinder', 'cone_s2', 'cone_s05'):

        print(f"\nRunning simulation for {config} with spatial method {spatial_method}...")
        t, c = run_model(cadet_path, output_path, config, col_disp_base[config],
                         polydeg=4, nelem=64, n_points=2201,
                         spatial_method=spatial_method)

        results[config] = (t, c)
        area, mu1, mu2 = moments(t, c)
        tp = peak_time(t, c)
        print(f"  {config:10s}: peak_t={tp:7.3f} s   mu1={mu1:7.3f} s   "
              f"sigma^2={mu2:.5f} s^2   area={area:.5f}")

    # ---- validation metrics, see src/validation/validation_metrics.py ----
    print("\n" + "=" * 70)
    print("Validation metrics -- Gritti et al. (2019), Fig. 8 "
          "(Bombesin, gradient)")
    print("=" * 70)
    metrics = {}
    for config in ('cylinder', 'cone_s2', 'cone_s05'):
        t_sim, c_sim = results[config]
        t_ref, c_ref = REFERENCE[config]
        metrics[config] = compute_metrics(config, t_sim, c_sim, t_ref, c_ref)

    # ---- one comparison plot per column ----
    colors = {'cylinder': 'k', 'cone_s2': 'tab:red', 'cone_s05': 'tab:blue'}
    labels = {'cylinder': r'$\rho_s=1$', 'cone_s2': r'$\rho_s=2$', 'cone_s05': r'$\rho_s=0.5$'}
    for config in ('cylinder', 'cone_s2', 'cone_s05'):
        t_sim, c_sim = results[config]
        t_ref, c_ref = REFERENCE[config]
        scale = metrics[config]['amplitude_scale']
        color = colors[config]
        label = labels[config]

        fig, ax = plt.subplots(figsize=(7.5, 5.8))
        fontsize = 15
        ax.plot(t_sim, scale * c_sim, '-', color=color, lw=1.5,
                label=f"{label} (CADET)")
        ax.plot(t_ref, c_ref, 'o', color=color, ms=3, mfc='none',
                label=f"{label} (Gritti 2019)")
        ax.set_xlabel('Time [s]', fontsize=fontsize)
        ax.set_ylabel('Absorbance [AU]', fontsize=fontsize)
        ax.set_xlim(270, 306)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.text(
            0.98, 0.75, f"NRMSE: {metrics[config]['nrmse_%']:.2f}%", transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=fontsize, bbox=dict(
            boxstyle='round,pad=0.3', facecolor='white', alpha=0.7
            )
        )
        ax.legend(loc='upper right', fontsize=fontsize, ncol=1)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        outpath = os.path.join(output_path, f'Gritti2019_fig8_comparison_{config}_{spatial_method}.png')
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"\nSaved comparison plot to {outpath}")

    ordered = [metrics[config] for config in ('cylinder', 'cone_s2', 'cone_s05')]
    vm.print_metrics_table(ordered, time_unit='s')
    print(f"  cylinder  : calibrated mu_2 = {ordered[0]['mu2_sim_full']:.6g} s^2 "
          f"vs. Table 3 target {ordered[0]['mu2_ref']:.6g} s^2, which is why "
          f"its Delta mu_2 is left empty above")
    vm.dump_metrics(output_path, 'Gritti2019_fig8',
                    'Gradient Bombesin', ordered, time_unit='s')
    return ordered


if __name__ == '__main__':
    main()
