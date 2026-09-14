# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 7 from:

    F. Gritti, J. Belanger, G. Izzo, W. Leveille, "On the performance of
    conically shaped columns: Theory and practice", J. Chromatogr. A 1593
    (2019) 34-46. https://doi.org/10.1016/j.chroma.2019.01.055

Self-contained script: model definition, run, comparison plot, and
validation metrics. Further explanation on model and parameter selection is
provided under Gritti2019_fig7.md

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

# ---------------------------------------------------------------------------
# Physical parameters (SI units)
# ---------------------------------------------------------------------------
PI = np.pi
L = 0.15                      # m, column length (both column types)
EPS_T = 0.65                   # total porosity (paper's own assumed value, Sec. 4.1.4)

RE_CYL = 1.5e-3                # m, cylindrical column radius
FV_CYL = 0.35e-6 / 60.0         # m^3/s (0.35 mL/min)
VCOL_CYL = 1.06e-6              # m^3, empty cylindrical column volume

RE_SMALL = 1.05e-3              # m, frustum small-end radius (2.1 mm i.d.)
RE_LARGE = 2.10e-3              # m, frustum large-end radius (4.2 mm i.d.)
FV_CON = 0.40e-6 / 60.0          # m^3/s (0.40 mL/min)
VCOL_CON = 1.21e-6               # m^3, empty conical column volume

BETA_PER_MIN = 0.07              # gradient steepness [1/min] (Fig. 2 caption)
PHI0 = 0.60                      # starting ACN volume fraction
PHI_FINAL = 0.95                 # final ACN volume fraction
TG_MIN = (PHI_FINAL - PHI0) / BETA_PER_MIN   # = 5.0 min, gradient duration

K_ISO = 1.08                     # isocratic k(phi=0.75), valerophenone (text, p.44)
PHI_ISO = 0.75

H_VALEROPHENONE = 9.5e-6         # m, isocratic plate height, cylinder, 0.35 mL/min (text, p.43)
                                 # -- reference/context value only (see COL_DISP_PROBE below).

# Van Deemter fit to Gritti et al. Fig. 5 (H(v) = A + B/v + C*v), identical values
# to Gritti2019_fig6.py's VD_A/VD_B/VD_C (re-derived there from the digitized
# Fig. 5 data; hardcoded here too so this script stays self-contained/importable on
# its own). See fig6's module docstring for the digitization/fit procedure and
# fig6's plot_fig5_verification() for the fit-quality check (RMSE=0.071 um).
VD_A = 3.15452704e-06   # m
VD_B = 4.52688414e-09   # m^2/s
VD_C = 2.23971679e-03   # s

VINJ = 0.5e-9                    # m^3, injection volume (0.5 uL, Sec. 3.4.2)

TR_GRAD_CYL_MIN = 4.656          # Table 2, cylindrical column, first moment [min]
TR_GRAD_S2_MIN = 4.659           # Table 2, cone rho_s=2
TR_GRAD_S05_MIN = 4.674          # Table 2, cone rho_s=0.5

# Table 2, valerophenone, SECOND CENTRAL MOMENT [min^2] under gradient
# conditions -- the calibration target for the per-column dispersion fix
# (see root-cause docstring section above).
MU2_GRAD_CYL = 0.00096
MU2_GRAD_S2 = 0.00080
MU2_GRAD_S05 = 0.00072


def tau_ref_min(re, Fv):
    """Reference time L/u0(0) [min] at a column's own inlet radius re."""
    u0 = Fv / (EPS_T * PI * re ** 2)
    return (L / u0) / 60.0


T0_CYL_MIN = tau_ref_min(RE_CYL, FV_CYL)
T0_S2_MIN = tau_ref_min(RE_SMALL, FV_CON)     # cone rho_s=2: inlet = small end
T0_S05_MIN = tau_ref_min(RE_LARGE, FV_CON)    # cone rho_s=0.5: inlet = large end


def _bisect(f, a, b, xtol=1e-13, max_iter=200):
    """Minimal dependency-free bisection root finder (replaces
    scipy.optimize.brentq -- this script intentionally imports nothing beyond
    cadet/numpy/matplotlib). f is assumed continuous with f(a) and f(b) of
    opposite sign (checked below)."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if np.sign(fa) == np.sign(fb):
        raise ValueError(f"Root not bracketed: f({a})={fa}, f({b})={fb}")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0.0 or 0.5 * (b - a) < xtol:
            return m
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5 * (a + b)


def derive_lssm_parameters():
    """Derive (S, k0) for valerophenone from the paper's own Eqs. (28) & (34)
    evaluated at the cylindrical column only (rho_s=1), using the paper's own
    tabulated k(0.75)=1.08 and Table-2 gradient retention time (4.656 min).
    See module docstring, "Reparameterization"."""
    tau_e1_target = TR_GRAD_CYL_MIN / T0_CYL_MIN

    def resid(S):
        k0 = K_ISO * np.exp(S * (PHI_ISO - PHI0))
        G = S * BETA_PER_MIN * T0_CYL_MIN
        tau_e1 = 1.0 + (1.0 / G) * np.log(1.0 + G * k0)
        return tau_e1 - tau_e1_target

    S = _bisect(resid, 1e-3, 50.0, xtol=1e-13)
    k0 = K_ISO * np.exp(S * (PHI_ISO - PHI0))
    return S, k0


def tau_e1_analytic(S, k0, beta, t0, s):
    """Paper's Eq. (34): dimensionless gradient elution time at column
    outlet for a frustum of ratio s=rho_s, evaluated with reference time t0
    (=L/u0(0), at that orientation's own inlet radius)."""
    G = S * beta * t0
    poly = 1.0 + s + s ** 2
    return poly / 3.0 + (1.0 / G) * np.log(1.0 + (G * k0 / 3.0) * poly)


S_LSSM, K0_LSSM = derive_lssm_parameters()
GAMMA1 = -S_LSSM
QMAX1 = 1.0e4                      # arbitrary large placeholder (linear/dilute limit)
A_PREFACTOR = K0_LSSM * np.exp(S_LSSM * PHI0)   # = k(phi=0)-equivalent prefactor
KA1 = A_PREFACTOR * EPS_T / (1.0 - EPS_T) / QMAX1
KD1 = 1.0

COL_DISP_PROBE = 1.0        # dimensionless scale factor probe value for calibrate_dispersion()
                            # (1.0 = Fig. 5 VAN_DEEMTER curve exactly as measured/fitted, unscaled).
COL_DISP_MODIFIER = 1.0e-6  # dimensionless scale factor for the ACN modifier: a tiny fraction
                            # of valerophenone's own H(v) curve, giving near-plug-flow transport.

# ---------------------------------------------------------------------------
# CADET model definition
# ---------------------------------------------------------------------------
COLUMNS = {
    'cylinder': dict(geometry='AXIAL_FLOW_FRUSTUM', Fv=FV_CYL, forward_flow=1,
                      tR_ref=TR_GRAD_CYL_MIN, mu2_ref=MU2_GRAD_CYL,
                      color='k', label=r'$\rho_s=1$'),
    'cone_s2': dict(geometry='AXIAL_FLOW_FRUSTUM', Fv=FV_CON, forward_flow=0,
                     tR_ref=TR_GRAD_S2_MIN, mu2_ref=MU2_GRAD_S2,
                     color='tab:red', label=r'$\rho_s=2$'),
    'cone_s05': dict(geometry='AXIAL_FLOW_FRUSTUM', Fv=FV_CON, forward_flow=1,
                      tR_ref=TR_GRAD_S05_MIN, mu2_ref=MU2_GRAD_S05,
                      color='tab:blue', label=r'$\rho_s=0.5$'),
}

# Populated by calibrate_dispersion() in __main__ before the production runs;
# maps column key -> calibrated COL_DISPERSION length-scale [m] for
# valerophenone (component 1). Falls back to the COL_DISP_PROBE value if a
# column has not (yet) been calibrated, e.g. when get_model() is imported
# and used standalone/interactively.
H_EFF = {}


def get_model(cadet_path, column, spatial_method='DG', nelem=128, polydeg=4, ncol=800,
              n_points=3000, t_end_min=7.5, col_disp_valerophenone=None):
    """Build the CADET model for one of the three column configurations
    ('cylinder', 'cone_s2', 'cone_s05') and return a ready-to-run `Cadet`
    instance. The model tree is built directly on the `Cadet` object's own
    `.root` attribute -- which the `cadet` package itself already provides as
    an addict.Dict-like nested structure -- so this script does not need to
    import addict (or anything else) itself.

    Flow sheet: unit_000=INLET (2 components: 0=ACN modifier, 1=valerophenone)
    -> unit_001=COLUMN (native geometry) -> unit_002=OUTLET.

    col_disp_valerophenone: dimensionless COL_DISPERSION scale factor for
        component 1 (valerophenone), multiplying the Fig. 5 VAN_DEEMTER
        H(v) curve (VD_A, VD_B, VD_C -- same fit as Gritti2019_fig6.py):
        Dax(z) = col_disp_valerophenone * H(v(z))*v(z)/2 via
        COL_DISPERSION_DEP='VAN_DEEMTER'. Defaults to the per-column
        calibrated value in H_EFF (see calibrate_dispersion() and the
        "Dispersion calibration" docstring section); falls back to the uncalibrated probe
        value (COL_DISP_PROBE=1.0, i.e. the Fig. 5 curve exactly as
        measured/fitted) if that column has not been calibrated yet.
    """
    cfg = COLUMNS[column]
    if col_disp_valerophenone is None:
        col_disp_valerophenone = H_EFF.get(column, COL_DISP_PROBE)
    Fv = cfg['Fv']
    t_inj = VINJ / Fv                       # s, injection pulse duration
    tg_s = TG_MIN * 60.0                    # s, gradient duration
    t_end = t_end_min * 60.0
    beta_per_s = BETA_PER_MIN / 60.0

    cadet_obj = Cadet(install_path=cadet_path)
    m = cadet_obj.root
    m.input.model.nunits = 3

    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, Fv,
        1.0, 2.0, -1.0, -1.0, Fv,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # --- Inlet: component 0 = ACN modifier (gradient), component 1 =
    # valerophenone (narrow injection pulse at t=0) ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 2

    # Section 0: [0, t_inj) -- injection pulse + gradient ramp already started
    m.input.model.unit_000.sec_000.const_coeff = [PHI0, 1.0]
    m.input.model.unit_000.sec_000.lin_coeff = [beta_per_s, 0.0]
    m.input.model.unit_000.sec_000.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_000.cube_coeff = [0.0, 0.0]
    # Section 1: [t_inj, tg) -- gradient ramp continues, analyte back to 0
    phi_at_tinj = PHI0 + beta_per_s * t_inj
    m.input.model.unit_000.sec_001.const_coeff = [phi_at_tinj, 0.0]
    m.input.model.unit_000.sec_001.lin_coeff = [beta_per_s, 0.0]
    m.input.model.unit_000.sec_001.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_001.cube_coeff = [0.0, 0.0]
    # Section 2: [tg, t_end) -- isocratic hold at phi_final
    m.input.model.unit_000.sec_002.const_coeff = [PHI_FINAL, 0.0]
    m.input.model.unit_000.sec_002.lin_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_002.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_002.cube_coeff = [0.0, 0.0]

    # --- Column ---
    col = m.input.model.unit_001
    col.unit_type = 'COLUMN_MODEL_1D'
    col.geometry = cfg['geometry']
    col.ncomp = 2
    col.bed_length = L
    if column == 'cylinder':
        col.cross_section_area_small_end = PI * RE_CYL ** 2
        col.cross_section_area_large_end = col.cross_section_area_small_end
    elif column in ('cone_s2', 'cone_s05'):
        col.cross_section_area_small_end = PI * RE_SMALL ** 2
        col.cross_section_area_large_end = PI * RE_LARGE ** 2
    else:
        raise ValueError(f"Unexpected column key: {column}")
    col.forward_flow = [cfg['forward_flow']]
    col.total_porosity = EPS_T
    col.npartype = 1
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [COL_DISP_MODIFIER, col_disp_valerophenone]
    col.col_dispersion_dep = 'VAN_DEEMTER'
    col.col_dispersion_dep_a = VD_A
    col.col_dispersion_dep_b = VD_B
    col.col_dispersion_dep_c = VD_C
    col.init_c = [PHI0, 0.0]

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    if spatial_method == 'DG':
        col.discretization.SPATIAL_METHOD = 'DG'
        col.discretization.POLYDEG = polydeg
        col.discretization.NELEM = nelem
        col.discretization.USE_COLLOCATION_DG = 0
        col.dispersion_spatial_dependence_polydeg = polydeg
    elif spatial_method == 'FV':
        col.discretization.SPATIAL_METHOD = 'FV'
        col.discretization.NCOL = ncol
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

    # --- particle_type_000: Lumped-Rate-Model-without-pores mode ---
    col.particle_type_000.nbound = [0, 1]
    col.particle_type_000.has_film_diffusion = 0
    col.particle_type_000.init_cs = [0.0]

    col.particle_type_000.adsorption_model = 'MOBILE_PHASE_MODULATOR'
    col.particle_type_000.adsorption.is_kinetic = 0
    col.particle_type_000.adsorption.mpm_ka = [0.0, KA1]
    col.particle_type_000.adsorption.mpm_kd = [0.0, KD1]
    col.particle_type_000.adsorption.mpm_qmax = [0.0, QMAX1]
    col.particle_type_000.adsorption.mpm_gamma = [0.0, GAMMA1]
    col.particle_type_000.adsorption.mpm_beta = [0.0, 0.0]
    # Scalar linearization threshold for the c_{p,0}^beta term (only matters
    # for beta!=0; kept far below the gradient's phi range 0.6-0.95 so the
    # full nonlinear branch is always used in this model).
    col.particle_type_000.adsorption.mpm_linear_threshold = 1e-6

    m.input.model.unit_002.ncomp = 2
    m.input.model.unit_002.unit_type = 'OUTLET'

    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_000.write_solution_outlet = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_002.write_solution_outlet = 0

    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 3
    m.input.solver.sections.section_continuity = [0, 0]
    m.input.solver.sections.section_times = [0.0, t_inj, tg_s, t_end]
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-10
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, t_end, n_points)

    return cadet_obj


def run_column(cadet_path, output_path, column, **kwargs):
    c = get_model(cadet_path, column, **kwargs)
    c.filename = os.path.join(output_path, f'Gritti2019_fig7_{column}.h5')
    c.save()
    rc = c.run_simulation()
    if rc.return_code != 0:
        raise RuntimeError(f"CADET failed ({column}): {getattr(rc, 'error_message', rc)}")
    c.load_from_file()
    t = np.asarray(c.root.output.solution.solution_times)
    outlet = np.asarray(c.root.output.solution.unit_001.solution_outlet)
    c_modifier = outlet[:, 0]
    c_valerophenone = outlet[:, 1]
    return t, c_modifier, c_valerophenone


def second_central_moment(t, c):
    """First and second central moment, using the shared pulse-moment
    definition so that the dispersion calibration target below and the
    reported Delta mu_2 are computed identically."""
    _, m1, m2 = vm.pulse_moments(t, c)
    return m1, m2


def calibrate_dispersion(cadet_path, output_path, column, nelem=64,
                         probe_value=COL_DISP_PROBE):
    """Calibrate the dimensionless COL_DISPERSION scale factor for
    valerophenone on this column (multiplying the Fig. 5 VAN_DEEMTER H(v)
    curve, see get_model()) so that the full gradient-elution PDE simulation
    reproduces THIS COLUMN'S OWN measured second central moment (Table 2,
    mu2_ref) -- see the "Dispersion calibration" docstring section for why
    a per-column scale factor is needed at all (in short: the cylinder's real
    peak is genuinely tailed -- a packing/wall effect no symmetric-dispersion
    model, VAN_DEEMTER or otherwise, can capture -- while the cones are
    genuinely Gaussian and expected to need only a small, ~unity, correction
    on top of the real measured H(v) curve). Variance scales essentially
    exactly linearly with the configured dispersion scale factor for this
    problem (verified separately to <0.1% by direct probing at 1x/2x/3x the
    baseline value), so a single probe run plus closed-form rescaling is used
    instead of an iterative optimizer."""
    t, _, c_val = run_column(cadet_path, output_path, column, spatial_method='DG',
                             nelem=nelem, polydeg=4,
                             col_disp_valerophenone=probe_value)
    _, var_probe = second_central_moment(t, c_val)
    target_var_s2 = COLUMNS[column]['mu2_ref'] * 3600.0   # min^2 -> s^2
    return probe_value * (target_var_s2 / var_probe), var_probe / 3600.0


# ---------------------------------------------------------------------------
# Digitized reference data
# ---------------------------------------------------------------------------
def load_digitized(path=None):
    """Load the digitized CSV. Each curve keeps only its own valid (non-NaN)
    samples and its own x-grid -- the three curves do not fully share pixel
    columns in the source image (partial occlusion of the red "cone rho_s=2"
    trace by the black/blue traces where they overlap), so a shared x-grid
    with NaN gaps is deliberately NOT assumed downstream."""
    if path is None:
        path = os.path.join(HERE, 'Gritti2019_fig7_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    t = data['time_s']
    out = {}
    for key, col in (('cylinder', 'cylinder_AU'), ('cone_s2', 'cone_s2_AU'), ('cone_s05', 'cone_s05_AU')):
        y = data[col]
        valid = ~np.isnan(y)
        out[key] = (t[valid], y[valid])
    return out


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------
def compute_metrics(column, t_sim, c_sim, t_ref, c_ref):
    """The four unified validation metrics -- see src/validation/validation_metrics.py
    for their definitions, which are shared verbatim by all six case studies.

    ``c_sim`` is the RAW simulated outlet; the per-column least-squares
    Absorbance-[AU] amplitude fit that the arbitrary-unit reference requires
    is performed inside the metric routine (see "AU-scale amplitude" in the
    module docstring for why it is fit per column and not shared).

    Delta mu_1 is taken against Table 2's measured first moment. Delta mu_2
    is deliberately left EMPTY for every column of this figure: the
    dispersion scale factor was calibrated column by column so as to
    reproduce exactly Table 2's mu_2 (see calibrate_dispersion()), so the
    agreement is fitted rather than predicted and would be misleading in a
    validation table. The calibrated-vs-target mu_2 values are still printed
    below as a diagnostic.
    """
    cfg = COLUMNS[column]
    t_inj = VINJ / cfg['Fv']
    return vm.standard_metrics(
        name=column,
        t_sim=t_sim, c_sim=c_sim, t_ref=t_ref, c_ref=c_ref,
        kind=vm.PULSE,
        mu1_ref=cfg['tR_ref'] * 60.0,          # Table 2 [min] -> [s]
        mu2_ref=cfg['mu2_ref'] * 3600.0,       # Table 2 [min^2] -> [s^2]
        ref_label='Gritti Table 2',
        mu2_calibrated=True,
        amplitude='lsq',
        mass_in=1.0 * t_inj,   # inlet valerophenone concentration is 1.0 (arbitrary units)
        mass_label='simulated outlet integral vs. the analytically known '
                   'injected mass C0*t_inj',
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
from pathlib import Path
CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "output" / "validation"

def main(cadet_path=CADET_PATH, output_path=OUTPUT_PATH):

    os.makedirs(output_path, exist_ok=True)

    print("Derived LSSM parameters (valerophenone): "
          f"S={S_LSSM:.4f}, k0={K0_LSSM:.4f}, gamma={GAMMA1:.4f}, KA={KA1:.4e}")
    print(f"COL_DISPERSION probe scale factor on the Fig. 5 VAN_DEEMTER H(v) curve: "
          f"{COL_DISP_PROBE:.4g} (1.0 = curve exactly as measured/fitted; "
          f"VD_A={VD_A:.4e} m, VD_B={VD_B:.4e} m^2/s, VD_C={VD_C:.4e} s)")
    print(f"t0 [min]: cylinder={T0_CYL_MIN:.4f}  cone_s2={T0_S2_MIN:.4f}  "
          f"cone_s05={T0_S05_MIN:.4f}")

    print("\nCalibrating per-column dispersion SCALE FACTOR (on top of the real, "
          "measured Fig. 5 VAN_DEEMTER H(v) curve) against each column's own "
          "measured gradient second moment (Table 2) -- see 'Dispersion "
          "calibration' in the module docstring: this scale factor is expected "
          "to come out close to 1.0 for the (genuinely Gaussian) cones, and "
          "substantially larger for the (genuinely tailed) cylinder:")
    for col in ('cylinder', 'cone_s2', 'cone_s05'):
        # NELEM=128 (matching production resolution) is required for the
        # calibration probe itself: cone_rho_s=2's second central moment is
        # under-converged at NELEM=64 (see "Numerical resolution" in the
        # module docstring), which would otherwise bake a resolution error
        # into the calibrated scale factor.
        disp_scale, var_probe_min2 = calibrate_dispersion(
            cadet_path, output_path, col, nelem=128)
        H_EFF[col] = disp_scale
        print(f"  {col:10s}: probe (VAN_DEEMTER, scale=1.0) variance={var_probe_min2:.6f} min^2  "
              f"Table 2 target={COLUMNS[col]['mu2_ref']:.6f} min^2  "
              f"-> calibrated dispersion scale factor={disp_scale:.4f} "
              f"(x{disp_scale/COL_DISP_PROBE:.3f} of the probe value)")

    print("\nAnalytic (paper Eq. 34) cross-check, using ONLY parameters "
          "derived from the cylindrical column:")
    for col, s in (('cylinder', 1.0), ('cone_s2', 2.0), ('cone_s05', 0.5)):
        t0 = {'cylinder': T0_CYL_MIN, 'cone_s2': T0_S2_MIN, 'cone_s05': T0_S05_MIN}[col]
        tR_pred = tau_e1_analytic(S_LSSM, K0_LSSM, BETA_PER_MIN, t0, s) * t0
        tR_actual = COLUMNS[col]['tR_ref']
        print(f"  {col:10s}: analytic tR={tR_pred:.4f} min  Table 2={tR_actual:.4f} min  "
              f"rel.err={100*abs(tR_pred-tR_actual)/tR_actual:.3g}%")

    print("\nLoading digitized reference data...")
    ref = load_digitized()

    print("\nRunning CADET simulations")

    spatial_method = 'DG'
    
    results = {}

    for col in ('cylinder', 'cone_s2', 'cone_s05'):
        print(f"  {col} ...")
        t, c_mod, c_val = run_column(cadet_path, output_path, col,
                                     spatial_method=spatial_method, nelem=128, polydeg=4)
        results[col] = (t, c_mod, c_val)

    # Per-column least-squares AU-scale fit (matching the convention used in
    # Gritti2019_fig8.py); see "AU-scale amplitude" in the module docstring
    # for why this is a display convention and why it is fit independently
    # per column rather than with one shared scale.
    print("\nFitting a per-column least-squares Absorbance-[AU] scale factor "
          "(simulated vs. digitized), independently for each column -- see "
          "'AU-scale amplitude' in the module docstring for why no shared "
          "AU-per-concentration scale factor is used.")

    print("\nValidation metrics:")
    all_metrics = {}
    for col in ('cylinder', 'cone_s2', 'cone_s05'):
        t, c_mod, c_val = results[col]
        t_ref, c_ref = ref[col]

        # The per-column least-squares Absorbance-[AU] scale (c_val is in
        # arbitrary CADET concentration units) is fit inside the shared
        # metric routine and returned as 'amplitude_scale'.
        metrics = compute_metrics(col, t, c_val, t_ref, c_ref)
        all_metrics[col] = metrics
        scale = metrics['amplitude_scale']

        # --- comparison plot ---
        fontsize = 15
        fig, ax = plt.subplots(figsize=(7.5, 5.8))
        t, c_mod, c_val = results[col]
        cfg = COLUMNS[col]
        ax.plot(t, scale * c_val, '-', color=cfg['color'], lw=1.5,
                label=f"{cfg['label']} (CADET)")
        t_ref, c_ref = ref[col]
        ax.plot(t_ref, c_ref, 'o', color=cfg['color'], ms=3, mfc='none',
                label=f"{cfg['label']} (Gritti 2019)")
        ax.set_xlabel('Time [s]', fontsize=fontsize)
        ax.set_ylabel('Absorbance [AU]', fontsize=fontsize)
        ax.set_xlim(270, 295)
        ax.set_ylim(None, 0.2)
        # ax.set_title('Gritti et al. (2019), Fig. 7 -- valerophenone gradient elution\n',
        #              fontsize=fontsize)
        nrmse = all_metrics[col]['nrmse_%']
        ax.text(
            0.98, 0.75, f"NRMSE: {nrmse:.2f}%", transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=fontsize, bbox=dict(
            boxstyle='round,pad=0.3', facecolor='white', alpha=0.7
            )
        )
        ax.legend(loc='upper right', fontsize=fontsize, ncol=1)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=fontsize)
        fig.tight_layout()
        outpath = os.path.join(output_path, f'Gritti2019_fig7_comparison_{col}_{spatial_method}.png')
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"\nSaved comparison plot to {outpath}")

    metrics = [all_metrics[col] for col in ('cylinder', 'cone_s2', 'cone_s05')]
    print("\n" + "=" * 70)
    print("Validation metrics -- Gritti et al. (2019), Fig. 7 "
          "(valerophenone, gradient)")
    print("=" * 70)
    vm.print_metrics_table(metrics, time_unit='s')
    for m in metrics:
        print(f"  [diagnostic] {m['name']:10s}: calibrated mu_2 = "
              f"{m['mu2_sim_full']:.6g} s^2 vs. Table 2 target {m['mu2_ref']:.6g} s^2 "
              f"(fitted by construction -- hence the empty Delta mu_2 above)")
    vm.dump_metrics(output_path, 'Gritti2019_fig7',
                    'Gradient valerophenone', metrics, time_unit='s')
    return metrics


if __name__ == '__main__':
    main()
