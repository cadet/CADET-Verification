# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 14.5 from:

    T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
    2nd ed., Springer, 2015, Chapter 14 ("Multicomponent Radial Flow
    Chromatography"), p. 201, Fig. 14.5: "Binary elution with an inert
    mobile phase in inward flow RFC".

Self-contained script: model definition, run, comparison plot, and
validation metrics. Further explanation on model and parameter selection is
provided under Gu2015_fig14_5.md.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from addict import Dict
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Paper's parameters, exactly as printed in the GUI screenshot (Fig. 14.5,
# PDF p. 209 / printed p. 201).
# ---------------------------------------------------------------------------
V0 = 0.04
TAU_IMP = 0.5     # dimensionless injection (pulse) duration
TAU_MAX_SIM = 16.0  # paper's own printed x-axis range (tmax = 16)

PAPER = {
    1: dict(PeL=100.0, eta=10.0, Bi_V1=10.0, C0=0.20, a=1.0, b=2.0),
    2: dict(PeL=120.0, eta=12.0, Bi_V1=12.0, C0=0.20, a=10.0, b=20.0),
}

# thermodynamic consistency check (a_i/b_i must be identical for all comps)
_qmax_dimless = [PAPER[i]['a'] / PAPER[i]['b'] for i in PAPER]
assert np.allclose(_qmax_dimless, _qmax_dimless[0]), \
    "Langmuir saturation capacities a_i/b_i are not thermodynamically consistent!"

# ---------------------------------------------------------------------------
# Reparameterization -- native radial geometry, physical (SI) scales.
# See "Reparameterization" in the module docstring above for the derivation.
# ---------------------------------------------------------------------------
X1 = 0.05                             # outer column radius [m] (inward-flow inlet)
X0 = X1 * np.sqrt(V0 / (1.0 + V0))    # inner radius [m]; V0 = X0^2/(X1^2-X0^2)
BED_LENGTH = X1 - X0                  # radial bed thickness [m]
CYL_HEIGHT = 1.0                      # arbitrary cylinder height [m]
EPS_B = 0.40
EPS_P = 0.40
RP = 5.0e-5                           # particle radius [m]
V_REF = 1.0e-4                        # interstitial velocity at X1 (V=1) [m/s]
V_CHAR = 2.0 * V_REF * X1 / (X1 + X0)  # transit-time characteristic velocity
CONC_UNIT = 1.0

for i, p in PAPER.items():
    p['Db_V1'] = V_REF * BED_LENGTH / p['PeL']
    p['Dp'] = p['eta'] * RP ** 2 * V_CHAR / (EPS_P * BED_LENGTH)
    p['k_V1'] = p['Bi_V1'] * p['eta'] * RP * V_CHAR / BED_LENGTH
    p['C0_phys'] = p['C0'] * CONC_UNIT
    p['ka'] = p['b'] / CONC_UNIT
    p['kd'] = 1.0
    p['qmax'] = (p['a'] / p['b']) * CONC_UNIT
    # COL_DISPERSION config value for COL_DISPERSION_DEP='POWER_LAW',
    # EXPONENT=1 (see "CADET-specific bookkeeping" in the module docstring):
    # Db_i(X) = COL_DISPERSION[i]*v(X), so supply Db_i|V=1/v(X1).
    p['col_dispersion_value'] = p['Db_V1'] / V_REF
    # FILM_DIFFUSION config value for FILM_DIFFUSION_DEP='POWER_LAW',
    # EXPONENT=1/3 (Eq. 14.16: k_i(V) ~ v^(1/3); see "CADET-specific
    # bookkeeping" above), so supply k_i|V=1/v(X1)^(1/3).
    p['film_diffusion_value'] = p['k_V1'] / V_REF ** (1.0 / 3.0)

# "iave=2" constant-Bi fallback (paper's own alternative to true
# position-dependent Bi_i(V), Eq. 14.16 at V=0.5) -- used when
# film_diffusion_velocity_dep=False in get_model() (see there).
IAVE2_FACTOR = ((1.0 - V0) / (0.5 + V0)) ** (1.0 / 6.0)
for i, p in PAPER.items():
    p['k_avg'] = p['k_V1'] * IAVE2_FACTOR

Q_FLOW = V_REF * X1 * 2.0 * np.pi * CYL_HEIGHT * EPS_B  # inlet flow [m^3/s]

TAU_IMP_PHYS = TAU_IMP * BED_LENGTH / V_CHAR
T_END = TAU_MAX_SIM * BED_LENGTH / V_CHAR  # physical end time [s]


def dimless_time(t_phys):
    """Map physical simulation time [s] to the paper's tau = v_char*t/(X1-X0)."""
    return np.asarray(t_phys) * V_CHAR / BED_LENGTH


# ---------------------------------------------------------------------------
# CADET model definition
# ---------------------------------------------------------------------------
def get_model(ncol=120, par_ncells=4, n_points=800, bulk_discretization='FV',
              dg_polydeg=4, col_dispersion_velocity_dep=True,
              film_diffusion_velocity_dep=True):
    """
    col_dispersion_velocity_dep: if True (default), Db_i(X) ~ v(X) via
        COL_DISPERSION_DEP='POWER_LAW'. If False, COL_DISPERSION is held
        constant at Db_i|V=1 everywhere.
    film_diffusion_velocity_dep: if True (default), k_i(X) ~ v(X)^(1/3) via
        FILM_DIFFUSION_DEP='POWER_LAW'. If False, falls back to the paper's
        own "iave=2" constant-Bi approximation (k_i evaluated once at
        V=0.5)."""
    
    m = Dict()
    m.input.model.nunits = 3

    # Two sections: 0 = injection pulse (0 < tau < tau_imp), feed = C0_i;
    # 1 = elution/wash (tau > tau_imp), feed = 0 (pure inert mobile phase),
    # per Eq. (14.12), index = 2. No "priming" section is needed here: V=1
    # is always the inlet by construction (see Step 1 discussion), so a
    # plain forward-flow axial column already represents inward-flow RFC.
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, Q_FLOW,
        1.0, 2.0, -1.0, -1.0, Q_FLOW,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # --- Inlet: pulse of duration tau_imp at C0_i, then pure carrier (0) ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 2
    feed_pulse = [PAPER[1]['C0_phys'], PAPER[2]['C0_phys']]
    feed_zero = [0.0, 0.0]
    for sec, feed in (('sec_000', feed_pulse), ('sec_001', feed_zero)):
        m.input.model.unit_000[sec].const_coeff = feed
        m.input.model.unit_000[sec].lin_coeff = [0.0, 0.0]
        m.input.model.unit_000[sec].quad_coeff = [0.0, 0.0]
        m.input.model.unit_000[sec].cube_coeff = [0.0, 0.0]

    # --- Column: CADET's native radial-flow geometry ---
    col = Dict()
    col.unit_type = 'COLUMN_MODEL_1D'
    col.geometry = 'RADIAL_FLOW_CYLINDER_SHELL'
    col.ncomp = 2
    col.npartype = 1
    col.par_type_volfrac = 1
    col.cross_section_area_outer = 2.0 * np.pi * X1 * CYL_HEIGHT
    col.cylinder_height = CYL_HEIGHT
    col.bed_length = BED_LENGTH
    col.col_porosity = EPS_B
    if col_dispersion_velocity_dep:
        col.col_dispersion = [PAPER[1]['col_dispersion_value'], PAPER[2]['col_dispersion_value']]
        col.col_dispersion_dep = 'POWER_LAW'
        col.col_dispersion_dep_exponent = 1.0
    else:
        col.col_dispersion = [PAPER[1]['Db_V1'], PAPER[2]['Db_V1']]
    # Gu (2015) uses inward flow, i.e. from the outer to the inner radius, which is
    # the default direction of CADET's radial flow geometry.
    col.forward_flow = [1]
    col.init_c = [0.0, 0.0]

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    if bulk_discretization == 'DG':
        col.discretization.SPATIAL_METHOD = 'DG'
        col.discretization.POLYDEG = dg_polydeg
        col.discretization.NELEM = ncol
        col.discretization.USE_COLLOCATION_DG = 0
        col.dispersion_spatial_dependence_polydeg = dg_polydeg
    else:
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

    # --- Particles: GENERAL_RATE_PARTICLE (film + pore diffusion, spherical),
    # instantaneous local equilibrium multicomponent Langmuir ---
    col.particle_type_000.nbound = [1, 1]
    col.particle_type_000.init_cp = [0.0, 0.0]
    col.particle_type_000.init_cs = [0.0, 0.0]

    col.particle_type_000.has_film_diffusion = 1
    if film_diffusion_velocity_dep:
        col.particle_type_000.film_diffusion = [PAPER[1]['film_diffusion_value'], PAPER[2]['film_diffusion_value']]
        col.particle_type_000.film_diffusion_dep = 'POWER_LAW'
        col.particle_type_000.film_diffusion_dep_exponent = 1.0 / 3.0
    else:
        col.particle_type_000.film_diffusion = [PAPER[1]['k_avg'], PAPER[2]['k_avg']]
    col.particle_type_000.has_pore_diffusion = 1
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.par_geom = 'SPHERE'
    col.particle_type_000.par_coreradius = 0.0
    col.particle_type_000.par_porosity = EPS_P
    col.particle_type_000.par_radius = RP
    col.particle_type_000.pore_diffusion = [PAPER[1]['Dp'], PAPER[2]['Dp']]
    col.particle_type_000.surface_diffusion = [0.0, 0.0]

    col.particle_type_000.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    col.particle_type_000.adsorption.is_kinetic = 0  # rapid/local equilibrium
    col.particle_type_000.adsorption.mcl_ka = [PAPER[1]['ka'], PAPER[2]['ka']]
    col.particle_type_000.adsorption.mcl_kd = [PAPER[1]['kd'], PAPER[2]['kd']]
    col.particle_type_000.adsorption.mcl_qmax = [PAPER[1]['qmax'], PAPER[2]['qmax']]

    if bulk_discretization == 'DG':
        col.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.PAR_NELEM = par_ncells
        col.particle_type_000.discretization.PAR_POLYDEG = dg_polydeg
    else:
        col.particle_type_000.discretization.SPATIAL_METHOD = 'FV'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.NCELLS = par_ncells
        col.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2

    m.input.model.unit_001 = col

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
    m.input.solver.sections.nsec = 2
    m.input.solver.sections.section_continuity = [0]
    m.input.solver.sections.section_times = [0.0, TAU_IMP_PHYS, T_END]
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-10
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, T_END, n_points)

    return m


def run_model(cadet_path, output_path, ncol=120, par_ncells=4, n_points=800,
              fname='Gu2015_fig14_5.h5', **kwargs):

    model = get_model(ncol=ncol, par_ncells=par_ncells, n_points=n_points, **kwargs)

    c = Cadet(install_path=cadet_path)
    c.root.input = model.input
    c.filename = os.path.join(output_path, fname)
    c.save()
    rc = c.run_simulation()
    if rc.return_code != 0:
        raise RuntimeError(f"CADET failed: {getattr(rc, 'error_message', rc)}")
    c.load_from_file()
    t = np.asarray(c.root.output.solution.solution_times)
    outlet = np.asarray(c.root.output.solution.unit_001.solution_outlet)  # (ntime, ncomp)
    return t, outlet


# ---------------------------------------------------------------------------
# Reference (digitized) data
# ---------------------------------------------------------------------------
def load_digitized(path=None):
    if path is None:
        path = os.path.join(HERE, 'Gu2015_fig14_5_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    names = data.dtype.names

    def pick(*candidates):
        for c in candidates:
            if c in names:
                return data[c]
        return None
    tau = pick('tau', 'time_dimensionless', 'x', 'f0')
    c1 = pick('c1_dimensionless', 'component_1', 'c1', 'f1')
    c2 = pick('c2_dimensionless', 'component_2', 'c2', 'f2')
    if tau is None or c1 is None or c2 is None:
        raise RuntimeError(f"Could not identify columns in digitized CSV; found: {names}")
    return tau, c1, c2


# ---------------------------------------------------------------------------
# Validation metrics (classic pulse/elution chromatogram analysis -- both
# components return to baseline, so peak position, first-moment elution
# time, mass balance, and MSE are all directly applicable).
# ---------------------------------------------------------------------------
def compute_metrics(tau_sim, c1_sim, c2_sim, tau_ref, c1_ref, c2_ref):
    metrics = {}

    for name, c_sim_native, c_ref, C0 in (
        ('component_1', c1_sim, c1_ref, PAPER[1]['C0']),
        ('component_2', c2_sim, c2_ref, PAPER[2]['C0']),
    ):
        m = {}

        # interpolate CADET solution onto the reference (digitized) tau grid
        c_sim_i = np.interp(tau_ref, tau_sim, c_sim_native)

        # 1) Peak position (time of maximum concentration)
        i_sim = np.argmax(c_sim_native)
        i_ref = np.nanargmax(c_ref)
        t_peak_sim = tau_sim[i_sim]
        t_peak_ref = tau_ref[i_ref]
        m['peak_time_sim'] = t_peak_sim
        m['peak_time_ref'] = t_peak_ref
        m['peak_time_relerr_%'] = 100 * abs(t_peak_sim - t_peak_ref) / t_peak_ref
        m['peak_height_sim'] = c_sim_native[i_sim]
        m['peak_height_ref'] = c_ref[i_ref]
        m['peak_height_relerr_%'] = 100 * abs(c_sim_native[i_sim] - c_ref[i_ref]) / c_ref[i_ref]

        # 2) Elution time (first moment): int(t*c dt) / int(c dt), over the
        #    full simulated time window (CADET's own dense time grid, not
        #    the sparser digitized grid, for accuracy).
        def first_moment(t, c):
            c = np.clip(c, 0.0, None)
            return np.trapz(t * c, t) / np.trapz(c, t)

        tm_sim = first_moment(tau_sim, c_sim_native)
        tm_ref = first_moment(tau_ref, np.nan_to_num(c_ref))
        m['moment_time_sim'] = tm_sim
        m['moment_time_ref'] = tm_ref
        m['moment_time_relerr_%'] = 100 * abs(tm_sim - tm_ref) / tm_ref

        # 3) Mass balance: injected mass vs. eluted mass, both expressed in
        #    C/C0-normalized dimensionless units (c1_sim, c2_sim, c1_ref,
        #    c2_ref are all already C0-normalized). The injected pulse has
        #    normalized concentration 1 for duration tau_imp, so its
        #    normalized "mass" (area) is exactly tau_imp. comp. 2's
        #    long tail is not fully captured within tau_max=16
        #    (matching the paper's own truncated plot window),
        #    so a residual undershoot here is expected and consistent
        #    with the reference curve, not necessarily a bug.
        injected = TAU_IMP
        eluted_sim = np.trapz(np.clip(c_sim_native, 0.0, None), tau_sim)
        eluted_ref = np.trapz(np.nan_to_num(np.clip(c_ref, 0.0, None)), tau_ref)
        m['mass_injected_dimless'] = injected
        m['mass_eluted_sim_dimless'] = eluted_sim
        m['mass_eluted_ref_dimless'] = eluted_ref
        m['mass_balance_relerr_%'] = 100 * abs(eluted_sim - injected) / injected
        m['mass_sim_vs_ref_relerr_%'] = 100 * abs(eluted_sim - eluted_ref) / eluted_ref

        # 4) Chromatogram MSE over the full digitized time window
        m['mse'] = np.nanmean((c_sim_i - c_ref) ** 2)
        # Normalized RMSE (% of the reference peak height) -- raw MSE is NOT
        # comparable across components with different amplitude scales
        # (comp. 1 peaks ~0.58, comp. 2 ~0.16 -- almost 4x apart) or
        # across scripts with different C/C0 ranges -- same convention as
        # the Gritti case studies (Gritti2019_fig6/7/8.py).
        m['nrmse_%'] = 100 * np.sqrt(m['mse']) / m['peak_height_ref']

        metrics[name] = m

    return metrics


def print_metrics(metrics):
    for comp, m in metrics.items():
        print(f"\n--- {comp} ---")
        print(f"  Peak position   : sim={m['peak_time_sim']:.4g}  ref={m['peak_time_ref']:.4g}"
              f"  rel.err={m['peak_time_relerr_%']:.3g}%")
        print(f"  Peak height     : sim={m['peak_height_sim']:.4g}  ref={m['peak_height_ref']:.4g}"
              f"  rel.err={m['peak_height_relerr_%']:.3g}%")
        print(f"  Elution time    : sim={m['moment_time_sim']:.4g}  ref={m['moment_time_ref']:.4g}"
              f"  rel.err={m['moment_time_relerr_%']:.3g}%   [first moment int(t c dt)/int(c dt)]")
        print(f"  Mass balance    : injected={m['mass_injected_dimless']:.4g}  "
              f"eluted(sim)={m['mass_eluted_sim_dimless']:.4g}  "
              f"rel.err(sim vs inj)={m['mass_balance_relerr_%']:.3g}%   "
              f"rel.err(sim vs ref)={m['mass_sim_vs_ref_relerr_%']:.3g}%")
        print(f"  Chromatogram MSE: {m['mse']:.4g}  (NRMSE={m['nrmse_%']:.2f}% of peak height)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
from pathlib import Path
CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "output" / "validation"

def main(cadet_path=CADET_PATH, output_path=OUTPUT_PATH):

    os.makedirs(output_path, exist_ok=True)

    print("Physical (SI) parameters derived from the paper's dimensionless groups:")
    for i, p in PAPER.items():
        print(f"  Component {i}: Db(V=1)={p['Db_V1']:.4g} m^2/s (COL_DISPERSION={p['col_dispersion_value']:.4g}), "
              f"Dp={p['Dp']:.4g} m^2/s, "
              f"k(V=1)={p['k_V1']:.4g} m/s (FILM_DIFFUSION={p['film_diffusion_value']:.4g}), "
              f"ka={p['ka']:.4g}, kd={p['kd']:.4g}, qmax={p['qmax']:.4g}")
    print(f"  X0={X0:.4g} m, X1={X1:.4g} m, bed_length={BED_LENGTH:.4g} m, "
          f"V_REF={V_REF:.4g} m/s, V_CHAR={V_CHAR:.4g} m/s, "
          f"Q={Q_FLOW:.4g} m^3/s, tau_imp={TAU_IMP}, T_END={T_END:.4g} s")

    print("\nRunning CADET simulation")

    model_kwargs = {
        'film_diffusion_velocity_dep': True,
        'col_dispersion_velocity_dep': True

    }

    spatial_method = 'DG'

    if spatial_method == 'DG':
        t_phys, outlet = run_model(cadet_path, output_path, ncol=64, par_ncells=2, dg_polydeg=4, bulk_discretization=spatial_method,
                                   n_points=400, fname=f'Gu2015_fig14_5_{spatial_method}.h5', **model_kwargs)
    elif spatial_method == 'FV':
        t_phys, outlet = run_model(cadet_path, output_path, ncol=256, par_ncells=8, dg_polydeg=None, bulk_discretization=spatial_method,
                                   n_points=400, fname=f'Gu2015_fig14_5_{spatial_method}.h5', **model_kwargs)

    tau_sim = dimless_time(t_phys)
    c1_sim = outlet[:, 0] / PAPER[1]['C0_phys']
    c2_sim = outlet[:, 1] / PAPER[2]['C0_phys']

    print("Loading digitized reference data...")
    tau_ref, c1_ref, c2_ref = load_digitized()

    print("Computing validation metrics...")
    metrics = compute_metrics(tau_sim, c1_sim, c2_sim, tau_ref, c1_ref, c2_ref)
    print_metrics(metrics)

    # --- comparison plot ---
    fontsize = 15
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tau_sim, c1_sim, '-', color='tab:red', label='comp. 1 (CADET)')
    ax.plot(tau_sim, c2_sim, '-', color='black', label='comp. 2 (CADET)')
    ax.plot(tau_ref, c1_ref, 'o', color='tab:red', ms=3, mfc='none',
            label='comp. 1 (Gu 2015)')
    ax.plot(tau_ref, c2_ref, 's', color='black', ms=3, mfc='none',
            label='comp. 2 (Gu 2015)')
    ax.set_xlabel('Dimensionless time', fontsize=fontsize)#, ' + r'$\tau = vt/(X_1-X_0)$')
    ax.set_ylabel('Dimensionless concentration', fontsize=fontsize)#, ' + r'$C/C_0$')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 0.65)
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax.set_title('Gu (2015), Fig. 14.5 -- binary elution with inert mobile phase, inward-flow RFC', fontsize=fontsize)

    # add a box with MSE, peak position and height deviation
    peak_text = f"Peak comp. 1: {metrics['component_1']['peak_time_relerr_%']:.4g}\nPeak comp. 2: {metrics['component_2']['peak_time_relerr_%']:.4g}"
    height_text = f"Peak Deviation comp. 1: {metrics['component_1']['peak_time_relerr_%']:.4g}\nPeak Deviation comp. 2: {metrics['component_2']['peak_time_relerr_%']:.4g}\nHeight Deviation comp. 1: {metrics['component_1']['peak_height_relerr_%']:.4g}\nHeight Deviation comp. 2: {metrics['component_2']['peak_height_relerr_%']:.4g}"
    mse_text = f"NRMSE comp. 1: {metrics['component_1']['nrmse_%']:.2f}%\n"+f"NRMSE comp. 2: {metrics['component_2']['nrmse_%']:.2f}%"
    box_text = mse_text # + "\n" + peak_text + "\n" + height_text
    ax.text(0.975, 0.6, box_text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top', horizontalalignment='right', multialignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.legend(loc='upper right', fontsize=fontsize)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(output_path, f'Gu2015_fig14_5_comparison_{spatial_method}.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved comparison plot to {outpath}")

if __name__ == '__main__':
    main()
