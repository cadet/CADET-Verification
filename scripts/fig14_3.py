# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 14.3 from:

    T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
    2nd ed., Springer, 2015, Chapter 14 ("Multicomponent Radial Flow
    Chromatography"), p. 199: "Simulation of binary frontal adsorption in
    inward flow RFC".

This is a self-contained script: model definition, run, comparison plot and
validation metrics.

===========================================================================
Model and Parameter identification
===========================================================================
Governing model: Sections 14.1-14.2 of Gu (2015). Binary frontal adsorption
(breakthrough) in an inward-flow radial flow column (RFC). Multi- component
Langmuir isotherm with rapid-equilibrium, film and pore diffusion,
spherical particles. Velocity dependent bulk dispersion and film diffusion.

Target: Fig. 14.3 (PDF p. 207 / printed p. 199), with the exact Fortran
RATERFC.FOR data-file dump for that figure reproduced on the same page:

    in/outward = -1 (inward flow)        V0 = 0.04000     iave = 0
    nsp=2  nelemb=15  nc=2  index=1 (frontal/breakthrough)
    epsip (eps_p) = 0.400   epsib (eps_b) = 0.400

    component   PeL     eta     Bi(V=1)   C0      a       b
        1       100.00  10.000  10.000    0.20000 1.000   2.000
        2        80.00   8.000   8.000    0.20000 10.000  20.000

    (a_i/b_i = 0.5 for both components -> thermodynamically consistent
    Langmuir saturation capacity, as required.)

"""
import os

import numpy as np
import matplotlib.pyplot as plt
from addict import Dict
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))

CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

# ---------------------------------------------------------------------------
# Step 1: paper's parameters, exactly as printed in the Fortran data dump
# on p. 199 (verified against the rendered figure page).
# ---------------------------------------------------------------------------
V0 = 0.04
TAU_MAX_SIM = 6.0  # simulate out to the figure's full x-axis range (paper's
                    # own run only went to tau_max=4.0, but the figure axis
                    # and plateau region extend to tau=6)

PAPER = {
    1: dict(PeL=100.0, eta=10.0, Bi_V1=10.0, C0=0.20, a=1.0, b=2.0),
    2: dict(PeL=80.0, eta=8.0, Bi_V1=8.0, C0=0.20, a=10.0, b=20.0),
}

# thermodynamic consistency check (a_i/b_i must be identical for all comps)
_qmax_dimless = [PAPER[i]['a'] / PAPER[i]['b'] for i in PAPER]
assert np.allclose(_qmax_dimless, _qmax_dimless[0]), \
    "Langmuir saturation capacities a_i/b_i are not thermodynamically consistent!"

# ---------------------------------------------------------------------------
# Step 2: reparameterization -- native radial geometry, physical (SI) scales.
# See Step 1 discussion above for the full derivation.
# ---------------------------------------------------------------------------
X1 = 0.05                             # outer column radius [m] (inward-flow inlet)
X0 = X1 * np.sqrt(V0 / (1.0 + V0))    # inner radius [m]; V0 = X0^2/(X1^2-X0^2)
BED_LENGTH = X1 - X0                  # radial bed thickness [m]
CYL_HEIGHT = 1.0                      # arbitrary cylinder height [m], fixed by flow rate
EPS_B = 0.40
EPS_P = 0.40
RP = 5.0e-5                           # particle radius [m]
V_REF = 1.0e-4                        # interstitial velocity at X1 (V=1) [m/s]
V_CHAR = 2.0 * V_REF * X1 / (X1 + X0) # transit-time characteristic velocity
CONC_UNIT = 1.0                       # reference concentration scale

for i, p in PAPER.items():
    p['Db_V1'] = V_REF * BED_LENGTH / p['PeL']
    p['Dp'] = p['eta'] * RP ** 2 * V_CHAR / (EPS_P * BED_LENGTH)
    p['k_V1'] = p['Bi_V1'] * p['eta'] * RP * V_CHAR / BED_LENGTH
    p['C0_phys'] = p['C0'] * CONC_UNIT
    p['ka'] = p['b'] / CONC_UNIT
    p['kd'] = 1.0
    p['qmax'] = (p['a'] / p['b']) * CONC_UNIT
    # COL_DISPERSION config value: with COL_DISPERSION_DEP='POWER_LAW' and
    # EXPONENT=1, CADET computes Db_i(X) = COL_DISPERSION[i] * v(X), so we
    # supply Db_i|V=1 / v(X1) here to get Db_i(X1) = Db_i|V=1 exactly.
    p['col_dispersion_value'] = p['Db_V1'] / V_REF
    # FILM_DIFFUSION config value: with FILM_DIFFUSION_DEP='POWER_LAW' and
    # EXPONENT=1/3 (per Eq. (14.16): k_i(V) ~ v^(1/3), the same relationship
    # expressed w.r.t. the actual local velocity that CADET's native
    # POWER_LAW dependency multiplies by), we supply k_i|V=1 / v(X1)^(1/3)
    # here to get k_i(X1) = k_i|V=1 exactly.
    p['film_diffusion_value'] = p['k_V1'] / V_REF ** (1.0 / 3.0)

# "iave=2" constant-Bi fallback (paper's own alternative to true
# position-dependent Bi_i(V), Eq. 14.16 at V=0.5) -- used when
# film_diffusion_velocity_dep=False in get_model() (see there).
IAVE2_FACTOR = ((1.0 - V0) / (0.5 + V0)) ** (1.0 / 6.0)
for i, p in PAPER.items():
    p['k_avg'] = p['k_V1'] * IAVE2_FACTOR

Q_FLOW = V_REF * X1 * 2.0 * np.pi * CYL_HEIGHT * EPS_B  # inlet flow [m^3/s]

T_END = TAU_MAX_SIM * BED_LENGTH / V_CHAR  # physical end time [s]


def dimless_time(t_phys):
    """Map physical simulation time [s] to the paper's tau = v_char*t/(X1-X0)."""
    return np.asarray(t_phys) * V_CHAR / BED_LENGTH


# ---------------------------------------------------------------------------
# Step 5: CADET model definition
# ---------------------------------------------------------------------------
def get_model(ncol=120, par_ncells=4, n_points=400, spatial_method='FV',
              dg_polydeg=4, col_dispersion_velocity_dep=True,
              film_diffusion_velocity_dep=True):
    """
    col_dispersion_velocity_dep: if True (default), Db_i(X) ~ v(X) via
        COL_DISPERSION_DEP='POWER_LAW' (Eq. 14.15, see Step 1 point (3)). If
        False, COL_DISPERSION is held constant at Db_i|V=1 everywhere (an
        ablation of the paper's own model, since the paper does not offer a
        non-dependent variant of the dispersion relationship).
    film_diffusion_velocity_dep: if True (default), k_i(X) ~ v(X)^(1/3) via
        FILM_DIFFUSION_DEP='POWER_LAW' (Eq. 14.16, see Step 1 point (6)). If
        False, falls back to the paper's own "iave=2" constant-Bi
        approximation (k_i evaluated once at V=0.5) used by earlier versions
        of this script."""
    
    m = Dict()
    m.input.model.nunits = 3

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

    # --- Inlet: frontal / breakthrough feed, both components held at C0_i
    # from time zero (index=1 in the paper's Fortran code) ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 2
    feed = [PAPER[1]['C0_phys'], PAPER[2]['C0_phys']]
    m.input.model.unit_000.sec_000.const_coeff = feed
    m.input.model.unit_000.sec_000.lin_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_000.quad_coeff = [0.0, 0.0]
    m.input.model.unit_000.sec_000.cube_coeff = [0.0, 0.0]

    # --- Column: CADET's native radial-flow geometry (see Step 1 for the
    # two bug fixes and the COL_DISPERSION_DEP mechanism this relies on) ---
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
    col.forward_flow = [0]
    col.init_c = [0.0, 0.0]

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    if spatial_method == 'DG':
        col.discretization.SPATIAL_METHOD = 'DG'
        col.discretization.POLYDEG = dg_polydeg
        col.discretization.NELEM = ncol
        col.discretization.USE_COLLOCATION_DG = 0
        # Required whenever COL_DISPERSION_DEP is set with DG bulk
        # discretization (quadrature degree for the variable-dispersion
        # integral) -- see Step 1 point (4).
        col.dispersion_spatial_dependence_polydeg = 2
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

    if spatial_method == 'FV':
        col.particle_type_000.discretization.SPATIAL_METHOD = 'FV'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.NCELLS = par_ncells
        col.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2
    elif spatial_method == 'DG':
        col.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.PAR_NELEM = par_ncells
        col.particle_type_000.discretization.PAR_POLYDEG = dg_polydeg

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
    m.input.solver.sections.nsec = 1
    m.input.solver.sections.section_continuity = []
    m.input.solver.sections.section_times = [0.0, T_END]
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-10
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, T_END, n_points)

    return m


def run_model(ncol=240, par_ncells=8, dg_polydeg=None, n_points=400, fname='fig14_3.h5', **kwargs):

    model = get_model(ncol=ncol, par_ncells=par_ncells, dg_polydeg=dg_polydeg, n_points=n_points,**kwargs)

    sim = Cadet(install_path=CADET_PATH)
    sim.root.input = model.input
    sim.filename = os.path.join(HERE, fname)
    sim.save()
    rc = sim.run_simulation()
    if rc.return_code != 0:
        raise RuntimeError(f"CADET failed: {getattr(rc, 'error_message', rc)}")
    sim.load_from_file()
    t = np.asarray(sim.root.output.solution.solution_times)
    outlet = np.asarray(sim.root.output.solution.unit_001.solution_outlet)  # (ntime, ncomp)
    return t, outlet


# ---------------------------------------------------------------------------
# Step 4: reference (digitized) data
# ---------------------------------------------------------------------------
def load_digitized(path=None):
    if path is None:
        path = os.path.join(HERE, 'fig14_3_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    return data['time_dimensionless'], data['c1_dimensionless'], data['c2_dimensionless']


# ---------------------------------------------------------------------------
# Step 6: validation metrics (adapted for a FRONTAL/breakthrough chromatogram
# -- classic pulse peak/first-moment analysis does not directly apply since
# both components approach a nonzero plateau (C/C0 -> 1) rather than
# returning to baseline; see per-metric notes below)
# ---------------------------------------------------------------------------
def t_at_level(t, c, level, rising_after=None):
    """First crossing time of c(t) through `level` (linear interpolation).
    If rising_after is given, only search t >= rising_after."""
    t = np.asarray(t)
    c = np.asarray(c)
    if rising_after is not None:
        mask = t >= rising_after
        t, c = t[mask], c[mask]
    idx = np.where(np.diff(np.sign(c - level)) > 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    t0, t1 = t[i], t[i + 1]
    c0, c1 = c[i], c[i + 1]
    frac = (level - c0) / (c1 - c0)
    return t0 + frac * (t1 - t0)


def compute_metrics(tau_sim, c1_sim, c2_sim, tau_ref, c1_ref, c2_ref):
    metrics = {}

    # interpolate CADET solution onto the reference (digitized) time grid
    # for MSE and plateau comparisons
    c1_sim_i = np.interp(tau_ref, tau_sim, c1_sim)
    c2_sim_i = np.interp(tau_ref, tau_sim, c2_sim)

    for name, c_sim, c_sim_i, c_ref in (
        ('component_1', c1_sim, c1_sim_i, c1_ref),
        ('component_2', c2_sim, c2_sim_i, c2_ref),
    ):
        m = {}

        # 1) Peak position (only meaningful for component 1, which shows
        #    competitive-Langmuir roll-up / overshoot above C/C0=1; component
        #    2 rises monotonically to its plateau and has no true interior
        #    peak, so this metric is reported as N/A there).
        if name == 'component_1':
            i_sim = np.argmax(c_sim)
            i_ref = np.argmax(c_ref)
            t_peak_sim = tau_sim[i_sim]
            t_peak_ref = tau_ref[i_ref]
            m['peak_time_sim'] = t_peak_sim
            m['peak_time_ref'] = t_peak_ref
            m['peak_time_relerr_%'] = 100 * abs(t_peak_sim - t_peak_ref) / t_peak_ref
            m['peak_height_sim'] = c_sim[i_sim]
            m['peak_height_ref'] = c_ref[i_ref]
            m['peak_height_relerr_%'] = 100 * abs(c_sim[i_sim] - c_ref[i_ref]) / c_ref[i_ref]
        else:
            m['peak_time_sim'] = np.nan
            m['peak_time_ref'] = np.nan
            m['peak_time_relerr_%'] = np.nan
            m['peak_height_sim'] = np.nan
            m['peak_height_ref'] = np.nan
            m['peak_height_relerr_%'] = np.nan

        # 2) "Elution time" analog: classic first-moment analysis
        #    (int t*c dt / int c dt) diverges for a step/frontal input since
        #    c(t) does not return to zero. We instead report the
        #    breakthrough time at 50% of the feed concentration (t50), the
        #    standard adapted metric for breakthrough curves.
        t50_sim = t_at_level(tau_sim, c_sim, 0.5)
        t50_ref = t_at_level(tau_ref, c_ref, 0.5)
        m['t50_sim'] = t50_sim
        m['t50_ref'] = t50_ref
        m['t50_relerr_%'] = 100 * abs(t50_sim - t50_ref) / t50_ref

        # 3) "Mass balance" analog: for a frontal input, overall mass
        #    balance is trivially satisfied once the column has reached
        #    saturation (all feed either passes through or is retained in
        #    the stationary phase, and outlet concentration must return to
        #    the feed value C/C0=1). We therefore check that both the
        #    simulated and the digitized curves converge to the same
        #    plateau value C/C0=1 at the end of the time window (deviation
        #    indicates either a units/scaling error or an under-resolved
        #    simulation).
        plateau_window = tau_ref >= (tau_ref.max() - 0.5)
        plateau_ref = np.nanmean(c_ref[plateau_window])
        plateau_sim = np.nanmean(c_sim_i[plateau_window])
        m['plateau_sim'] = plateau_sim
        m['plateau_ref'] = plateau_ref
        m['plateau_relerr_%'] = 100 * abs(plateau_sim - plateau_ref) / plateau_ref
        m['plateau_vs_feed_relerr_%'] = 100 * abs(plateau_sim - 1.0)

        # 4) Chromatogram MSE over the full digitized time window
        m['mse'] = np.nanmean((c_sim_i - c_ref) ** 2)

        metrics[name] = m

    return metrics


def print_metrics(metrics):
    for comp, m in metrics.items():
        print(f"\n--- {comp} ---")
        print(f"  Peak position   : sim={m['peak_time_sim']:.4g}  ref={m['peak_time_ref']:.4g}"
              f"  rel.err={m['peak_time_relerr_%']:.3g}%" if not np.isnan(m['peak_time_sim'])
              else "  Peak position   : N/A (monotonic breakthrough, no overshoot)")
        if not np.isnan(m['peak_height_sim']):
            print(f"  Peak height     : sim={m['peak_height_sim']:.4g}  ref={m['peak_height_ref']:.4g}"
                  f"  rel.err={m['peak_height_relerr_%']:.3g}%")
        print(f"  t50 (bt. time)  : sim={m['t50_sim']:.4g}  ref={m['t50_ref']:.4g}"
              f"  rel.err={m['t50_relerr_%']:.3g}%   [adapted 'elution time' metric]")
        print(f"  Plateau (C/C0)  : sim={m['plateau_sim']:.4g}  ref={m['plateau_ref']:.4g}"
              f"  rel.err={m['plateau_relerr_%']:.3g}%  (vs. feed=1: {m['plateau_vs_feed_relerr_%']:.3g}%)"
              "   [adapted 'mass balance' metric]")
        print(f"  Chromatogram MSE: {m['mse']:.4g}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Physical (SI) parameters derived from the paper's dimensionless groups:")
    for i, p in PAPER.items():
        print(f"  Component {i}: Db(V=1)={p['Db_V1']:.4g} m^2/s (COL_DISPERSION={p['col_dispersion_value']:.4g}), "
              f"Dp={p['Dp']:.4g} m^2/s, "
              f"k(V=1)={p['k_V1']:.4g} m/s (FILM_DIFFUSION={p['film_diffusion_value']:.4g}), "
              f"ka={p['ka']:.4g}, kd={p['kd']:.4g}, qmax={p['qmax']:.4g}")
    print(f"  X0={X0:.4g} m, X1={X1:.4g} m, bed_length={BED_LENGTH:.4g} m, "
          f"V_REF={V_REF:.4g} m/s, V_CHAR={V_CHAR:.4g} m/s, "
          f"Q={Q_FLOW:.4g} m^3/s, T_END={T_END:.4g} s")

    print("\nRunning CADET simulation...")

    model_kwargs = {
        'film_diffusion_velocity_dep': True,
        'col_dispersion_velocity_dep': True

    }

    spatial_method = 'DG'

    if spatial_method == 'DG':
        t_phys, outlet = run_model(ncol=128, par_ncells=8, dg_polydeg=4, n_points=400, fname=f'fig14_3_{spatial_method}.h5', **model_kwargs)
    elif spatial_method == 'FV':
        t_phys, outlet = run_model(ncol=240, par_ncells=8, dg_polydeg=None, n_points=400, fname=f'fig14_3_{spatial_method}.h5', **model_kwargs)

    tau_sim = dimless_time(t_phys)
    c1_sim = outlet[:, 0] / PAPER[1]['C0_phys']
    c2_sim = outlet[:, 1] / PAPER[2]['C0_phys']

    print("Loading digitized reference data...")
    tau_ref, c1_ref, c2_ref = load_digitized()

    print("Computing validation metrics...")
    metrics = compute_metrics(tau_sim, c1_sim, c2_sim, tau_ref, c1_ref, c2_ref)
    print_metrics(metrics)

    # --- comparison plot ---
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.plot(tau_sim, c1_sim, '-', color='tab:blue', label='Component 1 (CADET)')
    ax.plot(tau_sim, c2_sim, '-', color='tab:orange', label='Component 2 (CADET)')
    ax.plot(tau_ref, c1_ref, 'o', color='tab:blue', ms=3, mfc='none',
            label='Component 1 (Gu 2015)')
    ax.plot(tau_ref, c2_ref, 's', color='tab:orange', ms=3, mfc='none',
            label='Component 2 (Gu 2015)')
    ax.set_xlabel('Dimensionless time')#, ' + r'$\tau = v_{char}t/(X_1-X_0)$')
    ax.set_ylabel('Dimensionless concentration')#, ' + r'$C/C_0$')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.4)
    # ax.set_title('Gu (2015), Fig. 14.3 -- binary frontal adsorption, inward-flow RFC\n', fontsize=12)

    # add a box with MSE, peak position and height deviation
    peak_text = f"Peak Component 1: {metrics['component_1']['peak_time_relerr_%']:.4g}\nPeak Component 2: {metrics['component_2']['peak_time_relerr_%']:.4g}"
    height_text = f"Peak Deviation Component 1: {metrics['component_1']['peak_time_relerr_%']:.4g}\nPeak Deviation Component 2: {metrics['component_2']['peak_time_relerr_%']:.4g}\nHeight Deviation Component 1: {metrics['component_1']['peak_height_relerr_%']:.4g}\nHeight Deviation Component 2: {metrics['component_2']['peak_height_relerr_%']:.4g}"
    mse_text = f"MSE Component 1: {metrics['component_1']['mse']:.4g}\nMSE Component 2: {metrics['component_2']['mse']:.4g}"
    box_text = mse_text # + "\n" + peak_text + "\n" + height_text
    ax.text(0.95, 0.95, box_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.legend(loc='center right', fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(HERE, f'fig14_3_comparison{spatial_method}.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved comparison plot to {outpath}")
