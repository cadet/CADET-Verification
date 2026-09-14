# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 14.3 from:

    T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
    2nd ed., Springer, 2015, Chapter 14 ("Multicomponent Radial Flow
    Chromatography"), p. 199: "Simulation of binary frontal adsorption in
    inward flow RFC".

Self-contained script: model definition, run, comparison plot and
validation metrics. Further explanation on model and parameter selection is
provided under Gu2015_fig14_3.md.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from addict import Dict
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
# Step 1: paper's parameters, exactly as printed in the Fortran data dump
# on p. 199.
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
# See the module docstring, Sec. 3, for the full derivation.
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
    # COL_DISPERSION config value (Eq. 14.15 bookkeeping, docstring Sec. 3):
    # CADET computes Db_i(X) = COL_DISPERSION[i] * v(X), so we supply
    # Db_i|V=1 / v(X1) here to get Db_i(X1) = Db_i|V=1 exactly.
    p['col_dispersion_value'] = p['Db_V1'] / V_REF
    # FILM_DIFFUSION config value (Eq. 14.16 bookkeeping, docstring Sec. 3):
    # CADET computes k_i(X) = FILM_DIFFUSION[i] * v(X)^(1/3), so we supply
    # k_i|V=1 / v(X1)^(1/3) here to get k_i(X1) = k_i|V=1 exactly.
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
# Step 3: CADET model definition
# ---------------------------------------------------------------------------
def get_model(ncol=120, par_ncells=4, n_points=400, spatial_method='FV',
              dg_polydeg=4, col_dispersion_velocity_dep=True,
              film_diffusion_velocity_dep=True):
    """
    col_dispersion_velocity_dep: if True (default), Db_i(X) ~ v(X) via
        COL_DISPERSION_DEP='POWER_LAW' (Eq. 14.15). If False, COL_DISPERSION
        is held constant at Db_i|V=1 everywhere (an ablation of the paper's
        own model, since the paper does not offer a non-dependent variant of
        the dispersion relationship).
    film_diffusion_velocity_dep: if True (default), k_i(X) ~ v(X)^(1/3) via
        FILM_DIFFUSION_DEP='POWER_LAW' (Eq. 14.16). If False, falls back to
        the paper's own "iave=2" constant-Bi approximation (k_i evaluated
        once at V=0.5, see docstring Sec. 3)."""
    
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

    # --- Column ---
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
    if spatial_method == 'DG':
        col.discretization.SPATIAL_METHOD = 'DG'
        col.discretization.POLYDEG = dg_polydeg
        col.discretization.NELEM = ncol
        col.discretization.USE_COLLOCATION_DG = 0
        # Quadrature degree DG uses to integrate the (now spatially varying)
        # dispersion coefficient; required whenever COL_DISPERSION_DEP is
        # active with DG bulk discretization.
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


def run_model(cadet_path, output_path, ncol=240, par_ncells=8, dg_polydeg=None,
              n_points=400, fname='Gu2015_fig14_3.h5', **kwargs):

    model = get_model(ncol=ncol, par_ncells=par_ncells, dg_polydeg=dg_polydeg, n_points=n_points,**kwargs)

    sim = Cadet(install_path=cadet_path)
    sim.root.input = model.input
    sim.filename = os.path.join(output_path, fname)
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
        path = os.path.join(HERE, 'Gu2015_fig14_3_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    return data['time_dimensionless'], data['c1_dimensionless'], data['c2_dimensionless']


# ---------------------------------------------------------------------------
# Step 5: validation metrics -- the four unified numbers shared by all six
# case studies, see src/validation/validation_metrics.py.
#
# Both components of this figure are FRONTAL (breakthrough) responses: they
# approach a nonzero plateau at C/C0 = 1 instead of returning to baseline,
# so int(t*c dt)/int(c dt) would simply grow with the upper integration
# limit. Their moments are therefore taken of the underlying residence time
# distribution E = dF/dt of the normalised front F = c/c_plateau, evaluated
# by parts so that nothing has to be differentiated numerically; mu_1 is
# then the stoichiometric breakthrough time and mu_2 the variance of the
# front. See the validation_metrics module docstring for the derivation.
#
# Gu (2015) prints no moment table for this figure, so the reference for
# both moments is the digitized chromatogram itself. The mu_2 column uses
# the digitized second moment rather than the usual peak-height fallback:
# these curves have no peak, only a plateau, whose height error is
# degenerate (both curves are normalised to C/C0 = 1 by construction),
# whereas the width of the front is digitized reliably.
# ---------------------------------------------------------------------------
def saturated_inventory_dimensionless(component):
    """On-column inventory of one component at full feed saturation, in the
    same dimensionless units as int(c/C0 dtau).

    A frontal run retains a full saturated column load at the end of the
    simulation, so the outlet integral alone cannot close the mass balance;
    that inventory has to be added back. At equilibrium with the feed the
    column holds, per unit bed volume,

        eps_b*c_f  +  (1-eps_b)*( eps_p*c_f + (1-eps_p)*q*(c_f) )

    and, because the dimensionless time tau is scaled such that
    Q_FLOW*(X1-X0)/V_CHAR = eps_b*V_col exactly, dividing by eps_b*V_col*C0
    turns that into the dimensionless form used here -- no explicit column
    volume or flow rate is needed.

    q*(c_f) is the multi-component Langmuir loading at the feed composition,
    q_i = qmax_i*K_i*c_i / (1 + sum_j K_j*c_j) with K_j = ka_j/kd_j, i.e.
    exactly CADET's MULTI_COMPONENT_LANGMUIR at quasi-stationary equilibrium.
    """
    denom = 1.0 + sum(p['ka'] / p['kd'] * p['C0_phys'] for p in PAPER.values())
    p = PAPER[component]
    q_star = p['qmax'] * (p['ka'] / p['kd']) * p['C0_phys'] / denom
    per_bed_volume = (EPS_B * p['C0_phys']
                      + (1.0 - EPS_B) * (EPS_P * p['C0_phys']
                                         + (1.0 - EPS_P) * q_star))
    return per_bed_volume / (EPS_B * p['C0_phys'])


def compute_metrics(tau_sim, c1_sim, c2_sim, tau_ref, c1_ref, c2_ref):
    """Return the four unified metrics for both components."""
    metrics = []
    for name, component, c_sim, c_ref in (
        ('component_1', 1, c1_sim, c1_ref),
        ('component_2', 2, c2_sim, c2_ref),
    ):
        metrics.append(vm.standard_metrics(
            name=name,
            t_sim=tau_sim, c_sim=c_sim, t_ref=tau_ref, c_ref=c_ref,
            kind=vm.FRONTAL,
            mu2_fallback=vm.MU2_FROM_DIGITIZED,
            # Gu's curves are already C/C0-normalised, so no amplitude fit.
            amplitude=1.0,
            # Both components are fed at C/C0 = 1 for the whole run, so the
            # mass fed in dimensionless units is simply the run duration.
            mass_in=float(tau_sim[-1]),
            mass_retained=saturated_inventory_dimensionless(component),
            mass_label='outlet integral + saturated on-column inventory vs. mass fed '
                       '(a frontal run ends with the column loaded, so the retained '
                       'equilibrium inventory has to be added back)',
        ))
    return metrics


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
          f"Q={Q_FLOW:.4g} m^3/s, T_END={T_END:.4g} s")

    print("\nRunning CADET simulation...")

    model_kwargs = {
        'film_diffusion_velocity_dep': False,
        'col_dispersion_velocity_dep': True

    }

    spatial_method = 'DG'

    if spatial_method == 'DG':
        t_phys, outlet = run_model(cadet_path, output_path, ncol=64, par_ncells=2, dg_polydeg=4, spatial_method=spatial_method,
                                   n_points=400, fname=f'Gu2015_fig14_3_{spatial_method}.h5', **model_kwargs)
    elif spatial_method == 'FV':
        t_phys, outlet = run_model(cadet_path, output_path, ncol=256, par_ncells=8, dg_polydeg=None, spatial_method=spatial_method,
                                    n_points=400, fname=f'Gu2015_fig14_3_{spatial_method}.h5', **model_kwargs)

    tau_sim = dimless_time(t_phys)
    c1_sim = outlet[:, 0] / PAPER[1]['C0_phys']
    c2_sim = outlet[:, 1] / PAPER[2]['C0_phys']

    print("Loading digitized reference data...")
    tau_ref, c1_ref, c2_ref = load_digitized()

    print("Computing validation metrics...")
    metrics = compute_metrics(tau_sim, c1_sim, c2_sim, tau_ref, c1_ref, c2_ref)
    print("=" * 70)
    print("Validation metrics -- Gu (2015), Fig. 14.3 (binary frontal adsorption)")
    print("=" * 70)
    vm.print_metrics_table(metrics, time_unit='tau')
    by_name = {m['name']: m for m in metrics}

    # --- comparison plot ---
    fontsize = 15
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.plot(tau_sim, c1_sim, '-', color='tab:blue', label='comp. 1 (CADET)')
    ax.plot(tau_sim, c2_sim, '-', color='tab:orange', label='comp. 2 (CADET)')
    ax.plot(tau_ref, c1_ref, 'o', color='tab:blue', ms=3, mfc='none',
            label='comp. 1 (Gu 2015)')
    ax.plot(tau_ref, c2_ref, 's', color='tab:orange', ms=3, mfc='none',
            label='comp. 2 (Gu 2015)')
    ax.set_xlabel('Dimensionless time', fontsize=fontsize)#, ' + r'$\tau = v_{char}t/(X_1-X_0)$')
    ax.set_ylabel('Dimensionless concentration', fontsize=fontsize)#, ' + r'$C/C_0$')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.4)
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax.set_title('Gu (2015), Fig. 14.3 -- binary frontal adsorption, inward-flow RFC\n', fontsize=fontsize)

    # add a box with the chromatogram NRMSE of both components
    box_text = (f"NRMSE comp. 1: {by_name['component_1']['nrmse_%']:.2f}%\n"
                f"NRMSE comp. 2: {by_name['component_2']['nrmse_%']:.2f}%")
    ax.text(0.975, 0.95, box_text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top', horizontalalignment='right', multialignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.legend(fontsize=fontsize)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(output_path, f'Gu2015_fig14_3_comparison_{spatial_method}.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved comparison plot to {outpath}")

    vm.dump_metrics(output_path, 'Gu2015_fig14_3',
                    'Binary frontal adsorption', metrics, time_unit='tau')
    return metrics


if __name__ == '__main__':
    main()
