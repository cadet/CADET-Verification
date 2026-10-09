# -*- coding: utf-8 -*-
"""
Reproduction of the m-xylene, Sh = 2 curve of Fig. 2 from:

    S. Brandani, "Kinetics of liquid phase batch adsorption experiments",
    Adsorption 27 (2021) 353-368, https://doi.org/10.1007/s10450-020-00258-9
    (open access). The underlying system is the finite bath experiment of
    Santacesaria et al., Ind. Eng. Chem. Process Des. Dev. 21 (1982) 440-445.

A batch uptake experiment: a closed, well stirred vessel holding a known
volume of solution and a known mass of adsorbent, i.e. CADET's FINITE_BATH
with a general rate particle. Fig. 2 is itself a simulation, so this is a
code-to-code comparison and not a validation against measurement; see
Brandani2021_fig2.md, which also derives the parameter mapping used below.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from addict import Dict
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))

# The metric definitions shared by all validation case studies live one
# directory up. Adding it to sys.path keeps this script runnable both
# directly and as an import.
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
import validation_metrics as vm  # noqa: E402


# ---------------------------------------------------------------------------
# Parameters as printed in Brandani's Table 1, m-xylene column
# ---------------------------------------------------------------------------
EPS_P = 0.20            # macropore void fraction of the beads [-]
TORTUOSITY = 2.15       # tortuosity [-]
RP = 0.65e-3            # bead radius [m]
D_M = 2.15e-9           # molecular diffusivity [m^2/s]
C0 = 250.0              # initial liquid concentration [mol/m^3]
RHO_S = 1400.0          # bead envelope density [kg/m^3]
M_S = 40.0e-3           # mass of adsorbent [kg]
V_F = 200.0e-6          # volume of fluid [m^3]
QS = 2450.0             # Langmuir saturation capacity [mol/m^3 of solid]
B_LANG = 0.006          # Langmuir affinity [m^3/mol]

T_END = 3000.0          # the figure's full time axis [s]

# --- derived CADET parameters, see the markdown file -----------------------
# Multiplying Brandani's Eq. 10 and CADET's bead equation into the same form
# identifies D_p = D_m/tau; the Sh = 2 film coefficient is k_F = D_m/R_p,
# which reproduces the 3.31e-6 m/s of Table 1.
PORE_DIFFUSION = D_M / TORTUOSITY
FILM_DIFFUSION = D_M / RP

V_S = M_S / RHO_S              # bead volume [m^3]; RHO_S is the envelope density
ALPHA = V_S / V_F              # Brandani's volume ratio [-]
EPS_B = V_F / (V_F + V_S)      # CADET's BULK_POROSITY [-]

# CADET's MULTI_COMPONENT_LANGMUIR gives q = qmax*(ka/kd)*c/(1+(ka/kd)*c) at
# equilibrium, so the affinity is b = ka/kd.
MCL_KA = B_LANG
MCL_KD = 1.0


def particle_phase_concentration(c_liquid):
    """Brandani's Q from the liquid concentration via his Eq. 2.

    The vessel is closed, so V_F*c0 = V_F*c(t) + V_S*Q(t) holds at every
    instant and Q = (c0 - c)/alpha. Fig. 2 plots Q, not c.
    """
    return (C0 - np.asarray(c_liquid, dtype=float)) / ALPHA


def equilibrium_state():
    """(c_inf, Q_inf) where the isotherm and the mass balance agree.

    Used only as a sanity print: it is an independent check that the
    parameter set reproduces the plateau of the published figure.
    """
    from scipy.optimize import brentq
    q_eq = lambda c: EPS_P * c + (1.0 - EPS_P) * QS * B_LANG * c / (1.0 + B_LANG * c)
    c_inf = brentq(lambda c: C0 - ALPHA * q_eq(c) - c, 1e-12, C0)
    return c_inf, q_eq(c_inf)


# ---------------------------------------------------------------------------
# CADET model definition
# ---------------------------------------------------------------------------
def get_model(par_nelem=8, par_polydeg=4, n_points=601, spatial_method='DG',
              par_ncells=32):
    """Closed finite bath with one general rate particle type.

    The vessel is closed, so both flow rates are zero. CADET still wants the
    inlet and outlet units of a flow sheet, and they simply stay inactive;
    LIQUID_VOLUME likewise drops out of the bulk balance when F_in = F_out = 0
    and is given only to document the experiment.
    """
    m = Dict()
    m.input.model.nunits = 3

    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, 0.0,
        1.0, 2.0, -1.0, -1.0, 0.0,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # --- Inlet: inactive, the vessel is closed ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 1
    m.input.model.unit_000.sec_000.const_coeff = [0.0]
    m.input.model.unit_000.sec_000.lin_coeff = [0.0]
    m.input.model.unit_000.sec_000.quad_coeff = [0.0]
    m.input.model.unit_000.sec_000.cube_coeff = [0.0]

    # --- Finite bath ---
    bath = Dict()
    bath.unit_type = 'FINITE_BATH'
    bath.ncomp = 1
    bath.npartype = 1
    bath.par_type_volfrac = 1
    bath.liquid_volume = V_F
    bath.bulk_porosity = EPS_B
    # The experiment starts with the solute in the bulk and clean beads.
    bath.init_c = [C0]

    bath.discretization.USE_ANALYTIC_JACOBIAN = 1

    # --- Particle: spherical, film + pore diffusion, Langmuir isotherm in
    # local equilibrium with the pore fluid (Brandani's Eq. 11) ---
    bath.particle_type_000.nbound = [1]
    bath.particle_type_000.init_cp = [0.0]
    bath.particle_type_000.init_cs = [0.0]

    bath.particle_type_000.has_film_diffusion = 1
    bath.particle_type_000.film_diffusion = [FILM_DIFFUSION]
    bath.particle_type_000.has_pore_diffusion = 1
    bath.particle_type_000.has_surface_diffusion = 0
    bath.particle_type_000.par_geom = 'SPHERE'
    bath.particle_type_000.par_coreradius = 0.0
    bath.particle_type_000.par_porosity = EPS_P
    bath.particle_type_000.par_radius = RP
    bath.particle_type_000.pore_diffusion = [PORE_DIFFUSION]
    bath.particle_type_000.surface_diffusion = [0.0]

    bath.particle_type_000.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    bath.particle_type_000.adsorption.is_kinetic = 0   # local equilibrium
    bath.particle_type_000.adsorption.mcl_ka = [MCL_KA]
    bath.particle_type_000.adsorption.mcl_kd = [MCL_KD]
    bath.particle_type_000.adsorption.mcl_qmax = [QS]

    bath.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT'
    if spatial_method == 'DG':
        bath.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
        bath.particle_type_000.discretization.PAR_POLYDEG = par_polydeg
        bath.particle_type_000.discretization.PAR_NELEM = par_nelem
    elif spatial_method == 'FV':
        bath.particle_type_000.discretization.SPATIAL_METHOD = 'FV'
        bath.particle_type_000.discretization.NCELLS = par_ncells

    m.input.model.unit_001 = bath

    m.input.model.unit_002.ncomp = 1
    m.input.model.unit_002.unit_type = 'OUTLET'

    # --- return group ---
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_000.write_solution_outlet = 0
    m.input['return'].unit_001.write_solution_outlet = 1
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


def run_model(cadet_path, output_path, fname='Brandani2021_fig2.h5', **kwargs):
    model = get_model(**kwargs)

    sim = Cadet(install_path=cadet_path)
    sim.root.input = model.input
    sim.filename = os.path.join(output_path, fname)
    sim.save()
    rc = sim.run_simulation()
    if rc.return_code != 0:
        raise RuntimeError(f"CADET failed: {getattr(rc, 'error_message', rc)}")
    sim.load_from_file()
    t = np.asarray(sim.root.output.solution.solution_times)
    outlet = np.asarray(sim.root.output.solution.unit_001.solution_outlet)
    return t, outlet.reshape(len(t), -1)[:, 0]


# ---------------------------------------------------------------------------
# Reference data, extracted from the figure
# ---------------------------------------------------------------------------
def load_digitized(path=None):
    if path is None:
        path = os.path.join(HERE, 'Brandani2021_fig2_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    return data['time_s'], data['Q_mol_per_m3']


# ---------------------------------------------------------------------------
# Metric
#
# Only the NRMSE of the shared table carries over to a batch uptake curve.
# It is computed here with the definition of validation_metrics.py --
# root mean square deviation on the overlap of the two time axes, with the
# simulation interpolated onto the reference grid, normalised by max|Q_ref|
# -- rather than through vm.standard_metrics, whose other three columns are
# undefined for this experiment: an uptake curve has neither a peak nor an
# elution time, and a closed vessel has no injected mass to balance against
# an outlet integral. See Brandani2021_fig2.md.
# ---------------------------------------------------------------------------
def compute_nrmse(t_sim, q_sim, t_ref, q_ref):
    t_sim, q_sim = vm.clean_curve(t_sim, q_sim)
    t_ref, q_ref = vm.clean_curve(t_ref, q_ref)
    lo, hi = vm.common_window(t_sim, t_ref)
    t_ref_w, q_ref_w = vm.restrict(t_ref, q_ref, lo, hi)
    q_sim_on_ref = np.interp(t_ref_w, t_sim, q_sim)

    ref_amplitude = float(np.nanmax(np.abs(q_ref_w)))
    mse = float(np.nanmean((q_sim_on_ref - q_ref_w) ** 2))
    return {
        'window': (lo, hi),
        'n_ref': len(t_ref_w),
        'ref_amplitude': ref_amplitude,
        'mse': mse,
        'nrmse_%': 100.0 * np.sqrt(mse) / ref_amplitude,
        'max_abs_dev': float(np.nanmax(np.abs(q_sim_on_ref - q_ref_w))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# The finite bath unit operation was added in CADET-Core commit 31229b58 on
# the feature/finiteBath branch, so this has to point at a build at least that
# recent. The default is the shared install the other case studies use, which
# only carries the unit operation once that branch is merged; until then,
# point it at a build of the branch.
CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "output" / "validation"


def main(cadet_path=CADET_PATH, output_path=OUTPUT_PATH):
    os.makedirs(output_path, exist_ok=True)

    c_inf, q_inf = equilibrium_state()
    print("Parameters derived from Brandani (2021), Table 1, m-xylene:")
    print(f"  PORE_DIFFUSION  = D_m/tau = {PORE_DIFFUSION:.4g} m^2/s")
    print(f"  FILM_DIFFUSION  = D_m/R_p = {FILM_DIFFUSION:.4g} m^2/s   (Sh = 2)")
    print(f"  BULK_POROSITY   = {EPS_B:.4f}   (alpha = V_S/V_F = {ALPHA:.6f})")
    print(f"  MCL_KA/MCL_KD   = {MCL_KA:.4g}/{MCL_KD:.4g} = b,  MCL_QMAX = {QS:.4g}")
    print(f"  equilibrium from isotherm + mass balance: "
          f"c_inf = {c_inf:.2f} mol/m^3, Q_inf = {q_inf:.1f} mol/m^3")

    print("\nRunning CADET simulation...")
    t, c = run_model(cadet_path, output_path, par_nelem=8, par_polydeg=4,
                     n_points=601, spatial_method='DG')
    q = particle_phase_concentration(c)
    print(f"  final state: c = {c[-1]:.2f} mol/m^3, Q = {q[-1]:.1f} mol/m^3 "
          f"({100.0 * q[-1] / q_inf:.1f}% of equilibrium)")

    print("\nLoading reference data extracted from Fig. 2...")
    t_ref, q_ref = load_digitized()

    metric = compute_nrmse(t, q, t_ref, q_ref)
    print("=" * 70)
    print("Brandani (2021), Fig. 2 -- m-xylene, c0 = 250 mol/m^3, M_S = 40 g, Sh = 2")
    print("=" * 70)
    print(f"  window              : {metric['window'][0]:.1f} .. {metric['window'][1]:.1f} s "
          f"({metric['n_ref']} reference points)")
    print(f"  reference amplitude : {metric['ref_amplitude']:.1f} mol/m^3")
    print(f"  max |deviation|     : {metric['max_abs_dev']:.3f} mol/m^3")
    print(f"  NRMSE           [%] : {metric['nrmse_%']:.4g}")

    # --- comparison plot ---
    fontsize = 15
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.plot(t_ref, q_ref, 'o', color='tab:red', markersize=4,
            markevery=60, markerfacecolor='none',
            label='Brandani (2021), Fig. 2')
    ax.plot(t, q, '-', color='tab:blue', linewidth=1.8, label='CADET, finite bath')
    ax.axhline(q_inf, color='0.6', linestyle=':', linewidth=1.2)
    ax.annotate(r'$Q_\infty$ (isotherm + mass balance)', xy=(0.02, q_inf),
                xytext=(60, q_inf - 60), fontsize=fontsize - 4, color='0.4')
    ax.set_xlabel('Time [s]', fontsize=fontsize)
    ax.set_ylabel(r'$\bar{Q}$ [mol m$^{-3}$]', fontsize=fontsize)
    ax.set_title('m-xylene on Y zeolite, batch uptake, Sh = 2\n'
                 r'$c_0 = 250$ mol m$^{-3}$, $M_S = 40$ g',
                 fontsize=fontsize - 2)
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, None)
    ax.tick_params(labelsize=fontsize - 3)
    ax.legend(fontsize=fontsize - 3, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(output_path, 'Brandani2021_fig2.png')
    fig.savefig(out_png, dpi=150)
    print(f"\nComparison plot written to {out_png}")

    return metric


if __name__ == '__main__':
    main()
