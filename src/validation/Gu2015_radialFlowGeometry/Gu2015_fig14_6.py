# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 14.6 from:

    T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
    2nd ed., Springer, 2015, Chapter 14 ("Multicomponent Radial Flow
    Chromatography"), p. 210: "Simulation of affinity RFC with inward flow".

The model, the source of the parameters and the conversion of Gu's
dimensionless groups into CADET parameters are explained in
Gu2015_fig14_6.md.
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
# directly and as an import (scripts/verify_geometries.py calls main()).
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
import validation_metrics as vm  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters read off the Fig. 10.14 GUI screenshot on p. 138, applied to
# the RFC geometry with V0 = 0.04 as stated on p. 210
# ---------------------------------------------------------------------------
V0 = 0.04
TIMP = 14.0     # protein frontal-loading duration (dimensionless tau)
TSHIFT = 15.0   # switch to soluble-ligand elution feed (dimensionless tau)
TAU_MAX_SIM = 60.0

EPS_B = 0.40
EPS_P_TOTAL = 0.45
EXF = 0.8                     # size-exclusion factor, identical for i=1,2,3
EPS_AP = EXF * EPS_P_TOTAL    # accessible particle porosity = F_acc*eps_p

# component  PeL   eta   Bi    C_inf     C0         Daa   Dad
PAPER = {
    1: dict(PeL=300.0, eta=10.0, Bi_V1=40.0, C_inf=1.0e-5, C0=1.0e-6, Daa=2.0, Dad=0.2),  # protein
    2: dict(PeL=300.0, eta=10.0, Bi_V1=40.0, C_inf=0.0,    C0=5.0e-6, Daa=2.0, Dad=0.2),  # soluble ligand
    3: dict(PeL=300.0, eta=10.0, Bi_V1=40.0, C_inf=0.0,    C0=1.0e-6, Daa=0.0, Dad=0.0),  # complex (PI)
}
DA1A, DA1D = PAPER[1]['Daa'], PAPER[1]['Dad']
DA2A, DA2D = PAPER[2]['Daa'], PAPER[2]['Dad']
C1_INF = PAPER[1]['C_inf'] / PAPER[1]['C0']

# ---------------------------------------------------------------------------
# Dimensionless groups -> physical (SI-like) parameters, see the markdown file
# ---------------------------------------------------------------------------
X1 = 0.05                             # outer column radius [m] (inward-flow inlet)
X0 = X1 * np.sqrt(V0 / (1.0 + V0))    # inner radius [m]; V0 = X0^2/(X1^2-X0^2)
BED_LENGTH = X1 - X0                  # radial bed thickness [m]
CYL_HEIGHT = 1.0                      # arbitrary cylinder height [m]
RP = 5.0e-5                           # particle radius [m]
V_REF = 1.0e-4                        # interstitial velocity at X1 (V=1) [m/s]
V_CHAR = 2.0 * V_REF * X1 / (X1 + X0)  # transit-time characteristic velocity
CONC_UNIT = 1.0                        # reference concentration scale

for i, p in PAPER.items():
    p['Db_V1'] = V_REF * BED_LENGTH / p['PeL']
    p['Dp'] = p['eta'] * RP ** 2 * V_CHAR / (EPS_AP * BED_LENGTH)
    p['k_V1'] = p['Bi_V1'] * p['eta'] * RP * V_CHAR / BED_LENGTH
    p['C0_phys'] = p['C0'] * CONC_UNIT
    p['C_inf_phys'] = p['C_inf'] * CONC_UNIT
    # CADET computes Db_i(X) = COL_DISPERSION[i] * v(X) and
    # k_i(X) = FILM_DIFFUSION[i] * v(X)^(1/3) at the local velocity, so the
    # configured values are the paper's coefficients divided by the reference
    # velocity; that reproduces Db_i|V=1 and k_i|V=1 at X1.
    p['col_dispersion_value'] = p['Db_V1'] / V_REF
    p['film_diffusion_value'] = p['k_V1'] / V_REF ** (1.0 / 3.0)

# Gu's "iave=2" alternative to a position-dependent Bi_i(V): Eq. 14.16
# evaluated once at V=0.5. Used when film_diffusion_velocity_dep=False.
IAVE2_FACTOR = ((1.0 - V0) / (0.5 + V0)) ** (1.0 / 6.0)
for i, p in PAPER.items():
    p['k_avg'] = p['k_V1'] * IAVE2_FACTOR

C0_1_PHYS = PAPER[1]['C0_phys']
C0_2_PHYS = PAPER[2]['C0_phys']
C0_3_PHYS = PAPER[3]['C0_phys']  # == C0_1_PHYS by construction (book convention)

# Size exclusion: CADET's PORE_ACCESSIBILITY factor gives the same split as
# the book, with eps_ap = F_acc*eps_p in pore transport and the film boundary
# condition and the true (1-eps_p) in the solid phase. The saturation
# capacity is therefore the paper's own C_inf,1, unchanged.
QMAX1_PHYS = PAPER[1]['C_inf_phys']

KA1 = DA1A * V_CHAR / (BED_LENGTH * C0_1_PHYS)
KD1 = DA1D * V_CHAR / BED_LENGTH
KA2 = DA2A * V_CHAR / (BED_LENGTH * C0_1_PHYS)   # book-chapter convention: C0_1, not C0_2
KD2 = DA2D * V_CHAR / BED_LENGTH

Q_FLOW = V_REF * X1 * 2.0 * np.pi * CYL_HEIGHT * EPS_B  # inlet flow [m^3/s]

T_IMP = TIMP * BED_LENGTH / V_CHAR
T_SHIFT = TSHIFT * BED_LENGTH / V_CHAR
T_END = TAU_MAX_SIM * BED_LENGTH / V_CHAR


def dimless_time(t_phys):
    """Map physical simulation time [s] to the paper's tau = v_char*t/(X1-X0)."""
    return np.asarray(t_phys) * V_CHAR / BED_LENGTH


# ---------------------------------------------------------------------------
# CADET model definition
# ---------------------------------------------------------------------------
def get_model(ncol=200, par_ncells=4, n_points=900, spatial_method='FV',
              dg_polydeg=4, col_dispersion_velocity_dep=True,
              film_diffusion_velocity_dep=True):
    """
    col_dispersion_velocity_dep: if True (default), Db_i(X) ~ v(X) via
        COL_DISPERSION_DEP='POWER_LAW'. If False, COL_DISPERSION is held
        constant at Db_i|V=1 everywhere.
    film_diffusion_velocity_dep: if True (default), k_i(X) ~ v(X)^(1/3) via
        FILM_DIFFUSION_DEP='POWER_LAW'. If False, Gu's own "iave=2"
        approximation is used, i.e. k_i evaluated once at V=0.5."""
    m = Dict()
    m.input.model.nunits = 3

    # Three sections: 0 = frontal protein loading (0 <= tau < 14),
    # 1 = wash with inert mobile phase (14 <= tau < 15), 2 = elution with a
    # constant soluble-ligand feed (tau >= 15). The flow direction is inward
    # throughout and never switches.
    sec_times = [0.0, T_IMP, T_SHIFT, T_END]
    n_sections = 3

    m.input.model.connections.nswitches = n_sections
    for s in range(n_sections):
        key = f'switch_{s:03d}'
        m.input.model.connections[key].connections = [
            0.0, 1.0, -1.0, -1.0, Q_FLOW,
            1.0, 2.0, -1.0, -1.0, Q_FLOW,
        ]
        m.input.model.connections[key].section = s

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # --- Inlet: 3 components (1=protein, 2=soluble ligand, 3=complex,
    # which is never fed) ---
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 3

    feed_by_section = [
        [C0_1_PHYS, 0.0, 0.0],   # sec 0: frontal protein loading
        [0.0, 0.0, 0.0],         # sec 1: wash (inert mobile phase)
        [0.0, C0_2_PHYS, 0.0],   # sec 2: elution with soluble ligand
    ]
    for s, feed in enumerate(feed_by_section):
        key = f'sec_{s:03d}'
        m.input.model.unit_000[key].const_coeff = feed
        m.input.model.unit_000[key].lin_coeff = [0.0, 0.0, 0.0]
        m.input.model.unit_000[key].quad_coeff = [0.0, 0.0, 0.0]
        m.input.model.unit_000[key].cube_coeff = [0.0, 0.0, 0.0]

    # --- Column ---
    col = Dict()
    col.unit_type = 'COLUMN_MODEL_1D'
    col.geometry = 'RADIAL_FLOW_CYLINDER_SHELL'
    col.ncomp = 3
    col.npartype = 1
    col.par_type_volfrac = 1
    col.cross_section_area_outer = 2.0 * np.pi * X1 * CYL_HEIGHT
    col.cylinder_height = CYL_HEIGHT
    col.bed_length = BED_LENGTH
    col.col_porosity = EPS_B
    if col_dispersion_velocity_dep:
        col.col_dispersion = [PAPER[1]['col_dispersion_value'], PAPER[2]['col_dispersion_value'], PAPER[3]['col_dispersion_value']]
        col.col_dispersion_dep = 'POWER_LAW'
        col.col_dispersion_dep_exponent = 1.0
    else:
        col.col_dispersion = [PAPER[1]['Db_V1'], PAPER[2]['Db_V1'], PAPER[3]['Db_V1']]
    # Gu (2015) uses inward flow, i.e. from the outer to the inner radius, which is
    # the default direction of CADET's radial flow geometry.
    col.forward_flow = [1, 1, 1]
    col.init_c = [0.0, 0.0, 0.0]

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    if spatial_method == 'DG':
        col.discretization.SPATIAL_METHOD = 'DG'
        col.discretization.POLYDEG = dg_polydeg
        col.discretization.NELEM = ncol
        col.discretization.USE_COLLOCATION_DG = 0
        col.dispersion_spatial_dependence_polydeg = dg_polydeg
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

    # --- Bulk-phase (liquid) reaction: P(1) + I(2) <-> PI(3) ---
    col.nreac_liquid = 1
    col.liquid_reaction_000.type = 'MASS_ACTION_LAW'
    col.liquid_reaction_000.mal_stoichiometry = [-1.0, -1.0, 1.0]
    col.liquid_reaction_000.mal_kfwd = [KA2]
    col.liquid_reaction_000.mal_kbwd = [KD2]

    # --- Particles: spherical, with film and pore diffusion; the paper
    # lists the same transport parameters for all three components ---
    col.particle_type_000.nbound = [1, 0, 0]  # only component 1 (protein) binds
    col.particle_type_000.init_cp = [0.0, 0.0, 0.0]
    col.particle_type_000.init_cs = [0.0]

    col.particle_type_000.has_film_diffusion = 1
    if film_diffusion_velocity_dep:
        col.particle_type_000.film_diffusion = [PAPER[1]['film_diffusion_value'], PAPER[2]['film_diffusion_value'], PAPER[3]['film_diffusion_value']]
        col.particle_type_000.film_diffusion_dep = 'POWER_LAW'
        col.particle_type_000.film_diffusion_dep_exponent = 1.0 / 3.0
    else:
        col.particle_type_000.film_diffusion = [PAPER[1]['k_avg'], PAPER[2]['k_avg'], PAPER[3]['k_avg']]
    col.particle_type_000.has_pore_diffusion = 1
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.par_geom = 'SPHERE'
    col.particle_type_000.par_coreradius = 0.0
    col.particle_type_000.par_porosity = EPS_P_TOTAL
    # Size exclusion: eps_ap = PORE_ACCESSIBILITY*PAR_POROSITY replaces the
    # porosity in pore transport and the film boundary condition, while the
    # solid phase retains (1-eps_p). ExF is the same for all components here.
    col.particle_type_000.pore_accessibility = [EXF, EXF, EXF]
    col.particle_type_000.par_radius = RP
    col.particle_type_000.pore_diffusion = [PAPER[1]['Dp'], PAPER[2]['Dp'], PAPER[3]['Dp']]
    col.particle_type_000.surface_diffusion = [0.0, 0.0, 0.0]

    # Pore-liquid reaction: same reaction and same rate constants as in the
    # bulk, because eps_ap cancels out of Eq. 10.11 (see the markdown file).
    col.particle_type_000.nreac_liquid = 1
    col.particle_type_000.liquid_reaction_000.type = 'MASS_ACTION_LAW'
    col.particle_type_000.liquid_reaction_000.mal_stoichiometry = [-1.0, -1.0, 1.0]
    col.particle_type_000.liquid_reaction_000.mal_kfwd = [KA2]
    col.particle_type_000.liquid_reaction_000.mal_kbwd = [KD2]

    # Kinetic (non-equilibrium) Langmuir binding, component 1 only
    # (Eq. 10.12: dq1/dt = ka1*cp1*(qmax1 - q1) - kd1*q1). Entries for
    # components 2,3 are unused since nbound=0 there.
    col.particle_type_000.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    col.particle_type_000.adsorption.is_kinetic = 1
    col.particle_type_000.adsorption.mcl_ka = [KA1, 0.0, 0.0]
    col.particle_type_000.adsorption.mcl_kd = [KD1, 0.0, 0.0]
    col.particle_type_000.adsorption.mcl_qmax = [QMAX1_PHYS, 0.0, 0.0]

    if spatial_method == 'DG':
        col.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.PAR_NELEM = par_ncells
        col.particle_type_000.discretization.PAR_POLYDEG = 2
    elif spatial_method == 'FV':
        col.particle_type_000.discretization.SPATIAL_METHOD = 'FV'
        col.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        col.particle_type_000.discretization.NCELLS = par_ncells
        col.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2
    else:
        raise ValueError(f"Unsupported spatial method: {spatial_method}")

    m.input.model.unit_001 = col

    m.input.model.unit_002.ncomp = 3
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
    m.input.solver.sections.nsec = n_sections
    m.input.solver.sections.section_continuity = [0, 0]
    m.input.solver.sections.section_times = sec_times
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-10
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, sec_times[-1], n_points)

    return m


def run_model(cadet_path, output_path, ncol=256, par_ncells=4, n_points=900,
              fname='Gu2015_fig14_6.h5', spatial_method='FV', **kwargs):
    model = get_model(ncol=ncol, par_ncells=par_ncells, n_points=n_points, spatial_method=spatial_method, **kwargs)
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
# Reference data, digitized from the figure on p. 210 by pixel colour
# thresholding. The protein is the teal solid curve, the soluble ligand the
# black dashed one and the complex the navy solid one. The dashed ligand
# curve is sampled with gaps where the dashes are, which does not matter
# because it is smooth and slowly varying there.
# ---------------------------------------------------------------------------
def load_digitized(path=None):
    if path is None:
        path = os.path.join(HERE, 'Gu2015_fig14_6_digitized.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)

    def clean(tcol, ccol):
        t, c = data[tcol], data[ccol]
        mask = ~(np.isnan(t) | np.isnan(c))
        t, c = t[mask], c[mask]
        order = np.argsort(t)
        return t[order], c[order]

    t1, c1 = clean('time_protein', 'protein')
    t2, c2 = clean('time_soluble_ligand', 'soluble_ligand')
    t3, c3 = clean('time_complex', 'complex')
    return (t1, c1), (t2, c2), (t3, c3)


# ---------------------------------------------------------------------------
# Validation metrics, see src/validation/validation_metrics.py
#
# The three species need two different curve kinds. Protein and complex
# elute as pulses that return to baseline, so their moments are the usual
# int(t*c dt)/int(c dt) and its central second counterpart. The soluble
# ligand is fed from tau_shift onwards and never switched off, so its outlet
# rises to a plateau instead; its moments are taken of the residence time
# distribution of the normalised front, which makes mu_1 the stoichiometric
# breakthrough time and mu_2 the variance of the front.
#
# Gu prints no moment table for this figure, so both moments are compared
# against the digitized chromatogram. Protein and complex fall back to the
# peak-height error in the mu_2 column; the soluble ligand uses its
# digitized mu_2, since a plateauing curve has no peak height to compare.
#
# The mass balance is reported on the protein row as an atom balance.
# P + I <-> PI is a 1:1 reaction and the complex is normalised by the
# protein feed concentration, so the protein fed has to leave the column
# either free or as complex: int(c_protein dtau) + int(c_complex dtau)
# against the fed area 1*tau_imp. The other two rows have no closed balance
# of their own -- the complex is never fed, and the ligand feed is never
# switched off while part of it reacts -- so they are left empty.
# ---------------------------------------------------------------------------
def compute_metrics(tau_sim, sims, tau_refs, refs):
    """sims/refs: dicts {'protein': c_arr, 'soluble_ligand': c_arr, 'complex': c_arr}"""
    protein_out = vm.trapezoid(np.clip(sims['protein'], 0.0, None), tau_sim)
    complex_out = vm.trapezoid(np.clip(sims['complex'], 0.0, None), tau_sim)

    spec = {
        'protein': dict(
            kind=vm.PULSE, mu2_fallback=vm.MU2_PEAK_HEIGHT,
            mass_in=1.0 * TIMP,
            # standard_metrics() integrates this species' own outlet and adds
            # mass_extra_out, giving int(c_protein) + int(c_complex).
            mass_extra_out=complex_out,
            mass_label='protein-equivalent atom balance: free protein + complex '
                       'eluted vs. protein fed (1*tau_imp); P + I <-> PI is 1:1 '
                       'and the complex is normalised by the protein feed. NOT a '
                       'pure conservation check: a little protein is still bound '
                       'on the column at tau_max, and that physical residual is '
                       'included in the deviation',
            mass_exact=False,
        ),
        'soluble_ligand': dict(
            kind=vm.FRONTAL, mu2_fallback=vm.MU2_FROM_DIGITIZED,
            mass_in=None,
            mass_label='not closed: the ligand displacer feed is never switched off '
                       'and part of it is consumed by complex formation',
        ),
        'complex': dict(
            kind=vm.PULSE, mu2_fallback=vm.MU2_PEAK_HEIGHT,
            mass_in=None,
            mass_label='not closed: the complex is a reaction product and is never '
                       'fed, so it has no injected mass of its own (it is instead '
                       "accounted for in the protein row's atom balance)",
        ),
    }

    metrics = []
    for name in ('protein', 'soluble_ligand', 'complex'):
        metrics.append(vm.standard_metrics(
            name=name,
            t_sim=tau_sim, c_sim=sims[name],
            t_ref=tau_refs[name], c_ref=refs[name],
            # Gu's curves are already C/C0-normalised, so no amplitude fit.
            amplitude=1.0,
            **spec[name],
        ))
    metrics_by_name = {m['name']: m for m in metrics}
    metrics_by_name['protein']['protein_free_out'] = protein_out
    metrics_by_name['protein']['protein_as_complex_out'] = complex_out
    return metrics


def print_atom_balance(tau_sim, c1_sim, c2_sim, c3_sim):
    """Diagnostic printout: since P + I -> PI is a 1:1 reaction and c3 is
    normalized by C0_1 (the book's convention), a protein-equivalent balance
    should hold up to the protein still bound at the end of the run:
        protein fed (=1*tau_imp) ~= int(c1_out dtau) + int(c3_out dtau)
                                     + protein remaining bound on-column
    """
    fed = 1.0 * TIMP
    out_protein = vm.trapezoid(c1_sim, tau_sim)
    out_complex = vm.trapezoid(c3_sim, tau_sim)
    print("\n--- Diagnostic: protein-equivalent atom balance ---")
    print(f"  Protein fed (1*tau_imp)              : {fed:.4g}")
    print(f"  int(c1_out dtau) [free protein out]  : {out_protein:.4g}")
    print(f"  int(c3_out dtau) [as complex out]    : {out_complex:.4g}")
    print(f"  Sum (free + complex, protein-equiv.) : {out_protein + out_complex:.4g}")
    residual = fed - (out_protein + out_complex)
    print(f"  Residual (interpreted as protein still bound on-column"
          f" at tau_max={TAU_MAX_SIM}): {residual:.4g}  ({100 * residual / fed:.3g}% of fed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "output" / "validation"

def main(cadet_path=CADET_PATH, output_path=OUTPUT_PATH):

    os.makedirs(output_path, exist_ok=True)

    print("Physical (SI-like) parameters derived from the paper's dimensionless groups:")
    for i, p in PAPER.items():
        print(f"  Component {i}: Db(V=1)={p['Db_V1']:.4g} m^2/s (COL_DISPERSION={p['col_dispersion_value']:.4g}), "
              f"Dp={p['Dp']:.4g} m^2/s, "
              f"k(V=1)={p['k_V1']:.4g} m/s (FILM_DIFFUSION={p['film_diffusion_value']:.4g}), "
              f"C0={p['C0_phys']:.4g}")
    print(f"  KA1={KA1:.4g}, KD1={KD1:.4g}, QMAX1={QMAX1_PHYS:.4g}  (component-1 kinetic Langmuir)")
    print(f"  KA2(=KFWD)={KA2:.4g}, KD2(=KBWD)={KD2:.4g}  (shared P+I<->PI mass-action reaction)")
    print(f"  X0={X0:.4g} m, X1={X1:.4g} m, bed_length={BED_LENGTH:.4g} m, "
          f"Q={Q_FLOW:.4g} m^3/s, T_END={T_END:.4g} s")
    print(f"  eps_b={EPS_B}, eps_p(total)={EPS_P_TOTAL} (PAR_POROSITY), ExF={EXF} (PORE_ACCESSIBILITY), "
          f"eps_ap=ExF*eps_p={EPS_AP:.4g}")

    print("\nRunning CADET simulation...")

    spatial_method = 'DG'
    if spatial_method == 'DG':
        t_phys, outlet = run_model(cadet_path, output_path, spatial_method=spatial_method, dg_polydeg=4, ncol=64, par_ncells=3)
    elif spatial_method == 'FV':
        t_phys, outlet = run_model(cadet_path, output_path, spatial_method=spatial_method, ncol=256, par_ncells=4)

    tau_sim = dimless_time(t_phys)
    c1_sim = outlet[:, 0] / C0_1_PHYS
    c2_sim = outlet[:, 1] / C0_2_PHYS
    c3_sim = outlet[:, 2] / C0_3_PHYS

    print("Loading digitized reference data...")
    (t1_ref, c1_ref), (t2_ref, c2_ref), (t3_ref, c3_ref) = load_digitized()

    print("Computing validation metrics...")
    sims = {'protein': c1_sim, 'soluble_ligand': c2_sim, 'complex': c3_sim}
    tau_refs = {'protein': t1_ref, 'soluble_ligand': t2_ref, 'complex': t3_ref}
    refs = {'protein': c1_ref, 'soluble_ligand': c2_ref, 'complex': c3_ref}
    metrics = compute_metrics(tau_sim, sims, tau_refs, refs)
    print("=" * 70)
    print("Validation metrics -- Gu (2015), Fig. 14.6 (affinity RFC with "
          "soluble-ligand displacement)")
    print("=" * 70)
    vm.print_metrics_table(metrics, time_unit='tau')
    by_name = {m['name']: m for m in metrics}
    print_atom_balance(tau_sim, c1_sim, c2_sim, c3_sim)

    # --- comparison plot ---
    fontsize = 15
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(tau_sim, c1_sim, '-', color='#3dc498', lw=1.8, label='Protein (CADET)')
    ax.plot(tau_sim, c2_sim, '--', color='black', lw=1.5, label='Soluble ligand (CADET)')
    ax.plot(tau_sim, c3_sim, '-', color='#39408f', lw=1.8, label='Complex (CADET)')
    ax.plot(t1_ref, c1_ref, 'o', color='#3dc498', ms=3, mfc='none', mew=1.0,
            label='Protein (Gu 2015)')
    ax.plot(t2_ref, c2_ref, 's', color='black', ms=3, mfc='none', mew=1.0,
            label='Soluble ligand (Gu 2015)')
    ax.plot(t3_ref, c3_ref, '^', color='#39408f', ms=3, mfc='none', mew=1.0,
            label='Complex (Gu 2015)')
    ax.set_xlabel('Dimensionless time', fontsize=fontsize)
    ax.set_ylabel('Dimensionless concentration', fontsize=fontsize)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.2)
    ax.tick_params(axis='both', labelsize=fontsize)

    # NRMSE of all three species, as a box inside the axes
    nrmse_text = (f"NRMSE Protein: {by_name['protein']['nrmse_%']:.2f}%\n"
                  f"NRMSE Soluble ligand: {by_name['soluble_ligand']['nrmse_%']:.2f}%\n"
                  f"NRMSE Complex: {by_name['complex']['nrmse_%']:.2f}%")
    ax.text(0.98, 0.15, nrmse_text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='bottom', horizontalalignment='right', multialignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.legend(loc='center right', fontsize=fontsize)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(output_path, f'Gu2015_fig14_6_comparison_{spatial_method}.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved comparison plot to {outpath}")

    vm.dump_metrics(output_path, 'Gu2015_fig14_6',
                    'Affinity RFC displacement', metrics, time_unit='tau')
    return metrics


if __name__ == '__main__':
    main()
