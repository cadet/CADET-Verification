# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 14.6 from:

    T. Gu, "Mathematical Modeling and Scale-Up of Liquid Chromatography",
    2nd ed., Springer, 2015, Chapter 14 ("Multicomponent Radial Flow
    Chromatography"), p. 210: "Simulation of affinity RFC with inward flow".

Self-contained script: model definition, run, comparison plot and
validation metrics. Further explanation on model and parameter selection is
provided under Gu2015_fig14_6.md.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from addict import Dict
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))
INSTALL_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

# ---------------------------------------------------------------------------
# Paper's parameters, read off the Fig. 10.14 GUI screenshot (p. 138),
# applied per p. 210 to RFC with V0=0.04.
# ---------------------------------------------------------------------------
V0 = 0.04
TIMP = 14.0     # protein frontal-loading duration (dimensionless tau)
TSHIFT = 15.0   # switch to soluble-ligand elution feed (dimensionless tau)
TAU_MAX_SIM = 60.0

EPS_B = 0.40
EPS_P_TOTAL = 0.45
EXF = 0.8                     # size-exclusion factor, identical for i=1,2,3
EPS_AP = EXF * EPS_P_TOTAL    # accessible particle porosity used in CADET

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
# Reparameterization -- native radial geometry, physical (SI-like) scales.
# See docstring Sec. 5 for the full derivation.
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
    # COL_DISPERSION config value for COL_DISPERSION_DEP='POWER_LAW',
    # EXPONENT=1: Db_i(X) = COL_DISPERSION[i]*v(X), so supply Db_i|V=1/v(X1).
    p['col_dispersion_value'] = p['Db_V1'] / V_REF
    # FILM_DIFFUSION config value for FILM_DIFFUSION_DEP='POWER_LAW',
    # EXPONENT=1/3 (per Eq. 14.16: k_i(V) ~ v^(1/3), expressed w.r.t. the
    # local velocity CADET's POWER_LAW dependency multiplies by), so supply
    # k_i|V=1/v(X1)^(1/3).
    p['film_diffusion_value'] = p['k_V1'] / V_REF ** (1.0 / 3.0)

# "iave=2" constant-Bi fallback (paper's own alternative to true
# position-dependent Bi_i(V), Eq. 14.16 at V=0.5) -- used when
# film_diffusion_velocity_dep=False in get_model() (see there).
IAVE2_FACTOR = ((1.0 - V0) / (0.5 + V0)) ** (1.0 / 6.0)
for i, p in PAPER.items():
    p['k_avg'] = p['k_V1'] * IAVE2_FACTOR

C0_1_PHYS = PAPER[1]['C0_phys']
C0_2_PHYS = PAPER[2]['C0_phys']
C0_3_PHYS = PAPER[3]['C0_phys']  # == C0_1_PHYS by construction (book convention)

# Particle-porosity correction (docstring Sec. 4): CADET's single
# PAR_POROSITY=eps_ap makes its solid-phase weight (1-eps_ap) instead of the
# book's own (1-eps_p); rescaling QMAX1 by (1-eps_p)/(1-eps_ap) exactly
# compensates, since the kinetic Langmuir ODE is linear in qmax for fixed
# ka1/kd1.
QMAX1_CORRECTION = (1.0 - EPS_P_TOTAL) / (1.0 - EPS_AP)
QMAX1_PHYS = PAPER[1]['C_inf_phys'] * QMAX1_CORRECTION

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
        FILM_DIFFUSION_DEP='POWER_LAW'. If False, falls back to the paper's
        own "iave=2" constant-Bi approximation (k_i evaluated once at
        V=0.5)."""
    m = Dict()
    m.input.model.nunits = 3

    # 3 sections: 0 = frontal protein loading (0 <= tau < 14); 1 = wash with
    # inert mobile phase (14 <= tau < 15); 2 = elution with soluble ligand
    # feed held constant (tau >= 15). Genuine inward flow throughout
    # (FORWARD_FLOW=[0,0,0], single unchanging direction across all 3
    # sections): V=1 (Gu's inward-flow RFC inlet) is CADET's z=0 inlet.
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

    # --- Column: CADET's native radial-flow geometry (docstring Sec. 3 for
    # the COL_DISPERSION_DEP/FILM_DIFFUSION_DEP velocity-dependence mechanism) ---
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
    col.forward_flow = [0, 0, 0]
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

    # --- Particles: GENERAL_RATE_PARTICLE (film + pore diffusion,
    # spherical), identical transport parameters for all 3 components ---
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
    col.particle_type_000.par_porosity = EPS_AP
    col.particle_type_000.par_radius = RP
    col.particle_type_000.pore_diffusion = [PAPER[1]['Dp'], PAPER[2]['Dp'], PAPER[3]['Dp']]
    col.particle_type_000.surface_diffusion = [0.0, 0.0, 0.0]

    # Pore-liquid-phase reaction (same reaction, same rate constants -- see
    # docstring Sec. 3 for why eps_ap cancels between bulk and pore liquid).
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


def run_model(ncol=256, par_ncells=4, n_points=900, fname='Gu2015_fig14_6.h5', spatial_method='FV', **kwargs):
    model = get_model(ncol=ncol, par_ncells=par_ncells, n_points=n_points, spatial_method=spatial_method, **kwargs)
    c = Cadet(install_path=INSTALL_PATH)
    c.root.input = model.input
    c.filename = os.path.join(HERE, fname)
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
#
# Digitized from the rendered p. 210 figure using pixel colour-thresholding.
# Curves: protein = teal/green solid, soluble ligand = black dashed,
# complex = navy solid. Extraction quality was checked visually by
# overlaying the digitized points against a matplotlib reproduction next to
# a crop of the original page (Gu2015_fig14_6_digitized_reference_check.png,
# alongside this script) -- the digitized curves reproduce the original
# figure's shape, timing, and peak heights closely; the only expected
# artifact is a sparse/gapped sampling of the dashed "soluble ligand" curve
# (dash gaps => missing x-samples there), which is immaterial since the
# curve is smooth and slowly varying in that region.
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
# Validation metrics
# ---------------------------------------------------------------------------
def first_moment(t, c, t_lo=None, t_hi=None):
    t = np.asarray(t)
    c = np.clip(np.asarray(c), 0.0, None)
    if t_lo is not None or t_hi is not None:
        lo = t_lo if t_lo is not None else t.min()
        hi = t_hi if t_hi is not None else t.max()
        mask = (t >= lo) & (t <= hi)
        t, c = t[mask], c[mask]
    area = np.trapz(c, t)
    if area <= 0:
        return np.nan, np.nan
    moment = np.trapz(t * c, t) / area
    return moment, area


def t_at_level(t, c, level, direction='rising'):
    t = np.asarray(t)
    c = np.asarray(c)
    d = np.diff(np.sign(c - level))
    if direction == 'rising':
        idx = np.where(d > 0)[0]
    else:
        idx = np.where(d < 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    t0, t1 = t[i], t[i + 1]
    c0, c1 = c[i], c[i + 1]
    frac = (level - c0) / (c1 - c0)
    return t0 + frac * (t1 - t0)


def compute_metrics(tau_sim, sims, tau_refs, refs):
    """sims/refs: dicts {'protein': c_arr, 'soluble_ligand': c_arr, 'complex': c_arr}"""
    metrics = {}
    for name in ('protein', 'soluble_ligand', 'complex'):
        c_sim = sims[name]
        tau_ref, c_ref = tau_refs[name], refs[name]
        c_sim_i = np.interp(tau_ref, tau_sim, c_sim)

        m = {}

        # 1) Peak position -- meaningful for protein and complex (both show
        # a genuine interior maximum). Soluble ligand rises monotonically
        # to a plateau (no interior peak), so this metric is reported N/A.
        if name in ('protein', 'complex'):
            i_sim = np.argmax(c_sim)
            i_ref = np.argmax(c_ref)
            t_peak_sim, t_peak_ref = tau_sim[i_sim], tau_ref[i_ref]
            m['peak_time_sim'] = t_peak_sim
            m['peak_time_ref'] = t_peak_ref
            m['peak_time_relerr_%'] = 100 * abs(t_peak_sim - t_peak_ref) / t_peak_ref
            m['peak_height_sim'] = c_sim[i_sim]
            m['peak_height_ref'] = c_ref[i_ref]
            m['peak_height_relerr_%'] = 100 * abs(c_sim[i_sim] - c_ref[i_ref]) / c_ref[i_ref]
        else:
            m['peak_time_sim'] = m['peak_time_ref'] = m['peak_time_relerr_%'] = np.nan
            m['peak_height_sim'] = m['peak_height_ref'] = m['peak_height_relerr_%'] = np.nan

        # 2) Elution time (first moment). For protein and complex, c(t)
        # returns close to baseline within the simulated window
        # [0, tau_max=60]. For soluble ligand, c(t) plateaus near 1 and
        # never returns to baseline, so the first moment is dominated by
        # (and diverges with) the upper integration limit; we instead
        # report the breakthrough time t50 (time to cross 50% of the FINAL
        # plateau value), the standard adapted metric for a non-eluting/
        # plateauing curve.
        if name == 'soluble_ligand':
            plateau_val_sim = np.nanmean(c_sim[tau_sim >= tau_sim.max() - 2.0])
            plateau_val_ref = np.nanmean(c_ref[tau_ref >= tau_ref.max() - 2.0])
            t50_sim = t_at_level(tau_sim, c_sim, 0.5 * plateau_val_sim)
            t50_ref = t_at_level(tau_ref, c_ref, 0.5 * plateau_val_ref)
            m['elution_metric'] = 't50 (50% of final plateau), adapted for a non-eluting curve'
            m['elution_time_sim'] = t50_sim
            m['elution_time_ref'] = t50_ref
            m['elution_time_relerr_%'] = 100 * abs(t50_sim - t50_ref) / t50_ref
        else:
            mu_sim, _ = first_moment(tau_sim, c_sim, t_lo=0.0, t_hi=tau_sim.max())
            mu_ref, _ = first_moment(tau_ref, c_ref, t_lo=0.0, t_hi=tau_ref.max())
            m['elution_metric'] = 'first moment int(t*c dt)/int(c dt) over [0, tau_max]'
            m['elution_time_sim'] = mu_sim
            m['elution_time_ref'] = mu_ref
            m['elution_time_relerr_%'] = 100 * abs(mu_sim - mu_ref) / mu_ref

        # 3) Mass balance. For protein: total fed (=C0_1 for 0<=tau<14,
        # i.e. area=14 in dimensionless (C/C0)*tau units) vs.
        # int(c1_out dtau) over the full run. Because protein reacts with
        # the soluble ligand to form the complex, most of the loaded
        # protein leaves the column AS COMPLEX, not as free protein -- so
        # a component-1-only mass balance is expected to show a large
        # "deficit" that reflects real chemistry (conversion + residual
        # binding), not a modeling error (see the printed atom-balance
        # check). For soluble ligand, the feed is a "displacer" that is
        # never turned off (per Eq. 14.12's index=4 "displacer" clause), so
        # its cumulative mass balance only makes sense over a bounded
        # window; per the task's guidance we use the window
        # [tau_shift, tau_max] = [15, 60] (a "suitably long integration
        # window" starting when the ligand feed switches on) and compare
        # int(c2_out dtau) there against the ligand fed over the same
        # window (=1.0*(tau_max-tau_shift)=45 in (C/C0)*tau units); some
        # ligand mass is expected to be "missing" here too since part of it
        # reacts to form the complex rather than exiting as free ligand.
        # For complex: it is not fed at all, so mass balance is simply
        # int(c3_out dtau) over the full run compared between simulation
        # and digitized reference.
        if name == 'protein':
            fed = 1.0 * TIMP
            out_sim = np.trapz(c_sim, tau_sim)
            out_ref = np.trapz(c_ref, tau_ref)
            m['mass_balance_metric'] = 'int(c_out dtau) vs. protein fed (=1*tau_imp); ' \
                                        'deficit reflects protein retained on-column + ' \
                                        'converted to complex (see printed atom-balance check)'
            m['mass_fed'] = fed
            m['mass_out_sim'] = out_sim
            m['mass_out_ref'] = out_ref
            m['mass_relerr_sim_vs_fed_%'] = 100 * abs(out_sim - fed) / fed
            m['mass_relerr_sim_vs_ref_%'] = 100 * abs(out_sim - out_ref) / out_ref if out_ref > 0 else np.nan
        elif name == 'soluble_ligand':
            window = (tau_sim >= TSHIFT)
            out_sim = np.trapz(c_sim[window], tau_sim[window])
            window_ref = (tau_ref >= TSHIFT)
            out_ref = np.trapz(c_ref[window_ref], tau_ref[window_ref]) if window_ref.sum() > 1 else np.nan
            fed = 1.0 * (TAU_MAX_SIM - TSHIFT)
            m['mass_balance_metric'] = f'int(c_out dtau) over [tau_shift={TSHIFT}, tau_max={TAU_MAX_SIM}] ' \
                                        'vs. ligand fed over same window; deficit reflects ligand ' \
                                        'consumed by ongoing complex formation (see atom-balance check)'
            m['mass_fed'] = fed
            m['mass_out_sim'] = out_sim
            m['mass_out_ref'] = out_ref
            m['mass_relerr_sim_vs_fed_%'] = 100 * abs(out_sim - fed) / fed
            m['mass_relerr_sim_vs_ref_%'] = 100 * abs(out_sim - out_ref) / out_ref if (out_ref and out_ref > 0) else np.nan
        else:  # complex
            out_sim = np.trapz(c_sim, tau_sim)
            out_ref = np.trapz(c_ref, tau_ref)
            m['mass_balance_metric'] = 'int(c_out dtau), sim vs. digitized reference (no independent ' \
                                        'feed-side reference for a non-fed product species)'
            m['mass_fed'] = np.nan
            m['mass_out_sim'] = out_sim
            m['mass_out_ref'] = out_ref
            m['mass_relerr_sim_vs_fed_%'] = np.nan
            m['mass_relerr_sim_vs_ref_%'] = 100 * abs(out_sim - out_ref) / out_ref if out_ref > 0 else np.nan

        # 4) Chromatogram MSE over the full digitized time window
        m['mse'] = np.nanmean((c_sim_i - c_ref) ** 2)
        # Normalized RMSE (% of the reference curve's own peak amplitude) --
        # raw MSE is NOT comparable across species with very different
        # amplitude scales (protein/complex peak vs. soluble_ligand's
        # plateau) or across scripts with different C/C0 ranges -- same
        # convention as the Gritti case studies
        # (Gritti2019_fig6/7/8.py). max|c_ref| (not peak_height_ref,
        # which is NaN for soluble_ligand's monotonic plateau) is used as
        # the reference amplitude so this works uniformly for both peaked
        # and plateauing curves.
        m['nrmse_%'] = 100 * np.sqrt(m['mse']) / np.nanmax(np.abs(c_ref))

        metrics[name] = m

    return metrics


def print_metrics(metrics):
    for comp, m in metrics.items():
        print(f"\n--- {comp} ---")
        if not np.isnan(m['peak_time_sim']):
            print(f"  Peak position   : sim tau={m['peak_time_sim']:.4g}  ref tau={m['peak_time_ref']:.4g}"
                  f"  rel.err={m['peak_time_relerr_%']:.3g}%")
            print(f"  Peak height     : sim={m['peak_height_sim']:.4g}  ref={m['peak_height_ref']:.4g}"
                  f"  rel.err={m['peak_height_relerr_%']:.3g}%")
        else:
            print("  Peak position   : N/A (monotonic rise to plateau, no interior peak)")
        print(f"  Elution metric  : {m['elution_metric']}")
        print(f"  Elution time    : sim={m['elution_time_sim']:.4g}  ref={m['elution_time_ref']:.4g}"
              f"  rel.err={m['elution_time_relerr_%']:.3g}%")
        print(f"  Mass balance    : {m['mass_balance_metric']}")
        fed_str = f"{m['mass_fed']:.4g}" if not np.isnan(m['mass_fed']) else "N/A"
        print(f"    fed={fed_str}  out(sim)={m['mass_out_sim']:.4g}  out(ref)={m['mass_out_ref']:.4g}")
        if not np.isnan(m['mass_relerr_sim_vs_fed_%']):
            print(f"    sim vs. fed rel.err={m['mass_relerr_sim_vs_fed_%']:.3g}%")
        if not np.isnan(m['mass_relerr_sim_vs_ref_%']):
            print(f"    sim vs. ref rel.err={m['mass_relerr_sim_vs_ref_%']:.3g}%")
        print(f"  Chromatogram MSE: {m['mse']:.4g}  (NRMSE={m['nrmse_%']:.2f}% of reference peak amplitude)")


def print_atom_balance(tau_sim, c1_sim, c2_sim, c3_sim):
    """Diagnostic (not one of the 4 core metrics): since P + I -> PI is a
    1:1 reaction and c3 is normalized by C0_1 (book convention), a
    "protein-equivalent" balance should approximately hold:
        protein fed (=1*tau_imp) ~= int(c1_out dtau) + int(c3_out dtau)
                                     + protein remaining bound on-column
    """
    fed = 1.0 * TIMP
    out_protein = np.trapz(c1_sim, tau_sim)
    out_complex = np.trapz(c3_sim, tau_sim)
    print("\n--- Diagnostic: protein-equivalent atom balance (not one of the 4 core metrics) ---")
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
if __name__ == '__main__':

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
    print(f"  eps_b={EPS_B}, eps_p(total)={EPS_P_TOTAL}, ExF={EXF}, eps_ap(CADET PAR_POROSITY)={EPS_AP}")

    print("\nRunning CADET simulation (native radial geometry, genuine inward flow -- see script docstring)...")

    spatial_method = 'DG'
    if spatial_method == 'DG':
        t_phys, outlet = run_model(spatial_method=spatial_method, dg_polydeg=4, ncol=64, par_ncells=3)
    elif spatial_method == 'FV':
        t_phys, outlet = run_model(spatial_method=spatial_method, ncol=256, par_ncells=4)

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
    print_metrics(metrics)
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
    # fig.suptitle('Gu (2015), Fig. 14.6 -- affinity RFC with inward flow', y=0.985, fontsize=fontsize)
    # ax.set_title('CADET native radial geometry; velocity-scaled dispersion and film\n',
    #               fontsize=fontsize)
    # add an NRMSE box, same convention as the other case-study scripts
    nrmse_text = (f"NRMSE Protein: {metrics['protein']['nrmse_%']:.2f}%\n"
                  f"NRMSE Soluble ligand: {metrics['soluble_ligand']['nrmse_%']:.2f}%\n"
                  f"NRMSE Complex: {metrics['complex']['nrmse_%']:.2f}%")
    ax.text(0.98, 0.15, nrmse_text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='bottom', horizontalalignment='right', multialignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.legend(loc='center right', fontsize=fontsize)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(HERE, f'Gu2015_fig14_6_comparison_{spatial_method}.png')
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved comparison plot to {outpath}")
