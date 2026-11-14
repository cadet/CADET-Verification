# -*- coding: utf-8 -*-
"""
Six-component ion-exchange chromatography problem (plus salt), using the
parameter values and operating conditions of an in-house Novo Nordisk A/S
chromatography workflow as reported in Section 4.2 of

    Meyer et al., 2026, Computers and Chemical Engineering,
    "ChromOps.jl: High-order simulation and discrete forward sensitivity
     analysis for chromatography models".

Model parameters are Table 3, isotherm parameters Table 4, and the inlet
program Eqs. (35)-(40). The published chromatogram is Figure 1.

The paper's lumped-rate model is CADET's `COLUMN_MODEL_1D` with
`HOMOGENEOUS_PARTICLE` particles under an exact change of state variable; see
`get_binding_configuration` for the mapping and
`validation/Meyer2026_chromOpsIEX/Meyer2026_fig1.md` for the full derivation,
the validation against an independent solver of the paper's own equations, and
two documented inconsistencies between the paper's equations and its Figure 1.

Only `get_model` and its helpers live here; the reproduction of Figure 1 is
`validation/Meyer2026_chromOpsIEX/Meyer2026_fig1.py`.
"""

import numpy as np
from addict import Dict

# Available-sites ("free ligand") term of the SMA isotherm. The paper prints
# one form in Eq. (4) but its Figure 1 can only be reproduced with the other;
# see Meyer2026_fig1.md, Section "Two inconsistencies". The default is the
# printed equation, so this setting never asserts anything the paper does not.
#
#   'paper_eq4'  -- qbar = Lambda - sum_j (nu_j + sigma_j) q_j, i.e. Eq. (4)
#                   verbatim and CADET's standard SMA (SMA_SIGMA = Table 4).
#   'bound_salt' -- qbar = Lambda - sum_j nu_j q_j, which is the bound-salt
#                   concentration q_s that the paper itself integrates in
#                   Eq. (5). Realized as SMA_SIGMA = 0. Reproduces Figure 1;
#                   note that sigma_i then no longer enters the model at all.
SHIELDING_CONVENTIONS = ('paper_eq4', 'bound_salt')


def get_binding_configuration(is_kinetic: bool = False, par_porosity: float = 0.66,
                              shielding: str = 'paper_eq4'):
    """
    SMA parameters mapped from the paper's driving-force form (Eq. 4) onto
    CADET's Brooks & Cramer mass-action SMA, transforming *input parameters
    only*.

    The paper states its solid-phase balance (Eq. 2) in terms of q_i, the
    adsorbed concentration per pore-liquid volume, whereas CADET's solid-phase
    state c^s_i is per solid volume. The two are related by the constant factor

        q_i = alpha * c^s_i,        alpha := (1 - eps_p) / eps_p,

    which is a change of state variable, not an approximation: substituting it
    into CADET's pore balance and bulk film term reproduces the paper's Eqs. (2)
    and (1) exactly (see the .md). In the isotherm it gives

        Lambda - sum_j (nu_j + sigma_j) q_j = alpha * (Lambda_CADET - sum_j (nu_j + sigma_j) c^s_j)

    for Lambda_CADET = Lambda / alpha, so the porosity factor is absorbed
    completely and no residual scaling is left over.

    Kinetics, however, differ in form. Factoring qbar^(-nu_i) out of the paper's
    Eq. (4) leaves CADET's mass-action law exactly, times a state-dependent
    prefactor:

        dq_i/dt |paper = (ka_bar/alpha) * qbar^(-nu_i)
                         * [ qbar^(nu_i) c^p_i - c^p_s^(nu_i) c^s_i / (keq_i alpha^(nu_i - 1)) ]

    The prefactor is strictly positive, so both laws share the same zero set:
    the *equilibrium isotherm is identical*, giving the exact and
    state-independent ratio

        ka_i / kd_i = keq_i * alpha^(nu_i - 1),

    while the transient rates cannot be matched by any constant ka_i, kd_i (the
    required forward coefficient (ka_bar/alpha) qbar^(-nu_i) varies by ~20
    orders of magnitude over the load).

    Rapid-equilibrium binding (`is_kinetic = False`) is therefore used, which
    imposes the exactly-mapped isotherm and is the exact common limit of both
    laws. That this limit is reached at the paper's ka_bar = 10 1/s is not
    assumed but verified: an independent solver of the paper's equations with
    its own driving-force kinetics
    (`validation/Meyer2026_chromOpsIEX/Meyer2026_fig1_paper_equations.py`)
    agrees with this CADET setup to within 6 s in peak time and 0.7 % in peak
    height for all six components, under both shielding conventions.

    The reference concentrations keep the parameter values well-scaled; without
    them ka, kd would be O(1e-60) due to the nu-th powers of absolute
    concentrations (nu ~ 20). They cancel from the equilibrium condition as long
    as SMA_REFQ = SMA_REFC0.
    """
    if shielding not in SHIELDING_CONVENTIONS:
        raise ValueError(f"shielding must be one of {SHIELDING_CONVENTIONS}, got {shielding!r}")

    alpha = (1.0 - par_porosity) / par_porosity

    lambda_paper = 324.7  # ionic capacity, Table 3; 0.3247 mol/L = 324.7 mol/m^3
    sma_lambda = lambda_paper / alpha

    nu = np.array([22.0, 22.0, 21.0, 20.0, 10.0, 23.0])   # Table 4
    sigma = np.full(6, 3.0)                                # Table 4
    keq = np.array([1.0e5, 1.0e5, 3.0e4, 5.0e2, 5.0, 1.0e6])
    ka_bar = 10.0  # effective adsorption-rate coefficient, Table 4

    ka = np.full(6, ka_bar / alpha)
    kd = ka_bar / (keq * alpha ** nu)

    # 'bound_salt' moves the shielding out of the available-sites term; sigma_i
    # appears nowhere else in the model, so setting it to zero is exactly
    # equivalent to replacing qbar by the bound-salt concentration q_s.
    sma_sigma = np.zeros(6) if shielding == 'bound_salt' else sigma

    return {
        'is_kinetic': 1 if is_kinetic else 0,
        'sma_ka': np.concatenate(([0.0], ka)),
        'sma_kd': np.concatenate(([0.0], kd)),
        'sma_lambda': sma_lambda,
        'sma_refq': sma_lambda,
        'sma_refc0': sma_lambda,
        'sma_nu': np.concatenate(([0.0], nu)),
        'sma_sigma': np.concatenate(([0.0], sma_sigma)),
        }


def get_column_geometry_configuration(geometry: str):

    # for all geometries:
    # velocity = 7.51e-5 m/s = 7.51e-3 cm/s
    # bulk porosity is 0.37
    # bed length is 10mm = 0.01m
    # col radius at inlet was not given but we assume it to be 0.0035m and compute the adequat flow rate from there to match the velocity
    col_radius = 0.0035
    # flow rate Q = velocity / (cross_section * col_porosity)
    axial_flow_cross_section_area = np.pi * (col_radius ** 2)

    if geometry == 'AXIAL_FLOW_CYLINDER':
        return {
            # A = v * \pi * r^2 * \varepsilon
            'cross_section_area': axial_flow_cross_section_area,
            'col_length': 0.01,
            'bed_length': 0.01,
        }
    elif geometry == 'RADIAL_FLOW_CYLINDER_SHELL': # note: not considered in the original source
        return {
            # A = 2 * pi * \rho * L^b -> \rho = A / 2.0 / pi / L^b
            'cross_section_area': axial_flow_cross_section_area,
            'col_length': 0.00025, # height
            'col_radius_outer': axial_flow_cross_section_area / 2.0 / np.pi / 0.00025,
            'col_radius_inner': axial_flow_cross_section_area / 2.0 / np.pi / 0.00025 - 0.01,
        }
    elif geometry == 'AXIAL_FLOW_FRUSTUM': # note: not considered in the original source
        return {
            'cross_section_area': axial_flow_cross_section_area,
            'col_radius_large_end': col_radius,
            'col_radius_small_end': col_radius * 0.75,
            'col_radius_outer': col_radius,
            'col_radius_inner': col_radius * 0.75,
            'col_length': 0.01,
        }
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def get_model(
        spatial_method_bulk, axNElem,
        column_geometry='AXIAL_FLOW_CYLINDER',
        shielding='paper_eq4',
        **kwargs):

    model = Dict()

    model.input.model.nunits = 2

    column = Dict()
    if column_geometry == 'AXIAL_FLOW_CYLINDER':
        column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    elif column_geometry == 'RADIAL_FLOW_CYLINDER_SHELL':
        column.UNIT_TYPE = 'RADIAL_COLUMN_MODEL_1D'
    elif column_geometry == 'AXIAL_FLOW_FRUSTUM':
        column.UNIT_TYPE = 'FRUSTUM_COLUMN_MODEL_1D'
    else:
        raise ValueError(f"Unknown column geometry: {column_geometry}")
    column.geometry = column_geometry
    column.update(get_column_geometry_configuration(column_geometry))

    column.npartype = 1
    column.col_porosity = 0.37

    column.ncomp = 7
    column.init_c = [40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    column.col_dispersion = 1.5e-9 # 1.5e-5 cm^2 / s = 1.5e-9 m^2 / s
    column.forward_flow = 1

    # Spatial discretization of interstitial / bulk volume
    if spatial_method_bulk > 0:
        column.discretization.SPATIAL_METHOD = 'DG'
        column.discretization.POLYNOMIAL_INTEGRATION_TYPE = kwargs.get('POLYNOMIAL_INTEGRATION_TYPE', 0)
        column.discretization.POLYDEG = spatial_method_bulk
        column.discretization.NELEM = axNElem
    else:
        column.discretization.SPATIAL_METHOD = 'FV'
        column.discretization.NCOL = axNElem
        column.discretization.RECONSTRUCTION = 'WENO'
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = 3
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10
        column.discretization.SCHUR_SAFETY = 1.0e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = 1

    # particle_type = 'HOMOGENEOUS_PARTICLE'
    column.particle_type_000.has_film_diffusion = 1
    column.particle_type_000.par_geom = 'SPHERE'
    # The paper lumps the particle radius and the film mass-transfer coefficient
    # into the single parameter k_MT, so r_p is not reported. Only the ratio
    # k_f/r_p enters CADET (see below), so this assumed value cancels from the
    # mapping and has no effect on the solution.
    column.particle_type_000.par_radius = 4.5e-05
    column.particle_type_000.par_coreradius = 0.0
    # Table 3's particle porosity eps_p = 0.66 is used literally, as the pore volume fraction
    # CADET's PAR_POROSITY expects. The paper's pore-phase mass balance (eq. 2) is
    #   d(c_p,i)/dt + d(q_i)/dt = kMT,i*(c_i - c_p,i)         [coefficient 1 on d(q_i)/dt]
    # while CADET's HOMOGENEOUS_PARTICLE (LRMP) pore balance is
    #   d(c_p,i)/dt + (1-eps_p)/eps_p * d(c^s_i)/dt = 3/(eps_p*par_radius) * film_diffusion * (c_i - c_p,i).
    # The paper's q_i (adsorbed amount per pore-liquid volume) and CADET's c^s_i (per solid
    # volume) are related by q_i = (1-eps_p)/eps_p * c^s_i, which turns eq. (2) into CADET's
    # pore balance exactly. This substitution only rescales *input parameters* (eps_p, the
    # film diffusion coefficient below, and the SMA lambda/ka/kd in
    # get_binding_configuration()); no state variable is transformed -- CADET's own c^s_i then
    # automatically equals q_i * eps_p/(1-eps_p).
    par_porosity = 0.66
    column.particle_type_000.par_porosity = par_porosity
    # we compute film diffusion coefficient from the lumped parameter in the source, which is 1.39/s and 0.139/s for salt and proteins respectively.
    # The paper's bulk balance (eq. 1) uses the coefficient (1-eps_c)/eps_c * eps_p * kMT,i, while CADET's HOMOGENEOUS_PARTICLE
    # (LRMP) model uses (1-eps_c)/eps_c * (3/par_radius) * film_diffusion. Equating the two gives
    # kMT,i = (3/par_radius) * film_diffusion / eps_p, i.e. film_diffusion = kMT,i * par_radius * eps_p / 3.
    # (par_porosity above is used here too, so the bulk-equation match holds for whichever eps_p we use.)
    fd_salt = 1.39 * column.particle_type_000.par_radius * column.particle_type_000.par_porosity / 3.0
    fd_protein = 0.139 * column.particle_type_000.par_radius * column.particle_type_000.par_porosity / 3.0
    column.particle_type_000.film_diffusion = [fd_salt, fd_protein, fd_protein, fd_protein, fd_protein, fd_protein, fd_protein]

    column.particle_type_000.adsorption_model = 'STERIC_MASS_ACTION'
    column.particle_type_000.adsorption = get_binding_configuration(
        is_kinetic=False, par_porosity=par_porosity, shielding=shielding)
    column.particle_type_000.nbound = [1, 1, 1, 1, 1, 1, 1]

    init_salt = column.particle_type_000.adsorption['sma_lambda']
    column.particle_type_000.init_cs = [init_salt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    column.particle_type_000.init_cp = [40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    model.input.model.unit_000 = column

    # Flow sheet
    # velocity = 7.51e-5 m/s = 7.51e-3 cm/s
    interstitial_velocity = 7.51e-5
    flowRate = interstitial_velocity * column.cross_section_area * column.col_porosity
    model.input.model.connections.connections_include_ports = 1
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        1.0, 0.0, -1.0, -1.0, -1.0, -1.0, flowRate
    ]
    model.input.model.connections.switch_000.section = 0

    # Inlet / Feed unit
    model.input.model.unit_001.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_001.ncomp = 7
    # Salt gradient program, Eqs. (37)-(40); rectangular protein injection over
    # the first 360 s, Eqs. (35)-(36).
    deltaT = [360.0, 360, 900, 720, 7200, 720]
    section_times = [0.0]
    salt_start = [40.0, 40.0, 40.0, 240.0, 240.0, 1040.0]
    salt_end = [40.0, 40.0, 240.0, 240.0, 640.0, 1040.0]
    # c_feed = 1e-5 * [138.1 3.046 30.87 6.092 16.65 8.326] mol/L, Eq. (36)
    c_feed = [1.381, 0.03046, 0.3087, 0.06092, 0.1665, 0.08326]

    for i in range(len(deltaT)):

        section_times.append(section_times[-1] + deltaT[i])

        salt_const_coeff = salt_start[i]
        salt_lin_coeff = (salt_end[i] - salt_start[i]) / deltaT[i]

        if i == 0:
            model.input.model.unit_001[f'sec_{i:03d}'].const_coeff = [salt_const_coeff] + c_feed
        else:
            model.input.model.unit_001[f'sec_{i:03d}'].const_coeff = [salt_const_coeff, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.input.model.unit_001[f'sec_{i:03d}'].lin_coeff = [salt_lin_coeff, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.input.model.unit_001[f'sec_{i:03d}'].cube_coeff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.input.model.unit_001[f'sec_{i:03d}'].quad_coeff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    model.input.model.unit_001.UNIT_TYPE = 'INLET'

    # Global system solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    # Time integration / solver
    model.input.solver.nthreads = 1
    model.input.solver.consistent_init_mode = 5
    model.input.solver.sections.nsec = len(section_times) - 1
    model.input.solver.sections.section_continuity = [ 0, 0, 0, 0, 0, 0 ]
    model.input.solver.sections.section_times = section_times
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_reftol', 1e-6)
    model.input.solver.time_integrator.ALGTOL = kwargs['idas_reftol'] * 100 if 'idas_reftol' in kwargs else 1e-5
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.time_integrator.RELTOL = kwargs['idas_reftol'] * 100 if 'idas_reftol' in kwargs else 1e-5

    # Return data
    model.input.solver.user_solution_times = np.linspace(0, section_times[-1], int(section_times[-1]) + 1)
    model.input['return'].split_components_data = 0
    model.input['return'].unit_000.write_coordinates = kwargs.get('write_solution_bulk', False) or kwargs.get('write_solution_particle', False)
    model.input['return'].unit_000.write_solution_bulk =  kwargs.get('write_solution_bulk', False)
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].unit_000.write_solution_particle = kwargs.get('write_solution_particle', False)
    model.input['return'].unit_000.write_solution_solid = kwargs.get('write_solution_particle', False)
    model.input['return'].write_solution_times = 1

    return model
