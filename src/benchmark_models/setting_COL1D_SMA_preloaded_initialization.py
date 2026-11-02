# -*- coding: utf-8 -*-
"""

This script defines preloaded column settings with a quasi-stationary steric
mass action isotherm, used to stress the nonlinear solver of the consistent
initialization, see src/consistent_initialization.py.

With ``IS_KINETIC = 0`` the isotherm is algebraic and consistent initialization
has to solve, per particle shell and per axial node, the nonlinear system of the
conserved moieties of the participating phases together with the equilibrium
conditions of the isotherm,

.. math::

    \\varepsilon_p c_{p,i} + (1 - \\varepsilon_p) q_i
        &= \\varepsilon_p c_{p,i}^0 + (1 - \\varepsilon_p) q_i^0 \\\\
    k_{a,i} c_{p,i} \\bar{q}_0^{\\nu_i} - k_{d,i} q_i c_{p,0}^{\\nu_i} &= 0 \\\\
    q_0 &= \\Lambda - \\sum_{j \\geq 1} \\nu_j q_j

with the shielded salt capacity
:math:`\\bar{q}_0 = \\Lambda - \\sum_{j \\geq 1} (\\nu_j + \\sigma_j) q_j`.

This system is trivial when the column starts empty, because then every protein
concentration and every bound concentration is zero and the solution is the
initial state. It becomes hard as soon as the column is preloaded, and the
settings here are built around the two preloaded cases that CADET users have
reported on the CADET forum:

- ``nu_integer`` / ``nu_fractional`` reproduce the mechanism of
  https://forum.cadet-web.de/t/failing-simulation-for-non-integer-nu-in-sma/354,
  where a simulation of the case study of Saleh et al. fails for the fitted
  characteristic charges :math:`\\nu = (7.38, 7.50, 7.70)` and runs after they
  are rounded to integers. With fractional exponents, an iterate that leaves the
  physical range makes :math:`c_{p,0}^{\\nu_i}` undefined, whereas for integer
  exponents it stays finite, which is why rounding hides the problem.

- the low salt cases reproduce
  https://forum.cadet-web.de/t/bisma-model-error/942, where only the salt was
  given a nonzero initial value and the factorization inside the consistent
  initialization failed.

The difficulty is controlled by two knobs: the initial salt concentration, which
sets how strongly the protein equilibrium is driven towards the solid phase, and
whether the characteristic charges are integers.

The geometry, transport and isotherm parameters are those of the four component
load-wash-elute benchmark of Breuer et al. (2023),
doi:10.1016/j.compchemeng.2023.108340, see
src/benchmark_models/setting_Col1D_SMA_4comp_LWE_benchmark1.py, so that the
cases differ from an established benchmark only in their initial state and in
the characteristic charges.

"""

import numpy as np
from addict import Dict


# Characteristic charges of the load-wash-elute benchmark, and the fitted
# fractional charges of the forum case and their rounded counterparts.
NU_BENCHMARK = [0.0, 4.7, 5.29, 3.7]
NU_FRACTIONAL = [0.0, 7.38, 7.50, 7.70]
NU_INTEGER = [0.0, 7.0, 8.0, 7.0]


def get_binding_configuration(nu=None, is_kinetic=False):
    """Steric mass action parameters of the load-wash-elute benchmark."""

    return {
        'is_kinetic': 1 if is_kinetic else 0,
        'sma_ka': [0.0, 35.5, 1.59, 7.7],
        'sma_kd': [0.0, 1000.0, 1000.0, 1000.0],
        'sma_lambda': 1200.0,
        'sma_nu': list(NU_BENCHMARK if nu is None else nu),
        'sma_sigma': [0.0, 11.83, 10.6, 10.0],
        }


def get_model(init_salt=50.0, init_protein=(1.0, 1.0, 1.0), nu=None,
              init_bound=None, consistency_solver=None, t_end=1.0,
              nelem=2, polydeg=2, idas_abstol=1e-10, idas_algtol=1e-10,
              idas_reltol=1e-8, use_analytic_jacobian=True,
              consistent_init_mode=1, particle_type='HOMOGENEOUS_PARTICLE'):
    """Build the CADET-Core configuration of a preloaded column case.

    Parameters
    ----------
    init_salt : float
        Initial salt concentration in the bulk and the particle liquid phase.
        The difficulty knob of the family: the lower the salt, the stronger the
        proteins are driven onto the resin and the further the equilibrium lies
        from the initial state.
    init_protein : sequence of float
        Initial protein concentrations in the bulk and the particle liquid
        phase. All zero gives the trivial initialization problem of an empty
        column, any nonzero entry preloads the column.
    nu : sequence of float or None
        Characteristic charges, ``None`` uses those of the benchmark. See
        ``NU_FRACTIONAL`` and ``NU_INTEGER`` for the forum case.
    init_bound : sequence of float or None
        Initial bound concentrations, ``None`` uses the fully salt loaded resin
        ``[sma_lambda, 0, 0, 0]`` of the benchmark.
    consistency_solver : dict or None
        Contents of the ``consistency_solver`` group, e.g.
        ``{'SOLVER_NAME': 'PTC'}``. ``None`` leaves the CADET-Core default.
    t_end : float
        Length of the single time section. Only the initialization is of
        interest, so one short section suffices.
    nelem, polydeg : int
        Discontinuous Galerkin discretization of the bulk.
    idas_abstol, idas_algtol, idas_reltol : float
        Time integrator tolerances.
    use_analytic_jacobian : bool
        Use the analytic Jacobian instead of automatic differentiation.
    consistent_init_mode : int
        ``CONSISTENT_INIT_MODE``, 1 is full initialization.
    particle_type : string
        ``HOMOGENEOUS_PARTICLE`` for a lumped rate model with pores or
        ``GENERAL_RATE_PARTICLE`` for the general rate model, which solves the
        same initialization problem on every particle shell.

    Returns
    -------
    addict.Dict
        CADET-Core configuration.
    """

    binding = get_binding_configuration(nu=nu)
    n_comp = 4
    initial_liquid = [float(init_salt)] + [float(c) for c in init_protein]
    if init_bound is None:
        init_bound = [binding['sma_lambda'], 0.0, 0.0, 0.0]

    model = Dict()
    model.input.model.nunits = 2

    column = Dict()
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.geometry = 'AXIAL_FLOW_CYLINDER'
    cross_section_area = np.pi * 0.0035 ** 2
    column.cross_section_area = cross_section_area
    column.inlet_cross_section_area = cross_section_area
    column.bed_length = 0.014
    column.forward_flow = 1
    column.col_porosity = 0.37
    column.total_porosity = 0.8425
    column.npartype = 1
    column.ncomp = n_comp
    column.init_c = initial_liquid
    column.col_dispersion = 5.75e-08

    column.discretization.SPATIAL_METHOD = 'DG'
    column.discretization.USE_COLLOCATION_DG = 1
    column.discretization.POLYDEG = polydeg
    column.discretization.NELEM = nelem
    column.discretization.USE_ANALYTIC_JACOBIAN = 1 if use_analytic_jacobian else 0
    if consistency_solver is not None:
        column.discretization.consistency_solver = dict(consistency_solver)

    column.particle_type_000.has_film_diffusion = 1
    column.particle_type_000.par_geom = 'SPHERE'
    column.particle_type_000.par_radius = 4.5e-05
    column.particle_type_000.par_coreradius = 0.0
    column.particle_type_000.par_porosity = 0.75
    column.particle_type_000.film_diffusion = [6.9e-06] * n_comp

    if particle_type == 'GENERAL_RATE_PARTICLE':
        column.particle_type_000.has_pore_diffusion = 1
        column.particle_type_000.pore_diffusion = [7.00e-10, 6.07e-11,
                                                  6.07e-11, 6.07e-11]
        column.particle_type_000.has_surface_diffusion = 0
        column.particle_type_000.surface_diffusion = [0.0] * n_comp
        column.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        column.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
        column.particle_type_000.discretization.PAR_POLYDEG = 1
        column.particle_type_000.discretization.PAR_NELEM = 1

    column.particle_type_000.nbound = [1] * n_comp
    column.particle_type_000.init_cp = initial_liquid
    column.particle_type_000.init_cs = [float(q) for q in init_bound]
    column.particle_type_000.adsorption_model = 'STERIC_MASS_ACTION'
    column.particle_type_000.adsorption = binding

    model.input.model.unit_000 = column

    flow_rate = 0.000575 * cross_section_area * column.col_porosity
    model.input.model.connections.connections_include_ports = 1
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        1.0, 0.0, -1.0, -1.0, -1.0, -1.0, flow_rate
        ]
    model.input.model.connections.switch_000.section = 0

    model.input.model.unit_001.UNIT_TYPE = 'INLET'
    model.input.model.unit_001.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_001.ncomp = n_comp
    # The inlet repeats the initial liquid phase, so that the boundary adds no
    # transient of its own on top of the initialization.
    model.input.model.unit_001.sec_000.const_coeff = initial_liquid
    for coefficient in ['lin_coeff', 'quad_coeff', 'cube_coeff']:
        model.input.model.unit_001.sec_000[coefficient] = [0.0] * n_comp

    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    model.input.solver.consistent_init_mode = consistent_init_mode
    model.input.solver.consistent_init_mode_sens = 1
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 1
    model.input.solver.sections.section_continuity = []
    model.input.solver.sections.section_times = [0.0, t_end]
    model.input.solver.time_integrator.ABSTOL = idas_abstol
    model.input.solver.time_integrator.ALGTOL = idas_algtol
    model.input.solver.time_integrator.RELTOL = idas_reltol
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-14
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.user_solution_times = [0.0, t_end]

    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_solution_bulk = 1
    model.input['return'].unit_000.write_solution_particle = 1
    model.input['return'].unit_000.write_solution_solid = 1
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].write_solution_times = 1

    return model


def equilibrium_residual(c_particle, c_solid, binding, par_porosity,
                         init_particle, init_solid):
    """Residual of the initialization system at one particle shell.

    Returns the relative residual of the three kinds of equation that the
    consistent initialization has to satisfy, so that a returned initial state
    can be judged without knowing the exact solution.

    Parameters
    ----------
    c_particle, c_solid : np.ndarray
        Particle liquid and bound concentrations of one shell.
    binding : dict
        Isotherm parameters as returned by ``get_binding_configuration()``.
    par_porosity : float
        Particle porosity.
    init_particle, init_solid : np.ndarray
        Configured initial particle liquid and bound concentrations, which fix
        the conserved moieties.

    Returns
    -------
    dict
        ``equilibrium`` is the largest relative residual over the protein
        equilibrium conditions, ``electroneutrality`` the relative residual of
        the salt relation and ``moiety`` the largest relative violation of the
        conserved moieties.
    """

    k_a = np.asarray(binding['sma_ka'], float)
    k_d = np.asarray(binding['sma_kd'], float)
    nu = np.asarray(binding['sma_nu'], float)
    sigma = np.asarray(binding['sma_sigma'], float)
    lam = float(binding['sma_lambda'])

    c_particle = np.asarray(c_particle, float)
    c_solid = np.asarray(c_solid, float)

    # Salt capacity available for binding, reduced by the steric shielding of
    # the bound proteins.
    shielded = lam - np.sum((nu[1:] + sigma[1:]) * c_solid[1:])

    equilibrium = 0.0
    with np.errstate(all='ignore'):
        for comp in range(1, len(c_particle)):
            adsorption = k_a[comp] * c_particle[comp] * shielded ** nu[comp]
            desorption = k_d[comp] * c_solid[comp] * c_particle[0] ** nu[comp]
            scale = max(abs(adsorption), abs(desorption), np.finfo(float).tiny)
            residual = abs(adsorption - desorption) / scale
            if not np.isfinite(residual):
                residual = np.inf
            equilibrium = max(equilibrium, residual)

    # Electroneutrality fixes the bound salt.
    bound_salt = lam - np.sum(nu[1:] * c_solid[1:])
    electroneutrality = abs(c_solid[0] - bound_salt) / max(abs(lam),
                                                           np.finfo(float).tiny)

    conserved = (par_porosity * c_particle + (1.0 - par_porosity) * c_solid)
    conserved_init = (par_porosity * np.asarray(init_particle, float)
                      + (1.0 - par_porosity) * np.asarray(init_solid, float))
    scale = np.maximum(np.abs(conserved_init), np.finfo(float).tiny)
    moiety = float(np.max(np.abs(conserved - conserved_init) / scale))

    return {
        'equilibrium': float(equilibrium),
        'electroneutrality': float(electroneutrality),
        'moiety': moiety,
        }
