# -*- coding: utf-8 -*-
"""

This script defines protonation equilibria as rapid-equilibrium mass action law
reactions in the bulk liquid phase of a one dimensional column, used to stress
the nonlinear solver of the consistent initialization, see
src/consistent_initialization.py.

A reaction group with ``IS_KINETIC = 0`` is not integrated in time. CADET-Core
replaces the balance equations of the participating components by the conserved
moieties of the equilibrium reactions, i.e. the left nullspace of their
stoichiometric matrix, plus one algebraic equation per equilibrium reaction, see
the conserved moiety reformulation in CADET-Core. Consistent initialization then
has to project the configured ``INIT_C`` onto that equilibrium manifold, which
is exactly the nonlinear system

.. math::

    L c = L c_0, \\qquad v_l(c) = 0 \\quad l = 1, \\dots, n_\\mathrm{react}

Matthias Bur's bachelor thesis "Solving nonlinear systems with the
pseudo-transient-continuation method" (University of Cologne, 2026)
Section 4.3 is reproduced exactly by ``ph_buffer_case()``,
and ``single_protonation_case()`` is a one reaction reduction of it whose
difficulty is controlled by a single equilibrium constant.

The inlet of the column carries those equilibrium concentrations while the
column is initialized off equilibrium. The initial state after a successful
projection is therefore a steady state of the whole unit, so that the time
integration that follows the initialization cannot fail. This can be verified
by initializing the column at the reference equilibrium, in which case all
nonlinear solvers return it to machine precision.

"""

import numpy as np
from addict import Dict
from scipy.optimize import brentq


# A protonation family is a chain of species from the most protonated form to
# the fully deprotonated one, e.g. Dap2+ -> Dap+ -> Dap, together with the
# equilibrium constants of the steps between them, the total concentration of
# the family and the species that holds that total initially. Reaction l of such
# a step reads
#
#     AH  <->  A- + H+,      v_l = k_l^fwd [AH] - k_l^bwd [A-] [H+],
#
# and with the backward rates fixed to one, k_l^fwd is the equilibrium constant.
# A family is given as a dictionary with the keys 'species', 'constants',
# 'total' and 'initial'.


# The ten protonation equilibria of the pH buffer model of the thesis, section
# 4.3, in the order in which the thesis lists them. The equilibrium constants
# are its k^+ vector and the backward rates are all one. The initial species of
# every family is the one that carries 10 in the thesis initial value C0, i.e.
# the free base for the amines and the undissociated acid for acetate and
# lactate.
_PH_BUFFER_FAMILIES_ = [
    {'species': ['Dap2+', 'Dap+', 'Dap'], 'constants': [2.29e-06, 2.4e-08],
     'total': 10.0, 'initial': 'Dap'},
    {'species': ['Dea+', 'Dea'], 'constants': [1.31e-06],
     'total': 10.0, 'initial': 'Dea'},
    {'species': ['Tris+', 'Tris'], 'constants': [8.47e-06],
     'total': 10.0, 'initial': 'Tris'},
    {'species': ['Imi+', 'Imi'], 'constants': [1.02e-04],
     'total': 10.0, 'initial': 'Imi'},
    {'species': ['Bts+', 'Bts'], 'constants': [3.28e-04],
     'total': 10.0, 'initial': 'Bts'},
    {'species': ['Pip2+', 'Pip+', 'Pip'], 'constants': [4.6e-03, 1.86e-07],
     'total': 10.0, 'initial': 'Pip'},
    {'species': ['Ace', 'Ace-'], 'constants': [1.75e-02],
     'total': 10.0, 'initial': 'Ace'},
    {'species': ['Lac', 'Lac-'], 'constants': [1.38e-01],
     'total': 10.0, 'initial': 'Lac'},
]

# Initial proton concentration of the thesis pH buffer model, its C0[19].
_PH_BUFFER_INIT_H_ = 10.0 ** -0.5


def protonation_network(families, init_h):
    """Assemble a reaction network of protonation equilibria.

    Parameters
    ----------
    families : list of dict
        One entry per protonation family with the keys ``species`` (the species
        names from the most protonated form to the fully deprotonated one),
        ``constants`` (the equilibrium constants of the steps between
        consecutive species), ``total`` (the total concentration of the family)
        and ``initial`` (the species that holds that total in the initial
        state, by default the most protonated one). A family of ``m`` species
        contributes ``m - 1`` reactions and needs ``m - 1`` constants.
    init_h : float
        Initial proton concentration.

    Returns
    -------
    dict
        Description of the network with the keys ``names`` (component names, the
        proton last), ``stoichiometry`` (``NCOMP x NREACT``), ``k_fwd``,
        ``k_bwd``, ``init_c`` and ``families``.
    """

    names = []
    for family in families:
        if len(family['constants']) != len(family['species']) - 1:
            raise ValueError(
                f"Family {family['species']} needs "
                f"{len(family['species']) - 1} equilibrium constants, got "
                f"{len(family['constants'])}"
            )
        names.extend(family['species'])
    names.append('H+')

    n_comp = len(names)
    n_react = sum(len(family['constants']) for family in families)

    stoichiometry = np.zeros((n_comp, n_react))
    k_fwd = []
    init_c = np.zeros(n_comp)

    index = {name: i for i, name in enumerate(names)}
    reaction = 0
    for family in families:
        species = family['species']
        initial = family.get('initial', species[0])
        if initial not in species:
            raise ValueError(f"Initial species {initial} is not part of "
                             f"family {species}")
        init_c[index[initial]] = family['total']
        for step, constant in enumerate(family['constants']):
            stoichiometry[index[species[step]], reaction] = -1.0
            stoichiometry[index[species[step + 1]], reaction] = 1.0
            stoichiometry[index['H+'], reaction] = 1.0
            k_fwd.append(constant)
            reaction += 1
    init_c[index['H+']] = init_h

    return {
        'names': names,
        'stoichiometry': stoichiometry,
        'k_fwd': k_fwd,
        'k_bwd': [1.0] * n_react,
        'init_c': init_c,
        'families': families,
        }


def proton_weights(network):
    """Weights of the proton moiety of a protonation network.

    The conserved moieties of the network are the family totals and the total
    amount of bound and free protons. The latter weighs a species by the number
    of protons it carries relative to the fully deprotonated form of its family,
    and the proton itself by one.
    """

    weights = np.zeros(len(network['names']))
    index = {name: i for i, name in enumerate(network['names'])}
    for family in network['families']:
        species = family['species']
        for step, name in enumerate(species):
            weights[index[name]] = len(species) - 1 - step
    weights[index['H+']] = 1.0
    return weights


def conservation_matrix(network):
    """Basis of the conserved moieties of a protonation network.

    One row per family, which weighs every species of that family by one and
    thus states that its total concentration is conserved, plus the proton
    moiety of ``proton_weights()``. Together these span the left nullspace of
    the stoichiometric matrix, which is what the conserved moiety reformulation
    of CADET-Core puts in place of the balance equations of the participating
    components.
    """

    index = {name: i for i, name in enumerate(network['names'])}
    rows = []
    for family in network['families']:
        row = np.zeros(len(network['names']))
        for name in family['species']:
            row[index[name]] = 1.0
        rows.append(row)
    rows.append(proton_weights(network))
    return np.array(rows)


def equilibrium_concentrations(network):
    """Analytic equilibrium of a protonation network, for the given ``init_c``.

    For a fixed proton concentration :math:`h` the equilibria of one family fix
    the ratios of its species, so its distribution follows from its total
    concentration alone. The equilibrium is therefore determined by the single
    scalar equation that the proton moiety of the resulting distribution equals
    the proton moiety of the initial state, which is solved by bisection.

    Parameters
    ----------
    network : dict
        Network as returned by ``protonation_network()``.

    Returns
    -------
    np.ndarray
        Equilibrium concentrations in the component order of the network.
    """

    weights = proton_weights(network)
    proton_moiety = float(weights @ network['init_c'])

    def speciate(h):
        concentrations = np.zeros(len(network['names']))
        index = {name: i for i, name in enumerate(network['names'])}
        for family in network['families']:
            species = family['species']
            # Abundance of every species relative to the most protonated form.
            relative = [1.0]
            for constant in family['constants']:
                relative.append(relative[-1] * constant / h)
            relative = np.array(relative)
            most_protonated = family['total'] / relative.sum()
            for step, name in enumerate(species):
                concentrations[index[name]] = most_protonated * relative[step]
        concentrations[index['H+']] = h
        return concentrations

    def balance(h):
        return float(weights @ speciate(h)) - proton_moiety

    # The proton moiety is strictly increasing in h, so the root is unique and
    # bracketed by any sufficiently wide interval.
    h = brentq(balance, 1e-16, 1e4, xtol=1e-20, rtol=8.9e-16, maxiter=500)
    return speciate(h)


def equilibrium_rate_terms(network, concentrations):
    """Forward and backward terms of every reaction of a protonation network.

    The exponents are the CADET-Core defaults for a mass action law reaction,
    i.e. they are taken from the stoichiometry.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Forward and backward terms, one entry per reaction.
    """

    stoichiometry = network['stoichiometry']
    concentrations = np.asarray(concentrations, float)
    forward = np.zeros(stoichiometry.shape[1])
    backward = np.zeros(stoichiometry.shape[1])
    for reaction in range(stoichiometry.shape[1]):
        column = stoichiometry[:, reaction]
        exp_fwd = np.maximum(0.0, -column)
        exp_bwd = np.maximum(0.0, column)
        forward[reaction] = network['k_fwd'][reaction] * np.prod(
            concentrations ** exp_fwd
            )
        backward[reaction] = network['k_bwd'][reaction] * np.prod(
            concentrations ** exp_bwd
            )
    return forward, backward


def equilibrium_rates(network, concentrations):
    """Reaction rates of a protonation network, zero at equilibrium."""

    forward, backward = equilibrium_rate_terms(network, concentrations)
    return forward - backward


def get_model(network, consistency_solver=None, t_end=1.0, nelem=1, polydeg=1,
              idas_abstol=1e-10, idas_algtol=1e-10, idas_reltol=1e-8,
              use_analytic_jacobian=True, consistent_init_mode=1,
              init_c=None, inlet_c=None):
    """Build the CADET-Core configuration of a rapid-equilibrium reaction case.

    Parameters
    ----------
    network : dict
        Network as returned by ``protonation_network()``.
    consistency_solver : dict or None
        Contents of the ``consistency_solver`` group, e.g.
        ``{'SOLVER_NAME': 'PTC'}``. ``None`` leaves the CADET-Core default,
        which is a composite of ``ATRN_ERR`` and ``LEVMAR``.
    t_end : float
        Length of the single time section. Only the initialization is of
        interest here, so one short section suffices.
    nelem, polydeg : int
        Discontinuous Galerkin discretization of the bulk. The initialization
        problem is solved per axial node and is the same at every node, so the
        coarsest grid already exercises it.
    idas_abstol, idas_algtol, idas_reltol : float
        Time integrator tolerances.
    use_analytic_jacobian : bool
        Use the analytic Jacobian. Note that automatic differentiation of the
        conserved moiety reformulation is not supported, see the limitations
        section of src/consistent_initialization.py.
    consistent_init_mode : int
        ``CONSISTENT_INIT_MODE``, 1 is full initialization.
    init_c : np.ndarray or None
        Overrides the initial concentrations of the network, which is how the
        control simulation of src/consistent_initialization.py starts a case at
        its own reference equilibrium.
    inlet_c : np.ndarray or None
        Overrides the inlet concentrations, which otherwise are the analytic
        equilibrium of the network. Needed for a network that is not a
        protonation network and thus has no analytic equilibrium, see
        ``check_rapid_equilibrium_support()``.

    Returns
    -------
    addict.Dict
        CADET-Core configuration.
    """

    n_comp = len(network['names'])
    initial = network['init_c'] if init_c is None else np.asarray(init_c, float)
    inlet = (equilibrium_concentrations(network) if inlet_c is None
             else np.asarray(inlet_c, float))

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
    # No particles, so the bulk fills the column and both porosities are one.
    column.col_porosity = 1.0
    column.total_porosity = 1.0
    column.npartype = 0
    column.ncomp = n_comp
    column.init_c = list(initial)
    column.col_dispersion = 5.75e-08

    column.discretization.SPATIAL_METHOD = 'DG'
    column.discretization.USE_COLLOCATION_DG = 1
    column.discretization.POLYDEG = polydeg
    column.discretization.NELEM = nelem
    column.discretization.USE_ANALYTIC_JACOBIAN = 1 if use_analytic_jacobian else 0
    if consistency_solver is not None:
        column.discretization.consistency_solver = dict(consistency_solver)

    column.NREAC_LIQUID = 1
    column.liquid_reaction_000.TYPE = 'MASS_ACTION_LAW'
    # Rapid equilibrium: the reactions are not integrated in time, their rates
    # are fixed by the requirement that the net flux vanishes.
    column.liquid_reaction_000.IS_KINETIC = 0
    column.liquid_reaction_000.MAL_KFWD = list(network['k_fwd'])
    column.liquid_reaction_000.MAL_KBWD = list(network['k_bwd'])
    # NCOMP x NREACT in row major storage.
    column.liquid_reaction_000.MAL_STOICHIOMETRY = list(
        network['stoichiometry'].flatten()
        )

    model.input.model.unit_000 = column

    flow_rate = 0.000575 * cross_section_area
    model.input.model.connections.connections_include_ports = 1
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        1.0, 0.0, -1.0, -1.0, -1.0, -1.0, flow_rate
        ]
    model.input.model.connections.switch_000.section = 0

    model.input.model.unit_001.UNIT_TYPE = 'INLET'
    model.input.model.unit_001.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_001.ncomp = n_comp
    # The inlet carries the equilibrium, see the module docstring.
    model.input.model.unit_001.sec_000.const_coeff = list(inlet)
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
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].write_solution_times = 1

    return model


def ph_buffer_network():
    """The pH buffer model of the thesis, section 4.3.

    Nineteen species in ten protonation equilibria, nine conserved moieties and
    ten algebraic equilibrium equations. The thesis reports that a plain Newton
    solver fails on this system because the Jacobian is singular at the initial
    iterate, while pseudo-transient continuation converges in 38 iterations.
    """

    return protonation_network(_PH_BUFFER_FAMILIES_, _PH_BUFFER_INIT_H_)


def single_protonation_network(equilibrium_constant, total=10.0, init_h=0.0):
    """One protonation equilibrium ``AH <-> A- + H+``.

    The smallest case of the family and the one whose difficulty is easiest to
    reason about: the smaller the equilibrium constant, the smaller the
    equilibrium proton concentration, and the further the equilibrium lies from
    the initial state, in which all of the family sits in its protonated form.

    Parameters
    ----------
    equilibrium_constant : float
        Forward rate of the single reaction; the backward rate is one.
    total : float
        Total concentration of the family.
    init_h : float
        Initial proton concentration.
    """

    return protonation_network(
        [{'species': ['AH', 'A-'], 'constants': [equilibrium_constant],
          'total': total, 'initial': 'AH'}],
        init_h,
        )
