# -*- coding: utf-8 -*-
"""

This script defines a test template that compares the nonlinear solvers of the
consistent initialization of CADET-Core on initialization problems that are hard
enough to tell them apart.

Background
----------

Whenever a unit operation holds algebraic equations, i.e. a quasi-stationary
binding model or a rapid-equilibrium reaction, the configured initial values are
in general inconsistent with those equations and CADET-Core has to project them
onto the solution manifold before the time integration can start. That
projection is a nonlinear equation system and is solved by one of the solvers
under ``/input/model/unit_XXX/discretization/consistency_solver``, see
:ref:`non_consistency_solver_parameters` in the CADET-Core documentation:

- ``ATRN_RES``: adaptive trust-region Newton, NLEQ-RES
- ``ATRN_ERR``: robust adaptive trust-region Newton, NLEQ-ERR
- ``LEVMAR``: Levenberg-Marquardt
- ``PTC``: pseudo-transient continuation
- ``COMPOSITE``: applies several solvers in sequence; the CADET-Core default is
  ``ATRN_ERR`` followed by ``LEVMAR``

Failures of this projection are a recurring topic of the CADET forum, usually
reported as a simulation that does not start or that stalls at a section
transition, see for instance
https://forum.cadet-web.de/t/failing-simulation-for-non-integer-nu-in-sma/354,
https://forum.cadet-web.de/t/bisma-model-error/942,
https://forum.cadet-web.de/t/error-when-using-quasi-stationary-binding/703 and
https://forum.cadet-web.de/t/issue-with-rapid-equilibrium/252. The usual
workaround is to fall back to a lean initialization, which simply does not solve
the algebraic equations, so the underlying initialization problem stays
unsolved.

What this study does
--------------------

For every case and every nonlinear solver, one simulation of a single short time
section is run, because only the initial state is of interest, and the resulting
initial state is judged by three measures that do not depend on the solver under
test:

- ``status``: whether CADET-Core completed the simulation at all, and if not,
  whether it failed on the first step at :math:`t = 0`, which is what an
  inconsistent initial state looks like from the outside, or later
- ``residual``: the largest relative residual of the algebraic equations,
  evaluated in Python at the initial state that CADET-Core reports. This catches
  a solver that returns a point that is not a solution without CADET-Core
  noticing
- ``reference_error``: the largest absolute deviation from an independently
  computed exact solution, where one is available

Each case also runs one control simulation that does not depend on the solvers,
which establishes that a failure belongs to the initialization and not to the
rest of the setting, see ``_control_result()``.

Case families
-------------

``rapid_equilibrium_reaction``
    Protonation equilibria as a rapid-equilibrium mass action law reaction in
    the bulk liquid phase, see
    src/benchmark_models/setting_COL1D_rapidEqMAL_initialization.py. These are
    the CADET-Core counterpart of the experiments of Matthias Bur's bachelor
    thesis on the pseudo-transient continuation method, including its pH buffer
    model of 19 species in 10 reactions. Their exact equilibrium is computed
    analytically, so the accuracy of every solver is known and not only its
    residual. The difficulty is controlled by the equilibrium constant: the
    smaller it is, the further the equilibrium lies from the configured initial
    state and the larger the dynamic range that the solver has to cross.

``quasi_stationary_binding``
    Preloaded columns with a quasi-stationary steric mass action isotherm, see
    src/benchmark_models/setting_COL1D_SMA_preloaded_initialization.py, built
    around the two preloaded cases reported on the forum. The difficulty is
    controlled by the initial salt concentration and by whether the
    characteristic charges are integers.

Adding a case
-------------

A case is a dictionary with a ``build`` callable that takes a
``consistency_solver`` configuration and returns a CADET-Core configuration, an
``evaluate`` callable that turns a finished simulation into the measures above,
and a ``control`` callable that returns a configuration whose success does not
depend on the initialization. See ``_rapid_equilibrium_cases()`` and
``_binding_cases()``; both only assemble such dictionaries from the settings
modules, so a new case family needs a settings module and one more builder.

Known limitations
-----------------

- The column models of CADET-Core do not check whether the nonlinear solve of
  the algebraic equations converged, so a failed consistent initialization is
  never reported as such. It surfaces as a failed factorization of the time
  derivative system that follows it, as a residual that has become NaN because
  an iterate left the physical range, or as a time integration that gives up on
  its first step, see ``_is_initialization_failure()``. A failing case can be
  diagnosed further by rerunning its stored h5 file with a debug build of
  ``cadet-cli -L Debug``, which logs the consistency error and the initialized
  state. Run the suite with ``--delete-h5-files=false`` to keep those files.
- A solver can also return a point that is not a solution without CADET-Core
  noticing at all, in which case the simulation runs and the ``residual``
  column of the report is large. This is why the study reports the residual and
  not only whether a simulation completed.

"""

import json
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
from cadet import Cadet

import src.utility.convergence as convergence
from src.benchmark_models import setting_COL1D_rapidEqMAL_initialization as mal
from src.benchmark_models import setting_COL1D_SMA_preloaded_initialization as sma


# Nonlinear solvers of the consistent initialization, as configured under
# discretization/consistency_solver. ``None`` leaves the CADET-Core default,
# which is a composite of ATRN_ERR and LEVMAR.
NONLINEAR_SOLVERS = {
    'DEFAULT': None,
    'ATRN_RES': {'SOLVER_NAME': 'ATRN_RES'},
    'ATRN_ERR': {'SOLVER_NAME': 'ATRN_ERR'},
    'LEVMAR': {'SOLVER_NAME': 'LEVMAR'},
    'PTC': {'SOLVER_NAME': 'PTC'},
    }

# Outcome of a single run.
CONVERGED = 'converged'
INITIALIZATION_FAILED = 'initialization_failed'
INTEGRATION_FAILED = 'integration_failed'

# Equilibrium constants of the single protonation ladder. The first is a
# balanced equilibrium, the last two are those of acetic and diaminopropane
# protonation in the pH buffer model of the thesis.
_SINGLE_PROTONATION_CONSTANTS_ = [1.0, 1.75e-02, 1.31e-06, 2.4e-08]

# Initial salt concentrations of the preloaded column family, from the loading
# conditions of the load-wash-elute benchmark down to a rinsed column.
_PRELOADED_SALT_ = [500.0, 100.0, 50.0, 1.0]


def _run(model, file_path, cadet_path):
    """Run one simulation and report outcome, message and wall clock time."""

    simulation = Cadet(install_path=cadet_path)
    simulation.root = model
    simulation.filename = str(file_path)
    simulation.save()

    start = time.perf_counter()
    information = simulation.run_simulation()
    elapsed = time.perf_counter() - start

    message = '\n'.join(
        part.strip() for part in
        [getattr(information, 'log', '') or '', information.error_message or '']
        if part and part.strip()
        )
    if information.return_code == 0:
        simulation.load_from_file()
        return CONVERGED, message, elapsed, simulation

    status = INITIALIZATION_FAILED if _is_initialization_failure(message) \
        else INTEGRATION_FAILED
    return status, message, elapsed, None


# CADET-Core reports a failed consistent initialization in several ways: the
# nonlinear solve of the algebraic equations is not checked at all in the column
# models, so it surfaces either as a failed factorization of the time derivative
# system that follows it, as a residual that has become NaN because an iterate
# left the physical range, or as a time integration that gives up on its very
# first step because the initial state is inconsistent. None of these can be
# confused with a failure later in the section.
_INITIALIZATION_FAILURE_MARKERS_ = [
    'consistentinitial',
    'factorize() failed',
    'nan or infinite',
    'initialization failed',
    'at t = 0.000000',
    ]


def _is_initialization_failure(message):
    """Whether a CADET-Core failure belongs to the consistent initialization."""

    lowered = message.lower()
    return any(marker in lowered for marker in _INITIALIZATION_FAILURE_MARKERS_)


def _initial_bulk(simulation, n_comp):
    """Bulk concentrations of all axial nodes at the first solution time."""

    solution = np.asarray(
        convergence.get_simulation(simulation).root.output.solution.unit_000
        .solution_bulk
        )
    return solution[0].reshape(-1, n_comp)


def _initial_particle(simulation, n_comp):
    """Particle liquid and bound concentrations at the first solution time."""

    root = convergence.get_simulation(simulation).root.output.solution.unit_000
    particle = np.asarray(root.solution_particle)[0].reshape(-1, n_comp)
    solid = np.asarray(root.solution_solid)[0].reshape(-1, n_comp)
    return particle, solid


def _evaluate_rapid_equilibrium(simulation, network):
    """Measures of a rapid-equilibrium reaction case."""

    n_comp = len(network['names'])
    concentrations = _initial_bulk(simulation, n_comp)
    reference = mal.equilibrium_concentrations(network)
    conservation = mal.conservation_matrix(network)
    moieties = conservation @ network['init_c']

    residual = 0.0
    reference_error = 0.0
    moiety_error = 0.0
    min_value = float(np.min(concentrations))
    for state in concentrations:
        forward, backward = mal.equilibrium_rate_terms(network, state)
        # The net rate is compared to the size of the terms it cancels, so that
        # a reaction of small turnover does not look solved by default.
        scale = np.maximum(np.maximum(np.abs(forward), np.abs(backward)),
                           np.finfo(float).tiny)
        residual = max(residual,
                       float(np.max(np.abs(forward - backward) / scale)))
        reference_error = max(reference_error,
                              float(np.max(np.abs(state - reference))))
        moiety_error = max(
            moiety_error,
            float(np.max(np.abs(conservation @ state - moieties)
                         / np.maximum(np.abs(moieties),
                                      np.finfo(float).tiny))),
            )

    return {
        'residual': residual,
        'reference_error': reference_error,
        'moiety_error': moiety_error,
        'min_value': min_value,
        }


def _evaluate_binding(simulation, binding, par_porosity, init_particle,
                      init_solid):
    """Measures of a quasi-stationary binding case."""

    n_comp = len(init_particle)
    particle, solid = _initial_particle(simulation, n_comp)

    residual = 0.0
    moiety_error = 0.0
    for shell in range(particle.shape[0]):
        measures = sma.equilibrium_residual(
            particle[shell], solid[shell], binding, par_porosity,
            init_particle, init_solid,
            )
        residual = max(residual, measures['equilibrium'],
                       measures['electroneutrality'])
        moiety_error = max(moiety_error, measures['moiety'])

    return {
        'residual': residual,
        'reference_error': None,
        'moiety_error': moiety_error,
        'min_value': float(min(np.min(particle), np.min(solid))),
        }


def check_rapid_equilibrium_support(cadet_path, output_path):
    """Check whether CADET-Core honours ``IS_KINETIC = 0`` for reactions.

    Builds the smallest possible rapid-equilibrium case, a single reaction
    ``A <-> B`` with equilibrium constant 2 started at ``[1, 0]``, whose
    equilibrium is ``[1/3, 2/3]``. A build that ignores ``IS_KINETIC = 0``
    integrates the reaction kinetically and reports the configured initial state
    instead, which this detects.

    Returns
    -------
    bool
        True if the conserved moiety reformulation is in effect.
    """

    network = {
        'names': ['A', 'B'],
        'stoichiometry': np.array([[-1.0], [1.0]]),
        'k_fwd': [2.0],
        'k_bwd': [1.0],
        'init_c': np.array([1.0, 0.0]),
        'families': [],
        }
    # Two components without a proton, so the network has no analytic
    # equilibrium routine and the inlet, which carries the equilibrium, is
    # given explicitly.
    model = mal.get_model(network, inlet_c=[1.0 / 3.0, 2.0 / 3.0])

    status, message, _, simulation = _run(
        model, Path(output_path) / 'support_check_rapid_equilibrium.h5',
        cadet_path,
        )
    if status != CONVERGED:
        print('Rapid-equilibrium support check could not be run: ' + message)
        return False

    initial = _initial_bulk(simulation, 2)[0]
    supported = bool(np.max(np.abs(initial - np.array([1.0 / 3.0, 2.0 / 3.0])))
                     < 1e-6)
    if not supported:
        print('CADET-Core reports ' + np.array2string(initial, precision=6)
              + ' as the initial state of a rapid-equilibrium reaction whose '
              'equilibrium is [0.333333, 0.666667]. This build ignores '
              'IS_KINETIC = 0 for reaction models; the rapid-equilibrium cases '
              'are skipped.')
    return supported


def _cadet_version(output_path):
    """CADET-Core version and commit of the support check simulation."""

    file_path = Path(output_path) / 'support_check_rapid_equilibrium.h5'
    try:
        with h5py.File(file_path, 'r') as handle:
            version = handle['meta/CADET_VERSION'][()].decode()
            commit = handle['meta/CADET_COMMIT'][()].decode()
            branch = handle['meta/CADET_BRANCH'][()].decode()
        return f'{version} ({branch}, {commit[:10]})'
    except Exception:  # noqa: BLE001 - the version is informational only
        return 'unknown'


def _rapid_equilibrium_cases(small_test, use_analytic_jacobian):
    """Cases of the rapid-equilibrium reaction family."""

    cases = []

    constants = _SINGLE_PROTONATION_CONSTANTS_
    if small_test:
        constants = [constants[0], constants[-1]]

    for constant in constants:
        network = mal.single_protonation_network(constant)
        cases.append({
            'name': f'protonation_K{constant:.2e}',
            'group': 'rapid_equilibrium_reaction',
            'source': 'thesis experiment 1, reduced to one reaction',
            'network': network,
            'use_analytic_jacobian': use_analytic_jacobian,
            })

    network = mal.ph_buffer_network()
    cases.append({
        'name': 'ph_buffer_19comp',
        'group': 'rapid_equilibrium_reaction',
        'source': 'thesis section 4.3, pH buffer model',
        'network': network,
        'use_analytic_jacobian': use_analytic_jacobian,
        })

    for case in cases:
        network = case['network']
        case['build'] = (
            lambda consistency_solver, network=network, case=case:
            mal.get_model(
                network, consistency_solver=consistency_solver,
                use_analytic_jacobian=case['use_analytic_jacobian'],
                )
            )
        # Control: start the column at the analytic equilibrium, so that the
        # initialization has nothing left to do. If this succeeds, the setting
        # and its time integration are sound and every failure of the same case
        # belongs to the initialization.
        case['control'] = (
            lambda network=network, case=case: mal.get_model(
                network, init_c=mal.equilibrium_concentrations(network),
                use_analytic_jacobian=case['use_analytic_jacobian'],
                )
            )
        case['control_description'] = 'initialized at the analytic equilibrium'
        case['evaluate'] = (
            lambda simulation, network=network:
            _evaluate_rapid_equilibrium(simulation, network)
            )

    return cases


def _binding_cases(small_test, use_analytic_jacobian):
    """Cases of the quasi-stationary binding family."""

    cases = []

    salts = _PRELOADED_SALT_ if not small_test else [_PRELOADED_SALT_[0],
                                                     _PRELOADED_SALT_[-1]]

    variants = [('nu_benchmark', sma.NU_BENCHMARK,
                 'load-wash-elute benchmark charges, preloaded column'),
                ('nu_fractional', sma.NU_FRACTIONAL,
                 'forum topic 354, fitted fractional characteristic charges'),
                ('nu_integer', sma.NU_INTEGER,
                 'forum topic 354, the same charges rounded to integers')]
    if small_test:
        variants = [variants[0], variants[1]]

    for label, nu, source in variants:
        for salt in salts:
            cases.append({
                'name': f'sma_preloaded_{label}_salt{salt:g}',
                'group': 'quasi_stationary_binding',
                'source': source,
                'init_salt': salt,
                'nu': nu,
                'use_analytic_jacobian': use_analytic_jacobian,
                })

    for case in cases:
        settings = dict(init_salt=case['init_salt'], nu=case['nu'],
                        use_analytic_jacobian=case['use_analytic_jacobian'])
        binding = sma.get_binding_configuration(nu=case['nu'])
        init_particle = np.array([case['init_salt'], 1.0, 1.0, 1.0])
        init_solid = np.array([binding['sma_lambda'], 0.0, 0.0, 0.0])

        case['build'] = (
            lambda consistency_solver, settings=settings: sma.get_model(
                consistency_solver=consistency_solver, **settings
                )
            )
        # Control: the same setting on an empty column. The initialization
        # problem is then solved by the configured initial state itself, because
        # every protein concentration and every protein loading is zero, so a
        # control that converges establishes that geometry, transport, flow and
        # isotherm are sound and that only the preloading makes the case hard.
        case['control'] = (
            lambda settings=settings: sma.get_model(
                init_protein=(0.0, 0.0, 0.0),
                init_bound=[sma.get_binding_configuration()['sma_lambda'],
                            0.0, 0.0, 0.0],
                **settings
                )
            )
        case['control_description'] = 'the same setting on an empty column'
        case['evaluate'] = (
            lambda simulation, binding=binding, init_particle=init_particle,
            init_solid=init_solid: _evaluate_binding(
                simulation, binding, 0.75, init_particle, init_solid
                )
            )

    return cases


def build_cases(small_test=False, use_analytic_jacobian=True,
                include_reactions=True, include_binding=True):
    """Assemble the cases of the study."""

    cases = []
    if include_reactions:
        cases.extend(_rapid_equilibrium_cases(small_test,
                                              use_analytic_jacobian))
    if include_binding:
        cases.extend(_binding_cases(small_test, use_analytic_jacobian))
    return cases


def _control_result(case, output_path, cadet_path):
    """Run the solver independent control simulation of a case."""

    status, message, _, _ = _run(
        case['control'](),
        Path(output_path) / f"{case['name']}_control.h5",
        cadet_path,
        )
    return {
        'status': status,
        'description': case['control_description'],
        'message': message if status != CONVERGED else '',
        }


def run_case(case, solvers, output_path, cadet_path):
    """Run one case with every nonlinear solver and collect the measures."""

    results = {}
    for solver_name in solvers:
        model = case['build'](NONLINEAR_SOLVERS[solver_name])
        status, message, elapsed, simulation = _run(
            model,
            Path(output_path) / f"{case['name']}_{solver_name}.h5",
            cadet_path,
            )

        entry = {
            'status': status,
            'runtime': elapsed,
            'message': '' if status == CONVERGED else message,
            'residual': None,
            'reference_error': None,
            'moiety_error': None,
            'min_value': None,
            }
        if status == CONVERGED:
            try:
                entry.update(case['evaluate'](simulation))
            except Exception as error:  # noqa: BLE001 - reported, not raised
                entry['message'] = f'evaluation failed: {error}'
        results[solver_name] = entry

    return results


def _format_report(report):
    """Readable table of the study, returned as a string."""

    solvers = report['solvers']
    lines = []
    lines.append('Consistent initialization: nonlinear solver comparison')
    lines.append('CADET-Core: ' + report['cadet_version'])
    lines.append('')
    lines.append('Entries are the largest relative residual of the algebraic '
                 'equations at the')
    lines.append('reported initial state. "init" marks a failure of the '
                 'consistent initialization,')
    lines.append('"time" a failure of the time integration that followed a '
                 'successful one.')
    lines.append('')

    width = max(len(case['name']) for case in report['cases']) + 2
    header = 'case'.ljust(width) + ''.join(s.rjust(12) for s in solvers)
    lines.append(header)
    lines.append('-' * len(header))

    for case in report['cases']:
        row = case['name'].ljust(width)
        for solver in solvers:
            entry = case['results'][solver]
            if entry['status'] == INITIALIZATION_FAILED:
                row += 'init'.rjust(12)
            elif entry['status'] == INTEGRATION_FAILED:
                row += 'time'.rjust(12)
            elif entry['residual'] is None:
                row += '?'.rjust(12)
            else:
                row += f"{entry['residual']:.1e}".rjust(12)
        lines.append(row)

    lines.append('')
    lines.append('Deviation from the analytic equilibrium, where available:')
    lines.append('')
    lines.append(header)
    lines.append('-' * len(header))
    for case in report['cases']:
        if all(case['results'][s]['reference_error'] is None for s in solvers):
            continue
        row = case['name'].ljust(width)
        for solver in solvers:
            entry = case['results'][solver]
            if entry['reference_error'] is None:
                row += ('init' if entry['status'] == INITIALIZATION_FAILED
                        else 'time' if entry['status'] == INTEGRATION_FAILED
                        else '-').rjust(12)
            else:
                row += f"{entry['reference_error']:.1e}".rjust(12)
        lines.append(row)

    lines.append('')
    lines.append('Smallest concentration of the reported initial state. A '
                 'negative value is')
    lines.append('unphysical and is what makes a fractional exponent of a '
                 'concentration undefined:')
    lines.append('')
    lines.append(header)
    lines.append('-' * len(header))
    for case in report['cases']:
        row = case['name'].ljust(width)
        for solver in solvers:
            entry = case['results'][solver]
            if entry['min_value'] is None:
                row += ('init' if entry['status'] == INITIALIZATION_FAILED
                        else 'time' if entry['status'] == INTEGRATION_FAILED
                        else '-').rjust(12)
            else:
                row += f"{entry['min_value']:.1e}".rjust(12)
        lines.append(row)

    lines.append('')
    lines.append('Control simulations, which do not depend on the nonlinear '
                 'solver. A control')
    lines.append('that converges establishes that every failure of its case '
                 'belongs to the')
    lines.append('consistent initialization:')
    for case in report['cases']:
        control = case['control']
        lines.append(f"  {case['name'].ljust(width)}{control['status']}"
                     f"  ({control['description']})")

    diagnostics = {}
    for case in report['cases']:
        for solver in solvers:
            entry = case['results'][solver]
            if entry['status'] == CONVERGED:
                continue
            for line in entry['message'].splitlines():
                if '[Error' in line or 'NAN or infinite' in line:
                    diagnostics[line.strip()] = diagnostics.get(line.strip(),
                                                                0) + 1

    if diagnostics:
        lines.append('')
        lines.append('CADET-Core messages of the failing runs, by number of '
                     'runs:')
        for line, count in sorted(diagnostics.items(), key=lambda kv: -kv[1]):
            lines.append(f'  {count:4d}  {line[:150]}')

    return '\n'.join(lines) + '\n'


def _plot_report(report, file_path):
    """Status and accuracy matrix of the study.

    One cell per case and solver. A converged run is filled from a single hue
    ramp by the order of magnitude of its residual, a failed run is a neutral
    cell, and every cell carries a text label, so that the outcome never rests
    on colour alone.
    """

    solvers = report['solvers']
    cases = report['cases']
    if not cases:
        return

    exponents = np.full((len(cases), len(solvers)), np.nan)
    labels = np.empty((len(cases), len(solvers)), dtype=object)

    for row, case in enumerate(cases):
        for column, solver in enumerate(solvers):
            entry = case['results'][solver]
            if entry['status'] == INITIALIZATION_FAILED:
                labels[row, column] = 'init'
            elif entry['status'] == INTEGRATION_FAILED:
                labels[row, column] = 'time'
            elif entry['residual'] is None:
                labels[row, column] = '?'
            else:
                residual = max(entry['residual'], 1e-16)
                exponents[row, column] = np.log10(residual)
                labels[row, column] = f'{np.log10(residual):.0f}'

    figure, axes = plt.subplots(
        figsize=(1.35 * len(solvers) + 4.0, 0.42 * len(cases) + 2.6)
        )
    colormap = matplotlib.colormaps['Blues'].with_extremes(bad='#E3E5E8')
    finite = exponents[np.isfinite(exponents)]
    vmin = float(np.min(finite)) if finite.size else -16.0
    vmax = float(np.max(finite)) if finite.size else 0.0
    if vmax - vmin < 1.0:
        vmax = vmin + 1.0
    image = axes.imshow(np.ma.masked_invalid(exponents), cmap=colormap,
                        vmin=vmin, vmax=vmax, aspect='auto')

    for row in range(len(cases)):
        for column in range(len(solvers)):
            value = exponents[row, column]
            if np.isfinite(value):
                shade = (value - vmin) / (vmax - vmin)
                colour = 'white' if shade > 0.55 else '#1b1f24'
            else:
                colour = '#57606a'
            axes.text(column, row, labels[row, column], ha='center',
                      va='center', fontsize=8, color=colour)

    axes.set_xticks(range(len(solvers)), solvers, rotation=30, ha='right')
    axes.set_yticks(range(len(cases)), [case['name'] for case in cases],
                    fontsize=8)
    axes.set_xticks(np.arange(len(solvers) + 1) - 0.5, minor=True)
    axes.set_yticks(np.arange(len(cases) + 1) - 0.5, minor=True)
    axes.grid(which='minor', color='white', linewidth=2)
    axes.tick_params(which='minor', length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)

    bar = figure.colorbar(image, ax=axes, pad=0.02)
    bar.set_label('log10 of the relative residual of the algebraic equations,'
                  '\ndarker is a larger residual')
    bar.outline.set_visible(False)

    axes.set_title('Consistent initialization\n'
                   'grey cells failed: "init" in the initialization, "time" in '
                   'the time integration after it',
                   fontsize=10, loc='left')
    figure.tight_layout()
    figure.savefig(file_path, dpi=200)
    plt.close(figure)


def consistent_initialization_tests(n_jobs, small_test, output_path,
                                    cadet_path, solvers=None,
                                    use_analytic_jacobian=True):
    """Compare the nonlinear solvers of the consistent initialization.

    Parameters
    ----------
    n_jobs : int
        Accepted for a uniform interface with the other test groups but unused:
        the study measures the runtime of each initialization, so the
        simulations are run one after another. They are small and the whole
        study takes seconds.
    small_test : bool
        Run a reduced set of cases.
    output_path : string or Path
        Output folder of the simulations, the report and the figure.
    cadet_path : string
        CADET installation path.
    solvers : list of string or None
        Subset of ``NONLINEAR_SOLVERS`` to compare, ``None`` compares all.
    use_analytic_jacobian : bool
        Use the analytic Jacobian. Automatic differentiation of the conserved
        moiety reformulation is not supported, see the known limitations in the
        module docstring.

    Returns
    -------
    dict
        The report that is also written to ``consistent_initialization.json``.
    """

    output_path = str(output_path)
    os.makedirs(output_path, exist_ok=True)

    solvers = list(NONLINEAR_SOLVERS) if solvers is None else list(solvers)
    unknown = [name for name in solvers if name not in NONLINEAR_SOLVERS]
    if unknown:
        raise ValueError(f'Unknown nonlinear solvers: {unknown}')

    include_reactions = check_rapid_equilibrium_support(cadet_path, output_path)

    cases = build_cases(small_test=small_test,
                        use_analytic_jacobian=use_analytic_jacobian,
                        include_reactions=include_reactions)

    report = {
        'cadet_version': _cadet_version(output_path),
        'solvers': solvers,
        'rapid_equilibrium_supported': include_reactions,
        'use_analytic_jacobian': use_analytic_jacobian,
        'cases': [],
        }

    for case in cases:
        print(f"  consistent initialization: {case['name']}")
        results = run_case(case, solvers, output_path, cadet_path)
        report['cases'].append({
            'name': case['name'],
            'group': case['group'],
            'source': case['source'],
            'control': _control_result(case, output_path, cadet_path),
            'results': results,
            })

    with open(Path(output_path) / 'consistent_initialization.json', 'w') as f:
        json.dump(report, f, indent=4)

    text = _format_report(report)
    with open(Path(output_path) / 'consistent_initialization.txt', 'w') as f:
        f.write(text)
    print(text)

    _plot_report(report, Path(output_path) / 'consistent_initialization.png')

    return report
