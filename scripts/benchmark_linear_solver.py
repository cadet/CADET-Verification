# -*- coding: utf-8 -*-
"""

Performance benchmark of the linear solver of the 1D column models.

Measures what each linear solver costs to reach a given accuracy, over all
chromatography benchmarks. Two studies are defined, selected by
--benchmark-method:

  dg  The generalized DG unit operations, comparing the Eigen sparse LU
      against the block elimination of the particle states.

  fv  The FV discretization, comparing the dedicated arrow head FV unit
      operations against the generalized ones, the latter both with the Eigen
      sparse LU and with the block elimination.

Every series of a figure is measured in the same job, since compute time is
not comparable across machines. One job computes one setting, selected by
--benchmark-settings.

"""

# %% Include packages
import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

from cadetrdm import ProjectRepo

import src.bench_configs as bench_configs
import src.bench_func as bench_func
import src.utility.convergence as convergence
import src.utility.plotting as plotting
from src.utility.versionInfo import print_cadet_versions


# %% Reference data paths
reference_data_path = str(
    Path(__file__).resolve().parent.parent / 'data'
)


# %% Compared series
#
# The new solver is called BLOCK_JACOBIAN in ColumnModel1D and BANDED_JACOBIAN in the
# LRM without pores, and which of the two a setting lands in depends on its particle
# type, so both keys are written and each unit operation reads the one it knows.
#
# FV_ARROW_HEAD_OPTIMIZATION selects the unit operation itself: left on, a FV setting
# is computed by the dedicated arrow head FV unit operations, switched off it is
# computed by the generalized ones (ColumnModel1D, or the sparse LRM without pores for
# a particle without film transfer).

DG_VARIANTS = {
    'eigenLU': {},
    'blockElim': {'BLOCK_JACOBIAN': 1, 'BANDED_JACOBIAN': 1},
}

FV_VARIANTS = {
    'arrowHeadFV': {'FV_ARROW_HEAD_OPTIMIZATION': 1},
    'eigenLU': {'FV_ARROW_HEAD_OPTIMIZATION': 0},
    'blockElim': {'FV_ARROW_HEAD_OPTIMIZATION': 0,
                  'BLOCK_JACOBIAN': 1, 'BANDED_JACOBIAN': 1},
}


# %% Figure matrix

AXES = [
    ("Sim. time", "compute", "Compute time in seconds"),
    ("DoF", "dof", "Degrees of freedom"),
    ("idas n_timesteps", "timesteps", "IDAS time steps"),
]

NORMS = [
    ("Max. error", "Linf", r"$L^\infty$ error in mol $/m^3$"),
    ("$L^2$ error", "L2", r"$L^2$ error in mol $/m^3$"),
]


# %% Benchmark definition

def benchmark_groups(method, small_test=False, ref_filepath=reference_data_path):
    """Benchmark configurations of one spatial method, by column geometry."""

    if method == 'dg':
        return [
            bench_configs.axial_flow_benchmark_dg(
                small_test=small_test, ref_filepath=ref_filepath),
            bench_configs.radial_flow_benchmark_dg(
                small_test=small_test, ref_filepath=ref_filepath),
        ]

    if method == 'fv':
        return [
            bench_configs.axial_flow_benchmark_fv(
                small_test=small_test, ref_filepath=ref_filepath),
            bench_configs.radial_flow_benchmark_fv(
                small_test=small_test, ref_filepath=ref_filepath),
        ]

    raise ValueError("Unknown benchmark method " + str(method))


def setting_names(method):
    """Names of every setting of one spatial method, in benchmark order.

    Used to enumerate the jobs of the workflow, so it must not need a CADET
    installation and is called without reference solutions.
    """

    names = []

    for config in benchmark_groups(method, ref_filepath=None):
        names.extend(config['cadet_config_names'])

    return names


def select_setting(method, name, small_test):
    """One setting of one spatial method, as a benchmark configuration."""

    for config in benchmark_groups(method, small_test=small_test):

        if name not in config['cadet_config_names']:
            continue

        index = config['cadet_config_names'].index(name)

        return {key: [copy.deepcopy(value[index])] for key, value in config.items()}

    raise ValueError(
        "Unknown setting " + str(name) + " for method " + str(method)
        + ", available: " + ", ".join(setting_names(method))
    )


def apply_variant(config, variant, options):
    """Writes the numerical options of one series into a benchmark configuration.

    The options do not reach the method group name of the convergence json, so
    each series also gets its own setting name. That separates both the h5 file
    names and the convergence json, which are overlaid again when plotting.
    """

    config['cadet_config_names'] = [
        name + '_' + variant for name in config['cadet_config_names']
    ]

    for model, unit_id in zip(config['cadet_config_jsons'], config['unit_IDs']):

        unit = model['input']['model']['unit_' + unit_id]

        if 'discretization' not in unit:
            raise ValueError(
                "Unit " + str(unit_id) + " has no discretization group to configure"
            )

        for key, value in options.items():
            unit['discretization'][key] = value

    return config


# %% Run

def run_setting(method, name, output_path, cadet_path, n_reruns, small_test):
    """Measures every series of one setting and returns its convergence jsons."""

    variants = DG_VARIANTS if method == 'dg' else FV_VARIANTS

    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Each series keeps its own configuration objects, because
    # run_convergence_analysis resolves a missing reference into ref_files and drops
    # the corresponding level from ax_discs in place. Handing the same objects back
    # for the rebuild is what makes the second pass agree with the first; rebuilding
    # the configuration in between would drop a further level.
    configs = {
        variant: apply_variant(
            select_setting(method, name, small_test), variant, options)
        for variant, options in variants.items()
    }

    json_paths = {}

    for variant, config in configs.items():

        print("\n=== " + name + ", " + variant + " ===\n")

        result_names = bench_func.run_convergence_analysis(
            output_path=output_path,
            cadet_path=cadet_path,
            cadet_configs=config['cadet_config_jsons'],
            cadet_config_names=config['cadet_config_names'],
            include_sens=config['include_sens'],
            ref_files=config['ref_files'],
            unit_IDs=config['unit_IDs'],
            which=config['which'],
            ax_methods=config['ax_methods'],
            ax_discs=config['ax_discs'],
            par_methods=config['par_methods'],
            par_discs=config['par_discs'],
            idas_abstol=config['idas_abstol'],
            n_jobs=1,
            rerun_sims=True,
            disc_refinement_functions=config['disc_refinement_functions'],
        )

        json_paths[variant] = Path(output_path) / result_names[0]

    if n_reruns > 1:

        print("\n=== " + name + ", keeping the fastest of "
              + str(n_reruns) + " runs ===\n")

        convergence.mult_sim_rerun(str(output_path), cadet_path, n_reruns)

        for variant, config in configs.items():

            bench_func.run_convergence_analysis(
                output_path=output_path,
                cadet_path=cadet_path,
                cadet_configs=config['cadet_config_jsons'],
                cadet_config_names=config['cadet_config_names'],
                include_sens=config['include_sens'],
                ref_files=config['ref_files'],
                unit_IDs=config['unit_IDs'],
                which=config['which'],
                ax_methods=config['ax_methods'],
                ax_discs=config['ax_discs'],
                par_methods=config['par_methods'],
                par_discs=config['par_discs'],
                idas_abstol=config['idas_abstol'],
                n_jobs=1,
                rerun_sims=False,
                disc_refinement_functions=config['disc_refinement_functions'],
            )

    write_metadata(name, configs, json_paths, output_path, cadet_path, n_reruns)

    return json_paths


def write_metadata(name, configs, json_paths, output_path, cadet_path, n_reruns):
    """Records what a figure of this setting may not be read without."""

    from cadet import Cadet

    cadet = Cadet()
    cadet.install_path = cadet_path

    metadata = {
        'setting': name,
        'n_reruns': n_reruns,
        'n_jobs': 1,
        'cadet_path': str(cadet_path),
        'cadet_version': cadet.version,
        'cadet_commit_hash': cadet.commit_hash,
        'series': {
            variant: {
                'convergence_file': Path(json_paths[variant]).name,
                'idas_abstol': config['idas_abstol'][0],
                'reference': str(config['ref_files'][0]),
                'ax_methods': config['ax_methods'][0],
                'ax_discs': config['ax_discs'][0],
                'par_methods': config['par_methods'][0],
                'par_discs': config['par_discs'][0],
            }
            for variant, config in configs.items()
        },
    }

    with open(Path(output_path) / ('benchmark_' + name + '.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2, default=str)


# %% Plot

def plot_setting(name, json_paths, output_path, section='outlet'):
    """Draws the figure matrix of one setting, one line per compared series."""

    variants = {}

    for variant, path in json_paths.items():

        if not Path(path).exists():
            print("Missing convergence file " + str(path) + ", skipping " + variant)
            continue

        with open(path) as handle:
            variants[variant] = json.load(handle)['convergence']

    written = []

    for x_key, x_tag, xlabel in AXES:

        for y_key, y_tag, ylabel in NORMS:

            series = {}

            for variant, methods in variants.items():

                for method_name, method in methods.items():

                    if not isinstance(method, dict) or section not in method:
                        continue

                    data = method[section]
                    if x_key not in data or y_key not in data:
                        continue

                    label = variant + ', ' + plotting.method_label(method_name)
                    series[label] = method

            if not series:
                continue

            fig, ax = plotting.plot_metric(
                series, x_key, y_key, section=section,
                xscale='log', yscale='log',
                xlabel=xlabel, ylabel=ylabel, title=name,
            )

            figure_path = Path(output_path) / (
                name + '_performance_' + x_tag + '_' + y_tag + '.png')
            fig.savefig(figure_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            written.append(figure_path)

    return written


# %% Fixtures

@pytest.fixture
def benchmark_method(request):
    return request.config.getoption("--benchmark-method")


@pytest.fixture
def benchmark_settings(request):
    return request.config.getoption("--benchmark-settings")


@pytest.fixture
def n_reruns(request):
    return request.config.getoption("--n-reruns")


@pytest.fixture
def small_test(request):
    return request.config.getoption("--small-test")


@pytest.fixture
def commit_message(request):
    return request.config.getoption("--commit-message")


@pytest.fixture
def rdm_debug_mode(request):
    return request.config.getoption("--rdm-debug-mode")


@pytest.fixture
def rdm_push(request):
    return request.config.getoption("--rdm-push")


@pytest.fixture
def branch_name(request):
    return request.config.getoption("--branch-name")


# %% Run with CADET-RDM

def test_linear_solver_benchmark(
    benchmark_method, benchmark_settings, n_reruns, small_test,
    commit_message, rdm_debug_mode, branch_name, rdm_push
):

    sys.path.append(str(Path(".")))

    if benchmark_method not in ('dg', 'fv'):
        raise ValueError(
            "--benchmark-method must be dg or fv, got " + str(benchmark_method))

    available = setting_names(benchmark_method)

    if benchmark_settings in (None, '', 'all'):
        selected = available
    else:
        selected = [name.strip() for name in benchmark_settings.split(',')
                    if name.strip()]

    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(
            "Unknown settings " + ", ".join(unknown)
            + ", available: " + ", ".join(available))

    project_repo = ProjectRepo(branch=branch_name)
    base_path = project_repo.output_path / "benchmark_linear_solver" / benchmark_method
    cadet_path = convergence.get_cadet_path()

    with project_repo.track_results(results_commit_message=commit_message,
                                    debug=rdm_debug_mode):

        print_cadet_versions(cadet_path)

        for name in selected:

            output_path = base_path / name

            json_paths = run_setting(
                method=benchmark_method,
                name=name,
                output_path=output_path,
                cadet_path=cadet_path,
                n_reruns=n_reruns,
                small_test=small_test,
            )

            plot_setting(name, json_paths, output_path)

    if rdm_push:
        project_repo.push()
