"""

This script executes the column geometry verification and validation studies for a paper publication

""" 

#%% Include packages
import os
import sys
from pathlib import Path
import pytest
from joblib import Parallel, delayed

from cadetrdm import ProjectRepo

import src.utility.convergence as convergence
from src.utility.versionInfo import print_cadet_versions

from src.column_geometries import geometry_tests
from src.validation.Gritti2019_frustumGeometry.Gritti2019_fig6 import main as Gritti2019_fig6
from src.validation.Gritti2019_frustumGeometry.Gritti2019_fig7 import main as Gritti2019_fig7
from src.validation.Gritti2019_frustumGeometry.Gritti2019_fig8 import main as Gritti2019_fig8
from src.validation.Gu2015_radialFlowGeometry.Gu2015_fig14_3 import main as Gu2015_fig14_3
from src.validation.Gu2015_radialFlowGeometry.Gu2015_fig14_5 import main as Gu2015_fig14_5
from src.validation.Gu2015_radialFlowGeometry.Gu2015_fig14_6 import main as Gu2015_fig14_6
from src.bench_configs import SMA_performance_benchmark
from src.bench_configs import langmuir_performance_benchmark
from src.bench_configs import add_benchmark
from src.bench_configs import GEOMETRY_NAMES, GEOMETRY_REFERENCES, geometry_reference
from src.bench_func import create_object_from_config
from src.bench_func import run_simulation_in_verification
from src.bench_func import run_convergence_analysis

@pytest.fixture
def small_test(request):
    return request.config.getoption("--small-test")

@pytest.fixture
def n_jobs(request):
    return request.config.getoption("--n-jobs")

@pytest.fixture
def delete_h5_files(request):
    return request.config.getoption("--delete-h5-files")

@pytest.fixture
def run_EOC_tests(request):
    return request.config.getoption("--run-eoc-tests")

@pytest.fixture
def run_validation_tests(request):
    return request.config.getoption("--run-validation-tests")

@pytest.fixture
def n_reruns(request):
    return request.config.getoption("--n-reruns")


@pytest.fixture
def run_performance_sma_tests(request):
    return request.config.getoption("--run-performance-sma-tests")


@pytest.fixture
def run_performance_langmuir_tests(request):
    return request.config.getoption("--run-performance-langmuir-tests")

@pytest.fixture
def column_geometries(request):
    return request.config.getoption("--column-geometries")

@pytest.fixture
def sma_particle_resolutions(request):
    return request.config.getoption("--sma-particle-resolutions")

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


@pytest.fixture
def reference_setting(request):
    return request.config.getoption("--reference-setting")

@pytest.fixture
def reference_geometry(request):
    return request.config.getoption("--reference-geometry")

# The geometries are selected by the prefix their output files carry, which is
# shorter on the command line than the CADET geometry name.
REFERENCE_GEOMETRIES = {
    prefix: geometry for geometry, prefix in GEOMETRY_NAMES.items()
    }


def selected_reference(option, value, choices):
    """The one choice a reference run is given, as named on the command line."""
    if value is None:
        raise ValueError(
            option + ' is required to compute a reference and is one of '
            + ', '.join(choices) + '.'
            )

    if value.strip() not in choices:
        raise ValueError(
            option + ' is one of ' + ', '.join(choices) + ', got ' + value + '.'
            )

    return value.strip()


def compute_geometry_reference(setting, geometry, output_path, cadet_path):
    """Simulate the reference solution of one performance benchmark.

    Every setting of a performance benchmark is compared against a single
    reference, shared by all of its spatial methods, which is what makes the
    methods comparable to one another. Without one,
    bench_func.run_convergence_analysis falls back to self-convergence, where
    each method takes its own finest level as its reference, and the last levels
    of every sweep then say more about that level than about the method.

    A reference is one simulation at a resolution beyond the sweep it serves and
    at the time integration tolerance of Breuer et al. (2023),
    doi:10.1016/j.compchemeng.2023.108340, and costs hours rather than minutes,
    which is why one run computes one of them. It is written next to the
    simulations of the sweep, which is where the sweep looks it up by name.
    """

    reference = geometry_reference(
        setting=setting, column_geometry=REFERENCE_GEOMETRIES[geometry]
        )

    simulation = create_object_from_config(
        config_data=reference['cadet_config_json'],
        setting_name=reference['setting_name'],
        unit_id=reference['unit_id'],
        ax_method=reference['ax_method'],
        ax_cells=reference['ax_cells'],
        par_method=reference['par_method'],
        par_cells=reference['par_cells'],
        output_path=output_path,
        idas_abstol=reference['idas_abstol'],
        USE_COLLOCATION_DG=reference['use_collocation_dg'],
        include_sens=False,
        )

    print(
        f"Computing the {setting} reference on the {geometry} geometry: "
        f"{simulation.filename}"
        )

    run_simulation_in_verification([simulation], cadet_path)

    print(
        f"Done in {convergence.get_compute_time(simulation.filename):.1f} "
        f"seconds, {convergence.get_idas_timesteps(simulation.filename):.0f} "
        "time steps."
        )
    print(
        f"Pass {Path(simulation.filename).name} as the ref_file of the "
        "benchmark configuration of this setting to compare its spatial methods "
        "against it."
        )


def test_selected_model_groups(
    commit_message, rdm_debug_mode, branch_name, rdm_push, small_test, n_jobs, delete_h5_files,
    run_EOC_tests, run_performance_sma_tests, run_performance_langmuir_tests,
    run_validation_tests, n_reruns, column_geometries, sma_particle_resolutions,
    reference_setting, reference_geometry,
):

    sys.path.append(str(Path(".")))
    project_repo = ProjectRepo(branch=branch_name)
    output_path = project_repo.output_path / "test_cadet-core"
    cadet_path = convergence.get_cadet_path()

    with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):

        print_cadet_versions(cadet_path)

        delete_h5_files = False
        n_jobs = -1
        small_test = False
        n_reruns = 0

        # Computing no reference is the default, since one of them is a job of
        # its own: naming a setting and a geometry asks for that one reference
        # and nothing else.
        if reference_setting is not None or reference_geometry is not None:

            chromatography_path = str(output_path) + "/chromatography"
            os.makedirs(chromatography_path, exist_ok=True)

            compute_geometry_reference(
                setting=selected_reference(
                    '--reference-setting', reference_setting,
                    sorted(GEOMETRY_REFERENCES)
                    ),
                geometry=selected_reference(
                    '--reference-geometry', reference_geometry,
                    sorted(REFERENCE_GEOMETRIES)
                    ),
                output_path=chromatography_path,
                cadet_path=cadet_path,
                )

        if run_EOC_tests:

            geometry_tests(
                n_jobs=n_jobs,
                small_test=small_test,
                output_path=output_path,
                cadet_path=cadet_path,
                regenerate_references=False,
                delete_h5_files=delete_h5_files,
                )
            
            if delete_h5_files:
                convergence.delete_h5_files(str(output_path) + "/transport")

        performance_benchmarks = []
        if run_performance_sma_tests:
            performance_benchmarks.append([
                SMA_performance_benchmark(
                    small_test=small_test,
                    column_geometry=geometry,
                    particle_type=particle_type
                )
                for geometry in column_geometries
                for particle_type in sma_particle_resolutions
            ])
        if run_performance_langmuir_tests:
            performance_benchmarks.append([
                langmuir_performance_benchmark(
                    small_test=small_test, column_geometry=geometry)
                for geometry in column_geometries
                ])

        for performance_benchmark in performance_benchmarks:

            chromatography_path = str(output_path) + "/chromatography"
            os.makedirs(chromatography_path, exist_ok=True)

            cadet_configs = []
            cadet_config_names = []
            include_sens = []
            ref_files = []
            unit_IDs = []
            which = []
            idas_abstol = []
            ax_methods = []
            ax_discs = []
            par_methods = []
            par_discs = []
            disc_refinement_functions = []

            for addition in performance_benchmark:

                add_benchmark(
                    cadet_configs, include_sens, ref_files, unit_IDs, which,
                    ax_methods, ax_discs, par_methods, par_discs,
                    idas_abstol=idas_abstol,
                    cadet_config_names=cadet_config_names, addition=addition,
                    disc_refinement_functions=disc_refinement_functions
                    )

            run_convergence_analysis(
                output_path=chromatography_path,
                cadet_path=cadet_path,
                cadet_configs=cadet_configs,
                cadet_config_names=cadet_config_names,
                include_sens=include_sens,
                ref_files=ref_files,
                unit_IDs=unit_IDs,
                which=which,
                ax_methods=ax_methods,
                ax_discs=ax_discs,
                par_methods=par_methods,
                par_discs=par_discs,
                idas_abstol=idas_abstol,
                USE_COLLOCATION_DG=0,
                # Serial on purpose, whatever --n-jobs says: this benchmark
                # measures compute times, and simulations that share cores do
                # not have comparable ones.
                n_jobs=n_jobs,
                rerun_sims=True,
                disc_refinement_functions=disc_refinement_functions
                )

            # A single compute time carries whatever else the machine was doing,
            # so a benchmark repeats every simulation and keeps the fastest, and
            # then rebuilds the tables from the files it just updated.
            if n_reruns:
                convergence.mult_sim_rerun(
                    chromatography_path, str(cadet_path), n_reruns)
                run_convergence_analysis(
                    output_path=chromatography_path,
                    cadet_path=cadet_path,
                    cadet_configs=cadet_configs,
                    cadet_config_names=cadet_config_names,
                    include_sens=include_sens,
                    ref_files=ref_files,
                    unit_IDs=unit_IDs,
                    which=which,
                    ax_methods=ax_methods,
                    ax_discs=ax_discs,
                    par_methods=par_methods,
                    par_discs=par_discs,
                    idas_abstol=idas_abstol,
                    USE_COLLOCATION_DG=0,
                    n_jobs=1,
                    rerun_sims=False,
                    disc_refinement_functions=disc_refinement_functions
                    )

        if run_validation_tests:

            validation_path = str(output_path) + "/validation"

            # The validation studies are independent of each other and write
            # different files, so they run next to each other.
            Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(study)(cadet_path=cadet_path, output_path=validation_path)
                for study in [
                    Gritti2019_fig6,
                    Gritti2019_fig7,
                    Gritti2019_fig8,
                    Gu2015_fig14_3,
                    Gu2015_fig14_5,
                    Gu2015_fig14_6,
                    ]
                )
            
            if delete_h5_files:
                convergence.delete_h5_files(str(output_path) + "/validation")           

