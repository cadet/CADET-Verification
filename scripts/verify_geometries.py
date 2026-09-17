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


def test_selected_model_groups(
    commit_message, rdm_debug_mode, branch_name, rdm_push, small_test, n_jobs, delete_h5_files,
    run_EOC_tests, run_performance_sma_tests, run_performance_langmuir_tests,
    run_validation_tests, n_reruns, column_geometries, sma_particle_resolutions,
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

