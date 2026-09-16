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
from src.bench_configs import geometry_performance_benchmark
from src.bench_configs import add_benchmark
from src.bench_func import run_convergence_analysis

# Reference solutions of the performance benchmarks live in the repository, so
# that a run in continuous integration finds them after a plain checkout.
reference_data_path = str(Path(__file__).resolve().parent.parent / "data")


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
    run_validation_tests, n_reruns,
):

    sys.path.append(str(Path(".")))
    project_repo = ProjectRepo(branch=branch_name)
    output_path = project_repo.output_path / "test_cadet-core"
    cadet_path = convergence.get_cadet_path()

    with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):

        print_cadet_versions(cadet_path)

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

        # The two performance benchmarks of Breuer et al. (2023) on the radial
        # flow and the conical column geometry, the four-component GRM with
        # kinetic SMA binding of Fig. 5 and the two-component LRM with
        # rapid-equilibrium Langmuir binding of Figs. 7 and 8 in its less
        # disperse variant. They are selected separately, because the Langmuir
        # one is by far the more expensive of the two and because two benchmarks
        # sharing a machine do not have comparable compute times.
        #
        # small_test runs two refinement levels fewer per series than the
        # publication; small_test=False runs its exact steps, see
        # bench_configs.geometry_performance_benchmark.
        performance_cases = []
        if run_performance_sma_tests:
            performance_cases.append('SMA')
        if run_performance_langmuir_tests:
            performance_cases.append('langmuir')

        for performance_case in performance_cases:

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

            # The WENO finite volume scheme and DG of degrees three and four,
            # all measured against the same stored reference per geometry.
            for spatial_method in [0, 3, 4]:

                addition = geometry_performance_benchmark(
                    case=performance_case, spatial_method=spatial_method,
                    small_test=small_test, ref_filepath=reference_data_path,
                    cadet_path=cadet_path, output_path=chromatography_path
                    )

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
                n_jobs=1,
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

