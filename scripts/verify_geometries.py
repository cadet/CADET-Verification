"""

This script executes the column geometry verification and validation studies for a paper publication

""" 
  
#%% Include packages
import os
import sys
from pathlib import Path
import pytest
from joblib import Parallel, delayed
import copy

from cadetrdm import ProjectRepo

from src import bench_func
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
def run_performance_tests(request):
    return request.config.getoption("--run-performance-tests")

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
    run_EOC_tests, run_performance_tests, run_validation_tests,
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

        if run_performance_tests:

            os.makedirs(output_path, exist_ok=True)

            # Define settings and benchmarks

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

            # FV cells
            ax_disc = [
                [bench_func.disc_list(8, 6 if not small_test else 3)],
                [bench_func.disc_list(32, 9 if not small_test else 3)],
                [bench_func.disc_list(8, 6 if not small_test else 3)],
                [bench_func.disc_list(32, 9 if not small_test else 3)]
                ]
            par_disc = [[bench_func.disc_list(1, 6 if not small_test else 3)], [None], [bench_func.disc_list(1, 6 if not small_test else 3)], [None]]

            addition = geometry_performance_benchmark(spatial_method=0, ax_disc=ax_disc, par_disc=par_disc, small_test=small_test)

            add_benchmark(
                cadet_configs, include_sens, ref_files, unit_IDs, which,
                ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
                cadet_config_names=cadet_config_names, addition=addition,
                disc_refinement_functions=disc_refinement_functions
                )

            # DG elements
            ax_disc = [
                [bench_func.disc_list(4, 5 if not small_test else 3)],
                [bench_func.disc_list(8, 7 if not small_test else 3)],
                [bench_func.disc_list(4, 5 if not small_test else 3)],
                [bench_func.disc_list(8, 7 if not small_test else 3)]
            ]
            par_disc = [
                [bench_func.disc_list(1, 5 if not small_test else 3)],
                [None],
                [bench_func.disc_list(1, 5 if not small_test else 3)],
                [None]
            ]

            addition = geometry_performance_benchmark(spatial_method=3, ax_disc=copy.deepcopy(ax_disc), par_disc=copy.deepcopy(par_disc), small_test=small_test)

            add_benchmark(
                cadet_configs, include_sens, ref_files, unit_IDs, which,
                ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
                cadet_config_names=cadet_config_names, addition=addition,
                disc_refinement_functions=disc_refinement_functions
                )

            addition = geometry_performance_benchmark(spatial_method=4, ax_disc=copy.deepcopy(ax_disc), par_disc=copy.deepcopy(par_disc), small_test=small_test)

            add_benchmark(
                cadet_configs, include_sens, ref_files, unit_IDs, which,
                ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
                cadet_config_names=cadet_config_names, addition=addition,
                disc_refinement_functions=disc_refinement_functions
                )

            run_convergence_analysis(
                output_path=output_path+ "/chromatography",
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
                n_jobs=n_jobs,
                rerun_sims=True,
                disc_refinement_functions = disc_refinement_functions
                # For which='bulk', exactly one of the following two must be given:
                #
                # time_point: solution time index at which spatial error norms are
                # evaluated. Must hit a time at which the concentration front is still
                # inside the column (here t = 125s); at the end of the simulation the
                # column is empty again.
                # time_point=500,
                #
                # normed_coord: normalized axial coordinate z/L in [0, 1] at which
                # temporal (outlet-like) error norms are evaluated; normed_coord=1.0
                # is equivalent to the outlet solution.
                # normed_coord=1.0,
            )
            
            if delete_h5_files:
                convergence.delete_h5_files(str(output_path) + "/chromatography")

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

