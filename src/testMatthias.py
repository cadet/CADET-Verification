# -*- coding: utf-8 -*-
"""

This script executes all the CADET-Verification tests for CADET-Core.
Modify the input in the 'User Input' section if needed.
To test if the script works, specify rdm_debug_mode and small_test as true.

Only specify rdm_debug_mode as False if you are sure that this run shall be
saved to the output repository!

""" 
  
#%% Include packages
import sys
from pathlib import Path
import pytest

from cadetrdm import ProjectRepo

import src.utility.convergence as convergence

import eoc_studies.eoc_Col1D_langLRM_2comp_benchmark as langmuir

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
    run_binding_tests, run_chromatography_tests, run_chromatography_sensitivity_tests, run_chromatography_system_tests,
    run_crystallization_tests, run_MCT_tests, run_2Dmodels_tests
):

    sys.path.append(str(Path(".")))
    project_repo = ProjectRepo(branch=branch_name)
    output_path = project_repo.output_path / "test_cadet-core"
    cadet_path = convergence.get_cadet_path()

    with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):

        langmuir.mainFunc(
            n_jobs=n_jobs,
            small_test=small_test,
            output_path=str(output_path) + "/chromatography",
            cadet_path=cadet_path
        )
        if delete_h5_files:
            convergence.delete_h5_files(str(output_path) + "/chromatography")
    
        if rdm_push:
            project_repo.push()