# -*- coding: utf-8 -*-
"""

This script executes all the CADET-Verification tests for CADET-Core.
Modify the input in the 'User Input' section if needed.
To test if the script works, specify rdm_debug_mode and small_test as true.

Only specify rdm_debug_mode as False if you are sure that this run shall be
saved to the output repository!

""" 
  
#%% Include packages
import os
import sys
from pathlib import Path
import re
from joblib import Parallel, delayed
import numpy as np
import pytest

from cadet import Cadet
from cadetrdm import ProjectRepo

import src.utility.convergence as convergence
import src.bench_func as bench_func
import src.bench_configs as bench_configs

import src.chromatography as chromatography
import src.bindings as bindings
import src.crystallization as crystallization
import src.MCT as MCT
import src.chrom_systems as chrom_systems
import src.twoDimChromatography as twoDimChromatography
import src.chromatography_sensitivities as chromatography_sensitivities

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
def run_binding_tests(request):
    return request.config.getoption("--run-binding-tests")

@pytest.fixture
def run_chromatography_tests(request):
    return request.config.getoption("--run-chromatography-tests")

@pytest.fixture
def run_chromatography_sensitivity_tests(request):
    return request.config.getoption("--run-chromatography-sensitivity-tests")

@pytest.fixture
def run_chromatography_system_tests(request):
    return request.config.getoption("--run-chromatography-system-tests")

@pytest.fixture
def run_crystallization_tests(request):
    return request.config.getoption("--run-crystallization-tests")

@pytest.fixture
def run_MCT_tests(request):
    return request.config.getoption("--run-mct-tests")

@pytest.fixture
def run_2Dmodels_tests(request):
    return request.config.getoption("--run-2dmodels-tests")

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

        if run_chromatography_tests:
            chromatography.chromatography_tests(
                n_jobs=n_jobs,
                small_test=small_test,
                sensitivities=True,
                output_path=str(output_path) + "/chromatography",
                cadet_path=cadet_path
            )
            if delete_h5_files:
                convergence.delete_h5_files(str(output_path) + "/chromatography")

        if run_binding_tests:
            bindings.binding_tests(
                n_jobs=n_jobs, cadet_path=cadet_path,
                output_path=str(output_path) + "/chromatography/binding"
            )
            if delete_h5_files:
                convergence.delete_h5_files(str(output_path) + "/chromatography/binding")
            
        if run_chromatography_sensitivity_tests:
            chromatography_sensitivities.chromatography_sensitivity_tests(
                    n_jobs=n_jobs,
                    small_test=small_test,
                    output_path=str(output_path) + "/chromatography/sensitivity",
                    cadet_path=cadet_path
            )
            if delete_h5_files:
                    convergence.delete_h5_files(str(output_path) + "/chromatography/sensitivity")

        if run_chromatography_system_tests:
            chrom_systems.chromatography_systems_tests(
                n_jobs=n_jobs,
                small_test=small_test,
                output_path=str(output_path) + "/chromatography/systems",
                cadet_path=cadet_path,
                analytical_reference=True,
                reference_data_path=str(project_repo.output_path.parent) + '/data/CASEMA_reference'
            )
            if delete_h5_files:
                convergence.delete_h5_files(str(output_path) + "/chromatography/systems")

        if run_crystallization_tests:
            crystallization.crystallization_tests(
                n_jobs=n_jobs,
                small_test=small_test,
                output_path=str(output_path) + "/crystallization",
                cadet_path=cadet_path
            )
            if delete_h5_files:
                convergence.delete_h5_files(str(output_path) + "/crystallization")

        if run_2Dmodels_tests:
            twoDimChromatography.GRM2D_linBnd_tests(
                n_jobs=n_jobs,
                small_test=small_test,
                output_path=str(output_path) + "/2Dchromatography",
                cadet_path=cadet_path,
                reference_data_path=str(project_repo.output_path.parent / 'data'),
                use_CASEMA_reference=True,
                rerun_sims=True
            )
            if delete_h5_files:
                convergence.delete_h5_files(str(output_path) + "/2Dchromatography")

        if run_MCT_tests:
            MCT.MCT_tests(
                n_jobs=n_jobs,
                small_test=small_test,
                output_path=str(output_path) + "/mct",
                cadet_path=cadet_path
            )
            if delete_h5_files:
                convergence.delete_h5_files(str(output_path) + "/mct")

    if rdm_push:
        project_repo.push()