
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


small_test = True
n_jobs = -1
delete_h5_files = False

run_binding_tests = False
run_chromatography_tests = False
run_MCT_tests = False
run_chromatography_sensitivity_tests = False
run_chromatography_system_tests = False
run_crystallization_tests = False
run_2Dmodels_tests = True

commit_message = "testitest"
rdm_debug_mode = True
rdm_push = False
branch_name = "feature/generalized_unit"


database_path = (
    "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database"
    "/-/raw/core_tests/cadet_config/test_cadet-core/"
)
sys.path.append(str(Path(".")))
project_repo = ProjectRepo(branch=branch_name)
output_path = project_repo.output_path / "test_cadet-core"
cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master4_generalizedUnit_4cc363a\aRELEASE"
# convergence.get_cadet_path()

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

    if run_MCT_tests:
        MCT.MCT_tests(
            n_jobs=n_jobs,
            database_path=database_path + "mct/",
            small_test=small_test,
            output_path=str(output_path) + "/mct",
            cadet_path=cadet_path
        )
        if delete_h5_files:
            convergence.delete_h5_files(str(output_path) + "/mct")

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

if rdm_push:
    project_repo.push()