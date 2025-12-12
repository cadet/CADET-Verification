
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

import src.twoDimChromatography as twoDimChromatography


small_test = True
n_jobs = -1
delete_h5_files = False
run_2Dmodels_tests = True

commit_message = "testitest"
rdm_debug_mode = True
rdm_push = False
branch_name = "feature/generalized_unit"


sys.path.append(str(Path(".")))
project_repo = ProjectRepo(branch=branch_name)
output_path = project_repo.output_path / "test_cadet-core"
# cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master4_crysPartII_d0888cb\aRELEASE"
cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master5_generalizedUnit_f1a1972\aRELEASE"
# convergence.get_cadet_path()

with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):

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