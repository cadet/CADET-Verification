# This file is similar to the src/verify.py file, but meant for debugging, ie without pytest fixtures


#%% Include packages
import sys
from pathlib import Path

import src.utility.convergence as convergence

import src.chromatography as chromatography


small_test = False
n_jobs = -1
delete_h5_files = False

sys.path.append(str(Path(".")))
output_path = Path.cwd() / "output" / "test_cadet-core"
cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\CADET_eigen5_v6.0.0-alpha.1\aRELEASE"
# convergence.get_cadet_path()

chromatography.chromatography_tests(
    n_jobs=n_jobs,
    small_test=small_test,
    sensitivities=True,
    output_path=str(output_path) + "/chromatography",
    cadet_path=cadet_path
)
if delete_h5_files:
    convergence.delete_h5_files(str(output_path) + "/chromatography")

