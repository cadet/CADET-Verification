#%% Include packages
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.utility.convergence as convergence
from src.utility.versionInfo import print_cadet_versions

import src.twoDimChromatographyDG as twoDimChromatographyDG
import src.DG2D_radEpsB as DG2D_radEpsB
import src.DG2D_radConst as DG2D_radConst

small_test = True
n_jobs = -1
delete_h5_files = False

output_path = Path.cwd().parent / "output" / "test_cadet-core"

# cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
cadet_path = convergence.get_cadet_path()

print_cadet_versions(cadet_path)


#%% radially constant, single outlet against CASEMA

DG2D_radConst.GRM2D_linBnd_tests(
    n_jobs=n_jobs,
    small_test=small_test,
    output_path=str(output_path) + "/2Dchromatography",
    cadet_path=cadet_path,
    rerun_sims=True
)
if delete_h5_files:
    convergence.delete_h5_files(str(output_path) + "/2Dchromatography")

'''
We get ???

'''


#%% radially variable epsB in two zones against CASEMA

DG2D_radEpsB.GRM2D_linBnd_tests(
    n_jobs=n_jobs,
    small_test=small_test,
    output_path=str(output_path) + "/2Dchromatography",
    cadet_path=cadet_path,
    rerun_sims=True
)
if delete_h5_files:
    convergence.delete_h5_files(str(output_path) + "/2Dchromatography")

'''
We get second order convergence

'''


#%% tests similar to the FV ones, outlet zones against CASEMA for different linear settings

twoDimChromatographyDG.GRM2D_linBnd_tests(
    n_jobs=n_jobs,
    small_test=small_test,
    output_path=str(output_path) + "/2Dchromatography",
    cadet_path=cadet_path,
    rerun_sims=True
)
if delete_h5_files:
    convergence.delete_h5_files(str(output_path) + "/2Dchromatography")
    
'''
We get second order convergence

'''







