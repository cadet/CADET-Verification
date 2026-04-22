# -*- coding: utf-8 -*-
"""

This scipt is similar to the src/verify.py file, but meant for debugging
without pytest fixtures and cadet-rdm

"""

#%% Include packages
import sys
from pathlib import Path

import src.utility.convergence as convergence
from src.utility.versionInfo import print_cadet_versions

import src.transport_convDisp as transport_convDisp
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

run_transport_tests = True
run_binding_tests = False
run_chromatography_tests = False
run_MCT_tests = False
run_chromatography_sensitivity_tests = False
run_chromatography_system_tests = False
run_crystallization_tests = False
run_2Dmodels_tests = False

sys.path.append(str(Path(".")))
output_path = Path.cwd() / "output" / "test_cadet-core"
cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\parDiffOpFV\aRELEASE"
# cadet_path = convergence.get_cadet_path()

print_cadet_versions(cadet_path)

if run_transport_tests:
    transport_convDisp.transport_tests(
        n_jobs=n_jobs,
        small_test=small_test,
        output_path=str(output_path) + "/transport",
        cadet_path=cadet_path
    )
    if delete_h5_files:
        convergence.delete_h5_files(str(output_path) + "/transport")
        
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
        reference_data_path=str(Path.cwd()) + '/data/CASEMA_reference'
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
        reference_data_path=str(Path.cwd() / 'data'),
        use_CASEMA_reference=True,
        rerun_sims=True
    )
    if delete_h5_files:
        convergence.delete_h5_files(str(output_path) + "/2Dchromatography")
