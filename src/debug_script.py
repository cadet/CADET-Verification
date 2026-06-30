# debug_chromatography.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path

ref_path = Path(__file__).resolve().parent.parent / "data" / "CADET-Core_reference" / "chromatography" / "ref_LRM_langmuir.h5"
ref_path_list = [str(ref_path)]
for i in range(8):
    ref_path_list.insert(0, None)

from chromatography import chromatography_tests
from cadet import Cadet


chromatography_tests(
    n_jobs=1,
    small_test=True,
    sensitivities=False,
    output_path="output/debug",
    cadet_path= None,
    ref_filepaths= ref_path_list)