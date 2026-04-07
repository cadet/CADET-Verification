# -*- coding: utf-8 -*-
"""

This script defines binding model tests

"""

import os
# from joblib import Parallel, delayed

from src.benchmark_models import setting_GRM_ACT_2comp_benchmark1
from src.benchmark_models import setting_GRM_SplineBnd_knots_Shallow_7

#%%

def binding_tests(n_jobs, cadet_path, output_path):
    
    os.makedirs(output_path, exist_ok=True)
    
    setting_GRM_ACT_2comp_benchmark1.get_model(
        use_ion_conc=False, cadet_path=cadet_path,
        output_path=output_path, run_simulation=True, plot_result=True
        )
    
    setting_GRM_ACT_2comp_benchmark1.get_model(
        use_ion_conc=True, cadet_path=cadet_path,
        output_path=output_path, run_simulation=True, plot_result=True
        )

    setting_GRM_SplineBnd_knots_Shallow_7.get_model(
        cadet_path, output_path, run_simulation=True, plot_result=True
        )
