# -*- coding: utf-8 -*-
"""

This script defines binding model tests

"""

import os
# from joblib import Parallel, delayed

from src.benchmark_models import setting_GRM_ACT_2comp_benchmark1
from src.benchmark_models import setting_GRM_SplineBnd_knots_Shallow_7
from src.binding_GPR import test_GPR_binding
from src.binding_train_gpr_langmuir import binding_train_GPR_langmuir2Comp, binding_train_GPR_langmuir1Comp

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
    
    test_GPR_binding(output_path, cadet_path)
    
    binding_train_GPR_langmuir2Comp(cadet_path, output_path)
    
    binding_train_GPR_langmuir1Comp(cadet_path, output_path)
