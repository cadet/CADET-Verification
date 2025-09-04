# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:57:05 2025

@author: jmbr
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import src.benchmark_models.settings_crystallization as settings_crystallization
import src.crystallization_partII as cryPartII
import src.crystallization_partI as cryPartI
import src.crystallization as cry
import src.utility.convergence as convergence


#%%

cadet_path = r"C:\Users\jmbr\Desktop\CADET_compiled\master2_preGeneralUnit_4483702\aRELEASE\bin\cadet-cli.exe"
small_test = True
output_path = r"C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core"

#%%

# # cryPartI.CSTR_PBM_growth_EOC_test(small_test, output_path, cadet_path)
# cry.crystallization_tests(
#         n_jobs=1, database_path=None,
#         small_test=small_test, output_path=output_path, cadet_path=cadet_path,
#         run_primary_dynamics_tests = True, # part I
#         run_secondary_dynamics_tests = False, # part II without (partly) redundant tests
#         run_full_secondary_dynamics_tests = False # full part II tests
#         )

#%% settings to consider

#### DONE
# run_CSTR_aggregation_test = 1, # does not need a precomputed reference
# run_CSTR_fragmentation_test = 1, # does not need a precomputed reference
# run_CSTR_aggregation_fragmentation_test = 0, # does not need a precomputed reference


#### TODO

# run_CSTR_PBM_aggregation_fragmentation_test = 1,
# run_DPFR_constAgg_test = 0, # not included in test pipeline per default, due to redundancy
# run_DPFR_constFrag_test = 1,
# run_DPFR_NGGR_aggregation_test = 1,
# run_DPFR_aggregation_fragmentation_test = 0 # not included in test pipeline per default, due to redundancy











import numpy as np
from joblib import Parallel, delayed

# Wrap each simulation in a function
def run_CSTR_PBM():
    x_c, x_max = 1e-6, 1000e-6  # m
    n_x = 1536
    cycle_time = 300            # s
    time_res = 100
    t = np.linspace(0, cycle_time, time_res)

    model, x_grid, x_ct = settings_crystallization.CSTR_PBM_aggregation_fragmentation(
        n_x, x_c, x_max, 1, t, cadet_path, output_path
    )
    model.save()
    return model.run_simulation()


def run_Agg_DPFR():
    n_x = 384
    n_col = 192
    x_c, x_max = 1e-6, 1000e-6
    x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

    cycle_time = 300
    t = np.linspace(0, cycle_time, 200)

    model, x_grid, x_ct = settings_crystallization.Agg_DPFR(
        n_x, n_col, x_c, x_max, 1, t, cadet_path, output_path
    )
    model.save()
    return model.run_simulation()


def run_Frag_DPFR():
    n_x = 384
    n_col = 192
    x_c, x_max = 1e-6, 1000e-6
    x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

    cycle_time = 300
    t = np.linspace(0, cycle_time, 200)

    model, x_grid, x_ct = settings_crystallization.Frag_DPFR(
        n_x, n_col, x_c, x_max, 1, t, cadet_path, output_path
    )
    model.save()
    return model.run_simulation()


def run_DPFR_NGGR():
    n_x = 384
    n_col = 192
    x_c, x_max = 1e-6, 1000e-6
    x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

    cycle_time = 200
    t = np.linspace(0, cycle_time, 200 + 1)

    model, x_grid, x_ct = settings_crystallization.DPFR_PBM_NGGR_aggregation(
        n_x, n_col, x_c, x_max, 1, 1, t, cadet_path, output_path
    )
    model.save()
    return model.run_simulation()


def run_Agg_Frag_DPFR():
    n_x = 384
    n_col = 192
    x_c, x_max = 1e-6, 1000e-6
    x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

    cycle_time = 300
    t = np.linspace(0, cycle_time, 200)

    model, x_grid, x_ct = settings_crystallization.Agg_Frag_DPFR(
        n_x, n_col, x_c, x_max, 1, t, cadet_path, output_path
    )
    model.save()
    return model.run_simulation()


results = Parallel(n_jobs=-1)(  # use all available cores
    delayed(func)() for func in [
        run_CSTR_PBM,
        run_Agg_DPFR,
        run_Frag_DPFR,
        run_DPFR_NGGR,
        run_Agg_Frag_DPFR,
    ]
)














#%%

# small_test=True

# cryPartII.PBM_aggregation_fragmentation_EOC_test(cadet_path, small_test, output_path)


# #%% run_CSTR_PBM_aggregation_fragmentation_test

# import numpy as np

# # define params
# x_c, x_max = 1e-6, 1000e-6  # m
# n_x = 100
# cycle_time = 300            # s
# time_res = 100
# t = np.linspace(0, cycle_time, time_res)

# # numerical ref solution
# model, x_grid, x_ct = settings_crystallization.CSTR_PBM_aggregation_fragmentation(
#     800, x_c, x_max, 1, t, cadet_path, output_path)
# model.save()
# return_data = model.run_simulation()


# #%% run_CSTR_PBM_aggregation_fragmentation_test

# # define params
# n_x = 100
# n_col = 100
# x_c, x_max = 1e-6, 1000e-6            # m
# x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

# cycle_time = 300                      # s
# t = np.linspace(0, cycle_time, 200)



# model, x_grid, x_ct = settings_crystallization.Agg_DPFR(
#     n_x, n_col, x_c, x_max, 1, t, cadet_path, output_path)
# model.save()
# return_data = model.run_simulation()

# #%% run_DPFR_constFrag_test


# # system setup
# n_x = 100
# n_col = 100

# x_c, x_max = 1e-6, 1000e-6            # m
# x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

# cycle_time = 300                      # s
# t = np.linspace(0, cycle_time, 200)

# '''
# @note: There is no analytical solution in this case. Hence, we use a numerical reference
# '''

# model, x_grid, x_ct = settings_crystallization.Frag_DPFR(
#     n_x, n_col, x_c, x_max, 1, t, cadet_path, output_path)
# model.save()
# return_data = model.run_simulation()
    
# #%% run_DPFR_NGGR_aggregation_test

# # set up
# n_x = 100
# n_col = 100
# x_c, x_max = 1e-6, 1000e-6       # m
# x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

# # simulation time
# cycle_time = 200                 # s
# t = np.linspace(0, cycle_time, 200+1)

# model, x_grid, x_ct = settings_crystallization.DPFR_PBM_NGGR_aggregation(
#     n_x, n_col, x_c, x_max, 1, 1, t, cadet_path, output_path)
# model.save()
# return_data = model.run_simulation()
    

# #%% run_DPFR_aggregation_fragmentation_test
    
# # system setup
# n_x = 100
# n_col = 100

# x_c, x_max = 1e-6, 1000e-6            # m
# x_grid, x_ct = settings_crystallization.get_log_space(n_x, x_c, x_max)

# cycle_time = 300                      # s
# t = np.linspace(0, cycle_time, 200)

# '''
# @note: There is no analytical solution in this case. We are using a result as the reference solution.
# '''

# model, x_grid, x_ct = settings_crystallization.Agg_Frag_DPFR(
#     n_x, n_col, x_c, x_max, 1, t, cadet_path, output_path)
# model.save()
# return_data = model.run_simulation()
    
    