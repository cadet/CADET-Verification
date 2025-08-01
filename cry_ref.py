# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:51:51 2025

@author: jmbr
"""

import src.benchmark_models.settings_crystallization as cry
import src.utility.convergence as convergence



cadet_path = convergence.get_cadet_path()

output_path = r"C:\Users\jmbr\software\CADET-Verification"

#%%

# model = cry.CSTR_PBM_growth(2000, cadet_path, output_path)
# model.save()
# model.run_simulation()


#%%

# N_x_ref   = 200 + 2
# N_col_ref = 200

# x_max = 900e-6 # um

# ## get ref solution
    
# model = cry.DPFR_PBM_primarySecondaryNucleationGrowth(N_x_ref, N_col_ref, cadet_path, output_path)
# model.save()
# model.run_simulation()




#%%

from joblib import Parallel, delayed

# List of model constructors
model_constructors = [
    # cry.CSTR_PBM_growthSizeDep,
    cry.CSTR_PBM_primaryNucleationAndGrowth,
    cry.CSTR_PBM_primarySecondaryNucleationAndGrowth
    # cry.CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion
]

# Function to run one simulation
def run_model(nx, model_class):
    model = model_class(nx, cadet_path, output_path)
    model.save()
    model.run_simulation()

# Run in parallel
Parallel(n_jobs=-1)(delayed(run_model)(1602, model_class) for model_class in model_constructors)









