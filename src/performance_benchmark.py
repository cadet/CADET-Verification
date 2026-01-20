# -*- coding: utf-8 -*-
import shutil
import glob
import os
import sys
from pathlib import Path
import re
sys.path.append(str(Path(".")))
import json

from cadet import Cadet
from cadetrdm import ProjectRepo

import chromatography as bench
import utility.convergence as convergence



#%% Computation of the performance benchmark

# Specify the two cadet-cli that should be compared.
# Give the corresponding output_paths a meaningful name, add commit message


cadet_path2 = r"C:\Users\jmbr\Desktop\CADET_compiled\master4_v600a1_518d41b\aRELEASE"
cadet_path1 = r"C:\Users\jmbr\Desktop\CADET_compiled\NANerror\aRELEASE"

# cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE" # r"C:\Users\jmbr\Desktop\CADET_compiled\master3_generalUnit_f1a1972\aRELEASE"

model = Cadet()

model.cadet_path = cadet_path1

commit_message = "Performance benchmark residual NAN check"


project_repo = ProjectRepo()
output_path2 = project_repo.output_path / "test_cadet-core" / "masterPerformance"
output_path1 = project_repo.output_path / "test_cadet-core" / "featurePerformance"

n_reruns = 2

n_jobs = -1

rdm_debug_mode = 1

small_test = 1

delete_h5_files = 1


#%%

with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):
    
    for i in range(0, n_reruns):
    
        output_path_tmp1 = str(output_path1) + "_run" + str(i)
        output_path_tmp2 = str(output_path2) + "_run" + str(i)
        
        bench.chromatography_tests(
            n_jobs=n_jobs,
            small_test=small_test, sensitivities=0,
            output_path=output_path_tmp1, cadet_path=cadet_path1)
            
        bench.chromatography_tests(
            n_jobs=n_jobs,
            small_test=small_test, sensitivities=0,
            output_path=output_path_tmp2, cadet_path=cadet_path2)
        
        if delete_h5_files:
                convergence.delete_h5_files(output_path_tmp1)
                convergence.delete_h5_files(output_path_tmp2)
        
        numMethodPattern = re.compile(r'^(FV|DG_P[1-9](parP[1-9])?)$')
        
        if i > 0: # update simualtion times, keep the fastest
            
            for output_path_tmp in [output_path_tmp1, output_path_tmp2]:
            
                for tmp_file_path in glob.glob(os.path.join(output_path_tmp, '*.json')):
                    
                    filename = os.path.basename(tmp_file_path)
                    main_path = re.sub(r'_run\d+', '', output_path_tmp)
                    main_file_path = os.path.join(main_path, filename)
            
                    if not os.path.exists(main_file_path):
                        continue
                    
                    with open(tmp_file_path, 'r') as f_tmp, open(main_file_path, 'r') as f_main:
                        tmp_data = json.load(f_tmp)
                        main_data = json.load(f_main)
                
                    try:
                        # iterate over all keys under convergence
                        for method in main_data['convergence'].keys():
                            if numMethodPattern.match(method):
                                simtime_new = tmp_data['convergence'][method]['outlet']['Sim. time']
                                simtime_old = main_data['convergence'][method]['outlet']['Sim. time']
                
                                for j in range(len(simtime_new)):
                                    if simtime_new[j] < simtime_old[j]:
                                        main_data['convergence'][method]['outlet']['Sim. time'][j] = simtime_new[j]
                
                        with open(main_file_path, 'w') as f_main:
                            json.dump(main_data, f_main, indent=4)
                
                    except KeyError:
                        print(f"Skipping {filename}: missing some key.")
            
        else: # move to performance directory
            os.makedirs(output_path1, exist_ok=True)
            for file_path in glob.glob(os.path.join(output_path_tmp1, '*.json')):
                shutil.copy(file_path, output_path1)
            os.makedirs(output_path2, exist_ok=True)
            for file_path in glob.glob(os.path.join(output_path_tmp2, '*.json')):
                shutil.copy(file_path, output_path2)
        
        
        
# %% Actual comparison of the data

import src.utility.compareConvergenceData as compare

# Local data comparison
compare.compare_json_directories(
    str(output_path1),
    str(output_path2),
    print_only_convergence_differences=False
    )


# github repo data comparison, requires github token
# compare.compare_json_directories(
#     ("cadet", "CADET-Verification-Output", "2025-08-13_22-00-53_release/cadet-core_v504_329536f", "test_cadet-core"),
#     ("cadet", "CADET-Verification-Output", "2025-09-15_12-48-47_release/cadet-core_v5.1.0_c4c48ec", "test_cadet-core"),
#     print_only_convergence_differences=False
# )




