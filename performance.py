# -*- coding: utf-8 -*-
import shutil
import glob
import os
import sys
from pathlib import Path
sys.path.append(str(Path(".")))
import json

from cadet import Cadet
from cadetrdm import ProjectRepo

import src.generalized_unit_benchmark as bench

cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE" # r"C:\Users\jmbr\Cadet_testBuild\CADET-Core\out\install\aRELEASE"

small_test = 0

project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "performance"

database_path = (
        "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database"
        "/-/raw/core_tests/cadet_config/test_cadet-core/"
    ) + "chromatography/"

# #%%

# bench.chromatography_tests(
#     n_jobs=-1, database_path=database_path,
#     small_test=small_test, sensitivities=0,
#     output_path=output_path, cadet_path=cadet_path)

#%%

from cadetrdm import ProjectRepo
project_repo = ProjectRepo()

n_reruns = 10

rdm_debug_mode = False

commit_message = "Performance comparison old vs new generalized units"

#%%

with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):
    
    for i in range(0, n_reruns):
    
        output_path_tmp = str(output_path) + str(i)
        
        bench.chromatography_tests(
            n_jobs=-1, database_path=database_path,
            small_test=small_test, sensitivities=0,
            output_path=output_path_tmp, cadet_path=cadet_path)
        
        if i > 0: # update simualtion times, keep the fastest
            
            for tmp_file_path in glob.glob(os.path.join(output_path_tmp, '*.json')):
                filename = os.path.basename(tmp_file_path)
                main_file_path = os.path.join(output_path, filename)
        
                if not os.path.exists(main_file_path):
                    continue
                
                with open(tmp_file_path, 'r') as f_tmp, open(main_file_path, 'r') as f_main:
                    tmp_data = json.load(f_tmp)
                    main_data = json.load(f_main)
                    
                try:
                    simtime_vec_new = tmp_data['convergence']['DG_P3']['outlet']['Sim. time']
                    simtime_vec_old = main_data['convergence']['DG_P3']['outlet']['Sim. time']
                
                    for j in range(0, len(simtime_vec_new)):
                        if simtime_vec_new[j] < simtime_vec_old[j]:
                            main_data['convergence']['DG_P3']['outlet']['Sim. time'][j] = simtime_vec_new[j]
                            
                    with open(main_file_path, 'w') as f_main:
                        json.dump(main_data, f_main, indent=4)
                            
                except KeyError:
                    print(f"Skipping {filename}: missing some key.")
            
        else: # move to performance directory
            os.makedirs(output_path, exist_ok=True)
            for file_path in glob.glob(os.path.join(output_path_tmp, '*.json')):
                shutil.copy(file_path, output_path)
        