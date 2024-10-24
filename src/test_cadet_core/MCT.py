import bench_func as bf
import utility.convergence as convergence
import re
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np

from cadet import Cadet
from cadetrdm import ProjectRepo


database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database/-/raw/core_tests/cadet_config/test_cadet-core/chromatography/"
sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core" / "chromatography"
os.makedirs(output_path, exist_ok=True)

# specify a source build cadet_path
cadet_path = r'C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master4_MMkinetics_Commit_a17024ae\aRELEASE\bin\cadet-cli.exe'
Cadet.cadet_path = cadet_path

# commit_message = f"Recreation of MCT reference files"
# with project_repo.track_results(results_commit_message=commit_message, debug=False):

n_jobs = -1

model = bf.create_object(database_path,
                         cadet_config_json = 'configuration_LRM_dynLin_1comp_benchmark2_FV_Z256.json',
                         output_path = str(output_path)
                         )

model.run()
model.load()
model.save()
