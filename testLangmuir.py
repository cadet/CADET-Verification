
import matplotlib.pyplot as plt
import numpy as np

from cadet import Cadet

import src.benchmark_models.setting_Col1D_langLRM_2comp_benchmark1 as langSetting
import src.utility.convergence as convergence

radModel = Cadet()

radModel.root = langSetting.get_model(spatial_method_bulk=0)

radModel.filename = "radGRM.h5"
radModel.save()
return_data = radModel.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")

radModel.load_from_file()

outlet = convergence.get_solution(radModel, which='outlet')
solution_time = convergence.get_solution_times(radModel)

plt.plot(solution_time, outlet, label='radial fwd flow')
