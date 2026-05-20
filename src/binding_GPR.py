import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cadet import Cadet
import src.utility.convergence as convergence

import src.benchmark_models.setting_Col1D_GPR as setting_Col1D_GPR

def test_GPR_binding(output_path:str, cadet_path:str):
    
    _reference_data_path_ = str(
        Path(__file__).resolve().parent.parent / 'data' / 'CADET-Core_reference' / 'binding'
    )
    
    #%% Test Shallow_RBF_15
    
    setting_name = 'Shallow_RBF_15'
    
    model = Cadet()
    model.install_path = cadet_path
    model.root.input = setting_Col1D_GPR.get_model(f"{setting_name}")
    model.filename = output_path + f"/GPR_{setting_name}.h5"
    return_data = model.run_simulation()
    model.save()
    
    if not return_data.return_code == 0:
        raise Exception(f"simulation failed with error {return_data.error_message}\n and LOG\n {return_data.log}")
    
    model.load_from_file()
    outlet = convergence.get_solution(model, which='outlet')
    solutionTime = convergence.get_solution_times(model)
    plt.plot(solutionTime, outlet, label='sim c')
    
    
    modelRef = Cadet()
    modelRef.filename = _reference_data_path_ + f"/GPR_{setting_name}.h5"
    modelRef.load_from_file()
    outletRef = convergence.get_solution(modelRef, which='outlet')
    solutionTimeRef = convergence.get_solution_times(modelRef)
    plt.plot(solutionTimeRef, outletRef, label='ref c', linestyle='dashed')
    plt.title(f'GPR_{setting_name}')
    plt.legend()
    
    # ensure same start/end times
    if solutionTime[0] != solutionTimeRef[0] or solutionTime[-1] != solutionTimeRef[-1]:
        raise ValueError(
            f"Time ranges do not match: "
            f"[{solutionTimeRef[0]}, {solutionTimeRef[-1]}] vs "
            f"[{solutionTime[0]}, {solutionTime[-1]}]"
        )
    
    # common timeline
    t_common = np.linspace(solutionTimeRef[0], solutionTimeRef[-1], 10000)
    
    # interpolate
    y1 = np.interp(t_common, solutionTimeRef, outletRef)
    y2 = np.interp(t_common, solutionTime, outlet)
    
    abs_max_diff = np.max(np.abs(y1-y2))
    txt = f"Max |error| = {abs_max_diff:.3e}"
    # place box in top-left corner of axes
    plt.text(
        0.02, 0.98,                # x,y in axes coordinates
        txt,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.8
        )
    )
    
    plt.savefig(output_path + '/GPR_{setting_name}.png')
    plt.show()
    
    
    #%% Test Shallow_MLP_7
    
    setting_name = 'Shallow_MLP_7'
    
    model = Cadet()
    model.install_path = cadet_path
    model.root.input = setting_Col1D_GPR.get_model(f"{setting_name}")
    model.filename = output_path + f'\{setting_name}.h5'
    model.save()
    return_data = model.run_simulation()
    
    if not return_data.return_code == 0:
        raise Exception(f"simulation failed with error {return_data.error_message}\n and LOG\n {return_data.log}")
    
    model.load_from_file()
    outlet = convergence.get_solution(model, which='outlet')
    solutionTime = convergence.get_solution_times(model)
    plt.plot(solutionTime, outlet, label='sim c')
    
    
    modelRef = Cadet()
    modelRef.filename = _reference_data_path_ + f"/GPR_{setting_name}.h5"
    modelRef.load_from_file()
    outletRef = convergence.get_solution(modelRef, which='outlet')
    solutionTimeRef = convergence.get_solution_times(modelRef)
    plt.plot(solutionTimeRef, outletRef, label='ref c', linestyle='dashed')
    plt.title(f'GPR_{setting_name}')
    plt.legend()
    
    # ensure same start/end times
    if solutionTime[0] != solutionTimeRef[0] or solutionTime[-1] != solutionTimeRef[-1]:
        raise ValueError(
            f"Time ranges do not match: "
            f"[{solutionTimeRef[0]}, {solutionTimeRef[-1]}] vs "
            f"[{solutionTime[0]}, {solutionTime[-1]}]"
        )
    
    # common timeline
    t_common = np.linspace(solutionTimeRef[0], solutionTimeRef[-1], 10000)
    
    # interpolate
    y1 = np.interp(t_common, solutionTimeRef, outletRef)
    y2 = np.interp(t_common, solutionTime, outlet)
    
    abs_max_diff = np.max(np.abs(y1-y2))
    txt = f"Max |error| = {abs_max_diff:.3e}"
    # place box in top-left corner of axes
    plt.text(
        0.02, 0.98,                # x,y in axes coordinates
        txt,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.8
        )
    )
    
    plt.show()
