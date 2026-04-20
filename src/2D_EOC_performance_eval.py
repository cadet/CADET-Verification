
import matplotlib.pyplot as plt
import json
from pathlib import Path

import src.utility.convergence as convergence


file_path = r"C:\Users\jmbr\Downloads\convergence_2DGRM3Zone_noBnd_1Comp.json"
# file_path = str(Path.cwd()) + r"\output\test_cadet-core\2Dchromatography\convergence_2DGRM3Zone_noBnd_1Comp - Kopie.json"

with open(file_path, 'r') as f:
    convergenceData = json.load(f)

    simTimesDG = convergenceData['convergence']['DG_P3']['outlet']['Sim. time']
    maxErrorsDG = convergenceData['convergence']['DG_P3']['outlet']['$L^1$ error']
    
    simTimesFV = convergenceData['convergence']['FV']['outlet']['Sim. time']
    maxErrorsFV = convergenceData['convergence']['FV']['outlet']['$L^1$ error']
    
    
    convergence.std_plot(x_axis=simTimesDG, y_axis=maxErrorsDG, label='DGP3')
    convergence.std_plot(x_axis=simTimesFV, y_axis=maxErrorsFV, label='FV')
    convergence.std_plot_prep(benchmark_plot=True)
    plt.savefig(r"C:\Users\jmbr\Downloads\convergence_2DGRM3Zone_noBnd_1Comp.png")
    plt.show()

