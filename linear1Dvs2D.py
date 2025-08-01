# -*- coding: utf-8 -*-

import sys
from pathlib import Path
sys.path.append(str(Path(".")))

from cadet import Cadet

import src.benchmark_models.setting_Col2D_linBnd_1comp_benchmark1 as settingLinear

Cadet.cadet_path = r"C:\Users\jmbr\Cadet_testBuild\CADET_2DmodelsDG\out\install\aRELEASE"

nComp = 4
nRadZones = 1
polyDeg = 3
axRefinement = 1
#%%

model2D = Cadet()

uoType = "COLUMN_MODEL_2D"
spatialMethod = "DG"

model2D.root = settingLinear.get_model(
    column_resolution='2D',
    particle_type='GENERAL_RATE_PARTICLE',
    nComp = nComp,
    nRadZones=nRadZones,
    axRefinement=axRefinement, radRefinement=1,
    axP=polyDeg, radP=polyDeg, parP=polyDeg,
    return_bulk=True,
    return_particle=True,
    col_dispersion_radial=0.0
    )
model2D.filename = "test2D_linear.h5"
model2D.save()
data = model2D.run()
if data.return_code != 0:
    print(data.error_message)
    print(data.log)

#%%

model1D = Cadet()
model1D.root = settingLinear.get_model(
    column_resolution='1D',
    particle_type='GENERAL_RATE_PARTICLE',
    nComp = nComp,
    axRefinement=axRefinement,
    axP=polyDeg, parP=polyDeg,
    return_bulk=True,
    return_particle=True
    )
model1D.filename = "test1D_linear.h5"
model1D.save()
data = model1D.run()
if data.return_code != 0:
    print(data.error_message)
    print(data.log)

#%%

import numpy as np
import matplotlib.pyplot as plt

import src.utility.convergence as convergence


filename1D = r"C:\Users\jmbr\software\CADET-Verification\test1D_linear.h5"
outlet1D = convergence.get_outlet(filename1D, "000")
time1D = convergence.get_solution_times(filename1D)

filename2D = r"C:\Users\jmbr\software\CADET-Verification\test2D_linear.h5"
outlet2D = convergence.get_outlet(filename2D, "000")
time2D = convergence.get_solution_times(filename2D)


plt.title("1D solution")
if nComp > 1:
    plt.plot(time1D, outlet1D[:, 0:nComp])
else:
    plt.plot(time1D, outlet1D[:])
plt.show()
plt.title("2D solution port 0")
if nComp > 1:
    plt.plot(time2D, outlet2D[:, 0, 0:nComp])
else:
    plt.plot(time2D, outlet2D[:, 0])
plt.show()
plt.title("2D solution port " + str((polyDeg+1)*nRadZones-1))
if nComp > 1:
    plt.plot(time2D, outlet2D[:, (polyDeg+1)*nRadZones-1, 0:nComp])
else:
    plt.plot(time2D, outlet2D[:, 1])
plt.show()

if nComp > 1:
    error = np.max(np.abs(outlet2D[:, 0, :]-outlet2D[:, 1, :]))
else:
    error = np.max(np.abs(outlet2D[:, 0]-outlet2D[:, 1]))
print("port0 - port", (polyDeg+1)*nRadZones-1, "error: ", error)
if nComp > 1:
    error = np.max(np.abs(outlet2D[:, (polyDeg+1)*nRadZones-1, :]-outlet1D[:, :]))
else:
    error = np.max(np.abs(outlet2D[:, (polyDeg+1)*nRadZones-1]-outlet1D[:]))
    
print("2D port0 - 1D error: ", error)
