# -*- coding: utf-8 -*-

import sys
from pathlib import Path
sys.path.append(str(Path(".")))

from cadet import Cadet

import src.benchmark_models.setting_Col1D_SMA_4comp_LWE_benchmark1 as settingSMA1D
import src.benchmark_models.setting_Col2D_SMA_4comp_LWE_benchmark1 as settingSMA2D
import src.benchmark_models.setting_Col2D_linBnd_1comp_benchmark1 as settingLin2D

Cadet.cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

nRadZones = 2
polyDeg = 3
axRefinement = 1
write_column_concentrations = True
#%%

model2D = Cadet()

uoType = "COLUMN_MODEL_2D"
spatialMethod = "DG"

model2D.root = settingSMA2D.get_model(
    particle_type='GENERAL_RATE_PARTICLE',
    nRadZones=nRadZones,
    axRefinement=axRefinement, radRefinement=1,
    axP=polyDeg, radP=polyDeg, parP=polyDeg,
    return_bulk=write_column_concentrations,
    return_particle=write_column_concentrations,
    # col_dispersion_radial=0.0,
    idas_reftol=1e-10 # = abstol, reltol*100, algtol*100
    )

model2D.filename = "test2D_SMA.h5"
model2D.save()
model2D.run()

#%%

model1D = Cadet()

model1D.root = settingSMA1D.get_model(
    particle_type='GENERAL_RATE_PARTICLE',
    axRefinement=axRefinement,
    axP=polyDeg, radP=polyDeg, parP=polyDeg,
    return_bulk=write_column_concentrations,
    return_particle=write_column_concentrations,
    idas_reftol=1e-10 # = abstol, reltol*100, algtol*100
    )

model1D.filename = "test1D_SMA.h5"
model1D.save()
model1D.run()


#%%

import numpy as np
import matplotlib.pyplot as plt

import src.utility.convergence as convergence


filenameOldGRM = r"C:\Users\jmbr\software\CADET-Core\test\data\ref_GRM_reqSMA_4comp_sensbenchmark1_exIntDG_P3Z8_GSM_parP3parZ1.h5"
outletOld = convergence.get_outlet(filenameOldGRM, "000")
timeOld = convergence.get_solution_times(filenameOldGRM)
plt.title("1D Old solution - components")
plt.plot(timeOld, outletOld[:, 1:4])
plt.show()
plt.title("1D Old solution - salt")
plt.plot(timeOld, outletOld[:, 0])
plt.show()

#%%


filename1D = r"C:\Users\jmbr\software\CADET-Verification\test1D_SMA.h5"
outlet1D = convergence.get_outlet(filename1D, "000")
time1D = convergence.get_solution_times(filename1D)

filename2D = r"C:\Users\jmbr\software\CADET-Verification\test2D_SMA.h5"
outlet2D = convergence.get_outlet(filename2D, "000")
time2D = convergence.get_solution_times(filename2D)


plt.title("1D solution - components")
plt.plot(time1D, outlet1D[:, 1:4])
plt.show()
plt.title("1D solution - salt")
plt.plot(time1D, outlet1D[:, 0])
plt.show()
plt.title("2D solution - components port 0")
plt.plot(time2D, outlet2D[:, 0, 1:4])
plt.show()
plt.title("2D solution - salt port 0")
plt.plot(time2D, outlet2D[:, 0, 0])
plt.show()
# plt.title("2D solution - components port 1")
# plt.plot(time1D, outlet2D[:, 1, 1:4])
# plt.show()
# plt.title("2D solution - salt port 1")
# plt.plot(time1D, outlet2D[:, 1, 0])
# plt.show()

print("error with salt")
error = np.max(np.abs(outlet2D[:, 0, :]-outlet2D[:, 1, :]))
print("port0 - port1 error: ", error)
error = np.max(np.abs(outlet2D[:, 0, :]-outlet1D[:, :]))
print("2D port0 - 1D error: ", error)

print("error without salt")
error = np.max(np.abs(outlet2D[:, 0, 1:4]-outlet2D[:, 1, 1:4]))
print("port0 - port1 error: ", error)
error = np.max(np.abs(outlet2D[:, 0, 1:4]-outlet1D[:, 1:4]))
print("2D port0 - 1D error: ", error)

#%%

error = np.max(np.abs(outletOld[:, :]-outlet1D[:, :]))
print("error 1D old vs new with salt", error)
error = np.max(np.abs(outletOld[:, 1:4]-outlet1D[:, 1:4]))
print("error 1D old vs new without salt", error)


