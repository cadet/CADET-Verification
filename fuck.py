# -*- coding: utf-8 -*-

import sys
from pathlib import Path
sys.path.append(str(Path(".")))

from cadet import Cadet

import src.benchmark_models.setting_Col1D_SMA_4comp_LWE_benchmark1 as setting1D
import src.benchmark_models.setting_Col2D_SMA_4comp_LWE_benchmark1 as setting2D
import src.benchmark_models.setting_Col2D_linBnd_1comp_benchmark1 as settingLin2D

Cadet.cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

#%%

model = Cadet()
model.root = setting1D.get_model()
model.filename = "jojo1D.h5"
model.save()
model.run()

#%%

model = Cadet()
model.root = setting2D.get_model(particle_type=None)
model.filename = "jojo2D.h5"
model.save()
model.run()

#%%

model = Cadet()
model.root = settingLin2D.get_model2(particle_type=None, nComp=2, refinement=1)
model.filename = "1CompLin2D.h5"
model.save()
model.run()

#%%

# refinement 1, nComp 2
# -> we have 4 * (8*1) * 4 * (3*1) * (2 + 4 * 4) # axP * axZ * radP * radZ * (nComp + parP * 2nComp)
# = 6912 DOF
model = Cadet()
model.root = settingLin2D.get_model2(particle_type=None,
                                     nComp=2, nRadZones=1,
                                     axRefinement=0.5, axP=3, radP=1, parP=1,
                                     analytical_jacobian=True
                                     )
model.filename = "2CompLin2D.h5"
model.save()
model.run()




#%%

model = Cadet()

uoType = "COLUMN_MODEL_2D"
spatialMethod = "DG"

model.root = settingLin2D.get_model2(particle_type=None,
                                     nComp=2, nRadZones=1,
                                     axRefinement=0.5, axP=3, radP=1, parP=1,
                                     analytical_jacobian=True
                                     )
model.filename = "2CompLin2DcppTest.h5"
model.save()
model.run()