import sys
sys.path.insert(0, r"C:\Users\jmbr\software\CADET-Verification\src")
from benchmark_models.setting_radCol1D_lin_1comp_benchmark1 import get_model
from cadet import Cadet

install_path = r"C:\Users\jmbr\Desktop\CADET_compiled\master6_geomIntChange_25f9ff5\aRelease"

model_dict = get_model(spatial_method_bulk=0, particle_type='GENERAL_RATE_PARTICLE', spatial_method_par=0)

cadet = Cadet(install_path=install_path)
cadet.root = model_dict
cadet.filename = r"C:\Users\jmbr\software\CADET-Verification\scripts\scratch_test_radial.h5"
cadet.save()
rc = cadet.run()
print("returncode:", rc.return_code)
print("log:", rc.log[-3000:] if getattr(rc, 'log', None) else None)
print("error message:", getattr(rc, 'error_message', None))

