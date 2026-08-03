from cadet import Cadet
from src.benchmark_models import setting_Col1D_SMA_4comp_LWE_benchmark1 as benchmark1

particle_type = "GENERAL_RATE_PARTICLE" # None, HOMOGENEOUS_PARTICLE, GENERAL_RATE_PARTICLE, EQUILIBRIUM_PARTICLE
cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
output_path = r"C:\Users\jmbr\software\CADET-Verification\output"
unitId = '000'

modelAxial = Cadet()

modelAxial.install_path = cadet_path
modelAxial.root.input = benchmark1.get_model(
    ncomp=4,
    particle_type=particle_type,
    spatial_method_bulk=3, spatial_method_particle=3, refinement=2, idas_reftol=1e-6,
    column_geometry='AXIAL_FLOW_CYLINDER',
    POLYNOMIAL_INTEGRATION_TYPE=1,
    )['input']

modelAxial.filename = output_path + r"\col1D_axial_LWE_SMA.h5"
modelAxial.save()
return_data = modelAxial.run_simulation()
modelAxial.save()

if not return_data.return_code == 0:
    print(f"Axial simulation failed with return msg: {return_data.error_message}")

modelFrustum = Cadet()
modelFrustum.install_path = cadet_path
modelFrustum.root.input = benchmark1.get_model(
    ncomp=4,
    particle_type=particle_type,
    spatial_method_bulk=3, spatial_method_particle=3, refinement=2, idas_reftol=1e-6,
    column_geometry='AXIAL_FLOW_FRUSTUM', # AXIAL_FLOW_CYLINDER, AXIAL_FLOW_FRUSTUM, RADIAL_FLOW_CYLINDER_SHELL
    frustum_ratio=0.75, # ratio of the inlet radius to the outlet radius for frustum geometry
    )['input']

modelFrustum.filename = output_path + r"\col1D_frustum_LWE_SMA.h5"
modelFrustum.save()
return_data = modelFrustum.run_simulation()
modelFrustum.save()

if return_data.return_code == 0:
    print("Frustum simulation completed successfully.")
    # # compare with reference data for case frustum_ratio=1.0
    # refData = modelAxial.root.output.solution['unit_' + unitId].solution_outlet[:, :]
    # import numpy as np
    # diff = np.abs(modelFrustum.root.output.solution['unit_' + unitId].solution_outlet[:, :] - refData)
    # print("Maximum difference:", np.max(diff))
else:
    print(f"Frustum simulation failed with return msg: {return_data.error_message}")

modelRadial = Cadet()
modelRadial.install_path = cadet_path
modelRadial.root.input = benchmark1.get_model(
    ncomp=4,
    particle_type=particle_type,
    spatial_method_bulk=3, spatial_method_particle=3, refinement=2, idas_reftol=1e-6,
    column_geometry='RADIAL_FLOW_CYLINDER_SHELL', # AXIAL_FLOW_CYLINDER, AXIAL_FLOW_FRUSTUM, RADIAL_FLOW_CYLINDER_SHELL
    )['input']

modelRadial.filename = output_path + r"\col1D_radial_LWE_SMA.h5"
modelRadial.save()
return_data = modelRadial.run_simulation()
modelRadial.save()

if not return_data.return_code == 0:
    print(f"Radial simulation failed with return msg: {return_data.error_message}")

from matplotlib import pyplot as plt

plt.plot(modelAxial.root.output.solution.solution_times, modelAxial.root.output.solution.unit_000.solution_outlet[:, 0], label='salt axial')
plt.plot(modelFrustum.root.output.solution.solution_times, modelFrustum.root.output.solution.unit_000.solution_outlet[:, 0], label='salt frustum')
plt.plot(modelRadial.root.output.solution.solution_times, modelRadial.root.output.solution.unit_000.solution_outlet[:, 0], label='salt radial')
plt.legend()
plt.show()

for i in range(1, 4):
    plt.plot(modelAxial.root.output.solution.solution_times, modelAxial.root.output.solution.unit_000.solution_outlet[:, i], label=f'protein {i} axial')
    plt.plot(modelFrustum.root.output.solution.solution_times, modelFrustum.root.output.solution.unit_000.solution_outlet[:, i], label=f'protein {i} frustum')
    plt.plot(modelRadial.root.output.solution.solution_times, modelRadial.root.output.solution.unit_000.solution_outlet[:, i], label=f'protein {i} radial')
plt.legend()
plt.show()