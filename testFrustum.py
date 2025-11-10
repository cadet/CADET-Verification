
import matplotlib.pyplot as plt
import numpy as np

from cadet import Cadet

import src.benchmark_models.setting_Col1D_lin_1comp_benchmark1 as axSetting
import src.utility.convergence as convergence

Cadet.cadet_path = r"C:\Users\jmbr\Cadet_testBuild\CADET-Core\out\install\aRELEASE"


#%% Radial flow EOC test with pure advection

errors = []
EOC = []

for i in range(0, 8):

    refinement = 2**i
    
    frustModel = Cadet()
    
    frustModel.root = axSetting.get_model(
        0, particle_type="GENERAL_RATE_PARTICLE", spatial_method_particle=0,
        axRefinement=refinement, parZ=1,
        )
    
    frustModel.root.input.model.unit_001.UNIT_TYPE = "RADIAL_GENERAL_RATE_MODEL"
    
    col_porosity = 1.0
    
    frustum_inner_radius = np.sqrt((6.e-05 / 0.000575 / col_porosity) / np.pi)
    frustum_outer_radius = np.sqrt((6.e-05 / 0.000575 / col_porosity) / np.pi * 1.5)
    frustModel.root.input.model.unit_001.COL_RADIUS_INNER = frustum_inner_radius
    frustModel.root.input.model.unit_001.COL_RADIUS_OUTER = frustum_outer_radius
    
    frustModel.root.input.model.unit_001.col_dispersion = 0.0
    frustModel.root.input.model.unit_001.col_porosity = col_porosity
    frustModel.root.input.model.unit_001.particle_type_000.film_diffusion = 0.0
    frustModel.root.input.model.unit_001.particle_type_000.adsorption_model = "NONE"
    frustModel.root.input.solver.sections.section_times = [ 0.0, 10.0, 200.0 ]
    frustModel.root.input.solver.user_solution_times = np.linspace(0.0, 200.0, 200 + 1)
    
    frustModel.filename = "radAdvection.h5"
    
    frustModel.save()
    
    return_data = frustModel.run_simulation()
    
    if not return_data.return_code == 0:
        print(return_data.error_message)
        raise Exception(f"simulation failed")
    
    frustModel.load_from_file()
    
    outlet = convergence.get_solution(frustModel, which='outlet')
    solution_time = convergence.get_solution_times(frustModel)
    
    plt.plot(solution_time, outlet)
    plt.title('radial flow')
    
    # analytical
    from scipy.integrate import quad
    
    def velocity(x):
        
        radius = frustum_inner_radius + x * (frustum_outer_radius - frustum_inner_radius)
        
        return 6.e-05 / (col_porosity * 2.0 * np.pi * radius * 0.014)
    
    integral_val, error = quad(velocity, frustum_inner_radius, frustum_outer_radius)
    avg_velocity = integral_val / (frustum_outer_radius - frustum_inner_radius)
    pulse_start = (frustum_outer_radius - frustum_inner_radius) / avg_velocity
    pulse_end = pulse_start + 10.0
    
    reference = [1.0 if pulse_start <= x <= pulse_end else 0.0 for x in solution_time]
    error = abs(outlet - reference)
    errors.append(np.max(error))
    
    if i > 1:
        EOC.append(np.log(errors[i-2] / errors[i-1]) / np.log(2))
    
    plt.plot(solution_time, reference)
    
    plt.show()

print("errors:\n", errors)
print("EOC:\n", EOC)

#%% Frustum EOC test with pure advection


ref_model = Cadet()
ref_model.filename = r"C:\Users\jmbr\software\CADET-Verification\ref_frustAdvection.h5"
ref_model.load_from_file()
reference = convergence.get_outlet(ref_model)

errors = []
EOC = []

for i in range(0, 8):

    refinement = 2**i
    
    frustModel = Cadet()
    
    frustModel.root = axSetting.get_model(
        0, particle_type="GENERAL_RATE_PARTICLE", spatial_method_particle=0,
        axRefinement=refinement, parZ=1,
        )
    
    frustModel.root.input.model.unit_001.UNIT_TYPE = "FRUSTUM_GENERAL_RATE_MODEL"
    
    col_porosity = 1.0
    
    frustum_inner_radius = np.sqrt((6.e-05 / 0.000575 / col_porosity) / np.pi)
    frustum_outer_radius = np.sqrt((6.e-05 / 0.000575 / col_porosity) / np.pi * 1.5)
    frustModel.root.input.model.unit_001.COL_RADIUS_INNER = frustum_inner_radius
    frustModel.root.input.model.unit_001.COL_RADIUS_OUTER = frustum_outer_radius
    
    frustModel.root.input.model.unit_001.col_dispersion = 0.0
    frustModel.root.input.model.unit_001.col_porosity = col_porosity
    frustModel.root.input.model.unit_001.particle_type_000.film_diffusion = 0.0
    frustModel.root.input.model.unit_001.particle_type_000.adsorption_model = "NONE"
    frustModel.root.input.solver.sections.section_times = [ 0.0, 10.0, 200.0 ]
    frustModel.root.input.solver.user_solution_times = np.linspace(0.0, 200.0, 200 + 1)
    
    frustModel.filename = "frustAdvection.h5"
    
    frustModel.save()
    
    return_data = frustModel.run_simulation()
    
    if not return_data.return_code == 0:
        print(return_data.error_message)
        raise Exception(f"simulation failed")
    
    frustModel.load_from_file()
    
    outlet = convergence.get_solution(frustModel, which='outlet')
    solution_time = convergence.get_solution_times(frustModel)
    
    plt.plot(solution_time, outlet)
    plt.title('frustum flow')
    
    # analytical
    from scipy.integrate import quad
    
    def velocity(x):
        
        radius = frustum_inner_radius + x / 0.014 * (frustum_outer_radius - frustum_inner_radius)
        
        return 6.e-05 / (col_porosity * np.pi * np.square(radius))
    
    integral_val, error = quad(velocity, 0.0, 0.014)
    avg_velocity = integral_val / 0.014
    pulse_start = 0.014 / avg_velocity
    pulse_end = pulse_start + 10.0
    
    # reference = [1.0 if pulse_start <= x <= pulse_end else 0.0 for x in solution_time]
    error = abs(outlet - reference)
    errors.append(np.max(error))
    
    if i > 1:
        EOC.append(np.log(errors[i-2] / errors[i-1]) / np.log(2))
    
    plt.plot(solution_time, reference)
    
    plt.show()

print("errors:\n", errors)
print("EOC:\n", EOC)


#%% Comparison with pure advection

frustModel = Cadet()

frustModel.root = axSetting.get_model(
    0, particle_type="GENERAL_RATE_PARTICLE", spatial_method_particle=0,
    axRefinement=256, parZ=4,
    )

frustModel.root.input.model.unit_001.UNIT_TYPE = "FRUSTUM_GENERAL_RATE_MODEL"

col_porosity = 1.0

frustum_inner_radius = np.sqrt((6.e-05 / 0.000575 / col_porosity) / np.pi)
frustum_outer_radius = np.sqrt((6.e-05 / 0.000575 / col_porosity) / np.pi * 1.5)
frustModel.root.input.model.unit_001.COL_RADIUS_INNER = frustum_inner_radius
frustModel.root.input.model.unit_001.COL_RADIUS_OUTER = frustum_outer_radius

frustModel.root.input.model.unit_001.col_dispersion = 0.0
frustModel.root.input.model.unit_001.col_porosity = col_porosity
frustModel.root.input.model.unit_001.particle_type_000.adsorption_model = "NONE"
frustModel.root.input.solver.sections.section_times = [ 0.0, 10.0, 200.0 ]
frustModel.root.input.solver.user_solution_times = np.linspace(0.0, 200.0, 200 + 1)

frustModel.filename = "frustAdvection.h5"

frustModel.save()

return_data = frustModel.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")

frustModel.load_from_file()

outlet = convergence.get_solution(frustModel, which='outlet')
solution_time = convergence.get_solution_times(frustModel)

plt.plot(solution_time, outlet)
plt.title('frustum flow')
plt.show()

# analytical
from scipy.integrate import quad

def velocity(x):
    
    radius = frustum_inner_radius + x / 0.014 * (frustum_outer_radius - frustum_inner_radius)
    
    return 6.e-05 / (col_porosity * np.pi * np.square(radius))

integral_val, error = quad(velocity, 0.0, 0.014)
avg_velocity = integral_val / 0.014
print("inner velocity: ", velocity(0.0))
print("outer velocity: ", velocity(0.014))
print("avg. velocity: ", avg_velocity)
print("expected breakthrough: ", 0.014 / avg_velocity)
print("axial breakthrough: ", 0.014 / 5.57e-4)

print("outlet at t = ", int(0.014 / avg_velocity), ": ", outlet[int(0.014 / avg_velocity)])
print("outlet at t = ", int(0.014 / avg_velocity) + 1, ": ", outlet[int(0.014 / avg_velocity)])
print("outlet at t = ", int(0.014 / 5.57e-4), ": ", outlet[int(0.014 / 5.57e-4)])


#%% Comparison with radial
# Frustum vs Radial flow for a single-comp. linear binding setting,
# with similar bed height and cross sectional area at the inlet
# (which is at the smaller radius cross-section area in both settings)

frustModel = Cadet()

frustModel.root = axSetting.get_model(
    0, particle_type="GENERAL_RATE_PARTICLE", spatial_method_particle=0,
    axRefinement=16, parZ=4,
    )

frustModel.root.input.model.unit_001.UNIT_TYPE = "FRUSTUM_GENERAL_RATE_MODEL"

frustum_inner_radius = np.sqrt((6.e-05 / 0.000575 / 0.37) / np.pi)
frustum_outer_radius = np.sqrt((6.e-05 / 0.000575 / 0.37) / np.pi * 1.1)
frustModel.root.input.model.unit_001.COL_RADIUS_INNER = frustum_inner_radius
frustModel.root.input.model.unit_001.COL_RADIUS_OUTER = frustum_outer_radius

frustModel.filename = "frustGRM.h5"

frustModel.save()

return_data = frustModel.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")

frustModel.load_from_file()

outlet = convergence.get_solution(frustModel, which='outlet')
solution_time = convergence.get_solution_times(frustModel)

plt.plot(solution_time, outlet, label='frustum')

radModel = Cadet()

radModel.root = axSetting.get_model(
    0, particle_type="GENERAL_RATE_PARTICLE", spatial_method_particle=0,
    axRefinement=16, parZ=4, weno_order=1
    )

radModel.root.input.model.unit_001.UNIT_TYPE = "RADIAL_GENERAL_RATE_MODEL"

# cross sectional area is also specified
cross_section_area_inner = np.pi * frustum_inner_radius * frustum_inner_radius
cross_section_area_outer = np.pi * frustum_outer_radius * frustum_outer_radius
cylinder_inner_shell_radius = cross_section_area_inner / (2.0 * np.pi * 0.014) # r = A / (2 pi H)
cylinder_outer_shell_radius = cross_section_area_outer / (2.0 * np.pi * 0.014) # r = A / (2 pi H)
radModel.root.input.model.unit_001.COL_RADIUS_INNER = cylinder_inner_shell_radius
radModel.root.input.model.unit_001.COL_RADIUS_OUTER = cylinder_inner_shell_radius + 0.014

print("length of radial column: ", radModel.root.input.model.unit_001.COL_RADIUS_OUTER - radModel.root.input.model.unit_001.COL_RADIUS_INNER)

radModel.filename = "radGRM.h5"

radModel.save()

return_data = radModel.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")

radModel.load_from_file()

outlet = convergence.get_solution(radModel, which='outlet')
solution_time = convergence.get_solution_times(radModel)

plt.plot(solution_time, outlet, label='radial')
plt.title('frustum vs. radial flow for similar cross-section areas')
plt.legend()
plt.show()

#%% Comparison with axial

frustModel = Cadet()

frustModel.root = axSetting.get_model(
    0, particle_type="GENERAL_RATE_PARTICLE", spatial_method_particle=0,
    axRefinement=16, parZ=4,
    )

frustModel.root.input.model.unit_001.UNIT_TYPE = "FRUSTUM_GENERAL_RATE_MODEL"

frustModel.root.input.model.unit_001.COL_RADIUS_INNER = np.sqrt((6.e-05 / 0.000575 / 0.37) / np.pi)
frustModel.root.input.model.unit_001.COL_RADIUS_OUTER = np.sqrt((6.e-05 / 0.000575 / 0.37) / np.pi * 1.5)

frustModel.filename = "frustGRM.h5"

frustModel.save()

return_data = frustModel.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")

frustModel.load_from_file()

outlet = convergence.get_solution(frustModel, which='outlet')
solution_time = convergence.get_solution_times(frustModel)

plt.plot(solution_time, outlet)
plt.title('frustum flow')
plt.show()


axModel = Cadet()

axModel.root = axSetting.get_model(
    0, particle_type="GENERAL_RATE_PARTICLE", spatial_method_particle=0,
    axRefinement=16, parZ=4, weno_order=1
    )

axModel.filename = "axGRM.h5"

axModel.save()

return_data = axModel.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")

axModel.load_from_file()

outlet = convergence.get_solution(axModel, which='outlet')
solution_time = convergence.get_solution_times(axModel)

plt.plot(solution_time, outlet)
plt.title('axial flow')
plt.show()



#%% Comparison with axial sensitivities

frustModel.root = axSetting.get_GRM_sensbenchmark1(
    0, spatial_method_particle=0, axRefinement=4, parZ=4,
    )


frustModel.root.input.model.unit_001.UNIT_TYPE = "FRUSTUM_GENERAL_RATE_MODEL"

frustModel.root.input.model.unit_001.COL_RADIUS_INNER = np.sqrt((6.e-05 / 0.000575 / 0.37) / np.pi)
frustModel.root.input.model.unit_001.COL_RADIUS_OUTER = np.sqrt((6.e-05 / 0.000575 / 0.37) / np.pi)

frustModel.filename = "frustGRM.h5"

frustModel.save()

return_data = frustModel.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")

frustModel.load_from_file()

outlet = convergence.get_solution(frustModel, which='sens_outlet', sensIdx=0)
solution_time = convergence.get_solution_times(frustModel)

plt.plot(solution_time, outlet)
plt.show()


axModel = Cadet()

axModel.root = axSetting.get_GRM_sensbenchmark1(
    0, spatial_method_particle=0, axRefinement=4, parZ=4,
    weno_order=1
    )

axModel.filename = "axGRM.h5"

axModel.save()

return_data = axModel.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")

axModel.load_from_file()

outlet = convergence.get_solution(axModel, which='sens_outlet', sensIdx=0)
solution_time = convergence.get_solution_times(axModel)

plt.plot(solution_time, outlet)
plt.show()



