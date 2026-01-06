import numpy as np
from cadet import Cadet

Cadet.cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master7_fix2DGRM_a70da07\aRELEASE"

def kauthar_model(geometry):
    
    model = Cadet()
    
    flow_rate_factor = 1.0 / 2500.0 if geometry == "FRUSTUM" else 1.0
    
    model.root.input.model.connections.connections_include_dynamic_flow = np.int64(1)
    model.root.input.model.connections.connections_include_ports = np.int64(1)
    model.root.input.model.connections.nswitches = np.int64(2)
    model.root.input.model.connections.switch_000.connections = np.array([
      0.00000000e+00, 2.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     -1.00000000e+00,-1.00000000e+00, flow_rate_factor * 3.71016667e-05, 0.00000000e+00,
      0.00000000e+00, 0.00000000e+00, 2.00000000e+00, 3.00000000e+00,
      0.00000000e+00, 0.00000000e+00,-1.00000000e+00,-1.00000000e+00,
      flow_rate_factor * 3.71016667e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    model.root.input.model.connections.switch_000.section = np.int64(0)
    model.root.input.model.connections.switch_001.connections = np.array([
      1.00000000e+00, 2.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     -1.00000000e+00,-1.00000000e+00, flow_rate_factor * 3.76833333e-06, 0.00000000e+00,
      0.00000000e+00, 0.00000000e+00, 2.00000000e+00, 3.00000000e+00,
      0.00000000e+00, 0.00000000e+00,-1.00000000e+00,-1.00000000e+00,
      flow_rate_factor * 3.76833333e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    model.root.input.model.connections.switch_001.section = np.int64(1)
    model.root.input.model.nunits = np.int64(4)
    
    
    model.root.input.model.solver.gs_type = np.int64(1)
    model.root.input.model.solver.linear_solution_mode = np.int64(0)
    model.root.input.model.solver.max_krylov = np.int64(0)
    model.root.input.model.solver.max_restarts = np.int64(10)
    model.root.input.model.solver.schur_safety = np.float64(1e-08)
    
    
    model.root.input.model.unit_000.unit_type = np.bytes_(b'INLET')
    model.root.input.model.unit_000.inlet_type = np.bytes_(b'PIECEWISE_CUBIC_POLY')
    model.root.input.model.unit_000.ncomp = np.int64(1)
    model.root.input.model.unit_000.sec_000.const_coeff = np.array([0.04])
    model.root.input.model.unit_000.sec_000.cube_coeff = np.array([0.])
    model.root.input.model.unit_000.sec_000.lin_coeff = np.array([0.])
    model.root.input.model.unit_000.sec_000.quad_coeff = np.array([0.])
    
    
    model.root.input.model.unit_001.unit_type = np.bytes_(b'INLET')
    model.root.input.model.unit_001.inlet_type = np.bytes_(b'PIECEWISE_CUBIC_POLY')
    model.root.input.model.unit_001.ncomp = np.int64(1)
    model.root.input.model.unit_001.sec_000.const_coeff = np.array([0.])
    model.root.input.model.unit_001.sec_000.cube_coeff = np.array([0.])
    model.root.input.model.unit_001.sec_000.lin_coeff = np.array([0.])
    model.root.input.model.unit_001.sec_000.quad_coeff = np.array([0.])
    
    
    if geometry == "RADIAL":
        model.root.input.model.unit_002.unit_type = np.bytes_(b'RADIAL_GENERAL_RATE_MODEL')
        model.root.input.model.unit_002.col_length = np.float64(0.1)
        model.root.input.model.unit_002.col_radius_inner = np.float64(0.2)
        model.root.input.model.unit_002.col_radius_outer = np.float64(0.36)
        model.root.input.model.unit_002.velocity = np.int64(1)
        # cross_section_inner = 2.0 * np.pi * np.float64(0.2) * np.float64(0.1)
    elif geometry == "FRUSTUM":
        model.root.input.model.unit_002.unit_type = np.bytes_(b'FRUSTUM_GENERAL_RATE_MODEL')
        model.root.input.model.unit_002.col_length = np.float64(0.12)
        model.root.input.model.unit_002.col_radius_inner = np.float64(0.004)
        model.root.input.model.unit_002.col_radius_outer = np.float64(0.006)
        model.root.input.model.unit_002.velocity = np.int64(-1)
        # cross_section_inner = np.pi * np.float64(0.004) * np.float64(0.004)
    else:
        raise Exception(f"invalid geometry {geometry}")
        
    model.root.input.model.unit_002.col_porosity = np.float64(0.34)
    
    ## particle
    model.root.input.model.unit_002.npartype = 1
    model.root.input.model.unit_002.particle_type_000.par_geom = np.bytes_(b'SPHERE')
    model.root.input.model.unit_002.particle_type_000.nbound = np.array([1])
    model.root.input.model.unit_002.particle_type_000.par_porosity = np.float64(0.96)
    model.root.input.model.unit_002.particle_type_000.par_radius = np.float64(0.000125)
    
    model.root.input.model.unit_002.particle_type_000.has_film_diffusion = True
    model.root.input.model.unit_002.particle_type_000.has_pore_diffusion = True
    model.root.input.model.unit_002.particle_type_000.has_surface_diffusion = False
    model.root.input.model.unit_002.particle_type_000.film_diffusion = np.array([1.97e-05])
    model.root.input.model.unit_002.particle_type_000.pore_diffusion = np.array([6.299e-12])
    
    model.root.input.model.unit_002.particle_type_000.init_cp = np.array([0.0])
    model.root.input.model.unit_002.particle_type_000.init_cs = np.array([0.0])
    
    model.root.input.model.unit_002.particle_type_000.discretization.par_boundary_order = np.int64(2)
    model.root.input.model.unit_002.particle_type_000.discretization.par_disc_type = np.bytes_(b'EQUIDISTANT_PAR')
    model.root.input.model.unit_002.particle_type_000.discretization.ncells = np.int64(10)
    
    # adsorption
    model.root.input.model.unit_002.particle_type_000.adsorption.is_kinetic = np.False_
    model.root.input.model.unit_002.particle_type_000.adsorption.mcl_ka = np.array([110.88])
    model.root.input.model.unit_002.particle_type_000.adsorption.mcl_kd = np.array([1])
    model.root.input.model.unit_002.particle_type_000.adsorption.mcl_qmax = np.array([12.62626263])
    model.root.input.model.unit_002.particle_type_000.adsorption_model = np.bytes_(b'MULTI_COMPONENT_LANGMUIR')
    
    model.root.input.model.unit_002.col_dispersion = np.array([5.9e-07])
    model.root.input.model.unit_002.col_dispersion_dep = np.bytes_(b'POWER_LAW')
    model.root.input.model.unit_002.col_dispersion_dep_abs = np.True_
    model.root.input.model.unit_002.col_dispersion_dep_base = np.float64(1.0)
    model.root.input.model.unit_002.col_dispersion_dep_exponent = np.float64(1.0)
    model.root.input.model.unit_002.col_dispersion_multiplex = np.int64(3)
    model.root.input.model.unit_002.discretization.consistency_solver.init_damping = np.float64(0.01)
    model.root.input.model.unit_002.discretization.consistency_solver.max_iterations = np.int64(50)
    model.root.input.model.unit_002.discretization.consistency_solver.min_damping = np.float64(0.0001)
    model.root.input.model.unit_002.discretization.consistency_solver.solver_name = np.bytes_(b'LEVMAR')
    model.root.input.model.unit_002.discretization.consistency_solver.subsolvers = np.bytes_(b'LEVMAR')
    model.root.input.model.unit_002.discretization.fix_zero_surface_diffusion = np.False_
    model.root.input.model.unit_002.discretization.gs_type = np.True_
    model.root.input.model.unit_002.discretization.max_krylov = np.int64(0)
    model.root.input.model.unit_002.discretization.max_restarts = np.int64(10)
    model.root.input.model.unit_002.discretization.spatial_method = np.bytes_(b'FV')
    model.root.input.model.unit_002.discretization.ncol = np.int64(50)
    model.root.input.model.unit_002.discretization.schur_safety = np.float64(1e-08)
    model.root.input.model.unit_002.discretization.use_analytic_jacobian = np.True_
    model.root.input.model.unit_002.discretization.weno.boundary_model = np.int64(0)
    model.root.input.model.unit_002.discretization.weno.weno_eps = np.float64(1e-10)
    model.root.input.model.unit_002.discretization.weno.weno_order = np.int64(0)
    model.root.input.model.unit_002.init_c = np.array([0.])
    model.root.input.model.unit_002.ncomp = np.int64(1)
    
    
    model.root.input.model.unit_003.unit_type = np.bytes_(b'OUTLET')
    model.root.input.model.unit_003.ncomp = np.int64(1)
    
    
    model.root.input['return'].single_as_multi_port = np.True_
    model.root.input['return'].split_components_data = np.False_
    model.root.input['return'].split_ports_data = np.True_
    # model.root.input['return'].unit_000.write_coordinates = np.True_
    # model.root.input['return'].unit_000.write_sens_inlet = np.True_
    # model.root.input['return'].unit_000.write_sens_outlet = np.True_
    # model.root.input['return'].unit_000.write_sensdot_inlet = np.False_
    # model.root.input['return'].unit_000.write_sensdot_outlet = np.False_
    # model.root.input['return'].unit_000.write_soldot_inlet = np.False_
    # model.root.input['return'].unit_000.write_soldot_outlet = np.False_
    # model.root.input['return'].unit_000.write_solution_inlet = np.True_
    # model.root.input['return'].unit_000.write_solution_last_unit = np.False_
    # model.root.input['return'].unit_000.write_solution_outlet = np.True_
    # model.root.input['return'].unit_001.write_coordinates = np.True_
    # model.root.input['return'].unit_001.write_sens_inlet = np.True_
    # model.root.input['return'].unit_001.write_sens_outlet = np.True_
    # model.root.input['return'].unit_001.write_sensdot_inlet = np.False_
    # model.root.input['return'].unit_001.write_sensdot_outlet = np.False_
    # model.root.input['return'].unit_001.write_soldot_inlet = np.False_
    # model.root.input['return'].unit_001.write_soldot_outlet = np.False_
    # model.root.input['return'].unit_001.write_solution_inlet = np.True_
    # model.root.input['return'].unit_001.write_solution_last_unit = np.False_
    # model.root.input['return'].unit_001.write_solution_outlet = np.True_
    # model.root.input['return'].unit_002.write_coordinates = np.True_
    # model.root.input['return'].unit_002.write_sens_bulk = np.False_
    # model.root.input['return'].unit_002.write_sens_flux = np.False_
    # model.root.input['return'].unit_002.write_sens_inlet = np.True_
    # model.root.input['return'].unit_002.write_sens_outlet = np.True_
    # model.root.input['return'].unit_002.write_sens_particle = np.False_
    # model.root.input['return'].unit_002.write_sens_solid = np.False_
    # model.root.input['return'].unit_002.write_sensdot_bulk = np.False_
    # model.root.input['return'].unit_002.write_sensdot_flux = np.False_
    # model.root.input['return'].unit_002.write_sensdot_inlet = np.False_
    # model.root.input['return'].unit_002.write_sensdot_outlet = np.False_
    # model.root.input['return'].unit_002.write_sensdot_particle = np.False_
    # model.root.input['return'].unit_002.write_sensdot_solid = np.False_
    # model.root.input['return'].unit_002.write_soldot_bulk = np.False_
    # model.root.input['return'].unit_002.write_soldot_flux = np.False_
    # model.root.input['return'].unit_002.write_soldot_inlet = np.False_
    # model.root.input['return'].unit_002.write_soldot_outlet = np.False_
    # model.root.input['return'].unit_002.write_soldot_particle = np.False_
    # model.root.input['return'].unit_002.write_soldot_solid = np.False_
    # model.root.input['return'].unit_002.write_solution_bulk = np.True_
    # model.root.input['return'].unit_002.write_solution_flux = np.False_
    # model.root.input['return'].unit_002.write_solution_inlet = np.True_
    # model.root.input['return'].unit_002.write_solution_last_unit = np.False_
    # model.root.input['return'].unit_002.write_solution_outlet = np.True_
    # model.root.input['return'].unit_002.write_solution_particle = np.True_
    # model.root.input['return'].unit_002.write_solution_solid = np.False_
    model.root.input['return'].unit_003.write_coordinates = np.True_
    model.root.input['return'].unit_003.write_sens_inlet = np.True_
    model.root.input['return'].unit_003.write_sens_outlet = np.True_
    model.root.input['return'].unit_003.write_sensdot_inlet = np.False_
    model.root.input['return'].unit_003.write_sensdot_outlet = np.False_
    model.root.input['return'].unit_003.write_soldot_inlet = np.False_
    model.root.input['return'].unit_003.write_soldot_outlet = np.False_
    model.root.input['return'].unit_003.write_solution_inlet = np.True_
    model.root.input['return'].unit_003.write_solution_last_unit = np.False_
    model.root.input['return'].unit_003.write_solution_outlet = np.True_
    model.root.input['return'].write_sens_last = np.True_
    model.root.input['return'].write_solution_last = np.True_
    model.root.input['return'].write_solution_times = np.True_
    
    model.root.input.solver.consistent_init_mode = np.int64(1)
    model.root.input.solver.consistent_init_mode_sens = np.int64(1)
    model.root.input.solver.nthreads = np.int64(1)
    model.root.input.solver.sections.nsec = np.int64(2)
    model.root.input.solver.sections.section_continuity = np.array([0])
    model.root.input.solver.sections.section_times = np.array([    0.,19800.,31000.])
    model.root.input.solver.time_integrator.abstol = np.float64(1e-08)
    model.root.input.solver.time_integrator.algtol = np.float64(1e-12)
    model.root.input.solver.time_integrator.errortest_sens = np.False_
    model.root.input.solver.time_integrator.init_step_size = np.float64(1e-06)
    model.root.input.solver.time_integrator.max_convtest_fail = np.int64(1000000)
    model.root.input.solver.time_integrator.max_errtest_fail = np.int64(1000000)
    model.root.input.solver.time_integrator.max_newton_iter = np.int64(1000000)
    model.root.input.solver.time_integrator.max_newton_iter_sens = np.int64(1000000)
    model.root.input.solver.time_integrator.max_step_size = np.float64(0.0)
    model.root.input.solver.time_integrator.max_steps = np.int64(1000000)
    model.root.input.solver.time_integrator.reltol = np.float64(1e-06)
    model.root.input.solver.time_integrator.reltol_sens = np.float64(1e-12)
    model.root.input.solver.user_solution_times = np.linspace(0.0, 31000.0, 31001)
    
    model.filename = 'kauthar_frustum.h5'
    model.save()
    return model



import src.utility.convergence as convergence
import matplotlib.pyplot as plt


#%%
radModel = kauthar_model("RADIAL")
return_data = radModel.run_simulation()
if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")
radModel.save()

outlet = convergence.get_solution(radModel, which='outlet_port_000', unit='unit_003')
solution_time = convergence.get_solution_times(radModel)

plt.plot(solution_time, outlet, label='radial')
#%%
frustModel = kauthar_model("FRUSTUM")
return_data = frustModel.run_simulation()
if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")
frustModel.save()

outlet = convergence.get_solution(frustModel, which='outlet_port_000', unit='unit_003')
solution_time = convergence.get_solution_times(frustModel)

plt.plot(solution_time, outlet, label='frustum', linestyle='dashed')

plt.legend()
