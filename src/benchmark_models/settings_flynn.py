import numpy as np
from cadet import Cadet


model = Cadet()
model.root.input.model.connections.nswitches = np.int64(1)
model.root.input.model.connections.switch_000.connections = np.array([ 0.00000000e+00, 1.00000000e+00,-1.00000000e+00,-1.00000000e+00,
  1.09083078e-08, 1.00000000e+00, 2.00000000e+00,-1.00000000e+00,
 -1.00000000e+00, 1.09083078e-08])
model.root.input.model.connections.switch_000.section = np.int64(0)
model.root.input.model.nunits = np.int64(3)
model.root.input.model.solver.gs_type = np.int64(1)
model.root.input.model.solver.max_krylov = np.int64(0)
model.root.input.model.solver.max_restarts = np.int64(10)
model.root.input.model.solver.schur_safety = np.float64(1e-08)

model.root.input.model.unit_000.inlet_type = np.bytes_(b'PIECEWISE_CUBIC_POLY')
model.root.input.model.unit_000.ncomp = np.int64(3)
model.root.input.model.unit_000.sec_000.const_coeff = np.array([-1.78653518e+00, 2.65464006e-05, 1.70179972e-06])
model.root.input.model.unit_000.sec_000.cube_coeff = np.array([0.,0.,0.])
model.root.input.model.unit_000.sec_000.lin_coeff = np.array([0.,0.,0.])
model.root.input.model.unit_000.sec_000.quad_coeff = np.array([0.,0.,0.])
model.root.input.model.unit_000.sec_001.const_coeff = np.array([-1.78653518, 0.        , 0.        ])
model.root.input.model.unit_000.sec_001.cube_coeff = np.array([0.,0.,0.])
model.root.input.model.unit_000.sec_001.lin_coeff = np.array([0.,0.,0.])
model.root.input.model.unit_000.sec_001.quad_coeff = np.array([0.,0.,0.])
model.root.input.model.unit_000.sec_002.const_coeff = np.array([-1.78653518, 0.        , 0.        ])
model.root.input.model.unit_000.sec_002.cube_coeff = np.array([-8.43143424e-13, 0.00000000e+00, 0.00000000e+00])
model.root.input.model.unit_000.sec_002.lin_coeff = np.array([-0.00021525, 0.        , 0.        ])
model.root.input.model.unit_000.sec_002.quad_coeff = np.array([2.09495648e-08,0.00000000e+00,0.00000000e+00])
model.root.input.model.unit_000.sec_003.const_coeff = np.array([-2.7480333, 0.       , 0.       ])
model.root.input.model.unit_000.sec_003.cube_coeff = np.array([0.,0.,0.])
model.root.input.model.unit_000.sec_003.lin_coeff = np.array([0.,0.,0.])
model.root.input.model.unit_000.sec_003.quad_coeff = np.array([0.,0.,0.])
model.root.input.model.unit_000.unit_type = np.bytes_(b'INLET')


model.root.input.model.unit_001.particle_type_000.has_film_diffusion = True
model.root.input.model.unit_001.particle_type_000.has_pore_diffusion = True
model.root.input.model.unit_001.particle_type_000.has_surface_diffusion = False
model.root.input.model.unit_001.particle_type_000.film_diffusion = np.array([6.9e-06,6.9e-06,6.9e-06])
model.root.input.model.unit_001.particle_type_000.pore_diffusion = np.array([7.0e-9,1.2e-11,1.2e-11])
model.root.input.model.unit_001.particle_type_000.init_cp = np.array([-1.78653518, 0.0, 0.0])
model.root.input.model.unit_001.particle_type_000.init_cs = np.array([0.0, 0.0])
model.root.input.model.unit_001.particle_type_000.par_porosity = np.float64(0.65)
model.root.input.model.unit_001.particle_type_000.par_radius = np.float64(2.4999999999999998e-05)

model.root.input.model.unit_001.particle_type_000.nbound = np.array([0, 1, 1])
model.root.input.model.unit_001.particle_type_000.adsorption_model = np.bytes_(b'AFFINITY_COMPLEX_TITRATION')
model.root.input.model.unit_001.particle_type_000.adsorption.act_etaa = np.array([0.0, 0.66071299, 0.66071299])
model.root.input.model.unit_001.particle_type_000.adsorption.act_etag = np.array([0.0, 2.29260803, 2.29260803])
model.root.input.model.unit_001.particle_type_000.adsorption.act_ka = np.array([1.00e+00, 1.26e+07, 4.20e+07])
model.root.input.model.unit_001.particle_type_000.adsorption.act_kd = np.array([1., 1., 1.])
model.root.input.model.unit_001.particle_type_000.adsorption.act_pkaa = np.array([ 0.0, -1.94530529, -1.5])
model.root.input.model.unit_001.particle_type_000.adsorption.act_pkag = np.array([ 0.0, -1.17006982, -1.1])
model.root.input.model.unit_001.particle_type_000.adsorption.act_qmax = np.array([1.00000000e-10, 1.19047619e-02, 1.19047619e-02])
model.root.input.model.unit_001.particle_type_000.adsorption.is_kinetic = np.int64(1)

model.root.input.model.unit_001.particle_type_000.discretization.spatial_method = "FV"
model.root.input.model.unit_001.particle_type_000.discretization.ncells = np.int64(10)
model.root.input.model.unit_001.particle_type_000.discretization.par_disc_type = np.bytes_(b'EQUIDISTANT_PAR')


model.root.input.model.unit_001.unit_type = np.bytes_(b'COLUMN_MODEL_1D')
model.root.input.model.unit_001.init_c = np.array([-1.78653518, 0.0, 0.0])
model.root.input.model.unit_001.ncomp = np.int64(3)
model.root.input.model.unit_001.col_dispersion = np.float64(5.75e-08)
model.root.input.model.unit_001.col_length = np.float64(0.1)
model.root.input.model.unit_001.col_porosity = np.float64(0.4)
model.root.input.model.unit_001.cross_section_area = np.float64(1.9634954084936207e-05)
model.root.input.model.unit_001.npartype = 1

model.root.input.model.unit_001.discretization.spatial_method = "FV"
model.root.input.model.unit_001.discretization.ncol = np.int64(40)
model.root.input.model.unit_001.discretization.reconstruction = np.bytes_(b'WENO')
model.root.input.model.unit_001.discretization.schur_safety = np.float64(1e-08)
model.root.input.model.unit_001.discretization.weno.boundary_model = np.int64(0)
model.root.input.model.unit_001.discretization.weno.weno_eps = np.float64(1e-10)
model.root.input.model.unit_001.discretization.weno.weno_order = np.int64(3)
model.root.input.model.unit_001.discretization.gs_type = np.int64(1)
model.root.input.model.unit_001.discretization.max_krylov = np.int64(0)
model.root.input.model.unit_001.discretization.max_restarts = np.int64(10)
model.root.input.model.unit_001.discretization.use_analytic_jacobian = np.int64(1)

model.root.input.model.unit_002.ncomp = np.int64(3)
model.root.input.model.unit_002.unit_type = np.bytes_(b'OUTLET')

model.root.input['return'].split_components_data = np.int64(1)
model.root.input['return'].split_ports_data = np.int64(0)
model.root.input['return'].unit_000.write_solution_bulk = np.int64(0)
model.root.input['return'].unit_000.write_solution_inlet = np.int64(0)
model.root.input['return'].unit_000.write_solution_outlet = np.int64(1)
model.root.input['return'].unit_001.write_solution_bulk = np.int64(0)
model.root.input['return'].unit_001.write_solution_inlet = np.int64(0)
model.root.input['return'].unit_001.write_solution_outlet = np.int64(1)
model.root.input['return'].unit_002.write_solution_bulk = np.int64(0)
model.root.input['return'].unit_002.write_solution_inlet = np.int64(0)
model.root.input['return'].unit_002.write_solution_outlet = np.int64(1)

model.root.input.solver.nthreads = np.int64(1)
model.root.input.solver.sections.nsec = np.int64(4)
model.root.input.solver.sections.section_continuity = np.array([0,0,0,0])
model.root.input.solver.sections.section_times = np.array([
    0.        ,  795.3805228 , 2369.53846154, 13474.15384615, 14774.0])
model.root.input.solver.time_integrator.init_step_size = np.float64(1e-12)
model.root.input.solver.time_integrator.max_convtest_fail = np.int64(50)
model.root.input.solver.time_integrator.max_errtest_fail = np.int64(7)
model.root.input.solver.time_integrator.max_steps = np.int64(1000000)
model.root.input.solver.time_integrator.reltol = np.float64(0.001)
model.root.input.solver.time_integrator.abstol = np.float64(0.0001)
model.root.input.solver.time_integrator.algtol = np.float64(1e-08)
model.root.input.solver.user_solution_times = np.linspace(0.0, 14774.0, 14774 + 1)

model.filename = r'C:\Users\jmbr\OneDrive\Desktop\flynn.h5'
model.save()


#%%

data = model.run_simulation()

if not data.return_code == 0:
    print("error: ")
    print(data.error_message)
    print(data.log)

#%%

import matplotlib.pyplot as plt

model.load()

outlet_comp1 = model.root.output.solution.unit_002.solution_outlet_comp_001
outlet_comp2 = model.root.output.solution.unit_002.solution_outlet_comp_002
time = model.root.output.solution.solution_times



plt.plot(time, outlet_comp1, label='comp1')
plt.plot(time, outlet_comp2, label='comp2')
plt.legend()
plt.show()






