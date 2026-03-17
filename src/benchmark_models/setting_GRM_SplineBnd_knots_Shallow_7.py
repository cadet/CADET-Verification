# -*- coding: utf-8 -*-
"""

This script implements a spline based data-driven binding setting

""" 

import numpy as np
import matplotlib.pyplot as plt
import os

from cadet import Cadet

def get_model(cadet_path, output_path, run_simulation, plot_result):
    
    model = Cadet()
    
    model.install_path = cadet_path
    
    model.root.input.model.connections.nswitches = np.int32(1)
    model.root.input.model.connections.switch_000.connections = np.array([
        0.0000e+00, 1.0000e+00,-1.0000e+00,-1.0000e+00, 4.1667e-08,
        1.0000e+00, 2.0000e+00,-1.0000e+00,-1.0000e+00, 4.1667e-08]
        )
    model.root.input.model.connections.switch_000.section = np.int32(0)
    model.root.input.model.nunits = np.int32(3)
    model.root.input.model.solver.gs_type = np.int32(1)
    model.root.input.model.solver.max_krylov = np.int32(0)
    model.root.input.model.solver.max_restarts = np.int32(10)
    model.root.input.model.solver.schur_safety = np.float64(1e-08)
    
    
    model.root.input.model.unit_000.unit_type = np.bytes_(b'INLET')
    model.root.input.model.unit_000.inlet_type = np.bytes_(b'PIECEWISE_CUBIC_POLY')
    model.root.input.model.unit_000.ncomp = np.int32(1)
    model.root.input.model.unit_000.sec_000.const_coeff = np.array([5])
    model.root.input.model.unit_000.sec_000.cube_coeff = np.array([0.])
    model.root.input.model.unit_000.sec_000.lin_coeff = np.array([0.])
    model.root.input.model.unit_000.sec_000.quad_coeff = np.array([0.])
    model.root.input.model.unit_000.sec_001.const_coeff = np.array([5])
    model.root.input.model.unit_000.sec_001.cube_coeff = np.array([0.])
    model.root.input.model.unit_000.sec_001.lin_coeff = np.array([0.])
    model.root.input.model.unit_000.sec_001.quad_coeff = np.array([0.])
    
    
    model.root.input.model.unit_001.unit_type = np.bytes_(b'COLUMN_MODEL_1D')
    model.root.input.model.unit_001.npartype = np.int32(1)
    model.root.input.model.unit_001.ncomp = np.int32(1)
    model.root.input.model.unit_001.init_c = np.array([0.])
    model.root.input.model.unit_001.col_dispersion = np.float64(0.0)
    model.root.input.model.unit_001.col_length = np.float64(0.1)
    model.root.input.model.unit_001.col_porosity = np.float64(0.25)
    model.root.input.model.unit_001.cross_section_area = np.float64(9.503317777109126e-05)
    model.root.input.model.unit_001.discretization.gs_type = np.int32(1)
    model.root.input.model.unit_001.discretization.max_krylov = np.int32(0)
    model.root.input.model.unit_001.discretization.max_restarts = np.int32(10)
    model.root.input.model.unit_001.discretization.spatial_method = np.bytes_(b'FV')
    model.root.input.model.unit_001.discretization.ncol = np.int32(100)
    model.root.input.model.unit_001.discretization.reconstruction = np.bytes_(b'WENO')
    model.root.input.model.unit_001.discretization.schur_safety = np.float64(1e-08)
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = np.int32(1)
    model.root.input.model.unit_001.discretization.weno.boundary_model = np.int32(0)
    model.root.input.model.unit_001.discretization.weno.weno_eps = np.float64(1e-10)
    model.root.input.model.unit_001.discretization.weno.weno_order = np.int32(3)
    
    model.root.input.model.unit_001.particle_type_000.has_film_diffusion = np.int32(1)
    model.root.input.model.unit_001.particle_type_000.has_pore_diffusion = np.int32(1)
    model.root.input.model.unit_001.particle_type_000.has_surface_diffusion = np.int32(0)
    model.root.input.model.unit_001.particle_type_000.film_diffusion = np.array([6.38748621e-06])
    model.root.input.model.unit_001.particle_type_000.pore_diffusion = np.array([5.50724638e-12])
    model.root.input.model.unit_001.particle_type_000.par_porosity = np.float64(0.69)
    model.root.input.model.unit_001.particle_type_000.par_radius = np.float64(4.625e-05)
    model.root.input.model.unit_001.particle_type_000.init_cs = np.array([0.])
    model.root.input.model.unit_001.particle_type_000.init_cp = np.array([0.])
    model.root.input.model.unit_001.particle_type_000.nbound = np.array([1])
    model.root.input.model.unit_001.particle_type_000.adsorption_model = np.bytes_(b'SPLINE')
    model.root.input.model.unit_001.particle_type_000.adsorption.is_kinetic = np.int32(1)
    model.root.input.model.unit_001.particle_type_000.adsorption.ml_kkin = np.float64(1.0)
    model.root.input.model.unit_001.particle_type_000.adsorption.spline_model_parameters.c_vals_comp_000 = np.array([
        0.0, 0.19219219, 0.37837838, 0.75075075, 1.5015015 , 3.003003, 6.0
        ])
    model.root.input.model.unit_001.particle_type_000.adsorption.spline_model_parameters.cs_vals_comp_000_bnd_000 = np.array([
        1.29198742, 76.84778516, 110.14325476, 141.59781958, 165.80146634, 181.33282173, 190.23846417
        ])
    model.root.input.model.unit_001.particle_type_000.discretization.spatial_method = np.bytes_(b'FV')
    model.root.input.model.unit_001.particle_type_000.discretization.ncells = np.int32(15)
    model.root.input.model.unit_001.particle_type_000.discretization.par_disc_type = np.bytes_(b'EQUIDISTANT_PAR')
    
    
    model.root.input.model.unit_002.unit_type = np.bytes_(b'OUTLET')
    model.root.input.model.unit_002.ncomp = np.int32(1)
    
    
    model.root.input['return'].split_components_data = np.int32(0)
    model.root.input['return'].split_ports_data = np.int32(0)
    # model.root.input['return'].unit_000.write_coordinates = np.int32(1)
    # model.root.input['return'].unit_000.write_sens_outlet = np.int32(1)
    # model.root.input['return'].unit_000.write_solution_bulk = np.int32(1)
    # model.root.input['return'].unit_000.write_solution_flux = np.int32(1)
    # model.root.input['return'].unit_000.write_solution_inlet = np.int32(1)
    # model.root.input['return'].unit_000.write_solution_outlet = np.int32(1)
    # model.root.input['return'].unit_000.write_solution_particle = np.int32(1)
    # model.root.input['return'].unit_000.write_solution_solid = np.int32(1)
    # model.root.input['return'].unit_000.write_solution_volume = np.int32(1)
    # model.root.input['return'].unit_001.write_coordinates = np.int32(1)
    # model.root.input['return'].unit_001.write_sens_outlet = np.int32(1)
    # model.root.input['return'].unit_001.write_solution_bulk = np.int32(1)
    # model.root.input['return'].unit_001.write_solution_flux = np.int32(1)
    # model.root.input['return'].unit_001.write_solution_inlet = np.int32(1)
    model.root.input['return'].unit_001.write_solution_outlet = np.int32(1)
    # model.root.input['return'].unit_001.write_solution_particle = np.int32(1)
    # model.root.input['return'].unit_001.write_solution_solid = np.int32(1)
    # model.root.input['return'].unit_001.write_solution_volume = np.int32(1)
    # model.root.input['return'].unit_002.write_coordinates = np.int32(1)
    # model.root.input['return'].unit_002.write_sens_outlet = np.int32(1)
    # model.root.input['return'].unit_002.write_solution_bulk = np.int32(1)
    # model.root.input['return'].unit_002.write_solution_flux = np.int32(1)
    # model.root.input['return'].unit_002.write_solution_inlet = np.int32(1)
    # model.root.input['return'].unit_002.write_solution_outlet = np.int32(1)
    # model.root.input['return'].unit_002.write_solution_particle = np.int32(1)
    # model.root.input['return'].unit_002.write_solution_solid = np.int32(1)
    # model.root.input['return'].unit_002.write_solution_volume = np.int32(1)
    
    
    model.root.input.solver.nthreads = np.int32(1)
    model.root.input.solver.sections.nsec = np.int32(2)
    model.root.input.solver.sections.section_continuity = np.array([0,0])
    model.root.input.solver.sections.section_times = np.array([0.0e+00,5.0e+00,7.2e+03])
    
    model.root.input.solver.time_integrator.abstol = np.float64(1e-12)
    model.root.input.solver.time_integrator.algtol = np.float64(1e-8)
    model.root.input.solver.time_integrator.init_step_size = np.float64(1e-06)
    model.root.input.solver.time_integrator.max_steps = np.int32(1000000)
    model.root.input.solver.time_integrator.reltol = np.float64(1e-06)
    model.root.input.solver.user_solution_times = np.linspace(0.0, 7200.0, 601)
    
    if run_simulation:
        
        model.filename = os.path.join(output_path, 'GRM_SplineBnd_knots_Shallow_7.h5')
        model.save()
        return_data = model.run_simulation()
        
        if not return_data.return_code == 0:
            raise Exception(f"simulation failed with {return_data.error_message}\n and LOG:\n {return_data.log}")
        
        if plot_result:
    
            time = model.root.output.solution.solution_times
            outlet = model.root.output.solution.unit_001.solution_outlet
            
            plt.plot(time, outlet, label='Spline Binding')
            plt.xlabel(r'Time, $min$')
            plt.ylabel(r'Concentration, $g/L$')
            plt.legend(frameon=0)
            plt.savefig(output_path + '/GRM_SplineBnd_knots_Shallow_7.png')
            plt.show()
            plt.close()
        
    return model

#%% Simulation result comparison to reference solution

# cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

# model = get_model(cadet_path, save_model=True)

# data = model.run_simulation()

# if not data.return_code == 0:
#     print("Simulation failed with: " + data.error_message)
#     print(data.log)
# else:
    
#     model.load()
    
#     # data
    
#     time = model.root.output.solution.solution_times
#     outlet = model.root.output.solution.unit_001.solution_outlet
    
#     model_old = Cadet()
#     model_old.filename = r'C:\Users\jmbr\OneDrive\Desktop\Hybrid models\2026 Dev Work\Input Files\Splines\Spline_knots_Shallow_7.h5'
#     model_old.load_from_file()
#     outlet_old = model_old.root.output.solution.unit_001.solution_outlet
    
#     plt.plot(time, outlet, label='new sim')
#     plt.plot(time, outlet_old, label='old sim', linestyle='dashed')
#     plt.xlabel(r'Time, $min$')
#     plt.ylabel(r'Concentration, $g/L$')
#     plt.legend(frameon=0)
#     plt.show()
    
#     print("absolute max. difference: " + str(np.max(abs(outlet_old - outlet))))
