# -*- coding: utf-8 -*-
"""
Created June 2025

@author: jmbr
"""

import numpy as np
from addict import Dict

def get_model():

    model = Dict()
    
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = np.array(
        [0, 1, -1, -1, 6.e-05,
         1, 2, -1, -1, 6.e-05]
        )
    model.input.model.connections.switch_000.section = 0
    model.input.model.nunits = 3
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08
    
    
    # Inlet unit
    model.input.model.unit_000.unit_type = 'INLET'
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = 2
    model.input.model.unit_000.sec_000.const_coeff = np.array([1., 1.])
    model.input.model.unit_000.sec_000.cube_coeff = np.array([0., 0.])
    model.input.model.unit_000.sec_000.lin_coeff = np.array([0., 0.])
    model.input.model.unit_000.sec_000.quad_coeff = np.array([0., 0.])
    model.input.model.unit_000.sec_001.const_coeff = np.array([0., 0.])
    model.input.model.unit_000.sec_001.cube_coeff = np.array([0., 0.])
    model.input.model.unit_000.sec_001.lin_coeff = np.array([0., 0.])
    model.input.model.unit_000.sec_001.quad_coeff = np.array([0., 0.])
    
    # Column unit
    model.input.model.unit_001.unit_type = 'COLUMN_MODEL_1D'
    model.input.model.unit_001.ncomp = 2
    model.input.model.unit_001.velocity = 5.75e-4
    model.input.model.unit_001.col_dispersion = [5.75e-8]
    model.input.model.unit_001.col_length = 0.014
    model.input.model.unit_001.col_porosity = 0.37
    model.input.model.unit_001.col_porosity = 0.37 + (1.0 - 0.37) * (0.6 * 0.75 + 0.4 * 0.5)
    model.input.model.unit_001.init_c = np.array([0., 0.])
    model.input.model.unit_001.init_cp = np.array([0., 0.])
    model.input.model.unit_001.init_q = np.array([0., 0.])
    
    # Spatial discretization unit level
    model.input.model.unit_001.discretization.USE_ANALYTIC_JACOBIAN = True
    model.input.model.unit_001.discretization.SPATIAL_METHOD = 'DG'
    model.input.model.unit_001.discretization.POLYDEG = 3
    model.input.model.unit_001.discretization.NELEM = 10
    model.input.model.unit_001.discretization.EXACT_INTEGRATION = 0
    # model.input.model.unit_001.discretization.GS_TYPE = 0
    # model.input.model.unit_001.discretization.MAX_KRYLOV = 10
    # model.input.model.unit_001.discretization.MAX_RESTARTS = 100
    # model.input.model.unit_001.discretization.SCHUR_SAFETY = 0.1
    
    
    # Particles
    model.input.model.unit_001.npartype = 2
    model.input.model.unit_001.par_type_volfrac = [0.6, 0.4]
    
    
    # Particle type1: General rate particle
    model.input.model.unit_001.particle_type_000.particle_type = "GENERAL_RATE_PARTICLE"
    model.input.model.unit_001.particle_type_000.par_geom = ['SPHERE']
    model.input.model.unit_001.particle_type_000.par_coreradius = 0.0
    model.input.model.unit_001.particle_type_000.par_diffusion = [
        6.07e-11, 1e-11, # section0: comp0, comp1
        2*6.07e-11, 2*1e-11 # section1: comp0, comp1
        ]
    model.input.model.unit_001.particle_type_000.par_diffusion_multiplex = 1 # section, component
    model.input.model.unit_001.particle_type_000.par_diffusion_partype_dependent = 1
    model.input.model.unit_001.particle_type_000.film_diffusion = [6.9e-06, 2e-06]
    model.input.model.unit_001.particle_type_000.film_diffusion_multiplex = 1 # component and section
    model.input.model.unit_001.particle_type_000.film_diffusion_partype_dependent = 1
    
    model.input.model.unit_001.particle_type_000.par_porosity = 0.75
    model.input.model.unit_001.particle_type_000.par_radius = 4.5e-05
    model.input.model.unit_001.particle_type_000.par_surfdiffusion = [
        5e-11, 2e-11
        ]
    model.input.model.unit_001.particle_type_000.par_surfdiffusion_multiplex = 1 # component
    model.input.model.unit_001.particle_type_000.par_surfdiffusion_partype_dependent = 1
    
    # Binding
    model.input.model.unit_001.particle_type_000.adsorption_model = 'LINEAR'
    model.input.model.unit_001.particle_type_000.nbound = np.array([1, 1])
    model.input.model.unit_001.particle_type_000.adsorption.is_kinetic = True
    model.input.model.unit_001.particle_type_000.adsorption.lin_ka = np.array([3.55, 2.0])
    model.input.model.unit_001.particle_type_000.adsorption.lin_kd = np.array([0.1, 1.0])
    model.input.model.unit_001.particle_type_000.binding_partype_dependent = 1

    # Spatial discretization
    model.input.model.unit_001.particle_type_000.discretization.PAR_DISC_TYPE = ['EQUIDISTANT_PAR']
    model.input.model.unit_001.particle_type_000.discretization.SPATIAL_METHOD = "DG"
    model.input.model.unit_001.particle_type_000.discretization.PAR_POLYDEG = 3
    model.input.model.unit_001.particle_type_000.discretization.PAR_NELEM = 1


    # Particle type2: Homogeneous particle
    model.input.model.unit_001.particle_type_001.particle_type = "HOMOGENEOUS_PARTICLE"
    model.input.model.unit_001.particle_type_001.par_geom = ['SPHERE']
    model.input.model.unit_001.particle_type_001.film_diffusion = [1.0e-07, 4e-06]
    model.input.model.unit_001.particle_type_001.film_diffusion_multiplex = 1 # component and section
    model.input.model.unit_001.particle_type_001.film_diffusion_partype_dependent = 1
    model.input.model.unit_001.particle_type_001.par_porosity = 0.5
    model.input.model.unit_001.particle_type_001.par_radius = 5.5e-05
    
    # Binding
    model.input.model.unit_001.particle_type_001.adsorption_model = 'LINEAR'
    model.input.model.unit_001.particle_type_001.nbound = np.array([1, 1])
    model.input.model.unit_001.particle_type_001.adsorption.is_kinetic = True
    model.input.model.unit_001.particle_type_001.adsorption.lin_ka = np.array([4.55, 2.5])
    model.input.model.unit_001.particle_type_001.adsorption.lin_kd = np.array([0.2, 1.0])
    model.input.model.unit_001.particle_type_001.binding_partype_dependent = 1
    
    
    # Outlet unit
    model.input.model.unit_002.ncomp = 2
    model.input.model.unit_002.unit_type = 'OUTLET'
    
    
    # Return data
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_solution_bulk = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_coordinates = 0
    model.input['return'].unit_001.write_sens_bulk = 0
    model.input['return'].unit_001.write_sens_last = 0
    model.input['return'].unit_001.write_sens_outlet = 1
    model.input['return'].unit_001.write_solution_bulk = 0
    model.input['return'].unit_001.write_solution_inlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_001.write_solution_particle = 0
    model.input['return'].unit_002.write_solution_bulk = 0
    model.input['return'].unit_002.write_solution_inlet = 0
    model.input['return'].unit_002.write_solution_outlet = 0
    
    # Solver specifics
    model.input.solver.consistent_init_mode = 1
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = np.array([0])
    model.input.solver.sections.section_times = np.array([0.,  10., 1500.])
    model.input.solver.time_integrator.ABSTOL = 1e-8
    # model.input.solver.time_integrator.SENS_ABSTOL = 1e-4
    model.input.solver.time_integrator.ALGTOL = 1e-08
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.time_integrator.RELTOL = 1e-08
    model.input.solver.user_solution_times = np.linspace(0, 1500, 6001)

    return model


from cadet import Cadet
from cadetrdm import ProjectRepo
import matplotlib.pyplot as plt

cadet_path = r'C:/Users\jmbr\Cadet_testBuild\CADET_2DmodelsDG\out\install\aRELEASE\bin\cadet-cli.exe'

project_repo = ProjectRepo()
output_path = str(project_repo.output_path / "test_cadet-core")
run_simulation=1
plot_result=1

Cadet.cadet_path = cadet_path

model = Cadet()
model.root = get_model()
model.filename = output_path + '/COL1D_2parTypeMixed_2comp_benchmark1.h5'
model.save()

if run_simulation:
    data = model.run()
    if not data.return_code == 0:
        print(data.error_message)
        raise Exception(f"simulation failed")

    model.load()  
    
    time = model.root.output.solution.solution_times
    solution = model.root.output.solution.unit_001.solution_outlet
        
    if plot_result:
        plt.plot(time, solution[:,0], c="blue", label="comp0")
        plt.plot(time, solution[:,1], c="red", label="comp1")
        plt.legend()
        plt.savefig(output_path + '/COL1D_2parTypeMixed_2comp_benchmark1.png')
        plt.show()




