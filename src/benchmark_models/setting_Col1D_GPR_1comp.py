# -*- coding: utf-8 -*-
"""

This script defines an axial 1D transport model, 1 component, no particles

"""

import numpy as np
from addict import Dict


def get_GPR_kernel(kernel:str):
    
    config = Dict()
    config.adsorption_model = np.bytes_(b'GAUSSIAN_PROCESS_REGRESSION')

    if kernel == "Shallow_MLP_7":
        
        config.adsorption.kernel = np.bytes_(b'MLP')

        config.nbound = np.array([1])
        config.adsorption.is_kinetic = np.int32(1)
        
        config.adsorption.gpr_kkin = np.int32(1)
        config.adsorption.cp_ndim = np.int32(1)
        config.adsorption.cs_ndim = np.int32(1)
        config.adsorption.cp_vals = np.array(
            [ 0.0, 0.19219219, 0.37837838, 0.75075075, 1.5015015 , 3.003003 , 6.0 ]
            )
        config.adsorption.cs_vals = np.array(
            [ 1.29198742, 76.84778516,110.14325476,141.59781958,165.80146634,
              181.33282173,190.23846417 ]
            )

        config.adsorption.MLP_WEIGHT_VARIANCE = np.array([1.86490637e+01])
        config.adsorption.MLP_BIAS_VARIANCE = np.array([3.31357112e-04])
        config.adsorption.MLP_VARIANCE = np.array([7.89170771e+03])
        config.adsorption.GAUSSIAN_NOISE_VARIANCE = np.array([1.00000000e+00])
        
    elif kernel == "Shallow_RBF_15":
        
        config.adsorption.kernel = np.bytes_(b'RBF_Linear')
        
        config.nbound = np.array([1])
        config.adsorption.is_kinetic = np.int32(1)

        config.adsorption.gpr_kkin = np.int32(1)
        config.adsorption.cp_ndim = np.int32(1)
        config.adsorption.cs_ndim = np.int32(1)
        config.adsorption.cp_vals = np.array(
            [ 0.0, 0.42857143, 0.85714286, 1.28571429, 1.71428571, 2.14285714,
              2.57142857, 3.0,3.42857143, 3.85714286, 4.28571429, 4.71428571,
              5.14285714, 5.57142857, 6.0 ]
            )
        config.adsorption.cs_vals = np.array(
            [ 0.0,116.26794275,147.04992449,161.28318594,169.48561458,
              174.82014386,178.56705449,181.34328358,183.4827749 ,185.18204912,
              186.56429944,187.71067415,188.67680362,189.50209958,190.21526419 ]
            )
        
        config.adsorption.LINEAR_VARIANCE = np.array([6.89934599e+00])
        config.adsorption.RBF_VARIANCE = np.array([1.52543530e+04])
        config.adsorption.RBF_LENGTHSCALE = np.array([9.32978701e-01])
        config.adsorption.GAUSSIAN_NOISE_VARIANCE = np.array([1.47338643e+03])
        
    else:
        raise Exception("unsupported GPR binding kernel")
    
    return config

def get_model(kernel):
    
    if kernel not in ["Shallow_RBF_15", "Shallow_MLP_7"]:
        raise Exception("unsupported GPR binding kernel")
    
    config = Dict()
    
    config.model.connections.nswitches = np.int32(1)
    config.model.connections.switch_000.connections = np.array(
        [ 0.0000e+00, 1.0000e+00,-1.0000e+00,-1.0000e+00, 4.1667e-08,
          1.0000e+00, 2.0000e+00,-1.0000e+00,-1.0000e+00, 4.1667e-08 ]
        )
    config.model.connections.switch_000.section = np.int32(0)
    
    config.model.nunits = np.int32(3)
    
    config.model.unit_000.unit_type = np.bytes_(b'INLET')
    config.model.unit_000.inlet_type = np.bytes_(b'PIECEWISE_CUBIC_POLY')
    config.model.unit_000.ncomp = np.int32(1)
    config.model.unit_000.sec_000.const_coeff = np.array([5])
    config.model.unit_000.sec_000.cube_coeff = np.array([0.])
    config.model.unit_000.sec_000.lin_coeff = np.array([0.])
    config.model.unit_000.sec_000.quad_coeff = np.array([0.])
    config.model.unit_000.sec_001.const_coeff = np.array([5])
    config.model.unit_000.sec_001.cube_coeff = np.array([0.])
    config.model.unit_000.sec_001.lin_coeff = np.array([0.])
    config.model.unit_000.sec_001.quad_coeff = np.array([0.])
    
    config.model.unit_001.unit_type = np.bytes_(b'COLUMN_MODEL_1D')
    config.model.unit_001.ncomp = np.int32(1)
    config.model.unit_001.col_dispersion = np.float64(0.0)
    config.model.unit_001.col_length = np.float64(0.1)
    config.model.unit_001.npartype = 1
    config.model.unit_001.col_porosity = np.float64(0.25)
    config.model.unit_001.cross_section_area = np.float64(9.503317777109126e-05)
    config.model.unit_001.init_c = np.array([0.])
    # Discretization
    config.model.unit_001.discretization.spatial_method = "FV"
    config.model.unit_001.discretization.reconstruction = np.bytes_(b'WENO')
    config.model.unit_001.discretization.ncol = np.int32(100)
    config.model.unit_001.discretization.use_analytic_jacobian = np.int32(1)
    config.model.unit_001.discretization.weno.boundary_model = np.int32(0)
    config.model.unit_001.discretization.weno.weno_eps = np.float64(1e-10)
    config.model.unit_001.discretization.weno.weno_order = np.int32(3)
    config.model.unit_001.discretization.gs_type = np.int32(1)
    config.model.unit_001.discretization.max_krylov = np.int32(0)
    config.model.unit_001.discretization.max_restarts = np.int32(10)
    config.model.unit_001.discretization.schur_safety = np.float64(1e-08)

    # Particle
    config.model.unit_001.particle_type_000.par_porosity = np.float64(0.69)
    config.model.unit_001.particle_type_000.par_radius = np.float64(4.625e-05)
    config.model.unit_001.particle_type_000.has_film_diffusion = True
    config.model.unit_001.particle_type_000.has_pore_diffusion = True
    config.model.unit_001.particle_type_000.has_surface_diffusion = False
    config.model.unit_001.particle_type_000.film_diffusion = np.array([6.38748621e-06])
    config.model.unit_001.particle_type_000.pore_diffusion = np.array([5.50724638e-12])
    config.model.unit_001.particle_type_000.init_cp = np.array([0.])
    config.model.unit_001.particle_type_000.init_cs = np.array([0.])
    
    # Binding
    config.model.unit_001.particle_type_000.update(get_GPR_kernel(kernel))
    
    # Discretization
    config.model.unit_001.particle_type_000.discretization.spatial_method = "FV"
    config.model.unit_001.particle_type_000.discretization.ncells = np.int32(15)
    config.model.unit_001.particle_type_000.discretization.par_disc_type = np.bytes_(b'EQUIDISTANT_PAR')

    
    config.model.unit_002.ncomp = np.int32(1)
    config.model.unit_002.unit_type = np.bytes_(b'OUTLET')
    
    config['return'].split_components_data = np.int32(0)
    config['return'].split_ports_data = np.int32(0)
    # config['return'].unit_000.write_coordinates = np.int32(1)
    # config['return'].unit_000.write_sens_outlet = np.int32(1)
    # config['return'].unit_000.write_solution_bulk = np.int32(1)
    # config['return'].unit_000.write_solution_flux = np.int32(1)
    # config['return'].unit_000.write_solution_inlet = np.int32(1)
    # config['return'].unit_000.write_solution_outlet = np.int32(1)
    # config['return'].unit_000.write_solution_particle = np.int32(1)
    # config['return'].unit_000.write_solution_solid = np.int32(1)
    # config['return'].unit_000.write_solution_volume = np.int32(1)
    # config['return'].unit_001.write_coordinates = np.int32(1)
    # config['return'].unit_001.write_sens_outlet = np.int32(1)
    # config['return'].unit_001.write_solution_bulk = np.int32(1)
    # config['return'].unit_001.write_solution_flux = np.int32(1)
    # config['return'].unit_001.write_solution_inlet = np.int32(1)
    config['return'].unit_001.write_solution_outlet = np.int32(1)
    # config['return'].unit_001.write_solution_particle = np.int32(1)
    # config['return'].unit_001.write_solution_solid = np.int32(1)
    # config['return'].unit_001.write_solution_volume = np.int32(1)
    # config['return'].unit_002.write_coordinates = np.int32(1)
    # config['return'].unit_002.write_sens_outlet = np.int32(1)
    # config['return'].unit_002.write_solution_bulk = np.int32(1)
    # config['return'].unit_002.write_solution_flux = np.int32(1)
    # config['return'].unit_002.write_solution_inlet = np.int32(1)
    # config['return'].unit_002.write_solution_outlet = np.int32(1)
    # config['return'].unit_002.write_solution_particle = np.int32(1)
    # config['return'].unit_002.write_solution_solid = np.int32(1)
    # config['return'].unit_002.write_solution_volume = np.int32(1)
    
    config.solver.nthreads = np.int32(1)
    
    tEnd = 5.0e+03
    config.solver.sections.nsec = np.int32(2)
    config.solver.sections.section_continuity = np.array([0,0])
    config.solver.sections.section_times = np.array([0.0e+00, 5.0e+00, tEnd])
    
    config.model.solver.gs_type = np.int32(1)
    config.model.solver.max_krylov = np.int32(0)
    config.model.solver.max_restarts = np.int32(10)
    config.model.solver.schur_safety = np.float64(1e-08)
    
    config.solver.time_integrator.abstol = np.float64(1e-06)
    config.solver.time_integrator.algtol = np.float64(1e-10)
    config.solver.time_integrator.init_step_size = np.float64(1e-06)
    config.solver.time_integrator.max_steps = np.int32(1000000)
    config.solver.time_integrator.reltol = np.float64(1e-06)
    config.solver.user_solution_times = np.linspace(0.0, tEnd, int(tEnd) + 1)
    
    
    return config
