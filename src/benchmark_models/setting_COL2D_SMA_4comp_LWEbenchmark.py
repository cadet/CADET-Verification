# -*- coding: utf-8 -*-
"""
Created Jul 2025

@author: jmbr
"""

import numpy as np
from addict import Dict


def get_model():
    
    model = Dict()
    
    model.input.model.nunits = 2
    
    # Flow sheet
    model.input.model.connections.connections_include_ports = 1
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [ 1., 0.,-1.,-1.,-1.,-1., 1.]
    model.input.model.connections.switch_000.section = 0
    
    # Column unit
    model.input.model.unit_000.unit_type = b'GENERAL_RATE_MODEL'
    model.input.model.unit_000.col_length = 0.014
    model.input.model.unit_000.col_radius = 0.01
    model.input.model.unit_000.col_porosity = 0.37
    model.input.model.unit_000.total_porosity = 0.8425
    
    model.input.model.unit_000.ncomp = 4
    model.input.model.unit_000.init_q = [1200.,   0.,   0.,   0.]
    model.input.model.unit_000.init_c = [50., 0., 0., 0.]
    model.input.model.unit_000.col_dispersion = 5.75e-08
    model.input.model.unit_000.col_dispersion_radial = 1e-06
    model.input.model.unit_000.velocity = 0.000575
    
    model.input.model.unit_000.par_radius = 4.5e-05
    model.input.model.unit_000.par_coreradius = 0.0
    model.input.model.unit_000.par_porosity = 0.75
    model.input.model.unit_000.film_diffusion = [6.9e-06,6.9e-06,6.9e-06,6.9e-06]
    model.input.model.unit_000.par_diffusion = [7.00e-10,6.07e-11,6.07e-11,6.07e-11]
    model.input.model.unit_000.par_surfdiffusion = [0.,0.,0.,0.]
    
    model.input.model.unit_000.adsorption.is_kinetic = 0
    model.input.model.unit_000.adsorption.sma_ka = [ 0.0, 35.5, 1.59, 7.7 ]
    model.input.model.unit_000.adsorption.sma_kd = [ 0.0, 1000.0, 1000.0, 1000.0]
    model.input.model.unit_000.adsorption.sma_lambda = 1200.0
    model.input.model.unit_000.adsorption.sma_nu = [0.  ,4.7 ,5.29,3.7 ]
    model.input.model.unit_000.adsorption.sma_sigma = [ 0.  ,11.83,10.6 ,10.  ]
    model.input.model.unit_000.adsorption_model = b'STERIC_MASS_ACTION'
    
    model.input.model.unit_000.discretization.gs_type = 1
    model.input.model.unit_000.discretization.max_krylov = 0
    model.input.model.unit_000.discretization.max_restarts = 10
    model.input.model.unit_000.discretization.nbound = [1,1,1,1]
    model.input.model.unit_000.discretization.nelem = 8
    model.input.model.unit_000.discretization.par_disc_type = [b'EQUIDISTANT_PAR']
    model.input.model.unit_000.discretization.par_geom = [b'SPHERE']
    model.input.model.unit_000.discretization.par_nelem = 1
    model.input.model.unit_000.discretization.par_polydeg = 3
    model.input.model.unit_000.discretization.polydeg = 3
    model.input.model.unit_000.discretization.reconstruction = b'WENO'
    model.input.model.unit_000.discretization.schur_safety = 1e-08
    model.input.model.unit_000.discretization.spatial_method = b'DG'
    model.input.model.unit_000.discretization.use_analytic_jacobian = 1
    
    # Inlet / Feed unit
    model.input.model.unit_001.inlet_type = b'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_001.ncomp = 4
    model.input.model.unit_001.sec_000.const_coeff = [50., 1., 1., 1.]
    model.input.model.unit_001.sec_000.cube_coeff = [0.,0.,0.,0.]
    model.input.model.unit_001.sec_000.lin_coeff = [0.,0.,0.,0.]
    model.input.model.unit_001.sec_000.quad_coeff = [0.,0.,0.,0.]
    model.input.model.unit_001.sec_001.const_coeff = [50., 0., 0., 0.]
    model.input.model.unit_001.sec_001.cube_coeff = [0.,0.,0.,0.]
    model.input.model.unit_001.sec_001.lin_coeff = [0.,0.,0.,0.]
    model.input.model.unit_001.sec_001.quad_coeff = [0.,0.,0.,0.]
    model.input.model.unit_001.sec_002.const_coeff = [100.,  0.,  0.,  0.]
    model.input.model.unit_001.sec_002.cube_coeff = [0.,0.,0.,0.]
    model.input.model.unit_001.sec_002.lin_coeff = [0.2,0. ,0. ,0. ]
    model.input.model.unit_001.sec_002.quad_coeff = [0.,0.,0.,0.]
    model.input.model.unit_001.unit_type = 'INLET'

    # Global system solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    # Time integration / solver
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 3
    model.input.solver.sections.section_continuity = [0, 0]
    model.input.solver.sections.section_times = [   0.,  10.,  90.,1500.]
    model.input.solver.time_integrator.abstol = 1e-07
    model.input.solver.time_integrator.algtol = 9.999999999999999e-06
    model.input.solver.time_integrator.init_step_size = 1e-10
    model.input.solver.time_integrator.max_steps = 10000
    model.input.solver.time_integrator.reltol = 9.999999999999999e-06
    
    # Return data
    model.input.solver.user_solution_times = np.linspace(0, 1500, 1501)
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_coordinates = 0
    model.input['return'].unit_000.write_sens_bulk = 0
    model.input['return'].unit_000.write_sens_last = 0
    model.input['return'].unit_000.write_sens_outlet = 1
    model.input['return'].unit_000.write_solution_bulk = 0
    model.input['return'].unit_000.write_solution_flux = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].unit_000.write_solution_particle = 0
    model.input['return'].unit_000.write_solution_solid = 0
    model.input['return'].write_solution_times = 1
    
    return model


from cadet import Cadet

model = Cadet()
model.root = get_model()
model.filename = "jojo.h5"
model.save()
model.run()


