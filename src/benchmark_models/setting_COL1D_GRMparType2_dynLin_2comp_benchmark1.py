import urllib
import json
import re
import numpy as np
from addict import Dict


def get_model(spatial_method_bulk, spatial_method_particle):

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
    model.input.model.unit_001.npartype = 2
    model.input.model.unit_001.par_type_volfrac = [0.4, 0.6]
    model.input.model.unit_001.velocity = 0.000575
    model.input.model.unit_001.col_dispersion = [5.75e-08]
    model.input.model.unit_001.col_length = 0.014
    model.input.model.unit_001.col_porosity = 0.37
    model.input.model.unit_001.init_c = np.array([0., 0.])
    
    # Particle - Type 0
    model.input.model.unit_001.particle_type_000.has_film_diffusion = 1
    model.input.model.unit_001.particle_type_000.has_pore_diffusion = 1
    model.input.model.unit_001.particle_type_000.has_surface_diffusion = 1
    model.input.model.unit_001.particle_type_000.par_geom = 'SPHERE'
    model.input.model.unit_001.particle_type_000.film_diffusion = [6.9e-06, 2e-06, 0.1*6.9e-06, 0.1*2e-06]
    model.input.model.unit_001.particle_type_000.film_diffusion_multiplex = 1 # component and section dependent
    model.input.model.unit_001.particle_type_000.film_diffusion_partype_dependent = 0 # component and section dependent
    model.input.model.unit_001.particle_type_000.par_coreradius = 0.0
    model.input.model.unit_001.particle_type_000.pore_diffusion = [
        2*6.07e-11, 2*1e-11, # section0: type0comp0, type0comp1,
        2*2*6.07e-11, 2*2*1e-11 # section1: type0comp0, type0comp1,
        ]
    model.input.model.unit_001.particle_type_000.pore_diffusion_multiplex = 1 # section, parType and component dependent
    model.input.model.unit_001.particle_type_000.pore_diffusion_partype_dependent = 1
    model.input.model.unit_001.particle_type_000.par_porosity = 0.75
    model.input.model.unit_001.particle_type_000.par_radius = 4.5e-05
    model.input.model.unit_001.particle_type_000.surface_diffusion = [
        5e-11, 2e-11 # type0comp0, type0comp1
        ]
    model.input.model.unit_001.particle_type_000.surface_diffusion_multiplex = 0 # parType and component dependent
    model.input.model.unit_001.particle_type_000.surface_diffusion_partype_dependent = 1
    model.input.model.unit_001.particle_type_000.init_cp = np.array([0., 0.])
    model.input.model.unit_001.particle_type_000.init_cs = np.array([0., 0.])
    # Binding
    model.input.model.unit_001.particle_type_000.adsorption_model = 'LINEAR'
    model.input.model.unit_001.particle_type_000.nbound = np.array([1, 1])
    model.input.model.unit_001.particle_type_000.adsorption.is_kinetic = True
    model.input.model.unit_001.particle_type_000.adsorption.lin_ka = np.array([3.55, 2.0])
    model.input.model.unit_001.particle_type_000.adsorption.lin_kd = np.array([0.1, 1.0])
    
    # Particle - Type 1
    model.input.model.unit_001.particle_type_001.has_film_diffusion = 1
    model.input.model.unit_001.particle_type_001.has_pore_diffusion = 1
    model.input.model.unit_001.particle_type_001.has_surface_diffusion = 1
    model.input.model.unit_001.particle_type_001.par_geom = 'SPHERE'
    model.input.model.unit_001.particle_type_001.film_diffusion = [6.9e-06, 2e-06, 0.1*6.9e-06, 0.1*2e-06]
    model.input.model.unit_001.particle_type_001.film_diffusion_multiplex = 1 # component and section dependent
    model.input.model.unit_001.particle_type_001.film_diffusion_partype_dependent = 0
    model.input.model.unit_001.particle_type_001.par_coreradius = 0.0
    model.input.model.unit_001.particle_type_001.pore_diffusion = [
        6.07e-11, 1e-11, # section0: type1comp0, type1comp1
        2*6.07e-11, 2*1e-11 # section1: type1comp0, type1comp1
        ]
    model.input.model.unit_001.particle_type_001.pore_diffusion_multiplex = 1 # section, parType and component dependent
    model.input.model.unit_001.particle_type_001.pore_diffusion_partype_dependent = 1
    model.input.model.unit_001.particle_type_001.par_porosity = 0.75
    model.input.model.unit_001.particle_type_001.par_radius = 4.5e-05
    model.input.model.unit_001.particle_type_001.surface_diffusion = [
        2*5e-11, 2*2e-11 # type1comp0, type1comp1
        ]
    model.input.model.unit_001.particle_type_001.surface_diffusion_multiplex = 0 # parType and component dependent
    model.input.model.unit_001.particle_type_001.surface_diffusion_partype_dependent = 1
    model.input.model.unit_001.particle_type_001.init_cp = np.array([0., 0.])
    model.input.model.unit_001.particle_type_001.init_cs = np.array([0., 0.])
    # Binding
    model.input.model.unit_001.particle_type_001.adsorption_model = 'LINEAR'
    model.input.model.unit_001.particle_type_001.nbound = np.array([1, 1])
    model.input.model.unit_001.particle_type_001.adsorption.is_kinetic = True
    model.input.model.unit_001.particle_type_001.adsorption.lin_ka = np.array([3.55, 2.0])
    model.input.model.unit_001.particle_type_001.adsorption.lin_kd = np.array([0.1, 1.0])

    # Spatial discretization specifics (some of which will be overwritten)
    if spatial_method_bulk > 0:
        model.input.model.unit_001.discretization.SPATIAL_METHOD = 'DG'
        model.input.model.unit_001.discretization.POLYDEG = spatial_method_bulk
        model.input.model.unit_001.discretization.NELEM = 1
    else:
        model.input.model.unit_001.discretization.SPATIAL_METHOD = 'FV'
        model.input.model.unit_001.discretization.GS_TYPE = 0
        model.input.model.unit_001.discretization.MAX_KRYLOV = 10
        model.input.model.unit_001.discretization.MAX_RESTARTS = 100
        model.input.model.unit_001.discretization.NCOL = 8
        model.input.model.unit_001.discretization.RECONSTRUCTION = 'WENO'
        model.input.model.unit_001.discretization.SCHUR_SAFETY = 0.1
        model.input.model.unit_001.discretization.weno.BOUNDARY_MODEL = 0
        model.input.model.unit_001.discretization.weno.WENO_EPS = 1e-10
        model.input.model.unit_001.discretization.weno.WENO_ORDER = 3
        
        
    model.input.model.unit_001.discretization.USE_ANALYTIC_JACOBIAN = True

    if spatial_method_bulk > 0:
        model.input.model.unit_001.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
        model.input.model.unit_001.particle_type_000.discretization.PAR_POLYDEG = spatial_method_particle
        model.input.model.unit_001.particle_type_000.discretization.PAR_NELEM = 1
        model.input.model.unit_001.particle_type_000.discretization.PAR_DISC_TYPE = ['EQUIDISTANT_PAR']
        
        model.input.model.unit_001.particle_type_001.discretization.SPATIAL_METHOD = 'DG'
        model.input.model.unit_001.particle_type_001.discretization.PAR_POLYDEG = spatial_method_particle
        model.input.model.unit_001.particle_type_001.discretization.PAR_NELEM = 1
        model.input.model.unit_001.particle_type_001.discretization.PAR_DISC_TYPE = ['EQUIDISTANT_PAR']
    else:
        model.input.model.unit_001.particle_type_000.discretization.SPATIAL_METHOD = 'FV'
        model.input.model.unit_001.particle_type_000.discretization.NCELLS = 2
        model.input.model.unit_001.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2
        model.input.model.unit_001.particle_type_000.discretization.PAR_DISC_TYPE = ['EQUIDISTANT_PAR']
        
        model.input.model.unit_001.particle_type_001.discretization.SPATIAL_METHOD = 'FV'
        model.input.model.unit_001.particle_type_001.discretization.NCELLS = 2
        model.input.model.unit_001.particle_type_001.discretization.FV_BOUNDARY_ORDER = 2
        model.input.model.unit_001.particle_type_001.discretization.PAR_DISC_TYPE = ['EQUIDISTANT_PAR']
    
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


def add_sensitivity_GRMparType2_dynLin_1comp_benchmark1(model, sensName):

    sensDepIdx = {
        'FILM_DIFFUSION_SEC0': { 'sens_comp': 0,'sens_section': 0, 'sens_name': 'FILM_DIFFUSION' },
        'FILM_DIFFUSION_SEC1': { 'sens_comp': 0,'sens_section': 1, 'sens_name': 'FILM_DIFFUSION' },
        'SURFACE_DIFFUSION_PARTYPE0': { 'sens_comp': 0,'sens_partype': 0,'sens_section': 0, 'sens_name': 'PORE_DIFFUSION' },
        'SURFACE_DIFFUSION_PARTYPE1': { 'sens_comp': 0,'sens_partype': 1,'sens_section': 0, 'sens_name': 'PORE_DIFFUSION' },
        'SURFACE_DIFFUSION_COMP0': { 'sens_comp': 0,'sens_boundphase': 0, 'sens_partype': 0, 'sens_name': 'SURFACE_DIFFUSION' },
        'SURFACE_DIFFUSION_COMP1': { 'sens_comp': 1,'sens_boundphase': 0, 'sens_partype': 0, 'sens_name': 'SURFACE_DIFFUSION'}
    }

    if sensName not in sensDepIdx:
        raise Exception(f'Sensitivity dependencies for {sensName} unknown, please implement!')

    if 'sensitivity' in model['input']:
        model['input']['sensitivity']['NSENS'] += 1
    else:
        model['input']['sensitivity'] = {'NSENS': 1}
        model['input']['sensitivity']['sens_method'] = 'ad1'

    sensIdx = str(model['input']['sensitivity']['NSENS'] - 1).zfill(3)

    model['input']['sensitivity'][f'param_{sensIdx}'] = {}
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_name'] = str(sensName)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_unit'] = 1
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_partype'] = -1
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_reaction'] = -1
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_section'] = -1
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_boundphase'] = -1
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_comp'] = -1

    if sensName in sensDepIdx:
        param = model['input']['sensitivity'][f'param_{sensIdx}']
        for key, value in {**sensDepIdx[sensName]}.items():
            model['input']['sensitivity'][f'param_{sensIdx}'][key] = value

    return model


def get_sensbenchmark1(spatial_method_bulk, spatial_method_particle):

    model = get_model(spatial_method_bulk, spatial_method_particle)
    if 'sensitivity' in model.keys():
        model['input'].pop('sensitivity', None)
    model = add_sensitivity_GRMparType2_dynLin_1comp_benchmark1(model, 'FILM_DIFFUSION_SEC0')
    model = add_sensitivity_GRMparType2_dynLin_1comp_benchmark1(model, 'FILM_DIFFUSION_SEC1')
    model = add_sensitivity_GRMparType2_dynLin_1comp_benchmark1(model, 'SURFACE_DIFFUSION_PARTYPE0')
    model = add_sensitivity_GRMparType2_dynLin_1comp_benchmark1(model, 'SURFACE_DIFFUSION_PARTYPE1')
    model = add_sensitivity_GRMparType2_dynLin_1comp_benchmark1(model, 'SURFACE_DIFFUSION_COMP0')
    model = add_sensitivity_GRMparType2_dynLin_1comp_benchmark1(model, 'SURFACE_DIFFUSION_COMP1')

    return model
