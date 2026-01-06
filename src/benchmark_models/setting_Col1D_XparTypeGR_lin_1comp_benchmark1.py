"""

This script defines a setting with 2 or 4 particle types with different
parameters and linear bindings. In the 4 particle types setting, one particle
type has no adsorption and no bound state.
The setting is meant for software verification, not to define a physically
meaningful case study.

"""

from addict import Dict
import numpy as np

def get_model(
        spatial_method_bulk,
        spatial_method_particle,
        refinement=1,
        **kwargs):
    
    axNElem = 8 * kwargs.get('axRefinement', refinement)
    parNElem = kwargs.get('parZ', 1)
    
    model = Dict()
    
    model.input.model.nunits = 3
    
    # Flow sheet
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        0.e+00, 1.e+00,-1.e+00,-1.e+00, kwargs.get('flowRate', 6.e-05), 1.e+00, 2.e+00,-1.e+00,-1.e+00, kwargs.get('flowRate', 6.e-05)
        ]
    model.input.model.connections.switch_000.section = 0
    
    #%% Column unit
    column = Dict()
    
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.ncomp = 1
    column.col_dispersion = 5.75e-08
    column.col_length = kwargs.get('colLength', 0.014)
    column.cross_section_area = (kwargs.get('flowRate', 6.e-05) / 0.000575) / 0.37
    column.velocity = 0.000575
    column.col_porosity = 0.37
    column.npartype = kwargs.get('npartype', 1)
    column.par_type_volfrac = 1 if 'par_type_volfrac' not in kwargs else kwargs['par_type_volfrac']
    
    if spatial_method_bulk > 0:
        column.discretization.SPATIAL_METHOD = "DG"
        column.discretization.EXACT_INTEGRATION = kwargs.get('exact_integration', 0)
        column.discretization.POLYDEG = spatial_method_bulk
        column.discretization.NELEM = axNElem
    else:
        column.discretization.SPATIAL_METHOD = "FV"
        column.discretization.NCOL = axNElem
        column.discretization.RECONSTRUCTION = 'WENO'
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = kwargs.get('weno_order', 2)
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10
        column.discretization.SCHUR_SAFETY = 1.0e-8

    column.discretization.USE_ANALYTIC_JACOBIAN = 1
    column.init_c = [ 0.0 ]
        
    for parType in range(0, column.npartype):
        
        particle_group = f'particle_type_{parType:03d}'
        
        column[particle_group].film_diffusion = 6.9e-06 if 'film_diffusion' not in kwargs else kwargs['film_diffusion'][parType]
        column[particle_group].has_film_diffusion = 1
        column[particle_group].par_geom = 'SPHERE'
        column[particle_group].par_radius = 4.5e-05 if 'par_radius' not in kwargs else kwargs['par_radius'][parType]
        column[particle_group].par_coreradius = 0.0 if 'par_coreradius' not in kwargs else kwargs['par_coreradius'][parType]
        column[particle_group].par_porosity = 0.75 if 'par_porosity' not in kwargs else kwargs['par_porosity'][parType]
        
        column[particle_group].has_pore_diffusion = 1
        column[particle_group].pore_diffusion = 6.07e-11 if 'pore_diffusion' not in kwargs else kwargs['pore_diffusion'][parType]
        
        column[particle_group].nbound = [ 1 ] if 'nbound' not in kwargs else [kwargs['nbound'][parType]]
        surfDiff = 0.0 if 'surface_diffusion' not in kwargs else kwargs['surface_diffusion'][parType]
        
        if column[particle_group].nbound[0] > 0:
            column[particle_group].surface_diffusion = surfDiff
            column[particle_group].has_surface_diffusion = 1 if surfDiff > 0.0 else 0
        else:
            column[particle_group].has_surface_diffusion = 0
            
        if spatial_method_particle > 0:
            column.discretization.SPATIAL_METHOD = "DG"
            column[particle_group].discretization.PAR_POLYDEG = spatial_method_particle
            column[particle_group].discretization.PAR_NELEM = parNElem
        else:
            column.discretization.SPATIAL_METHOD = "FV"
            column[particle_group].discretization.NCELLS = parNElem
            column[particle_group].discretization.FV_BOUNDARY_ORDER = 2
            
        column[particle_group].discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        
        column[particle_group].adsorption.is_kinetic = 1 if 'is_kinetic' not in kwargs else kwargs['is_kinetic'][parType]
        column[particle_group].adsorption.lin_ka = [ 3.55 ] if 'lin_ka' not in kwargs else [kwargs['lin_ka'][parType]]
        column[particle_group].adsorption.lin_kd = [ 0.1 ] if 'lin_kd' not in kwargs else [kwargs['lin_kd'][parType]]
        column[particle_group].adsorption_model = 'LINEAR' if 'adsorption_model' not in kwargs else kwargs['adsorption_model'][parType]
        column[particle_group].init_cp = [ 0.0 ] if 'init_cp' not in kwargs else [kwargs['init_cp'][parType]]
        column[particle_group].init_cs = [ 0.0 ] if 'init_cs' not in kwargs else [kwargs['init_cs'][parType]]
    
    model.input.model.unit_001 = column
    
    #%% time integration parameters
    # non-linear solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08
    # time integration / solver specifics
    model.input.solver.consistent_init_mode = 1
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = [ 0 ]
    model.input.solver.sections.section_times = [ 0.0, 10.0, 1500.0 ]
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_reftol', 1e-12)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_reftol', 1e-10)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 10000
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reftol', 1e-10)
    model.input.solver.user_solution_times = np.linspace(0.0, 1500.0, 1500*4 + 1)
    
    #%% auxiliary units: inlet and outlet
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = 1
    model.input.model.unit_000.sec_000.const_coeff = [1.]
    model.input.model.unit_000.sec_000.cube_coeff = [0.]
    model.input.model.unit_000.sec_000.lin_coeff = [0.]
    model.input.model.unit_000.sec_000.quad_coeff = [0.]
    model.input.model.unit_000.sec_001.const_coeff = [0.]
    model.input.model.unit_000.sec_001.cube_coeff = [0.]
    model.input.model.unit_000.sec_001.lin_coeff = [0.]
    model.input.model.unit_000.sec_001.quad_coeff = [0.]
    model.input.model.unit_000.UNIT_TYPE = 'INLET'
    
    model.input.model.unit_002.ncomp = 1
    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    
    
    #%% return data
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
    
    return model
