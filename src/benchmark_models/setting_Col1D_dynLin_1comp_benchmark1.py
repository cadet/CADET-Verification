from addict import Dict
import numpy as np

def get_model(
        particle_type='GENERAL_RATE_PARTICLE',
        refinement=1, polyDeg=3,
        **kwargs):
    
    axNElem = 8 * kwargs.get('axRefinement', refinement)
    parNElem = kwargs.get('parZ', 1)
    
    model = Dict()
    
    model.input.model.nunits = 3
    
    # Flow sheet
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        0.e+00, 1.e+00,-1.e+00,-1.e+00, 6.e-05, 1.e+00, 2.e+00,-1.e+00,-1.e+00, 6.e-05
        ]
    model.input.model.connections.switch_000.section = 0
    
    #%% Column unit
    column = Dict()
    
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.ncomp = 1
    column.col_dispersion = 5.75e-08
    column.col_length = 0.014
    column.col_porosity = 0.37
    column.velocity = 0.000575
    column.npartype = 1
    column.par_type_volfrac = 1
    
    column.discretization.SPATIAL_METHOD = 'DG'
    # column.discretization.EXACT_INTEGRATION = 1
    column.discretization.POLYDEG = kwargs.get('axP', polyDeg)
    column.discretization.NELEM = axNElem
    column.discretization.USE_ANALYTIC_JACOBIAN = 1
    column.init_c = [ 0.0 ]
    
    column.particle_type_000.particle_type = particle_type
    column.particle_type_000.par_geom = 'SPHERE'
    column.particle_type_000.par_radius = 4.5e-05
    column.particle_type_000.par_coreradius = 0.0
    column.particle_type_000.par_porosity = 0.75
    column.particle_type_000.par_diffusion = 6.07e-11
    column.particle_type_000.par_surfdiffusion = kwargs.get('par_surfdiffusion', 0.0)
    column.particle_type_000.film_diffusion = 6.9e-06
    column.particle_type_000.nbound = [ 1 ]
    column.particle_type_000.adsorption.is_kinetic = True
    column.particle_type_000.adsorption.lin_ka = [ 3.55 ]
    column.particle_type_000.adsorption.lin_kd = [ 0.1 ]
    column.particle_type_000.adsorption_model = 'LINEAR'
    column.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
    column.particle_type_000.discretization.PAR_POLYDEG = kwargs.get('parP', polyDeg)
    column.particle_type_000.discretization.PAR_NELEM = parNElem
    column.init_cp = [ 0.0 ]
    column.init_cs = [ 0.0 ]
    
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
