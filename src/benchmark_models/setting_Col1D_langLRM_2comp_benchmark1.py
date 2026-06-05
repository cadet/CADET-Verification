"""

This script defines Langmuir adsorption case studies
used for numerical benchmarks in
https://doi.org/10.1016/j.compchemeng.2023.108340

"""

from addict import Dict
import numpy as np

def get_model(spatial_method_bulk, refinement=1, **kwargs):
    
    axNElem = 8 * refinement
    
    model = Dict()
    
    model.input.model.nunits = 3

    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1

    #%% Column unit
    column = Dict()
    
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.ncomp = 2
    column.npartype = 1
    column.col_dispersion = kwargs.get("col_dispersion", 1e-05)
    column.col_length = 1.0
    column.total_porosity = 0.4
    # Flow sheet
    #velocity = 0.1
    A = column.col_length * 0.5
    column.cross_section_area = A  

    Q = 0.1 * A * column.total_porosity # note that v = Q/(A*porosity)
    model.input.model.connections.switch_000.connections = [
        0.e+00, 1.e+00,-1.e+00,-1.e+00, Q, 
        1.e+00, 2.e+00,-1.e+00,-1.e+00, Q
        ]
    model.input.model.connections.switch_000.section = 0
    
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
        column.discretization.weno.WENO_ORDER = 2
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10
        column.discretization.SCHUR_SAFETY = 1.0e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = 1
    column.init_c = [ 0.0 , 0.0]

    # Particle Model
    column.particle_type_000.has_film_diffusion = False
    
    column.particle_type_000.nbound = [1, 1]
    column.particle_type_000.init_cs = [0.0, 0.0]
    
    # Adsorption - LANGMUIR for 2 components
    column.particle_type_000.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    column.particle_type_000.adsorption.is_kinetic = kwargs.get('is_kinetic', 0)
    
    # Langmuir parameters for 2 components
    column.particle_type_000.adsorption.mcl_ka = kwargs.get('mcl_ka', [0.1, 0.05])
    column.particle_type_000.adsorption.mcl_kd = kwargs.get('mcl_kd', [1.0, 1.0])                 
    column.particle_type_000.adsorption.mcl_qmax = kwargs.get('mcl_qmax', [10.0, 10.0])

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
    model.input.solver.sections.section_times = [ 0.0, 12.0, 40.0 ]
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_reftol', 1e-12)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_reftol', 1e-10)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 10000
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reftol', 1e-10)
    model.input.solver.user_solution_times = np.linspace(0.0, 40.0, 40*4 + 1)
    
    #%% auxiliary units: inlet and outlet
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = 2
    #Feed concentration section 1
    model.input.model.unit_000.sec_000.const_coeff = [10.0, 10.0]
    model.input.model.unit_000.sec_000.cube_coeff = [0., 0.]
    model.input.model.unit_000.sec_000.lin_coeff = [0., 0.]
    model.input.model.unit_000.sec_000.quad_coeff = [0., 0.]
    #Feed concentration section 2
    model.input.model.unit_000.sec_001.const_coeff = [0.0, 0.0]
    model.input.model.unit_000.sec_001.cube_coeff = [0., 0.]
    model.input.model.unit_000.sec_001.lin_coeff = [0., 0.]
    model.input.model.unit_000.sec_001.quad_coeff = [0., 0.]
    model.input.model.unit_000.UNIT_TYPE = 'INLET'
    
    model.input.model.unit_002.ncomp = 2
    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    
    
    #%% return data
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_solution_bulk = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 0

    model.input['return'].unit_001.write_coordinates = kwargs.get('write_solution_bulk', 0) or kwargs.get('write_solution_solid', 0)
    model.input['return'].unit_001.write_solution_bulk = kwargs.get('write_solution_bulk', 0) or kwargs.get('write_solution_particle', 0)
    model.input['return'].unit_001.write_solution_solid = kwargs.get('write_solution_solid', 0)
    model.input['return'].unit_001.write_solution_inlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1

    model.input['return'].unit_002.write_solution_bulk = 0
    model.input['return'].unit_002.write_solution_inlet = 0
    model.input['return'].unit_002.write_solution_outlet = 0
    
    return model
