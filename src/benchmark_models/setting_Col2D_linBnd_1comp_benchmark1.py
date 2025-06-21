import numpy as np
from addict import Dict
import copy

from . import helper_connections_2Dmodels as helper


def get_model(
        column_resolution='2D',
        particle_type='GENERAL_RATE_PARTICLE', nComp=1, nRadZones=3,
        refinement=1, polyDeg=3,
        analytical_jacobian=True,
        **kwargs
        ):
    
    if column_resolution == '1D':
        nRadZones = 1
    
    radNElem = nRadZones * kwargs.get('radRefinement', refinement)
    axNElem = 8 * kwargs.get('axRefinement', refinement)
    parNElem = 1
    
    model = Dict()
    
    model.input.model.nunits = 1 + 2 * nRadZones
    
    # Column unit
    column = Dict()
    column.UNIT_TYPE = 'COLUMN_MODEL_' + column_resolution
    
    column.col_length = 0.014
    column.col_radius = 0.01
    column.col_porosity = 0.37
    column.npartype = 0 if particle_type is None else 1
    
    column.ncomp = nComp
    column.init_c = [0.0] * nComp
    column.col_dispersion = 5.75e-08
    if column_resolution == '2D':
        column.col_dispersion_radial = kwargs.get('col_dispersion_radial', 5e-08)
    column.velocity = 0.000575

    # Spatial discretization of interstitial / bulk volume
    column.discretization.spatial_method = 'DG'
    column.discretization.exact_integration = 1
    if column_resolution == '2D':
        column.discretization.AX_POLYDEG = kwargs.get('axP', polyDeg)
        column.discretization.AX_NELEM = axNElem
        column.discretization.RADIAL_DISC_TYPE = 'EQUIDISTANT'
        column.discretization.RAD_POLYDEG = kwargs.get('radP', polyDeg)
        column.discretization.RAD_NELEM = radNElem
    elif column_resolution == '1D':
        column.discretization.POLYDEG = kwargs.get('axP', polyDeg)
        column.discretization.NELEM = axNElem
        
    column.discretization.USE_ANALYTIC_JACOBIAN = analytical_jacobian
    
    if particle_type is not None:
        
        column.particle_type_000.par_geom = ['SPHERE']
        column.particle_type_000.particle_type = particle_type
        column.particle_type_000.par_radius = 4.5e-05
        column.particle_type_000.par_coreradius = 0.0
        column.particle_type_000.par_porosity = 0.75
        column.particle_type_000.film_diffusion = [6.9e-06] * nComp
        column.particle_type_000.par_diffusion = [6.07e-11] * nComp
        column.particle_type_000.par_surfdiffusion = [0.0] * nComp
        column.particle_type_000.nbound = [1] * nComp
        column.init_cp = [0.0] * nComp
        column.init_cs = [0.0] * nComp
        
        column.particle_type_000.adsorption_model = 'LINEAR'
        column.particle_type_000.adsorption = {
                'is_kinetic': [ True ] * nComp,
                'lin_ka': [ 35.5 ] * nComp,
                'lin_kd': [ 1.0] * nComp
                }
        
        # Spatial discretization of particle volume
        column.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        column.particle_type_000.discretization.PAR_POLYDEG = kwargs.get('parP', polyDeg)
        column.particle_type_000.discretization.PAR_NELEM = parNElem
    
    model.input.model.unit_000 = column
    
    # Flow sheet, system
    if column_resolution == '2D':
        model.input.model.connections.connections_include_ports = 1
        model.input.model.connections.nswitches = 1
        connections, rad_coords = helper.generate_connections_matrix(
                rad_method=kwargs.get('radP', polyDeg), rad_cells=radNElem,
                velocity=column.velocity, porosity=column.col_porosity, col_radius=column.col_radius,
                add_inlet_per_port=nRadZones, add_outlet=True
            )
        model.input.model.connections.switch_000.connections = connections
        model.input.model.connections.switch_000.section = 0
    elif column_resolution == '1D':
        model.input.model.connections.connections_include_ports = 1
        model.input.model.connections.nswitches = 1
        model.input.model.connections.switch_000.connections = [
            1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0,
            0.0, 2.0, -1.0, -1.0, -1.0, -1.0, 1.0
        ]
        model.input.model.connections.switch_000.section = 0
    
    # Inlet / Feed unit
    inletUnit = Dict()
    inletUnit.inlet_type = 'PIECEWISE_CUBIC_POLY'
    inletUnit.ncomp = nComp
    inletUnit.sec_000.const_coeff = [1.0] * nComp
    inletUnit.sec_000.cube_coeff = [0.0] * nComp
    inletUnit.sec_000.lin_coeff = [0.0] * nComp
    inletUnit.sec_000.quad_coeff = [0.0] * nComp
    inletUnit.sec_001.const_coeff = [0.0] * nComp
    inletUnit.sec_001.cube_coeff = [0.0] * nComp
    inletUnit.sec_001.lin_coeff = [0.0] * nComp
    inletUnit.sec_001.quad_coeff = [0.0] * nComp
    inletUnit.UNIT_TYPE = 'INLET'
    
    outletUnit = Dict()
    outletUnit.UNIT_TYPE = 'OUTLET'
    outletUnit.ncomp = nComp
    
    # add inlet and outlet for each radial zone
    for rad in range(nRadZones):

        model.input.model['unit_' + str(rad + 1).zfill(3)
                    ] = copy.deepcopy(inletUnit)
        
        model.input.model['unit_' + str(nRadZones + 1 + rad).zfill(3)] = copy.deepcopy(outletUnit)
        
        model.input.model['return']['unit_' + str(nRadZones + 1 + rad).zfill(3)].write_solution_outlet = 1

    # Global system solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    # Time integration / solver
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 3
    model.input.solver.sections.section_continuity = [ 0, 0 ]
    model.input.solver.sections.section_times = [ 0.0, 10.0, 100.0]
    model.input.solver.time_integrator.ABSTOL = 1e-12
    model.input.solver.time_integrator.ALGTOL = 1e-10
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-12
    model.input.solver.time_integrator.MAX_STEPS = 10000
    model.input.solver.time_integrator.RELTOL = 1e-010
    
    # Return data
    model.input.solver.user_solution_times = np.linspace(0, 100, 101)
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_coordinates = kwargs.get('return_bulk', False) or kwargs.get('return_particle', False)
    model.input['return'].unit_000.write_sens_bulk = 0
    model.input['return'].unit_000.write_sens_last = 0
    model.input['return'].unit_000.write_sens_outlet = 1
    model.input['return'].unit_000.write_solution_bulk =  kwargs.get('return_bulk', False)
    model.input['return'].unit_000.write_solution_flux = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].unit_000.write_solution_particle = kwargs.get('return_particle', False)
    model.input['return'].unit_000.write_solution_solid = kwargs.get('return_particle', False)
    model.input['return'].write_solution_times = 1
    
    return model
