import numpy as np
from addict import Dict
import copy

import src.benchmark_models.helper_connections_2Dmodels as helper


def get_model(
        polyDeg, axNElem, radNElem, parNElem,
        particle_type="GENERAL_RATE_PARTICLE",
        **kwargs
        ):
    
    nRadZones = radNElem
    
    radNElem = radNElem
    axNElem = axNElem
    parNElem = parNElem
    
    model = Dict()
    
    model.input.model.nunits = 1 + 2 * nRadZones
    
    # Column unit
    column = Dict()
    column.UNIT_TYPE = 'COLUMN_MODEL_2D'
    column.col_length = 0.014
    column.col_radius = 0.01
    
    # Build a radially resolved porosity profile to mimic wall effects in packed columns.
    # The porosity is assumed lower in the bulk (core) and increases smoothly toward the wall,
    # where packing is less dense. An exponential function is used to create a smooth transition
    # between core and wall porosity over a characteristic decay length on the order of a few
    # particle diameters. This provides a physically plausible profile while remaining simple
    # and well-suited for visualizing radial gradients in the simulation.
    deltaR = column.col_radius / radNElem
    par_radius = 4.5e-05
    R = column.col_radius
    eps_inner = kwargs.get('eps_inner', 0.35)
    eps_wall = kwargs.get('eps_wall', 0.5)
    dp = 2 * par_radius
    lam = 2.0 * dp  # decay length
    
    eps_r = []
    
    # rad_coords, _ = helper.get_radCoords_and_crossSectionAreas(polyDeg, radNElem, column.col_radius)
    for r in range(radNElem):
        eps = eps_inner + (eps_wall - eps_inner) * np.exp(-(R - deltaR * (1.0 + r)) / lam) # 1.0 + r to get the porosity at the right edge of the element
        eps_r.append(eps)
    
    column.col_porosity = eps_r
    
    print("Interstitial velocity per radial zone: ", np.array(eps_r) / 0.37 * 0.000575)
    print("Porosity per radial zone: ", np.array(eps_r))
    
    column.npartype = 0 if particle_type is None else 1
    
    column.ncomp = 4
    column.init_c = [50.0, 0.0, 0.0, 0.0]
    column.col_dispersion_axial = 5.75e-08
    column.col_dispersion_radial = kwargs.get('col_dispersion_radial', 5e-08)
    column.velocity = 0.000575

    # Spatial discretization of interstitial / bulk volume
    column.discretization.spatial_method = 'DG'
    column.discretization.POLYNOMIAL_INTEGRATION_TYPE = 1
    column.discretization.AX_POLYDEG = kwargs.get('axP', polyDeg)
    column.discretization.AX_NELEM = axNElem
    column.discretization.RADIAL_DISC_TYPE = 'EQUIDISTANT'
    column.discretization.RAD_POLYDEG = kwargs.get('radP', polyDeg)
    column.discretization.RAD_NELEM = radNElem
    column.discretization.USE_ANALYTIC_JACOBIAN = 1
    
    if particle_type is not None:
        
        column.particle_type_000.par_geom = ['SPHERE']
        column.particle_type_000.par_radius = par_radius
        column.particle_type_000.par_coreradius = 0.0
        column.particle_type_000.par_porosity = 0.75
        column.particle_type_000.has_film_diffusion = True
        column.particle_type_000.has_pore_diffusion = True
        column.particle_type_000.has_surface_diffusion = False
        column.particle_type_000.film_diffusion = [6.9e-06, 6.9e-06, 6.9e-06, 6.9e-06]
        column.particle_type_000.pore_diffusion = [7.00e-10, 6.07e-11, 6.07e-11, 6.07e-11]
        column.particle_type_000.surface_diffusion = [0.,0.,0.,0.]
        column.particle_type_000.nbound = [1, 1, 1, 1]
        column.particle_type_000.init_cp = [50.0, 0.0, 0.0, 0.0]
        column.particle_type_000.init_cs = [1200.0, 0.0, 0.0, 0.0]
        
        column.particle_type_000.adsorption_model = 'STERIC_MASS_ACTION'
        column.particle_type_000.adsorption = {
                'is_kinetic': 1,
                'sma_ka': [ 0.0, 35.5, 1.59, 7.7 ],
                'sma_kd': [ 0.0, 1000.0, 1000.0, 1000.0],
                'sma_lambda': 1200.0,
                'sma_nu': [ 0.0, 4.7, 5.29, 3.7 ],
                'sma_sigma': [ 0.0, 11.83, 10.6, 10.0 ]
                }
        
        # Spatial discretization of particle volume
        column.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        column.particle_type_000.discretization.PAR_POLYDEG = kwargs.get('parP', polyDeg)
        column.particle_type_000.discretization.PAR_NELEM = parNElem
    
    model.input.model.unit_000 = column
    
    # Flow sheet, system
    model.input.model.connections.connections_include_ports = 1
    model.input.model.connections.nswitches = 1
    connections, rad_coords = helper.generate_connections_matrix(
            rad_method=polyDeg, rad_cells=radNElem,
            velocity=column.velocity, porosity=column.col_porosity[0], col_radius=column.col_radius,
            add_inlet_per_port=nRadZones, add_outlet=True
        )
    
    model.input.model.connections.switch_000.connections = connections
    model.input.model.connections.switch_000.section = 0
    
    # Inlet / Feed unit
    inletUnit = Dict()
    inletUnit.inlet_type = 'PIECEWISE_CUBIC_POLY'
    inletUnit.ncomp = 4
    inletUnit.sec_000.const_coeff = [50.0, 1.0, 1.0, 1.0]
    inletUnit.sec_000.cube_coeff = [0.0, 0.0, 0.0, 0.0]
    inletUnit.sec_000.lin_coeff = [0.0, 0.0, 0.0, 0.0]
    inletUnit.sec_000.quad_coeff = [0.0, 0.0, 0.0, 0.0]
    inletUnit.sec_001.const_coeff = [50.0, 0.0, 0.0, 0.0]
    inletUnit.sec_001.cube_coeff = [0.0, 0.0, 0.0, 0.0]
    inletUnit.sec_001.lin_coeff = [0.0, 0.0, 0.0, 0.0]
    inletUnit.sec_001.quad_coeff = [0.0, 0.0, 0.0, 0.0]
    inletUnit.sec_002.const_coeff = [100.0, 0.0, 0.0, 0.0]
    inletUnit.sec_002.cube_coeff = [0.0, 0.0, 0.0, 0.0]
    inletUnit.sec_002.lin_coeff = [0.2, 0.0, 0.0, 0.0]
    inletUnit.sec_002.quad_coeff = [0.0, 0.0, 0.0, 0.0]
    inletUnit.UNIT_TYPE = 'INLET'
    
    outletUnit = Dict()
    outletUnit.UNIT_TYPE = 'OUTLET'
    outletUnit.ncomp = 4
    
    # add inlet and outlet for each radial zone
    for rad in range(radNElem):

        model.input.model['unit_' + str(rad + 1).zfill(3)
                    ] = copy.deepcopy(inletUnit)
        model.input.model['unit_' + str(rad + 1).zfill(3)
                    ]['const_coeff'] = [50.0, 1.0 * rad, 1.0 * rad, 1.0 * rad]
        
        model.input.model['unit_' + str(radNElem + 1 + rad).zfill(3)] = copy.deepcopy(outletUnit)
        
        model.input.model['return']['unit_' + str(radNElem + 1 + rad).zfill(3)].write_solution_outlet = 1

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
    model.input.solver.sections.section_times = [ 0.0, 10.0, 90.0, 1500.0]
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_reftol', 1e-7)
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reftol', 1e-5)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_reftol', 1e-8)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-12
    model.input.solver.time_integrator.MAX_STEPS = 10000
    
    # Return data
    model.input.solver.user_solution_times = np.linspace(0.0, 1500.0, 1501)
    model.input['return'].split_components_data = 1
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_coordinates = kwargs.get('write_solution_bulk', False) or kwargs.get('write_solution_particle', False) or kwargs.get('write_solution_solid', False)
    model.input['return'].unit_000.write_sens_bulk = 0
    model.input['return'].unit_000.write_sens_last = 0
    model.input['return'].unit_000.write_sens_outlet = 0
    model.input['return'].unit_000.write_solution_bulk = kwargs.get('write_solution_bulk', False)
    model.input['return'].unit_000.write_solution_flux = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].unit_000.write_solution_particle = kwargs.get('write_solution_particle', False)
    model.input['return'].unit_000.write_solution_solid = kwargs.get('write_solution_solid', False)
    model.input['return'].write_solution_times = 1
    
    return model





from cadet import Cadet

model = Cadet()
model.install_path = r"C:\Users\jmbr\Desktop\CADET_compiled\master5_fixParCoords_783967a\aRELEASE"
polyDeg = 5
axNElem = 16
radNElem = 2
parNElem = 20

eps_wall=0.5

model.root = get_model(
    polyDeg=polyDeg, axNElem=axNElem, radNElem=radNElem, parNElem=parNElem,
    write_solution_bulk=True, write_solution_particle=True, write_solution_solid=True,
    eps_wall=eps_wall, eps_inner=0.35
    )
modelName = f"2DLWE_radInlet_epsRc{eps_wall}_DG_P{polyDeg}Z{axNElem}radZ{radNElem}parZ{parNElem}"
model.filename = r"C:\Users\jmbr\software/" + modelName + ".h5"

model.save()
# return_data = model.run_simulation()
# print(return_data.return_code)
# print(return_data.error_message)
# model.load()
# model.save()














