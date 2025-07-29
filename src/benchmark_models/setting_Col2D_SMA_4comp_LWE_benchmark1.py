import numpy as np
from addict import Dict
import copy

import numpy as np
from scipy.special import legendre

def q_and_L(poly_deg, x):
    P = legendre(poly_deg)
    dP = P.deriv()
    L = P(x)
    q = P.deriv()(x)
    q_der = P.deriv(2)(x)
    return L, q, q_der


def lgl_nodes_weights(poly_deg):
    if poly_deg < 1:
        raise ValueError("Polynomial degree must be at least 1!")

    nodes = np.zeros(poly_deg + 1)
    weights = np.zeros(poly_deg + 1)
    pi = np.pi
    tolerance = 1e-15
    n_iterations = 10

    if poly_deg == 1:
        nodes[0] = -1
        nodes[1] = 1
        weights[0] = 1
        weights[1] = 1
    else:
        nodes[0] = -1
        nodes[poly_deg] = 1
        weights[0] = 2.0 / (poly_deg * (poly_deg + 1.0))
        weights[poly_deg] = weights[0]

        for j in range(1, (poly_deg + 1) // 2):
            x = -np.cos(pi * (j + 0.25) / poly_deg - 3 /
                        (8.0 * poly_deg * pi * (j + 0.25)))
            for k in range(n_iterations):
                L, q, q_der = q_and_L(poly_deg, x)
                dx = q / q_der
                x -= dx
                if abs(dx) <= tolerance * abs(x):
                    break
            nodes[j] = x
            nodes[poly_deg - j] = -x
            L, q, q_der = q_and_L(poly_deg, x)
            weights[j] = 2.0 / (poly_deg * (poly_deg + 1.0) * L**2)
            weights[poly_deg - j] = weights[j]

        if poly_deg % 2 == 0:
            L, q, q_der = q_and_L(poly_deg, 0.0)
            nodes[poly_deg // 2] = 0
            weights[poly_deg // 2] = 2.0 / (poly_deg * (poly_deg + 1.0) * L**2)

    return nodes, weights
# # Example usage:
# poly_deg = 5  # Change this to the desired order
# nodes, weights = lgl_nodes_weights(poly_deg)

# # Display the results
# print("nodes:", nodes)
# print("Weights:", weights)
# print("wiki node", np.sqrt(1/3 - 2*np.sqrt(7) / 21))
# print("wiki weight", (14 + np.sqrt(7))/30)
# print("Sum of weights:", sum(weights))
# print("Inv weights:", np.reciprocal(weights))


def generate_connections_matrix(rad_method, rad_cells,
                                velocity, porosity, col_radius,
                                add_inlet_per_port=True, add_outlet=False):
    """Computes the connections matrix with const. velocity flow rates, and radial coordinates.
    Equidistant cell/element spacing is assumed.
    
    Parameters
    ----------
    rad_method : int
        radial method / polynomial degree
    rad_cells : int
        radial number of cells
    velocity : float
        column velocity (constant)
    porosity : float
        column porosity (constant)
    col_radius : float
        column radius (constant)
    add_inlet_per_port : int | bool
        specifies how many radial zones are used either by number or by true to specify one per port
    add_outlet : bool
        specifies whetehr or not an outlet is connected per radial zone
    
    Returns
    -------
    List of float, List of float
        Connections matrix, radial coordinates.
    """

    nRadPoints = (rad_method + 1) * rad_cells

    # we want the same velocity within each radial zone and use an equidistant radial grid, ie we adjust the volumetric flow rate accordingly in each port
    # 1. compute cross sections

    subcellCrossSectionAreas = []
    rad_coords = []

    if rad_method > 0:

        nodes, weights = lgl_nodes_weights(rad_method)
        # scale the weights to radial element spacing
        # note that weights need to be scaled to 1 later, to give us the size of the corresponding subcells
        # print(sum(weights) / 2.0 - 1.0 < 1E-15)
        deltaR = col_radius / rad_cells
        for rIdx in range(rad_cells):
            jojoL = rIdx * deltaR
            for node in range(rad_method + 1):
                jojoR = jojoL + weights[node] / 2.0 * deltaR
                # print("Left boundary: ", jojoL)
                # print("Right boundary: ", jojoR)
                subcellCrossSectionAreas.append(
                    np.pi * (jojoR ** 2 - jojoL ** 2))
                rad_coords.append(
                    rIdx * deltaR + (nodes[node] + 1) / 2.0 * deltaR)
                jojoL = jojoR
    else:
        deltaR = col_radius / nRadPoints
        jojoL = 0.0
        for rIdx in range(nRadPoints):
            rad_coords.append(rIdx * deltaR + deltaR / 2.0)
            jojoR = jojoL + deltaR
            subcellCrossSectionAreas.append(np.pi * (jojoR ** 2 - jojoL ** 2))
            jojoL = jojoR

    # create flow rates for each zone
    flowRates = []
    columnIdx = 0  # always needs to be the first unit
    
    for rad in range(nRadPoints):
        flowRates.append(subcellCrossSectionAreas[rad] * porosity * velocity)
    # create connections matrix
    connections = []
    # add inlet connections
    if add_inlet_per_port:

        nRadialZones = rad_cells if add_inlet_per_port is True else add_inlet_per_port

        if not rad_cells % nRadialZones == 0:
            raise Exception(
                f"Number of rad_cells {rad_cells} is not a multiple of radial zones {nRadialZones}")

        for rad in range(nRadPoints):
            zone = int(rad / (nRadPoints / nRadialZones))
            connections += [zone + 1, columnIdx,
                            0, rad, -1, -1, flowRates[rad]]
            if add_outlet:
                connections += [columnIdx, nRadialZones + 1 + zone,
                                rad, 0, -1, -1, flowRates[rad]]
    else:
        for rad in range(nRadPoints):
            connections += [1, columnIdx, 0, rad, -1, -1, flowRates[rad]]
            if add_outlet:
                connections += [columnIdx, nRadPoints + 1 + rad,
                                rad, 0, -1, -1, flowRates[rad]]
                
    return connections, rad_coords

def get_model(particle_type='GENERAL_RATE_PARTICLE'):
    
    polyDeg = 3
    radNElem = 3
    nRadZones = 3 # number of radial zones
    
    model = Dict()
    
    model.input.model.nunits = 1 + 2 * nRadZones
    
    # Column unit
    column = Dict()
    column.unit_type = 'COLUMN_MODEL_2D'
    column.col_length = 0.014
    column.col_radius = 0.01
    column.col_porosity = 0.37
    column.npartype = 0 if particle_type is None else 1
    
    column.ncomp = 4
    column.init_c = [50.0, 0.0, 0.0, 0.0]
    column.col_dispersion = 5.75e-08
    column.col_dispersion_radial = 1e-06
    column.velocity = 0.000575

    # Spatial discretization of interstitial / bulk volume
    column.discretization.spatial_method = 'DG'
    column.discretization.exact_integration = 0
    column.discretization.ax_polydeg = polyDeg
    column.discretization.ax_nelem = 8
    column.discretization.radial_disc_type = 'EQUIDISTANT'
    
    column.discretization.rad_polydeg = polyDeg
    column.discretization.rad_nelem = radNElem
    column.discretization.use_analytic_jacobian = 1
    
    if particle_type is not None:
        
        column.particle_type_000.par_geom = ['SPHERE']
        column.particle_type_000.particle_type = particle_type
        column.particle_type_000.par_radius = 4.5e-05
        column.particle_type_000.par_coreradius = 0.0
        column.particle_type_000.par_porosity = 0.75
        column.particle_type_000.film_diffusion = [6.9e-06, 6.9e-06, 6.9e-06, 6.9e-06]
        column.particle_type_000.par_diffusion = [7.00e-10, 6.07e-11, 6.07e-11, 6.07e-11]
        column.particle_type_000.par_surfdiffusion = [0.,0.,0.,0.]
        column.particle_type_000.nbound = [1, 1, 1, 1]
        column.init_cp = [50.0, 0.0, 0.0, 0.0]
        column.init_cs = [1200.0, 0.0, 0.0, 0.0]
        
        column.particle_type_000.adsorption_model = 'STERIC_MASS_ACTION'
        column.particle_type_000.adsorption = {
                'is_kinetic': 0,
                'sma_ka': [ 0.0, 35.5, 1.59, 7.7 ],
                'sma_kd': [ 0.0, 1000.0, 1000.0, 1000.0],
                'sma_lambda': 1200.0,
                'sma_nu': [ 0.0, 4.7, 5.29, 3.7 ],
                'sma_sigma': [ 0.0, 11.83, 10.6, 10.0 ]
                }
        
        # Spatial discretization of particle volume
        column.particle_type_000.discretization.par_disc_type = 'EQUIDISTANT_PAR'
        column.particle_type_000.discretization.par_polydeg = polyDeg
        column.particle_type_000.discretization.par_nelem = 1
    
    model.input.model.unit_000 = column
    
    # Flow sheet, system
    model.input.model.connections.connections_include_ports = 1
    model.input.model.connections.nswitches = 1
    connections, rad_coords = generate_connections_matrix(
            rad_method=polyDeg, rad_cells=radNElem,
            velocity=column.velocity, porosity=column.col_porosity, col_radius=column.col_radius,
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
    inletUnit.unit_type = 'INLET'
    
    outletUnit = Dict()
    outletUnit.unit_type = 'OUTLET'
    outletUnit.ncomp = 4
    
    # add inlet and outlet for each radial zone
    for rad in range(radNElem):

        model.input.model['unit_' + str(rad + 1).zfill(3)
                    ] = copy.deepcopy(inletUnit)
        
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
    model.input.solver.time_integrator.abstol = 1e-07
    model.input.solver.time_integrator.algtol = 1e-05
    model.input.solver.time_integrator.init_step_size = 1e-10
    model.input.solver.time_integrator.max_steps = 10000
    model.input.solver.time_integrator.reltol = 1e-05
    
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

