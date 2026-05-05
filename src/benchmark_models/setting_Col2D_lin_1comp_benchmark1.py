# -*- coding: utf-8 -*-
"""

This script defines model settings considered for the verification of the
2D flow chromatography models in CADET-Core

"""

import numpy as np
from addict import Dict
import re
import copy

from src.utility import convergence
import src.benchmark_models.helper_setup_2Dmodels as helper


# %% Verification settings with radial variation


def get_model(
        axMethod=0, axNElem=8,
        radMethod=0, radNElem=3,
        parMethod=0, parNElem=2,
        nRadialZones=1,  # discontinuous radial inlet zones (equidistant)
        save_path="C:/Users/jmbr/JupyterNotebooks/",
        file_name=None,
        transport_model=None,
        **kwargs
):

    nRadPoints = (radMethod + 1) * radNElem
    nInlets = nRadialZones
    nOutlets = nRadialZones

    column = Dict()

    if transport_model is not None:
        column.UNIT_TYPE = transport_model
    else:
        column.UNIT_TYPE = 'COLUMN_MODEL_2D'
    nComp = 1
    column.NCOMP = nComp

    column.COL_LENGTH = 0.014
    column.COL_RADIUS = 0.0035
    column.CROSS_SECTION_AREA = np.pi * column.COL_RADIUS**2
    
    column.NPARTYPE = kwargs.get('npartype', 1)
    
    column.COL_POROSITY = kwargs.get('COL_POROSITY', [0.37])
    
    if column.NPARTYPE > 0:
        column.PAR_TYPE_VOLFRAC = kwargs.get('par_type_volfrac', 1.0)
        # column.PAR_TYPE_VOLFRAC_MULTIPLEX = 0

    column.COL_DISPERSION_AXIAL = 5.75e-8
    column.COL_DISPERSION_RADIAL = kwargs.get('col_dispersion_radial', 5e-8)
    
    for parType in range(column.NPARTYPE):
        
        groupName = 'particle_type_' + str(parType).zfill(3)
        
        column[groupName].has_film_diffusion = 1

        # binding parameters
        if column.NPARTYPE > 1:
            
            column[groupName].FILM_DIFFUSION = kwargs['film_diffusion'][parType]
            column[groupName].has_pore_diffusion = kwargs['pore_diffusion'][parType] > 0.0
            column[groupName].PAR_RADIUS = kwargs['par_radius'][parType]
            column[groupName].PAR_POROSITY = kwargs['par_porosity'][parType]
            column[groupName].nbound = [kwargs['nbound'][parType]]
            column[groupName].adsorption_model = kwargs['adsorption_model'][parType]
            column[groupName].PORE_DIFFUSION = kwargs['pore_diffusion'][parType]
            
            if not column[groupName].nbound == [0]:
                column[groupName].has_surface_diffusion = kwargs['surface_diffusion'][parType] > 0.0
                column[groupName].SURFACE_DIFFUSION = kwargs['surface_diffusion'][parType]
                column[groupName].init_cs = [kwargs['init_cs'][parType]]
            else:
                column[groupName].has_surface_diffusion = 0
            column[groupName].adsorption.is_kinetic = kwargs['adsorption.is_kinetic'][parType]
            column[groupName].adsorption.lin_ka = kwargs['adsorption.lin_ka'][parType]
            column[groupName].adsorption.lin_kd = kwargs['adsorption.lin_kd'][parType]
            column[groupName].init_cp = kwargs['init_cp'][parType]
        else:
            column[groupName].has_pore_diffusion = kwargs.get('pore_diffusion', 6.07e-11) > 0.0
            column[groupName].has_surface_diffusion = kwargs.get('surface_diffusion', 0.0) > 0.0
            column[groupName].PAR_RADIUS = kwargs.get('par_radius', 45E-6)
            column[groupName].PAR_POROSITY = kwargs.get('par_porosity', 0.75)
            column[groupName].FILM_DIFFUSION = kwargs.get('film_diffusion', 6.9e-6)
            column[groupName].PORE_DIFFUSION = kwargs.get('pore_diffusion', 6.07e-11)
            column[groupName].SURFACE_DIFFUSION = kwargs.get('surface_diffusion', 0.0)
            column[groupName].adsorption_model = kwargs.get('adsorption_model', 'LINEAR')
            column[groupName].nbound = kwargs.get('nbound', [1])
            column[groupName].adsorption.is_kinetic = kwargs.get('adsorption.is_kinetic', 0)
            column[groupName].adsorption.lin_ka = kwargs.get('adsorption.lin_ka', 35.5)
            column[groupName].adsorption.lin_kd = kwargs.get('adsorption.lin_kd', 1.0)
            column[groupName].init_cp = kwargs.get('init_cp', [0])
            column[groupName].init_cs = kwargs.get('init_cs', [0])
        if parMethod > 0:
            column[groupName].discretization.SPATIAL_METHOD = 'DG'
            column[groupName].discretization.PAR_POLYDEG = parMethod
            column[groupName].discretization.PAR_NELEM = parNElem
        elif parMethod == 0:
            column[groupName].discretization.SPATIAL_METHOD = 'FV'
            column[groupName].discretization.NCELLS = parNElem
            column[groupName].discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'

    if 'INIT_C' in kwargs:

        rad_coords = np.zeros(nRadPoints)
        ax_coords = np.zeros((axMethod+1)*axNElem)
        ax_delta = column.COL_LENGTH / axNElem
        rad_delta = column.COL_RADIUS / radNElem

        if axMethod > 0:
            ax_nodes, _ = helper.lgl_nodes_weights(axMethod)
            rad_nodes, _ = helper.lgl_nodes_weights(radMethod)

            for idx in range(axNElem):
                ax_coords[idx * (axMethod+1): (idx + 1) * (axMethod+1)
                          ] = convergence.map_xi_to_z(ax_nodes, idx, ax_delta)
            for idx in range(radNElem):
                rad_coords[idx * (radMethod+1): (idx + 1) * (radMethod+1)
                           ] = convergence.map_xi_to_z(rad_nodes, idx, rad_delta)
        else:
            ax_coords = np.array([idx * ax_delta for idx in range(axNElem)])
            rad_coords = np.array([rad_delta / 2.0 + idx * rad_delta for idx in range(radNElem)])

        column.init_c = kwargs['INIT_C'](ax_coords, rad_coords)
    else:
        column.init_c = [0] * nComp

    if axMethod > 0:
        column.discretization.SPATIAL_METHOD = "DG"
        if re.search("2D", column.UNIT_TYPE):
            column.discretization.AX_POLYDEG = axMethod
            column.discretization.AX_NELEM = axNElem
            column.discretization.RAD_POLYDEG = radMethod
            column.discretization.RAD_NELEM = radNElem
        else:
            column.discretization.POLYDEG = axMethod
            column.discretization.NELEM = axNElem
            column.discretization.POLYNOMIAL_INTEGRATION_TYPE = 1
    elif axMethod == 0:
        column.discretization.SPATIAL_METHOD = "FV"
        column.discretization.NCOL = axNElem
        column.discretization.NRAD = radNElem
        column.discretization.SCHUR_SAFETY = 1.0e-8
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = 3
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10

    if axMethod >= 0:
        column.discretization.USE_ANALYTIC_JACOBIAN = True
    column.discretization.RADIAL_DISC_TYPE = 'EQUIDISTANT'
    column.PORTS = nRadPoints

    inletUnit = Dict()

    inletUnit.INLET_TYPE = 'PIECEWISE_CUBIC_POLY'
    inletUnit.UNIT_TYPE = 'INLET'
    inletUnit.NCOMP = nComp
    inletUnit.sec_000.CONST_COEFF = [kwargs.get('INLET_CONST', 1.0)] * nComp
    inletUnit.sec_001.CONST_COEFF = [0.0] * nComp
    inletUnit.ports = 1
    
    # define cadet model using the unit-dicts above
    model = Dict()

    model.model.nunits = 1 + nInlets + nOutlets

    # Store solution
    model['return'].split_components_data = 0
    model['return'].split_ports_data = kwargs.get('SPLIT_PORTS_DATA', 1)
    model['return']['unit_000'].WRITE_SOLUTION_INLET = kwargs.get(
        'WRITE_SOLUTION_INLET', 0)
    model['return']['unit_000'].WRITE_SOLUTION_FLUX = kwargs.get(
        'WRITE_SOLUTION_FLUX', 0)
    model['return']['unit_000'].WRITE_SOLUTION_OUTLET = kwargs.get(
        'WRITE_SOLUTION_OUTLET', 1)
    model['return']['unit_000'].WRITE_SOLUTION_BULK = kwargs.get(
        'WRITE_SOLUTION_BULK', 0)
    model['return']['unit_000'].WRITE_SOLUTION_PARTICLE = kwargs.get(
        'WRITE_SOLUTION_PARTICLE', 0)
    model['return']['unit_000'].WRITE_SOLUTION_SOLID = kwargs.get(
        'WRITE_SOLUTION_SOLID', 0)
    model['return']['unit_000'].WRITE_COORDINATES = 1
    model['return']['unit_000'].WRITE_SENS_OUTLET = kwargs.get(
        'WRITE_SENS_OUTLET', 0)

    # Tolerances for the time integrator
    if axMethod >= 0:
        model.solver.time_integrator.USE_MODIFIED_NEWTON = kwargs.get(
            'USE_MODIFIED_NEWTON', 0)
        model.solver.time_integrator.ABSTOL = 1e-6
        model.solver.time_integrator.ALGTOL = 1e-10
        model.solver.time_integrator.RELTOL = 1e-6
        model.solver.time_integrator.INIT_STEP_SIZE = 1e-6
        model.solver.time_integrator.MAX_STEPS = 1000000
    
        # Solver settings
        model.model.solver.GS_TYPE = 1
        model.model.solver.MAX_KRYLOV = 0
        model.model.solver.MAX_RESTARTS = 10
        model.model.solver.SCHUR_SAFETY = 1e-8
    
        # Run the simulation on single thread
        model.solver.NTHREADS = 1
        model.solver.CONSISTENT_INIT_MODE = 3
    
    # Sections
    model.solver.sections.NSEC = 2
    model.solver.sections.SECTION_TIMES = [0.0, 10.0, 1500.0]

    
    # Note: this velocity is only applied to the first zone.
    # Other zones might have different velocity depending on the porosity
    zone0Velocity = 3.45 / (100.0 * 60.0)  # 3.45 cm/min
    column.VELOCITY = zone0Velocity
    
    # get connections matrix
    if re.search("2D", column.UNIT_TYPE):
        connections, rad_coords = helper.generate_connections_matrix(
            rad_method=radMethod, rad_cells=radNElem,
            velocity=zone0Velocity, porosity=column.COL_POROSITY[0], col_radius=column.COL_RADIUS,
            add_inlet_per_port=nInlets, add_outlet=True
        )

    else:
        Q = np.pi * column.COL_RADIUS**2 * zone0Velocity
        connections = [1, 0, -1, -1, Q]
        rad_coords = [column.COL_RADIUS / 2.0]

    outletUnit = Dict()
    outletUnit.UNIT_TYPE = 'OUTLET'
    outletUnit.NCOMP = nComp
            
    # Set units
    model.model['unit_000'] = column
    model.model['unit_001'] = copy.deepcopy(inletUnit)

    if kwargs.get('rad_inlet_profile', None) is None:
        for rad in range(max(1, nRadialZones)):

            model.model['unit_' + str(rad + 1).zfill(3)
                        ] = copy.deepcopy(inletUnit)

            constCoeff = kwargs['inlet_function'](rad)

            model.model['unit_' + str(rad + 1).zfill(
                3)].sec_000.CONST_COEFF = constCoeff

            model.model['unit_' + str(nRadialZones + 1 + rad).zfill(3)] = copy.deepcopy(outletUnit)
            model['return']['unit_' + str(nRadialZones + 1 + rad).zfill(3)] = model['return']['unit_000']

    else:
        for rad in range(nRadPoints):

            model.model['unit_' + str(rad + 1).zfill(3)
                        ] = copy.deepcopy(inletUnit)

            model.model['unit_' + str(rad + 1).zfill(
                3)].sec_000.CONST_COEFF = [kwargs['rad_inlet_profile'](rad_coords[rad], column.COL_RADIUS)] * nComp

            model.model['unit_' + str(nRadPoints + 1 + rad).zfill(3)] = copy.deepcopy(outletUnit)
            model['return']['unit_' + str(nRadPoints + 1 + rad).zfill(3)] = model['return']['unit_000']

    model.model.connections.NSWITCHES = 1
    model.model.connections.switch_000.SECTION = 0
    model.model.connections.switch_000.connections = connections
    model.model.connections.connections_include_ports = 1

    model.solver.sections.SECTION_CONTINUITY = [0,]
    model.solver.USER_SOLUTION_TIMES = np.linspace(0, 1500, 1501)

    return {'input': model}
