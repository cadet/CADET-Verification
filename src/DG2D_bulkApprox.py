# -*- coding: utf-8 -*-
"""

"""

# %% import packages and files
import os
import copy
import numpy as np
import json
import shutil
from pathlib import Path
from functools import partial

from cadet import Cadet

import src.utility.convergence as convergence
import src.bench_configs as bench_configs
import src.bench_func as bench_func
from src.benchmark_models import setting_Col2D_lin_1comp_benchmark1
import src.benchmark_models.helper_setup_2Dmodels as helper


_reference_data_path_ = str(
    Path(__file__).resolve().parent.parent / 'data' / 'CADET-Core_reference'
)

def init_c(x, rho, col_length, col_radius):

    # normalize coordinates to [0, 1]
    x_hat = x / col_length
    r_hat = rho / col_radius

    # smooth single hill: zero at boundaries, max at center
    c = (np.sin(np.pi * x_hat) ** 2) * (np.sin(np.pi * r_hat) ** 2)

    return c

def get_settings(small_test):
    return [
        {
            'npartype': 0,
            'nRadialZones': 2,
            'name': '2DDPFR2Zone_bulkApprox_1Comp',
            # 'reference': convergence.get_solution(
            #     _reference_data_path_ + '/transport/2DDPFR2Zone_radEps_1Comp_DG_axP3Z64_radP3Z32.h5', unit='unit_003', which='outlet'
            # ),
            'reference': None,
            'inlet_function': partial(helper.constInlet,
                                      const=0.0),
            'velocity': 0.0,
            'col_dispersion_axial': 0.0,
            'col_dispersion_radial': 0.0,
            'init_function': partial(init_c, col_length = 0.014, col_radius = 0.0035),
            'WRITE_SOLUTION_BULK': True,
            'tEnd': 0.1, 'tSec1': 0.05,
            'USER_SOLUTION_TIMES': [0.1]
        }
    ]


def GRM2D_linBnd_tests(
        n_jobs, small_test,
        output_path, cadet_path,
        rerun_sims=True):

    os.makedirs(output_path, exist_ok=True)

    ref_file_names = [
        None
        # 'transport/2DDPFR2Zone_radEps_1Comp_DG_axP3Z64_radP3Z32.h5'
        ]


    # %% Define benchmarks

    def refine_disc_bulkApprox(
            config_data, disc_idx, setting_name,
            spatial_discretization,
            time_integrator=None,
            unit_id = '000',
            only_return_name=False,
            **kwargs
            ):

        config_copy = copy.deepcopy(config_data)

        # update discretization
        
        if time_integrator is not None:
            config_copy['input']['solver']['time_integrator'] = time_integrator

        axNElem = spatial_discretization['AX_NELEM'] * 2** (disc_idx)
        radNElem = spatial_discretization['RAD_NELEM'] * 2** (disc_idx)
        ax_method = spatial_discretization['AX_POLYDEG']
        rad_method = spatial_discretization['RAD_POLYDEG']
        
        config_copy['input']['model']['unit_' + unit_id]['discretization'].update(spatial_discretization)
        config_copy['input']['model']['unit_' + unit_id]['discretization']['AX_NELEM'] = axNElem
        config_copy['input']['model']['unit_' + unit_id]['discretization']['RAD_NELEM'] = radNElem
        
        col_length = config_copy['input']['model']['unit_' + unit_id]['COL_LENGTH']
        col_radius = config_copy['input']['model']['unit_' + unit_id]['COL_RADIUS']

        config_copy['input']['model']['unit_' + unit_id].init_state = helper.init_state_2D(
            method=ax_method, axNElem=axNElem, radNElem=radNElem,
            L=col_length, RHO=col_radius,
            init_function=kwargs['init_function']
        )

        # replicate zonal parameters for each element
        
        colPorosity = []
        
        for zoneIdx in range(kwargs['nRadialZones']):
            
            epsB = config_copy['input']['model']['unit_' + unit_id].COL_POROSITY[zoneIdx]
            
            for elemIdx in range(int(radNElem / kwargs['nRadialZones'])):
            
                colPorosity.append(epsB)
            
        config_copy['input']['model']['unit_' + unit_id].COL_POROSITY = colPorosity

        constant_velocity = np.isscalar(colPorosity) or len(colPorosity) == 1

        # update connections
        
        config_copy['input']['model']['unit_'+ unit_id].PORTS = (rad_method + 1 ) * radNElem

        n_units = config_copy['input']['model']['nunits']
        nInlets = int((n_units - 1) / 2)
        add_inlet_per_port = nInlets
        
        config_copy['input']['model'].nunits = n_units
                
        connections, rad_coords = helper.generate_connections_matrix(
            rad_method=rad_method, rad_cells=radNElem,
            velocity=config_copy['input']['model']['unit_' +
                                                   unit_id].VELOCITY,
            porosity=config_copy['input']['model']['unit_' +
                                                   unit_id].COL_POROSITY,
            col_radius=config_copy['input']['model']['unit_' +
                                                     unit_id].COL_RADIUS,
            constant_velocity=constant_velocity,
            add_inlet_per_port=add_inlet_per_port, add_outlet=True
        )

        if add_inlet_per_port is True:
            for rad in range(unit_id * (rad_method + 1)):
        
                config_copy['input']['model']['unit_' +
                                              str(rad + 1).zfill(3)] = copy.deepcopy(config_copy['input']['model']['unit_001'])

                if kwargs.get('rad_inlet_profile', None) is not None:
                    config_copy['input']['model']['unit_001'].sec_000.CONST_COEFF = kwargs['rad_inlet_profile'](
                        rad_coords[rad], config_copy['input']['model']['unit_000'].COL_RADIUS)

        config_copy['input'].model.connections.switch_000.connections = connections
    
        # create and return object
        
        config_name = convergence.generate_2D_name(
            setting_name,
            spatial_discretization['AX_POLYDEG'], axNElem,
            spatial_discretization['RAD_POLYDEG'], radNElem
            )
        
        model = Cadet()
        model.root.input = config_copy['input']
        
        if output_path is not None:

            model.filename = str(output_path) + '/' + config_name

            if only_return_name:
                return model.filename
            else:
                model.save()
                return model

    time_integrator_2dgrm = {
        'ABSTOL' : 1e-10, 'RELTOL' : 1e-8, 'ALGTOL' : 1e-10,
        'USE_MODIFIED_NEWTON' : True,
        'init_step_size' : 1e-10,
        'max_steps' : 1000000
        }
    
    spatial_discretization = {
        'AX_POLYDEG': 3, 'AX_NELEM': 4, 
        'RAD_POLYDEG': 3, 'RAD_NELEM': 2, 
        'SPATIAL_METHOD' : 'DG',
        'USE_ANALYTIC_JACOBIAN': True
        }

    settings = get_settings(small_test)

    cadet_configs = []
    config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    rad_methods = []
    rad_discs = []
    par_methods = []
    par_discs = []
    refinement_IDs = []
    disc_refinement_functions = []

    def GRM2D_DG_Benchmark(small_test=False, **kwargs):

        nDisc = 4 if small_test else 4
        nRadialZones = kwargs['nRadialZones']

        benchmark_config = {
            'cadet_config_jsons': [
                setting_Col2D_lin_1comp_benchmark1.get_model(
                    radNElem=nRadialZones,
                    rad_inlet_profile=None,
                    axMethod=3, **kwargs)
            ],
            'include_sens': [
                False
            ],
            'ref_files': [
                [kwargs.get('reference', None)]
            ],
            'refinement_ID': [
                '000'
            ],
            'unit_IDs': [  # note that we consider radial zone 0
                str(nRadialZones + 1 + 0).zfill(3)
            ],
            'which': [
                'outlet'
            ],
            'idas_abstol': [
                [1e-10]
            ],
            'ax_methods': [
                [3]
            ],
            'ax_discs': [
                [bench_func.disc_list(4, nDisc)]
            ],
            'rad_methods': [
                [3]
            ],
            'rad_discs': [
                [bench_func.disc_list(nRadialZones, nDisc)]
            ],
            'par_methods': [
                [None]
            ],
            'par_discs': [
                [None]
            ],
            'disc_refinement_functions' : [[
                partial(refine_disc_bulkApprox,
                         setting_name=kwargs['name'],
                         spatial_discretization=copy.deepcopy(spatial_discretization),
                         time_integrator=time_integrator_2dgrm,
                         nRadialZones=nRadialZones,
                         init_function=kwargs['init_function']
                         )
                ]]
        }

        return benchmark_config

    # %% create benchmark configurations

    for setting in settings:
        
        addition = GRM2D_DG_Benchmark(small_test=small_test, **setting)

        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
            par_methods=par_methods, par_discs=par_discs,
            idas_abstol=idas_abstol,
            refinement_IDs=refinement_IDs,
            disc_refinement_functions=disc_refinement_functions,
            addition=addition)

        config_names.extend([setting['name']])

    # %% Run convergence analysis

    bench_func.run_convergence_analysis(
        output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods, ax_discs=copy.deepcopy(ax_discs),
        rad_methods=rad_methods, rad_discs=copy.deepcopy(rad_discs),
        par_methods=par_methods, par_discs=copy.deepcopy(par_discs),
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rad_inlet_profile=None,
        rerun_sims=rerun_sims,
        refinement_IDs=refinement_IDs,
        disc_refinement_functions=disc_refinement_functions
    )

# Compute bulk EOC

file_path = r'C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core\2Dchromatography'
simulation_names = [
    '2DDPFR2Zone_bulkApprox_1Comp_DG_axP3Z4_radP3Z2',
    '2DDPFR2Zone_bulkApprox_1Comp_DG_axP3Z8_radP3Z4',
    '2DDPFR2Zone_bulkApprox_1Comp_DG_axP3Z16_radP3Z8',
    '2DDPFR2Zone_bulkApprox_1Comp_DG_axP3Z32_radP3Z16',
]

def compute_bulk_EOC(file_path, simulation_names, unit_ID, which='bulk'):
    
    errors = []

    domain = (0.014, 0.0035)  # col_length, col_radius
    X = np.linspace(0.0, domain[0], 100)
    RHO = np.linspace(0.0, domain[1], 100)
    XX, RR = np.meshgrid(X, RHO, indexing='ij')
    reference = init_c(XX, RR, col_length=domain[0], col_radius=domain[1]).reshape(-1)

    for sim_name in simulation_names:

        sim_path = os.path.join(file_path, sim_name + '.h5')

        solution = convergence.get_solution(sim_path, unit='unit_' + unit_ID, which=which)

        sim = convergence.get_simulation(sim_path)
        ax_coords = convergence.get_axial_coordinates(sim, unit=unit_ID)
        rad_coords = convergence.get_radial_coordinates(sim, unit=unit_ID)

        polyDeg = sim.root.input.model.unit_000.discretization.ax_polydeg
        nCellsX = sim.root.input.model.unit_000.discretization.ax_nelem
        nCellsY = sim.root.input.model.unit_000.discretization.rad_nelem

        # pack coords into shape (N, 2)
        orig_coords = np.array(np.meshgrid(ax_coords, rad_coords)).T.reshape(-1, 2)
        output_coords = np.array(np.meshgrid(X, RHO)).T.reshape(-1, 2)

        print(solution.shape, orig_coords.shape, output_coords.shape)

        interpolated_solution = convergence.get_interpolated_solution_2d(
            orig_values=solution, orig_coords=orig_coords, domain=domain, output_coords=output_coords,
            polyDeg=polyDeg, nCellsX=nCellsX, nCellsY=nCellsY
        )

        print(interpolated_solution.shape)

        error = np.linalg.norm(interpolated_solution - reference) / np.linalg.norm(reference)

        errors.append(error)

    for i in range(1, len(errors)):
        eoc = np.log(errors[i-1] / errors[i]) / np.log(2)
        print(f"Refinement {i}: Error = {errors[i]:.2e}, EOC = {eoc:.2f}")

    # plot reference
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.contourf(X, RHO, reference.reshape(len(X), len(RHO)), levels=50, cmap='viridis')
    plt.colorbar(label='Initial Concentration')
    plt.xlabel('Axial Position (m)')
    plt.ylabel('Radial Position (m)')
    plt.title('Initial Concentration Distribution')
    plt.show()

compute_bulk_EOC(file_path, simulation_names, unit_ID='000', which='bulk')
