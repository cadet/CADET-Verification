# -*- coding: utf-8 -*-
"""

This file contains the software verification code for the FV implementation of
the 2DGRM. The results of this convergence analysis are published in Rao et al.
    'Two-dimensional general rate model with particle size distribution in CADET
    calibrated with high-definition CFD simulated intra-column data' (2025)

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


# %% Reference data paths
_reference_data_path_ = str(
    Path(__file__).resolve().parent.parent / 'data' / 'CASEMA_reference'
)

# _reference_data_path_ = str(
#     Path(__file__).resolve().parent.parent / 'data' / 'CADET-Core_reference'
# )


# %% We define multiple settings convering binding modes, surface diffusion and
# multiple particle types. All settings consider three radial zones.
# An analytical solution can be provided and the EOC is computed for three
# radial zones. Ultimately, the discrete maximum norm of the zonal errors is
# considered to compute the EOC.


def get_settings(small_test):
    return [
        {  # PURE COLUMN TRANSPORT CASE
            'npartype': 0,
            # 'col_dispersion_radial' : 0.0,
            'nRadialZones': 2,
            'COL_POROSITY': np.linspace(0.35, 0.5, 2),
            'name': '2DDPFR2Zone_radEps_1Comp',
            # 'reference': convergence.get_solution(
            #     _reference_data_path_ + '/transport/2DDPFR2Zone_1Comp_DG_axP3Z32_radP3Z16.h5', unit='unit_003', which='outlet'
            # )
            'reference': None,
            'inlet_function': partial(helper.constInlet,
                                      const=1.0)
        }
    ]


def GRM2D_linBnd_tests(
        n_jobs, small_test,
        output_path, cadet_path,
        rerun_sims=True):

    os.makedirs(output_path, exist_ok=True)

    # To test only a subset of settings, comment out the corresponding ref_file_name and the setup in `get_settings`
    
    ref_file_names = [
        None
        ]


    # %% Define benchmarks

    def refine_disc_radEps(
            config_data, disc_idx, setting_name,
            spatial_discretization,
            time_integrator=None,
            unit_id = '001',
            only_return_name=False,
            **kwargs
            ):

        config_copy = copy.deepcopy(config_data)

        # update discretization
        
        if time_integrator is not None:
            config_copy['input']['solver']['time_integrator'] = time_integrator

        axNElem = spatial_discretization['AX_NELEM'] * 2** (disc_idx)
        radNElem = spatial_discretization['RAD_NELEM'] * 2** (disc_idx)
        rad_method = config_copy['input']['model']['unit_' + unit_id]['discretization']['RAD_POLYDEG']

        config_copy['input']['model']['unit_' + unit_id]['discretization'].update(spatial_discretization)
        config_copy['input']['model']['unit_' + unit_id]['discretization']['AX_NELEM'] = axNElem
        config_copy['input']['model']['unit_' + unit_id]['discretization']['RAD_NELEM'] = radNElem
        
        # replicate zonal parameters for each element
        
        colPorosity = []
        
        for zoneIdx in range(kwargs['nRadialZones']):
            
            epsB = config_copy['input']['model']['unit_' + unit_id].COL_POROSITY[zoneIdx]
            
            for elemIdx in range(int(radNElem / kwargs['nRadialZones'])):
            
                colPorosity.append(epsB)
            
        config_copy['input']['model']['unit_' + unit_id].COL_POROSITY = colPorosity

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
                                                   unit_id].COL_POROSITY[0],
            col_radius=config_copy['input']['model']['unit_' +
                                                     unit_id].COL_RADIUS,
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
            spatial_discretization['AX_POLYDEG'], spatial_discretization['AX_NELEM'],
            spatial_discretization['RAD_POLYDEG'], spatial_discretization['RAD_NELEM']
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
        'USE_MODIFIED_NEWTON' : False,
        'init_step_size' : 1e-10,
        'max_steps' : 1000000
        }
    
    spatial_discretization = {
        'AX_POLYDEG': 3, 'AX_NELEM': 4, 
        'RAD_POLYDEG': 3, 'RAD_NELEM': 2, 
        'SPATIAL_METHOD' : 'DG',
        'USE_ANALYTIC_JACOBIAN': True, 'USE_MODIFIED_NEWTON' : False
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

        nDisc = 3 if small_test else 4
        nRadialZones = kwargs['nRadialZones']

        benchmark_config = {
            'cadet_config_jsons': [
                setting_Col2D_lin_1comp_benchmark1.get_model(
                    radNElem=nRadialZones,
                    rad_inlet_profile=None,
                    USE_MODIFIED_NEWTON=0, axMethod=3, **kwargs)
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
                'outlet' # outlet_port_000
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
                partial(refine_disc_radEps,
                         setting_name=kwargs['name'],
                         spatial_discretization=copy.deepcopy(spatial_discretization),
                         time_integrator=time_integrator_2dgrm,
                         nRadialZones=nRadialZones
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
        ax_methods=ax_methods, ax_discs=ax_discs,
        rad_methods=rad_methods, rad_discs=rad_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rad_inlet_profile=None,
        rerun_sims=True,
        refinement_IDs=refinement_IDs,
        disc_refinement_functions=disc_refinement_functions
    )

    # We compute the discrete norm of the errors from each zone

    def copy_json_file(source_file, destination_file):
        try:
            shutil.copy(source_file, destination_file)
        except FileNotFoundError:
            print(f"File {source_file} not found!")
        except Exception as e:
            print(f"An error occurred: {e}")

    def rename_json_file(original_file, new_file):
        try:
            os.replace(original_file, new_file)
        except FileNotFoundError:
            print(f"File {original_file} not found!")
        except Exception as e:
            print(f"An error occurred: {e}")

    for settingIdx in range(len(ref_file_names)):
        
        # save old results under new name for corresponding port
        old_name = str(output_path) + '/convergence_' + \
            settings[settingIdx]['name'] + '.json'
        new_name = str(output_path) + '/convergence_' + 'port' + \
            str(0).zfill(3) + '_' + settings[settingIdx]['name'] + '.json'
        rename_json_file(old_name, new_name)
        
        nRadialZones = settings[settingIdx]['nRadialZones']

        for target_zone in range(1, nRadialZones):

            # get the references at the other ports
            tmp_ref_files = [
                [convergence.get_solution(
                    _reference_data_path_ + '/' + ref_file_names[settingIdx], unit='unit_' + str(nRadialZones + 1 + target_zone).zfill(3), which='outlet'
                )]
            ]

            unit_IDs = [str(nRadialZones + 1 + target_zone).zfill(3)]
    
            bench_func.run_convergence_analysis(
                output_path=output_path,
                cadet_path=cadet_path,
                cadet_configs=[cadet_configs[settingIdx]],
                cadet_config_names=[config_names[settingIdx]],
                include_sens=[include_sens[settingIdx]],
                ref_files=tmp_ref_files,
                unit_IDs=unit_IDs,
                which=[which[settingIdx]],
                ax_methods=[ax_methods[settingIdx]], ax_discs=[ax_discs[settingIdx]],
                rad_methods=[rad_methods[settingIdx]], rad_discs=[rad_discs[settingIdx]],
                par_methods=[par_methods[settingIdx]], par_discs=[par_discs[settingIdx]],
                idas_abstol=[idas_abstol[settingIdx]],
                n_jobs=n_jobs,
                rad_inlet_profile=None,
                rerun_sims=False,
                refinement_IDs=[refinement_IDs[settingIdx]]
            )
    
            # save new results under new name for corresponding port
            old_name = str(output_path) + '/convergence_' + \
                settings[settingIdx]['name'] + '.json'
            new_name = str(output_path) + '/convergence_' + 'port' + \
                str(target_zone).zfill(3) + '_' + \
                settings[settingIdx]['name'] + '.json'
            rename_json_file(old_name, new_name)

    # Calculate Discrete Maximum Norm over all radial zones

    for settingIdx in range(len(settings)):

        # create target file based off the first file
        target_name = str(output_path) + '/convergence_' + \
            settings[settingIdx]['name'] + '.json'
        copy_name = str(output_path) + '/convergence_' + \
            'port000_' + settings[settingIdx]['name'] + '.json'
        copy_json_file(copy_name, target_name)

        nRadialZones = settings[settingIdx]['nRadialZones']

        for target_zone in range(nRadialZones):

            file_name = str(output_path) + '/convergence_' + 'port' + \
                str(target_zone).zfill(3) + '_' + \
                settings[settingIdx]['name'] + '.json'

            with open(file_name, "r") as file:
                data = json.load(file)

            if target_zone == 0:
                disc = data['convergence']['DG_P3']['outlet']['$N_e^z$']
                maxError = np.array(
                    data['convergence']['DG_P3']['outlet']['Max. error'])
                L1Error = np.array(
                    data['convergence']['DG_P3']['outlet']['$L^1$ error'])
                L2Error = np.array(
                    data['convergence']['DG_P3']['outlet']['$L^2$ error'])
            else:  # maximum norm
                maxError = np.maximum(maxError, np.array(
                    data['convergence']['DG_P3']['outlet']['Max. error']))
                L1Error = np.maximum(L1Error, np.array(
                    data['convergence']['DG_P3']['outlet']['$L^1$ error']))
                L2Error = np.maximum(L2Error, np.array(
                    data['convergence']['DG_P3']['outlet']['$L^2$ error']))

        maxEOC = np.insert(
            convergence.calculate_eoc(disc, maxError), 0, 0.0)
        L1EOC = np.insert(convergence.calculate_eoc(disc, L1Error), 0, 0.0)
        L2EOC = np.insert(convergence.calculate_eoc(disc, L2Error), 0, 0.0)

        with open(target_name, "r") as file:
            target_data = json.load(file)

        target_data['convergence']['DG_P3']['outlet']['Max. error'] = maxError.tolist()
        target_data['convergence']['DG_P3']['outlet']['Max. EOC'] = maxEOC.tolist()
        target_data['convergence']['DG_P3']['outlet']['$L^1$ error'] = L1Error.tolist()
        target_data['convergence']['DG_P3']['outlet']['$L^1$ EOC'] = L1EOC.tolist()
        target_data['convergence']['DG_P3']['outlet']['$L^2$ error'] = L2Error.tolist()
        target_data['convergence']['DG_P3']['outlet']['$L^2$ EOC'] = L2EOC.tolist()

        print("2D chromatography convergence for setting no. ", settingIdx)
        print(target_data)
        
        with open(target_name, "w") as file:
            json.dump(target_data, file, indent=4)
        
        new_name = str(output_path) + '/convergence_portsMaxNorm_' + \
            settings[settingIdx]['name'] + '.json'
        
        rename_json_file(target_name, new_name)
