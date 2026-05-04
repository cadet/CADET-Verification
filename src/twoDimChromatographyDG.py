# -*- coding: utf-8 -*-
"""

This file contains the software verification code for the FV implementation of
the 2DGRM. The results of this convergence analysis are published in Rao et al.
    'Two-dimensional general rate model with particle size distribution in CADET
    calibrated with high-definition CFD simulated intra-column data' (2025)

"""

# %% import packages and files
import os
import numpy as np
import json
import shutil
from pathlib import Path

import src.utility.convergence as convergence
import src.bench_configs as bench_configs
import src.bench_func as bench_func
from src.benchmark_models import settings_2Dchromatography


# %% Reference data paths
_reference_data_path_ = str(
    Path(__file__).resolve().parent.parent / 'data' / 'CASEMA_reference'
)

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
            # If set to true, solution time 0.0 is ignored since its not computed by the analytical solution (CADET-Semi-Analytic)
            'nRadialZones': 2,
            'name': '2DDPFR2Zone_1Comp',
            'reference': convergence.get_solution(
                _reference_data_path_ + '/2DDPFR3Zone_1comp.h5', unit='unit_000', which='outlet_port_' + str(0).zfill(3)
            )
        },
        # {  # 1parType, dynamic binding, no surface diffusion
        #     'nRadialZones': nRadialZones,
        #     'name': '2DGRM3Zone_dynLin_1Comp',
        #     'par_method': 3,
        #     'adsorption_model': 'LINEAR',
        #     'adsorption.is_kinetic': 1,
        #     'surface_diffusion': 0.0,
        #     'reference': convergence.get_solution(
        #         _reference_data_path_ + '/2DGRM3Zone_dynLin_1Comp.h5', unit='unit_000', which='outlet_port_' + str(0).zfill(3)
        #     )
        # },
        # {  # 1parType, dynamic binding, with surface diffusion
        #     'nRadialZones': nRadialZones,
        #     'name': '2DGRMsd3Zone_dynLin_1Comp',
        #     'par_method': 3,
        #     'adsorption_model': 'LINEAR',
        #     'adsorption.is_kinetic': 1,
        #     'surface_diffusion': 1e-11,
        #     'reference': convergence.get_solution(
        #         _reference_data_path_ + '/2DGRMsd3Zone_dynLin_1Comp.h5', unit='unit_000', which='outlet_port_' + str(0).zfill(3)
        #     )
        # },
        # {  # 1parType, req binding, no surface diffusion
        #     'nRadialZones': nRadialZones,
        #     'name': '2DGRM3Zone_reqLin_1Comp',
        #     'par_method': 3,
        #     'adsorption_model': 'LINEAR',
        #     'adsorption.is_kinetic': 0,
        #     'surface_diffusion': 0.0,
        #     'init_cp': [0.0],
        #     'init_cs': [0.0],
        #     'reference': convergence.get_solution(
        #         _reference_data_path_ + '/2DGRM3Zone_reqLin_1Comp.h5', unit='unit_000', which='outlet_port_' + str(0).zfill(3)
        #     )
        # },
        # {  # 1parType, req binding, with surface diffusion
        #     'nRadialZones': nRadialZones,
        #     'name': '2DGRMsd3Zone_reqLin_1Comp',
        #     'par_method': 3,
        #     'adsorption_model': 'LINEAR',
        #     'adsorption.is_kinetic': 0,
        #     'surface_diffusion': 1e-11,
        #     'init_cp': [0.0],
        #     'init_cs': [0.0],
        #     'reference': convergence.get_solution(
        #         _reference_data_path_ + '/2DGRMsd3Zone_reqLin_1Comp.h5', unit='unit_000', which='outlet_port_' + str(0).zfill(3)
        #     )
        # },
        # {  # 4parType:
        #     'nRadialZones': nRadialZones,
        #     'name': '2DGRM2parType3Zone_1Comp' if small_test else '2DGRM4parType3Zone_1Comp',
        #     'par_method': 3,
        #     'npartype': 2 if small_test else 4,
        #     'par_type_volfrac': [0.5, 0.5] if small_test else [0.3, 0.35, 0.15, 0.2],
        #     'par_radius': [45E-6, 75E-6] if small_test else [45E-6, 75E-6, 25E-6, 60E-6],
        #     'par_porosity': [0.75, 0.7] if small_test else [0.75, 0.7, 0.8, 0.65],
        #     'nbound': [1, 1] if small_test else [1, 1, 0, 1],
        #     'init_cp': [0.0, 0.0] if small_test else [0.0, 0.0, 0.0, 0.0],
        #     'init_cs': [0.0, 0.0] if small_test else [0.0, 0.0, 0.0, 0.0],
        #     'film_diffusion': [6.9E-6, 6E-6] if small_test else [6.9E-6, 6E-6, 6.5E-6, 6.7E-6],
        #     'pore_diffusion': [5E-11, 3E-11] if small_test else [6.07E-11, 5E-11, 3E-11, 4E-11],
        #     'surface_diffusion': [5E-11, 0.0] if small_test else [1E-11, 5E-11, 0.0, 0.0],
        #     'adsorption_model': ['LINEAR', 'LINEAR'] if small_test else ['LINEAR', 'LINEAR', 'NONE', 'LINEAR'],
        #     'adsorption.is_kinetic': [0, 1] if small_test else [0, 1, 0, 0],
        #     'adsorption.lin_ka': [35.5, 4.5] if small_test else [35.5, 4.5, 0, 0.25],
        #     'adsorption.lin_kd': [1.0, 0.15] if small_test else [1.0, 0.15, 0, 1.0],
        #     'reference': convergence.get_solution(
        #         _reference_data_path_ + '/2DGRM2parType3Zone_1Comp.h5' if small_test else _reference_data_path_ + '/2DGRM4parType3Zone_1Comp.h5',
        #         unit='unit_000', which='outlet_port_' + str(0).zfill(3)
        #     )
        # }
    ]


def GRM2D_linBnd_tests(
        n_jobs, small_test,
        output_path, cadet_path,
        rerun_sims=True):

    os.makedirs(output_path, exist_ok=True)

    # To test only a subset of settings, comment out the corresponding ref_file_name and the setup in `get_settings`
    
    ref_file_names = [
        '2DDPFR2Zone_1Comp.h5',
        '2DGRM3Zone_dynLin_1Comp.h5',
        '2DGRMsd3Zone_dynLin_1Comp.h5',
        '2DGRM3Zone_reqLin_1Comp.h5',
        '2DGRMsd3Zone_reqLin_1Comp.h5',
        '2DGRM2parType3Zone_1Comp.h5' if small_test else '2DGRM4parType3Zone_1Comp.h5'
        ]


    # %% Define benchmarks

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

    def GRM2D_DG_Benchmark(small_test=False, **kwargs):

        nDisc = 3 if small_test else 4
        nRadialZones = kwargs['nRadialZones']

        benchmark_config = {
            'cadet_config_jsons': [
                settings_2Dchromatography.GRM2D_linBnd_benchmark1(
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
            ]
        }

        return benchmark_config

    # %% create benchmark configurations

    for setting in [settings[0]]:
        
        addition = GRM2D_DG_Benchmark(small_test=small_test, **setting)

        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
            par_methods=par_methods, par_discs=par_discs,
            idas_abstol=idas_abstol,
            refinement_IDs=refinement_IDs,
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
        rerun_sims=rerun_sims,
        refinement_IDs=refinement_IDs
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
                    _reference_data_path_ + '/' + ref_file_names[settingIdx], unit='unit_000', which='outlet_port_' + str(target_zone).zfill(3)
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
