# -*- coding: utf-8 -*-
"""

This script defines benchmark configurations, e.g. whether or not to include
sensitivities, the considered numerical methods, etc.
The individual model settings considered are specified in the CADET-Database
github project.

"""

import os
import json
import copy
from functools import partial

import src.bench_func as bench_func
import src.utility.convergence as convergence
from src import analytical
from src import geometry_references

from src.benchmark_models import settings_2Dchromatography
from src.benchmark_models import settings_columnSystems
from src.benchmark_models import setting_Col1D_lin_1comp_benchmark1
from src.benchmark_models import setting_Col1D_SMA_4comp_LWE_benchmark1
from src.benchmark_models import setting_radCol1D_LRM_lin_1comp_benchmark1
from src.benchmark_models import setting_radCol1D_lin_1comp_benchmark1
from src.benchmark_models import setting_COL1D_GRMparType2_dynLin_2comp_benchmark1
from src.benchmark_models import setting_Col1D_XparTypeGR_lin_1comp_benchmark1
from src.benchmark_models import setting_Col1D_langLRM_2comp_benchmark1
from src.benchmark_models import setting_Col1D_pureTransport_1comp_benchmark1
from src.benchmark_models.setting_Col1D_pureTransport_1comp_benchmark1 import create_convergence_object


# %% benchmark templates

# TODO add Langmuir setting used in Breuer et al

_benchmark_settings_ = [
    'full_chromatography_benchmark',
    'chromatography_benchmark_without_GRMLWE',
    'linear_chromatography_benchmark',
    'LWE_chromatography_benchmark',
    'radial_flow_benchmark_fv',
    'radial_flow_benchmark_dg',
    # Individual settings
    'LRM_dynLin_1comp_benchmark1',
    'LRMP_dynLin_1comp_benchmark1',
    'GRM_dynLin_1comp_benchmark1',
    'LRM_reqSMA_4comp_benchmark1',
    'LRMP_reqSMA_4comp_benchmark1',
    'GRM_reqSMA_4comp_benchmark1',
    'LRM_langmuir_2comp_benchmark1'
]

# %%


def run_benchmark(
        setting, disc_methods, cadet_path, output_path, database_path,
        n_jobs=-1, benchmark_size='mid', include_sensitivity=False, ref_files=None,
        N_RUNS=1, **kwargs):
    """ Runs and saves a convergence/performance benchmark.

    Parameters
    ----------
    cadet_path : String 
        absolute path to CADET executable.
    output_path : String 
        absolute path to output folder.
    N_RUNS : Int 
        Number of runs. For multiple runs, the simulation times
        are overwritten by the best/fastest. The saved h5 files will still hold
        the compute times of the last run. Defaults to one
    Returns
    -------
    List of Strings
        Names of result json files.
    """

    if setting not in _benchmark_settings_:
        raise ValueError('Unknown setting ' + str(setting) + '.')

    cadet_config_jsons = []
    include_sens = []
    ref_files_ = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    par_methods = []
    par_discs = []

    for methodIdx in range(len(disc_methods)):
        addition = eval(setting)(
            disc_methods[methodIdx], benchmark_size, include_sensitivity, ref_files)
        add_benchmark(
            cadet_config_jsons, include_sens, ref_files_, unit_IDs, which,
            idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
            addition)

    os.makedirs(output_path, exist_ok=True)

    first_iteration = True

    while True:

        # run convergence benchmark
        results = bench_func.run_convergence_analysis(
            database_path=database_path, output_path=output_path,
            cadet_path=cadet_path,
            cadet_config_jsons=cadet_config_jsons,
            include_sens=include_sens,
            ref_files=ref_files_,
            unit_IDs=unit_IDs,
            which=which,
            ax_methods=ax_methods,
            ax_discs=ax_discs,
            par_methods=par_methods,
            par_discs=par_discs,
            idas_abstol=idas_abstol,
            n_jobs=n_jobs,
            rerun_sims=bool(N_RUNS),
            **kwargs
        )

        if first_iteration:
            sim_times = copy.deepcopy(ax_discs)

        # get simulation times
        for resultIdx in range(len(results)):
            try:
                with open(output_path / results[resultIdx], 'r') as json_file:

                    data = json.load(json_file)['convergence']

                    for methodIdx in range(len(ax_methods[resultIdx])):

                        axMethod = ax_methods[resultIdx][methodIdx]
                        if axMethod == 0:
                            method_string = "FV"
                        else:
                            method_string = "DG_P" + \
                                str(axMethod) if axMethod > 0 else "FV"
                            if par_methods[resultIdx][methodIdx] is not None:
                                method_string += "parP" + \
                                    str(par_methods[resultIdx][methodIdx])

                        sim_times[resultIdx][methodIdx] = data[method_string][which[resultIdx]]['Sim. time']

            except FileNotFoundError:
                raise Exception('file not found')

        if first_iteration:
            sim_times_old = sim_times
            N_RUNS = N_RUNS - 1
            first_iteration = False
            if N_RUNS <= 0:
                break
            else:
                continue
        else:  # sim_times has the new compute times, sim_times_old the old ones
            # get smallest compute time each and write into file!
            sim_times = [[min(a, b) for a, b in zip(sub_list1, sub_list2)]
                         for sub_list1, sub_list2 in zip(sim_times, sim_times_old)]

            for resultIdx in range(len(results)):
                try:
                    with open(output_path / results[resultIdx], 'r') as json_file:

                        json_data = json.load(json_file)

                        for methodIdx in range(len(ax_methods[resultIdx])):

                            axMethod = ax_methods[resultIdx][methodIdx]
                            if axMethod == 0:
                                method_string = "FV"
                            else:
                                method_string = "DG_P" + \
                                    str(axMethod) if axMethod > 0 else "FV"
                                if par_methods[resultIdx][methodIdx] is not None:
                                    method_string += "parP" + \
                                        str(par_methods[resultIdx][methodIdx])

                            json_data['convergence'][method_string][which[resultIdx]
                                                                    ]['Sim. time'] = sim_times[resultIdx][methodIdx]

                    # Write the updated data back to the JSON file
                    with open(output_path / results[resultIdx], 'w') as json_file:
                        json.dump(json_data, json_file, indent=4)

                except FileNotFoundError:
                    raise Exception('file not found')

            sim_times_old = sim_times

        N_RUNS = N_RUNS - 1
        if N_RUNS <= 0:
            break

    return results


def axial_flow_benchmark_fv(small_test=False, sensitivities=False, ref_filepath=None):

    # Load analytical references for linear 1-component benchmarks
    if ref_filepath is not None:
        ref_LRM = convergence.get_solution(ref_filepath+'/CASEMA_reference/LRM_dynLin_1comp_benchmark1.h5')
        ref_LRMP = convergence.get_solution(ref_filepath+'/CASEMA_reference/LRMP_dynLin_1comp_benchmark1.h5')
        ref_GRM = convergence.get_solution(ref_filepath+'/CASEMA_reference/GRM_dynLin_1comp_benchmark1.h5')
        ref_GRMsd = convergence.get_solution(ref_filepath+'/CASEMA_reference/GRMsd_dynLin_1comp_benchmark1.h5')
        ref_LRMlangmuir = convergence.get_solution(ref_filepath+'/CADET-Core_reference/chromatography/LRM_lang_2comp_benchmark1.h5')
    else:
        ref_LRM = None
        ref_LRMP = None
        ref_GRM = None
        ref_GRMsd = None
        ref_LRMlangmuir = None

    n_settings = 9

    benchmark_config = {
        'cadet_config_jsons': [
            setting_Col1D_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=0, particle_type='EQUILIBRIUM_PARTICLE'
                ),
            setting_Col1D_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=0, particle_type='HOMOGENEOUS_PARTICLE'
                ),
            setting_Col1D_lin_1comp_benchmark1.get_model(
               spatial_method_bulk=0, spatial_method_particle=0,
               particle_type='GENERAL_RATE_PARTICLE'
               ),
            setting_Col1D_lin_1comp_benchmark1.get_model(
               spatial_method_bulk=0, spatial_method_particle=0,
               particle_type='GENERAL_RATE_PARTICLE', surface_diffusion=5E-11
               ),
            setting_Col1D_SMA_4comp_LWE_benchmark1.get_model(
                spatial_method_bulk=0, particle_type='EQUILIBRIUM_PARTICLE'
                ),
            setting_Col1D_SMA_4comp_LWE_benchmark1.get_model(
               spatial_method_bulk=0, particle_type='HOMOGENEOUS_PARTICLE'
               ),
           setting_Col1D_SMA_4comp_LWE_benchmark1.get_model(
              spatial_method_bulk=0, spatial_method_particle=0,
              particle_type='GENERAL_RATE_PARTICLE'
              ),
           setting_Col1D_XparTypeGR_lin_1comp_benchmark1.get_model(
               spatial_method_bulk=0, spatial_method_particle=0,
            **{ # 4parType:
                'par_method': 0,
                'npartype': 2 if small_test else 4,
                'par_type_volfrac': [0.5, 0.5] if small_test else [0.3, 0.35, 0.15, 0.2],
                'par_radius': [45E-6, 75E-6] if small_test else [45E-6, 75E-6, 25E-6, 60E-6],
                'par_porosity': [0.75, 0.7] if small_test else [0.75, 0.7, 0.8, 0.65],
                'nbound': [1, 1] if small_test else [1, 1, 0, 1],
                'init_cp': [0.0, 0.0] if small_test else [0.0, 0.0, 0.0, 0.0],
                'init_cs': [0.0, 0.0] if small_test else [0.0, 0.0, 0.0, 0.0],
                'film_diffusion': [6.9E-6, 6E-6] if small_test else [6.9E-6, 6E-6, 6.5E-6, 6.7E-6],
                'pore_diffusion': [5E-11, 3E-11] if small_test else [6.07E-11, 5E-11, 3E-11, 4E-11],
                'surface_diffusion': [5E-11, 0.0] if small_test else [1E-11, 5E-11, 0.0, 0.0],
                'adsorption_model': ['LINEAR', 'LINEAR'] if small_test else ['LINEAR', 'LINEAR', 'NONE', 'LINEAR'],
                'is_kinetic': [0, 1] if small_test else [0, 1, 0, 0],
                'lin_ka': [35.5, 4.5] if small_test else [35.5, 4.5, 0, 0.25],
                'lin_kd': [1.0, 0.15] if small_test else [1.0, 0.15, 0, 1.0]
            }),
            setting_Col1D_langLRM_2comp_benchmark1.get_model(
                spatial_method_bulk=0
                )
        ],
        'cadet_config_names': [
            'LRM_dynLin_1comp_benchmark1',
            'LRMP_dynLin_1comp_benchmark1',
            'GRM_dynLin_1comp_benchmark1',
            'GRMsd_dynLin_1comp_benchmark1',
            'LRM_reqSMA_4comp_benchmark1',
            'LRMP_reqSMA_4comp_benchmark1',
            'GRM_reqSMA_4comp_benchmark1',
            'GRM_4parTypeLin_4comp_benchmark1',
            'LRM_langmuir_2comp_benchmark1'

        ],
        'include_sens': [True] * n_settings if sensitivities else [False] * n_settings,
        'ref_files': [
            [ref_LRM], [ref_LRMP], [ref_GRM], [ref_GRMsd],
            [None], [None], [None], [None], [ref_LRMlangmuir]
        ],
        'unit_IDs': [
            '001', '001', '001', '001', '000', '000', '000', '001', '001'
        ],
        'which': [
            'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet'
        ],
        'idas_abstol': [
            [1e-12], [1e-12], [1e-12], [1e-12], [1e-10], [1e-10], [1e-8], [1e-6], [1e-8]
        ],
        'ax_methods': [
            [0], [0], [0], [0], [0], [0], [0], [0], [0]
        ],
        'ax_discs': [
            [bench_func.disc_list(8, 8 if not small_test else 3)],
            [bench_func.disc_list(8, 8 if not small_test else 3)],
            [bench_func.disc_list(8, 8 if not small_test else 3)],
            [bench_func.disc_list(8, 8 if not small_test else 3)],
            [bench_func.disc_list(8, 6 if not small_test else 3)],
            [bench_func.disc_list(8, 6 if not small_test else 3)],
            [bench_func.disc_list(8, 6 if not small_test else 3)],
            [bench_func.disc_list(8, 4 if not small_test else 3)],
            [bench_func.disc_list(32, 9 if not small_test else 3)]
        ],
        'par_methods': [
            [None], [None], [0], [0], [None], [None], [0], [0], [None]
        ],
        'par_discs': [
            [None],
            [None],
            [bench_func.disc_list(1, 8 if not small_test else 3)],
            [bench_func.disc_list(1, 8 if not small_test else 3)],
            [None],
            [None],
            [bench_func.disc_list(1, 6 if not small_test else 3)],
            [bench_func.disc_list(1, 4 if not small_test else 3)],
            [None]
        ],
        'disc_refinement_functions' : [
            [bench_func.create_object_from_config] for _ in range(n_settings)
            ]
    }

    return benchmark_config


def axial_flow_benchmark_dg(small_test=False, sensitivities=False, ref_filepath=None):

    # Load analytical references for linear 1-component benchmarks
    if ref_filepath is not None:
        ref_LRM = convergence.get_solution(ref_filepath+'/CASEMA_reference/LRM_dynLin_1comp_benchmark1.h5')
        ref_LRMP = convergence.get_solution(ref_filepath+'/CASEMA_reference/LRMP_dynLin_1comp_benchmark1.h5')
        ref_GRM = convergence.get_solution(ref_filepath+'/CASEMA_reference/GRM_dynLin_1comp_benchmark1.h5')
        ref_GRMsd = convergence.get_solution(ref_filepath+'/CASEMA_reference/GRMsd_dynLin_1comp_benchmark1.h5')
        ref_LRMlangmuir = convergence.get_solution(ref_filepath+'/CADET-Core_reference/chromatography/LRM_lang_2comp_benchmark1.h5')
    else:
        ref_LRM = None
        ref_LRMP = None
        ref_GRM = None
        ref_GRMsd = None
        ref_LRMlangmuir = None

    n_settings = 9

    benchmark_config = {
        'cadet_config_jsons': [
            setting_Col1D_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=3, particle_type='EQUILIBRIUM_PARTICLE'
                ),
            setting_Col1D_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=3, particle_type='HOMOGENEOUS_PARTICLE'
                ),
            setting_Col1D_lin_1comp_benchmark1.get_model(
               spatial_method_bulk=3, spatial_method_particle=3,
               particle_type='GENERAL_RATE_PARTICLE'
               ),
            setting_Col1D_lin_1comp_benchmark1.get_model(
               spatial_method_bulk=3, spatial_method_particle=3,
               particle_type='GENERAL_RATE_PARTICLE', surface_diffusion=5E-11
               ),
            setting_Col1D_SMA_4comp_LWE_benchmark1.get_model(
                spatial_method_bulk=3, particle_type='EQUILIBRIUM_PARTICLE'
                ),
            setting_Col1D_SMA_4comp_LWE_benchmark1.get_model(
                spatial_method_bulk=3, particle_type='HOMOGENEOUS_PARTICLE'
                ),
            setting_Col1D_SMA_4comp_LWE_benchmark1.get_model(
               spatial_method_bulk=3, spatial_method_particle=3,
               particle_type='GENERAL_RATE_PARTICLE'
               ),
            setting_Col1D_XparTypeGR_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=0, spatial_method_particle=0,
             **{ # 4parType:
                 'par_method': 0,
                 'npartype': 2 if small_test else 4,
                 'par_type_volfrac': [0.5, 0.5] if small_test else [0.3, 0.35, 0.15, 0.2],
                 'par_radius': [45E-6, 75E-6] if small_test else [45E-6, 75E-6, 25E-6, 60E-6],
                 'par_porosity': [0.75, 0.7] if small_test else [0.75, 0.7, 0.8, 0.65],
                 'nbound': [1, 1] if small_test else [1, 1, 0, 1],
                 'init_cp': [0.0, 0.0] if small_test else [0.0, 0.0, 0.0, 0.0],
                 'init_cs': [0.0, 0.0] if small_test else [0.0, 0.0, 0.0, 0.0],
                 'film_diffusion': [6.9E-6, 6E-6] if small_test else [6.9E-6, 6E-6, 6.5E-6, 6.7E-6],
                 'pore_diffusion': [5E-11, 3E-11] if small_test else [6.07E-11, 5E-11, 3E-11, 4E-11],
                 'surface_diffusion': [5E-11, 0.0] if small_test else [1E-11, 5E-11, 0.0, 0.0],
                 'adsorption_model': ['LINEAR', 'LINEAR'] if small_test else ['LINEAR', 'LINEAR', 'NONE', 'LINEAR'],
                 'is_kinetic': [0, 1] if small_test else [0, 1, 0, 0],
                 'lin_ka': [35.5, 4.5] if small_test else [35.5, 4.5, 0, 0.25],
                 'lin_kd': [1.0, 0.15] if small_test else [1.0, 0.15, 0, 1.0]
             }),
            setting_Col1D_langLRM_2comp_benchmark1.get_model(
                spatial_method_bulk=3
                )
        ],
        'cadet_config_names': [
            'LRM_dynLin_1comp_benchmark1',
            'LRMP_dynLin_1comp_benchmark1',
            'GRM_dynLin_1comp_benchmark1',
            'GRMsd_dynLin_1comp_benchmark1',
            'LRM_reqSMA_4comp_benchmark1',
            'LRMP_reqSMA_4comp_benchmark1',
            'GRM_reqSMA_4comp_benchmark1',
            'GRM_4parTypeLin_4comp_benchmark1',
            'LRM_langmuir_2comp_benchmark1'
        ],
        'include_sens': [True] * n_settings if sensitivities else [False] * n_settings,
        'ref_files': [
            [ref_LRM], [ref_LRMP], [ref_GRM], [ref_GRMsd],
            [None], [None], [None], [None], [ref_LRMlangmuir]
        ],
        'unit_IDs': [
            '001', '001', '001', '001', '000', '000', '000', '001', '001'
        ],
        'which': [
            'outlet'
        ] * n_settings,
        'idas_abstol': [
           [1e-12], [1e-12], [1e-12], [1e-12], [1e-10], [1e-10], [1e-8], [1e-6], [1e-10]
        ],
        'ax_methods': [
            [3], [3], [3], [3], [3], [3], [3], [2], [3]
        ],
        'ax_discs': [
            [bench_func.disc_list(1, 8 if not small_test else 3)],
            [bench_func.disc_list(1, 9 if not small_test else 3)],
            [bench_func.disc_list(8, 5 if not small_test else 3)],
            [bench_func.disc_list(8, 5 if not small_test else 3)],
            [bench_func.disc_list(4, 6 if not small_test else 3)],
            [bench_func.disc_list(4, 6 if not small_test else 3)],
            [bench_func.disc_list(4, 5 if not small_test else 3)],
            [bench_func.disc_list(2, 4 if not small_test else 3)],
            [bench_func.disc_list(8, 7 if not small_test else 3)]
        ],
        'par_methods': [
            [None], [None], [3], [3], [None], [None], [3], [2], [None]
        ],
        'par_discs': [
            [None],
            [None],
            [bench_func.disc_list(1, 5 if not small_test else 3)],
            [bench_func.disc_list(1, 5 if not small_test else 3)],
            [None],
            [None],
            [bench_func.disc_list(1, 5 if not small_test else 3)],
            [bench_func.disc_list(1, 4 if not small_test else 3)],
            [None]
        ],
        'disc_refinement_functions' : [
            [bench_func.create_object_from_config] for _ in range(n_settings)
            ]
    }

    return benchmark_config


def paper_geometry_test_benchmark(setting_name,
                             small_test=False, ref_filepath=None,
                             user_solution_times_unit_state=[1.25],
                            **model_kwargs
                             ):

    reference = analytical.load_reference(setting_name, ref_filepath)

    n_settings = 1
    
    benchmark_config = {
        'cadet_config_jsons': [
            setting_Col1D_pureTransport_1comp_benchmark1.get_model(
                spatial_method_bulk=0,
                write_solution_bulk=True,
                **model_kwargs
                ),
        ],
        'cadet_config_names': [
            setting_name,
        ],
        'include_sens': [False] * n_settings,
        'ref_files': [
            [reference]
        ],
        'unit_IDs': [
            '001',
        ],
        'which': [
            'bulk' # requires exactly one of the kwargs time_point or normed_coord
            # 'outlet'
        ] * n_settings,
        'idas_abstol': [
            [1e-15],
        ],
        'ax_methods': [
            [0]
        ],
        'ax_discs': [
            [bench_func.disc_list(1, 13 if not small_test else 3)],
        ],
        'par_methods': [
            [None],
        ],
        'par_discs': [
            [None],
        ],
        'disc_refinement_functions' : [
            [partial(create_convergence_object, setting_name=setting_name, model_kwargs={**model_kwargs, 'spatial_method_bulk': 0}, user_solution_times_unit_state=user_solution_times_unit_state)] for _ in range(n_settings)
             ]
    }

    return benchmark_config


def paper_geometry_transport_benchmark(setting_name,
                             small_test=False, ref_filepath=None,
                             cadet_reference=None,
                            **model_kwargs
                             ):
    """Pure transport EOC benchmark of one column geometry.

    Compared against the analytical solution where one exists, otherwise against
    the stored CADET reference named by cadet_reference, and otherwise against the
    finest resolution of the sweep itself, see src/geometry_references.py.

    The methods default to FV (0) and DG of degrees one to four. Pass
    ax_methods to restrict them, e.g. to the DG degrees for a geometry that is
    only implemented for DG, and n_levels to shorten the refinement sweeps.
    """

    default_n_levels = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8}

    ax_methods = model_kwargs.pop('ax_methods', [0, 1, 2, 3, 4])
    n_levels = model_kwargs.pop(
        'n_levels', [default_n_levels[method] for method in ax_methods])
    if small_test:
        n_levels = [3] * len(ax_methods)
    ax_discs = [bench_func.disc_list(1, n) for n in n_levels]
    idas_abstol = [1e-15] * len(ax_methods)

    base_config = setting_Col1D_pureTransport_1comp_benchmark1.get_model(
        spatial_method_bulk=3, write_solution_bulk=True, **model_kwargs
    )

    # The nodal initial condition depends on the grid, so every refinement level
    # is rebuilt from scratch instead of only updating NELEM/NCOL.
    refinement_functions = [
        partial(create_convergence_object, setting_name=setting_name,
                model_kwargs={**model_kwargs, 'spatial_method_bulk': method})
        for method in ax_methods
    ]

    references = geometry_references.resolve(
        setting_name=setting_name, ax_methods=ax_methods, data_dir=ref_filepath,
        cadet_reference=cadet_reference,
    )

    n_settings = 1

    benchmark_config = {
        'cadet_config_jsons': [base_config],
        'cadet_config_names': [setting_name],
        'include_sens': [False] * n_settings,
        'ref_files': [references],
        'unit_IDs': ['001'],
        'which': [
            'bulk'  # requires exactly one of the kwargs time_point or normed_coord
        ] * n_settings,
        'idas_abstol': [idas_abstol],
        'ax_methods': [ax_methods],
        'ax_discs': [ax_discs],
        'par_methods': [[None] * len(ax_methods)],
        'par_discs': [[None] * len(ax_methods)],
        'disc_refinement_functions': [refinement_functions],
    }

    return benchmark_config


def paper_geometry_LRMPdynLin_benchmark(setting_name,
                             small_test=False, ref_filepath=None,
                             cadet_reference=None,
                            **kwargs
                             ):
    """Linear binding LRMP EOC benchmark of one column geometry.

    No analytical solution is available with particles, so this is compared against
    the stored CADET reference named by cadet_reference, and otherwise against the
    finest resolution of the sweep itself, see src/geometry_references.py.
    """

    ax_methods = [0, 1, 2, 3, 4]
    n_levels = [13, 13, 12, 11, 10] if not small_test else [3] * len(ax_methods)
    ax_discs = [bench_func.disc_list(1, n) for n in n_levels]
    idas_abstol = [1e-15] * len(ax_methods)

    base_config = setting_Col1D_lin_1comp_benchmark1.get_model(
        spatial_method_bulk=3, particle_type='HOMOGENEOUS_PARTICLE',
        column_geometry=kwargs['column_geometry'],
        write_solution_bulk=True, user_solution_times_unit_state=[12.0]
    )

    refinement_functions = [bench_func.create_object_from_config] * len(ax_methods)

    references = geometry_references.resolve(
        setting_name=setting_name, ax_methods=ax_methods, data_dir=ref_filepath,
        cadet_reference=cadet_reference,
    )

    n_settings = 1

    benchmark_config = {
        'cadet_config_jsons': [base_config],
        'cadet_config_names': [setting_name],
        'include_sens': [False] * n_settings,
        'ref_files': [references],
        'unit_IDs': ['001'],
        'which': [
            'bulk'  # requires exactly one of the kwargs time_point or normed_coord
        ] * n_settings,
        'idas_abstol': [idas_abstol],
        'ax_methods': [ax_methods],
        'ax_discs': [ax_discs],
        'par_methods': [[None] * len(ax_methods)],
        'par_discs': [[None] * len(ax_methods)],
        'disc_refinement_functions': [refinement_functions],
    }

    return benchmark_config


# Refinement steps of the performance benchmarks of Breuer et al. (2023),
# doi:10.1016/j.compchemeng.2023.108340, run here on the radial flow and the
# conical column geometry instead of the axial flow cylinder of the publication.
#
# Per physical case and spatial method, (first level, number of levels), every
# level doubling. For the SMA case the axial and the particle grid are refined
# together, starting at one particle element.
#
# The steps are those of the published figures: Fig. 5 starts at four axial and
# one particle element and shows FV up to 512 and 128, DG P3 up to 32 and 8 and
# DG P4 up to 16 and 4; Fig. 3, the same load-wash-elute setting as an LRMP,
# starts at four and shows FV up to 512 cells, DG P3 up to 32 and DG P4 up to 16
# elements; Table S5, the left panel of Fig. 7, lists FV from 32 to 65536 cells
# and both DG degrees from 8 to 1024 elements. Only the less disperse Langmuir
# setting is covered, which is what the settings file and the stored references
# hold.
#
# The LRMP carries no particle grid at all, only film diffusion, so its sweeps
# refine nothing but the bulk. That is what makes it the cleanest of the three
# for comparing bulk discretizations: no particle discretization error enters
# the measured error, and no choice of particle to bulk resolution has to be
# made and defended.
_GEOMETRY_BENCHMARK_STEPS_ = {
    'SMA': {
        0: (4, 8),
        3: (4, 4),
        4: (4, 3),
        5: (4, 3),
        },
    'LRMP_SMA': {
        0: (4, 8),
        3: (4, 4),
        4: (4, 3),
        5: (4, 3),
        },
    'langmuir': {
        0: (32, 12),
        3: (8, 8),
        4: (8, 8),
        5: (8, 8),
        },
    }

# Particle elements of the coarsest level of the SMA case.
_GEOMETRY_BENCHMARK_PAR_START_ = 1

# A small test runs this many refinement levels fewer per series, but never
# fewer than two, since a single level is not a convergence study.
_GEOMETRY_BENCHMARK_SMALL_TEST_FEWER_ = 2
_GEOMETRY_BENCHMARK_MIN_LEVELS_ = 2

# Reference solution per physical case and geometry, shared by every spatial
# method of that case so that the methods are comparable to one another. They
# live in the repository under data/CADET-Core_reference/chromatography. The SMA
# case takes the finest DG solution available, the Langmuir case the finest FV
# one: the DG schemes oscillate on the self-sharpening Langmuir fronts, which is
# the very effect that benchmark measures, so the monotone scheme is the safer
# reference there.
_GEOMETRY_BENCHMARK_REFERENCES_ = {
    ('SMA', 'radial'):
        'radial_GRM_reqSMA_4comp_benchmark1_cDG_P4Z256_DGexInt_parP4parZ64.h5',
    ('SMA', 'frustum'):
        'frustum_GRM_reqSMA_4comp_benchmark1_cDG_P4Z256_DGexInt_parP4parZ64.h5',
    ('langmuir', 'radial'):
        'radial_LRM_langmuir_2comp_benchmark1_FV_Z8192.h5',
    ('langmuir', 'frustum'):
        'frustum_LRM_langmuir_2comp_benchmark1_FV_Z8192.h5',
    # None is stored for the LRMP yet, so its reference is simulated at double
    # the last refinement step; drop a file of this name in to reuse one.
    ('LRMP_SMA', 'radial'):
        'radial_LRMP_reqSMA_4comp_benchmark1_reference.h5',
    ('LRMP_SMA', 'frustum'):
        'frustum_LRMP_reqSMA_4comp_benchmark1_reference.h5',
    }

_GEOMETRY_BENCHMARK_GEOMETRIES_ = {
    'radial': 'RADIAL_FLOW_CYLINDER_SHELL',
    'frustum': 'AXIAL_FLOW_FRUSTUM',
    }

# The SMA study runs either as a general rate model, which resolves the particle
# in space and is Fig. 5 of the publication, or as a lumped rate model with
# pores, which carries no particle grid and so compares nothing but the bulk
# discretizations. They are separate internal cases because they differ in their
# reference, in the names of their files and in whether their sweeps refine a
# particle grid at all.
_SMA_PARTICLE_RESOLUTION_CASES_ = {0: 'LRMP_SMA', 1: 'SMA'}


def geometry_benchmark_case(case, sma_particle_resolution=1):
    """Internal case key of a physical case and its particle treatment.

    sma_particle_resolution is 0 for an LRMP and 1 for a GRM, and applies to the
    SMA case only; the Langmuir case has no particles to resolve.
    """

    if case != 'SMA':
        return case

    if sma_particle_resolution not in _SMA_PARTICLE_RESOLUTION_CASES_:
        raise ValueError(
            'sma_particle_resolution must be 0 for an LRMP or 1 for a GRM, got '
            + str(sma_particle_resolution) + '.'
            )

    return _SMA_PARTICLE_RESOLUTION_CASES_[sma_particle_resolution]


def geometry_benchmark_geometries(selection=None):
    """Resolve a column geometry selection into the geometries to run.

    Accepts the short key, the CADET geometry name, 'both' or None for all of
    them, or a sequence of any of those. Selecting one at a time is what makes
    the expensive sweeps tractable: each geometry brings its own reference, so
    running one does not compute the other.
    """

    if selection is None or (isinstance(selection, str)
                             and selection.lower() in ('both', 'all')):
        return tuple(_GEOMETRY_BENCHMARK_GEOMETRIES_)

    if isinstance(selection, str):
        selection = [selection]

    resolved = []
    for name in selection:
        if name in _GEOMETRY_BENCHMARK_GEOMETRIES_:
            resolved.append(name)
            continue
        matches = [key for key, geometry in _GEOMETRY_BENCHMARK_GEOMETRIES_.items()
                   if str(name).upper() == geometry]
        if not matches:
            raise ValueError(
                'Unknown column geometry ' + repr(name) + '; expected one of '
                + str(sorted(_GEOMETRY_BENCHMARK_GEOMETRIES_)) + ', '
                + str(sorted(_GEOMETRY_BENCHMARK_GEOMETRIES_.values()))
                + " or 'both'."
                )
        resolved.append(matches[0])

    return tuple(resolved)

_GEOMETRY_BENCHMARK_UNIT_ = {'SMA': '000', 'LRMP_SMA': '000', 'langmuir': '001'}

# Spatial method a computed reference of a case uses: a finite volume scheme for
# the Langmuir case, DG of the highest degree of the study for the SMA case.
_GEOMETRY_BENCHMARK_REFERENCE_METHOD_ = {'SMA': 4, 'LRMP_SMA': 4, 'langmuir': 0}

_GEOMETRY_BENCHMARK_SETTINGS_ = {
    'SMA': setting_Col1D_SMA_4comp_LWE_benchmark1,
    'LRMP_SMA': setting_Col1D_SMA_4comp_LWE_benchmark1,
    'langmuir': setting_Col1D_langLRM_2comp_benchmark1,
    }

# Name of the setting, which prefixes its simulation files and its convergence
# json, and has to name the transport model and the number of components.
_GEOMETRY_BENCHMARK_CONFIG_NAME_ = {
    'SMA': 'GRM_reqSMA_4comp_benchmark1',
    'LRMP_SMA': 'LRMP_reqSMA_4comp_benchmark1',
    'langmuir': 'LRM_langmuir_2comp_benchmark1',
    }

# Only the GRM resolves the particle in space; the LRMP has film diffusion but
# no particle grid, and the LRM has no particles at all.
_GEOMETRY_BENCHMARK_PARTICLE_GRID_ = {
    'SMA': True, 'LRMP_SMA': False, 'langmuir': False,
    }


def _geometry_benchmark_model_kwargs(case, spatial_method):
    """Particle arguments of the setting of one case."""

    if case == 'SMA':
        return dict(spatial_method_particle=spatial_method,
                    particle_type='GENERAL_RATE_PARTICLE')

    if case == 'LRMP_SMA':
        return dict(particle_type='HOMOGENEOUS_PARTICLE')

    return {}


def _geometry_benchmark_axial_points(spatial_method, n_ax):
    """Axial discrete points of one discretization, the measure of resolution."""

    return n_ax if spatial_method == 0 else n_ax * (spatial_method + 1)


def _geometry_benchmark_reference(case, geometry, ref_filepath):
    """Stored reference solution of one case and geometry, and its resolution.

    Returns (solution, axial points), or (None, None) when no reference is
    stored, upon which run_convergence_analysis falls back on the finest level
    of every method as the reference of that same method.
    """

    if ref_filepath is None:
        return None, None

    path = os.path.join(ref_filepath, 'CADET-Core_reference', 'chromatography',
                        _GEOMETRY_BENCHMARK_REFERENCES_[(case, geometry)])

    if not os.path.isfile(path):
        return None, None

    unit = 'unit_' + _GEOMETRY_BENCHMARK_UNIT_[case]
    disc = convergence.get_simulation(path).root.input.model[unit].discretization

    method = convergence.get_case_insensitive(disc, 'SPATIAL_METHOD')
    method = method.decode() if isinstance(method, bytes) else str(method)

    if method.upper() == 'FV':
        points = int(convergence.get_case_insensitive(disc, 'NCOL'))
    else:
        points = int(convergence.get_case_insensitive(disc, 'NELEM')) * (
            int(convergence.get_case_insensitive(disc, 'POLYDEG')) + 1)

    return convergence.get_solution(path, unit=unit), points


def _geometry_benchmark_levels(case, spatial_method, small_test):
    """First level and number of refinement levels of one series.

    The steps of the publication, or two levels fewer for a small test, never
    fewer than two, since a single level is not a convergence study.
    """

    n_ax_start, n_levels = _GEOMETRY_BENCHMARK_STEPS_[case][spatial_method]

    if small_test:
        n_levels = max(n_levels - _GEOMETRY_BENCHMARK_SMALL_TEST_FEWER_,
                       _GEOMETRY_BENCHMARK_MIN_LEVELS_)

    return n_ax_start, n_levels


def _geometry_benchmark_sweep_resolution(case, small_test):
    """Axial points of the finest level of the whole sweep of one case.

    Across all spatial methods, since they share one reference and it has to
    out-resolve every one of them.
    """

    return max(
        _geometry_benchmark_axial_points(
            spatial_method,
            bench_func.disc_list(*_geometry_benchmark_levels(
                case, spatial_method, small_test))[-1])
        for spatial_method in _GEOMETRY_BENCHMARK_STEPS_[case]
        )


def _geometry_benchmark_computed_reference(case, geometry, small_test,
                                           cadet_path, output_path):
    """Simulate the reference of one case and geometry and return its solution.

    Used when no stored reference resolves the sweep. The discretization is the
    one of the case, a finite volume scheme for the Langmuir case and DG of the
    highest degree the study uses for the SMA case, refined one step beyond the
    sweep, i.e. at double the last refinement step. Where that is still not
    finer than the sweep as a whole, which happens when the reference method is
    not the one reaching furthest, it is doubled again until it is.

    The file is written to the output folder and reused, so that the three
    spatial methods of a case simulate it once rather than three times.
    """

    if cadet_path is None or output_path is None:
        raise ValueError(
            'The ' + case + ' benchmark on the ' + geometry + ' geometry needs '
            'a reference that is not stored in the reference data folder, so it '
            'has to be simulated, for which geometry_performance_benchmark needs '
            'cadet_path and output_path.'
            )

    method = _GEOMETRY_BENCHMARK_REFERENCE_METHOD_[case]
    n_ax_start, n_levels = _geometry_benchmark_levels(case, method, small_test)

    # One refinement beyond the sweep, i.e. double the last refinement step.
    target = 2 * _geometry_benchmark_sweep_resolution(case, small_test)
    n_levels += 1
    while _geometry_benchmark_axial_points(
            method, bench_func.disc_list(n_ax_start, n_levels)[-1]) < target:
        n_levels += 1

    n_ax = bench_func.disc_list(n_ax_start, n_levels)[-1]
    n_par = (bench_func.disc_list(_GEOMETRY_BENCHMARK_PAR_START_, n_levels)[-1]
             if _GEOMETRY_BENCHMARK_PARTICLE_GRID_[case] else None)

    settings = _GEOMETRY_BENCHMARK_SETTINGS_[case]
    model_kwargs = _geometry_benchmark_model_kwargs(case, method)

    unit = 'unit_' + _GEOMETRY_BENCHMARK_UNIT_[case]

    build_kwargs = dict(
        setting_name=geometry + '_' + case + '_benchmark_reference',
        unit_id=_GEOMETRY_BENCHMARK_UNIT_[case],
        ax_method=method, ax_cells=n_ax,
        par_method=None if n_par is None else method, par_cells=n_par,
        output_path=str(output_path), idas_abstol=1e-8, include_sens=False,
        )

    # The name first, without writing anything: a reference that has already
    # been simulated is reused, so that the spatial methods of a case simulate
    # it once rather than once each.
    filename = bench_func.create_object_from_config(
        config_data=copy.deepcopy(settings.get_model(
            column_geometry=_GEOMETRY_BENCHMARK_GEOMETRIES_[geometry],
            spatial_method_bulk=method,
            **model_kwargs
            )),
        only_return_name=True, **build_kwargs
        )

    if os.path.isfile(filename):
        try:
            solution = convergence.get_solution(filename, unit=unit)
            print('Reference of the ' + case + ' benchmark on the ' + geometry
                  + ' geometry: reusing ' + os.path.basename(filename))
            return solution
        except ValueError:
            # An input file without results, from an interrupted run.
            pass

    print('Reference of the ' + case + ' benchmark on the ' + geometry
          + ' geometry: no stored reference resolves this sweep, simulating '
          + ('FV' if method == 0 else 'DG P' + str(method)) + ' at N_e = '
          + str(n_ax) + ('' if n_par is None else ' and N_e^p = ' + str(n_par))
          + ', double the last refinement step.')

    simulation = bench_func.create_object_from_config(
        config_data=copy.deepcopy(settings.get_model(
            column_geometry=_GEOMETRY_BENCHMARK_GEOMETRIES_[geometry],
            spatial_method_bulk=method,
            **model_kwargs
            )),
        **build_kwargs
        )

    bench_func.run_simulation_in_verification([simulation], str(cadet_path))

    return convergence.get_solution(simulation.filename, unit=unit)


def geometry_performance_benchmark(case, spatial_method, small_test=False,
                                   ref_filepath=None,
                                   geometries=None, sma_particle_resolution=1,
                                   cadet_path=None, output_path=None):
    """Performance benchmark of one physical case on the column geometries.

    case is 'SMA', the four-component GRM with kinetic steric mass action
    binding of Fig. 5 of Breuer et al. (2023), 'LRMP_SMA', the same setting as
    an LRMP, which is Fig. 3 and which resolves no particle at all and so
    compares nothing but the bulk discretizations, or 'langmuir', the
    two-component LRM with rapid-equilibrium Langmuir binding of Figs. 7 and 8
    in its less disperse variant. spatial_method is 0 for the WENO finite volume scheme and
    the polynomial degree for DG, used for the axial and the particle
    discretization alike.

    The refinement steps are those of the publication, see
    _GEOMETRY_BENCHMARK_STEPS_, or two levels fewer per series for a small test.

    geometries selects the column geometries, one of them or both, see
    geometry_benchmark_geometries. sma_particle_resolution selects the particle
    treatment of the SMA case, 0 for an LRMP and 1 for a GRM, see
    geometry_benchmark_case.

    Every spatial method of a case is measured against the same reference, which
    is what makes them comparable to one another: the stored one where it
    resolves the sweep, and otherwise one simulated at double the last
    refinement step, which needs cadet_path and output_path.
    """

    case = geometry_benchmark_case(case, sma_particle_resolution)

    if case not in _GEOMETRY_BENCHMARK_STEPS_:
        raise ValueError(
            'case must be one of ' + str(sorted(_GEOMETRY_BENCHMARK_STEPS_))
            + ', got ' + str(case) + '.'
            )

    settings = _GEOMETRY_BENCHMARK_SETTINGS_[case]

    model_kwargs = _geometry_benchmark_model_kwargs(case, spatial_method)

    n_ax_start, n_levels = _geometry_benchmark_levels(
        case, spatial_method, small_test)
    needed = 2 * _geometry_benchmark_sweep_resolution(case, small_test)

    cadet_configs = []
    cadet_config_names = []
    ref_files = []
    ax_discs = []
    par_discs = []

    for geometry in geometry_benchmark_geometries(geometries):

        reference, reference_points = _geometry_benchmark_reference(
            case, geometry, ref_filepath)

        if reference is None or reference_points < needed:
            if reference is not None:
                print('Reference of the ' + case + ' benchmark on the '
                      + geometry + ' geometry: the stored one resolves '
                      + str(reference_points) + ' axial points, which does not '
                      'out-resolve this sweep.')
            reference = _geometry_benchmark_computed_reference(
                case, geometry, small_test, cadet_path, output_path)

        cadet_configs.append(settings.get_model(
            column_geometry=_GEOMETRY_BENCHMARK_GEOMETRIES_[geometry],
            spatial_method_bulk=spatial_method,
            **model_kwargs
            ))
        cadet_config_names.append(
            geometry + '_' + _GEOMETRY_BENCHMARK_CONFIG_NAME_[case])
        ref_files.append([reference])
        ax_discs.append([bench_func.disc_list(n_ax_start, n_levels)])
        par_discs.append(
            [bench_func.disc_list(_GEOMETRY_BENCHMARK_PAR_START_, n_levels)]
            if _GEOMETRY_BENCHMARK_PARTICLE_GRID_[case] else [None]
            )

    n_settings = len(cadet_configs)

    return {
        'cadet_config_jsons': cadet_configs,
        'cadet_config_names': cadet_config_names,
        'include_sens': [False] * n_settings,
        'ref_files': ref_files,
        'unit_IDs': [_GEOMETRY_BENCHMARK_UNIT_[case]] * n_settings,
        'which': ['outlet'] * n_settings,
        'idas_abstol': [[1e-8]] * n_settings,
        'ax_methods': [[spatial_method]] * n_settings,
        'ax_discs': ax_discs,
        'par_methods': [
            [spatial_method if _GEOMETRY_BENCHMARK_PARTICLE_GRID_[case] else None]
            ] * n_settings,
        'par_discs': par_discs,
        'disc_refinement_functions': [
            [bench_func.create_object_from_config] for _ in range(n_settings)
            ]
        }


# %% Further sensitivity benchmark configuration used in CADET-Core tests (FV and DG)


def sensitivity_benchmark1(spatial_method, small_test):
    
    if spatial_method not in ["DG", "FV"]:
        raise ValueError(
            f"spatial method must be FV or DG.")

    if spatial_method == "FV":
        spatial_method_polyDeg = 0
    elif spatial_method == "DG":
        spatial_method_polyDeg = 3

    benchmark_config = {
        'cadet_config_jsons': [
            setting_Col1D_lin_1comp_benchmark1.get_LRM_sensbenchmark1(
                spatial_method_bulk=spatial_method_polyDeg
                ),
            setting_Col1D_lin_1comp_benchmark1.get_LRMP_sensbenchmark1(
                spatial_method_bulk=spatial_method_polyDeg
                ),
            setting_Col1D_lin_1comp_benchmark1.get_GRM_sensbenchmark1(
               spatial_method_bulk=spatial_method_polyDeg,
               spatial_method_particle=spatial_method_polyDeg
               ),
            setting_Col1D_lin_1comp_benchmark1.get_GRM_sensbenchmark2(
               spatial_method_bulk=spatial_method_polyDeg,
               spatial_method_particle=spatial_method_polyDeg
               ),
            setting_Col1D_SMA_4comp_LWE_benchmark1.get_LRM_sensbenchmark1(
                spatial_method_bulk=spatial_method_polyDeg
                ),
            setting_Col1D_SMA_4comp_LWE_benchmark1.get_LRMP_sensbenchmark1(
                spatial_method_bulk=spatial_method_polyDeg
                ),
            setting_Col1D_SMA_4comp_LWE_benchmark1.get_GRM_sensbenchmark1(
               spatial_method_bulk=spatial_method_polyDeg,
               spatial_method_particle=spatial_method_polyDeg
               )
        ],
        'cadet_config_names': [
            'LRM_dynLin_1comp_sensbenchmark1',
            'LRMP_dynLin_1comp_sensbenchmark1',
            'GRM_dynLin_1comp_sensbenchmark1',
            'GRM_dynLin_1comp_sensbenchmark2',
            'LRM_reqSMA_4comp_sensbenchmark1',
            'LRMP_reqSMA_4comp_sensbenchmark1',
            'GRM_reqSMA_4comp_sensbenchmark1'
        ],
        'include_sens': [True] * 7,
        'ref_files': [
            [None], [None], [None], [None], [None], [None], [None]
        ],
        'unit_IDs': [
            '001', '001', '001', '001', '000', '000', '000'
        ],
        'which': [
            'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet'
        ],
        'idas_abstol': [
            [1e-10], [1e-10], [1e-10], [1e-10], [1e-10], [1e-10], [1e-8]
        ],
        'ax_methods': [[3]] * 7 if spatial_method == "DG" else [[0]] * 7,
        'ax_discs': [
            [bench_func.disc_list(2 if spatial_method == "DG" else 8, 4 if not small_test else 3)],
            [bench_func.disc_list(2 if spatial_method == "DG" else 8, 4 if not small_test else 3)],
            [bench_func.disc_list(4 if spatial_method == "DG" else 8, 4 if not small_test else 3)],
            [bench_func.disc_list(4 if spatial_method == "DG" else 8, 4 if not small_test else 3)],
            [bench_func.disc_list(4 if spatial_method == "DG" else 8, 4 if not small_test else 3)],
            [bench_func.disc_list(4 if spatial_method == "DG" else 8, 4 if not small_test else 3)],
            [bench_func.disc_list(4 if spatial_method == "DG" else 8, 3 if not small_test else 3)]
        ],
        'par_methods':
            [[None], [None], [3], [3], [None], [None], [3]] if spatial_method == "DG" else [[None], [None], [0], [0], [None], [None], [0]],
        'par_discs': [
            [None],
            [None],
            [bench_func.disc_list(1 if spatial_method == "DG" else 2, 4 if not small_test else 3)],
            [bench_func.disc_list(1 if spatial_method == "DG" else 2, 4 if not small_test else 3)],
            [None],
            [None],
            [bench_func.disc_list(1 if spatial_method == "DG" else 2, 3 if not small_test else 3)]
        ]
    }

    return benchmark_config


def sensitivity_benchmark2(spatial_method, small_test):
    
    if spatial_method not in ["DG", "FV"]:
        raise ValueError(
            f"spatial method must be FV or DG.")

    if spatial_method == "FV":
        spatial_method_polyDeg = 0
    elif spatial_method == "DG":
        spatial_method_polyDeg = 3

    benchmark_config = {
        'cadet_config_jsons': [
            setting_COL1D_GRMparType2_dynLin_2comp_benchmark1.get_sensbenchmark1(
                spatial_method_bulk=spatial_method_polyDeg, 
                spatial_method_particle=spatial_method_polyDeg)
        ],
        'cadet_config_names': [
            'GRMparType2_dynLin_2comp_sensbenchmark1'
        ],
        'include_sens': [True] * 1,
        'ref_files': [
            [None] * 1
        ],
        'unit_IDs': [
            '001'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-8]
        ],
        'ax_methods': [[3]] * 1 if spatial_method == "DG" else [[0]] * 1,
        'ax_discs': [
            [bench_func.disc_list(2 if spatial_method == "DG" else 4, 4 if not small_test else 3)]
        ],
        'par_methods':
            [[3]] if spatial_method == "DG" else [[0]],
        'par_discs': [
            [bench_func.disc_list(1 if spatial_method == "DG" else 2, 4 if not small_test else 3)]
        ]
    }

    return benchmark_config


def radial_flow_benchmark_fv(small_test=False, sensitivities=False, ref_filepath=None):

    if ref_filepath is not None:
        ref_LRM = convergence.get_solution(ref_filepath+'/CADET-Core_reference/chromatography/radLRM_dynLin_1comp_benchmark1_DG_P3Z256.h5')
        ref_LRMP = convergence.get_solution(ref_filepath+'/CADET-Core_reference/chromatography/radLRMP_dynLin_1comp_benchmark1_DG_P3Z128.h5')
        ref_GRM = convergence.get_solution(ref_filepath+'/CADET-Core_reference/chromatography/radGRM_dynLin_1comp_benchmark1_cDG_P3Z128_DGexInt_parP3parZ16.h5')
    else:
        ref_LRM = None
        ref_LRMP = None
        ref_GRM = None

    benchmark_config = {
        'cadet_config_jsons': [
            setting_radCol1D_LRM_lin_1comp_benchmark1.get_sensbenchmark1(
                spatial_method_bulk=0
                ),
            setting_radCol1D_lin_1comp_benchmark1.get_LRMP_sensbenchmark1(
                spatial_method_bulk=0
                ),
            setting_radCol1D_lin_1comp_benchmark1.get_GRM_sensbenchmark1(
                spatial_method_bulk=0, spatial_method_par=0
                )
        ] if sensitivities else [
            setting_radCol1D_LRM_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=0
                ),
            setting_radCol1D_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=0, particle_type="HOMOGENEOUS_PARTICLE"
                ),
            setting_radCol1D_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=0, spatial_method_par=0,
                particle_type="GENERAL_RATE_PARTICLE"
                )
        ],
        'cadet_config_names': [
            'radLRM_dynLin_1comp_sensbenchmark1',
            'radLRMP_dynLin_1comp_sensbenchmark1',
            'radGRM_dynLin_1comp_sensbenchmark1'
        ] if sensitivities else [
            'radLRM_dynLin_1comp_benchmark1',
            'radLRMP_dynLin_1comp_benchmark1',
            'radGRM_dynLin_1comp_benchmark1'
        ],
        'include_sens': [True] * 3 if sensitivities else [False] * 3,
        'ref_files': [
            [ref_LRM], [ref_LRMP], [ref_GRM]
        ],
        'unit_IDs': [
            '001', '001', '001'
        ],
        'which': [
            'outlet', 'outlet', 'outlet'
        ],
        'idas_abstol': [
            [1e-10], [1e-10], [1e-10]
        ],
        'ax_methods': [
            [0], [0], [0]
        ],
        'ax_discs': [
            [bench_func.disc_list(8, 11 if not small_test else 3)],
            [bench_func.disc_list(8, 7 if not small_test else 3)],
            [bench_func.disc_list(8, 5 if not small_test else 3)]
        ],
        'par_methods': [
            [None], [None], [0]
        ],
        'par_discs': [
            [None],
            [None],
            [bench_func.disc_list(1, 5 if not small_test else 3)]
        ],
        'disc_refinement_functions' : [
            [bench_func.create_object_from_config] for _ in range(3)
            ]
    }

    return benchmark_config


def radial_flow_benchmark_dg(small_test=False, sensitivities=False, ref_filepath=None):

    if ref_filepath is not None:
        ref_LRM = convergence.get_solution(ref_filepath+'/CADET-Core_reference/chromatography/radLRM_dynLin_1comp_benchmark1_DG_P3Z256.h5')
        ref_LRMP = convergence.get_solution(ref_filepath+'/CADET-Core_reference/chromatography/radLRMP_dynLin_1comp_benchmark1_DG_P3Z128.h5')
        ref_GRM = convergence.get_solution(ref_filepath+'/CADET-Core_reference/chromatography/radGRM_dynLin_1comp_benchmark1_cDG_P3Z128_DGexInt_parP3parZ16.h5')
    else:
        ref_LRM = None
        ref_LRMP = None
        ref_GRM = None

    benchmark_config = {
        'cadet_config_jsons': [
            setting_radCol1D_LRM_lin_1comp_benchmark1.get_sensbenchmark1(
                spatial_method_bulk=0
                ),
            setting_radCol1D_lin_1comp_benchmark1.get_LRMP_sensbenchmark1(
                spatial_method_bulk=0
                ),
            setting_radCol1D_lin_1comp_benchmark1.get_GRM_sensbenchmark1(
                spatial_method_bulk=0, spatial_method_par=0
                )
        ] if sensitivities else [
            setting_radCol1D_LRM_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=0
                ),
            setting_radCol1D_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=0, particle_type="HOMOGENEOUS_PARTICLE"
                ),
            setting_radCol1D_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=0, spatial_method_par=0,
                particle_type="GENERAL_RATE_PARTICLE"
                )
        ],
        'cadet_config_names': [
            'radLRM_dynLin_1comp_sensbenchmark1',
            'radLRMP_dynLin_1comp_sensbenchmark1',
            'radGRM_dynLin_1comp_sensbenchmark1'
        ] if sensitivities else [
            'radLRM_dynLin_1comp_benchmark1',
            'radLRMP_dynLin_1comp_benchmark1',
            'radGRM_dynLin_1comp_benchmark1'
        ],
        'include_sens': [True] * 3 if sensitivities else [False] * 3,
        'ref_files': [
            [ref_LRM], [ref_LRMP], [ref_GRM]
        ],
        'unit_IDs': [
            '001', '001', '001'
        ],
        'which': [
            'outlet', 'outlet', 'outlet'
        ],
        'idas_abstol': [
            [1e-10], [1e-10], [1e-10]
        ],
        'ax_methods': [
            [3], [3], [3]
        ],
        'ax_discs': [
            [bench_func.disc_list(2, 7 if not small_test else 3)],
            [bench_func.disc_list(2, 5 if not small_test else 3)],
            [bench_func.disc_list(8, 4 if not small_test else 3)]
        ],
        'par_methods': [
            [None], [None], [3]
        ],
        'par_discs': [
            [None],
            [None],
            [bench_func.disc_list(1, 4 if not small_test else 3)]
        ],
        'disc_refinement_functions' : [
            [bench_func.create_object_from_config] for _ in range(3)
            ]
    }

    return benchmark_config


def check_input_config(disc_method, test_size, test_sizes):
    if not disc_method >= 0:
        raise Exception('disc_method must be 0 for FV or N_d > 0 for DG')
    if isinstance(test_size, str):
        if test_size not in ['large', 'mid', 'small']:
            raise ValueError(
                f"test_size must be integer or in ['large', 'mid', 'small'].")
        if str(disc_method) in test_sizes.keys():
            test_size = test_sizes[str(disc_method)].get(test_size, 'mid')
        else:
            raise ValueError(
                f"No defined test size {test_size} for disc_method {disc_method}.")
    elif not isinstance(test_size, int):
        raise Exception(
            'test_size must be integer or string (mid/large/small)')
    return test_size


def expand_dict(compact_dict):
    expanded_dict = {}
    for keys, value in compact_dict.items():
        for key in keys.split('|'):
            expanded_dict[key] = value
    return expanded_dict


def LRM_dynLin_1comp_benchmark1(
        disc_method, test_size='mid', include_sens=True, ref_file=None):

    adj = 0 if ref_file == None else 1
    test_sizes = {
        '0': {'large': 15 - adj, 'mid': 8, 'small': 3},
        '3|4|5': {'large': 9 - adj, 'mid': 5, 'small': 3}
    }
    test_sizes = expand_dict(test_sizes)

    test_size = check_input_config(disc_method, test_size, test_sizes)

    benchmark_config = {
        'cadet_config_jsons': [
            'LRM_dynLin_1comp_sensbenchmark1_FV_Z256.json'
        ],
        'include_sens': [include_sens],
        'ref_files': [
            [ref_file]
        ],
        'unit_IDs': [
            '001'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-8 if include_sens else 1e-10]
        ],
        'ax_methods': [
            [disc_method]
        ],
        'ax_discs': [
            [bench_func.disc_list(8 if disc_method == 0 else 1, test_size)]
        ],
        'par_methods': [
            [None]
        ],
        'par_discs': [
            [None]
        ]
    }
    return benchmark_config


def LRMP_dynLin_1comp_benchmark1(
        disc_method, test_size='mid', include_sens=True, ref_file=None):

    adj = 0 if ref_file == None else 1
    test_sizes = {
        '0': {'large': 12 - adj, 'mid': 6, 'small': 3},
        '3|4|5': {'large': 5 - adj, 'mid': 5, 'small': 3}
    }
    test_sizes = expand_dict(test_sizes)

    test_size = check_input_config(disc_method, test_size, test_sizes)

    benchmark_config = {
        'cadet_config_jsons': [
            'LRMP_dynLin_1comp_sensbenchmark1_FV_Z32.json'
        ],
        'include_sens': [include_sens],
        'ref_files': [
            [ref_file]
        ],
        'unit_IDs': [
            '001'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-8 if include_sens else 1e-10]
        ],
        'ax_methods': [
            [disc_method]
        ],
        'ax_discs': [
            [bench_func.disc_list(8, test_size)]
        ],
        'par_methods': [
            [disc_method]
        ],
        'par_discs': [
            [bench_func.disc_list(1 if disc_method == 0 else 1, test_size)]
        ]
    }
    return benchmark_config


def GRM_dynLin_1comp_benchmark1(
        disc_method, test_size='mid', include_sens=True, ref_file=None):

    adj = 0 if ref_file == None else 1
    test_sizes = {
        '0': {'large': 15 - adj, 'mid': 6, 'small': 3},
        '3|4|5': {'large': 9 - adj, 'mid': 5, 'small': 3}
    }
    test_sizes = expand_dict(test_sizes)

    test_size = check_input_config(disc_method, test_size, test_sizes)

    benchmark_config = {
        'cadet_config_jsons': [
            'GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json'
        ],
        'include_sens': [include_sens],
        'ref_files': [
            [ref_file]
        ],
        'unit_IDs': [
            '001'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-8 if include_sens else 1e-10]
        ],
        'ax_methods': [
            [disc_method]
        ],
        'ax_discs': [
            [bench_func.disc_list(8 if disc_method == 0 else 1, test_size)]
        ],
        'par_methods': [
            [disc_method]
        ],
        'par_discs': [
            [bench_func.disc_list(1, test_size)]
        ]
    }
    return benchmark_config


def LRM_reqSMA_4comp_benchmark1(
        disc_method, test_size='mid', include_sens=True, ref_file=None):

    adj = 0 if ref_file == None else 1
    test_sizes = {
        '0': {'large': 12 - adj, 'mid': 5, 'small': 3},
        '3|4|5': {'large': 6 - adj, 'mid': 4, 'small': 3}
    }
    test_sizes = expand_dict(test_sizes)

    test_size = check_input_config(disc_method, test_size, test_sizes)

    benchmark_config = {
        'cadet_config_jsons': [
            'LRM_reqSMA_4comp_sensbenchmark1_FV_Z64.json'
        ],
        'include_sens': [include_sens],
        'ref_files': [
            [ref_file]
        ],
        'unit_IDs': [
            '000'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-8 if include_sens else 1e-10]
        ],
        'ax_methods': [
            [disc_method]
        ],
        'ax_discs': [
            [bench_func.disc_list(8 if disc_method == 0 else 4, test_size)]
        ],
        'par_methods': [
            [None]
        ],
        'par_discs': [
            [None]
        ]
    }
    return benchmark_config


def LRMP_reqSMA_4comp_benchmark1(
        disc_method, test_size='mid', include_sens=True, ref_file=None):

    adj = 0 if ref_file == None else 1
    test_sizes = {
        '0': {'large': 11 - adj, 'mid': 5, 'small': 3},
        '3|4|5': {'large': 6 - adj, 'mid': 4, 'small': 3}
    }
    test_sizes = expand_dict(test_sizes)

    test_size = check_input_config(disc_method, test_size, test_sizes)

    benchmark_config = {
        'cadet_config_jsons': [
            'LRMP_reqSMA_4comp_sensbenchmark1_FV_Z32.json'
        ],
        'include_sens': [include_sens],
        'ref_files': [
            [ref_file]
        ],
        'unit_IDs': [
            '000'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-8 if include_sens else 1e-10]
        ],
        'ax_methods': [
            [disc_method]
        ],
        'ax_discs': [
            [bench_func.disc_list(8 if disc_method == 0 else 4, test_size)]
        ],
        'par_methods': [
            [None]
        ],
        'par_discs': [
            [None]
        ]
    }
    return benchmark_config


def GRM_reqSMA_4comp_benchmark1(
        disc_method, test_size='mid', include_sens=True, ref_file=None):

    adj = 0 if ref_file == None else 1
    test_sizes = {
        '0': {'large': 11 - adj, 'mid': 5, 'small': 3},
        '3|4|5': {'large': 5 - adj, 'mid': 4, 'small': 3}
    }
    test_sizes = expand_dict(test_sizes)

    test_size = check_input_config(disc_method, test_size, test_sizes)

    benchmark_config = {
        'cadet_config_jsons': [
            'GRM_reqSMA_4comp_sensbenchmark1_FV_Z16parZ2.json'
        ],
        'include_sens': [include_sens],
        'ref_files': [
            [ref_file]
        ],
        'unit_IDs': [
            '000'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-7 if include_sens else 1e-8]
        ],
        'ax_methods': [
            [disc_method]
        ],
        'ax_discs': [
            [bench_func.disc_list(4, test_size)]
        ],
        'par_methods': [
            [disc_method]
        ],
        'par_discs': [
            [bench_func.disc_list(1, test_size)]
        ]
    }
    return benchmark_config


def linear_chromatography_benchmark(
        disc_method, test_size='mid', include_sens=True, ref_files=None):

    benchmark_config = LRM_dynLin_1comp_benchmark1(
        disc_method, test_size, include_sens, ref_files)

    merge_benchmark(benchmark_config,
                    LRMP_dynLin_1comp_benchmark1(
                        disc_method, test_size, include_sens, ref_files)
                    )
    merge_benchmark(benchmark_config,
                    GRM_dynLin_1comp_benchmark1(
                        disc_method, test_size, include_sens, ref_files)
                    )

    return benchmark_config


def LWE_chromatography_benchmark(
        disc_method, test_size='mid', include_sens=True, ref_files=None):

    benchmark_config = LRM_reqSMA_4comp_benchmark1(
        disc_method, test_size, include_sens, ref_files)

    merge_benchmark(benchmark_config,
                    LRMP_reqSMA_4comp_benchmark1(
                        disc_method, test_size, include_sens, ref_files)
                    )
    merge_benchmark(benchmark_config,
                    GRM_reqSMA_4comp_benchmark1(
                        disc_method, test_size, include_sens, ref_files)
                    )

    return benchmark_config


def chromatography_benchmark_without_GRMLWE(
        disc_method, test_size='mid', include_sens=True, ref_files=None):

    benchmark_config = LRM_reqSMA_4comp_benchmark1(
        disc_method, test_size, include_sens, ref_files)

    merge_benchmark(benchmark_config,
                    LRMP_reqSMA_4comp_benchmark1(
                        disc_method, test_size, include_sens, ref_files)
                    )
    merge_benchmark(benchmark_config,
                    linear_chromatography_benchmark(
                        disc_method, test_size, include_sens, ref_files)
                    )

    return benchmark_config


def full_chromatography_benchmark(
        disc_method, test_size='mid', include_sens=True, ref_files=None):

    benchmark_config = linear_chromatography_benchmark(
        disc_method, test_size, include_sens, ref_files)

    merge_benchmark(benchmark_config,
                    LWE_chromatography_benchmark(
                        disc_method, test_size, include_sens, ref_files)
                    )

    return benchmark_config


def GRM2D_FV_benchmark(small_test=False, **kwargs):

    nDisc = 4 if small_test else 6
    nRadialZones=kwargs.get('nRadialZones', 3)
    
    benchmark_config = {
        'cadet_config_jsons': [
            settings_2Dchromatography.GRM2D_linBnd_benchmark1(
                radNElem=nRadialZones,
                rad_inlet_profile=None,
                USE_MODIFIED_NEWTON=0, axMethod=0, **kwargs)
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
        'unit_IDs': [ # note that we consider radial zone 0
            str(nRadialZones + 1 + 0).zfill(3) if kwargs.get('analytical_reference', 0) else '000'
        ],
        'which': [
            'outlet' if kwargs.get('analytical_reference', 0) else 'radial_outlet' # outlet_port_000
        ],
        'idas_abstol': [
            [1e-10]
        ],
        'ax_methods': [
            [0]
        ],
        'ax_discs': [
            [bench_func.disc_list(4, nDisc)]
        ],
        'rad_methods': [
            [0]
        ],
        'rad_discs': [
            [bench_func.disc_list(nRadialZones, nDisc)]
        ],
        'par_methods': [
            [0]
        ],
        'par_discs': [ # same number of particle cells as radial cells
            [bench_func.disc_list(nRadialZones, nDisc)]
        ]
    }

    return benchmark_config


def smb1_systems_tests(n_jobs, database_path, output_path,
                       cadet_path, small_test=False, **kwargs):

    nDisc = 4 if small_test else 5

    benchmark_config = {
        'cadet_config_jsons': [
            settings_columnSystems.SMB_model1(
                nDisc, 4, 1)
        ],
        'include_sens': [
            False
        ],
        'ref_files': [
            [None]
        ],
        'unit_IDs': [
            '003'
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
        'par_methods': [
            [None]
        ],
        'par_discs': [
            [None]
        ]
    }

    return benchmark_config


def cyclic_systems_tests(n_jobs, output_path,
                         cadet_path, small_test=False, **kwargs):

    # The analytical reference for the cyclic case has only ~1e-8 accuracy
    # Note that there is no proven error bound for the cyclic case, only estimations
    nDisc = 5 if kwargs.get('analytical_reference', False) else 7

    benchmark_config = {
        'cadet_config_jsons': [
            settings_columnSystems.Cyclic_model1(
                nDisc, 4, 1, analytical_reference=kwargs.get('analytical_reference', False))
        ],
        'include_sens': [
            False
        ],
        'ref_files': [
            [None]
        ],
        'unit_IDs': [
            '002' if kwargs.get('analytical_reference', False) else '003'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-12]
        ],
        'ax_methods': [
            [2]
        ],
        'ax_discs': [
            [bench_func.disc_list(1, nDisc)]
        ],
        'par_methods': [
            [None]
        ],
        'par_discs': [
            [None]
        ]
    }

    return benchmark_config


def acyclic_systems_tests(n_jobs, output_path,
                          cadet_path, small_test=False, **kwargs):

    nDisc = 5 if small_test else 5

    benchmark_config = {
        'cadet_config_jsons': [
            settings_columnSystems.Acyclic_model1(
                nDisc, 4, 1, analytical_reference=kwargs.get('analytical_reference', False))
        ],
        'include_sens': [
            False
        ],
        'ref_files': [
            [None]
        ],
        'unit_IDs': [
            '006'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-12]
        ],
        'ax_methods': [
            [3]
        ],
        'ax_discs': [
            [bench_func.disc_list(4, nDisc)]
        ],
        'par_methods': [
            [None]
        ],
        'par_discs': [
            [None]
        ]
    }

    return benchmark_config


def smb_systems_tests(n_jobs, output_path,
                       cadet_path, small_test=False, **kwargs):

    nDisc = 3 if small_test else 6

    benchmark_config = {
        'cadet_config_jsons': [
            settings_columnSystems.SMB_model1(nDisc, 3, 1)
        ],
        'include_sens': [
            False
        ],
        'ref_files': [
            [None]
        ],
        'unit_IDs': [
            '003'
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
            [bench_func.disc_list(3, nDisc)]
        ],
        'par_methods': [
            [None]
        ],
        'par_discs': [
            [None]
        ]
    }

    return benchmark_config


def merge_benchmark(benchmark_config1, benchmark_config2):

    for key in benchmark_config1.keys():
        benchmark_config1[key].extend(benchmark_config2[key])


def add_benchmark(cadet_config_jsons, include_sens, ref_files, unit_IDs, which,
                  ax_methods, ax_discs,
                  par_methods=None, par_discs=None,
                  rad_methods=None, rad_discs=None,
                  idas_abstol=None,
                  refinement_IDs=None,
                  cadet_config_names=None,
                  addition=None,
                  disc_refinement_functions = None):

    if addition is None:
        addition = {}

    cadet_config_jsons.extend(addition['cadet_config_jsons'])
    include_sens.extend(addition['include_sens'])
    ref_files.extend(addition['ref_files'])
    unit_IDs.extend(addition['unit_IDs'])
    which.extend(addition['which'])
    
    ax_methods.extend(addition['ax_methods'])
    ax_discs.extend(addition['ax_discs'])
    
    if idas_abstol is not None:
        idas_abstol.extend(addition['idas_abstol'])
    if disc_refinement_functions is not None:
        disc_refinement_functions.extend(addition['disc_refinement_functions'])
    if par_methods is not None:
        par_methods.extend(addition['par_methods'])
        par_discs.extend(addition['par_discs'])
    if rad_methods is not None:
        rad_methods.extend(addition['rad_methods'])
        rad_discs.extend(addition['rad_discs'])
    if refinement_IDs is not None:
        if 'refinement_ID' in addition.keys():
            refinement_IDs.extend(addition['refinement_ID'])
    if cadet_config_names is not None:
        cadet_config_names.extend(addition['cadet_config_names'])
        
