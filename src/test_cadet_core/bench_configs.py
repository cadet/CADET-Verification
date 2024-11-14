# -*- coding: utf-8 -*-
"""
Created May 2024

This script defines benchmark configurations, e.g. whether or not to include
sensitivities, the considered numerical methods, etc.
The individual model settings considered are specified in the CADET-Database
github project.

@author: jmbr
"""

import os
import json
import copy

import bench_func

# %% benchmark templates

_benchmark_settings_ = [
    'full_chromatography_benchmark',
    'chromatography_benchmark_without_GRMLWE',
    'linear_chromatography_benchmark',
    'LWE_chromatography_benchmark',
    'radial_flow_benchmark',
    # Individual settings
    'LRM_dynLin_1comp_benchmark1',
    'LRMP_dynLin_1comp_benchmark1',
    'GRM_dynLin_1comp_benchmark1',
    'LRM_reqSMA_4comp_benchmark1',
    'LRMP_reqSMA_4comp_benchmark1',
    'GRM_reqSMA_4comp_benchmark1',
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


# %% FV benchmark configuration used in CADET-Core tests


def fv_benchmark(small_test=False, sensitivities=False):

    benchmark_config = {
        'cadet_config_jsons': [
            'configuration_LRM_dynLin_1comp_sensbenchmark1_FV_Z256.json',
            'configuration_LRMP_dynLin_1comp_sensbenchmark1_FV_Z32.json',
            'configuration_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json',
            'configuration_LRM_reqSMA_4comp_sensbenchmark1_FV_Z64.json',
            'configuration_LRMP_reqSMA_4comp_sensbenchmark1_FV_Z32.json',
            'configuration_GRM_reqSMA_4comp_sensbenchmark1_FV_Z16parZ2.json'
        ],
        'include_sens': [True] * 6 if sensitivities else [False] * 6,
        'ref_files': [
            [None], [None], [None], [None], [None], [None]
        ],
        'unit_IDs': [
            '001', '001', '001', '000', '000', '000'
        ],
        'which': [
            'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet'
        ],
        'idas_abstol': [
            [1e-10], [1e-10], [1e-10], [1e-10], [1e-10], [1e-8]
        ],
        'ax_methods': [
            [0], [0], [0], [0], [0], [0]
        ],
        'ax_discs': [
            [bench_func.disc_list(8, 15 if not small_test else 3)],
            [bench_func.disc_list(8, 15 if not small_test else 3)],
            [bench_func.disc_list(8, 12 if not small_test else 3)],
            [bench_func.disc_list(8, 12 if not small_test else 3)],
            [bench_func.disc_list(8, 11 if not small_test else 3)],
            [bench_func.disc_list(8, 11 if not small_test else 3)]
        ],
        'par_methods': [
            [None], [None], [0], [None], [None], [0]
        ],
        'par_discs': [
            [None],
            [None],
            [bench_func.disc_list(1, 12 if not small_test else 3)],
            [None],
            [None],
            [bench_func.disc_list(1, 11 if not small_test else 3)]
        ]
    }

    return benchmark_config

# %% DG benchmark configuration used in CADET-Core tests


def dg_benchmark(small_test=False, sensitivities=False):

    benchmark_config = {
        'cadet_config_jsons': [
            'configuration_LRM_dynLin_1comp_sensbenchmark1_FV_Z256.json',
            'configuration_LRMP_dynLin_1comp_sensbenchmark1_FV_Z32.json',
            'configuration_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json',
            'configuration_LRM_reqSMA_4comp_sensbenchmark1_FV_Z64.json',
            'configuration_LRMP_reqSMA_4comp_sensbenchmark1_FV_Z32.json',
            'configuration_GRM_reqSMA_4comp_sensbenchmark1_FV_Z16parZ2.json'
        ],
        'include_sens': [True] * 6 if sensitivities else [False] * 6,
        'ref_files': [
            [None], [None], [None], [None], [None], [None]
        ],
        'unit_IDs': [
            '001', '001', '001', '000', '000', '000'
        ],
        'which': [
            'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet'
        ],
        'idas_abstol': [
            [1e-10], [1e-10], [1e-10], [1e-10], [1e-10], [1e-8]
        ],
        'ax_methods': [
            [3], [3], [3], [3], [3], [3],
        ],
        'ax_discs': [
            [bench_func.disc_list(1, 9 if not small_test else 3)],
            [bench_func.disc_list(1, 9 if not small_test else 3)],
            [bench_func.disc_list(8, 5 if not small_test else 3)],
            [bench_func.disc_list(4, 6 if not small_test else 3)],
            [bench_func.disc_list(4, 6 if not small_test else 3)],
            [bench_func.disc_list(4, 4 if not small_test else 3)]
        ],
        'par_methods': [
            [None], [None], [3], [None], [None], [3]
        ],
        'par_discs': [
            [None],
            [None],
            [bench_func.disc_list(1, 5 if not small_test else 3)],
            [None],
            [None],
            [bench_func.disc_list(1, 4 if not small_test else 3)]
        ]
    }

    return benchmark_config


# %% Radial flow (FV) benchmark configuration used in CADET-Core tests


def radial_flow_benchmark(small_test=False, sensitivities=False):

    benchmark_config = {
        'cadet_config_jsons': [
            'radial/configuration_radLRM_dynLin_1comp_sensbenchmark1_FV_Z256.json',
            'radial/configuration_radLRMP_dynLin_1comp_sensbenchmark1_FV_Z32.json',
            'radial/configuration_radGRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json'
        ],
        'include_sens': [True] * 3 if sensitivities else [False] * 3,
        'ref_files': [
            [None], [None], [None]
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
            [bench_func.disc_list(8, 6 if not small_test else 3)]
        ],
        'par_methods': [
            [None], [None], [0]
        ],
        'par_discs': [
            [None],
            [None],
            [bench_func.disc_list(1, 6 if not small_test else 3)]
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
            'configuration_LRM_dynLin_1comp_sensbenchmark1_FV_Z256.json'
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
            'configuration_LRMP_dynLin_1comp_sensbenchmark1_FV_Z32.json'
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
            'configuration_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.json'
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
            'configuration_LRM_reqSMA_4comp_sensbenchmark1_FV_Z64.json'
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
            'configuration_LRMP_reqSMA_4comp_sensbenchmark1_FV_Z32.json'
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
            'configuration_GRM_reqSMA_4comp_sensbenchmark1_FV_Z16parZ2.json'
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


def merge_benchmark(benchmark_config1, benchmark_config2):

    for key in benchmark_config1.keys():
        benchmark_config1[key].extend(benchmark_config2[key])


def add_benchmark(cadet_config_jsons, include_sens, ref_files, unit_IDs, which,
                  idas_abstol, ax_methods, ax_discs,
                  par_methods=None, par_discs=None,
                  rad_methods=None, rad_discs=None,
                  addition=None):

    if addition is None:
        addition = {}

    cadet_config_jsons.extend(addition['cadet_config_jsons'])
    include_sens.extend(addition['include_sens'])
    ref_files.extend(addition['ref_files'])
    unit_IDs.extend(addition['unit_IDs'])
    which.extend(addition['which'])
    idas_abstol.extend(addition['idas_abstol'])
    ax_methods.extend(addition['ax_methods'])
    ax_discs.extend(addition['ax_discs'])
    if par_methods is not None:
        par_methods.extend(addition['par_methods'])
        par_discs.extend(addition['par_discs'])
    if rad_methods is not None:
        rad_methods.extend(addition['rad_methods'])
        rad_discs.extend(addition['rad_discs'])
