# -*- coding: utf-8 -*-
"""

This script defines chromatography tests.

"""

# %% Include packages
import os
from pathlib import Path

import src.bench_configs as bench_configs
import src.bench_func as bench_func


# %% Reference data paths
reference_data_path = str(
    Path(__file__).resolve().parent.parent / 'data' / 'CASEMA_reference'
)


# %% Run with CADET-RDM

def chromatography_tests(n_jobs, small_test, sensitivities,
                         output_path, cadet_path):

    os.makedirs(output_path, exist_ok=True)

    # Define settings and benchmarks

    cadet_configs = []
    cadet_config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    par_methods = []
    par_discs = []
    disc_refinement_functions = []

    addition = bench_configs.radial_flow_benchmark(small_test=small_test)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
        cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions = disc_refinement_functions)

    addition = bench_configs.fv_benchmark(
        small_test=small_test, sensitivities=sensitivities,
        ref_filepath=reference_data_path
        )

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol,
        cadet_config_names=cadet_config_names, addition=addition,
        disc_refinement_functions = disc_refinement_functions
        )
    
    addition = bench_configs.dg_benchmark(
        small_test=small_test, sensitivities=sensitivities,
        ref_filepath=reference_data_path
        )

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol,
        cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions = disc_refinement_functions)

    bench_func.run_convergence_analysis(
        output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=cadet_config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods,
        ax_discs=ax_discs,
        par_methods=par_methods,
        par_discs=par_discs,
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rerun_sims=True,
        disc_refinement_functions = disc_refinement_functions
    )