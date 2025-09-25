# -*- coding: utf-8 -*-
"""

This script defines chromatography tests.

"""

# %% Include packages
import os
import sys
from pathlib import Path
import re
from joblib import Parallel, delayed
import numpy as np

from cadet import Cadet
from cadetrdm import ProjectRepo

import src.bench_configs as bench_configs
import src.bench_func as bench_func


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

    addition = bench_configs.radial_flow_benchmark(small_test=small_test)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        cadet_config_names=cadet_config_names, addition=addition)

    addition = bench_configs.fv_benchmark(small_test=small_test)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        cadet_config_names=cadet_config_names, addition=addition)
    
    
    addition = bench_configs.dg_benchmark(
        small_test=small_test, sensitivities=False)

    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol, ax_methods, ax_discs, par_methods, par_discs,
        cadet_config_names=cadet_config_names, addition=addition)

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
        rerun_sims=True
    )