"""

This script defines the performance benchmarks of the column geometries, i.e. of
the radial flow cylinder shell and the frustum.

Two chromatography settings are run in both geometries with the WENO finite
volume scheme and DG of polynomial degrees three to five: the four component
steric mass action load-wash-elute setting of a lumped rate model with pores, see
src/benchmark_models/setting_Col1D_SMA_4comp_LWE_benchmark1.py, and the two
component Langmuir setting of a lumped rate model without pores, see
src/benchmark_models/setting_Col1D_langLRM_2comp_benchmark1.py. Both settings and
their refinement steps are the ones of Breuer et al. (2023),
doi:10.1016/j.compchemeng.2023.108340, here on the geometries of the column
geometry publication.

What these studies measure is not the order of convergence, which the EOC studies
of src/column_geometries.py report, but what each method costs to reach a given
approximation error. Every method of a setting is therefore measured against the
same stored reference solution, which is finer than every level of its sweep, see
src/bench_configs.py and data/CADET-Core_reference/chromatography: runtimes of
different methods can only be compared at errors measured against the same
solution. A setting whose reference is missing falls back to the self-convergence
of the convergence analysis, in which case its methods are no longer comparable
to one another.

The sweeps of the publication run for days, which is more than a verification run
can spend, so they are shortened here by their most expensive levels, and a small
test runs the same three levels as the axial chromatography benchmarks;
full_refinement restores the sweeps of the publication.

"""

import os
from pathlib import Path

import src.bench_configs as bench_configs
import src.bench_func as bench_func


# Settings of the study, named as src/bench_configs.py names their stored
# reference solutions, and the geometries they are run in. The axial cylinder is
# left out, since its settings are already covered by the axial chromatography
# benchmarks, and so is the general rate model setting of the publication, whose
# sweeps refine the bulk and the particle grid at once and cost accordingly.
_SETTINGS_ = ['sma_lrmp', 'langmuir']

_GEOMETRIES_ = ['RADIAL_FLOW_CYLINDER_SHELL', 'AXIAL_FLOW_FRUSTUM']


# Root of the reference data; the reference solutions live in the chromatography
# subdirectory of the CADET-Core references.
_reference_data_path_ = str(Path(__file__).resolve().parent.parent / 'data')


def geometry_performance_tests(n_jobs, small_test, output_path, cadet_path,
                               full_refinement=False):
    """Run the performance benchmarks of the column geometries.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs of the convergence analysis, -1 uses all cores.
    small_test : bool
        Run strongly shortened refinement sweeps.
    output_path : string or Path
        Output folder of the simulations and the convergence data.
    cadet_path : string
        CADET installation path.
    full_refinement : bool
        Carry the sweeps as far as the publication does. This takes days and
        belongs on a machine that can spend them.
    """

    output_path = str(output_path)

    os.makedirs(output_path, exist_ok=True)

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

    for setting in _SETTINGS_:

        for geometry in _GEOMETRIES_:

            reference = bench_configs.geometry_reference_simulation(
                setting, geometry, _reference_data_path_
                )

            if setting == 'langmuir':
                addition = bench_configs.langmuir_performance_benchmark(
                    small_test=small_test, ref_file=reference,
                    column_geometry=geometry,
                    full_refinement=full_refinement
                    )
            else:
                addition = bench_configs.SMA_performance_benchmark(
                    small_test=small_test, ref_file=reference,
                    column_geometry=geometry,
                    particle_type='HOMOGENEOUS_PARTICLE',
                    full_refinement=full_refinement
                    )

            bench_configs.add_benchmark(
                cadet_configs, include_sens, ref_files, unit_IDs, which,
                ax_methods, ax_discs, par_methods, par_discs,
                idas_abstol=idas_abstol,
                cadet_config_names=cadet_config_names, addition=addition,
                disc_refinement_functions=disc_refinement_functions,
                )

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
        disc_refinement_functions=disc_refinement_functions,
    )
