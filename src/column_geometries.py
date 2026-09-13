"""

This script defines the EOC studies of the column geometries, i.e. of the radial
flow cylinder shell, the frustum, and the smoothly varying cross section.

Every physical setting is compared against exactly one reference, which all spatial
methods of that setting share: the analytical solution of the initial value problem
where one exists (advection only and dispersion only, see src/analytical.py), and
otherwise a stored CADET solution (combined advection and dispersion, and the
settings with particles). Both kinds live under data/, see src/geometry_references.py.

The studies are defined by geometry_tests, which src/transport_convDisp.py calls, so
that all transport EOC studies are run from one place. src/column_geometries.py
runs the geometry studies on their own.

"""

from pathlib import Path
import os

import src.utility.convergence as convergence
from src import analytical
from src import bench_configs
from src import bench_func


# Pure transport settings: setting name, geometry, advection, dispersion, spatial
# methods (None uses the default FV and DG degrees one to four), CADET reference.
#
# The CADET reference is the file name of a stored solution in the reference data
# directory, see src/geometry_references.py. It is only consulted for a setting that
# has no analytical solution, which here is the smoothly varying cross section, whose
# sine area profile no closed form covers. None means that the setting brings no
# reference of its own, upon which the EOC test falls back to using the finest
# resolution of the sweep as its reference.
#
# The smoothly varying cross section is prescribed at the DG nodes and therefore has
# no FV variant, so it is restricted to the DG degrees.
_TRANSPORT_SETTINGS_ = [
    ('radialAdvDPFR_1comp_benchmark1', 'RADIAL_FLOW_CYLINDER_SHELL', True, False, None,
     None),
    ('radialDispDPFR_1comp_benchmark1', 'RADIAL_FLOW_CYLINDER_SHELL', False, True, None,
     None),
    ('radialDPFR_1comp_benchmark1', 'RADIAL_FLOW_CYLINDER_SHELL', True, True, None,
     None),
    ('frustumAdvDPFR_1comp_benchmark1', 'AXIAL_FLOW_FRUSTUM', True, False, None,
     None),
    ('frustumDispDPFR_1comp_benchmark1', 'AXIAL_FLOW_FRUSTUM', False, True, None,
     None),
    ('frustumDPFR_1comp_benchmark1', 'AXIAL_FLOW_FRUSTUM', True, True, None,
     None),
    ('smoothlyVaryingAdvDPFR_1comp_benchmark1', 'SMOOTHLY_VARYING', True, False, [1, 2, 3, 4],
     'smoothlyVaryingAdvDPFR_1comp_benchmark1_DG_P4Z8192.h5'),
    ('smoothlyVaryingDispDPFR_1comp_benchmark1', 'SMOOTHLY_VARYING', False, True, [1, 2, 3, 4],
     'smoothlyVaryingDispDPFR_1comp_benchmark1_DG_P4Z8192.h5'),
    ('smoothlyVaryingDPFR_1comp_benchmark1', 'SMOOTHLY_VARYING', True, True, [1, 2, 3, 4],
     'smoothlyVaryingDPFR_1comp_benchmark1_DG_P4Z8192.h5'),
]

# Linear binding LRMP settings: setting name, geometry, CADET reference. No analytical
# solution exists with particles, so these have no analytical reference to fall back on.
_LRMP_SETTINGS_ = [
    ('radialLRMP_dynLin_1comp_benchmark1', 'RADIAL_FLOW_CYLINDER_SHELL', None),
    ('frustumLRMP_dynLin_1comp_benchmark1', 'AXIAL_FLOW_FRUSTUM', None),
]

# Solution time at which the bulk error norms are evaluated. The bulk state is written
# only at these times, so the time index used below refers to this list.
_USER_SOLUTION_TIMES_UNIT_STATE_ = [5.0]


def geometry_tests(n_jobs, small_test, output_path, cadet_path,
                   regenerate_references=False, delete_h5_files=False):
    """Run the EOC studies of the column geometries.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs of the convergence analysis, -1 uses all cores.
    small_test : bool
        Run strongly shortened refinement sweeps.
    output_path : string or Path
        Output folder; the simulations are written to its chromatography subfolder.
    cadet_path : string
        CADET installation path.
    regenerate_references : bool
        Rewrite the analytical references from the model settings before running.
        The references are fixed inputs of the study: running it compares against the
        stored ones and never recomputes them, so that a result cannot silently change
        together with the reference it is measured against. Rebuilding them is an act
        of maintenance rather than part of running the study, and its result belongs
        in a commit of its own. The CADET references are never computed here at all;
        each setting names the stored file it uses, see _TRANSPORT_SETTINGS_.
    delete_h5_files : bool
        Delete the simulation files after the analysis.
    """

    output_path = Path(output_path)

    # The convergence analysis writes into the chromatography subdirectory, which has
    # to exist before the first simulation is saved.
    os.makedirs(output_path / "transport", exist_ok=True)

    # Root of the reference data; the analytical and the CADET references live in
    # their respective subdirectories, see src/geometry_references.py.
    reference_data_path = str(Path(__file__).resolve().parent.parent / 'data')

    if regenerate_references:
        analytical.generate_geometry_references(
            output_dir=str(Path(reference_data_path) / 'CADET-Verification_reference'),
            user_solution_times_unit_state=_USER_SOLUTION_TIMES_UNIT_STATE_,
        )

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

    def add(addition):
        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol,
            cadet_config_names=cadet_config_names, addition=addition,
            disc_refinement_functions=disc_refinement_functions,
            )

    for (setting_name, geometry, advection, dispersion, methods,
         cadet_reference) in _TRANSPORT_SETTINGS_:

        model_kwargs = {
            'advection': advection,
            'dispersion': dispersion,
            'column_geometry': geometry,
            # further options of the setting are 'weno_order' and 'grid_type', the
            # latter being 'equidistant' or 'equivolume'
            }
        if methods is not None:
            model_kwargs['ax_methods'] = list(methods)

        add(bench_configs.paper_geometry_transport_benchmark(
            setting_name=setting_name,
            small_test=small_test, ref_filepath=reference_data_path,
            cadet_reference=cadet_reference,
            user_solution_times_unit_state=_USER_SOLUTION_TIMES_UNIT_STATE_,
            **model_kwargs
            ))

    for setting_name, geometry, cadet_reference in _LRMP_SETTINGS_:

        add(bench_configs.paper_geometry_LRMPdynLin_benchmark(
            setting_name=setting_name,
            small_test=small_test, ref_filepath=reference_data_path,
            cadet_reference=cadet_reference,
            user_solution_times_unit_state=_USER_SOLUTION_TIMES_UNIT_STATE_,
            **{'column_geometry': geometry}
            ))

    bench_func.run_convergence_analysis(
        output_path=output_path / "transport",
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
        rerun_sims=1,
        disc_refinement_functions=disc_refinement_functions,
        # For which='bulk', exactly one of the following two must be given:

        # time_point: solution time index at which spatial error norms are
        # evaluated. Must hit a time at which the concentration front is still
        # inside the column; at the end of the simulation the column is empty again.
        # The bulk state is written only at _USER_SOLUTION_TIMES_UNIT_STATE_, so
        # index 0 is the single time of that list.
        time_point=0,

        # normed_coord: normalized axial coordinate z/L in [0, 1] at which
        # temporal (outlet-like) error norms are evaluated; normed_coord=1.0
        # is equivalent to the outlet solution.
        # normed_coord=0.98,
    )

    if delete_h5_files:
        convergence.delete_h5_files(str(output_path) + "/transport")
