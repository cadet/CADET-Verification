"""

This scipt runs the verification studies for radial-flow and frustum
geometry discretizations (FV and DG)

"""

from pathlib import Path
import os

import src.utility.convergence as convergence
from src.utility.versionInfo import print_cadet_versions
from src import bench_configs
from src import bench_func

small_test = 0
n_jobs = -1
delete_h5_files = False

output_path = Path.cwd() / "output" / "test_cadet-core"

cadet_path = convergence.get_cadet_path()
cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

reference_data_path = str(Path(__file__).resolve().parent.parent / 'data' / 'CADET-Verification_reference')
reference_data_path = None

print_cadet_versions(cadet_path)

os.makedirs(output_path, exist_ok=True)

# Define settings and benchmarks
user_solution_times_unit_state = [5.0]

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

addition = bench_configs.cadet_configs = []
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

##################################
# Radial flow
##################################

addition = bench_configs.paper_geometry_transport_benchmark(
    setting_name='radialAdvDPFR_1comp_benchmark1',
    small_test=small_test, ref_filepath=reference_data_path,
    user_solution_times_unit_state=user_solution_times_unit_state,
    **{'advection' : True, 'dispersion' : False,
                    'column_geometry': 'RADIAL_FLOW_CYLINDER_SHELL'}
    )

bench_configs.add_benchmark(
    cadet_configs, include_sens, ref_files, unit_IDs, which,
    ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
    cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions=disc_refinement_functions,
    )

addition = bench_configs.paper_geometry_transport_benchmark(
    setting_name='radialDispDPFR_1comp_benchmark1',
    small_test=small_test, ref_filepath=reference_data_path,
    user_solution_times_unit_state=user_solution_times_unit_state,
    **{'advection' : False, 'dispersion' : True,
                    'column_geometry': 'RADIAL_FLOW_CYLINDER_SHELL'}
    )

bench_configs.add_benchmark(
    cadet_configs, include_sens, ref_files, unit_IDs, which,
    ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
    cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions=disc_refinement_functions,
    )

addition = bench_configs.paper_geometry_transport_benchmark(
    setting_name='radialDPFR_1comp_benchmark1',
    small_test=small_test, ref_filepath=reference_data_path,
    user_solution_times_unit_state=user_solution_times_unit_state,
    **{'advection' : True, 'dispersion' : True,
                    'column_geometry': 'RADIAL_FLOW_CYLINDER_SHELL'}
    )

bench_configs.add_benchmark(
    cadet_configs, include_sens, ref_files, unit_IDs, which,
    ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
    cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions=disc_refinement_functions,
    )

addition = bench_configs.paper_geometry_LRMPdynLin_benchmark(
    setting_name='radialLRMP_dynLin_1comp_benchmark1',
    small_test=small_test, ref_filepath=reference_data_path,
    user_solution_times_unit_state=user_solution_times_unit_state,
    **{'column_geometry': 'RADIAL_FLOW_CYLINDER_SHELL'}
    )

bench_configs.add_benchmark(
    cadet_configs, include_sens, ref_files, unit_IDs, which,
    ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
    cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions=disc_refinement_functions,
    )

##################################
# Frustum
##################################

addition = bench_configs.paper_geometry_transport_benchmark(
    setting_name='frustumAdvDPFR_1comp_benchmark1',
    small_test=small_test, ref_filepath=reference_data_path,
    user_solution_times_unit_state=user_solution_times_unit_state,
    **{'advection' : True, 'dispersion' : False,
                    'column_geometry': 'AXIAL_FLOW_FRUSTUM'}
    )

bench_configs.add_benchmark(
    cadet_configs, include_sens, ref_files, unit_IDs, which,
    ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
    cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions=disc_refinement_functions,
    )

addition = bench_configs.paper_geometry_transport_benchmark(
    setting_name='frustumDispDPFR_1comp_benchmark1',
    small_test=small_test, ref_filepath=reference_data_path,
    user_solution_times_unit_state=user_solution_times_unit_state,
    **{'advection' : False, 'dispersion' : True,
                    'column_geometry': 'AXIAL_FLOW_FRUSTUM'}
    )

bench_configs.add_benchmark(
    cadet_configs, include_sens, ref_files, unit_IDs, which,
    ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
    cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions=disc_refinement_functions,
    )

addition = bench_configs.paper_geometry_transport_benchmark(
    setting_name='frustumDPFR_1comp_benchmark1',
    small_test=small_test, ref_filepath=reference_data_path,
    user_solution_times_unit_state=user_solution_times_unit_state,
    **{'advection' : True, 'dispersion' : True,
                    'column_geometry': 'AXIAL_FLOW_FRUSTUM'}
    )

bench_configs.add_benchmark(
    cadet_configs, include_sens, ref_files, unit_IDs, which,
    ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
    cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions=disc_refinement_functions,
    )

addition = bench_configs.paper_geometry_LRMPdynLin_benchmark(
    setting_name='frustumLRMP_dynLin_1comp_benchmark1',
    small_test=small_test, ref_filepath=reference_data_path,
    user_solution_times_unit_state=user_solution_times_unit_state,
    **{'column_geometry': 'AXIAL_FLOW_FRUSTUM'}
    )

bench_configs.add_benchmark(
    cadet_configs, include_sens, ref_files, unit_IDs, which,
    ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
    cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions=disc_refinement_functions,
    )


# run convergence analysis

bench_func.run_convergence_analysis(
    output_path=output_path / "chromatography",
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
    disc_refinement_functions = disc_refinement_functions,
    # For which='bulk', exactly one of the following two must be given:
    
    # time_point: solution time index at which spatial error norms are
    # evaluated. Must hit a time at which the concentration front is still
    # inside the column; at the end of the simulation the
    # column is empty again.
    # With 601 solution times on [0, 30], index 50 corresponds to t = 2.5 s;
    # the bump center has then moved ~1/6 of the domain and is > 7 sigma away
    # from both boundaries. NOTE: index 0 compares the initial conditions
    # only (both are exact projections -> errors at machine precision).
    time_point=0,
    
    # normed_coord: normalized axial coordinate z/L in [0, 1] at which
    # temporal (outlet-like) error norms are evaluated; normed_coord=1.0
    # is equivalent to the outlet solution.
    # normed_coord=0.98,
)

if delete_h5_files:
    convergence.delete_h5_files(str(output_path) + "/chromatography")
