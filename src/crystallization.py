from joblib import Parallel, delayed
import os

import src.crystallization_partI as partI
import src.crystallization_partII as partII


def crystallization_tests(
        n_jobs, database_path, small_test, output_path, cadet_path,
        run_primary_dynamics_tests = True, # part I
        run_secondary_dynamics_tests = False, # part II without (partly) redundant tests
        run_full_secondary_dynamics_tests = False, # full part II tests
        ):
    
    os.makedirs(output_path, exist_ok=True)

    tasks = [ ]
    
    if run_primary_dynamics_tests:
        tasks.extend([
        delayed(partI.CSTR_PBM_growth_EOC_test)(small_test, output_path, cadet_path),
        delayed(partI.CSTR_PBM_growthSizeDep_EOC_test)(small_test, output_path, cadet_path),
        delayed(partI.CSTR_PBM_primaryNucleationAndGrowth_EOC_test)(small_test, output_path, cadet_path),
        delayed(partI.CSTR_PBM_primarySecondaryNucleationAndGrowth_EOC_test)(small_test, output_path, cadet_path),
        delayed(partI.CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion_EOC_test)(small_test, output_path, cadet_path),
        delayed(partI.DPFR_PBM_primarySecondaryNucleationGrowth_EOC_test)(small_test, output_path, cadet_path)
        ])
        
        
    if run_secondary_dynamics_tests or run_full_secondary_dynamics_tests:
        tasks.extend([
        delayed(partII.aggregation_EOC_test)(cadet_path, small_test, output_path),
        delayed(partII.fragmentation_EOC_test)(cadet_path, small_test, output_path),
        delayed(partII.PBM_aggregation_fragmentation_EOC_test)(cadet_path, small_test, output_path),
        delayed(partII.DPFR_constFragmentation_EOC_test)(cadet_path, small_test, output_path),
        delayed(partII.DPFR_NGGR_aggregation_EOC_test)(cadet_path, small_test, output_path)
        ])
        
        
    if run_full_secondary_dynamics_tests: # not included in test pipeline per default, due to redundancy
        tasks.extend([
        delayed(partII.aggregation_fragmentation_EOC_test)(cadet_path, small_test, output_path),
        delayed(partII.DPFR_constAggregation_EOC_test)(cadet_path, small_test, output_path),
        delayed(partII.DPFR_aggregation_fragmentation_EOC_test)(cadet_path, small_test, output_path)
        ])

    Parallel(n_jobs=n_jobs)(tasks)