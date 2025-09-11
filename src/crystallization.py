from joblib import Parallel, delayed
import os
from pathlib import Path

import src.crystallization_partI as partI
import src.crystallization_partII as partII


def crystallization_tests(
        n_jobs, database_path, small_test, output_path, cadet_path,
        run_primary_dynamics_tests = True, # part I
        run_secondary_dynamics_tests = True, # part II without (partly) redundant tests
        run_full_secondary_dynamics_tests = False, # full part II tests
        ):
    
    os.makedirs(output_path, exist_ok=True)

    reference_data_path = str(Path(__file__).resolve().parent.parent / 'data' / 'CADET-Core_reference' / 'crystallization')

    tasks = [ ]
    
    # add expensive tests to the list first
    
    if run_full_secondary_dynamics_tests: # not included in test pipeline per default, due to redundancy
        tasks.extend([
        delayed(partII.DPFR_constAggregation_EOC_test)(cadet_path, small_test, output_path, reference_data_path+'/ref_DPFR_Z192_aggregation_Z384.h5'),
        delayed(partII.DPFR_aggregation_fragmentation_EOC_test)(cadet_path, small_test, output_path, reference_data_path+'/ref_DPFR_Z192_aggFrag_Z384.h5')
        ])
        
    if run_secondary_dynamics_tests or run_full_secondary_dynamics_tests:
        tasks.extend([
        delayed(partII.PBM_aggregation_fragmentation_EOC_test)(cadet_path, small_test, output_path, reference_data_path+'/ref_PBM_Agg_Frag_Z1536.h5'),
        delayed(partII.DPFR_constFragmentation_EOC_test)(cadet_path, small_test, output_path, reference_data_path+'/ref_DPFR_Z192_fragmentation_Z384.h5'),
        delayed(partII.DPFR_NGGR_aggregation_EOC_test)(
            cadet_path, small_test=True, output_path=output_path,
            reference_solution_file=reference_data_path+'/ref_DPFR_Z192_NGGR_Z384.h5')
        ])
    
    if run_primary_dynamics_tests:
        tasks.extend([
        delayed(partI.DPFR_PBM_primarySecondaryNucleationGrowth_EOC_test)(small_test, output_path, cadet_path)
        ])
        
    # add less expensive tests to the list
    
    if run_primary_dynamics_tests:
        tasks.extend([
        delayed(partI.CSTR_PBM_growth_EOC_test)(small_test, output_path, cadet_path),
        delayed(partI.CSTR_PBM_growthSizeDep_EOC_test)(small_test, output_path, cadet_path),
        delayed(partI.CSTR_PBM_primaryNucleationAndGrowth_EOC_test)(small_test, output_path, cadet_path),
        delayed(partI.CSTR_PBM_primarySecondaryNucleationAndGrowth_EOC_test)(small_test, output_path, cadet_path),
        delayed(partI.CSTR_PBM_primaryNucleationGrowthGrowthRateDispersion_EOC_test)(small_test, output_path, cadet_path)
        ])
        
    if run_secondary_dynamics_tests or run_full_secondary_dynamics_tests:
        tasks.extend([
        delayed(partII.aggregation_EOC_test)(cadet_path, small_test, output_path),
        delayed(partII.fragmentation_EOC_test)(cadet_path, small_test, output_path)
        ]) 
        
    if run_full_secondary_dynamics_tests: # not included in test pipeline per default, due to redundancy
        tasks.extend([
        delayed(partII.aggregation_fragmentation_EOC_test)(cadet_path, small_test, output_path)
        ])

    Parallel(n_jobs=n_jobs)(tasks)