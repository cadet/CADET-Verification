import copy
import os
import sys
import numpy as np
from functools import partial
from pathlib import Path

from cadet import Cadet

# Running this file directly, e.g. via the run button of an IDE, puts src/ and
# not the repository root on sys.path, so the src.* imports below would fail.
# Only `python -m src.spatial_stability` gets this right by itself.
repository_root = Path(__file__).resolve().parents[1]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

import src.utility.convergence as convergence
import src.bench_configs as bench_configs
import src.bench_func as bench_func
from src.benchmark_models import setting_Col1D_lin_1comp_benchmark1
from src.benchmark_models import setting_Col1D_langLRM_2comp_benchmark1

spatial_discretization_DGsubFV = {
    
    'USE_ANALYTIC_JACOBIAN': True,
    'USE_MODIFIED_NEWTON' : False,

    'SPATIAL_METHOD' : 'DG',
    'POLYDEG': 3,
    'NELEM': 8,

    'stabilization' : {

        "STABILIZATION_METHOD": "SUBCELL_FV_LIMITING",
        "SUBCELL_FV_ORDER": 1,
        "SUBCELL_FV_MAX_BLENDING": 1.0,
        'BLENDING_THRESHOLD': 0.0,
        "SUBCELL_FV_LIMITER": "MINMOD",
        "SUBCELL_FV_BOUNDARY_TREATMENT": 0,

        'smoothness_indicator': {
            'RETURN_SMOOTHNESS_INDICATOR': True,
            'SMOOTHNESS_INDICATOR': "MODAL_ENERGY_LEGENDRE",
            'NODAL_COEFFICIENT_THRESHOLD': 0.0,
            'NODAL_COEFFICIENT_SHIFT': 0.0,
            'TIME_RELAXATION': 0.0,
            'NUM_MODES': 2,
            },
        }
    }

time_integrator = {
    'ABSTOL' : 1e-8, 'RELTOL' : 1e-8, 'ALGTOL' : 1e-8,
    'USE_MODIFIED_NEWTON' : False,
    'init_step_size' : 1e-6,
    # IDAS takes many steps right after the inlet pulse is switched off, which
    # the finer discretizations do not survive with a smaller step budget
    'max_steps' : 10000
    }

def n_refinements_to_ax_discs(spatial_discretization, n_refinements):
    """ Returns the sequence of axial cells/elements that refine_discretization
        produces, i.e. the base discretization doubled per refinement level.

    Parameters
    ----------
    spatial_discretization : Dictionary
        Base discretization, see refine_discretization
    n_refinements: int
        Number of refinement levels
    Returns
    -------
    List of Ints
        Number of cells (FV) or elements (DG) per refinement level
    """

    base = spatial_discretization['NCOL'] \
        if spatial_discretization['SPATIAL_METHOD'] == 'FV' \
        else spatial_discretization['NELEM']

    return [base * 2**disc_idx for disc_idx in range(n_refinements)]


def refine_discretization(config_data, disc_idx, setting_name,
                          spatial_discretization,
                          time_integrator=None,
                          unit_id='001',
                          output_path=None,
                          only_return_name=False,
                          **kwargs):
        """ Takes a CADET-configuration as dictionary, adjusts its discretization and returns the 
            Cadet-Object. Optionally saves the corresponding h5 config file.

        Parameters
        ----------
        config_data : Dictionary
            Dictionary with CADET configuration
        disc_idx: int
            current index of refinement starting at zero
        Returns
        -------
        Cadet-Object
            Cadet object.
        """

        # Adjust configuration to desired numerical refinement
        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator
            
        spatialMethod = 0 if spatial_discretization['SPATIAL_METHOD'] == 'FV' else spatial_discretization['POLYDEG']
        nCol = spatial_discretization['NCOL'] * 2** (disc_idx) if spatialMethod == 0 else spatial_discretization['NELEM'] * 2** (disc_idx)
        
        config_data['input']['model']['unit_' + unit_id]['discretization'].update(
            {k: v for k, v in spatial_discretization.items()}
            )
        
        if spatialMethod == 0:
            config_data['input']['model']['unit_' + unit_id]['discretization']['NCOL'] = nCol
            config_name = convergence.generate_1D_name(setting_name, 0, nCol)
        else:
            config_data['input']['model']['unit_' + unit_id]['discretization']['NELEM'] = nCol
            config_name = convergence.generate_1D_name(setting_name, spatial_discretization['POLYDEG'], nCol)

        model = Cadet()
        model.root.input = copy.deepcopy(config_data['input'])
        
        if output_path is not None:

            model.filename = str(output_path) + '/' + config_name

            if only_return_name:
                return model.filename

            model.save()

        return model


def stabilization_benchmark_dgsubfv(small_test=False, ref_filepath=None):

    if ref_filepath is not None:
        ref_DPFR = None # convergence.get_solution(ref_filepath+'/CASEMA_reference/LRM_dynLin_1comp_benchmark1.h5')
        ref_LRMlangmuir = convergence.get_solution(ref_filepath+'/CADET-Core_reference/chromatography/LRM_lang_2comp_benchmark1.h5')
    else:
        ref_DPFR = None
        ref_LRMlangmuir = None

    benchmark_config = {
        'cadet_config_jsons': [
            # Pure convection (no dispersion, no particles), i.e. the inlet step
            # pulse is transported through the column unchanged. With an
            # interstitial velocity of 5.75e-4 m/s and a bed length of 0.014 m,
            # the residence time is 24.3 s, so the pulse has fully left the
            # column at t = 34.3 s and the simulation can end at t = 50 s.
            setting_Col1D_lin_1comp_benchmark1.get_model(
                spatial_method_bulk=3, particle_type=None,
                col_dispersion=0.0, pulse_end=10.0, t_end=50.0
                ),
            # setting_Col1D_langLRM_2comp_benchmark1.get_model(
            #     spatial_method_bulk=3
            #     )
        ],
        'cadet_config_names': [
            'DPFR_stepPulse_1comp_benchmark1',
            # 'LRM_langmuir_2comp_benchmark1'
        ],
        'include_sens': [False],
        'ref_files': [
            [ref_DPFR]
        ],
        'unit_IDs': [
            '001'
        ],
        'which': ['outlet'],
        'idas_abstol': [
           [1e-12]
        ],
        'ax_methods': [
            [3]
        ],
        # These numbers only label the convergence tables, the actual
        # refinement is done by refine_discretization, so they are derived from
        # the same base discretization to keep both in sync.
        'ax_discs': [
            [n_refinements_to_ax_discs(spatial_discretization_DGsubFV, 6)],
        ],
        'par_methods': [
            [None]
        ],
        'par_discs': [
            [None]
        ],
        'disc_refinement_functions' : [
            [
                partial(refine_discretization,
                        setting_name="DPFR_stepPulse_1comp_subFV",
                        spatial_discretization=copy.deepcopy(spatial_discretization_DGsubFV),
                        time_integrator=time_integrator
                    ),
            ]
        ]
    }

    return benchmark_config

cadet_path = Path(r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE")
# Anchored at the repository and not at the working directory, which differs
# between running this file directly and running it as a module.
output_path = repository_root / "output" / "test_cadet-core" / "stabilization"
n_jobs = 1

# cadet-python creates Cadet() objects without an explicit install path in
# several places and falls back to autodetection via PATH, so make this build
# discoverable there.
os.environ['PATH'] = str(cadet_path / 'bin') + os.pathsep + os.environ['PATH']

output_path.mkdir(parents=True, exist_ok=True)

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

addition = stabilization_benchmark_dgsubfv()

bench_configs.add_benchmark(
    cadet_configs, include_sens, ref_files, unit_IDs, which,
    ax_methods, ax_discs, par_methods, par_discs, idas_abstol=idas_abstol, 
    cadet_config_names=cadet_config_names, addition=addition,
    disc_refinement_functions=disc_refinement_functions)

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
    # For which='bulk', exactly one of the following two must be given:
    #
    # time_point: solution time index at which spatial error norms are
    # evaluated. Must hit a time at which the concentration front is still
    # inside the column (here t = 125s); at the end of the simulation the
    # column is empty again.
    # time_point=500,
    #
    # normed_coord: normalized axial coordinate z/L in [0, 1] at which
    # temporal (outlet-like) error norms are evaluated; normed_coord=1.0
    # is equivalent to the outlet solution.
    # normed_coord=1.0,
)