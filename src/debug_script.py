# debug_chromatography.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chromatography import chromatography_tests
from cadet import Cadet

#Cadet.cadet_path = r"C:\...\cadet-cli.exe"  # dein Pfad

chromatography_tests(
    n_jobs=1,
    small_test=True,
    sensitivities=False,
    output_path="output/debug",
    cadet_path= None
)




""" 
def fv_benchmark(small_test=False, sensitivities=False):

    benchmark_config = {
        'cadet_config_jsons': [
            setting_Col1D_linLRM_1comp_benchmark1.get_model(
                spatial_method_bulk=0
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
        'include_sens': [True] * 9 if sensitivities else [False] * 9,
        'ref_files': [
            [None], [None], [None], [None], [None], [None], [None], [None], ['reference_method10_ref300.h5']
        ],
        'unit_IDs': [
            '001', '001', '001', '001', '000', '000', '000', '001', '001'
        ],
        'which': [
            'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet', 'outlet'
        ],
        'idas_abstol': [
            [1e-10], [1e-10], [1e-10], [1e-10], [1e-10], [1e-10], [1e-8], [1e-6], [1e-10]
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
            [bench_func.create_object_from_config] for _ in range(9)
            ]
    }

    return benchmark_config """