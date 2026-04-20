## -*- coding: utf-8 -*-

import subprocess

from cadet import Cadet

from src import twoDimChromatography
from src.benchmark_models import settings_2Dchromatography
from src.benchmark_models import settings_columnSystems
from src.benchmark_models import setting_Col1D_linLRM_1comp_benchmark1
from src.benchmark_models import setting_Col1D_lin_1comp_benchmark1
from src.benchmark_models import setting_Col1D_XparTypeGR_lin_1comp_benchmark1


executable_path = r"/home/jmbr/software/casema/code/build/release/src/casema-cli"
file_path = r"/home/jmbr/software/casema/code/build/release/src"

#%% 1D linear models

GRMlinBnd = Cadet()
GRMlinBnd.root = setting_Col1D_lin_1comp_benchmark1.get_model(
    spatial_method_bulk=-1, spatial_method_particle=-1,
    particle_type='GENERAL_RATE_PARTICLE'
    )

GRMlinBnd.root['input'].solver.casema_options  = {
    "ERROR_THRESHOLD": 1e-20,
    "NTHREADS": 4
    }

GRMlinBnd.filename = file_path + r"/GRM_dynLin_1comp_benchmark1.h5"
GRMlinBnd.save()

subprocess.run([executable_path, GRMlinBnd.filename], check=True)

LRMPlinBnd = Cadet()
LRMPlinBnd.root = setting_Col1D_lin_1comp_benchmark1.get_model(
    spatial_method_bulk=-1,
    particle_type='HOMOGENEOUS_PARTICLE'
    )

LRMPlinBnd.root['input'].solver.casema_options  = {
    "ERROR_THRESHOLD": 1e-20,
    "NTHREADS": 4
    }

LRMPlinBnd.filename = file_path + r"/LRMP_dynLin_1comp_benchmark1.h5"
LRMPlinBnd.save()

subprocess.run([executable_path, LRMPlinBnd.filename], check=True)

LRMlinBnd = Cadet()
LRMlinBnd.root = setting_Col1D_linLRM_1comp_benchmark1.get_model(
    spatial_method_bulk=-1
    )

LRMlinBnd.root['input'].solver.casema_options  = {
    "ERROR_THRESHOLD": 1e-20,
    "NTHREADS": 4
    }

LRMlinBnd.filename = file_path + r"/LRM_dynLin_1comp_benchmark1.h5"
LRMlinBnd.save()

subprocess.run([executable_path, LRMlinBnd.filename], check=True)

for small_test in [True, False]:
    
    mulParTypeModel = Cadet()
    
    mulParTypeModel.root = setting_Col1D_XparTypeGR_lin_1comp_benchmark1.get_model(
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
     })

    mulParTypeModel.root['input'].solver.casema_options  = {
     "ERROR_THRESHOLD": 1e-20,
     "NTHREADS": 4
     }

    if small_test:
        mulParTypeModel.filename = file_path + 'GRM_' + str(mulParTypeModel.root['input'].model.unit_001.npartype) + 'parTypeLin_4comp_benchmark1.h5'
    
    mulParTypeModel.save()
    
    subprocess.run([executable_path, mulParTypeModel.filename], check=True)

#%% acyclic model

acyclicModel = settings_columnSystems.Acyclic_model1(1, 1, 1)

acyclicModel.root['input'].solver.casema_options  = {
    "ERROR_THRESHOLD": 1e-20,
    "NTHREADS": 4
    }

acyclicModel.filename = file_path + r"/new_acyclicSystem1_LRMP_linBnd_1comp.h5"
acyclicModel.save()


subprocess.run([executable_path, acyclicModel.filename], check=True)

#%% cyclic model

cyclicModel = settings_columnSystems.Cyclic_model1(1, 1, 1)

cyclicModel.root['input'].solver.casema_options  = {
    "ERROR_THRESHOLD": 1e-20,
    "ABSCISSA": 0.0159585,
    "MAX_LAPLACE_SUMMANDS": 1000000,
    "NTHREADS": 4
    }

cyclicModel.filename = file_path + r"/new_acyclicSystem1_LRMP_linBnd_1comp.h5"
cyclicModel.save()


subprocess.run([executable_path, cyclicModel.filename], check=True)


#%% 2D linear models

GRM2DlinBnd = Cadet()

settings = twoDimChromatography.get_settings(
    use_CASEMA_reference=False, reference_data_path=None,
    small_test=True
    )

# add 4 parType setting (last in list)
settings.append(
    twoDimChromatography.get_settings(
        use_CASEMA_reference=False, reference_data_path=None,
        small_test=False
        )[-1]
    )

for setting in settings:
    
    GRM2DlinBnd.root = settings_2Dchromatography.GRM2D_linBnd_benchmark1(
        nRadialZones=3,
        radNElem=-3, # has to be -nRadialZones
        axMethod=-1,
        parMethod=-1
        )

    GRM2DlinBnd.root['input'].solver.casema_options  = {
        "ERROR_THRESHOLD": 1e-20,
        "ABSCISSA": 0.0315798,
        "MAX_HANKEL_SUMMANDS": 100,
        "NTHREADS": 4
        }
    
    GRM2DlinBnd.root['input'].model.unit_000.discretization.nrad = 3 # nRadialZones

    GRM2DlinBnd.filename = file_path + '/' + setting['name'] + '.h5'

    GRM2DlinBnd.save()
    
    subprocess.run([executable_path, GRM2DlinBnd.filename], check=True)