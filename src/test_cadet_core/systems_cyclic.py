# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:12:37 2023

@author: jespfra
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

import bench_configs
import bench_func

from cadet import Cadet
Cadet.cadet_path = r'C:\Users\pbzit\source\Test\out\install\aRELEASE\bin\cadet-cli'
cadet_path = Cadet.cadet_path
#%% General model options

def Cyclic_model1(nelem,polydeg,exactInt):
    
    
    #Setting up the model
    Cyclic_model = Cadet()


    #Speciy number of unit operations: input, column and output, 3
    Cyclic_model.root.input.model.nunits = 4
    
    #Specify # of components (salt,proteins)
    n_comp  = 1
    
    #First unit operation: inlet
    ## Source 1
    Cyclic_model.root.input.model.unit_000.unit_type = 'INLET'
    Cyclic_model.root.input.model.unit_000.ncomp = n_comp
    Cyclic_model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    
    
    ## Sink 
    Cyclic_model.root.input.model.unit_003.ncomp = n_comp
    Cyclic_model.root.input.model.unit_003.unit_type = 'OUTLET'
    
    ## Unit LRMP2
    Cyclic_model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES'
    Cyclic_model.root.input.model.unit_001.ncomp = n_comp 

    ## Geometry
    Cyclic_model.root.input.model.unit_001.col_porosity = 0.37
    Cyclic_model.root.input.model.unit_001.par_porosity = 0.75
    Cyclic_model.root.input.model.unit_001.col_dispersion = 2e-7
    Cyclic_model.root.input.model.unit_001.col_length = 1.4e-2
    Cyclic_model.root.input.model.unit_001.cross_section_area = 1 #From Lubke2007, is not important
    Cyclic_model.root.input.model.unit_001.film_diffusion = 6.9e-6
    Cyclic_model.root.input.model.unit_001.par_radius = 45e-6
    LRMP_Q3 = 3.45*1e-2 / 60 * 0.37

    #Isotherm specification
    Cyclic_model.root.input.model.unit_001.adsorption_model = 'LINEAR'
    Cyclic_model.root.input.model.unit_001.adsorption.is_kinetic = True    # Kinetic binding
    Cyclic_model.root.input.model.unit_001.adsorption.LIN_KA = [3.55] # m^3 / (mol * s)   (mobile phase)
    Cyclic_model.root.input.model.unit_001.adsorption.LIN_KD = [0.1]      # 1 / s (desorption)
    #Initial conditions
    Cyclic_model.root.input.model.unit_001.init_c = [0]
    Cyclic_model.root.input.model.unit_001.init_q = [0] #salt starts at max capacity
    
    
    ### Grid cells in column and particle: the most important ones - ensure grid-independent solutions
    Cyclic_model.root.input.model.unit_001.discretization.SPATIAL_METHOD = "DG"
    Cyclic_model.root.input.model.unit_001.discretization.nelem = nelem 
    
    #Polynomial order 
    Cyclic_model.root.input.model.unit_001.discretization.polydeg = polydeg
    Cyclic_model.root.input.model.unit_001.discretization.exact_integration = exactInt

    ### Bound states - for zero the compound does not bind, >1 = multiple binding sites
    Cyclic_model.root.input.model.unit_001.discretization.nbound = np.ones(n_comp,dtype=int)
    
    
    Cyclic_model.root.input.model.unit_001.discretization.par_disc_type = 'EQUIDISTANT_PAR'    
    Cyclic_model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    Cyclic_model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    Cyclic_model.root.input.model.unit_001.discretization.gs_type = 1
    Cyclic_model.root.input.model.unit_001.discretization.max_krylov = 0
    Cyclic_model.root.input.model.unit_001.discretization.max_restarts = 10
    Cyclic_model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8

    Cyclic_model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    Cyclic_model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    Cyclic_model.root.input.model.unit_001.discretization.weno.weno_order = 3
    
    ### Copy column models
    Cyclic_model.root.input.model.unit_002 = copy.deepcopy(Cyclic_model.root.input.model.unit_001)
    
    # Unit LRMP2 
    Cyclic_model.root.input.model.unit_003.adsorption.is_kinetic = False    # Kinetic binding
    Cyclic_model.root.input.model.unit_003.adsorption.LIN_KA = [35.5] # m^3 / (mol * s)   (mobile phase)
    Cyclic_model.root.input.model.unit_003.adsorption.LIN_KD = [1]      # 1 / s (desorption)
    
    
   

    
    #To write out last output to check for steady state
    Cyclic_model.root.input['return'].WRITE_SOLUTION_LAST = True



    #% Input and connections

    
    #Sections
    Cyclic_model.root.input.solver.sections.nsec = 2
    Cyclic_model.root.input.solver.sections.section_times = [0, 100, 6000]  
    
    ## Feed and Eluent concentration
    Cyclic_model.root.input.model.unit_000.sec_000.const_coeff = [1] #Inlet flowrate concentration

    Cyclic_model.root.input.model.unit_000.sec_001.const_coeff = [0] #Inlet flowrate concentration

    
    
    #Connections
    Cyclic_model.root.input.model.connections.nswitches = 1
    
    Cyclic_model.root.input.model.connections.switch_000.section = 0
    Cyclic_model.root.input.model.connections.switch_000.connections =[
        0, 1, -1, -1, LRMP_Q3/2,#flowrates, Q, m3/s
        1, 2, -1, -1, LRMP_Q3,
        2, 1, -1, -1, LRMP_Q3/2,
        2, 3, -1, -1, LRMP_Q3/2,
    ]

    
    #solution times
    Cyclic_model.root.input.solver.user_solution_times = np.linspace(0, 6000, 6000+1)
    
    
    
    #Time 
    # Tolerances for the time integrator
    Cyclic_model.root.input.solver.time_integrator.abstol = 1e-12 #absolute tolerance
    Cyclic_model.root.input.solver.time_integrator.algtol = 1e-10
    Cyclic_model.root.input.solver.time_integrator.reltol = 1e-10 #Relative tolerance
    Cyclic_model.root.input.solver.time_integrator.init_step_size = 1e-10
    Cyclic_model.root.input.solver.time_integrator.max_steps = 1000000
    
    
    
    #Solver options in general (not only for column although the same)
    Cyclic_model.root.input.model.solver.gs_type = 1
    Cyclic_model.root.input.model.solver.max_krylov = 0
    Cyclic_model.root.input.model.solver.max_restarts = 10
    Cyclic_model.root.input.model.solver.schur_safety = 1e-8
    Cyclic_model.root.input.solver.consistent_init_mode = 5 #necessary specifically for this sim
    Cyclic_model.root.input.solver.time_integrator.USE_MODIFIED_NEWTON = 1
    
    # Number of cores for parallel simulation
    Cyclic_model.root.input.solver.nthreads = 1
    
    
    #Specify which results we want to return
    # Return data
    Cyclic_model.root.input['return'].split_components_data = 0
    Cyclic_model.root.input['return'].split_ports_data = 0
    Cyclic_model.root.input['return'].unit_000.write_solution_bulk = 0
    Cyclic_model.root.input['return'].unit_000.write_solution_inlet = 0
    Cyclic_model.root.input['return'].unit_000.write_solution_outlet = 0
    Cyclic_model.root.input['return'].unit_001.write_solution_bulk = 0
    Cyclic_model.root.input['return'].unit_001.write_solution_inlet = 0
    Cyclic_model.root.input['return'].unit_001.write_solution_outlet = 1
    
    
    # Copy settings to the other unit operations
    Cyclic_model.root.input['return'].unit_002 = Cyclic_model.root.input['return'].unit_001
    Cyclic_model.root.input['return'].unit_003 = Cyclic_model.root.input['return'].unit_001

    
    
    return Cyclic_model

database_path = None
small_test = True
rdm_debug_mode = False
rerun_sims = False
output_path = "/"
n_jobs = 1
setting = []

def cyclic_systems_tests(n_jobs, database_path, output_path,
                                 cadet_path, small_test=False, **kwargs):

    nDisc = 4 if small_test else 6
    
    benchmark_config = {
        'cadet_config_jsons': [
            Cyclic_model1(nDisc,4,1)
        ],
        'include_sens': [
            False
        ],
        'ref_files': [
            [None]
        ],
        'unit_IDs': [
            '006'
        ],
        'which': [
            'outlet'
        ],
        'idas_abstol': [
            [1e-10]
        ],
        'ax_methods': [
            [3]
        ],
        'ax_discs': [
            [bench_func.disc_list(4, nDisc)]
        ],
        'par_methods': [
            [None]
        ],
        'par_discs': [
            [None]
        ]
    }

    return benchmark_config


def chromatography_systems_tests(n_jobs, database_path, small_test,
                                 output_path, cadet_path):

    cadet_configs = []
    config_names = []
    include_sens = []
    ref_files = [] # [[ref1], [ref2]]
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    rad_methods = []
    rad_discs = []
    par_methods = []
    par_discs = []
    
    # %% create benchmark configurations
    
    addition = cyclic_systems_tests(n_jobs, database_path, output_path, # todo
                                            cadet_path, small_test=small_test)
    
    bench_configs.add_benchmark(
        cadet_configs, include_sens, ref_files, unit_IDs, which,
        idas_abstol,
        ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
        par_methods=par_methods, par_discs=par_discs,
        addition=addition)
    
    config_names.extend([setting['smb1']])
    
    # addition = smb2_systems_tests(n_jobs, database_path, output_path, # todo
    #                                         cadet_path, small_test=small_test)
    
    # bench_configs.add_benchmark(
    #     cadet_configs, include_sens, ref_files, unit_IDs, which,
    #     idas_abstol,
    #     ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
    #     par_methods=par_methods, par_discs=par_discs,
    #     addition=addition)
    
    # config_names.extend([setting['smb2']])
    
    # %% Run convergence analysis
    
    Cadet.cadet_path = cadet_path
    
    bench_func.run_convergence_analysis(
        database_path=database_path, output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        rad_methods=rad_methods, rad_discs=rad_discs,
        par_methods=par_methods, par_discs=par_discs,
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rad_inlet_profile=None,
        rerun_sims=rerun_sims
    )







