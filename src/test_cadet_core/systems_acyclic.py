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

def Acyclic_model1(nelem,polydeg,exactInt):
    
    
    #Setting up the model
    Acyclic_model = Cadet()


    #Speciy number of unit operations: input, column and output, 3
    Acyclic_model.root.input.model.nunits = 7
    
    #Specify # of components (salt,proteins)
    n_comp  = 1
    
    #First unit operation: inlet
    ## Source 1
    Acyclic_model.root.input.model.unit_000.unit_type = 'INLET'
    Acyclic_model.root.input.model.unit_000.ncomp = n_comp
    Acyclic_model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    
    
    ## Source 2
    Acyclic_model.root.input.model.unit_001.unit_type = 'INLET'
    Acyclic_model.root.input.model.unit_001.ncomp = n_comp
    Acyclic_model.root.input.model.unit_001.inlet_type = 'PIECEWISE_CUBIC_POLY'
    
    ## Sink 
    Acyclic_model.root.input.model.unit_006.ncomp = n_comp
    Acyclic_model.root.input.model.unit_006.unit_type = 'OUTLET'
    
    ## Unit LRMP3 
    Acyclic_model.root.input.model.unit_002.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES'
    Acyclic_model.root.input.model.unit_002.ncomp = n_comp 

    ## Geometry
    Acyclic_model.root.input.model.unit_002.col_porosity = 0.37
    Acyclic_model.root.input.model.unit_002.par_porosity = 0.75
    Acyclic_model.root.input.model.unit_002.col_dispersion = 2e-7
    Acyclic_model.root.input.model.unit_002.col_length = 1.4e-2
    Acyclic_model.root.input.model.unit_002.cross_section_area = 1 #From Lubke2007, is not important
    Acyclic_model.root.input.model.unit_002.film_diffusion = 6.9e-6
    Acyclic_model.root.input.model.unit_002.par_radius = 45e-6
    LRMP_Q3 = 3.45*1e-2 / 60 * 0.37

    #Isotherm specification
    Acyclic_model.root.input.model.unit_002.adsorption_model = 'LINEAR'
    Acyclic_model.root.input.model.unit_002.adsorption.is_kinetic = True    # Kinetic binding
    Acyclic_model.root.input.model.unit_002.adsorption.LIN_KA = [3.55] # m^3 / (mol * s)   (mobile phase)
    Acyclic_model.root.input.model.unit_002.adsorption.LIN_KD = [0.1]      # 1 / s (desorption)
    #Initial conditions
    Acyclic_model.root.input.model.unit_002.init_c = [0]
    Acyclic_model.root.input.model.unit_002.init_q = [0] #salt starts at max capacity
    
    
    ### Grid cells in column and particle: the most important ones - ensure grid-independent solutions
    Acyclic_model.root.input.model.unit_002.discretization.SPATIAL_METHOD = "DG"
    Acyclic_model.root.input.model.unit_002.discretization.nelem = nelem 
    
    #Polynomial order 
    Acyclic_model.root.input.model.unit_002.discretization.polydeg = polydeg
    Acyclic_model.root.input.model.unit_002.discretization.exact_integration = exactInt

    ### Bound states - for zero the compound does not bind, >1 = multiple binding sites
    Acyclic_model.root.input.model.unit_002.discretization.nbound = np.ones(n_comp,dtype=int)
    
    
    Acyclic_model.root.input.model.unit_002.discretization.par_disc_type = 'EQUIDISTANT_PAR'    
    Acyclic_model.root.input.model.unit_002.discretization.use_analytic_jacobian = 1
    Acyclic_model.root.input.model.unit_002.discretization.reconstruction = 'WENO'
    Acyclic_model.root.input.model.unit_002.discretization.gs_type = 1
    Acyclic_model.root.input.model.unit_002.discretization.max_krylov = 0
    Acyclic_model.root.input.model.unit_002.discretization.max_restarts = 10
    Acyclic_model.root.input.model.unit_002.discretization.schur_safety = 1.0e-8

    Acyclic_model.root.input.model.unit_002.discretization.weno.boundary_model = 0
    Acyclic_model.root.input.model.unit_002.discretization.weno.weno_eps = 1e-10
    Acyclic_model.root.input.model.unit_002.discretization.weno.weno_order = 3
    
    ### Copy column models
    Acyclic_model.root.input.model.unit_003 = copy.deepcopy(Acyclic_model.root.input.model.unit_002)
    Acyclic_model.root.input.model.unit_004 = copy.deepcopy(Acyclic_model.root.input.model.unit_002)
    Acyclic_model.root.input.model.unit_005 = copy.deepcopy(Acyclic_model.root.input.model.unit_002)
    
    # Unit LRMP4 
    Acyclic_model.root.input.model.unit_003.col_length = 4.2e-2 
    Acyclic_model.root.input.model.unit_003.adsorption.is_kinetic = False    # Kinetic binding
    Acyclic_model.root.input.model.unit_003.adsorption.LIN_KA = [35.5] # m^3 / (mol * s)   (mobile phase)
    Acyclic_model.root.input.model.unit_003.adsorption.LIN_KD = [1]      # 1 / s (desorption)
    
    
    # Unit LRMP5 
    Acyclic_model.root.input.model.unit_004.adsorption.is_kinetic = False    # Kinetic binding
    Acyclic_model.root.input.model.unit_004.adsorption.LIN_KA = [21.4286] # m^3 / (mol * s)   (mobile phase)
    Acyclic_model.root.input.model.unit_004.adsorption.LIN_KD = [1]      # 1 / s (desorption)
    
    # Unit LRMP6
    Acyclic_model.root.input.model.unit_004.adsorption.LIN_KA = [4.55] # m^3 / (mol * s)   (mobile phase)
    Acyclic_model.root.input.model.unit_004.adsorption.LIN_KD = [0.12]      # 1 / s (desorption)

    
    #To write out last output to check for steady state
    Acyclic_model.root.input['return'].WRITE_SOLUTION_LAST = True



    #% Input and connections

    
    #Sections
    Acyclic_model.root.input.solver.sections.nsec = 3
    Acyclic_model.root.input.solver.sections.section_times = [0, 250, 300, 3000]  
    
    ## Feed and Eluent concentration
    Acyclic_model.root.input.model.unit_000.sec_000.const_coeff = [1] #Inlet flowrate concentration
    Acyclic_model.root.input.model.unit_001.sec_000.const_coeff = [0] #Desorbent stream
    
    Acyclic_model.root.input.model.unit_000.sec_001.const_coeff = [0] #Inlet flowrate concentration
    Acyclic_model.root.input.model.unit_001.sec_001.const_coeff = [5] #Desorbent stream
    
    Acyclic_model.root.input.model.unit_000.sec_002.const_coeff = [0] #Inlet flowrate concentration
    Acyclic_model.root.input.model.unit_001.sec_002.const_coeff = [0] #Desorbent stream
    
    
    #Connections
    Acyclic_model.root.input.model.connections.nswitches = 1
    
    Acyclic_model.root.input.model.connections.switch_000.section = 0
    Acyclic_model.root.input.model.connections.switch_000.connections =[
        0, 2, -1, -1, LRMP_Q3,#flowrates, Q, m3/s
        2, 4, -1, -1, LRMP_Q3/2,
        2, 5, -1, -1, LRMP_Q3/2,
        1, 3, -1, -1, LRMP_Q3,
        3, 4, -1, -1, LRMP_Q3/2,
        3, 5, -1, -1, LRMP_Q3/2,
        4, 6, -1, -1, LRMP_Q3,
        5, 6, -1, -1, LRMP_Q3,
    ]

    
    #solution times
    Acyclic_model.root.input.solver.user_solution_times = np.linspace(0, 3000, 3000+1)
    
    
    
    #Time 
    # Tolerances for the time integrator
    Acyclic_model.root.input.solver.time_integrator.abstol = 1e-12 #absolute tolerance
    Acyclic_model.root.input.solver.time_integrator.algtol = 1e-10
    Acyclic_model.root.input.solver.time_integrator.reltol = 1e-10 #Relative tolerance
    Acyclic_model.root.input.solver.time_integrator.init_step_size = 1e-10
    Acyclic_model.root.input.solver.time_integrator.max_steps = 1000000
    
    
    
    #Solver options in general (not only for column although the same)
    Acyclic_model.root.input.model.solver.gs_type = 1
    Acyclic_model.root.input.model.solver.max_krylov = 0
    Acyclic_model.root.input.model.solver.max_restarts = 10
    Acyclic_model.root.input.model.solver.schur_safety = 1e-8
    Acyclic_model.root.input.solver.consistent_init_mode = 5 #necessary specifically for this sim
    Acyclic_model.root.input.solver.time_integrator.USE_MODIFIED_NEWTON = 1
    
    # Number of cores for parallel simulation
    Acyclic_model.root.input.solver.nthreads = 1
    
    
    #Specify which results we want to return
    # Return data
    Acyclic_model.root.input['return'].split_components_data = 0
    Acyclic_model.root.input['return'].split_ports_data = 0
    Acyclic_model.root.input['return'].unit_000.write_solution_bulk = 0
    Acyclic_model.root.input['return'].unit_000.write_solution_inlet = 0
    Acyclic_model.root.input['return'].unit_000.write_solution_outlet = 0
    Acyclic_model.root.input['return'].unit_002.write_solution_bulk = 0
    Acyclic_model.root.input['return'].unit_002.write_solution_inlet = 0
    Acyclic_model.root.input['return'].unit_002.write_solution_outlet = 1
    
    
    # Copy settings to the other unit operations
    Acyclic_model.root.input['return'].unit_001 = Acyclic_model.root.input['return'].unit_000
    Acyclic_model.root.input['return'].unit_003 = Acyclic_model.root.input['return'].unit_002
    Acyclic_model.root.input['return'].unit_004 = Acyclic_model.root.input['return'].unit_002
    Acyclic_model.root.input['return'].unit_005 = Acyclic_model.root.input['return'].unit_002
    Acyclic_model.root.input['return'].unit_006 = Acyclic_model.root.input['return'].unit_002
    
    
    return Acyclic_model

database_path = None
small_test = True
rdm_debug_mode = False
rerun_sims = False
output_path = "/"
n_jobs = 1
setting = []

def acyclic_systems_tests(n_jobs, database_path, output_path,
                                 cadet_path, small_test=False, **kwargs):

    nDisc = 4 if small_test else 6
    
    benchmark_config = {
        'cadet_config_jsons': [
            Acyclic_model1(nDisc,4,1)
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
    
    addition = acyclic_systems_tests(n_jobs, database_path, output_path, # todo
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







