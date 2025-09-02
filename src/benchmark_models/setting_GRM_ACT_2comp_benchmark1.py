# -*- coding: utf-8 -*-
"""

This script implements an ACT binding setting for the GRM as used in Zhang et
al. (2025) 'An Affinity Complex Titration Isotherm for Mechanistic Modeling in
Protein A Chromatography'. The specific setting is given in the manuscript in
table 3, 5 ml MSS, mAb A.

""" 

import numpy as np
import matplotlib.pyplot as plt

from cadet import Cadet

def ACT_benchmark1(cadet_path, output_path,
                   run_simulation, plot_result): 

    ## set up
    injection_volume = 1e-6         ## m^3
    
    elution_pH_start = 5.5
    elution_pH_end = 3.3
    
    elution_start_volume = 8.66e-6  ## m^3
    gradient_length = 10            ## CV
    
    simulation_end_time = 1300
    Q = 3.5 /(6e7)                  ## volumetric flow rate m^3/s
    elution_start_time = elution_start_volume/Q
    
    column_length = 0.025        ## m
    column_volume = 5.025e-6     ## m^3
    protein_MW = 150             ## kDa
    
    c_feed = 10 / protein_MW     ## mol/m^3
    
    column_porosity = 0.31
    particle_porosity = 0.95
    
    Keq = 268                    ## ml/mg
    Q_max = 55.6                 ## mg/ml column volume
    
    elution_end_time = elution_start_time + gradient_length*column_volume/Q
    total_porosity = column_porosity + (1.0-column_porosity)*particle_porosity
    
    model = Cadet()
    model.install_path = cadet_path
    
    model.root.input.model.nunits = 3
    
    # Inlet 
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = 2
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    
    model.root.input.model.unit_001.unit_type = 'COLUMN_MODEL_1D'
    model.root.input.model.unit_001.ncomp = 2
    
    ## Column

    model.root.input.model.unit_001.col_length = column_length              # m              
    model.root.input.model.unit_001.cross_section_area = column_volume/column_length   # m^2
    model.root.input.model.unit_001.col_porosity = column_porosity              # 1
    model.root.input.model.unit_001.col_dispersion = 1.36e-8           # m^2/s       
    model.root.input.model.unit_001.init_c = [elution_pH_start, 0.0, ]
    ### discretization
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.SPATIAL_METHOD = "FV"
    model.root.input.model.unit_001.discretization.ncol = 50
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = 2
    model.root.input.model.unit_001.discretization.gs_type = 1
    model.root.input.model.unit_001.discretization.max_krylov = 0
    model.root.input.model.unit_001.discretization.max_restarts = 10
    model.root.input.model.unit_001.discretization.schur_safety = 1.0e-8

    ## Particles
    model.root.input.model.unit_001.npartype = 1
    model.root.input.model.unit_001.particle_type_000.has_film_diffusion = True
    model.root.input.model.unit_001.particle_type_000.has_pore_diffusion = True
    model.root.input.model.unit_001.particle_type_000.has_surface_diffusion = False
    model.root.input.model.unit_001.particle_type_000.par_porosity = particle_porosity              # 1      
    model.root.input.model.unit_001.particle_type_000.par_radius = 0.0425e-3            # m     
    model.root.input.model.unit_001.particle_type_000.film_diffusion = [1, 1.41e-5,]     # m/s
    model.root.input.model.unit_001.particle_type_000.pore_diffusion = [1, 1.99e-11]      # m^2/s  
    model.root.input.model.unit_001.particle_type_000.surface_diffusion = [0.0, 0.0]      
    
    model.root.input.model.unit_001.particle_type_000.nbound = [0, 1,]

    model.root.input.model.unit_001.particle_type_000.adsorption_model = 'AFFINITY_COMPLEX_TITRATION'
    
    model.root.input.model.unit_001.particle_type_000.adsorption.is_kinetic = 1
    model.root.input.model.unit_001.particle_type_000.adsorption.act_ka = [1.0, Keq*protein_MW*(1.0-total_porosity), ]          ##  m^3 solid phase / mol protein / s   ml/mg=m^3/kg    1 kda = 1 kg/mol
    model.root.input.model.unit_001.particle_type_000.adsorption.act_kd = [1.0, 1.0]                                            ## s^-1
    model.root.input.model.unit_001.particle_type_000.adsorption.act_qmax = [1e-10, Q_max/protein_MW/(1.0-total_porosity),]     ##  mol/m^3 solid phase mg/ml=kg/m^3 / 150 kg/mol = 1/150 mol/m^3
    
    model.root.input.model.unit_001.particle_type_000.adsorption.act_etaA = [0, 1.81,  ]
    model.root.input.model.unit_001.particle_type_000.adsorption.act_pkaA = [0, 2.07, ]
    model.root.input.model.unit_001.particle_type_000.adsorption.act_etaG = [0, 2.28, ]
    model.root.input.model.unit_001.particle_type_000.adsorption.act_pkaG = [0, 5.29, ]

    model.root.input.model.unit_001.particle_type_000.init_cp = [elution_pH_start, 0.0]
    model.root.input.model.unit_001.particle_type_000.init_cs = [0.0, 0.0]
    
    ### discretization
    model.root.input.model.unit_001.particle_type_000.discretization.SPATIAL_METHOD = "FV"
    model.root.input.model.unit_001.particle_type_000.discretization.par_disc_type = 'EQUIDISTANT_PAR'    
    model.root.input.model.unit_001.particle_type_000.discretization.NCELLS = 12
    
    
    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = 2
    
    model.root.input.solver.sections.nsec = 4
    model.root.input.solver.sections.section_times = [0.0, injection_volume/Q, elution_start_time, elution_end_time, simulation_end_time]   # s
    model.root.input.solver.sections.section_continuity = [0,0,0,0]
    
    model.root.input.model.unit_000.sec_000.const_coeff = [elution_pH_start, c_feed,] # mol / m^3       mg/ml = kg/m^3;  1 kda = 1 kg/mol
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0, 0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0, 0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0, 0.0,]
    
    model.root.input.model.unit_000.sec_001.const_coeff = [elution_pH_start, 0.0] # mol / m^3
    model.root.input.model.unit_000.sec_001.lin_coeff = [0.0, 0.0,]
    model.root.input.model.unit_000.sec_001.quad_coeff = [0.0, 0.0,]
    model.root.input.model.unit_000.sec_001.cube_coeff = [0.0, 0.0,]
    
    model.root.input.model.unit_000.sec_002.const_coeff = [elution_pH_start, 0.0] # mol / m^3
    model.root.input.model.unit_000.sec_002.lin_coeff = [-(elution_pH_start-elution_pH_end)/(elution_end_time - elution_start_time), 0.0, ]
    model.root.input.model.unit_000.sec_002.quad_coeff = [0.0, 0.0,]
    model.root.input.model.unit_000.sec_002.cube_coeff = [0.0, 0.0,]
    
    model.root.input.model.unit_000.sec_003.const_coeff = [elution_pH_end, 0.0] # mol / m^3
    model.root.input.model.unit_000.sec_003.lin_coeff = [0.0,0.0,]
    model.root.input.model.unit_000.sec_003.quad_coeff = [0.0,0.0,]
    model.root.input.model.unit_000.sec_003.cube_coeff = [0.0,0.0,]
    
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q]
    
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8
    
    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1
    
    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    
    # Return data
    model.root.input['return'].split_components_data = 1
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1
    
    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000
    
    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(0, simulation_end_time, 101)
    
    model.filename = output_path + '/GRM_ACT_2comp_benchmark1.h5'
    model.save()
    
    if run_simulation:
        data = model.run_simulation()
        if not data.return_code == 0:
            raise Exception(f"simulation failed with {data.error_message}")
            
        if plot_result:
    
            model.load_from_file() 
            time = model.root.output.solution.solution_times
            c = model.root.output.solution.unit_001.solution_outlet_comp_001
            pH_outlet = model.root.output.solution.unit_001.solution_outlet_comp_000
            pH_inlet = model.root.output.solution.unit_000.solution_outlet_comp_000

            measurement_factor = 710     ## converts conc to mAU
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(time[1:]/60*Q*6e7, c[1:]*150*measurement_factor, c="orange", label="mAb")
            ax.set_xlabel(r'Volume/mL')
            ax.set_ylabel(r'Protein/mAU')
            
            ax_ph = ax.twinx()
            ax_ph.plot(time/60*Q*6e7, pH_outlet, label='pH')
            ax_ph.set_ylabel('pH')
            plt.savefig(output_path + '/GRM_ACT_2comp_benchmark1.png')
            plt.close()