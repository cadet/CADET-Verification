import os
import pandas as pd
import numpy as np

from cadet import Cadet

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def ASM3h_SIMBA_test(output_path, cadet_path, data_path):

    # get data from SIMBA
    simba_data = pd.read_excel(data_path)

    # parameters for the simulation 
    ncomp = 13
    # S0: 0, SS: 1, SNH: 2, SNO: 3, SN2: 4, SALK: 5, SI: 6, XI: 7, XS: 8, XH: 9, XSTO: 10, XA: 11, XMI: 12
    init_c = [0.1, 6750, 16, 1350, 0, 5, 100, 25, 6750, 2500, 1, 50, 50]
    sim_time = 2.0


    # Cadet model setup
    model = Cadet(cadet_path)
    model.root.input.model.nunits = 3

    # Inlet
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = ncomp
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    # CSTR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = ncomp
    model.root.input.model.unit_001.init_liquid_volume = 1000.0
    model.root.input.model.unit_001.init_c = init_c
    model.root.input.model.unit_001.const_solid_volume = 0.0
    model.root.input.model.unit_001.use_analytic_jacobian = 1

    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = ncomp

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 1
    model.root.input['return'].unit_000.write_solution_inlet = 1
    model.root.input['return'].unit_000.write_solution_outlet = 1

    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    """Configure solver settings"""

    simba_time = np.array(simba_data["time"].values)
    model.root.input.solver.user_solution_times = simba_time
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, sim_time]
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1

    """Connect the units together"""
    # Connections
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, 50,  # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1]
        1, 2, -1, -1, 50   # [unit_001, unit_002, all components, all components, Q/ m^3*s^-1]
    ]

    # Inlet coefficients - no inflow
    model.root.input.model.unit_000.sec_000.const_coeff = [0.0, 206, 54.82, 0.0, 0.0, 9.525, 42.0, 101.1, 250.6, 100.2, 0.0, 0.0001, 112.8]
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0] * ncomp

    # Setup MM reaction with the new MICHAELIS_MENTEN model
    model.root.input.model.unit_001.reaction_model = 'ACTIVATED_SLUDGE_MODEL3'
    model.root.input.model.unit_001.reaction_bulk.asm3_insi = 0.01
    model.root.input.model.unit_001.reaction_bulk.asm3_inss = 0.03
    model.root.input.model.unit_001.reaction_bulk.asm3_inxi = 0.04
    model.root.input.model.unit_001.reaction_bulk.asm3_inxs = 0.03
    model.root.input.model.unit_001.reaction_bulk.asm3_inbm = 0.07
    model.root.input.model.unit_001.reaction_bulk.asm3_ivss_xi = 0.751879699 # not used
    model.root.input.model.unit_001.reaction_bulk.asm3_ivss_xs = 0.555555556 # not used
    model.root.input.model.unit_001.reaction_bulk.asm3_ivss_sto = 0.6 # not used
    model.root.input.model.unit_001.reaction_bulk.asm3_ivss_bm = 0.704225352
    model.root.input.model.unit_001.reaction_bulk.asm3_itss_vss_bm = 1.086956522


    model.root.input.model.unit_001.reaction_bulk.asm3_fiss_bm_prod = 1
    model.root.input.model.unit_001.reaction_bulk.asm3_fsi = 0
    model.root.input.model.unit_001.reaction_bulk.asm3_yh_aer = 0.8
    model.root.input.model.unit_001.reaction_bulk.asm3_yh_anox = 0.65

    model.root.input.model.unit_001.reaction_bulk.asm3_ysto_aer = 0.8375
    model.root.input.model.unit_001.reaction_bulk.asm3_ysto_anox = 0.7
    model.root.input.model.unit_001.reaction_bulk.asm3_fxi = 0.2
    model.root.input.model.unit_001.reaction_bulk.asm3_ya = 0.24
    model.root.input.model.unit_001.reaction_bulk.asm3_kh20 = 9
    model.root.input.model.unit_001.reaction_bulk.asm3_kx = 1
    model.root.input.model.unit_001.reaction_bulk.asm3_ksto20 = 12
    model.root.input.model.unit_001.reaction_bulk.asm3_mu_h20 = 3
    model.root.input.model.unit_001.reaction_bulk.asm3_eta_hno3 = 0.5
    model.root.input.model.unit_001.reaction_bulk.asm3_khO2 = 0.2
    model.root.input.model.unit_001.reaction_bulk.asm3_khss = 10
    model.root.input.model.unit_001.reaction_bulk.asm3_khno3 = 0.5
    model.root.input.model.unit_001.reaction_bulk.asm3_khnh4 = 0.01
    model.root.input.model.unit_001.reaction_bulk.asm3_khalk = 0.1
    model.root.input.model.unit_001.reaction_bulk.asm3_khsto = 0.1
    model.root.input.model.unit_001.reaction_bulk.asm3_mu_aut20 = 1.12
    model.root.input.model.unit_001.reaction_bulk.asm3_baut20 = 0.18
    model.root.input.model.unit_001.reaction_bulk.asm3_etah_end = 0.5
    model.root.input.model.unit_001.reaction_bulk.asm3_etan_end = 0.5
    model.root.input.model.unit_001.reaction_bulk.asm3_kno2 = 0.5
    model.root.input.model.unit_001.reaction_bulk.asm3_knnh4 = 0.7
    model.root.input.model.unit_001.reaction_bulk.asm3_knalk = 0.5
    model.root.input.model.unit_001.reaction_bulk.asm3_t = 12
    model.root.input.model.unit_001.reaction_bulk.asm3_bh20 = 0.33

    model.root.input.model.unit_001.reaction_bulk.asm3_v = 1000.0
    model.root.input.model.unit_001.reaction_bulk.asm3_io2 = 0.0

    model.filename = "ASM3h.h5"
    model.save()
    data = model.run()
    model.load()
    assert data.return_code == 0, f"MM simulation failed: {data.error_message}"

    
    # Get CADET simulation data
    cadet_time = model.root.output.solution.solution_times
    cadet_outlet = model.root.output.solution.unit_001.solution_outlet

    simba_time = simba_data["time"].values
    components = {
    'SS': {'cadet_idx': 1, 'simba_conc': simba_data["c_ss"].values},
    'SI': {'cadet_idx': 6, 'simba_conc': simba_data["c_si"].values},
    'XS': {'cadet_idx': 8, 'simba_conc': simba_data["c_xs"].values},
    'XI': {'cadet_idx': 7, 'simba_conc': simba_data["c_xi"].values},
    'XH': {'cadet_idx': 9,  'simba_conc': simba_data["c_xh"].values},
    'XA': {'cadet_idx': 11,  'simba_conc': simba_data["c_xa"].values},
    'SNH': {'cadet_idx': 2, 'simba_conc': simba_data["c_snh"].values},
    'SNO': {'cadet_idx': 3,  'simba_conc': simba_data["c_sno"].values},
    'SN2': {'cadet_idx': 4,  'simba_conc': simba_data["c_sn2"].values}
    }
    # Get CADET simulation data
    cadet_time = model.root.output.solution.solution_times
    cadet_outlet = model.root.output.solution.unit_001.solution_outlet

    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    # Compare each component
    for i, (name, data) in enumerate(components.items()):
        # Get CADET data for this component
        cadet_conc = cadet_outlet[:, data['cadet_idx']]
        
        # Plot on the corresponding subplot
        ax = axes[i]
        ax.plot(cadet_time, cadet_conc, 'b-', label=f'CADET {name}')
        ax.plot(simba_time, data['simba_conc'], 'ro', markersize=3, label=f'SIMBA {name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.grid(True)
        ax.legend()
    #.savefig(output_path)
    plt.show()
    

    for i, (name, data) in enumerate(components.items()):

        # interpolate CADET data to SIMBA time
        cadet_conc = cadet_outlet[:, data['cadet_idx']]
        cadet_interp = interp1d(cadet_time, cadet_conc, bounds_error=False, fill_value="extrapolate")
        cadet_interp_conc = cadet_interp(simba_time)

        abs_error = np.mean((cadet_interp_conc - data['simba_conc']))
        if abs_error > 0.1:
            ValueError(f"Warning: Absulute error for {name} is greater than 10%")
        else:
            print(f"Absolute error for {name} is within 10%")
        
        max_error = np.max(np.abs(cadet_interp_conc - data['simba_conc']))
        if max_error > 0.1:
            ValueError(f"Warning: Maximum error for {name} is greater than 10%")
        else:
            print(f"Maximum error for {name} is within 10%")

def ASM3h_with_MAL_in_CSTR_test(cadet_path):
    
    # discription
    """
    This function sets up and runs a CADET simulation for the ASM3h model with MAL
    It is using the new interface for reactions in CADET
    """
    # parameters for the simulation 
    ncomp = 13
    # S0: 0, SS: 1, SNH: 2, SNO: 3, SN2: 4, SALK: 5, SI: 6, XI: 7, XS: 8, XH: 9, XSTO: 10, XA: 11, XMI: 12
    init_c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    sim_time = 1.0


    # Cadet model setup
    model = Cadet(cadet_path)
    model.root.input.model.nunits = 1

    # CSTR
    model.root.input.model.unit_000.unit_type = 'CSTR'
    model.root.input.model.unit_000.ncomp = ncomp
    model.root.input.model.unit_000.init_liquid_volume = 1000.0
    model.root.input.model.unit_000.init_c = init_c
    model.root.input.model.unit_000.const_solid_volume = 0.0
    model.root.input.model.unit_000.use_analytic_jacobian = 1

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 1
    model.root.input['return'].unit_000.write_solution_inlet = 1
    model.root.input['return'].unit_000.write_solution_outlet = 1


    """Configure solver settings"""
    model.root.input.solver.user_solution_times = np.array([0.0, sim_time])
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, sim_time]
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1

    """Connect the units together"""
    # Connections
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
    ]


    # Setup MM reaction with the new MICHAELIS_MENTEN model
    model.root.input.model.unit_000.reaction_bulk.nreac = 2
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.reaction_type  = 'ACTIVATED_SLUDGE_MODEL3'
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_insi = 0.01
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_inss = 0.03
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_inxi = 0.04
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_inxs = 0.03
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_inbm = 0.07
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_ivss_xi = 0.751879699 # not used
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_ivss_xs = 0.555555556 # not used
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_ivss_sto = 0.6 # not used
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_ivss_bm = 0.704225352
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_itss_vss_bm = 1.086956522


    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_fiss_bm_prod = 1
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_fsi = 0
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_yh_aer = 0.8
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_yh_anox = 0.65

    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_ysto_aer = 0.8375
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_ysto_anox = 0.7
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_fxi = 0.2
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_ya = 0.24
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_kh20 = 9
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_kx = 1
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_ksto20 = 12
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_mu_h20 = 3
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_eta_hno3 = 0.5
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_khO2 = 0.2
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_khss = 10
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_khno3 = 0.5
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_khnh4 = 0.01
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_khalk = 0.1
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_khsto = 0.1
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_mu_aut20 = 1.12
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_baut20 = 0.18
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_etah_end = 0.5
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_etan_end = 0.5
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_kno2 = 0.5
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_knnh4 = 0.7
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_knalk = 0.5
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_t = 12
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_bh20 = 0.33

    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_v = 1000.0
    model.root.input.model.unit_000.reaction_bulk.reaction_model_000.asm3_io2 = 0.0

    model.root.input.model.unit_000.reaction_bulk.reaction_model_001.reaction_type = "MASS_ACTION_LAW"
    model.root.input.model.unit_000.reaction_bulk.reaction_model_001.mal_kfwd = [0.1]
    model.root.input.model.unit_000.reaction_bulk.reaction_model_001.mal_kbwd = [0.3]
    model.root.input.model.unit_000.reaction_bulk.reaction_model_001.mal_stoichiometry = [-1,1,0,0,0,0,0,0,0,0,0,0,0]


    model.filename = "ASM3h_with_mal_cstr.h5"
    model.save()
    data = model.run()
    model.load()
    assert data.return_code == 0, f"MM simulation failed: {data.error_message}"

    # plot results
    cadet_time = model.root.output.solution.solution_times
    cadet_outlet = model.root.output.solution.unit_000.solution_outlet

    plt.figure(figsize=(10, 6))
    plt.plot(cadet_time, cadet_outlet[:, 1], label='SS')
    plt.plot(cadet_time, cadet_outlet[:, 2], label='SNH')
    plt.plot(cadet_time, cadet_outlet[:, 3], label='SNO')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (mg/L)')
    plt.title('CADRE Simulation Results')
    plt.legend()
    plt.grid()
    plt.show()


def ASM3h_with_MAL_in_LRMP_test(cadet_path):
    
    # discription
    """
    This function sets up and runs a CADET simulation for the ASM3h model with MAL
    It is using the new interface for reactions in CADET
    """
    # parameters for the simulation 
    ncomp = 13
    # S0: 0, SS: 1, SNH: 2, SNO: 3, SN2: 4, SALK: 5, SI: 6, XI: 7, XS: 8, XH: 9, XSTO: 10, XA: 11, XMI: 12
    init_c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    sim_time = 1.0


    # Cadet model setup
    model = Cadet(cadet_path)
    model.root.input.model.nunits = 1


    model.root.input.model.unit_000.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES'
    model.root.input.model.unit_000.ncomp = ncomp
    model.root.input.model.unit_000.nbound = [1.0] * ncomp
    model.root.input.model.unit_000.init_liquid_volume = 1.0
    model.root.input.model.unit_000.init_c = init_c 
    model.root.input.model.unit_000.init_q = [0.0] * ncomp
    model.root.input.model.unit_000.col_dispersion = 1e-5
    model.root.input.model.unit_000.col_length  = 0.6
    model.root.input.model.unit_000.col_porosity  = 0.37
    model.root.input.model.unit_000.film_diffusion = [1e-5] * ncomp

    model.root.input.model.unit_000.par_porosity = 0.33
    model.root.input.model.unit_000.par_radius = 1e-6
    model.root.input.model.unit_000.cross_section_area = 1.0386890710931253E-4

    model.root.input.model.unit_000.discretization.ncol                  = 1
    model.root.input.model.unit_000.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_000.discretization.weno.boundary_model   = 0 
    model.root.input.model.unit_000.discretization.weno.weno_eps         = 1e-10
    model.root.input.model.unit_000.discretization.weno.weno_order       = 1
    model.root.input.model.unit_000.discretization.max_restarts          = 10
    model.root.input.model.unit_000.discretization.gs_type              = 1
    model.root.input.model.unit_000.discretization.max_krylov           = 0
    model.root.input.model.unit_000.discretization.schur_safety          = 1e-8
    model.root.input.model.unit_000.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_000.discretization.par_disc_type = 'EQUIDISTANT_PAR'

    """Configure solver settings"""
    model.root.input.solver.user_solution_times = np.linspace(0, sim_time, 1000)
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, sim_time]
    model.root.input.solver.sections.section_continuity = 0

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = []

    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 1
    model.root.input['return'].unit_000.write_solution_particle = 1
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 1
    model.root.input['return'].unit_000.write_sens_outlet = 0

    # Adsorption model -> have to be set
    model.root.input.model.unit_000.adsorption_model = 'LINEAR'
    model.root.input.model.unit_000.adsorption.is_kinetic  = True
    model.root.input.model.unit_000.adsorption.lin_ka = [0] * ncomp
    model.root.input.model.unit_000.adsorption.lin_kd = [0] * ncomp


    # Setup MM reaction with the new MICHAELIS_MENTEN model
    model.root.input.model.unit_000.reaction_particle_000.nreac = 2
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.reaction_type  = 'ACTIVATED_SLUDGE_MODEL3'
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_insi = 0.01
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_inss = 0.03
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_inxi = 0.04
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_inxs = 0.03
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_inbm = 0.07
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_ivss_xi = 0.751879699 # not used
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_ivss_xs = 0.555555556 # not used
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_ivss_sto = 0.6 # not used
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_ivss_bm = 0.704225352
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_itss_vss_bm = 1.086956522


    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_fiss_bm_prod = 1
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_fsi = 0
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_yh_aer = 0.8
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_yh_anox = 0.65

    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_ysto_aer = 0.8375
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_ysto_anox = 0.7
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_fxi = 0.2
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_ya = 0.24
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_kh20 = 9
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_kx = 1
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_ksto20 = 12
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_mu_h20 = 3
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_eta_hno3 = 0.5
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_khO2 = 0.2
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_khss = 10
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_khno3 = 0.5
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_khnh4 = 0.01
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_khalk = 0.1
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_khsto = 0.1
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_mu_aut20 = 1.12
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_baut20 = 0.18
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_etah_end = 0.5
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_etan_end = 0.5
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_kno2 = 0.5
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_knnh4 = 0.7
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_knalk = 0.5
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_t = 12
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_bh20 = 0.33

    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_v = 1000.0
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_000.asm3_io2 = 0.0

    model.root.input.model.unit_000.reaction_particle_000.reaction_model_001.reaction_type = "MASS_ACTION_LAW"
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_001.mal_kfwd = [1]
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_001.mal_kbwd = [0.5]
    model.root.input.model.unit_000.reaction_particle_000.reaction_model_001.mal_stoichiometry = [-1,1,0,0,-1,0,0,1,0,0,0,0,0]


    model.filename = "ASM3h_with_mal_cstr.h5"
    model.save()
    data = model.run()
    model.load()
    assert data.return_code == 0, f"MM simulation failed: {data.error_message}"

    # plot results
    cadet_time = model.root.output.solution.solution_times
    cadet_outlet = model.root.output.solution.unit_000.solution_outlet

    plt.figure(figsize=(10, 6))
    plt.plot(cadet_time, cadet_outlet[:, 1], label='SS')
    plt.plot(cadet_time, cadet_outlet[:, 2], label='SNH')
    plt.plot(cadet_time, cadet_outlet[:, 3], label='SNO')
    plt.plot(cadet_time, cadet_outlet[:, 4], label='SN2')
    plt.plot(cadet_time, cadet_outlet[:, 5], label='SALK')
    plt.plot(cadet_time, cadet_outlet[:, 6], label='SI')
    plt.plot(cadet_time, cadet_outlet[:, 7], label='XI')
    plt.plot(cadet_time, cadet_outlet[:, 8], label='XS')
    plt.plot(cadet_time, cadet_outlet[:, 9], label='XH')
    plt.plot(cadet_time, cadet_outlet[:, 10], label='XSTO')
    plt.plot(cadet_time, cadet_outlet[:, 11], label='XA')
    plt.plot(cadet_time, cadet_outlet[:, 12], label='XMI')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (mg/L)')
    plt.title('CADRE Simulation Results')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    cadet_path = r"C:\\Users\\berger\\CADET-Core\\out\\install\\DEBUG\\bin\\cadet-cli.exe"

    ASM3h_with_MAL_in_LRMP_test(cadet_path)
