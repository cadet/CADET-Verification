import os
import pandas as pd
import numpy as np

from cadet import Cadet

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def ASM3hC_SIMBA_test():

    # Set up Cadet Model
    """Setup basic CADET configuration used across all tests"""
    install_path = r'C:\Users\berger\CADET\out\install\DEBUG\bin\cadet-cli.exe'
    plot_results = True
    output_dir = r'C:\Users\berger\Documents\Projekts\MichealisMenten\output'
    
    model = Cadet()
    model.install_path = install_path
    
    model.root.input.model.nunits = 3
    ncomp = 13

    init_c = [0.1, 6750, 16, 1350, 0, 5, 100, 25, 6750, 2500, 1, 50, 50]

    sim_time = 2.0
    
    # Inlet
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = ncomp
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    
    # CSTR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = ncomp
    model.root.input.model.unit_001.init_liquid_volume = 1.0
    model.root.input.model.unit_001.init_c = init_c
    model.root.input.model.unit_001.const_solid_volume = 1.0
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
    model.root.input.solver.user_solution_times = np.linspace(0, sim_time, 1000)
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
        0, 1, -1, -1, 0.0,  # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1]
        1, 2, -1, -1, 0.0   # [unit_001, unit_002, all components, all components, Q/ m^3*s^-1]
    ]

    # Inlet coefficients - no inflow
    model.root.input.model.unit_000.sec_000.const_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0] * ncomp

    # Setup MM reaction with the new MICHAELIS_MENTEN model
    model.root.input.model.unit_001.reaction_model = 'ACTIVATED_SLUDGE_MODEL3'
    model.root.input.model.unit_001.reaction_bulk.asm_T = 12

    model.filename = f"{output_dir}/ASM3hC.h5"
    model.save()
    data = model.run()
    model.load()
    assert data.return_code == 0, f"MM simulation failed: {data.error_message}"

    
    # Get CADET simulation data
    cadet_time = model.root.output.solution.solution_times
    cadet_outlet = model.root.output.solution.unit_001.solution_outlet

    #get Simba simulation files (xml)

    simba_data = pd.read_excel("data/ref_CSTR_ASM3hC_simulation_results.xlsx")
    
    components = {
    'SS': {'cadet_idx': 0, 'simba_time': simba_data["t_ss"].values, 'simba_conc': simba_data["c_SS"].values},
    'SI': {'cadet_idx': 1, 'simba_time': simba_data["t_si"].values, 'simba_conc': simba_data["c_si"].values},
    'XS': {'cadet_idx': 2, 'simba_time': simba_data["t_xs"].values, 'simba_conc': simba_data["c_xs"].values},
    'XI': {'cadet_idx': 3, 'simba_time': simba_data["t_xi"].values, 'simba_conc': simba_data["c_xi"].values},
    'XH': {'cadet_idx': 4, 'simba_time': simba_data["t_xh"].values, 'simba_conc': simba_data["c_xh"].values},
    'XA': {'cadet_idx': 5, 'simba_time': simba_data["t_xa"].values, 'simba_conc': simba_data["c_xa"].values},
    'SNH': {'cadet_idx': 6, 'simba_time': simba_data["t_snh"].values, 'simba_conc': simba_data["c_snh"].values},
    'SNO': {'cadet_idx': 7, 'simba_time': simba_data["t_sno"].values, 'simba_conc': simba_data["c_sno"].values},
    'SN2': {'cadet_idx': 8, 'simba_time': simba_data["t_sn2"].values, 'simba_conc': simba_data["c_sn2"].values}
    }

    # Create figure for comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Compare each component
    for i, (name, data) in enumerate(components.items()):
        # Get CADET data for this component
        cadet_conc = cadet_outlet[:, data['cadet_idx']]
        cadet_interp = interp1d(cadet_time, cadet_conc, bounds_error=False, fill_value="extrapolate")
        cadet_interp_conc = cadet_interp(data['simba_time'])
        
        # Plot on the corresponding subplot
        ax = axes[i]
        ax.plot(cadet_time, cadet_conc, 'b-', label=f'CADET {name}')
        ax.plot(data['simba_time'], data['simba_conc'], 'ro', markersize=3, label=f'SIMBA {name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.grid(True)
        ax.legend()
        
    if output_dir:
        plt.savefig(output_dir, dpi=300, bbox_inches='tight')
    if plot_results:
        plt.show()

    for i, (name, data) in enumerate(components.items()):

        cadet_conc = cadet_outlet[:, data['cadet_idx']]
        cadet_interp = interp1d(cadet_time, cadet_conc, bounds_error=False, fill_value="extrapolate")
        cadet_interp_conc = cadet_interp(data['simba_time'])

        abs_error = np.mean((cadet_interp_conc - data['simba_conc']))
        rel_error = np.mean(np.abs((cadet_interp_conc - data['simba_conc'])/data['simba_conc'])) * 100

        if rel_error > 0.1:
            ValueError(f"Warning: Relative error for {name} is greater than 10%")
        
        if abs_error > 0.1:
                    ValueError(f"Warning: Absulute error for {name} is greater than 10%")