import unittest
import numpy as np
from cadet import Cadet
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp    

model = Cadet(r'C:\Users\jmbr\Cadet_testBuild\CADET_2DmodelsDG\out\install\aRELEASE\bin\cadet-cli.exe')
ncomp = 2
init_c = [1.0, 0.0]
sim_time = 10.0

km_a = 1.0
vmax = 1.0


"""Create a basic CSTR system with inlet, CSTR and outlet"""
# CSTR
model.root.input.model.nunits = 3

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
model.root.input.model.unit_001.use_analytic_jacobian = 0

# Outlet
model.root.input.model.unit_002.unit_type = 'OUTLET'
model.root.input.model.unit_002.ncomp = ncomp

# Return data
model.root.input['return'].split_components_data = 0
model.root.input['return'].split_ports_data = 0
model.root.input['return'].unit_001.write_solution_bulk = 0
model.root.input['return'].unit_001.write_solution_inlet = 0
model.root.input['return'].unit_001.write_solution_outlet = 1
model.root.input['return'].unit_001.write_sens_bulk = 0
model.root.input['return'].unit_001.write_sens_outlet = 1

# model.root.input['return'].unit_000 = model.root.input['return'].unit_001
# model.root.input['return'].unit_002 = model.root.input['return'].unit_001


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

# Inlet coefficients - no concentration added
model.root.input.model.unit_000.sec_000.const_coeff = [0.0] * ncomp
model.root.input.model.unit_000.sec_000.lin_coeff = [0.0] * ncomp
model.root.input.model.unit_000.sec_000.quad_coeff = [0.0] * ncomp
model.root.input.model.unit_000.sec_000.cube_coeff = [0.0] * ncomp


""" Configure the reaction system"""

model.root.input.model.unit_001.reaction_model = 'MASS_ACTION_LAW'
        
model.root.input.model.unit_001.reaction_bulk.mal_kfwd_bulk = [ 1.0 ]   
model.root.input.model.unit_001.reaction_bulk.mal_kbwd_bulk = [ 3.0 ]


model.root.input.model.unit_001.reaction_bulk.mal_stoichiometry_bulk = [-1,  1]

""" congifure the sensitivity analysis"""
model.root.input.sensitivity.nsens = 2
model.root.input.sensitivity.sens_method = 'ad1'

model.root.input.sensitivity.param_000.sens_unit = 1
model.root.input.sensitivity.param_000.sens_name = 'MAL_KFWD_BULK'
model.root.input.sensitivity.param_000.sens_comp = -1
model.root.input.sensitivity.param_000.sens_partype = -1
model.root.input.sensitivity.param_000.sens_reaction = 0
model.root.input.sensitivity.param_000.sens_boundphase = -1
model.root.input.sensitivity.param_000.sens_section = -1

model.root.input.sensitivity.param_001.sens_unit = 1
model.root.input.sensitivity.param_001.sens_name = 'MAL_STOICHIOMETRY_BULK'
model.root.input.sensitivity.param_001.sens_comp = 0
model.root.input.sensitivity.param_001.sens_partype = -1
model.root.input.sensitivity.param_001.sens_reaction = 0
model.root.input.sensitivity.param_001.sens_boundphase = -1
model.root.input.sensitivity.param_001.sens_section = -1



model.filename = "mal_sensitivity.h5"
model.save()
data_mm = model.run()
model.load()

print(data_mm.error_message)
print(data_mm.log)
