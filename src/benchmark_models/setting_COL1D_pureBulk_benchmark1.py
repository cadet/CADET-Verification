"""

This script defines a pure bulk benchmark for 1D cylindrical columns

"""

from addict import Dict
import numpy as np

def get_model(disc):
    
    model = Dict()
    
    model.input.model.nunits = 3
    
    # Flow sheet
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        0.e+00, 1.e+00,-1.e+00,-1.e+00, 6.e-05, 1.e+00, 2.e+00,-1.e+00,-1.e+00, 6.e-05
        ]
    model.input.model.connections.switch_000.section = 0
    
    #%% Column unit
    column = Dict()
    
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.ncomp = 1
    column.col_dispersion = 5.75e-08
    column.col_length = 0.014
    column.cross_section_area = (6.e-05 / 0.000575) / 0.37 # i.e. column.velocity = 0.000575
    column.npartype = 0
    column.col_porosity = 0.37
    column.total_porosity = column.col_porosity
    
    column.discretization = disc
    column.discretization.USE_ANALYTIC_JACOBIAN = disc.get('use_analytic_jacobian', 1)
    
    column.init_c = [ 0.0 ]
    
    model.input.model.unit_001 = column
    
    #%% time integration parameters
    # non-linear solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08
    # time integration / solver specifics
    model.input.solver.consistent_init_mode = 1
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = [ 0 ]
    model.input.solver.sections.section_times = [ 0.0, 10.0, 50.0 ]
    model.input.solver.time_integrator.ABSTOL = 1e-12
    model.input.solver.time_integrator.ALGTOL = 1e-10
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 10000
    model.input.solver.time_integrator.RELTOL = 1e-10
    model.input.solver.user_solution_times = np.linspace(0.0, 50.0, 50 + 1)
    
    #%% auxiliary units: inlet and outlet
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = 1
    model.input.model.unit_000.sec_000.const_coeff = [1.]
    model.input.model.unit_000.sec_000.cube_coeff = [0.]
    model.input.model.unit_000.sec_000.lin_coeff = [0.]
    model.input.model.unit_000.sec_000.quad_coeff = [0.]
    model.input.model.unit_000.sec_001.const_coeff = [0.]
    model.input.model.unit_000.sec_001.cube_coeff = [0.]
    model.input.model.unit_000.sec_001.lin_coeff = [0.]
    model.input.model.unit_000.sec_001.quad_coeff = [0.]
    model.input.model.unit_000.UNIT_TYPE = 'INLET'
    
    model.input.model.unit_002.ncomp = 1
    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    
    
    #%% return data
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_solution_bulk = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_coordinates = 0
    model.input['return'].unit_001.write_sens_bulk = 0
    model.input['return'].unit_001.write_sens_last = 0
    model.input['return'].unit_001.write_sens_outlet = 1
    model.input['return'].unit_001.write_solution_bulk = 0
    model.input['return'].unit_001.write_solution_inlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_001.write_solution_particle = 0
    model.input['return'].unit_002.write_solution_bulk = 0
    model.input['return'].unit_002.write_solution_inlet = 0
    model.input['return'].unit_002.write_solution_outlet = 0
    
    return model



from cadet import Cadet
import matplotlib.pyplot as plt
import src.utility.convergence as convergence

Cadet.cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

model = Cadet()


disc = Dict({
    "SPATIAL_METHOD": "FV",
    "NCOL": 100,
    "RECONSTRUCTION": "WENO",
    "weno": {
        "BOUNDARY_MODEL": 0,
        "WENO_EPS": 1e-10,
        "WENO_ORDER": 2
    },
    "GS_TYPE": 1,
    "MAX_KRYLOV": 0,
    "MAX_RESTARTS": 10,
    "SCHUR_SAFETY": 1.0e-8
})

disc = Dict({
    "SPATIAL_METHOD": "FV",
    "NCOL": 100,
    "RECONSTRUCTION": "KOREN",
    "koren": {
        "KOREN_EPS": 1e-10
    },
    "GS_TYPE": 1,
    "MAX_KRYLOV": 0,
    "MAX_RESTARTS": 10,
    "SCHUR_SAFETY": 1.0e-8
})

disc = Dict({
    "SPATIAL_METHOD": "FV",
    "NCOL": 100,
    "RECONSTRUCTION": "KOREN",
    "koren": {
        ""
        "KOREN_EPS": 1e-10
    },
    "GS_TYPE": 1,
    "MAX_KRYLOV": 0,
    "MAX_RESTARTS": 10,
    "SCHUR_SAFETY": 1.0e-8
})

model.root = get_model(disc)

model.filename = "weno1.h5"

model.save()

return_data = model.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")
    
model.load_from_file()

outlet = convergence.get_solution(model)
solution_time = convergence.get_solution_times(model)

plt.plot(solution_time, outlet)
plt.show()





