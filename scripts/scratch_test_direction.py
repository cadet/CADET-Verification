import numpy as np
from addict import Dict
from cadet import Cadet

install_path = r"C:\Users\jmbr\Desktop\CADET_compiled\master6_geomIntChange_25f9ff5\aRelease"

X0 = 0.002
X1 = 0.01
L = X1 - X0
H = 0.05
epsb = 0.4
v_ref = 1e-4
Q = v_ref * X1 * 2.0 * np.pi * H * epsb

eps_t = 1e-3  # tiny dummy section duration
tmax = 50.0

m = Dict()
m.input.model.nunits = 3
m.input.model.connections.nswitches = 1
m.input.model.connections.switch_000.connections = [
    0.0, 1.0, -1.0, -1.0, Q,
    1.0, 2.0, -1.0, -1.0, Q,
]
m.input.model.connections.switch_000.section = 0

m.input.model.solver.gs_type = 1
m.input.model.solver.max_krylov = 0
m.input.model.solver.max_restarts = 10
m.input.model.solver.schur_safety = 1e-8

m.input.model.unit_000.unit_type = 'INLET'
m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
m.input.model.unit_000.ncomp = 1
for i, val in enumerate([1.0, 1.0, 0.0]):
    sec = f"sec_{i:03d}"
    m.input.model.unit_000[sec].const_coeff = [val]
    m.input.model.unit_000[sec].lin_coeff = [0.0]
    m.input.model.unit_000[sec].quad_coeff = [0.0]
    m.input.model.unit_000[sec].cube_coeff = [0.0]

col = m.input.model.unit_001
col.unit_type = 'COLUMN_MODEL_1D'
col.ncomp = 1
col.geometry = 'RADIAL_FLOW_CYLINDER_SHELL'
col.cross_section_area_outer = 2.0 * np.pi * X1 * H
col.cylinder_height = H
col.bed_length = L
col.col_porosity = 1.0
col.total_porosity = 1.0
col.col_dispersion = [1e-8]
col.forward_flow = [0, 1, 1]
col.npartype = 0
col.init_c = [0.0]

col.discretization.use_analytic_jacobian = 1
col.discretization.spatial_method = 'FV'
col.discretization.ncol = 20
col.discretization.reconstruction = 'WENO'
col.discretization.weno.weno_order = 1
col.discretization.weno.weno_eps = 1e-10
col.discretization.weno.boundary_model = 0

m.input.model.unit_002.unit_type = 'OUTLET'
m.input.model.unit_002.ncomp = 1

m.input['return'].split_components_data = 0
m.input['return'].split_ports_data = 0
m.input['return'].unit_001.write_solution_bulk = 1
m.input['return'].unit_001.write_solution_outlet = 1
m.input['return'].unit_001.write_coordinates = 1

m.input.solver.consistent_init_mode = 1
m.input.solver.nthreads = 1
m.input.solver.sections.nsec = 3
m.input.solver.sections.section_continuity = [0, 0]
m.input.solver.sections.section_times = [0.0, eps_t, tmax / 2, tmax]
m.input.solver.time_integrator.abstol = 1e-10
m.input.solver.time_integrator.reltol = 1e-8
m.input.solver.time_integrator.algtol = 1e-10
m.input.solver.time_integrator.init_step_size = 1e-10
m.input.solver.time_integrator.max_steps = 1000000
m.input.solver.user_solution_times = np.linspace(0.0, tmax, 101)

cadet = Cadet(install_path=install_path)
cadet.root.input = m.input
cadet.filename = r"C:\Users\jmbr\software\CADET-Verification\scripts\scratch_test_direction.h5"
cadet.save()
rc = cadet.run_simulation()
print("returncode:", rc.return_code)
print("log:", rc.log[-3000:] if getattr(rc, 'log', None) else None)
print("error_message:", getattr(rc, 'error_message', None))
print(rc)

if rc.return_code == 0:
    cadet.load_from_file()
    bulk = np.array(cadet.root.output.solution.unit_001.solution_bulk)
    coords = np.array(cadet.root.output.coordinates.unit_001.axial_coordinates)
    print("coords (index0->last):", coords)
    for t_idx in [2, 5, 10]:
        print("t_idx", t_idx, "profile:", bulk[t_idx, :, 0])
