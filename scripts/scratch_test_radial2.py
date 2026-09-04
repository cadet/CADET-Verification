import numpy as np
from addict import Dict
from cadet import Cadet

install_path = r"C:\Users\jmbr\Desktop\CADET_compiled\master6_geomIntChange_25f9ff5\aRelease"

X0 = 0.002
X1 = 0.01
L = X1 - X0
H = 0.05  # cylinder height (arbitrary, cancels in dimensionless numbers)
epsb = 0.4
epsp = 0.4
Rp = 5e-5
v_ref = 1e-4
Q = v_ref * X1 * 2.0 * np.pi * H * epsb  # gives velocity_coeff = v_ref*X1, i.e. v(X1)=v_ref

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
m.input.model.unit_000.sec_000.const_coeff = [1.0]
m.input.model.unit_000.sec_000.lin_coeff = [0.0]
m.input.model.unit_000.sec_000.quad_coeff = [0.0]
m.input.model.unit_000.sec_000.cube_coeff = [0.0]

col = m.input.model.unit_001
col.unit_type = 'GENERAL_RATE_MODEL'
col.ncomp = 1
col.geometry = 'RADIAL_FLOW_CYLINDER_SHELL'
col.cross_section_area_outer = 2.0 * np.pi * X1 * H
col.cylinder_height = H
col.bed_length = L
col.col_porosity = epsb
col.total_porosity = epsb + (1 - epsb) * epsp
col.col_dispersion = [L / 100.0]
col.col_dispersion_dep = 'POWER_LAW'
col.col_dispersion_dep_exponent = 1.0
col.forward_flow = [0]
col.npartype = 1
col.par_type_volfrac = 1

col.discretization.use_analytic_jacobian = 1
col.discretization.spatial_method = 'FV'
col.discretization.ncol = 20
col.discretization.reconstruction = 'WENO'
col.discretization.weno.weno_order = 1
col.discretization.weno.weno_eps = 1e-10
col.discretization.weno.boundary_model = 0
col.discretization.gs_type = 1
col.discretization.max_krylov = 0
col.discretization.max_restarts = 10
col.discretization.schur_safety = 1e-8

col.particle_type_000.nbound = [1]
col.particle_type_000.adsorption_model = 'LINEAR'
col.particle_type_000.adsorption.is_kinetic = 0
col.particle_type_000.adsorption.lin_ka = [2.0]
col.particle_type_000.adsorption.lin_kd = [1.0]
col.particle_type_000.init_cp = [0.0]
col.particle_type_000.init_cs = [0.0]
col.particle_type_000.has_film_diffusion = 1
col.particle_type_000.film_diffusion = [1e-4]
col.particle_type_000.film_diffusion_dep = 'POWER_LAW'
col.particle_type_000.film_diffusion_dep_exponent = 1.0 / 3.0
col.particle_type_000.film_diffusion_dep_base = v_ref ** (-1.0 / 3.0)
col.particle_type_000.par_coreradius = 0.0
col.particle_type_000.par_porosity = epsp
col.particle_type_000.par_radius = Rp
col.particle_type_000.has_pore_diffusion = 1
col.particle_type_000.has_surface_diffusion = 0
col.particle_type_000.par_geom = 'SPHERE'
col.particle_type_000.pore_diffusion = [1e-10]
col.particle_type_000.surface_diffusion = [0.0]
col.particle_type_000.discretization.spatial_method = 'FV'
col.particle_type_000.discretization.par_disc_type = 'EQUIDISTANT_PAR'
col.particle_type_000.discretization.ncells = 2
col.particle_type_000.discretization.fv_boundary_order = 2

col.init_c = [0.0]

m.input.model.unit_002.unit_type = 'OUTLET'
m.input.model.unit_002.ncomp = 1

m.input['return'].split_components_data = 0
m.input['return'].split_ports_data = 0
m.input['return'].unit_001.write_solution_bulk = 1
m.input['return'].unit_001.write_solution_inlet = 0
m.input['return'].unit_001.write_solution_outlet = 1
m.input['return'].unit_001.write_coordinates = 1

m.input.solver.consistent_init_mode = 1
m.input.solver.nthreads = 1
m.input.solver.sections.nsec = 1
m.input.solver.sections.section_continuity = [0]
m.input.solver.sections.section_times = [0.0, 200.0]
m.input.solver.time_integrator.abstol = 1e-10
m.input.solver.time_integrator.reltol = 1e-8
m.input.solver.time_integrator.algtol = 1e-10
m.input.solver.time_integrator.init_step_size = 1e-8
m.input.solver.time_integrator.max_steps = 1000000
m.input.solver.user_solution_times = np.linspace(0.0, 200.0, 201)

cadet = Cadet(install_path=install_path)
cadet.root.input = m.input
cadet.filename = r"C:\Users\jmbr\software\CADET-Verification\scripts\scratch_test_radial2.h5"
cadet.save()
rc = cadet.run_simulation()
print("returncode:", rc.return_code)
print("log:", rc.log[-3000:] if getattr(rc, 'log', None) else None)
print("error message:", getattr(rc, 'error_message', None))

if rc.return_code == 0:
    cadet.load()
    sol = cadet.root.output.solution.unit_001.solution_outlet
    print("outlet shape:", np.array(sol).shape)
    print("max outlet:", np.max(sol))
