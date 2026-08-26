# -*- coding: utf-8 -*-
"""

This script defines an ion exchange chromatography problem with six components (plus salt),
using parameter values and operating conditions from an in-house Novo Nordisk A/S chromatography
workflow, as reported in Meyer et al., 2026, Computers and Chemical Engineering,
"ChromOps.jl: High-order simulation and discrete forward sensitivity analysis for chromatography models".

"""

import numpy as np
from addict import Dict


def get_binding_configuration(is_kinetic:bool=True):
    return {
        'is_kinetic': 1 if is_kinetic else 0,
        'sma_ka': [ 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        'sma_kd': [ 0.0, 1.0e-4, 1.0e-4, 1.0 / 3.0e3, 2.0e-2, 2.0, 1.0e-5],
        'sma_lambda': 324.7, # 324.7 mol / m^3 = 0.3247 mol / L
        'sma_nu': [ 0.0, 22, 22, 21, 20, 10, 23 ],
        'sma_sigma': [ 0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 ]
        }


def get_column_geometry_configuration(geometry: str):

    # for all geometries:
    # velocity = 7.51e-5 m/s = 7.51e-3 cm/s
    # bulk porosity is 0.37
    # bed length is 10mm = 0.01m
    # col radius at inlet was not given but we assume it to be 0.0035m and compute the adequat flow rate from there to match the velocity
    col_radius = 0.0035
    # flow rate Q = velocity / (cross_section * col_porosity)
    axial_flow_cross_section_area = np.pi * (col_radius ** 2)

    if geometry == 'AXIAL_FLOW_CYLINDER':
        return {
            # A = v * \pi * r^2 * \varepsilon
            'cross_section_area': axial_flow_cross_section_area,
            'col_length': 0.01,
            'bed_length': 0.01,
        }
    elif geometry == 'RADIAL_FLOW_CYLINDER_SHELL': # note: not considered in the original source
        return {
            # A = 2 * pi * \rho * L^b -> \rho = A / 2.0 / pi / L^b
            'cross_section_area': axial_flow_cross_section_area,
            'col_length': 0.00025, # height
            'col_radius_outer': axial_flow_cross_section_area / 2.0 / np.pi / 0.00025,
            'col_radius_inner': axial_flow_cross_section_area / 2.0 / np.pi / 0.00025 - 0.01,
        }
    elif geometry == 'AXIAL_FLOW_FRUSTUM': # note: not considered in the original source
        return {
            'cross_section_area': axial_flow_cross_section_area,
            'col_radius_large_end': col_radius,
            'col_radius_small_end': col_radius * 0.75,
            'col_radius_outer': col_radius,
            'col_radius_inner': col_radius * 0.75,
            'col_length': 0.01,
        }
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def get_model(
        spatial_method_bulk, axNElem,
        column_geometry='AXIAL_FLOW_CYLINDER',
        **kwargs):
        
    model = Dict()
    
    model.input.model.nunits = 2
    
    column = Dict()
    if column_geometry == 'AXIAL_FLOW_CYLINDER':
        column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    elif column_geometry == 'RADIAL_FLOW_CYLINDER_SHELL':
        column.UNIT_TYPE = 'RADIAL_COLUMN_MODEL_1D'
    elif column_geometry == 'AXIAL_FLOW_FRUSTUM':
        column.UNIT_TYPE = 'FRUSTUM_COLUMN_MODEL_1D'
    else:
        raise ValueError(f"Unknown column geometry: {column_geometry}")
    column.geometry = column_geometry
    column.update(get_column_geometry_configuration(column_geometry))

    column.npartype = 1
    column.col_porosity = 0.37
    
    column.ncomp = 7
    column.init_c = [40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    column.col_dispersion = 1.5e-9 # 1.5e-5 cm^2 / s = 1.5e-9 m^2 / s
    column.forward_flow = 1

    # Spatial discretization of interstitial / bulk volume
    if spatial_method_bulk > 0:
        column.discretization.SPATIAL_METHOD = 'DG'
        column.discretization.POLYNOMIAL_INTEGRATION_TYPE = kwargs.get('POLYNOMIAL_INTEGRATION_TYPE', 0)
        column.discretization.POLYDEG = spatial_method_bulk
        column.discretization.NELEM = axNElem
    else:
        column.discretization.SPATIAL_METHOD = 'FV'
        column.discretization.NCOL = axNElem
        column.discretization.RECONSTRUCTION = 'WENO'
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = 3
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10
        column.discretization.SCHUR_SAFETY = 1.0e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = 1

    # particle_type = 'HOMOGENEOUS_PARTICLE'
    column.particle_type_000.has_film_diffusion = 1
    column.particle_type_000.par_geom = 'SPHERE'
    # note: the radius is only assumed and not from the source, which lumps radius and film diffusion coefficient into a single parameter
    column.particle_type_000.par_radius = 4.5e-05
    column.particle_type_000.par_coreradius = 0.0
    column.particle_type_000.par_porosity = 0.66
    # we compute film diffusion coefficient from the lumped parameter in the source, which is 1.39/s and 0.139/s for salt and proteins respectively.
    # The paper's bulk balance (eq. 1) uses the coefficient (1-eps_c)/eps_c * eps_p * kMT,i, while CADET's HOMOGENEOUS_PARTICLE
    # (LRMP) model uses (1-eps_c)/eps_c * (3/par_radius) * film_diffusion. Equating the two gives
    # kMT,i = (3/par_radius) * film_diffusion / eps_p, i.e. film_diffusion = kMT,i * par_radius * eps_p / 3.
    fd_salt = 1.39 * column.particle_type_000.par_radius * column.particle_type_000.par_porosity / 3.0
    fd_protein = 0.139 * column.particle_type_000.par_radius * column.particle_type_000.par_porosity / 3.0
    column.particle_type_000.film_diffusion = [fd_salt, fd_protein, fd_protein, fd_protein, fd_protein, fd_protein, fd_protein]
    
    column.particle_type_000.adsorption_model = 'STERIC_MASS_ACTION'
    column.particle_type_000.adsorption = get_binding_configuration(is_kinetic=True)
    column.particle_type_000.nbound = [1, 1, 1, 1, 1, 1, 1]
    
    init_salt = column.particle_type_000.adsorption['sma_lambda']
    column.particle_type_000.init_cs = [init_salt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    column.particle_type_000.init_cp = [40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    model.input.model.unit_000 = column

    # Flow sheet
    # velocity = 7.51e-5 m/s = 7.51e-3 cm/s
    interstitial_velocity = 7.51e-5
    flowRate = interstitial_velocity * column.cross_section_area * column.col_porosity
    # print(f"Flow rate: {flowRate} m^3/s = {flowRate * 1e6} L/s = {flowRate * 1e6 * 60.0} L/min")
    # print(f"Interstitial velocity: {flowRate / column.cross_section_area / column.col_porosity} m/s")
    # print(f"Column cross section area: {column.cross_section_area} m^2")
    model.input.model.connections.connections_include_ports = 1
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        1.0, 0.0, -1.0, -1.0, -1.0, -1.0, flowRate
    ]
    model.input.model.connections.switch_000.section = 0
    
    # Inlet / Feed unit
    model.input.model.unit_001.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_001.ncomp = 7
    deltaT = [360.0, 360, 900, 720, 7200, 720]
    section_times = [0.0]
    salt_start = [40.0, 40.0, 40.0, 240.0, 240.0, 1040.0]
    salt_end = [40.0, 40.0, 240.0, 240.0, 640.0, 1040.0]

    for i in range(len(deltaT)):

        section_times.append(section_times[-1] + deltaT[i])

        salt_const_coeff = salt_start[i]
        salt_lin_coeff = (salt_end[i] - salt_start[i]) / deltaT[i]

        if i == 0:
            model.input.model.unit_001[f'sec_{i:03d}'].const_coeff = [salt_const_coeff, 1.381, 0.03046, 0.3087, 0.06092, 0.1665, 0.08326]
        else:
            model.input.model.unit_001[f'sec_{i:03d}'].const_coeff = [salt_const_coeff, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.input.model.unit_001[f'sec_{i:03d}'].lin_coeff = [salt_lin_coeff, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.input.model.unit_001[f'sec_{i:03d}'].cube_coeff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.input.model.unit_001[f'sec_{i:03d}'].quad_coeff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    print(f"Section times: {section_times}")

    model.input.model.unit_001.UNIT_TYPE = 'INLET'

    # Global system solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    # Time integration / solver
    model.input.solver.nthreads = 1
    model.input.solver.consistent_init_mode = 5
    model.input.solver.sections.nsec = len(section_times) - 1
    model.input.solver.sections.section_continuity = [ 0, 0, 0, 0, 0, 0 ]
    model.input.solver.sections.section_times = section_times
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_reftol', 1e-6)
    model.input.solver.time_integrator.ALGTOL = kwargs['idas_reftol'] * 100 if 'idas_reftol' in kwargs else 1e-5
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.time_integrator.RELTOL = kwargs['idas_reftol'] * 100 if 'idas_reftol' in kwargs else 1e-5
    
    # Return data
    model.input.solver.user_solution_times = np.linspace(0, section_times[-1], int(section_times[-1]) + 1)
    model.input['return'].split_components_data = 0
    model.input['return'].unit_000.write_coordinates = kwargs.get('write_solution_bulk', False) or kwargs.get('write_solution_particle', False)
    model.input['return'].unit_000.write_solution_bulk =  kwargs.get('write_solution_bulk', False)
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].unit_000.write_solution_particle = kwargs.get('write_solution_particle', False)
    model.input['return'].unit_000.write_solution_solid = kwargs.get('write_solution_particle', False)
    model.input['return'].write_solution_times = 1
    
    return model


from cadet import Cadet

sim = Cadet()
sim.install_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
sim.root = get_model(spatial_method_bulk=0, axNElem=64)
sim.filename = "NovoNordiskIEXbenchmark.h5"
sim.save()
return_data = sim.run_simulation()
sim.save()

if return_data.return_code != 0:
    raise RuntimeError(f"Simulation failed with return code {return_data.return_code} and message: {return_data.error_message}\n and LOG:\n {return_data.log}")

from matplotlib import pyplot as plt

# Extinction coefficient from Section 4.2
w = 1.167e4  # AU L mol^-1 cm^-1

times = sim.root.output.solution.solution_times
outlet = sim.root.output.solution.unit_000.solution_outlet

# Salt concentration
plt.figure()
plt.plot(times, outlet[:, 0], label='salt')
plt.xlabel('Time (s)')
plt.ylabel('Salt concentration (mol/m³)')
plt.legend()
plt.savefig("salt_concentration.png")

# Protein concentrations
plt.figure()
for i in range(1, 7):
    plt.plot(
        times,
        outlet[:, i] / 1000.0,
        label=f'protein {i}'
    )

plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.legend()
plt.savefig("protein_concentrations.png")

# Protein optical density — as in the paper
plt.figure()
for i in range(1, 7):
    od = (w / 1000.0) * outlet[:, i]
    plt.plot(times, od, label=f'protein {i}')

plt.xlabel('Time (s)')
plt.ylabel('OD (AU/cm)')
plt.legend()
plt.savefig("protein_optical_density.png")


import pandas as pd

df = pd.read_csv(
    "proteinA.csv",
    sep=";",
    decimal=",",
    header=None,
    names=["time", "concentrationOD"]
)

time = df["time"].to_numpy()
concentrationOD = df["concentrationOD"].to_numpy()

plt.figure()
plt.plot(time, concentrationOD, label='protein A ChromOps')
plt.plot(times, (w / 1000.0) * outlet[:, 1], label='protein A CADET')
plt.xlabel('Time (s)')
plt.ylabel('OD (AU/cm)')
plt.legend()
plt.savefig("proteinA_experimental_data.png")
plt.show()




# Feed concentrations from the paper, mol/m^3
cfeed = np.array([
    1.381,
    0.03046,
    0.3087,
    0.06092,
    0.1665,
    0.08326
])

# Integrated outlet concentration
outlet_area = np.array([
    np.trapz(outlet[:, i], times)
    for i in range(1, 7)
])

# Integrated inlet concentration
inlet_area = cfeed * 360.0

print("Protein    inlet area    outlet area    outlet/inlet")
for i in range(6):
    print(
        f"{i+1:7d}    "
        f"{inlet_area[i]:11.4f}    "
        f"{outlet_area[i]:12.4f}    "
        f"{outlet_area[i]/inlet_area[i]:12.4f}"
    )

import numpy as np

cadet_od = (w / 1000.0) * outlet[:, 1]

print("\n--- Protein A comparison ---")

print("CADET:")
print("  max OD =", cadet_od.max())
print("  area   =", np.trapz(cadet_od, times))

print("\nCSV:")
print("  max OD =", concentrationOD.max())
print("  area   =", np.trapz(concentrationOD, time))

print("\nRatio CSV / CADET:")
print("  max ratio  =", concentrationOD.max() / cadet_od.max())
print("  area ratio =", (
    np.trapz(concentrationOD, time) /
    np.trapz(cadet_od, times)
))