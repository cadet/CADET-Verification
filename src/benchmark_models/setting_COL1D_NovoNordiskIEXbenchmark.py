# -*- coding: utf-8 -*-
"""

This script defines an ion exchange chromatography problem with six components (plus salt),
using parameter values and operating conditions from an in-house Novo Nordisk A/S chromatography
workflow, as reported in Meyer et al., 2026, Computers and Chemical Engineering,
"ChromOps.jl: High-order simulation and discrete forward sensitivity analysis for chromatography models".

"""

import numpy as np
from addict import Dict


def get_binding_configuration(is_kinetic:bool=False, par_porosity:float=0.66):
    """
    SMA parameters mapped from the paper's driving-force form (Eq. 4) onto CADET's
    Brooks & Cramer mass-action SMA, transforming *input parameters only*.

    The paper states the solid-phase balance (Eq. 2) in terms of q_i, the adsorbed
    concentration per pore-liquid volume, whereas CADET's solid-phase state c^s_i is
    per solid volume. Eliminating q_i from the paper's equations via the constant
    factor q_i = alpha * c^s_i, alpha = (1 - eps_p)/eps_p, and dividing Eq. 4 by
    alpha yields (with Q := Lambda/alpha - sum_j (nu_j + sigma_j) c^s_j)

        d(c^s_i)/dt = (ka_bar/alpha) c^p_i
                      - (ka_bar/(keq_i alpha^nu_i)) (c^p_0 / Q)^nu_i c^s_i.

    CADET's SMA with reference concentrations refq = refc0 = Lambda/alpha reads

        d(c^s_i)/dt = ka_i c^p_i (Q/refq)^nu_i - kd_i c^s_i (c^p_0/refc0)^nu_i.

    Matching the two term by term requires ka_i, kd_i ~ (refq/Q)^nu_i, i.e. an exact
    match needs state-dependent rate "constants" -- the two kinetic laws genuinely
    differ. However, the ratio is state-independent:

        ka_i / kd_i = keq_i * alpha^(nu_i - 1)      (exact),

    so the paper's equilibrium isotherm is reproduced exactly for any Q, via

        ka_i = ka_bar / alpha,
        kd_i = ka_bar / (keq_i * alpha^nu_i).

    The kinetic-form difference is handled by using rapid-equilibrium binding
    (is_kinetic = 0): the paper's forward rate is ka_bar = 10/s at *any* loading,
    ~70x faster than film transfer (0.139/s), so its solid phase is effectively in
    local equilibrium with the pore liquid. CADET's kinetic mass-action form cannot
    mimic that with constant rates: its forward rate carries the factor
    (Q/refq)^nu_i, which collapses by many orders of magnitude at moderate loading
    (nu ~ 20), artificially freezing adsorption during the load phase and letting
    protein break through in a nonphysical early spike. Rapid-equilibrium binding
    imposes the exactly-mapped isotherm instead, consistent with the paper's
    near-equilibrium kinetics.

    The reference concentrations keep all parameter values well-scaled; without them
    ka, kd would be ~1e-60 due to the nu-th powers of absolute concentrations.
    """
    alpha = (1.0 - par_porosity) / par_porosity

    lambda_paper = 324.7  # ionic capacity, Table 3; 324.7 mol/m^3 = 0.3247 mol/L
    sma_lambda = lambda_paper / alpha

    nu = np.array([22.0, 22.0, 21.0, 20.0, 10.0, 23.0])
    keq = np.array([1.0e5, 1.0e5, 3.0e4, 5.0e2, 5.0, 1.0e6])
    ka_bar = 10.0  # effective adsorption-rate coefficient, Table 4

    ka = np.full(6, ka_bar / alpha)
    kd = ka_bar / (keq * alpha ** nu)

    return {
        'is_kinetic': 1 if is_kinetic else 0,
        'sma_ka': np.concatenate(([0.0], ka)),
        'sma_kd': np.concatenate(([0.0], kd)),
        'sma_lambda': sma_lambda,
        'sma_refq': sma_lambda,
        'sma_refc0': sma_lambda,
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
    # Table 3's particle porosity eps_p = 0.66 is used literally, as the pore volume fraction
    # CADET's PAR_POROSITY expects. The paper's pore-phase mass balance (eq. 2) is
    #   d(c_p,i)/dt + d(q_i)/dt = kMT,i*(c_i - c_p,i)         [coefficient 1 on d(q_i)/dt]
    # while CADET's HOMOGENEOUS_PARTICLE (LRMP) pore balance is
    #   d(c_p,i)/dt + (1-eps_p)/eps_p * d(c^s_i)/dt = 3/(eps_p*par_radius) * film_diffusion * (c_i - c_p,i).
    # The paper's q_i (adsorbed amount per pore-liquid volume) and CADET's c^s_i (per solid
    # volume) are related by q_i = (1-eps_p)/eps_p * c^s_i, which turns eq. (2) into CADET's
    # pore balance exactly. This substitution only rescales *input parameters* (eps_p, the
    # film diffusion coefficient below, and the SMA lambda/ka/kd in
    # get_binding_configuration()); no state variable is transformed -- CADET's own c^s_i then
    # automatically equals q_i * eps_p/(1-eps_p).
    par_porosity = 0.66
    column.particle_type_000.par_porosity = par_porosity
    # we compute film diffusion coefficient from the lumped parameter in the source, which is 1.39/s and 0.139/s for salt and proteins respectively.
    # The paper's bulk balance (eq. 1) uses the coefficient (1-eps_c)/eps_c * eps_p * kMT,i, while CADET's HOMOGENEOUS_PARTICLE
    # (LRMP) model uses (1-eps_c)/eps_c * (3/par_radius) * film_diffusion. Equating the two gives
    # kMT,i = (3/par_radius) * film_diffusion / eps_p, i.e. film_diffusion = kMT,i * par_radius * eps_p / 3.
    # (par_porosity above is used here too, so the bulk-equation match holds for whichever eps_p we use.)
    fd_salt = 1.39 * column.particle_type_000.par_radius * column.particle_type_000.par_porosity / 3.0
    fd_protein = 0.139 * column.particle_type_000.par_radius * column.particle_type_000.par_porosity / 3.0
    column.particle_type_000.film_diffusion = [fd_salt, fd_protein, fd_protein, fd_protein, fd_protein, fd_protein, fd_protein]
    
    column.particle_type_000.adsorption_model = 'STERIC_MASS_ACTION'
    column.particle_type_000.adsorption = get_binding_configuration(is_kinetic=False, par_porosity=par_porosity)
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
sim.root = get_model(spatial_method_bulk=0, axNElem=256)
sim.filename = "NovoNordiskIEXbenchmark.h5"
sim.save()
return_data = sim.run_simulation()
sim.save()

if return_data.return_code != 0:
    raise RuntimeError(f"Simulation failed with return code {return_data.return_code} and message: {return_data.error_message}\n and LOG:\n {return_data.log}")


# --- Comparison against the chromatogram extracted from the paper's Figure 1 ---
# Data extracted from the figure PNG by color matching (see extract_fig1_data.py in the
# repository root): per-component CSVs with columns time_s, OD_AU_per_cm.
# Note: points are sparse where curves overlap at the zero baseline (only the topmost-drawn
# curve owns those pixels); missing time points there mean ~0.
import os
from matplotlib import pyplot as plt

times = sim.root.output.solution.solution_times
outlet = sim.root.output.solution.unit_000.solution_outlet
w = 1.167e4  # AU L mol^-1 cm^-1

extracted_dir = "chromops_fig1_extracted"
component_names = ["A", "B", "C", "D", "E", "F"]
colors = plt.cm.tab10(np.arange(7))

for i, name in enumerate(component_names):
    plt.plot(times, (w / 1000.0) * outlet[:, i + 1], color=colors[i + 1], label=f"{name} CADET")
    ext = np.genfromtxt(os.path.join(extracted_dir, f"{name}.csv"), delimiter=",", skip_header=1)
    plt.plot(ext[:, 0], ext[:, 1], ".", color=colors[i + 1], ms=2.5, alpha=0.7)
plt.plot([], [], "k.", ms=2.5, label="paper plt. 1 (extracted)")
plt.ylabel("OD (AU/cm)")
plt.title("CADET (reparameterized, solid) vs. paper Figure 1 (extracted, dots)")
plt.legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.savefig("comparison_cadet_vs_paper_equations.png", dpi=150)
plt.show()


# --- Error metrics: CADET vs. paper Figure 1 (proteins only) ---
# Metrics are computed on the extracted time points of each component (the extracted
# curves are sparse near the zero baseline, so this restricts the comparison to where
# figure data actually exists). CADET is linearly interpolated onto those points.
#   peak_err   : relative peak height error
#   dt_peak    : peak time difference (CADET - figure), s
#   area_err   : relative error of the integrated OD signal (mass proxy)
#   NRMSE      : RMSE normalized by the figure's peak height
#   dt*        : uniform time shift applied to CADET that minimizes the RMSE
#   NRMSE(dt*) : shift-corrected NRMSE, i.e. remaining shape error after removing
#                the systematic time offset between simulation and figure
shifts = np.arange(-200.0, 800.0, 5.0)
print("\n--- Error metrics vs. extracted Figure 1 data (proteins only) ---")
print(f"{'comp':>4} {'peak_err':>9} {'dt_peak':>8} {'area_err':>9} {'NRMSE':>7} {'dt*':>6} {'NRMSE(dt*)':>11}")
nrmse_all, nrmse_shift_all = [], []
for i, name in enumerate(component_names):
    ext = np.genfromtxt(os.path.join(extracted_dir, f"{name}.csv"), delimiter=",", skip_header=1)
    t_ref, od_ref = ext[:, 0], ext[:, 1]
    od_sim_full = (w / 1000.0) * outlet[:, i + 1]
    od_sim = np.interp(t_ref, times, od_sim_full)

    peak_err = (od_sim_full.max() - od_ref.max()) / od_ref.max()
    dt_peak = times[od_sim_full.argmax()] - t_ref[od_ref.argmax()]
    area_sim = np.trapz(od_sim_full, times)
    area_ref = np.trapz(od_ref, t_ref)
    area_err = (area_sim - area_ref) / area_ref
    nrmse = np.sqrt(np.mean((od_sim - od_ref) ** 2)) / od_ref.max()

    # best uniform time shift of the CADET signal (positive = CADET shifted later)
    rmse_s = [np.sqrt(np.mean((np.interp(t_ref - s, times, od_sim_full) - od_ref) ** 2))
              for s in shifts]
    j = int(np.argmin(rmse_s))
    dt_star, nrmse_star = shifts[j], rmse_s[j] / od_ref.max()

    nrmse_all.append(nrmse)
    nrmse_shift_all.append(nrmse_star)
    print(f"{name:>4} {peak_err:>8.1%} {dt_peak:>7.0f}s {area_err:>8.1%} "
          f"{nrmse:>7.3f} {dt_star:>5.0f}s {nrmse_star:>11.3f}")

print(f"\nmean NRMSE           : {np.mean(nrmse_all):.3f}")
print(f"mean NRMSE (shifted) : {np.mean(nrmse_shift_all):.3f}")
print("A consistent dt* across components indicates a systematic time offset between")
print("the paper's figure and the stated inlet program, rather than a model mismatch.")
