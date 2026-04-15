# -*- coding: utf-8 -*-
"""

This script implements a salt-gradient elution of mAb1 
as shown in Briskot et al. (2021), J. Chromatogr. A 1654, 462439, Fig. 2b.
"""

import numpy as np
import matplotlib.pyplot as plt

from cadet import Cadet


e_ch  = 1.602176634e-19
N_A   = 6.02214076e23
k_B   = 1.380649e-23
eps_0 = 8.8541878128e-12
 
#Table 1 parameters
T       = 298.15       # K
eps_r   = 78.3
Gamma_L = 1.47e-6      # mol/m^2
zeta_L  = 0.0
pK_L    = 2.3

Vc_mL   = 22.3         # mL
Dax     = 10.0e-7      # m^2/s
dp      = 65e-6        # m
eps_v   = 0.34
eps_p   = 0.39

Vc   = Vc_mL * 1e-6    # m^3
Lc   = 0.215           # m
A_cs = Vc / Lc          # m^2

a_i     = 5.5e-9       # m
A_s_i   = 0.18e9       # 1/m
keff_i  = 0.51e-6      # m/s
Zi_ref  = 111.79
Z1_i    = 0.0
Z2_i    = 0.0
Zlat_i  = 61.87
log10_Delta_ref = -4.04
Delta_1_i = 0.0

pH_op   = 5.0
pH_ref  = pH_op

Im_low  = 69.97        # mol/m^3
Q       = 4.0333e-8    # m^3/s
c_feed  = 0.106        # mol/m^3

# Section durations
t_load = 74.15 * 60    # s
t_wash = 27.7 * 60     # s
t_grad = 88.1 * 60     # s
t_end  = 20 * 60       # s

# Component indices
IDX_PH      = 0
IDX_SALT    = 1
IDX_PROTEIN = 2
NCOMP       = 3

di = 9.0e-12           # m^2/s



def get_model(cadet_path, output_path, run_simulation, plot_result):

    model = Cadet()
    model.install_path = cadet_path

    model.root.input.model.nunits = 3

    # ── Inlet ──────────────────────────────────────────────────
    inlet = model.root.input.model.unit_000
    inlet.unit_type = 'INLET'
    inlet.ncomp = NCOMP
    inlet.inlet_type = 'PIECEWISE_CUBIC_POLY'

    pH_val = 10.0**pH_op

    # Section 0 — Loading
    sec0 = inlet.sec_000
    sec0.const_coeff = [pH_val, Im_low, c_feed]
    sec0.lin_coeff   = [0.0, 0.0, 0.0]
    sec0.quad_coeff  = [0.0, 0.0, 0.0]
    sec0.cube_coeff  = [0.0, 0.0, 0.0]

    # Section 1 — Wash
    sec1 = inlet.sec_001
    sec1.const_coeff = [pH_val, Im_low, 0.0]
    sec1.lin_coeff   = [0.0, 0.0, 0.0]
    sec1.quad_coeff  = [0.0, 0.0, 0.0]
    sec1.cube_coeff  = [0.0, 0.0, 0.0]

    # Section 2 — Salt gradient
    dIm_dt = 0.053
    sec2 = inlet.sec_002
    sec2.const_coeff = [pH_val, Im_low, 0.0]
    sec2.lin_coeff   = [0.0, dIm_dt, 0.0]
    sec2.quad_coeff  = [0.0, 0.0, 0.0]
    sec2.cube_coeff  = [0.0, 0.0, 0.0]

    # Section 3 — High salt
    sec3 = inlet.sec_003
    sec3.const_coeff = [pH_val, 0.335 * 1000, 0.0]
    sec3.lin_coeff   = [0.0, 0.0, 0.0]
    sec3.quad_coeff  = [0.0, 0.0, 0.0]
    sec3.cube_coeff  = [0.0, 0.0, 0.0]

    col = model.root.input.model.unit_001
    col.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES'
    col.ncomp = NCOMP
    col.col_length = Lc
    col.cross_section_area = A_cs
    col.col_dispersion = Dax
    col.col_porosity = eps_v
    col.npartype = 1
    col.init_c = [pH_val, Im_low, 0.0]
    col.use_analytic_jacobian = 1

    par = col.particle_type_000
    par.par_radius = 3.25e-5
    par.par_porosity = eps_p
    par.film_diffusion = [2.0e-7] * 3
    par.pore_diffusion = [9.0e-12, 9.0e-12, di]
    par.nbound = [0, 0, 1]
    par.init_cs = [0.0]

    # Discretization
    disc = col.discretization
    disc.ncol = 100
    disc.use_analytic_jacobian = 0
    disc.spatial_method = 'FV'
    disc.reconstruction = 'WENO'
    disc.weno.boundary_model = 0
    disc.weno.weno_eps = 1e-10
    disc.weno.weno_order = 3
    disc.gs_type = 1
    disc.max_krylov = 0
    disc.max_restarts = 10
    disc.schur_safety = 1e-8

    # CPA binding
    par.adsorption_model = 'COLLOIDAL_PARTICLE_ADSORPTION'
    ads = par.adsorption
    ads.is_kinetic = 1
    ads.cpa_is_kinetic = 1
    ads.cpa_proton_idx = IDX_PH
    ads.cpa_salt_idx   = IDX_SALT
    ads.cpa_temperature       = T
    ads.cpa_ionic_strength    = Im_low
    ads.cpa_permittivity      = eps_r
    ads.cpa_surface_density   = Gamma_L
    ads.cpa_charge_full_ligand = zeta_L
    ads.cpa_pk_ligand         = pK_L
    ads.cpa_surface_area     = [0.0, 0.0, A_s_i]
    ads.cpa_protein_radius   = [0.0, 0.0, a_i]
    ads.cpa_comp_lat_charge  = [0.0, 0.0, Zlat_i]
    ads.cpa_comp_charge_ref  = [0.0, 0.0, Zi_ref]
    ads.cpa_comp_charge_lin  = [0.0, 0.0, Z1_i]
    ads.cpa_comp_charge_quad = [0.0, 0.0, Z2_i]
    ads.cpa_ph_ref           = pH_ref
    ads.cpa_delta_ref        = [0.0, 0.0, log10_Delta_ref]
    ads.cpa_delta_lin        = [0.0, 0.0, Delta_1_i]
    ads.cpa_diffusion_coeff  = [0.0, 0.0, di]
    ads.maxiter = 1000

    outlet = model.root.input.model.unit_002
    outlet.unit_type = 'OUTLET'
    outlet.ncomp = NCOMP

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Sections
    t0 = 0.0
    t1 = t0 + t_load
    t2 = t1 + t_wash
    t3 = t2 + t_grad
    t4 = t3 + t_end

    model.root.input.solver.sections.nsec = 3
    model.root.input.solver.sections.section_times = [t0, t1, t2, t3, t4]
    model.root.input.solver.sections.section_continuity = [0, 0, 0]

    model.root.input.solver.user_solution_times = np.linspace(t0, t4, 2000)

    model.root.input.solver.time_integrator.abstol = 1e-8
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-8
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1
    model.root.input.solver.nthreads = 1

    # Return
    ret = model.root.input['return']
    ret.split_components_data = 1
    ret.split_ports_data = 0
    ret.unit_001.write_solution_bulk = 0
    ret.unit_001.write_solution_inlet = 1
    ret.unit_001.write_solution_outlet = 1
    ret.unit_001.write_solution_solid = 0
    ret.unit_002.write_solution_bulk = 0
    ret.unit_002.write_solution_inlet = 1
    ret.unit_002.write_solution_outlet = 0

    model.filename = output_path + '/CPA_LRMP_gradient_mAb1_benchmark2.h5'
    model.save()

    if run_simulation:

        return_data = model.run_simulation()

        if not return_data.return_code == 0:
            raise Exception(
                f"simulation failed with {return_data.error_message}"
                f"\n and LOG:\n {return_data.log}")

        if plot_result:

            model.load_from_file()

            times  = model.root.output.solution.solution_times / 60
            c_prot = model.root.output.solution.unit_001.solution_outlet_comp_002
            c_salt = model.root.output.solution.unit_001.solution_outlet_comp_001 / 1000

            fig, ax1 = plt.subplots(figsize=(8, 5))

            ax1.plot(times, c_prot, '-', color='black',
                     linewidth=1.5, label='UV 280 nm (sim)')
            ax1.set_xlabel('min', fontsize=12)
            ax1.set_ylabel('c', fontsize=12, color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.set_xlim(0, 200)
            ax1.set_ylim(0, 0.20)

            ax2 = ax1.twinx()
            ax2.plot(times, c_salt, '--', color='gray',
                     linewidth=1.0, label='Conductivity (sim)')
            ax2.set_ylabel(r'Im', fontsize=12, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.set_ylim(0, 0.4)

            ax1.set_title('Briskot Fig. 2b — mAb1 on SP Sepharose FF (CPA)',
                          fontsize=12, fontweight='bold')

            fig.tight_layout()
            fig.savefig(
                output_path + '/CPA_LRMP_gradient_mAb1_benchmark2.png',
                dpi=200, bbox_inches='tight')
            plt.show()
            plt.close()

    return model
