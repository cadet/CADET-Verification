# -*- coding: utf-8 -*-
"""

This script implements a salt-gradient elution of mAb1
as shown in Briskot et al. (2021), J. Chromatogr. A 1654, 462439, Fig. 2a.
"""

import numpy as np
import matplotlib.pyplot as plt

from cadet import Cadet


e_ch  = 1.602176634e-19
N_A   = 6.02214076e23
k_B   = 1.380649e-23
eps_0 = 8.8541878128e-12

T       = 298.15       # K
eps_r   = 78.3
Gamma_L = 2.89e-6      # mol/m^2
zeta_L  = 0.0
pK_L    = 2.3

# Column
Vc_mL   = 0.98         # mL
Lc_mm   = 50.0         # mm
Dax     = 0.06e-6      # m^2/s
dp      = 50e-6        # m
eps_v   = 0.53
eps_p   = 0.54

Vc   = Vc_mL * 1e-6    # m^3
Lc   = Lc_mm * 1e-3    # m
A_cs = Vc / Lc          # m^2

a_i     = 5.5e-9       # m
A_s_i   = 0.22e9       # 1/m
keff_i  = 1.01e-6      # m/s
Zi_ref  = 80.45
Z1_i    = 0.0
Z2_i    = 0.0
Zlat_i  = 19.07
log10_Delta_ref = -1.90
Delta_1_i = 0.0

pH_op   = 5.0
pH_ref  = pH_op

# Ionic strength
Im_buffer = 20.0                 # mol/m^3
Im_low    = Im_buffer
Im_high   = Im_buffer + 450.0   # mol/m^3
Im_strip  = Im_buffer + 1000.0  # mol/m^3

# Flow rate
Q_mLmin   = 0.08
Q         = Q_mLmin / 60.0 * 1e-6  # m^3/s

# Protein feed
c_feed_gL = 12.7
c_feed    = c_feed_gL / 150e3 * 1e3  # mol/m^3

# Section lengths in CV
CV_load  = 7.1
CV_wash  = 5.0
CV_grad  = 30.63
CV_strip = 5.0

t_CV   = Vc / Q
t_load = CV_load * t_CV
t_wash = CV_wash * t_CV
t_grad = CV_grad * t_CV
t_strip = CV_strip * t_CV

# Component indices
IDX_PH      = 0
IDX_SALT    = 1
IDX_PROTEIN = 2
NCOMP       = 3

di = 1.51957802009209e-16
other_diff = 1e-3



def get_model(cadet_path, output_path, run_simulation, plot_result):

    model = Cadet()
    model.install_path = cadet_path

    model.root.input.model.nunits = 3

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
    dIm_dt = (Im_high - Im_low) / t_grad
    sec2 = inlet.sec_002
    sec2.const_coeff = [pH_val, Im_low, 0.0]
    sec2.lin_coeff   = [0.0, dIm_dt, 0.0]
    sec2.quad_coeff  = [0.0, 0.0, 0.0]
    sec2.cube_coeff  = [0.0, 0.0, 0.0]

    # Section 3 — Strip
    sec3 = inlet.sec_003
    sec3.const_coeff = [pH_val, Im_strip, 0.0]
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
    col.use_analytic_jacobian = 0

    par = col.particle_type_000
    par.par_radius = dp / 2
    par.par_porosity = eps_p
    par.film_diffusion = [other_diff, other_diff, keff_i]
    par.pore_diffusion = [other_diff, other_diff, di]
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
    t4 = t3 + t_strip

    model.root.input.solver.sections.nsec = 5
    model.root.input.solver.sections.section_times = [t0, t1, t2, t3, t4]
    model.root.input.solver.sections.section_continuity = [0, 0, 0, 0]

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

    model.filename = output_path + '/CPA_LRMP_gradient_mAb1_benchmark1.h5'
    model.save()

    if run_simulation:

        return_data = model.run_simulation()

        if not return_data.return_code == 0:
            raise Exception(
                f"simulation failed with {return_data.error_message}"
                f"\n and LOG:\n {return_data.log}")

        if plot_result:

            model.load_from_file()

            times  = model.root.output.solution.solution_times
            c_prot = model.root.output.solution.unit_001.solution_outlet_comp_002
            c_salt = model.root.output.solution.unit_001.solution_inlet_comp_001

            vol_mL = times * Q * 1e6
            conductivity = c_salt * 0.1
            c_gL = c_prot * 150e3 * 1e-3
            absorbance = 1.4 * 0.04 * c_gL

            fig, ax1 = plt.subplots(figsize=(8, 5))

            ax1.plot(vol_mL, absorbance, '-', color='black',
                     linewidth=1.5, label='UV 280 nm (sim)')
            ax1.set_xlabel('volume [mL]', fontsize=12)
            ax1.set_ylabel('absorbance [AU]', fontsize=12, color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.set_xlim(0, 40)
            ax1.set_ylim(0, 0.6)

            ax2 = ax1.twinx()
            ax2.plot(vol_mL, conductivity, '--', color='gray',
                     linewidth=1.0, label='Conductivity (sim)')
            ax2.set_ylabel(r'conductivity [mS cm$^{-1}$]',
                           fontsize=12, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.set_ylim(0, 60)

            ax1.set_title('Briskot Fig. 2a — mAb1 on Poros 50 HS (CPA)',
                          fontsize=12, fontweight='bold')

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc='upper left', fontsize=9)

            fig.tight_layout()
            fig.savefig(
                output_path + '/CPA_LRMP_gradient_mAb1_benchmark1.png',
                dpi=200, bbox_inches='tight')
            plt.show()
            plt.close()

    return model
