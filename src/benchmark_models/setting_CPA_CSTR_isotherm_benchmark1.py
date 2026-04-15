# -*- coding: utf-8 -*-
"""

This script implements batch equilibrium CPA isotherms as shown in
Briskot et al. (2021), J. Chromatogr. A 1654, 462439, Fig. 6.
"""

import numpy as np
import matplotlib.pyplot as plt

from cadet import Cadet


e_ch  = 1.602176634e-19
N_A   = 6.02214076e23
k_B   = 1.380649e-23
eps_0 = 8.8541878128e-12

T       = 298.15
eps_r   = 78.3
A_s_i   = 0.24e9        # 1/m
a_i     = 5.5e-9        # m
Gamma_L = 1.86e-6       # mol/m^2
zeta_L  = 0.0
pK_L    = 2.3

# CPA protein parameters
pH_ref          = 6.25
Zi_ref          = 63.32
Z1_i            = -26.23
Z2_i            = 4.07
log10_Delta_ref = -3.61
Delta_1_i       = -94.11

# Experimental conditions
PH_VALUES  = [5.5, 6.0, 6.5, 7.0]
SIM_TIME   = 1e13
di         = 1e-15

PH_IM_MAP = {
    5.5: [20, 95, 145, 220],
    6.0: [20, 70, 95, 120],
    6.5: [20, 45, 70, 95],
    7.0: [20, 45, 95],
}

IM_MARKERS = {20: 'o', 45: 's', 70: 'D', 95: 'v',
              120: '<', 145: '>', 220: '^'}

EXPERIMENTAL_DATA = {
    5.5: {
        20:  {'c': [1.743e-2, 3.848e-2, 5.561e-2, 8.868e-2, 9.259e-2],
              'q': [3.345e+0, 3.467e+0, 3.178e+0, 3.558e+0, 3.315e+0]},
        95:  {'c': [7.214e-3, 2.345e-2, 4.449e-2, 6.523e-2, 6.673e-2,
                    8.086e-2, 8.176e-2, 1.007e-1, 1.025e-1],
              'q': [1.685e+0, 1.959e+0, 1.959e+0, 2.096e+0, 2.036e+0,
                    2.081e+0, 2.036e+0, 2.127e+0, 2.005e+0]},
        145: {'c': [2.705e-2, 3.667e-2, 6.854e-2, 7.275e-2, 8.417e-2,
                    1.314e-1, 1.32e-1, 1.007e-1],
              'q': [4.975e-1, 5.888e-1, 7.868e-1, 1.198e+0, 8.02e-1,
                    1.015e+0, 1e+0, 5.431e-1]},
        220: {'c': [1.285e-2, 1.942e-2, 5.289e-2, 8.546e-2, 1.312e-1,
                    1.377e-1, 1.425e-1],
              'q': [7.107e-2, 7.107e-2, 7.107e-2, 7.107e-2, 1.32e-1,
                    1.32e-1, 1.624e-1]},
    },
    6.0: {
        20:  {'c': [4.633e-3, 3.041e-2, 5.444e-2, 5.82e-2],
              'q': [3.837e+0, 3.658e+0, 3.97e+0, 3.822e+0]},
        70:  {'c': [4.923e-3, 1.911e-2, 3.388e-2, 5.618e-2, 6.197e-2,
                    7.905e-2, 8.108e-2],
              'q': [1.564e+0, 1.817e+0, 2.069e+0, 2.158e+0, 2.366e+0,
                    2.589e+0, 2.455e+0]},
        95:  {'c': [2.317e-2, 2.635e-2, 4.517e-2, 5.097e-2, 6.515e-2,
                    6.921e-2, 8.6e-2, 8.832e-2, 8.919e-2, 1.181e-1],
              'q': [1.163e+0, 1.178e+0, 1.282e+0, 1.223e+0, 1.446e+0,
                    1.416e+0, 1.416e+0, 1.609e+0, 1.49e+0, 1.55e+0]},
        120: {'c': [3.185e-2, 3.793e-2, 6.892e-2, 7.181e-2, 8.369e-2,
                    8.977e-2, 1.219e-1, 1.251e-1, 9.295e-2],
              'q': [1.98e-1, 2.277e-1, 4.802e-1, 4.505e-1, 4.505e-1,
                    5.396e-1, 9.703e-1, 8.515e-1, 2.871e-1]},
    },
    6.5: {
        20:  {'c': [4.5e-3, 1.26e-2, 1.92e-2, 5.28e-2, 5.46e-2],
              'q': [2.316e+0, 3.571e+0, 3.357e+0, 3.969e+0, 3.847e+0]},
        45:  {'c': [6e-4, 4.5e-3, 1.05e-2, 3.63e-2, 6.36e-2, 6.72e-2,
                    7.86e-2],
              'q': [9.541e-1, 1.719e+0, 1.888e+0, 2.286e+0, 2.439e+0,
                    2.224e+0, 2.393e+0]},
        70:  {'c': [1.26e-2, 2.28e-2, 4.11e-2, 6.09e-2, 7.83e-2, 7.98e-2,
                    9.96e-2, 1.035e-1],
              'q': [6.633e-1, 8.929e-1, 1.23e+0, 1.352e+0, 1.413e+0,
                    1.306e+0, 1.49e+0, 1.413e+0]},
        95:  {'c': [6.63e-2, 7.26e-2, 7.5e-2, 7.86e-2, 8.28e-2, 8.97e-2,
                    9.18e-2, 1.014e-1, 1.293e-1, 1.344e-1, 1.404e-1,
                    1.437e-1],
              'q': [4.184e-1, 5.255e-1, 6.173e-1, 6.173e-1, 4.184e-1,
                    4.949e-1, 4.184e-1, 4.184e-1, 6.02e-1, 8.469e-1,
                    4.949e-1, 5.561e-1]},
    },
    7.0: {
        20:  {'c': [8.806e-4, 2.935e-4, 1.937e-2, 5.46e-2, 5.607e-2,
                    6.164e-2],
              'q': [1.03e+0, 2.275e+0, 3.04e+0, 3.265e+0, 3.19e+0,
                    3.64e+0]},
        45:  {'c': [1.82e-2, 2.73e-2, 4.99e-2, 6.869e-2, 8.513e-2,
                    8.718e-2, 1.115e-1],
              'q': [5.5e-1, 7.9e-1, 1.12e+0, 1.24e+0, 1.39e+0, 1.24e+0,
                    1.495e+0]},
        95:  {'c': [7.867e-2, 8.307e-2, 1.221e-1, 1.391e-1],
              'q': [8.5e-2, 2.65e-1, 2.35e-1, 3.4e-1]},
    },
}


def _create_cpa_cstr(cadet_path, c_protein, pH, Im):

    model = Cadet()
    model.install_path = cadet_path

    model.root.input.model.nunits = 1

    unit = model.root.input.model.unit_000
    unit.unit_type = 'CSTR'
    unit.ncomp = 2
    unit.nbound = [0, 1]
    unit.init_liquid_volume = 1e6
    unit.const_solid_volume = 0.0
    unit.init_c = [10.0**pH, c_protein]
    unit.init_cs = [0.0]
    unit.use_analytic_jacobian = 1

    unit.adsorption_model = 'COLLOIDAL_PARTICLE_ADSORPTION'
    ads = unit.adsorption
    ads.is_kinetic = 1
    ads.cpa_proton_idx = 0
    ads.cpa_temperature = T
    ads.cpa_ionic_strength = Im
    ads.cpa_permittivity = eps_r
    ads.cpa_surface_density = Gamma_L
    ads.cpa_charge_full_ligand = zeta_L
    ads.cpa_pk_ligand = pK_L
    ads.cpa_surface_area   = [0.0, A_s_i]
    ads.cpa_protein_radius = [0.0, a_i]
    ads.cpa_maxiter = 1000
    ads.cpa_comp_lat_charge  = [0.0, 0.0]
    ads.cpa_comp_charge_ref  = [0.0, Zi_ref]
    ads.cpa_comp_charge_lin  = [0.0, Z1_i]
    ads.cpa_comp_charge_quad = [0.0, Z2_i]
    ads.cpa_ph_ref           = pH_ref
    ads.cpa_delta_ref        = [0.0, log10_Delta_ref]
    ads.cpa_delta_lin        = [0.0, Delta_1_i]
    ads.cpa_diffusion_coeff  = [0.0, di]

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = []

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    model.root.input.solver.user_solution_times = np.linspace(0, SIM_TIME, 200)
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, SIM_TIME]
    model.root.input.solver.sections.section_continuity = []

    model.root.input.solver.time_integrator.abstol = 1e-10
    model.root.input.solver.time_integrator.algtol = 1e-12
    model.root.input.solver.time_integrator.reltol = 1e-8
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1

    ret = model.root.input['return']
    ret.unit_000.split_components_data = 0
    ret.unit_000.split_ports_data = 0
    ret.unit_000.write_solution_bulk = 1
    ret.unit_000.write_solution_inlet = 0
    ret.unit_000.write_solution_outlet = 1
    ret.unit_000.write_solution_solid = 1

    return model


def _run_batch_equilibrium(cadet_path, output_path, c_protein, pH, Im):

    model = _create_cpa_cstr(cadet_path, c_protein, pH, Im)
    model.filename = output_path + '/cpa_cstr_isotherm_temp.h5'
    model.save()

    return_data = model.run_simulation()
    if not return_data.return_code == 0:
        raise Exception(
            f"simulation failed with {return_data.error_message}"
            f"\n and LOG:\n {return_data.log}")

    model.load_from_file()
    q_eq = model.root.output.solution.unit_000.solution_solid[-1, 0]
    c_eq = model.root.output.solution.unit_000.solution_bulk[-1, 1]
    return c_eq, q_eq



def get_model(cadet_path, output_path, run_simulation, plot_result):

    c_range = np.linspace(0.0, 0.15, 30)
    results = {}

    if run_simulation:
        for pH in PH_VALUES:
            results[pH] = {}
            im_list = PH_IM_MAP[pH]

            print(f"\n--- CPA isotherm pH {pH} ---")
            for Im in im_list:
                print(f"  Im={Im:3d} mM: ", end="")
                q_vals = []
                for c0 in c_range:
                    try:
                        _, q_eq = _run_batch_equilibrium(
                            cadet_path, output_path, c0, pH, Im)
                        q_vals.append(q_eq)
                    except Exception:
                        q_vals.append(np.nan)
                results[pH][Im] = np.array(q_vals)
                print(f"q_max={np.nanmax(results[pH][Im]):.3f}")

    if plot_result and run_simulation:

        fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                                 sharex=True, sharey=True)

        for idx, pH in enumerate(PH_VALUES):
            ax = axes[idx // 2][idx % 2]
            im_list = PH_IM_MAP[pH]

            for Im in im_list:
                if Im in EXPERIMENTAL_DATA[pH]:
                    exp = EXPERIMENTAL_DATA[pH][Im]
                    ax.scatter(exp['c'], exp['q'],
                               marker=IM_MARKERS[Im], s=50,
                               facecolors='none', edgecolors='black',
                               linewidths=1.5,
                               label=f'{Im} mM', zorder=10)

                ax.plot(c_range, results[pH][Im], '-', color='black',
                        linewidth=1.0, alpha=0.7, zorder=5)

            ax.set_xlim(0, 0.15)
            ax.set_ylim(0, 4.5)
            ax.set_title(f'pH {pH}', fontsize=12, fontweight='bold')
            ax.grid()
            if idx // 2 == 1:
                ax.set_xlabel(r'$c_{0,i}$ [mol m$^{-3}$]', fontsize=11)
            if idx % 2 == 0:
                ax.set_ylabel(r'$q_{v,i}$ [mol m$^{-3}$]', fontsize=11)
            if idx == 0:
                ax.legend(fontsize=8, loc='upper right', framealpha=0.9,
                          title='Ionic strength', title_fontsize=9)

        fig.suptitle('CPA isotherms — mAb on Poros XS (Briskot Fig. 6)',
                     fontsize=12, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(output_path + '/CPA_CSTR_isotherm_benchmark1.png',
                    dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()
