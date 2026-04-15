"""
Reproduces Fig. 2 from Briskot et al. (2021),
J. Chromatogr. A 1654, 462439.

mAb1 on Poros 50 HS — salt-gradient elution at constant pH.
Uses the CADET LRMP + CPA binding model.
All parameters are from Table 1 in Briskot et al. (2021) (mAb1).
"""

from cadet import Cadet
import numpy as np
import matplotlib.pyplot as plt
import os

CADET_INSTALL_PATH = "/Users/berger/fzj/cadet/CADET-Core/install_debug"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

e_ch  = 1.602176634e-19
N_A   = 6.02214076e23
k_B   = 1.380649e-23
eps_0 = 8.8541878128e-12

# Table 1
T       = 298.15       
eps_r   = 78.3
Gamma_L = 1.47e-6 
zeta_L  = 0.0
pK_L    = 2.3

Vc_mL   = 22.3         # mL
Dax     = 1.0e-7      # m²/s
dp      = 65e-6       #  m
eps_v   = 0.34     
eps_p   = 0.39

Vc   = Vc_mL * 1e-6    # m^3
Lc   = 0.215    # m
A_cs = Vc / Lc         # m^2

a_i     = 5.5e-9       # m
A_s_i   = 0.18e9       # 1/m
keff_i  = 0.51e-6      #m/s
k_kin   = 5.78e7
Zi_ref  = 111.79
Z1_i    = 0.0          
Z2_i    = 0.0
Zlat_i  = 61.87
log10_Delta_ref = -4.04
Delta_1_i = 0.0        

pH_ref  = 5.0

Im_low    = 69.97
Q         = 4.0333e-8

c_feed    = 0.106

#durations points
t_load  = 74.15 * 60
t_wash  = 27.7 * 60
t_grad  = 88.1 * 60
t_end =  20 * 60

# Component indices
IDX_PH      = 0
IDX_PROTEIN = 2

NCOMP = 4

def getProtonConcentration(pH):
    exp = -pH
    return (10**exp)/1e-3


def build_model():
    model = Cadet(CADET_INSTALL_PATH)
    root = model.root

    root.input.model.nunits = 3   # inlet, column, outlet

    inlet = root.input.model.unit_000
    inlet.unit_type = 'INLET'
    inlet.ncomp = NCOMP
    inlet.inlet_type = 'PIECEWISE_CUBIC_POLY'

    #  Sections: 1=load  2=wash  3=gradient  4=strip
    pc = getProtonConcentration(pH_ref)

    # Section 
    sec1 = inlet.sec_000
    sec1.const_coeff = [pc, Im_low, Im_low, c_feed]
    sec1.lin_coeff   = [0.0, 0.0, 0.0, 0.0]
    sec1.quad_coeff  = [0.0, 0.0, 0.0, 0.0]
    sec1.cube_coeff  = [0.0, 0.0, 0.0, 0.0]

    # Section 2
    sec2 = inlet.sec_001
    sec2.const_coeff = [pc, Im_low,Im_low, 0.0]
    sec2.lin_coeff   = [0.0, 0.0, 0.0,0.0]
    sec2.quad_coeff  = [0.0, 0.0, 0.0,0.0]
    sec2.cube_coeff  = [0.0, 0.0, 0.0,0.0]

    # Section 3
    dIm_dt = 0.053
    sec3 = inlet.sec_002
    sec3.const_coeff = [pc, Im_low,Im_low, 0.0]
    sec3.lin_coeff   = [0.0, dIm_dt, dIm_dt,0.0]
    sec3.quad_coeff  = [0.0, 0.0, 0.0,0.0]
    sec3.cube_coeff  = [0.0, 0.0, 0.0,0.0]

    sec4 = inlet.sec_003
    sec4.const_coeff = [pc, 0.345 * 1000, 0.345 * 1000, 0.0]
    sec4.lin_coeff   = [0.0, 0.0,0.0, 0.0]
    sec4.quad_coeff  = [0.0, 0.0,0.0, 0.0]
    sec4.cube_coeff  = [0.0, 0.0, 0.0,0.0]


    col = root.input.model.unit_001
    col.unit_type = 'COLUMN_MODEL_1D'
    col.ncomp = NCOMP
    col.bed_length = Lc
    col.cross_section_area = A_cs
    col.col_dispersion = Dax
    col.col_porosity = eps_v
    col.npartype = 1
    col.init_c = [pc, Im_low, Im_low, 0.0]
    col.use_analytic_jacobian = 0

    col.geometry = "AXIAL_FLOW_CYLINDER"
    col.forward_flow = 1

    # Particle type 0
    di = 9.0e-12
    par = col.particle_type_000
    par.par_radius = 3.25e-5
    par.par_porosity = eps_p
    par.has_film_diffusion = 1
    par.film_diffusion = [2.0e-7,2.0e-7,2.0e-7,keff_i]
    par.has_pore_diffusion = 0
    par.nbound = [0, 0, 0, 1]   # pH(0), na+, cl- , protein(1)
    par.init_cs = [0.0]      # initial solid-phase for protein
    par.geom = "SPHERE"


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


    par.adsorption_model = 'COLLOIDAL_PARTICLE_ADSORPTION'
    ads = par.adsorption
    ads.is_kinetic = 1
    ads.cpa_is_kinetic = 1
    ads.cpa_proton_idx = IDX_PH

    ads.cpa_temperature         = T
    ads.cpa_ionic_strength      = Im_low
    ads.cpa_permittivity        = eps_r
    ads.cpa_ligand_density      = Gamma_L
    ads.cpa_ligand_charge_full  = zeta_L
    ads.cpa_ligand_pk           = pK_L


    # Per-component arrays: [pH, salt, protein]
    ads.cpa_specific_surface_area     = [0.0, 0.0,0.0, A_s_i]
    ads.cpa_radius                    = [0.0, 0.0,0.0, a_i]
    ads.cpa_lat_charge                = [0.0, 0.0,0.0, Zlat_i]
    ads.cpa_effective_charge_coef     = [[0.0, 0.0,0.0, Zi_ref], [0.0, 0.0, 0.0,Z1_i], [0.0, 0.0,0.0, Z2_i]]
    ads.cpa_ph_ref                    = pH_ref
    ads.cpa_delta_ref                 = [0.0, 0.0,0.0, 10**(log10_Delta_ref)]
    ads.cpa_delta_lin                 = [0.0, 0.0,0.0, Delta_1_i]
    ads.cpa_kkin                      = [0.0, 0.0,0.0, k_kin]
    ads.cpa_ionic_valence               = [0,-1, -1,0]
    ads.cpa_maxiter                   = 1000

    outlet = root.input.model.unit_002
    outlet.unit_type = 'OUTLET'
    outlet.ncomp = NCOMP

    root.input.model.connections.nswitches = 1
    root.input.model.connections.switch_000.section = 0
    root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,   # inlet → column (volumetric flow)
        1, 2, -1, -1, Q,   # column → outlet
    ]

    #  Solver
    root.input.model.solver.gs_type = 1
    root.input.model.solver.max_krylov = 0
    root.input.model.solver.max_restarts = 10
    root.input.model.solver.schur_safety = 1e-8

    t0 = 0.0
    t1 = t0 + t_load
    t2 = t1 + t_wash
    t3 = t2 + t_grad
    t4 = t3 + t_end

    root.input.solver.sections.nsec = 3
    root.input.solver.sections.section_times = [t0, t1, t2, t3,t4]
    root.input.solver.sections.section_continuity = [0, 0,0]

    root.input.solver.user_solution_times = np.linspace(t0, t4, 2000)

    root.input.solver.time_integrator.abstol = 1e-8
    root.input.solver.time_integrator.algtol = 1e-10
    root.input.solver.time_integrator.reltol = 1e-6
    root.input.solver.time_integrator.init_step_size = 1e-8
    root.input.solver.time_integrator.max_steps = 1000000
    root.input.solver.consistent_init_mode = 1
    root.input.solver.nthreads = 1

    # Return
    ret = root.input['return']
    ret.split_components_data = 1
    ret.split_ports_data = 0
    ret.unit_001.write_solution_bulk = 0
    ret.unit_001.write_solution_inlet = 1
    ret.unit_001.write_solution_outlet = 1
    ret.unit_001.write_solution_solid = 0
    ret.unit_002.write_solution_bulk = 0
    ret.unit_002.write_solution_inlet = 1
    ret.unit_002.write_solution_outlet = 0

    return model


def run_and_plot():
    model = build_model()
    h5_path = os.path.join(RESULTS_DIR, 'cpa_fig2b.h5')
    model.filename = h5_path
    model.save()

    print("Running CADET simulation ...")
    msg = model.run()
    if msg.return_code != 0:
        print(f"ERROR: {msg.error_message}")
        return
    print("  done.")

    model.load()

    # Extract outlet solution
    times  = model.root.output.solution.solution_times / 60 
    c_prot = model.root.output.solution.unit_001.solution_outlet_comp_003
    c_na = model.root.output.solution.unit_001.solution_outlet_comp_001
    c_cl = model.root.output.solution.unit_001.solution_outlet_comp_002

    IM = 0.5 * (c_na + c_cl) / 1000


    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_uv   = 'black'
    color_cond = 'gray'

    ax1.plot(times, c_prot, '-', color=color_uv, linewidth=1.5, label='UV 280 nm (sim)')
    ax1.set_xlabel('min', fontsize=12)
    ax1.set_ylabel('c [mol/m^3] ', fontsize=12, color=color_uv)
    ax1.tick_params(axis='y', labelcolor=color_uv)
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 0.20)

    ax2 = ax1.twinx()
    ax2.plot(times , IM, '--', color=color_cond, linewidth=1.0, label='Conductivity (sim)')
    ax2.set_ylabel(r'Im [M]', fontsize=12, color=color_cond)
    ax2.tick_params(axis='y', labelcolor=color_cond)
    ax2.set_ylim(0, 0.4)

    ax1.set_title('Briskot Fig. 2b', fontsize=12, fontweight='bold')

    fig.tight_layout()
    outpath = os.path.join(RESULTS_DIR, 'cpa_fig2b.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {outpath}")


if __name__ == '__main__':
    run_and_plot()
