"""
EOC benchmark: radial FV reconstruction on pure-bulk transport (RADIAL_LUMPED_RATE_MODEL_WITHOUT_PORES).

Tested reconstruction variants:
  - WENO  order=2, equidistant
  - WENO  order=2, non-equidistant (GRID_FACES)
  - WENO  order=3, equidistant
  - WENO  order=3, non-equidistant (GRID_FACES)
  - Koren, equidistant
  - Koren, non-equidistant (GRID_FACES)

Model: COL_RADIUS_INNER=0.01, COL_RADIUS_OUTER=0.1, VELOCITY_COEFF=5e-5,
       COL_DISPERSION=1e-6, TOTAL_POROSITY=1.0
Reference solution: N_REF cells (finest grid).
Error metric: time-integrated L1 error at the column outlet.
EOC: log2(e_{N/2} / e_N)

Requires cadet-cli from FV_Features branch (radial GRID_FACES support).

Setup (one-time):
    mkdir -p /Users/yuvj/CADET/build/src/bin
    ln -sf /Users/yuvj/CADET/build/src/cadet-cli/cadet-cli /Users/yuvj/CADET/build/src/bin/

Usage:
    /Users/yuvj/CADET/build/venv/bin/python3 eoc_radialFV_nonEquidistant.py
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from functools import partial
import tempfile
import numpy as np
from pathlib import Path
from addict import Dict
from cadet import Cadet

# =========================================================================
# CONFIGURATION
# =========================================================================

CADET_PATH = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\radialFV\aRELEASE"

N_CELLS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]   # refinement levels
N_REF   = N_CELLS[-1] * 2             # fine-grid reference

### UPWIND TEST
# WENO3Eq with 4096 cells accuracy: ~ 1.0312e-08
# N_CELLS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 16384*2, 16384*4]   # refinement levels
# N_REF   = 4096


T_END     = 200.0
SOL_TIMES = np.linspace(0.0, T_END, 401)

RADIAL_GEOMETRY = (0.01, 0.1)  # [COL_RADIUS_INNER, COL_RADIUS_OUTER]

# =========================================================================
# GRID HELPERS
# =========================================================================

def grid_equidistant(r0, r1, n):
    return  np.linspace(r0, r1, n + 1)

def grid_equivolume(r0, r1, n):
    """
    Returns radial faces of a cylindrical shell such that each annular cell has the same volume.
    """
    # Compute equally spaced points in r^2
    r2_faces = np.linspace(r0**2, r1**2, n + 1)
    # Convert back to r
    return np.sqrt(r2_faces)

def grid_sinusoidal(r0, r1, n, alpha=0.3):
    """Sinusoidal perturbation of interior faces: face[i] += alpha*h*sin(2πi/n)."""
    faces = np.linspace(r0, r1, n + 1)
    h = (r1 - r0) / n
    for i in range(1, n):
        faces[i] += alpha * h * np.sin(2.0 * np.pi * i / n)
    return faces

# =========================================================================
# MODEL FACTORY
# =========================================================================

def radial_lrm_noBnd(ncol, reconstruction, weno_order=None, grid_faces=None):
    """Radial LRM-without-pores, 1 component, no binding (pure bulk transport)."""
    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, 1.0,
        1.0, 2.0, -1.0, -1.0, 1.0,
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
    m.input.model.unit_000.sec_000.lin_coeff   = [0.0]
    m.input.model.unit_000.sec_000.quad_coeff  = [0.0]
    m.input.model.unit_000.sec_000.cube_coeff  = [0.0]
    m.input.model.unit_000.sec_001.const_coeff = [0.0]
    m.input.model.unit_000.sec_001.lin_coeff   = [0.0]
    m.input.model.unit_000.sec_001.quad_coeff  = [0.0]
    m.input.model.unit_000.sec_001.cube_coeff  = [0.0]

    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_LUMPED_RATE_MODEL_WITHOUT_PORES'
    col.ncomp = 1
    col.npartype = 1
    col.particle_type_000.has_film_diffusion = 0
    col.particle_type_000.has_pore_diffusion = 0
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.nbound = [0]
    col.col_radius_inner = RADIAL_GEOMETRY[0]
    col.col_radius_outer = RADIAL_GEOMETRY[1]
    col.total_porosity = 1.0
    col.col_dispersion = [1e-6]
    col.velocity_coeff = 5e-5
    col.init_c = [0.0]

    disc = col.discretization
    disc.ncol = ncol
    disc.spatial_method = 'FV'
    disc.use_analytic_jacobian = 1
    disc.reconstruction = reconstruction
    disc.weno.weno_order = weno_order if weno_order is not None else 3
    disc.weno.weno_eps = 1e-10
    disc.weno.boundary_model = 0
    disc.koren.koren_eps = 1e-10
    if grid_faces is not None:
        disc.grid_faces = grid_faces.tolist()

    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = 1

    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0

    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 2
    m.input.solver.sections.section_continuity = [0]
    m.input.solver.sections.section_times = [0.0, 10.0, T_END]
    m.input.solver.time_integrator.abstol = 1e-10
    m.input.solver.time_integrator.reltol = 1e-8
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-6
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = SOL_TIMES

    return m

# =========================================================================
# SIMULATION RUNNER
# =========================================================================

def run_sim(model_dict, tmpdir, label):
    sim = Cadet()
    sim.root = model_dict
    sim.filename = str(Path(tmpdir) / f"{label}.h5")
    sim.save()
    ret = sim.run_simulation()
    if ret.return_code != 0:
        raise RuntimeError(f"Simulation '{label}' failed: {ret.error_message}, {ret.log}")
    sim.load_from_file()
    return np.squeeze(np.array(sim.root.output.solution.unit_001.solution_outlet))

# =========================================================================
# ERROR & EOC
# =========================================================================

def l1_error(sol, ref):
    dt = SOL_TIMES[1] - SOL_TIMES[0]
    return np.trapezoid(np.abs(sol - ref), dx=dt) / (SOL_TIMES[-1] - SOL_TIMES[0])


def compute_eoc(errors):
    eocs = [float('nan')]
    for i in range(1, len(errors)):
        eocs.append(np.log2(errors[i - 1] / errors[i]) if errors[i] > 0 else float('nan'))
    return eocs

# =========================================================================
# EOC STUDY
# =========================================================================

def eoc_study(label, reconstruction, weno_order, nonequid, grid_stretching, ref_sol, tmpdir):
    
    r0, r1 = RADIAL_GEOMETRY

    make_faces = grid_stretching if nonequid else grid_equidistant

    if ref_sol is None:
        faces_ref = make_faces(r0, r1, N_REF) if nonequid else None
        ref_sol = run_sim(radial_lrm_noBnd(N_REF, reconstruction, weno_order, faces_ref),
                          tmpdir, f"{label}_ref")

    errors = []
    for n in N_CELLS:
        faces = make_faces(r0, r1, n) if nonequid else None
        sol = run_sim(radial_lrm_noBnd(n, reconstruction, weno_order, faces),
                      tmpdir, f"{label}_N{n}")
        errors.append(l1_error(sol, ref_sol))

    return {'errors': errors, 'eocs': compute_eoc(errors)}

# =========================================================================
# TABLE PRINTER
# =========================================================================

def print_table(title, result):
    w = 36
    print(f"\n{'='*w}")
    print(title)
    print(f"{'-'*w}")
    print(f"{'N':>6}  {'L1 error':>12}  {'EOC':>6}")
    print(f"{'-'*w}")
    for n, err, eoc in zip(N_CELLS, result['errors'], result['eocs']):
        eoc_str = f"{eoc:6.3f}" if not np.isnan(eoc) else "   ---"
        print(f"{n:>6}  {err:>12.4e}  {eoc_str}")
    print(f"{'='*w}")

# =========================================================================
# TEST CASES
# =========================================================================

TESTS = [
    # ("upwindEq", "Updwind, equidistant", 'WENO', 1, False ),
    # ("radWENO2_eq",    "WENO (order=2), equidistant",     'WENO',  2,    False),
    # ("radWENO2_noneq", "WENO (order=2), non-equidistant", 'WENO',  2,    True ),
    ("radWENO3_eq",    "WENO (order=3), equidistant",     'WENO',  3,    False),
    ("radWENO3_noneq", "WENO (order=3), non-equidistant", 'WENO',  3,    True ),
    # ("radKOREN_eq",    "Koren, equidistant",               'KOREN', None, False),
    # ("radKOREN_noneq", "Koren, non-equidistant",           'KOREN', None, True ),
]

# =========================================================================
# MAIN
# =========================================================================

def main():
    Cadet.cadet_path = CADET_PATH

    print("=" * 60)
    print("Radial FV — EOC benchmark")
    print(f"Domain: r in {RADIAL_GEOMETRY}")
    print(f"Refinement levels: {N_CELLS}   Reference: N={N_REF}")
    print("=" * 60)

    # with tempfile.TemporaryDirectory(prefix="cadet_radial_eoc_") as tmpdir:

    tmpdir = Path.cwd() / "output" / "eoc"
        
    ref_sol = run_sim(radial_lrm_noBnd(
        N_REF, reconstruction="WENO", weno_order=3, grid_faces=None),
                      tmpdir, "WENO3Eq_ref")
    
    for label, description, recon, wo, nonequid in TESTS:
        
        grid_stretching = grid_equivolume # grid_sinusoidal
        
        result = eoc_study(
            label, recon, wo, nonequid,
            grid_stretching=grid_stretching, ref_sol=ref_sol,
            tmpdir=tmpdir
            )
        
        print_table(f"Radial FV with {description}", result)

    print("\nDone.")


def mainPlot():
    
    tmpdir = Path.cwd() / "output" / "eoc"
    
    nCells = 128
    
    weno3Eq = run_sim(
        radial_lrm_noBnd(
            nCells, reconstruction="WENO", weno_order=3, grid_faces=None),
        tmpdir, "WENO3Eq_ref"
            )
    
    plt.figure()
    plt.plot(SOL_TIMES, weno3Eq, label='WENO3Eq')
    plt.xlabel("time")
    plt.ylabel("concentration")
    plt.title("Different grid stretchings")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()