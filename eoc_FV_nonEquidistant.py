"""
EOC benchmark: axial FV reconstruction on pure-bulk transport (LUMPED_RATE_MODEL_WITHOUT_PORES).

Tested reconstruction variants:
  - WENO  order=2, equidistant
  - WENO  order=2, non-equidistant (GRID_FACES)
  - WENO  order=3, equidistant
  - WENO  order=3, non-equidistant (GRID_FACES)
  - Koren, equidistant
  - Koren, non-equidistant (GRID_FACES)

Model: COL_LENGTH=1.0, VELOCITY=0.01, COL_DISPERSION=1e-4, TOTAL_POROSITY=1.0
Reference solution: N_REF cells (finest grid).
Error metric: time-integrated L1 error at the column outlet.
EOC: log2(e_{N/2} / e_N)

Requires cadet-cli from feature/nonEqFVaxial branch (GRID_FACES support).

Setup (one-time):
    mkdir -p /Users/yuvj/CADET/build/src/bin
    ln -sf /Users/yuvj/CADET/build/src/cadet-cli/cadet-cli /Users/yuvj/CADET/build/src/bin/
    ln -sf /Users/yuvj/CADET/build/src/tools/createLWE     /Users/yuvj/CADET/build/src/bin/

Usage:
    /Users/yuvj/CADET/build/venv/bin/python3 eoc_FV_nonEquidistant.py
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

from src.benchmark_models.setting_axial_transport import axial_transport_model as model_config


# =========================================================================
# CONFIGURATION
# =========================================================================

CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

N_CELLS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]   # refinement levels
N_REF   = N_CELLS[-1] * 2                  # fine-grid reference

T_END     = 200.0
SOL_TIMES = np.linspace(0.0, T_END, 401)

AXIAL_GEOMETRY = (0.0, 1.0)  # [0, COL_LENGTH]

# =========================================================================
# GRID HELPERS
# =========================================================================

import numpy as np

def grid_equidistant(x0, x1, n):
    return  np.linspace(x0, x1, n + 1)

def grid_square(x0, x1, n):
    
    x = np.zeros(n+1)
    
    for i in range(0,n+1):
        
        x[i] = (i/(n+1))**2 * (x1-x0)
        
    return x

def grid_sinusoidal_perturbation(x0, x1, n, alpha=0.3):
    """
    Generate grid faces on [x0, x1] with a sinusoidal perturbation
    applied to interior nodes.

    Parameters
    ----------
    x0, x1 : float
        Domain boundaries.
    n : int
        Number of cells (n+1 faces returned).
    alpha : float, optional
        Perturbation strength relative to uniform spacing.

    Returns
    -------
    faces : ndarray
        Array of length n+1 containing face locations.
    """
    faces = np.linspace(x0, x1, n + 1)
    h = (x1 - x0) / n

    i = np.arange(1, n)
    faces[i] += alpha * h * np.sin(2.0 * np.pi * i / n)

    return faces


def grid_tanh_mapping(x0, x1, n, alpha=3.0):
    """
    Generate smoothly stretched grid faces using a tanh mapping.

    Clusters nodes symmetrically near the boundaries.

    Parameters
    ----------
    x0, x1 : float
        Domain boundaries.
    n : int
        Number of cells (n+1 faces returned).
    alpha : float, optional
        Stretching strength (alpha=0 -> uniform grid).

    Returns
    -------
    faces : ndarray
        Array of length n+1 containing face locations.
    """
    xi = np.linspace(0.0, 1.0, n + 1)

    stretched = (
        np.tanh(alpha * (xi - 0.5)) + np.tanh(alpha / 2.0)
    ) / (2.0 * np.tanh(alpha / 2.0))

    faces = x0 + (x1 - x0) * stretched
    return faces


def grid_left_cos_cluster(x0, x1, N, p=1.0):
    """
    Generate grid faces clustered toward the left boundary using
    a cosine-based stretching with tunable clustering.

    Parameters
    ----------
    x0, x1 : float
        Domain boundaries.
    N : int
        Number of cells (N+1 faces returned).
    p : float, optional
        Clustering exponent (>1 for stronger left clustering).

    Returns
    -------
    faces : ndarray
        Array of length N+1 containing face locations.
    """
    xi = np.linspace(0.0, 1.0, N + 1)
    stretched = 1.0 - np.cos(0.5 * np.pi * xi**p)
    faces = x0 + (x1 - x0) * stretched
    return faces

# =========================================================================
# MODEL FACTORY
# =========================================================================

def axial_lrm_noBnd(ncol, reconstruction, weno_order=None, grid_faces=None):
    """Axial LRM-without-pores, 1 component, no binding (pure bulk transport)."""
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
    col.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    col.ncomp = 1
    col.npartype = 1
    col.particle_type_000.has_film_diffusion = 0
    col.particle_type_000.has_pore_diffusion = 0
    col.particle_type_000.has_surface_diffusion = 0
    col.particle_type_000.nbound = [0]
    col.col_length = 1.0
    col.col_porosity = 1.0
    col.total_porosity = 1.0
    col.col_dispersion = [1e-4]
    col.velocity = 0.01
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
        raise RuntimeError(f"Simulation '{label}' failed: {ret.error_message}")
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

def eoc_study(label, reconstruction, weno_order, nonequid, grid_stretching,
              tmpdir, ref_sol=None,
              ):
    
    x0, x1 = AXIAL_GEOMETRY
    make_faces = grid_stretching if nonequid else grid_equidistant

    if ref_sol is None:
        faces_ref = make_faces(x0, x1, N_REF) if nonequid else None
        ref_sol = run_sim(model_config(N_REF, reconstruction, weno_order, faces_ref),
                          tmpdir, f"{label}_ref")

    errors = []
    for n in N_CELLS:
        faces = make_faces(x0, x1, n) if nonequid else None
        sol = run_sim(model_config(n, reconstruction, weno_order, faces),
                      tmpdir, f"{label}_N{n}")
        errors.append(l1_error(sol, ref_sol))

    return {'errors': errors, 'eocs': compute_eoc(errors)}

# =========================================================================
# TABLE PRINTER
# =========================================================================

def print_table(title, result):
    w = 32
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
    ("axWENO2_eq",    "WENO (order=2), equidistant",     'WENO',  2,    False),
    ("axWENO2_noneq", "WENO (order=2), non-equidistant", 'WENO',  2,    True ),
    ("axWENO3_eq",    "WENO (order=3), equidistant",     'WENO',  3,    False),
    ("axWENO3_noneq", "WENO (order=3), non-equidistant", 'WENO',  3,    True ),
    ("axKOREN_eq",    "Koren, equidistant",              'KOREN', None, False),
    ("axKOREN_noneq", "Koren, non-equidistant",          'KOREN', None, True ),
]

# =========================================================================
# MAIN
# =========================================================================


def main2():
    
    tmpdir = Path.cwd() / "output" / "eoc"
    
    nCells = 128
    
    weno3Eq = run_sim(
        model_config(
            nCells, reconstruction="WENO", weno_order=3, grid_faces=None),
        tmpdir, "WENO3Eq_ref"
        )
    
    # define full interval and starting interval which is more refined for non eq. grid
    xEnd = 1.0
    xFine = 1.0/100
    # Use ~ ten times more cells in fine interval
    h = xEnd / nCells
    nFineEqCells = int(np.ceil(xFine / h))
    nFineNonEqCells = nFineEqCells * 10
    nCoarseNonEqCells = nCells - nFineNonEqCells
    
    fistCells = np.linspace(0.0, xFine, nFineNonEqCells + 1)
    coarseCells = np.linspace(xFine, xEnd, nCoarseNonEqCells + 1)
    grid_faces = np.concatenate((fistCells, coarseCells[1:]))
    
    weno3nonEq = run_sim(
        model_config(
            128, reconstruction="WENO", weno_order=3, grid_faces=grid_faces),
        tmpdir, "WENO3Eq_ref"
        )
    
    plt.figure()
    plt.plot(SOL_TIMES, weno3Eq, label='WENO3 Eq')
    plt.plot(SOL_TIMES, weno3nonEq, label='WENO3 nonEq', linestyle='dashed')
    plt.xlabel("time (s)")
    plt.ylabel("concentration (mol/L)")
    plt.title("Comparison Eq. vs non-eq. grid")
    plt.legend()
    plt.show()
    
    refSol = run_sim(
        model_config(
            nCells*2, reconstruction="WENO", weno_order=3, grid_faces=None),
        tmpdir, "WENO3Eq_ref"
        )
    
    print("Eq. error: ", np.max(abs(refSol - weno3Eq)))
    print("Non-Eq. error: ", np.max(abs(refSol - weno3nonEq)))

def main():
    
    N = 50
    alpha = 3.0
    x0 = 0.0
    x1 = 1.0
    
    Cadet.cadet_path = CADET_PATH

    print("=" * 60)
    print("Axial FV — EOC benchmark")
    print(f"Refinement levels: {N_CELLS}   Reference: N={N_REF}")
    print("=" * 60)

    # with tempfile.TemporaryDirectory(prefix="cadet_eoc_") as tmpdir:
        
    tmpdir = Path.cwd() / "output" / "eoc"
        
    ref_sol = run_sim(model_config(
        N_REF, reconstruction="WENO", weno_order=3, grid_faces=None),
                      tmpdir, "WENO3Eq_ref")
    
    # grid_function = partial(grid_sinusoidal_perturbation, alpha=alpha)
    grid_function = partial(grid_tanh_mapping, alpha=alpha)
    # grid_function = partial(grid_left_cos_cluster, p=1.0)
    
    # Note: grid might get extremely small, e.g. first h~7.35×10−8, which crashes the simulation due to the huge gradients
    faces = grid_function(0.0, 1.0, N_CELLS[-1])
    dx = np.diff(faces)
    print("min dx =", dx.min(), "max dx =", dx.max())
    print("ratio =", dx.max()/dx.min())
    
    for label, description, recon, wo, nonequid in TESTS:
        
        result = eoc_study(
            label, recon, wo, nonequid,
            grid_function,
            tmpdir,
            ref_sol=ref_sol
            )
        print_table(f"Axial FV with {description}", result)

    print("\nDone.")
        
    x_left_cos_cluster_grid = grid_left_cos_cluster(x0, x1, N, p=1.0)
    x_sinusoidal = grid_sinusoidal_perturbation(x0, x1, N, alpha)
    x_tanh_mapping = grid_tanh_mapping(x0, x1, N, alpha)
    x_square = grid_square(x0, x1, N)
    
    comp_grid = np.linspace(x0,x1,N+1)
    
    plt.figure()
    plt.plot(comp_grid, x_left_cos_cluster_grid, marker='o', label='left cos cluster')
    plt.plot(comp_grid, x_sinusoidal, marker='o', label='sinusoidal')
    plt.plot(comp_grid, x_tanh_mapping, marker='o', label='tan')
    plt.plot(comp_grid, x_square, marker='o', label='square')
    plt.xlabel("Computational coordinate ξ")
    plt.ylabel("Physical coordinate x(ξ)")
    plt.title("Different grid stretchings")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()