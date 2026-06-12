import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

import numpy as np
from functools import partial
from numba import njit
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from cadet import Cadet

from src.utility import convergence
from src.benchmark_models import setting_Col2D_lin_1comp_benchmark1
import src.benchmark_models.helper_setup_2Dmodels as helper


# =============================
# configuration for eoc study
# =============================

output_path = Path.cwd().parent / "output" / "test_cadet-core"
cadet_path = convergence.get_cadet_path() # r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

snall_test = 0
n_jobs = -1

axP = 2
radP = axP

nRadialZones = 2

# note: last level will be taken as reference
axial_levels = [4, 8, 16, 32] if snall_test else [4, 8, 16, 32, 64]  # axial elements
radial_levels = [2, 4, 8, 16] if snall_test else [2, 4, 8, 16, 32] # [nRadialZones] * len(axial_levels)  # radial elements

def init_c(z, r):
    return np.linspace(1.0, 1.5, nRadialZones)

# L = 0.014
# R = 0.0035

# z = np.linspace(0, L, 300)
# r = np.linspace(0, R, 200)

# Z, RR = np.meshgrid(z, r, indexing="ij")

# C = init_c(Z, RR)

# plt.figure(figsize=(6, 4))
# plt.pcolormesh(Z, RR, C, shading="auto")
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar(label="init_c")
# plt.tight_layout()
# plt.show()

setting = {
    'npartype': 0,
    'nRadialZones': nRadialZones,
    'COL_POROSITY': np.linspace(0.3, 0.35, nRadialZones),
    'INIT_C': init_c,
    # 'COL_POROSITY': np.linspace(0.35, 0.9, nRadialZones),
    # Port 1 with reference bulk 64 rad 32
      # EOC 0->1: Linf=3.94, L1=4.16, L2=4.15
      # EOC 1->2: Linf=4.23, L1=3.25, L2=3.61
      # EOC 2->3: Linf=1.97, L1=1.81, L2=1.89
      # EOC 3->4: Linf=1.66, L1=1.68, L2=1.65
    'inlet_function': partial(helper.constInlet, const=1.0),
    'WRITE_SOLUTION_LAST': True,
    'WRITE_SOLUTION_BULK': True,
    'tEnd': 200, # 200 for outlet # 25 for bulk
    'USE_MODIFIED_NEWTON': True
}

# =============================
# HELPER FUNCTIONS
# =============================

def run_sim(axNElem, radNElem, **kwargs):

    kwargs.update({'name': f'run_ax{axNElem}_rad{radNElem}'})

    epsB = kwargs['COL_POROSITY']
    nElemPerZone = int(radNElem / nRadialZones)
    epsBVec = []
    for rad in range(nRadialZones):
        epsBVec.extend([epsB[rad]] * nElemPerZone)
    kwargs['COL_POROSITY'] = epsBVec

    model = setting_Col2D_lin_1comp_benchmark1.get_model(
        axMethod=axP, axNElem=axNElem,
        radMethod=radP, radNElem=radNElem,
        **kwargs)

    tEnd = kwargs.get('tEnd', 1500)

    model['input']['solver']['USER_SOLUTION_TIMES'] = np.linspace(0, tEnd, tEnd+1)
    model['input']['solver']['sections']['SECTION_TIMES'] = [0.0, 10.0, tEnd]

    sim = Cadet()
    sim.install_path = cadet_path
    sim.root = model
    sim.filename = str(output_path / f"sim_ax{axNElem}_rad{radNElem}.h5")
    sim.save()

    ret = sim.run_simulation()
    if ret.return_code != 0:
        raise RuntimeError(ret.error_message)

    sim.load_from_file()

    # extract solution
    axP_loc = sim.root.input.model.unit_000.discretization.ax_polydeg
    axZ = sim.root.input.model.unit_000.discretization.ax_nelem
    radP_loc = sim.root.input.model.unit_000.discretization.rad_polydeg
    radZ = sim.root.input.model.unit_000.discretization.rad_nelem

    bulkNPoints = (axP_loc + 1) * axZ * (radP_loc + 1) * radZ
    inletNPoints = (radP_loc + 1) * radZ

    # dimensionality like state: axial-radial-component major ordering
    bulk = np.array(sim.root.output.last_state_y)[inletNPoints:inletNPoints+bulkNPoints]

    ax_coords = sim.root.output.coordinates.unit_000.axial_coordinates
    rad_coords = sim.root.output.coordinates.unit_000.radial_coordinates

    if 'nRadialZones' in kwargs:
        outlets = []
    
        for rad in range(kwargs['nRadialZones']):
            
            outletIdx = 1 + rad + kwargs['nRadialZones']
            
            outlet = np.array(
                sim.root.output.solution[f'unit_{outletIdx:03d}'].solution_outlet
            )
    
            outlets.append(outlet)
    
        # shape: (time, nRadialZones)
        outlet = np.column_stack(outlets)
    
    else:
        outlet = np.array(
            sim.root.output.solution.unit_000.solution_outlet_port_000
        )
        
    times = np.array(sim.root.output.solution.solution_times)

    return bulk, ax_coords, rad_coords, outlet, times


def interpolate_2D_cartesian(vals, P1, P2, coords1, coords2,
                             outCoords1, outCoords2, outVals):
    """
    This function interpolates
        1) a 2D tensor-product nodal DG solution on Lobatto nodes to outCoords using barycentric interpolation,
        2) a 2D FV solution on cell centers to outCoords assuming piecewise constant values in each cell
    
    If an output coordinate coincides a node/cell center, the value at that node is taken
    If an output coordinate coincides with an element/cell interface, the average of the two interface values is taken
    """

    # check inputs
    if len(coords1) % (P1 + 1) != 0:
        raise ValueError("Length of coords1 must be divisible by P1 + 1")
    if len(coords2) % (P2 + 1) != 0:
        raise ValueError("Length of coords2 must be divisible by P2 + 1")
    if not np.all(np.diff(coords1) >= -1e-15): # include duplicate coordinate entries for interfaces
        raise ValueError("coords1 must be sorted in ascending order")
    if not np.all(np.diff(coords2) >= -1e-15): # include duplicate coordinate entries for interfaces
        raise ValueError("coords2 must be sorted in ascending order")
    if not np.all(np.diff(outCoords1) >= -1e-15): # include duplicate coordinate entries for interfaces
        raise ValueError("outCoords1 must be sorted in ascending order")
    if not np.all(np.diff(outCoords2) >= -1e-15): # include duplicate coordinate entries for interfaces
        raise ValueError("outCoords2 must be sorted in ascending order")
    if not (isinstance(vals, np.ndarray) and isinstance(outVals, np.ndarray)):
        raise ValueError("vals and outVals must be a numpy array")
    if vals.shape != (len(coords1), len(coords2)) or outVals.shape != (len(outCoords1), len(outCoords2)):
        raise ValueError(f"vals and outVals must have shape ({len(coords1)}, {len(coords2)}) and ({len(outCoords1)}, {len(outCoords2)})")
    if P1 < 0 or P2 < 0:
        raise ValueError("P must be >= 0")
    if outCoords1[0] < coords1[0] or outCoords1[-1] > coords1[-1]:
        raise ValueError("outCoords1 out of bounds")
    if np.any(coords1 < 0):
        raise ValueError("coords1 must be >= 0")
    if outCoords2[0] < coords2[0] or outCoords2[-1] > coords2[-1]:
        raise ValueError("outCoords2 out of bounds")
    if np.any(coords2 < 0):
        raise ValueError("coords2 must be >= 0")


    # allocations outside of kernel
    nodes1, _ = convergence.LGL_NodesWeights(P1)
    baryW1 = convergence.barycentric_weights(P1)

    nodes2, _ = convergence.LGL_NodesWeights(P2)
    baryW2 = convergence.barycentric_weights(P2)

    tmp_line = np.empty(max(len(coords1), len(coords2)))
    tmp_col  = np.empty(max(len(outCoords1), len(outCoords2)))

    tmp2D = np.empty((len(outCoords1), len(coords2)))

    # call optimized kernel
    return interpolate_2D_numba(
        vals, P1, P2,
        coords1, coords2,
        outCoords1, outCoords2,
        outVals,
        nodes1, baryW1,
        nodes2, baryW2,
        tmp_line, tmp_col, tmp2D
    )


@njit(fastmath=True)
def interpolate_2D_numba(vals, P1, P2,
                        coords1, coords2,
                        outCoords1, outCoords2,
                        outVals,
                        nodes1, baryW1,
                        nodes2, baryW2,
                        tmp_line, tmp_col, tmp2D):
    """
    Fully optimized 2D interpolation kernel.

    Requirements:
    - tmp_line shape = (len(coords2),)
    - tmp_col  shape = (len(outCoords1),)
    - No allocations inside
    """

    n1 = len(coords1)
    n2 = len(coords2)

    m1 = len(outCoords1)
    m2 = len(outCoords2)
        
    # loop over second dimension (columns)
    for j in range(n2):
        # gather column j into tmp_line
        for i in range(n1):
            tmp_line[i] = vals[i, j]

        # interpolate along coords1 → outCoords1
        interpolate_1D_cartesian_numba(
            tmp_line, P1, coords1, outCoords1, tmp_col, nodes1, baryW1
        )

        # store intermediate result into tmp_col
        for i in range(m1):
            tmp2D[i, j] = tmp_col[i]

    # now interpolate in second direction (rows)
    for i in range(m1):
        # tmp_line now used as row buffer (size n2 → m2 reuse carefully)

        # copy row
        for j in range(n2):
            tmp_line[j] = tmp2D[i, j]

        # interpolate along coords2 → outCoords2
        interpolate_1D_cartesian_numba(
            tmp_line[:n2], P2, coords2, outCoords2, tmp_col[:m2], nodes2, baryW2
        )

        # write back
        for j in range(m2):
            outVals[i, j] = tmp_col[j]

    return outVals


def interpolate_1D_cartesian(vals, P, coords, outCoords, outVals):
    """
    This function interpolates
    1) a 1D DG solution on Lobatto nodes to outCoords using barycentric interpolation,
    2) a 1D FV solution on cell centers to outCoords assuming piecewise constant values in each cell
    
    If an output coordinate coincides a node/cell center, the value at that node is taken
    If an output coordinate coincides with an element/cell interface, the average of the two interface values is taken
    """
    
    # check inputs
    if not (isinstance(vals, np.ndarray) and isinstance(outVals, np.ndarray)):
        raise ValueError("vals and outVals must be a numpy array")
    if vals.shape != (len(coords),) or outVals.shape != (len(outCoords),):
        raise ValueError(f"vals and outVals must have shape ({len(coords)},) and ({len(outCoords)},)")
    if P < 0:
        raise ValueError("P must be >= 0")
    if not np.all((outCoords >= coords[0]) & (outCoords <= coords[-1]) & (coords >= 0)):
        raise ValueError("All outCoords must be between the first and last coord and >= 0")
    if len(coords) % (P + 1) != 0:
        raise ValueError("Length of coords must be divisible by P + 1")
    if not np.all(np.diff(coords) > -1e-15): # include duplicate coordinate entries for interfaces
        raise ValueError("coords must be sorted in ascending order")
    if not np.all(np.diff(outCoords) > -1e-15): # include duplicate coordinate entries for interfaces
        raise ValueError("outCoords must be sorted in ascending order")

    nodes, _ = convergence.LGL_NodesWeights(P)
    baryWeights = convergence.barycentric_weights(P)

    # call optimized kernel
    return interpolate_1D_cartesian_numba(vals, P, coords, outCoords, outVals, nodes, baryWeights)


@njit(fastmath=True)
def interpolate_1D_cartesian_numba(vals, P, coords, outCoords, outVals, nodes, baryW):
    """
    Numba-optimized kernel for 1D interpolation.

    Assumptions (NOT checked here):
    - Inputs are valid and consistent
    - coords sorted ascending
    - DG nodes are Lobatto nodes per element
    - outCoords lie within domain

    This function:
    - performs no allocations
    - minimizes branching
    - is designed for maximum performance
    """

    n = len(coords)
    m = len(outCoords)

    nNodes = P + 1
    nCells = n // nNodes

    # ---------------- FV CASE ----------------
    if P == 0:
        deltaX = coords[1] - coords[0]
        cellIdx = 0

        for i in range(m):
            x = outCoords[i]

            while cellIdx < n - 1 and coords[cellIdx] + 0.5 * deltaX < x:
                if abs(coords[cellIdx] + 0.5 * deltaX - x) < 1e-14:
                    break
                cellIdx += 1

            if cellIdx < n - 1 and abs(coords[cellIdx] + 0.5 * deltaX - x) < 1e-14:
                outVals[i] = 0.5 * (vals[cellIdx] + vals[cellIdx + 1])
            else:
                outVals[i] = vals[cellIdx]

        return outVals

    # ---------------- DG CASE ----------------
    cellIdx = 0
    x_lIdx = 0

    for i in range(m):
        x = outCoords[i]

        # find cell (monotone scan)
        while cellIdx < nCells - 1:
            
            deltaX = coords[x_lIdx + nNodes - 1] - coords[x_lIdx]
            
            if coords[x_lIdx] + deltaX >= x:
                break
            cellIdx += 1
            x_lIdx += nNodes

        start = cellIdx * nNodes
        deltaX = coords[start + nNodes - 1] - coords[start]

        # --- node hit check ---
        hit = False
        for k in range(nNodes):
            idx = start + k
            if abs(x - coords[idx]) < 1e-14:

                if k == 0 and cellIdx != 0:
                    outVals[i] = 0.5 * (vals[idx] + vals[idx - 1])
                elif k == nNodes - 1 and cellIdx != nCells - 1:
                    outVals[i] = 0.5 * (vals[idx] + vals[idx + 1])
                else:
                    outVals[i] = vals[idx]

                hit = True
                break

        # --- barycentric interpolation ---
        if not hit:
            xi = (2.0 * (x - coords[start]) / deltaX) - 1.0

            num = 0.0
            den = 0.0

            for k in range(nNodes):
                diff = xi - nodes[k]
                tmp = baryW[k] / diff
                num += tmp * vals[start + k]
                den += tmp

            outVals[i] = num / den

    return outVals


def test_interpolate_2D_cartesian():
    # simple test case: 2D paraboloid sampled on Lobatto nodes
    P1 = 3
    P2 = 3

    coords1 = np.array([0.0, 0.25, 0.75, 1.0])
    coords2 = np.array([0.0, 0.5, 1.0])

    outCoords1 = np.array([0.125, 0.5, 0.875])
    outCoords2 = np.array([0.25, 0.75])

    vals = np.zeros((len(coords1), len(coords2)))
    for i in range(len(coords1)):
        for j in range(len(coords2)):
            vals[i, j] = coords1[i]**2 + coords2[j]**2

    outVals = np.empty((len(outCoords1), len(outCoords2)))

    interpolate_2D_cartesian(vals, P1, P2, coords1, coords2, outCoords1, outCoords2, outVals)

    expected_outVals = np.zeros((len(outCoords1), len(outCoords2)))
    for i in range(len(outCoords1)):
        for j in range(len(outCoords2)):
            expected_outVals[i, j] = outCoords1[i]**2 + outCoords2[j]**2

    assert np.allclose(outVals, expected_outVals), f"Expected {expected_outVals}, got {outVals}"


def test_interpolate_1D_cartesian():
    P = 3

    coords = np.array([0.0, 0.25, 0.75, 1.0])
    outCoords = np.array([0.125, 0.5, 0.875])

    vals = np.zeros(len(coords))
    for i in range(len(coords)):
        vals[i] = coords[i]**2

    outVals = np.empty(len(outCoords))

    interpolate_1D_cartesian(vals, P, coords, outCoords, outVals)

    expected_outVals = np.zeros(len(outCoords))
    for i in range(len(outCoords)):
        expected_outVals[i] = outCoords[i]**2

    assert np.allclose(outVals, expected_outVals), f"Expected {expected_outVals}, got {outVals}"


def collapse_duplicate_coords_1D(coords, vals, axis=0, tol=1e-14):
    """
    Collapse duplicated coordinates by averaging interface values.

    Parameters
    ----------
    coords : ndarray (n,)
    vals   : ndarray
        Data whose length along `axis` matches coords.
    axis : int
        Axis corresponding to coords.

    Returns
    -------
    coords_new
    vals_new
    """

    coords_new = []
    vals_new = []

    i = 0
    n = len(coords)

    while i < n:

        # duplicated interface node
        if i < n - 1 and abs(coords[i+1] - coords[i]) < tol:

            coords_new.append(coords[i])

            v0 = np.take(vals, i, axis=axis)
            v1 = np.take(vals, i + 1, axis=axis)

            vals_new.append(0.5 * (v0 + v1))

            i += 2

        else:

            coords_new.append(coords[i])

            vals_new.append(np.take(vals, i, axis=axis))

            i += 1

    coords_new = np.array(coords_new)
    vals_new = np.stack(vals_new, axis=axis)

    return coords_new, vals_new


def collapse_duplicate_coords_2D(ax, rad, u, tol=1e-14):
    """
    Collapse duplicated DG interface coordinates in both directions.
    """

    ax_new, u_new = collapse_duplicate_coords_1D(
        ax,
        u,
        axis=0,
        tol=tol
    )

    rad_new, u_new = collapse_duplicate_coords_1D(
        rad,
        u_new,
        axis=1,
        tol=tol
    )

    return ax_new, rad_new, u_new


def compute_errors(u, u_ref, ax, rad):
    
    dx = np.diff(ax).mean()
    dr = np.diff(rad).mean()

    err = np.abs(u - u_ref)

    L_inf = np.max(err)
    L1 = np.sum(err) * dx * dr
    L2 = np.sqrt(np.sum(err**2) * dx * dr)

    return L_inf, L1, L2


def compute_eoc(errors):
    eoc = []
    for i in range(1, len(errors)):
        e = errors[i]
        e_prev = errors[i-1]
        rate = np.log(e_prev / e) / np.log(2)
        eoc.append(rate)
    return eoc


#%% Bulk EOC

# ==========================================
# RUN ONE SIMULATION
# ==========================================

def run_case(axN, radN):

    u, ax, rad, outlet, t = run_sim(
        axN,
        radN,
        **setting
    )

    return {
        "axN": axN,
        "radN": radN,
        "u": u,
        "ax": ax,
        "rad": rad,
        "outlet": outlet,
        "t": t,
    }


# ==========================================
# RUN ALL CASES IN PARALLEL
# ==========================================

cases = list(zip(axial_levels, radial_levels))

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_case)(axN, radN)
    for axN, radN in cases
)

# map results by resolution
result_map = {
    (r["axN"], r["radN"]): r
    for r in results
}

# ==========================================
# EXTRACT REFERENCE SOLUTION
# ==========================================

ref_key = (axial_levels[-1], radial_levels[-1])

ref = result_map[ref_key]

u_ref = ref["u"]
ax_ref = ref["ax"]
rad_ref = ref["rad"]
outlet_ref = ref["outlet"]
t_ref = ref["t"]

u_ref = u_ref.reshape(len(ax_ref), len(rad_ref))

ax_ref, rad_ref, u_ref = collapse_duplicate_coords_2D(
    ax_ref,
    rad_ref,
    u_ref
)

# ensure shape = (time, nPorts)
if outlet_ref.ndim == 1:
    outlet_ref_use = outlet_ref[:, None]
else:
    outlet_ref_use = outlet_ref

dt = t_ref[1] - t_ref[0]

# ==========================================
# ERROR COMPUTATION
# ==========================================

errors_outlet_Linf = []
errors_outlet_L1 = []
errors_outlet_L2 = []

all_port_errors_Linf = []
all_port_errors_L1 = []
all_port_errors_L2 = []

# exclude reference level
for axN, radN in zip(axial_levels[:-1], radial_levels[:-1]):

    res = result_map[(axN, radN)]

    outlet = res["outlet"]
    t = res["t"]

    if outlet.ndim == 1:
        outlet = outlet[:, None]

    nPorts = outlet.shape[1]

    port_Linf = []
    port_L1 = []
    port_L2 = []

    for port in range(nPorts):

        err = np.abs(
            outlet[:, port]
            - outlet_ref_use[:, port]
        )

        Linf = np.max(err)
        L1 = np.sum(err) * dt
        L2 = np.sqrt(np.sum(err**2) * dt)

        port_Linf.append(Linf)
        port_L1.append(L1)
        port_L2.append(L2)

        print(
            f"ax={axN}, rad={radN}, port={port}: "
            f"Linf={Linf:.3e}, "
            f"L1={L1:.3e}, "
            f"L2={L2:.3e}"
        )

    all_port_errors_Linf.append(port_Linf)
    all_port_errors_L1.append(port_L1)
    all_port_errors_L2.append(port_L2)

    errors_outlet_Linf.append(np.max(port_Linf))
    errors_outlet_L1.append(np.max(port_L1))
    errors_outlet_L2.append(np.max(port_L2))

# ==========================================
# COMPUTE EOC PER PORT
# ==========================================

nPorts = len(all_port_errors_Linf[0])

port_eoc_Linf = []
port_eoc_L1 = []
port_eoc_L2 = []

for port in range(nPorts):

    err_Linf = [lvl[port] for lvl in all_port_errors_Linf]
    err_L1 = [lvl[port] for lvl in all_port_errors_L1]
    err_L2 = [lvl[port] for lvl in all_port_errors_L2]

    port_eoc_Linf.append(compute_eoc(err_Linf))
    port_eoc_L1.append(compute_eoc(err_L1))
    port_eoc_L2.append(compute_eoc(err_L2))

# ==========================================
# PRINT RESULTS
# ==========================================

print("\nPer-port Errors and EOC:")

for port in range(nPorts):

    print(f"\nPort {port}")

    for lvl in range(len(all_port_errors_Linf)):
        print(
            f"  Level {lvl}: "
            f"Linf={all_port_errors_Linf[lvl][port]:.3e}, "
            f"L1={all_port_errors_L1[lvl][port]:.3e}, "
            f"L2={all_port_errors_L2[lvl][port]:.3e}"
        )

    for lvl in range(len(port_eoc_Linf[port])):
        print(
            f"  EOC {lvl}->{lvl+1}: "
            f"Linf={port_eoc_Linf[port][lvl]:.2f}, "
            f"L1={port_eoc_L1[port][lvl]:.2f}, "
            f"L2={port_eoc_L2[port][lvl]:.2f}"
        )

# # =============================
# # PLOT REFERENCE SOLUTION
# # =============================

# X, Y = np.meshgrid(ax_ref, rad_ref)

# fig, ax = plt.subplots(figsize=(8, 5))

# pcm = ax.pcolormesh(
#     X,
#     Y,
#     u_ref.T,
#     shading='auto',
#     cmap='viridis'
# )

# cbar = fig.colorbar(pcm, ax=ax)
# cbar.set_label("Concentration")

# ax.set_xlabel("Axial coordinate")
# ax.set_ylabel("Radial coordinate")
# ax.set_title("Reference 2D Solution")

# plt.tight_layout()
# plt.show()


# # =============================
# # Bulk EOC LOOP
# # =============================

# errors_Linf = []
# errors_L1 = []
# errors_L2 = []

# for axN, radN in zip(axial_levels, radial_levels):
#     print(f"Running ax={axN}, rad={radN}")

#     u, axCoords, radCoords, _, _ = run_sim(axN, radN, **setting)
#     u = u.reshape(len(axCoords), len(radCoords))

#     # project to reference grid
#     u_interp = interpolate_2D_cartesian(u, axP, radP, axCoords, radCoords, ax_ref, rad_ref, np.empty((len(ax_ref), len(rad_ref))))

#     Linf, L1, L2 = compute_errors(u_interp, u_ref, ax_ref, rad_ref)

#     errors_Linf.append(Linf)
#     errors_L1.append(L1)
#     errors_L2.append(L2)


# # =============================
# # Bulk EOC RESULTS
# # =============================

# eoc_Linf = compute_eoc(errors_Linf)
# eoc_L1 = compute_eoc(errors_L1)
# eoc_L2 = compute_eoc(errors_L2)

# print("\nBulk Errors:")
# for i in range(len(errors_Linf)):
#     print(f"Level {i}: Linf={errors_Linf[i]:.3e}, L1={errors_L1[i]:.3e}, L2={errors_L2[i]:.3e}")

# print("\nBulk EOC:")
# for i in range(len(eoc_Linf)):
#     print(f"Level {i}->{i+1}: Linf={eoc_Linf[i]:.2f}, L1={eoc_L1[i]:.2f}, L2={eoc_L2[i]:.2f}")
    

    
    
#%% Tests

def build_dg_coords(nElem, P, xL=0.0, xR=1.0):
    """
    Construct DG nodal coordinates with duplicated interfaces.
    """

    nodes, _ = convergence.LGL_NodesWeights(P)

    coords = []

    dx = (xR - xL) / nElem

    for e in range(nElem):

        xl = xL + e * dx
        xr = xl + dx

        # map Lobatto nodes from [-1,1] to element
        x_loc = 0.5 * (xr - xl) * (nodes + 1.0) + xl

        coords.extend(x_loc)

    return np.array(coords)


def exact_solution(x, r, exp):
    return x**exp + r**exp


def test_2D_interpolation_convergence():

    P = 2

    levels = [1, 2, 4, 8, 16, 32]

    errors = []

    # -----------------------------
    # reference grid
    # -----------------------------
    ax_ref = build_dg_coords(levels[-1]*2, P)
    rad_ref = build_dg_coords(levels[-1]*2, P)

    Xref, Rref = np.meshgrid(ax_ref, rad_ref, indexing='ij')

    u_ref = exact_solution(Xref, Rref, P+1)

    # -----------------------------
    # convergence loop
    # -----------------------------
    for nElem in levels:

        print(f"\nTesting nElem = {nElem}")

        ax = build_dg_coords(nElem, P)
        rad = build_dg_coords(nElem, P)

        X, R = np.meshgrid(ax, rad, indexing='ij')

        u = exact_solution(X, R, P+1)

        # interpolate to reference grid
        u_interp = interpolate_2D_cartesian(
            u,
            P,
            P,
            ax,
            rad,
            ax_ref,
            rad_ref,
            np.empty((len(ax_ref), len(rad_ref)))
        )

        err = np.abs(u_interp - u_ref)

        Linf = np.max(err)

        errors.append(Linf)

        print(f"Linf = {Linf:.6e}")

    # -----------------------------
    # compute EOC
    # -----------------------------
    print("\nInterpolation EOC:")

    for i in range(1, len(errors)):

        eoc = np.log(errors[i-1] / errors[i]) / np.log(2)

        print(f"{levels[i-1]} -> {levels[i]} : {eoc:.4f}")
        
        
# test_2D_interpolation_convergence()