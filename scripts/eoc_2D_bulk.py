import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

import numpy as np
from functools import partial
from numba import njit

from cadet import Cadet

from src.utility import convergence
from src.benchmark_models import setting_Col2D_lin_1comp_benchmark1
import src.benchmark_models.helper_setup_2Dmodels as helper


# =============================
# SETTINGS
# =============================
output_path = Path.cwd() / "output" / "test_cadet-core"
cadet_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

axP = 2
radP = axP

axial_levels = [1, 2, 4, 8]  # axial elements
radial_levels = [1, 1, 1, 1]  # radial elements
# note: reference will have twice the resolution of the finest level

setting = {
    'npartype': 0,
    'nRadialZones': 1,
    'inlet_function': partial(helper.constInlet, const=1.0),
    'WRITE_SOLUTION_LAST': True,
    'WRITE_SOLUTION_BULK': True,
    'tEnd': 25,
    'USE_MODIFIED_NEWTON': True
}

# =============================
# HELPER FUNCTIONS
# =============================

def run_sim(axNElem, radNElem, **kwargs):

    kwargs.update({'name': f'run_ax{axNElem}_rad{radNElem}'})

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

    return bulk, ax_coords, rad_coords


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

    # call optimized kernel
    return interpolate_2D_numba(
        vals, P1, P2,
        coords1, coords2,
        outCoords1, outCoords2,
        outVals,
        nodes1, baryW1,
        nodes2, baryW2,
        tmp_line, tmp_col
    )


@njit(fastmath=True)
def interpolate_2D_numba(vals, P1, P2,
                        coords1, coords2,
                        outCoords1, outCoords2,
                        outVals,
                        nodes1, baryW1,
                        nodes2, baryW2,
                        tmp_line, tmp_col):
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

        # store intermediate result into outVals[:, j]
        for i in range(m1):
            outVals[i, j] = tmp_col[i]

    # now interpolate in second direction (rows)
    for i in range(m1):
        # tmp_line now used as row buffer (size n2 → m2 reuse carefully)

        # copy row
        for j in range(n2):
            tmp_line[j] = outVals[i, j]

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
            deltaX = coords[x_lIdx + nNodes] - coords[x_lIdx]
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


# =============================
# REFERENCE SOLUTION
# =============================

print("Running reference solution...")
u_ref, ax_ref, rad_ref = run_sim(axial_levels[-1]*2, radial_levels[-1]*2, **setting)

# reshape (important!)
u_ref = u_ref.reshape(len(ax_ref), len(rad_ref))


# =============================
# EOC LOOP
# =============================

errors_Linf = []
errors_L1 = []
errors_L2 = []

for axN, radN in zip(axial_levels, radial_levels):
    print(f"Running ax={axN}, rad={radN}")

    u, axCoords, radCoords = run_sim(axN, radN, **setting)
    u = u.reshape(len(axCoords), len(radCoords))

    # project to reference grid
    u_interp = interpolate_2D_cartesian(u, axP, radP, axCoords, radCoords, ax_ref, rad_ref, np.empty((len(ax_ref), len(rad_ref))))

    Linf, L1, L2 = compute_errors(u_interp, u_ref, ax_ref, rad_ref)

    errors_Linf.append(Linf)
    errors_L1.append(L1)
    errors_L2.append(L2)


# =============================
# EOC RESULTS
# =============================

eoc_Linf = compute_eoc(errors_Linf)
eoc_L1 = compute_eoc(errors_L1)
eoc_L2 = compute_eoc(errors_L2)

print("\nErrors:")
for i in range(len(errors_Linf)):
    print(f"Level {i}: Linf={errors_Linf[i]:.3e}, L1={errors_L1[i]:.3e}, L2={errors_L2[i]:.3e}")

print("\nEOC:")
for i in range(len(eoc_Linf)):
    print(f"Level {i}->{i+1}: Linf={eoc_Linf[i]:.2f}, L1={eoc_L1[i]:.2f}, L2={eoc_L2[i]:.2f}")