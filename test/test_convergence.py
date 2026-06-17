import numpy as np
import pytest

from src.utility import convergence


def test_Gauss_and_Lobatto_nodes():

    # test Legendre-Gauss
    with pytest.raises(ValueError):
        LGnodes, LGweights = convergence.legendre_gauss_nodes_and_weights(-1)

    LGnodes, LGweights = convergence.legendre_gauss_nodes_and_weights(0)
    np.testing.assert_almost_equal(LGnodes, np.array([0.0]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([2.0]), decimal=14)

    LGnodes, LGweights = convergence.legendre_gauss_nodes_and_weights(1)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-np.sqrt(1/3), np.sqrt(1/3)]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([1.0, 1.0]), decimal=14)

    LGnodes, LGweights = convergence.legendre_gauss_nodes_and_weights(2)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)]), decimal=14)
    np.testing.assert_almost_equal(
        LGweights, np.array([5/9, 8/9, 5/9]), decimal=14)

    LGnodes, LGweights = convergence.legendre_gauss_nodes_and_weights(4)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-0.906179845938664, -0.538469310105683, 0.0,
                           0.538469310105683, 0.906179845938664]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([
        0.236926885056189, 0.478628670499366, 0.568888888888889,
        0.478628670499366, 0.236926885056189]), decimal=14)

    # test Legendre-Gauss-Lobatto
    with pytest.raises(ValueError):
        LGnodes, LGweights = convergence.legendre_gauss_lobatto_nodes_and_weights(-1)
        LGnodes, LGweights = convergence.legendre_gauss_lobatto_nodes_and_weights(0)

    LGnodes, LGweights = convergence.legendre_gauss_lobatto_nodes_and_weights(1)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-1.0, 1.0]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([1.0, 1.0]), decimal=14)

    LGnodes, LGweights = convergence.legendre_gauss_lobatto_nodes_and_weights(2)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-1.0, 0.0, 1.0]), decimal=14)
    np.testing.assert_almost_equal(
        LGweights, np.array([1/3, 4/3, 1/3]), decimal=14)

    LGnodes, LGweights = convergence.legendre_gauss_lobatto_nodes_and_weights(4)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-1.0, -np.sqrt(3/7), 0.0, np.sqrt(3/7), 1.0]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([
        1/10, 49/90, 32/45, 49/90, 1/10]), decimal=14)


def test_map_z_to_xi():

    polyDeg = 4

    nNodes = polyDeg + 1

    nodes, weights = convergence.LGL_NodesWeights(polyDeg)

    x_lIdx = 0
    cellIdx = 2
    stretch = 0.1
    deltaZ = 2 * stretch
    orig_coords = (nodes + 1) * stretch + deltaZ * cellIdx

    mapped_orig_coords = np.zeros(nNodes)

    for node in range(nNodes):
        mapped_orig_coords[node] = convergence.map_z_to_xi(
            orig_coords[x_lIdx + node], cellIdx, deltaZ)

    if (abs(mapped_orig_coords - nodes) > 1e-14).any():
        raise ValueError(
            "getSolutionDG: mapped_orig_coords != nodes"
        )


def test_calculate_eoc():
    
    dof = [1, 2, 4, 8, 16]
    error = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    eoc_expected = [1, 1, 1, 1]
    eoc = convergence.calculate_eoc(dof, error)

    np.testing.assert_almost_equal(eoc, eoc_expected)

    with pytest.raises(ValueError):
        dof_zero = [1, 2, 4, 8, 0]
        eoc = convergence.calculate_eoc(dof_zero, error)

    with pytest.raises(ValueError):
        error_smaller_zero = [-0.1, 0.05, 0.025, 0.0125, 0.00625]
        eoc = convergence.calculate_eoc(dof, error_smaller_zero)

    # Test eps
    error_zero = [0.1, 0.05, 0.025, 0.0125, 0]

    eoc_expected = [1, 1, 1, 45.67807191]
    eoc = convergence.calculate_eoc(dof, error_zero)
    np.testing.assert_almost_equal(eoc, eoc_expected)


def test_convergency_table():

    # GRM test for L1/max error
    expected_order = 3
    method = np.array([1, 1]) * (expected_order - 1)
    discretizations = [[4, 8, 16], [1, 2, 4]]
    full_DOFs = False

    error_type = ["max"]
    initial_errors = np.random.rand(100)
    error_factors = np.zeros(len(discretizations[0]))
    for i in range(0, len(error_factors)):
        error_factors[i] = 2 ** (expected_order * i)
    abs_errors = np.zeros([len(error_factors), len(initial_errors)])

    for factor in range(0, len(error_factors)):
        abs_errors[factor] = initial_errors / error_factors[factor]

    header, table = convergence.convergency_table(method=method, disc=discretizations, abs_errors=abs_errors,
                                      error_types=error_type, full_DOFs=full_DOFs)

    np.testing.assert_array_equal(header, np.array(
        ["$N_e^z$", "$N_e^p$", 'Max. error', 'Max. EOC']))
    np.testing.assert_almost_equal(
        table[:, 0], discretizations[0])  # check axial disc
    np.testing.assert_almost_equal(
        table[:, 1], discretizations[1])  # check particle disc
    np.testing.assert_almost_equal(
        table[:, 2], np.amax(abs_errors, axis=1))  # check errors
    np.testing.assert_almost_equal(table[1:, 3], np.ones(
        len(discretizations[0])-1) * expected_order)  # check EOC

    # LRMP/LRM test for L1/max error
    expected_order = 3
    method = (expected_order - 1)
    discretizations = [4, 8, 16]
    full_DOFs = False

    error_type = ["max"]
    initial_errors = np.random.rand(100)
    error_factors = np.zeros(len(discretizations))
    for i in range(0, len(error_factors)):
        error_factors[i] = 2 ** (expected_order * i)
    abs_errors = np.zeros([len(error_factors), len(initial_errors)])

    for factor in range(0, len(error_factors)):
        abs_errors[factor] = initial_errors / error_factors[factor]

    header, table = convergence.convergency_table(method=method, disc=discretizations, abs_errors=abs_errors,
                                      error_types=error_type, full_DOFs=full_DOFs)

    np.testing.assert_array_equal(header, np.array(
        ["$N_e^z$", 'Max. error', 'Max. EOC']))
    np.testing.assert_almost_equal(
        table[:, 0], discretizations)  # check axial disc
    np.testing.assert_almost_equal(
        table[:, 1], np.amax(abs_errors, axis=1))  # check errors
    np.testing.assert_almost_equal(table[1:, 2], np.ones(
        len(discretizations)-1) * expected_order)  # check EOC


def test_get_unique_DGsolution():

    nCells = 3
    polyDeg = 3
    nComp = 2
    timePoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    nTime = len(timePoints)

    nNodes = polyDeg + 1
    nPoints = nCells*nNodes
    solution = np.zeros((nTime, nCells*nNodes, nComp))
    compare = np.zeros((nTime, nCells*nNodes-nCells+1, nComp))

    # fill timepoints
    for nodeIdx in range(nPoints):
        for compIdx in range(nComp):
            solution[:, nodeIdx, compIdx] = timePoints
            if nodeIdx < nCells*nNodes-nCells+1:
                compare[:, nodeIdx, compIdx] = timePoints

    solution[0, :, 0] = np.ones(nPoints)
    compare[0, :, 0] = np.ones(nCells*nNodes-nCells+1)
    solution[0, :, 1] = [1, 2, 3, 4,
                         5, 6, 7, 8,
                         9, 10, 11, 12]
    compare[0, :, 1] = [1, 2, 3, 4.5, 6, 7, 8.5, 10, 11, 12]
    solution[6, :, 1] = [1, 2, 3, 4,
                         5, 6, 7, 8,
                         9, 10, 11, 12]
    compare[6, :, 1] = [1, 2, 3, 4.5, 6, 7, 8.5, 10, 11, 12]

    # Test single space slice
    np.testing.assert_array_equal(convergence.get_unique_DGsolution(
        polyDeg, np.ones(nPoints)), np.ones(nPoints-nCells+1))
    np.testing.assert_array_equal(convergence.get_unique_DGsolution(
        polyDeg, solution[0, :, 1]), compare[0, :, 1])

    # Test full time x space x comp array
    test = convergence.get_unique_DGsolution(polyDeg, solution)
    np.testing.assert_array_equal(test, compare)

    # Test unique DG coords
    coords = [0, 0.1, 0.5, 0.9, 1.0, 1.0, 1.1,
              1.5, 1.9, 2.0, 2.0, 2.1, 2.5, 2.9, 3.0]
    compare_coords = [0, 0.1, 0.5, 0.9, 1.0,
                      1.1, 1.5, 1.9, 2.0, 2.1, 2.5, 2.9, 3.0]
    np.testing.assert_array_equal(
        convergence.get_unique_DGcoordinates(coords), compare_coords)


def test_get_interpolated_DGsolution():

    polyDeg = 4
    nCells = 30
    column_length = 1.0

    deltaZ = column_length / nCells
    nNodes = (polyDeg + 1)
    nPoints = nNodes * nCells
    nodes, weights = convergence.LGL_NodesWeights(polyDeg)

    x_l = np.linspace(0.0, column_length, num=nCells, endpoint=False)

    orig_coords = np.zeros(nPoints)
    orig_values = np.zeros(nPoints)

    for cellIdx in range(nCells):
        orig_coords[cellIdx * nNodes: (cellIdx + 1) *
                    nNodes] = convergence.map_xi_to_z(nodes, cellIdx, deltaZ)

    print("orig_coords:\n", orig_coords)

    # test linear concentration function
    orig_values = orig_coords
    output_coords = np.linspace(
        0.0, column_length, num=2*nPoints, endpoint=True)
    
    output_values = convergence.get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(output_values, output_coords, decimal=14)

    # test quadratic concentration function
    out_polyDeg = 3
    out_nodes, out_weights = convergence.LGL_NodesWeights(polyDeg)
    for cellIdx in range(nCells*2):
        output_coords[cellIdx * nNodes: (cellIdx + 1) *
                      nNodes] = convergence.map_xi_to_z(out_nodes, cellIdx, deltaZ/2)
    orig_values = np.square(orig_coords)
    # output_coords = np.linspace(0.0, column_length, num=int(nPoints/2), endpoint=True)
    output_values = convergence.get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(
        output_values, np.square(output_coords), decimal=14)

    # test sinus concentration function (limited accuracy)
    orig_values = np.sin(orig_coords)
    output_values = convergence.get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(
        output_values, np.sin(output_coords), decimal=6)

    # print("output_coords:\n", output_coords)
    # print("output_values:\n", output_values)

    # test two dimensional array
    output_coords = np.linspace(0.0, column_length, num=nPoints, endpoint=True)
    nTime = 10
    orig_values = np.zeros((nTime, nPoints))
    orig_values[0, :] = np.square(orig_coords)
    orig_values[5, :] = np.power(orig_coords, polyDeg)
    compare = np.zeros((nTime, nPoints))
    compare[0, :] = np.square(output_coords)
    compare[5, :] = np.power(output_coords, polyDeg)

    output_values = convergence.get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(output_values, compare, decimal=14)

    # test three dimensional array
    nTime = 12
    nComp = 3
    orig_values = np.zeros((nTime, nPoints, nComp))
    orig_values[0, :, 0] = np.square(orig_coords)
    orig_values[3, :, 0] = np.power(orig_coords, polyDeg)
    orig_values[7, :, 2] = np.power(orig_coords, polyDeg - 1)
    compare = np.zeros((nTime, nPoints, nComp))
    compare[0, :, 0] = np.square(output_coords)
    compare[3, :, 0] = np.power(output_coords, polyDeg)
    compare[7, :, 2] = np.power(output_coords, polyDeg - 1)

    output_values = convergence.get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(output_values, compare, decimal=14)
def test_get_interpolated_solution_2d_dg_grid():

    # -------------------------
    # DG parameters
    # -------------------------
    Lx, Ly = 1.0, 1.0
    polyDeg = 3
    nCellsX, nCellsY = 2, 2

    nNodes = polyDeg + 1

    # -------------------------
    # reference DG nodes
    # -------------------------
    nodes, weights = convergence.LGL_NodesWeights(polyDeg)

    # -------------------------
    # build DG grid in x/y
    # -------------------------
    dx = Lx / nCellsX
    dy = Ly / nCellsY

    Nx = nCellsX * nNodes
    Ny = nCellsY * nNodes

    # structured coordinates aligned with DG nodes
    orig_coords = np.zeros((Nx, Ny, 2))

    for cx in range(nCellsX):
        for cy in range(nCellsY):

            for i in range(nNodes):
                for j in range(nNodes):

                    ix = cx * nNodes + i
                    iy = cy * nNodes + j

                    # map reference LGL node -> physical cell
                    xi = nodes[i]
                    eta = nodes[j]

                    x = cx * dx + 0.5 * (xi + 1) * dx
                    y = cy * dy + 0.5 * (eta + 1) * dy

                    orig_coords[ix, iy, 0] = x
                    orig_coords[ix, iy, 1] = y

    # -------------------------
    # polynomial DG test function
    # (must be exactly representable in degree p)
    # -------------------------
    def u(x, y):
        return 1 + 2*x + 3*y + x*y + x**2 + y**2

    orig_values = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):
            orig_values[i, j] = u(orig_coords[i, j, 0], orig_coords[i, j, 1])

    # -------------------------
    # test points
    # -------------------------
    np.random.seed(0)
    Nout = 100
    output_coords = np.random.rand(Nout, 2)

    # -------------------------
    # call interpolation
    # -------------------------
    out = convergence.get_interpolated_solution_2d(
        orig_values,
        orig_coords,
        (Lx, Ly),
        output_coords,
        polyDeg,
        nCellsX,
        nCellsY
    )

    exact = np.array([u(x, y) for x, y in output_coords])
    
    np.testing.assert_almost_equal(
        out, exact, decimal=14
    )
