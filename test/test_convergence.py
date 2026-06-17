import numpy as np

from src.utility import convergence

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
