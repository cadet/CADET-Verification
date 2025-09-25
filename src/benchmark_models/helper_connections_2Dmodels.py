# -*- coding: utf-8 -*-
"""

This script implements helper functions to set up the connectivity/flow rate matrices for 2D settings

"""

import numpy as np
from scipy.special import legendre

def q_and_L(poly_deg, x):
    P = legendre(poly_deg)
    dP = P.deriv()
    L = P(x)
    q = P.deriv()(x)
    q_der = P.deriv(2)(x)
    return L, q, q_der


def lgl_nodes_weights(poly_deg):
    if poly_deg < 1:
        raise ValueError("Polynomial degree must be at least 1!")

    nodes = np.zeros(poly_deg + 1)
    weights = np.zeros(poly_deg + 1)
    pi = np.pi
    tolerance = 1e-15
    n_iterations = 10

    if poly_deg == 1:
        nodes[0] = -1
        nodes[1] = 1
        weights[0] = 1
        weights[1] = 1
    else:
        nodes[0] = -1
        nodes[poly_deg] = 1
        weights[0] = 2.0 / (poly_deg * (poly_deg + 1.0))
        weights[poly_deg] = weights[0]

        for j in range(1, (poly_deg + 1) // 2):
            x = -np.cos(pi * (j + 0.25) / poly_deg - 3 /
                        (8.0 * poly_deg * pi * (j + 0.25)))
            for k in range(n_iterations):
                L, q, q_der = q_and_L(poly_deg, x)
                dx = q / q_der
                x -= dx
                if abs(dx) <= tolerance * abs(x):
                    break
            nodes[j] = x
            nodes[poly_deg - j] = -x
            L, q, q_der = q_and_L(poly_deg, x)
            weights[j] = 2.0 / (poly_deg * (poly_deg + 1.0) * L**2)
            weights[poly_deg - j] = weights[j]

        if poly_deg % 2 == 0:
            L, q, q_der = q_and_L(poly_deg, 0.0)
            nodes[poly_deg // 2] = 0
            weights[poly_deg // 2] = 2.0 / (poly_deg * (poly_deg + 1.0) * L**2)

    return nodes, weights


def generate_connections_matrix(rad_method, rad_cells,
                                velocity, porosity, col_radius,
                                add_inlet_per_port=True, add_outlet=False):
    """Computes the connections matrix with const. velocity flow rates, and radial coordinates.
    Equidistant cell/element spacing is assumed.
    
    Parameters
    ----------
    rad_method : int
        radial method / polynomial degree
    rad_cells : int
        radial number of cells
    velocity : float
        column velocity (constant)
    porosity : float
        column porosity (constant)
    col_radius : float
        column radius (constant)
    add_inlet_per_port : int | bool
        specifies how many radial zones are used either by number or by true to specify one per port
    add_outlet : bool
        specifies whetehr or not an outlet is connected per radial zone
    
    Returns
    -------
    List of float, List of float
        Connections matrix, radial coordinates.
    """

    nRadPoints = (rad_method + 1) * rad_cells

    # we want the same velocity within each radial zone and use an equidistant radial grid, ie we adjust the volumetric flow rate accordingly in each port
    # 1. compute cross sections

    subcellCrossSectionAreas = []
    rad_coords = []

    if rad_method > 0:

        nodes, weights = lgl_nodes_weights(rad_method)
        # scale the weights to radial element spacing
        # note that weights need to be scaled to 1 later, to give us the size of the corresponding subcells
        # print(sum(weights) / 2.0 - 1.0 < 1E-15)
        deltaR = col_radius / rad_cells
        for rIdx in range(rad_cells):
            jojoL = rIdx * deltaR
            for node in range(rad_method + 1):
                jojoR = jojoL + weights[node] / 2.0 * deltaR
                # print("Left boundary: ", jojoL)
                # print("Right boundary: ", jojoR)
                subcellCrossSectionAreas.append(
                    np.pi * (jojoR ** 2 - jojoL ** 2))
                rad_coords.append(
                    rIdx * deltaR + (nodes[node] + 1) / 2.0 * deltaR)
                jojoL = jojoR
    else:
        deltaR = col_radius / nRadPoints
        jojoL = 0.0
        for rIdx in range(nRadPoints):
            rad_coords.append(rIdx * deltaR + deltaR / 2.0)
            jojoR = jojoL + deltaR
            subcellCrossSectionAreas.append(np.pi * (jojoR ** 2 - jojoL ** 2))
            jojoL = jojoR

    # create flow rates for each zone
    flowRates = []
    columnIdx = 0  # always needs to be the first unit
    
    for rad in range(nRadPoints):
        flowRates.append(subcellCrossSectionAreas[rad] * porosity * velocity)
    # create connections matrix
    connections = []
    # add inlet connections
    if add_inlet_per_port:

        nRadialZones = rad_cells if add_inlet_per_port is True else add_inlet_per_port

        if not rad_cells % nRadialZones == 0:
            raise Exception(
                f"Number of rad_cells {rad_cells} is not a multiple of radial zones {nRadialZones}")

        for rad in range(nRadPoints):
            zone = int(rad / (nRadPoints / nRadialZones))
            connections += [zone + 1, columnIdx,
                            0, rad, -1, -1, flowRates[rad]]
            if add_outlet:
                connections += [columnIdx, nRadialZones + 1 + zone,
                                rad, 0, -1, -1, flowRates[rad]]
    else:
        for rad in range(nRadPoints):
            connections += [1, columnIdx, 0, rad, -1, -1, flowRates[rad]]
            if add_outlet:
                connections += [columnIdx, nRadPoints + 1 + rad,
                                rad, 0, -1, -1, flowRates[rad]]
                
    return connections, rad_coords

