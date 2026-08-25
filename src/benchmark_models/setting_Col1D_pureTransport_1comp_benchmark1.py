# -*- coding: utf-8 -*-
"""

This script defines a pure transport (no particles) 1D column case study for
numerical benchmarks, in particular (super-)convergence studies of the bulk
solution.

The setting is an initial value problem: the inlet concentration is constant
zero and the concentration is initialized with a smooth profile, which is
evaluated on the spatial discretization grid (DG: LGL nodes per element,
FV: cell centers). Advection and dispersion can be switched on/off
independently.

Note that the nodal initial condition (INIT_STATE) depends on the spatial
grid, i.e. the model must be rebuilt for every refinement level (a pure
NELEM/NCOL update as done in bench_func.create_object_from_config would leave
a stale initial condition).
To this end, this module provides create_convergence_object, which is
compatible with the disc_refinement_functions interface of
bench_func.run_convergence_analysis, e.g.

    from functools import partial
    import src.benchmark_models.setting_Col1D_pureTransport_1comp_benchmark1 as setting

    model_kwargs = dict(
        spatial_method_bulk=3, column_geometry='AXIAL_FLOW_FRUSTUM',
        advection=True, dispersion=True, write_solution_bulk=True
    )

    addition = {
        'cadet_config_jsons': [setting.get_model(**model_kwargs)],
        'cadet_config_names': ['frustCol1D_pureTransport_1comp_benchmark1'],
        ...
        'ax_discs': [[bench_func.disc_list(8, 9)]],  # = 8 * 2^disc_idx elements
        'disc_refinement_functions': [[
            partial(setting.create_convergence_object,
                    setting_name='frustCol1D_pureTransport_1comp_benchmark1',
                    base_refinement=1, model_kwargs=model_kwargs)
        ]]
    }

where ax_discs must be consistent with 8 * base_refinement * 2^disc_idx.

"""

import copy

from addict import Dict
import numpy as np
from cadet import Cadet

from src.utility.convergence import LGL_NodesWeights, generate_1D_name
from src.benchmark_models.setting_Col1D_lin_1comp_benchmark1 import (
    get_column_geometry_configuration
)


def get_initial_profile(**kwargs):
    """Return the smooth initial concentration profile c0(z/L).

    Defaults to a Gaussian bump

        c0(z*) = init_amplitude * exp(-(z* - init_center)^2 / (2 init_stddev^2))

    in normalized coordinates z* = z/L, whose default parameters decay below
    machine precision at the boundaries and are thus compatible with the
    constant zero inlet.

    Parameters
    ----------
    init_profile : callable, optional
        Custom profile: takes normalized coordinates z/L in [0, 1] as np.array
        and returns the concentration values. Overrides the Gaussian.
    init_center : float
        Center of the Gaussian in normalized coordinates, defaults to 0.5.
    init_stddev : float
        Standard deviation of the Gaussian in normalized coordinates,
        defaults to 0.05.
    init_amplitude : float
        Amplitude of the Gaussian, defaults to 1.0.

    Returns
    -------
    callable
        Profile function z/L in [0, 1] (np.array) -> concentration (np.array).
    """
    if 'init_profile' in kwargs:
        return kwargs['init_profile']

    center = kwargs.get('init_center', 0.5)
    stddev = kwargs.get('init_stddev', 0.05)
    amplitude = kwargs.get('init_amplitude', 1.0)

    def gaussian_bump(normed_coords):
        return amplitude * np.exp(
            -(np.asarray(normed_coords) - center) ** 2 / (2.0 * stddev ** 2)
        )

    return gaussian_bump

from scipy.special import erf


def get_normalized_grid_faces(
        n_elem,
        geometry,
        grid_type='equidistant',
        radial_inner_radius=None,
        radial_outer_radius=None):
    """Return the normalized cell face coordinates xi in [0, 1].

    Parameters
    ----------
    n_elem : int
        Number of FV cells.
    geometry : str
        'AXIAL_FLOW_CYLINDER', 'RADIAL_FLOW_CYLINDER_SHELL' or
        'AXIAL_FLOW_FRUSTUM'.
    grid_type : str
        'equidistant': uniform cells in the transport coordinate.
        'equivolume': cells of equal geometric volume; reduces to
        'equidistant' for the axial cylinder (and for a degenerate frustum
        with equal end radii). Equivolume grids nest under dyadic refinement,
        so fine-grid reference solutions can be restricted exactly.
    radial_inner_radius, radial_outer_radius : float
        End radii, required for the radial shell and frustum geometries.

    Returns
    -------
    np.ndarray
        Normalized face coordinates, shape (n_elem + 1,), with exact
        endpoints 0 and 1.
    """
    if grid_type not in ('equidistant', 'equivolume'):
        raise ValueError(f"Unknown grid_type: {grid_type}")

    xi = np.arange(n_elem + 1, dtype=float) / n_elem

    if grid_type == 'equidistant' or geometry == 'AXIAL_FLOW_CYLINDER':
        return xi

    r_in = radial_inner_radius
    r_out = radial_outer_radius
    if r_in is None or r_out is None:
        raise ValueError(
            "radial_inner_radius and radial_outer_radius are required for "
            f"equivolume grids in geometry {geometry}."
        )
    delta_r = r_out - r_in
    if delta_r == 0.0:
        # Degenerate frustum = cylinder: equal volumes = equal widths.
        return xi

    if geometry == 'RADIAL_FLOW_CYLINDER_SHELL':
        # Cell volume ~ rho^2 difference: rho_j = sqrt(rho_in^2 + xi_j*(rho_out^2 - rho_in^2))
        rho = np.sqrt(r_in**2 + xi * (r_out**2 - r_in**2))
        faces = (rho - r_in) / delta_r
    else:
        # Frustum: volume ~ sigma^3 difference in the apex coordinate
        # sigma = z + r_in/slope with slope = delta_r / L; in normalized
        # coordinates s = xi + r_in/delta_r:
        s_in = r_in / delta_r
        s_out = s_in + 1.0
        s = np.cbrt(s_in**3 + xi * (s_out**3 - s_in**3))
        faces = s - s_in

    faces[0] = 0.0
    faces[-1] = 1.0
    return faces


def get_initial_cell_averages(
        n_elem,
        geometry,
        radial_inner_radius=None,
        radial_outer_radius=None,
        xi_faces=None,
        **kwargs):
    """Return exact FV cell averages of the initial profile.

    The FV state is defined as the geometry-weighted cell average

        cbar_i = integral_{cell} A(x) c(x) dx
                / integral_{cell} A(x) dx.

    The coordinate used by the initial profile is the normalized
    computational coordinate xi in [0, 1].

    Geometry conventions
    --------------------
    AXIAL_FLOW_CYLINDER:
        A(xi) = const

    RADIAL_FLOW_CYLINDER_SHELL:
        A(xi) proportional to r(xi)

    AXIAL_FLOW_FRUSTUM:
        A(xi) proportional to r(xi)^2

    where

        r(xi) = r_inner + (r_outer - r_inner) * xi.

    For the default Gaussian initial profile, the integrals are
    evaluated analytically. For a custom ``init_profile``, sufficiently
    high-order Gauss-Legendre quadrature is used.

    Parameters
    ----------
    n_elem : int
        Number of FV cells.
    geometry : str
        One of:
            'AXIAL_FLOW_CYLINDER',
            'RADIAL_FLOW_CYLINDER_SHELL',
            'AXIAL_FLOW_FRUSTUM'.
    radial_inner_radius : float, optional
        Inner radius for radial/frustum geometries.
    radial_outer_radius : float, optional
        Outer radius for radial/frustum geometries.
    xi_faces : np.ndarray, optional
        Normalized cell face coordinates in [0, 1], shape (n_elem + 1,).
        Defaults to the equidistant grid; pass the result of
        get_normalized_grid_faces for non-equidistant (e.g. equivolume)
        grids so that the initial condition is the exact geometry-weighted
        average on the actual cells.
    kwargs : dict
        Initial profile parameters.

    Returns
    -------
    np.ndarray
        Geometry-weighted FV cell averages.
    """

    if geometry not in (
            'AXIAL_FLOW_CYLINDER',
            'RADIAL_FLOW_CYLINDER_SHELL',
            'AXIAL_FLOW_FRUSTUM'):
        raise ValueError(
            f"Unknown geometry: {geometry}"
        )

    # Computational cells in xi in [0, 1] (equidistant unless faces are given).
    if xi_faces is None:
        xi_left = np.arange(n_elem, dtype=float) / n_elem
        xi_right = (np.arange(n_elem, dtype=float) + 1.0) / n_elem
    else:
        xi_faces = np.asarray(xi_faces, dtype=float)
        if xi_faces.shape != (n_elem + 1,):
            raise ValueError(
                "xi_faces must have shape (n_elem + 1,)"
            )
        xi_left = xi_faces[:-1]
        xi_right = xi_faces[1:]

    # ------------------------------------------------------------------
    # Geometry weight A(xi), up to an irrelevant constant factor.
    #
    # Axial cylinder:
    #     A = const
    #
    # Radial cylinder shell:
    #     A ~ r
    #
    # Axial frustum:
    #     A ~ r^2
    # ------------------------------------------------------------------

    if geometry == 'AXIAL_FLOW_CYLINDER':
        geometry_power = 0

    elif geometry in (
            'RADIAL_FLOW_CYLINDER_SHELL',
            'AXIAL_FLOW_FRUSTUM'):

        if radial_inner_radius is None:
            raise ValueError(
                "radial_inner_radius is required for "
                f"{geometry}."
            )

        if radial_outer_radius is None:
            raise ValueError(
                "radial_outer_radius is required for "
                f"{geometry}."
            )

        delta_r = (
            radial_outer_radius
            - radial_inner_radius
        )

        geometry_power = (
            1
            if geometry == 'RADIAL_FLOW_CYLINDER_SHELL'
            else 2
        )

    # ------------------------------------------------------------------
    # Custom profile: use high-order quadrature.
    # ------------------------------------------------------------------

    if 'init_profile' in kwargs:

        profile = get_initial_profile(**kwargs)
        quadrature_order = kwargs.get(
            'init_quadrature_order',
            32
        )

        nodes, weights = np.polynomial.legendre.leggauss(
            quadrature_order
        )

        centers = 0.5 * (xi_left + xi_right)
        half_widths = 0.5 * (
            xi_right - xi_left
        )

        xi = (
            centers[:, None]
            + half_widths[:, None] * nodes[None, :]
        )

        c = profile(xi)

        if geometry_power == 0:
            area = np.ones_like(xi)

        elif geometry_power == 1:
            r = (
                radial_inner_radius
                + delta_r * xi
            )
            area = r

        else:
            r = (
                radial_inner_radius
                + delta_r * xi
            )
            area = r ** 2

        numerator = half_widths * np.sum(
            weights[None, :] * area * c,
            axis=1
        )

        denominator = half_widths * np.sum(
            weights[None, :] * area,
            axis=1
        )

        return numerator / denominator

    # ------------------------------------------------------------------
    # Default Gaussian profile.
    #
    # c(xi) =
    #     amplitude * exp(-(xi-center)^2/(2 sigma^2))
    #
    # We analytically integrate
    #
    #     int A(xi) c(xi) dxi
    #
    # for A = 1, r, or r^2.
    # ------------------------------------------------------------------

    center = kwargs.get('init_center', 0.5)
    stddev = kwargs.get('init_stddev', 0.05)
    amplitude = kwargs.get('init_amplitude', 1.0)

    sqrt_2_sigma = np.sqrt(2.0) * stddev

    def I0(xi):
        """Integral of c(xi)."""
        return (
            amplitude
            * stddev
            * np.sqrt(np.pi / 2.0)
            * erf(
                (xi - center) / sqrt_2_sigma
            )
        )

    def I1(xi):
        """Integral of xi*c(xi)."""
        gaussian = np.exp(
            -(xi - center) ** 2
            / (2.0 * stddev ** 2)
        )

        return amplitude * (
            center
            * stddev
            * np.sqrt(np.pi / 2.0)
            * erf(
                (xi - center) / sqrt_2_sigma
            )
            - stddev ** 2 * gaussian
        )

    def I2(xi):
        """Integral of xi^2*c(xi)."""
        gaussian = np.exp(
            -(xi - center) ** 2
            / (2.0 * stddev ** 2)
        )

        erf_term = erf(
            (xi - center) / sqrt_2_sigma
        )

        return amplitude * (
            (
                center ** 2
                + stddev ** 2
            )
            * stddev
            * np.sqrt(np.pi / 2.0)
            * erf_term
            - stddev ** 2
            * (
                xi + center
            )
            * gaussian
        )

    # Differences of the Gaussian moments over each cell.
    C0 = I0(xi_right) - I0(xi_left)

    if geometry_power >= 1:
        C1 = I1(xi_right) - I1(xi_left)

    if geometry_power >= 2:
        C2 = I2(xi_right) - I2(xi_left)

    # ------------------------------------------------------------------
    # Geometry = constant:
    #
    #     A = const
    #
    # integral A*c dxi / integral A dxi
    # ------------------------------------------------------------------

    if geometry_power == 0:

        denominator = (
            xi_right - xi_left
        )

        numerator = C0

    # ------------------------------------------------------------------
    # Radial cylindrical shell:
    #
    #     A ~ r
    #       = r_inner + delta_r*xi
    #
    # integral A*c dxi
    # =
    # r_inner * I0
    # + delta_r * I1
    # ------------------------------------------------------------------

    elif geometry_power == 1:

        denominator = (
            radial_inner_radius
            * (xi_right - xi_left)
            + 0.5
            * delta_r
            * (
                xi_right ** 2
                - xi_left ** 2
            )
        )

        numerator = (
            radial_inner_radius * C0
            + delta_r * C1
        )

    # ------------------------------------------------------------------
    # Axial frustum:
    #
    #     A ~ r^2
    #       = (r_inner + delta_r*xi)^2
    #
    #     = r_inner^2
    #       + 2*r_inner*delta_r*xi
    #       + delta_r^2*xi^2
    # ------------------------------------------------------------------

    else:

        denominator = (
            radial_inner_radius ** 2
            * (
                xi_right - xi_left
            )
            + radial_inner_radius
            * delta_r
            * (
                xi_right ** 2
                - xi_left ** 2
            )
            + delta_r ** 2 / 3.0
            * (
                xi_right ** 3
                - xi_left ** 3
            )
        )

        numerator = (
            radial_inner_radius ** 2 * C0
            + 2.0
            * radial_inner_radius
            * delta_r
            * C1
            + delta_r ** 2 * C2
        )

    return numerator / denominator


def get_grid_coordinates(spatial_method_bulk, n_elem):
    """Return the normalized spatial discretization grid on [0, 1].

    Parameters
    ----------
    spatial_method_bulk : int
        Polynomial degree (DG) or 0 (FV).
    n_elem : int
        Number of DG elements or FV cells.

    Returns
    -------
    np.array
        DG: LGL nodes of all elements (n_elem * (polyDeg + 1) points, element
        interfaces are duplicated as in the CADET-Core DG state vector);
        FV: cell centers (n_elem points).
    """
    h = 1.0 / n_elem

    if spatial_method_bulk == 0:
        return (np.arange(n_elem) + 0.5) * h

    nodes, _ = LGL_NodesWeights(spatial_method_bulk)
    element_offsets = np.arange(n_elem)[:, None] * h
    return (element_offsets + 0.5 * (nodes[None, :] + 1.0) * h).reshape(-1)


def get_model(
        spatial_method_bulk,
        refinement=1,
        **kwargs):
    """Create the pure transport initial value problem configuration.

    Parameters
    ----------
    spatial_method_bulk : int
        Polynomial degree (DG) or 0 (FV).
    refinement : int
        Spatial refinement factor, the number of elements/cells is
        8 * refinement.
    column_geometry : string
        'AXIAL_FLOW_CYLINDER', 'RADIAL_FLOW_CYLINDER_SHELL' or
        'AXIAL_FLOW_FRUSTUM'.
    advection : bool
        Switches advection on/off (off sets the flow rate to zero, i.e. pure
        diffusion with vanishing Danckwerts boundary fluxes).
    dispersion : bool
        Switches axial dispersion on/off.
    kwargs : dict
        col_dispersion (5.75e-8), porosity (0.37), t_end (30.0),
        n_solution_times (601), idas_reftol (1e-12),
        write_solution_bulk (1), initial profile parameters
        (see get_initial_profile) and axRefinement, weno_order,
        POLYNOMIAL_INTEGRATION_TYPE as in setting_Col1D_lin_1comp_benchmark1.

    Returns
    -------
    addict.Dict
        CADET configuration.
    """

    column_geometry = kwargs.get('column_geometry', 'AXIAL_FLOW_CYLINDER')
    advection = kwargs.get('advection', True)
    dispersion = kwargs.get('dispersion', True)
    grid_type = kwargs.get('grid_type', 'equidistant')

    axNElem = int(8 * kwargs.get('axRefinement', refinement))

    # residence time L / v = 0.014 / 0.000575 ~ 24.3s for the default t_end
    flow_rate = 6.e-05 if advection else 0.0
    col_dispersion = kwargs.get('col_dispersion', 5.75e-08) if dispersion else 0.0
    if not (advection or dispersion):
        raise ValueError(
            "At least one of advection and dispersion must be switched on."
        )

    t_end = kwargs.get('t_end', 30.0)

    model = Dict()

    model.input.model.nunits = 3

    # Flow sheet
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        0.e+00, 1.e+00, -1.e+00, -1.e+00, flow_rate,
        1.e+00, 2.e+00, -1.e+00, -1.e+00, flow_rate
        ]
    model.input.model.connections.switch_000.section = 0

    #%% Column unit
    column = Dict()
    if column_geometry == 'AXIAL_FLOW_CYLINDER':
        column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    elif column_geometry == 'RADIAL_FLOW_CYLINDER_SHELL':
        column.UNIT_TYPE = 'RADIAL_COLUMN_MODEL_1D'
    elif column_geometry == 'AXIAL_FLOW_FRUSTUM':
        column.UNIT_TYPE = 'FRUSTUM_COLUMN_MODEL_1D'
    else:
        raise ValueError(f"Unknown column geometry: {column_geometry}")
    column.geometry = column_geometry
    column.update(get_column_geometry_configuration(column_geometry))
    column.forward_flow = 1

    column.ncomp = 1
    column.col_dispersion = col_dispersion
    column.col_porosity = kwargs.get('porosity', 0.37)
    column.total_porosity = kwargs.get('porosity', 0.37)

    # pure bulk transport, no particles
    column.npartype = 0

    if spatial_method_bulk > 0:
        if grid_type != 'equidistant':
            raise ValueError(
                "grid_type='" + grid_type + "' is only supported for the FV "
                "spatial method (spatial_method_bulk=0)."
            )
        column.discretization.SPATIAL_METHOD = "DG"
        column.discretization.POLYNOMIAL_INTEGRATION_TYPE = kwargs.get('POLYNOMIAL_INTEGRATION_TYPE', 0)
        column.discretization.POLYDEG = spatial_method_bulk
        column.discretization.NELEM = axNElem
    elif spatial_method_bulk == 0:
        column.discretization.SPATIAL_METHOD = "FV"
        column.discretization.NCOL = axNElem
        column.discretization.RECONSTRUCTION = 'WENO'
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = kwargs.get('weno_order', 3)
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10
        column.discretization.SCHUR_SAFETY = 1.0e-8
    if spatial_method_bulk >= 0:
        column.discretization.USE_ANALYTIC_JACOBIAN = 1

    # Initial condition.
    #
    # DG DOFs are point values at the LGL nodes, whereas FV DOFs represent
    # cell averages. Using cell-center point values for FV would therefore
    # introduce an O(h^2) initialization error and can mask the higher-order
    # convergence of WENO.
    if spatial_method_bulk == 0:

        radial_inner_radius = None
        radial_outer_radius = None
        if column_geometry == 'RADIAL_FLOW_CYLINDER_SHELL':
            radial_inner_radius = column.col_radius_inner
            radial_outer_radius = column.col_radius_outer
        elif column_geometry == 'AXIAL_FLOW_FRUSTUM':
            radial_inner_radius = column.col_radius_inner
            radial_outer_radius = column.col_radius_outer

        # Normalized cell faces; non-equidistant (equivolume) grids are passed
        # to CADET-Core via GRID_FACES in physical coordinates (radius for the
        # radial shell, axial position for the frustum) and used consistently
        # for the initial geometry-weighted cell averages below.
        xi_faces = get_normalized_grid_faces(
            axNElem,
            geometry=column_geometry,
            grid_type=grid_type,
            radial_inner_radius=radial_inner_radius,
            radial_outer_radius=radial_outer_radius,
        )

        if np.max(np.abs(np.diff(xi_faces) - 1.0 / axNElem)) > 1e-15:
            if column_geometry == 'RADIAL_FLOW_CYLINDER_SHELL':
                phys_faces = (
                    radial_inner_radius
                    + (radial_outer_radius - radial_inner_radius) * xi_faces
                )
                phys_faces[0] = radial_inner_radius
                phys_faces[-1] = radial_outer_radius
            else:
                phys_faces = column.col_length * xi_faces
                phys_faces[0] = 0.0
                phys_faces[-1] = column.col_length
            column.discretization.GRID_FACES = phys_faces.tolist()

        column.INIT_STATE = get_initial_cell_averages(
            axNElem,
            geometry=column_geometry,
            radial_inner_radius=radial_inner_radius,
            radial_outer_radius=radial_outer_radius,
            xi_faces=xi_faces,
            **kwargs
        ).tolist()
    else:
        initial_profile = get_initial_profile(**kwargs)
        grid = get_grid_coordinates(spatial_method_bulk, axNElem)
        column.INIT_STATE = np.asarray(
            initial_profile(grid),
            dtype=float
        ).tolist()

    model.input.model.unit_001 = column
    # model.input.model.unit_001.init_c = [0.0]

    #%% time integration parameters
    if spatial_method_bulk >= 0:
        # non-linear solver
        model.input.model.solver.gs_type = 1
        model.input.model.solver.max_krylov = 0
        model.input.model.solver.max_restarts = 10
        model.input.model.solver.schur_safety = 1e-08
        # time integration / solver specifics
        model.input.solver.consistent_init_mode = 1
        model.input.solver.consistent_init_mode_sens = 3
        model.input.solver.nthreads = 1
        model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_reftol', 1e-12)
        model.input.solver.time_integrator.ALGTOL = kwargs['idas_reftol'] * 100 if 'idas_reftol' in kwargs else 1e-10
        model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
        model.input.solver.time_integrator.MAX_STEPS = 10000
        model.input.solver.time_integrator.RELTOL = kwargs['idas_reftol'] * 100 if 'idas_reftol' in kwargs else 1e-10

    model.input.solver.sections.nsec = 1
    model.input.solver.sections.section_times = [0.0, t_end]
    model.input.solver.user_solution_times = np.linspace(0.0, t_end, 601)
    if 'user_solution_times_unit_state' in kwargs:
        if kwargs['user_solution_times_unit_state'] is not None:
            model.input.solver.user_solution_times_unit_state = kwargs['user_solution_times_unit_state']

    #%% auxiliary units: inlet (constant zero) and outlet
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = 1
    model.input.model.unit_000.sec_000.const_coeff = [0.0]
    model.input.model.unit_000.sec_000.cube_coeff = [0.]
    model.input.model.unit_000.sec_000.lin_coeff = [0.]
    model.input.model.unit_000.sec_000.quad_coeff = [0.]
    model.input.model.unit_000.UNIT_TYPE = 'INLET'

    model.input.model.unit_002.ncomp = 1
    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'

    #%% return data
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_coordinates = kwargs.get('write_solution_bulk', 1)
    model.input['return'].unit_001.write_solution_inlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_001.write_solution_bulk = kwargs.get('write_solution_bulk', 1)
    model.input['return'].unit_002.write_solution_inlet = 0
    model.input['return'].unit_002.write_solution_outlet = 0

    return model


def create_convergence_object(
        config_data=None,
        setting_name='Col1D_pureTransport_1comp_benchmark1',
        unit_id='001',
        disc_idx=0,
        output_path=None,
        idas_abstol=None,
        only_return_name=False,
        base_refinement=1,
        model_kwargs=None,
        grid_type=None,
        **kwargs):
    """Create the Cadet object for one refinement level of the EOC sweep.

    Compatible with the disc_refinement_functions interface of
    bench_func.run_convergence_analysis (see module docstring for usage).
    In contrast to bench_func.create_object_from_config, the configuration is
    rebuilt from scratch via get_model for every refinement level, since the
    nodal initial condition depends on the spatial grid. config_data is
    therefore ignored.

    Parameters
    ----------
    config_data : dict
        Ignored, exists for interface compatibility.
    setting_name : string
        Name prefix of the simulation files.
    unit_id : string
        Unit ID of the refined column unit, must be '001' for this setting.
    disc_idx : int
        Index of the refinement level, starting at zero.
    output_path : string
        Path to the output folder.
    idas_abstol : float
        Absolute time integration tolerance, mapped to get_model's
        idas_reftol if that is not given in model_kwargs.
    only_return_name : bool
        If True, only the simulation file name is returned.
    base_refinement : int
        Refinement of the coarsest level, i.e. level disc_idx has
        8 * base_refinement * 2^disc_idx elements/cells.
    model_kwargs : dict
        Keyword arguments passed to get_model, must contain
        spatial_method_bulk.
    grid_type : str, optional
        'equidistant' or 'equivolume' (see get_normalized_grid_faces);
        overrides model_kwargs['grid_type'] if given. Equivolume grids are
        non-equidistant for the radial shell and frustum geometries (CADET
        receives them via GRID_FACES) and nest under the dyadic refinement
        of this sweep, so the fine reference can be restricted exactly.

    Returns
    -------
    Cadet object or string
        Cadet object with saved h5 file, or the file name if only_return_name.
    """
    if unit_id != '001':
        raise ValueError(
            "This setting defines the column as unit_001, got unit_id=" + unit_id
        )

    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    if 'spatial_method_bulk' not in model_kwargs:
        raise ValueError(
            "model_kwargs must specify spatial_method_bulk."
        )
    if grid_type is not None:
        model_kwargs['grid_type'] = grid_type
    if idas_abstol is not None:
        model_kwargs.setdefault('idas_reftol', idas_abstol)

    config = get_model(refinement=base_refinement * 2 ** disc_idx, **model_kwargs, **kwargs)

    disc = config['input']['model']['unit_' + unit_id]['discretization']
    if disc['SPATIAL_METHOD'] == "DG":
        config_name = generate_1D_name(setting_name, disc['POLYDEG'], disc['NELEM'])
    else:
        config_name = generate_1D_name(setting_name, 0, disc['NCOL'])

    model = Cadet()
    model.root.input = copy.deepcopy(config['input'])

    if output_path is not None:
        model.filename = str(output_path) + '/' + config_name
        if only_return_name:
            return model.filename
        model.save()

    return model
