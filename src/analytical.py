# -*- coding: utf-8 -*-
"""

Analytical reference solutions for the 1D column pure-transport benchmarks
used in the column geometry convergence studies, see
scripts/column_geometries.py and
src/benchmark_models/setting_Col1D_pureTransport_1comp_benchmark1.py.

Model
-----
CADET-Core solves the variable cross section convection dispersion equation in
conservative form on the transport coordinate x in [0, L] (L = BED_LENGTH),

    w(x) dc/dt + d/dx ( q c - D w(x) dc/dx ) = 0,

where w is the (normalized) flow-directional cross section area and q the
constant interstitial volumetric flux. Both are geometry specific:

    geometry                     | w(x)    | q
    -----------------------------|---------|------------------
    AXIAL_FLOW_CYLINDER          | 1       | Q / (eps A)
    RADIAL_FLOW_CYLINDER_SHELL   | rho(x)  | Q / (2 pi H eps)
    AXIAL_FLOW_FRUSTUM           | r(x)^2  | Q / (pi eps)

With the linear "radius" s(x) = s_0 + m x, m = (s_1 - s_0) / L, where

    RADIAL_FLOW_CYLINDER_SHELL: s_0 = inner radius, s_1 = outer radius
                                (rho = s, and m = 1 since L = s_1 - s_0)
    AXIAL_FLOW_FRUSTUM:         s_0 = large end radius, s_1 = small end radius

all three cases are covered by w(x) = s(x)^p with the geometry weight power
p = 0, 1, 2, respectively. The local interstitial velocity is q / w(x).

CADET-Core applies Danckwerts boundary conditions, which for forward flow read

    q c(0, t) - D w(0) dc/dx(0, t) = q c_in(t),      dc/dx(L, t) = 0,

i.e. the inlet face carries the purely convective influx q c_in and the outlet
is a natural ("do nothing") outflow. This is how both the FV kernels
(RadialConvectionDispersionKernelFV.hpp) and the variable cross section DG
operator (ConvectionDispersionOperatorDG.cpp, computeNumericalFluxes) treat the
two boundary faces.

The benchmark is an initial value problem: c_in = 0 and the column is
initialized with a smooth profile c_0 in the normalized coordinate xi = x / L,
by default a Gaussian bump.

Which cases have an analytical solution
---------------------------------------
Pure advection (D = 0), all three geometries:
    Exact and elementary. The equation reduces to w c_t + q c_x = 0, i.e. c is
    constant along the characteristics of the volume coordinate
    V(x) = int_0^x w ds, which satisfy V(x(t)) = V(x(0)) + q t. Hence

        c(x, t) = c_0(xi_0),   V(x_0) = V(x) - q t,   xi_0 = x_0 / L,

    and c = c_in = 0 wherever the characteristic enters through the inlet.

Pure dispersion (q = 0), all three geometries:
    Exact Sturm-Liouville eigenfunction expansion. For q = 0 both Danckwerts
    conditions degenerate to zero flux, and separation of variables in the
    s coordinate gives

        c(x, t) = sum_n a_n phi_n(s(x)) exp(-D m^2 kappa_n^2 t),

    with (s^p phi')' + kappa^2 s^p phi = 0 and phi'(s_a) = phi'(s_b) = 0.
    The eigenfunctions are elementary or classical special functions,

        p = 0: cos(kappa x)                       (Fourier cosine series)
        p = 1: J_0, Y_0 combination               (annulus diffusion)
        p = 2: (sin(kappa s), cos(kappa s)) / s   (spherical shell diffusion)

    all of which are covered by

        phi(s) = s^-nu [ Y_(nu+1)(kappa s_a) J_nu(kappa s)
                         - J_(nu+1)(kappa s_a) Y_nu(kappa s) ],

    with nu = (p - 1) / 2, where the eigenvalues kappa_n are the roots of

        J_(nu+1)(kappa s_a) Y_(nu+1)(kappa s_b)
        - J_(nu+1)(kappa s_b) Y_(nu+1)(kappa s_a).

    The expansion coefficients follow from the w-weighted scalar products
    a_n = <c_0, phi_n>_w / <phi_n, phi_n>_w and are evaluated with composite
    Gauss-Legendre quadrature, i.e. to machine precision for smooth c_0. The
    series converges spectrally; for the default Gaussian the coefficients
    decay like exp(-(kappa_n sigma)^2 / 2).

Advection *and* dispersion: no usable analytical solution.
    The Danckwerts problem is not self-adjoint and its eigenfunctions carry the
    factor exp(int q / (2 D w) dx). For the axial cylinder this yields the
    classical Brenner series, for the radial shell the eigenfunctions are
    s^nu J_(+-nu)(kappa s) with nu = q / (2 D) ~ 900, and for the frustum the
    resulting Sturm-Liouville weight exp(q / (D m s)) is not of any standard
    type. In all three cases the expansion is conditioned like exp(Pe) with the
    column Peclet number Pe = u L / D = 140 for this benchmark, i.e. the
    eigenfunctions span ~e^140 in magnitude across the column and the series
    cannot be summed to a useful accuracy in double precision. The combined
    advection-dispersion settings therefore keep the self-convergence reference
    (finest discretization) in the EOC study.

Usage
-----
Regenerate the reference files consumed by scripts/column_geometries.py:

    python -m src.analytical

This writes one h5 file per analytically available setting to
data/CADET-Verification_reference/. The references are stored in the layout of
a CADET simulation output (a high order DG representation of the analytical
solution) so that they can be passed to bench_func.run_convergence_analysis
via ref_files and are handled by the existing bulk error machinery:
convergence.calculate_bulk_convergence_errors integrates the reference
polynomial over each FV cell with the geometry weight w (which gives the exact
geometry weighted cell averages that the FV degrees of freedom represent) and
interpolates it onto the Gauss points of each DG simulation grid.

"""

import os

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, yv


GEOMETRIES = (
    'AXIAL_FLOW_CYLINDER',
    'RADIAL_FLOW_CYLINDER_SHELL',
    'AXIAL_FLOW_FRUSTUM',
)

# Representation of the written reference files: a DG solution of polynomial
# degree REFERENCE_POLYDEG on REFERENCE_NELEM uniform elements, holding the
# nodal (LGL) point values of the analytical solution. The reference therefore
# carries an interpolation error of O(h^(POLYDEG + 1)); the defaults keep it at
# machine precision for the benchmark's profiles, see
# reference_representation_error.
REFERENCE_POLYDEG = 8
REFERENCE_NELEM = 2048

# Default number of eigenmodes of the pure dispersion expansion, and the
# resolution of the composite Gauss-Legendre rule used for the expansion
# coefficients (n_sub = QUADRATURE_SUBINTERVALS_PER_MODE * n_modes). The
# defaults keep the reconstruction of the initial profile (the worst case,
# since no mode is damped yet) at ~2e-14, see DispersionExpansion.
N_EIGENMODES = 120
QUADRATURE_POINTS_PER_SUBINTERVAL = 20
QUADRATURE_SUBINTERVALS_PER_MODE = 8

# Solution times of the bulk solution at which the references are generated.
# These must match user_solution_times_unit_state of the convergence study,
# since the EOC analysis compares the reference and the simulations at the same
# solution time index; src/column_geometries.py passes its own value.
DEFAULT_SOLUTION_TIMES = (5.0,)

# Default reference data directory, relative to the project root.
REFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'CADET-Verification_reference'
)


# %% Reading model parameters from a CADET configuration


def _lookup(config, key, default=None):
    """Return config[key] from a CADET configuration, ignoring key case.

    Parameters
    ----------
    config : dict
        Configuration (sub)group, e.g. an addict.Dict as returned by the
        benchmark model builders or a group of a loaded h5 file.
    key : string
        Field name.
    default : object
        Returned if the field does not exist.

    Returns
    -------
    object
        Field value or default.
    """
    for candidate in (key.lower(), key.upper(), key):
        if candidate in config:
            return config[candidate]
    return default


def _scalar(value):
    """Return a python float from a scalar-like CADET field."""
    return float(np.asarray(value).ravel()[0])


def _string(value):
    """Return an upper case python string from a string-like CADET field."""
    value = np.asarray(value).ravel()[0]
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).upper()


class ColumnGeometry:
    """Geometry of a 1D column in the unified w(x) = s(x)^p description.

    Attributes
    ----------
    kind : string
        One of GEOMETRIES.
    bed_length : float
        Length L of the transport coordinate x in [0, L]. For the radial
        cylinder shell this is the shell thickness, i.e. x = rho - s_0.
    weight_power : int
        Geometry weight power p, i.e. w(x) = s(x)^p.
    s_0, s_1 : float
        Linear radius s(x) = s_0 + m x at the inlet (x = 0) and the outlet
        (x = L) of the transport coordinate. Both are 1 for the axial
        cylinder, whose geometry weight is constant.
    slope : float
        m = (s_1 - s_0) / L.
    """

    def __init__(self, kind, bed_length, weight_power, s_0, s_1):
        if kind not in GEOMETRIES:
            raise ValueError(f"Unknown geometry: {kind}")
        self.kind = kind
        self.bed_length = float(bed_length)
        self.weight_power = int(weight_power)
        self.s_0 = float(s_0)
        self.s_1 = float(s_1)
        self.slope = (self.s_1 - self.s_0) / self.bed_length

    def __repr__(self):
        return (
            f"ColumnGeometry({self.kind}, bed_length={self.bed_length}, "
            f"weight_power={self.weight_power}, s_0={self.s_0}, "
            f"s_1={self.s_1})"
        )

    def radius(self, xi):
        """Return the linear radius s at the normalized coordinate xi = x / L."""
        return self.s_0 + (self.s_1 - self.s_0) * np.asarray(xi, dtype=float)

    def weight(self, xi):
        """Return the geometry weight w = s^p at xi = x / L."""
        if self.weight_power == 0:
            return np.ones_like(np.asarray(xi, dtype=float))
        return self.radius(xi) ** self.weight_power

    def normalized_coordinate(self, s):
        """Return xi = x / L for a linear radius s (inverse of radius)."""
        if self.s_1 == self.s_0:
            raise ValueError(
                "The linear radius is constant, xi cannot be recovered from it."
            )
        return (np.asarray(s, dtype=float) - self.s_0) / (self.s_1 - self.s_0)

    def mirrored(self):
        """Return the same geometry with the two ends swapped.

        Used to express flow towards decreasing x as flow towards increasing x
        on the mirrored geometry, see reversed_flow_solution.
        """
        return ColumnGeometry(self.kind, self.bed_length, self.weight_power,
                              self.s_1, self.s_0)

    def flux_coefficient(self, flow_rate, porosity, cross_section_area=None,
                         cylinder_height=None):
        """Return the constant interstitial volumetric flux q of the geometry.

        Parameters
        ----------
        flow_rate : float
            Volumetric flow rate Q into the column.
        porosity : float
            Column (interstitial) porosity.
        cross_section_area : float
            Cross section area, required for AXIAL_FLOW_CYLINDER.
        cylinder_height : float
            Cylinder height H, required for RADIAL_FLOW_CYLINDER_SHELL.

        Returns
        -------
        float
            q, such that the local interstitial velocity is q / w(x).
        """
        if self.kind == 'AXIAL_FLOW_CYLINDER':
            return flow_rate / (porosity * cross_section_area)
        if self.kind == 'RADIAL_FLOW_CYLINDER_SHELL':
            return flow_rate / (2.0 * np.pi * cylinder_height * porosity)
        return flow_rate / (np.pi * porosity)


def flows_towards_increasing_x(geometry_kind, forward_flow):
    """Return whether the flow runs towards increasing x, i.e. increasing index.

    Mirrors the convention of CADET-Core (see
    VariableCrossSectionConvectionDispersionOperatorBaseDG::internalForwardFlow and
    RadialConvectionDispersionOperatorBaseFV::configure): the transport coordinate x
    runs along the flow path for the axial cylinder and the frustum, so forward flow
    means increasing x. For the radial cylinder shell, x is the radius itself and
    increases from the inner to the outer radius, whereas the default (forward) flow
    direction runs from the larger to the smaller radius, i.e. towards decreasing x.

    Parameters
    ----------
    geometry_kind : string
        One of GEOMETRIES.
    forward_flow : bool
        Value of the FORWARD_FLOW field.

    Returns
    -------
    bool
        True if the flow runs towards increasing x.
    """
    if geometry_kind == 'RADIAL_FLOW_CYLINDER_SHELL':
        return not forward_flow
    return bool(forward_flow)


def geometry_from_unit_config(unit):
    """Create a ColumnGeometry from a CADET column unit configuration.

    Uses the same fields and the same derivation of the end radii as
    CADET-Core, see ConvectionDispersionOperatorDG.cpp,
    VariableCrossSectionConvectionDispersionOperatorBaseDG::configureModelDiscretization.

    Parameters
    ----------
    unit : dict
        Column unit configuration, i.e. input/model/unit_XXX.

    Returns
    -------
    ColumnGeometry
    """
    kind = _string(_lookup(unit, 'GEOMETRY'))
    bed_length = _scalar(_lookup(unit, 'BED_LENGTH'))

    if kind == 'AXIAL_FLOW_CYLINDER':
        return ColumnGeometry(kind, bed_length, 0, 1.0, 1.0)

    if kind == 'RADIAL_FLOW_CYLINDER_SHELL':
        height = _scalar(_lookup(unit, 'CYLINDER_HEIGHT'))
        area_outer = _scalar(_lookup(unit, 'CROSS_SECTION_AREA_OUTER'))
        # A = 2 pi r H => r = A / (2 pi H); the inner radius follows from the
        # shell thickness BED_LENGTH.
        radius_outer = area_outer / (2.0 * np.pi * height)
        radius_inner = radius_outer - bed_length
        return ColumnGeometry(kind, bed_length, 1, radius_inner, radius_outer)

    if kind == 'AXIAL_FLOW_FRUSTUM':
        area_large = _scalar(_lookup(unit, 'CROSS_SECTION_AREA_LARGE_END'))
        area_small = _scalar(_lookup(unit, 'CROSS_SECTION_AREA_SMALL_END'))
        # A = pi r^2 => r = sqrt(A / pi); the large end is at x = 0.
        radius_large = np.sqrt(area_large / np.pi)
        radius_small = np.sqrt(area_small / np.pi)
        return ColumnGeometry(kind, bed_length, 2, radius_large, radius_small)

    raise ValueError(f"Unknown geometry: {kind}")


def transport_parameters_from_config(config, unit_id='001'):
    """Extract the transport parameters of a column unit from a CADET config.

    Parameters
    ----------
    config : dict
        Full CADET configuration, i.e. containing an 'input' group.
    unit_id : string
        Unit ID of the column (000-999).

    Returns
    -------
    dict
        geometry (ColumnGeometry), flux_coefficient q, col_dispersion D,
        flow_rate, porosity and forward_flow.
    """
    model = _lookup(_lookup(config, 'input'), 'model')
    unit = _lookup(model, 'unit_' + unit_id)

    geometry = geometry_from_unit_config(unit)
    porosity = _scalar(_lookup(unit, 'COL_POROSITY'))
    dispersion = _scalar(_lookup(unit, 'COL_DISPERSION'))
    forward_flow = bool(_scalar(_lookup(unit, 'FORWARD_FLOW', 1)))

    # Volumetric flow rate into the column unit, read from the (single switch)
    # connections matrix, whose rows are
    # [unit_from, unit_to, port_from, port_to, flow_rate].
    connections = np.asarray(
        _lookup(
            _lookup(_lookup(model, 'connections'), 'switch_000'),
            'CONNECTIONS'
        ),
        dtype=float
    ).reshape(-1, 5)
    inflow = connections[connections[:, 1] == float(int(unit_id))]
    if inflow.shape[0] != 1:
        raise ValueError(
            f"Expected exactly one connection into unit_{unit_id}, "
            f"found {inflow.shape[0]}."
        )
    flow_rate = float(inflow[0, 4])

    return {
        'geometry': geometry,
        'flux_coefficient': geometry.flux_coefficient(
            flow_rate, porosity,
            cross_section_area=_scalar(
                _lookup(unit, 'CROSS_SECTION_AREA', np.nan)
            ),
            cylinder_height=_scalar(
                _lookup(unit, 'CYLINDER_HEIGHT', np.nan)
            ),
        ),
        'col_dispersion': dispersion,
        'flow_rate': flow_rate,
        'porosity': porosity,
        'forward_flow': forward_flow,
    }


# %% Initial condition


def gaussian_bump(center=0.5, stddev=0.05, amplitude=1.0):
    """Return the default initial profile c_0(xi) of the benchmark.

    Parameters
    ----------
    center, stddev, amplitude : float
        Gaussian parameters in the normalized coordinate xi = x / L.

    Returns
    -------
    callable
        c_0 : xi (np.array in [0, 1]) -> concentration (np.array).
    """
    def profile(xi):
        return amplitude * np.exp(
            -(np.asarray(xi, dtype=float) - center) ** 2 / (2.0 * stddev ** 2)
        )

    return profile


# %% Pure advection: exact characteristics


def advection_solution(xi, t, geometry, flux_coefficient, init_profile=None):
    """Exact solution of the pure advection problem w c_t + q c_x = 0.

    The characteristics of the volume coordinate V(x) = int_0^x w(s) ds are
    straight lines, V(x(t)) = V(x(0)) + q t, along which c is constant. With
    w = s^p and s(x) = s_0 + m x this gives, for m != 0,

        s(x_0)^(p+1) = s(x)^(p+1) - (p + 1) m q t,

    and x_0 = x - q t / s_0^p for the constant cross section case m = 0.
    Points whose characteristic originates outside the column are set to the
    (zero) inlet concentration.

    Parameters
    ----------
    xi : array_like
        Normalized coordinates x / L in [0, 1].
    t : float
        Time.
    geometry : ColumnGeometry
        Column geometry.
    flux_coefficient : float
        Interstitial volumetric flux q.
    init_profile : callable
        Initial profile c_0(xi), defaults to the benchmark's Gaussian bump.

    Returns
    -------
    np.ndarray
        Concentration at xi, same shape as xi.
    """
    if init_profile is None:
        init_profile = gaussian_bump()

    xi = np.asarray(xi, dtype=float)
    length = geometry.bed_length
    power = geometry.weight_power
    slope = geometry.slope

    if slope == 0.0:
        # Constant cross section: uniform transport velocity q / s_0^p.
        xi_foot = xi - flux_coefficient * t / (geometry.s_0 ** power) / length
    else:
        s_end = geometry.radius(xi)
        s_foot_powered = (
            s_end ** (power + 1)
            - (power + 1) * slope * flux_coefficient * t
        )
        # A non-positive value means the characteristic reaches the apex of the
        # geometry, which cannot happen inside a valid column; guard the root.
        s_foot = np.where(
            s_foot_powered > 0.0,
            np.abs(s_foot_powered) ** (1.0 / (power + 1)),
            0.0
        )
        xi_foot = np.where(
            s_foot_powered > 0.0,
            geometry.normalized_coordinate(s_foot),
            -1.0
        )

    inside = xi_foot >= 0.0
    return np.where(inside, init_profile(np.clip(xi_foot, 0.0, None)), 0.0)


# %% Pure dispersion: Sturm-Liouville eigenfunction expansion


def _composite_gauss_legendre(a, b, n_sub, n_gauss):
    """Return nodes and weights of a composite Gauss-Legendre rule on [a, b].

    Parameters
    ----------
    a, b : float
        Interval bounds.
    n_sub : int
        Number of equally sized subintervals.
    n_gauss : int
        Number of Gauss-Legendre points per subinterval.

    Returns
    -------
    np.ndarray
        Quadrature nodes, shape (n_sub * n_gauss,).
    np.ndarray
        Quadrature weights, shape (n_sub * n_gauss,).
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    edges = np.linspace(a, b, n_sub + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    half_widths = 0.5 * np.diff(edges)
    points = centers[:, None] + half_widths[:, None] * nodes[None, :]
    quad_weights = half_widths[:, None] * weights[None, :]
    return points.ravel(), quad_weights.ravel()


def _neumann_eigenvalue_residual(kappa, s_a, s_b, order):
    """Return the zero-flux eigenvalue condition of the s^p weighted problem.

    The condition is the 2x2 determinant of the two boundary conditions
    phi'(s_a) = phi'(s_b) = 0, which, using
    d/ds [s^-nu Z_nu(kappa s)] = -kappa s^-nu Z_(nu+1)(kappa s) for both
    Bessel functions Z = J, Y, reduces to

        J_(nu+1)(kappa s_a) Y_(nu+1)(kappa s_b)
        - J_(nu+1)(kappa s_b) Y_(nu+1)(kappa s_a) = 0.

    It is scaled by kappa sqrt(s_a s_b) pi / 2, which tends to
    sin(kappa (s_b - s_a)) for large kappa, so that the residual stays O(1).

    Parameters
    ----------
    kappa : array_like
        Eigenvalue candidates.
    s_a, s_b : float
        Interval bounds of the linear radius, s_a < s_b.
    order : float
        Bessel order nu + 1 = (p + 1) / 2.

    Returns
    -------
    np.ndarray
        Scaled residual, same shape as kappa.
    """
    kappa = np.asarray(kappa, dtype=float)
    scale = 0.5 * np.pi * kappa * np.sqrt(s_a * s_b)
    return scale * (
        jv(order, kappa * s_a) * yv(order, kappa * s_b)
        - jv(order, kappa * s_b) * yv(order, kappa * s_a)
    )


def _neumann_eigenvalues(s_a, s_b, weight_power, n_modes):
    """Return the positive eigenvalues kappa_n of the zero-flux problem.

    For weight_power = 0 the eigenvalues are the exact Fourier values
    kappa_n = n pi / (s_b - s_a). Otherwise the roots of
    _neumann_eigenvalue_residual are bracketed around their asymptotic
    values n pi / (s_b - s_a) and refined with Brent's method.

    Parameters
    ----------
    s_a, s_b : float
        Interval bounds of the linear radius, s_a < s_b.
    weight_power : int
        Geometry weight power p.
    n_modes : int
        Number of positive eigenvalues.

    Returns
    -------
    np.ndarray
        Eigenvalues kappa_1 ... kappa_n_modes in ascending order.
    """
    spacing = np.pi / (s_b - s_a)

    if weight_power == 0:
        return spacing * np.arange(1, n_modes + 1, dtype=float)

    order = 0.5 * (weight_power + 1)

    # The residual behaves like sin(kappa (s_b - s_a)) for kappa s_a >> 1, so
    # each root is bracketed by the asymptotic value +- half a period. Scan the
    # bracket for the sign change to stay robust for the lowest modes, where
    # the asymptotic behaviour is not yet exact.
    eigenvalues = np.zeros(n_modes)
    n_scan = 24

    for mode in range(1, n_modes + 1):
        lower = (mode - 0.5) * spacing
        upper = (mode + 0.5) * spacing
        grid = np.linspace(max(lower, 1e-8 * spacing), upper, n_scan)
        residual = _neumann_eigenvalue_residual(grid, s_a, s_b, order)

        sign_change = np.nonzero(np.sign(residual[:-1]) * np.sign(residual[1:]) < 0.0)[0]
        if sign_change.size == 0:
            raise RuntimeError(
                f"No sign change of the eigenvalue condition in bracket "
                f"[{lower}, {upper}] for mode {mode}."
            )
        # The bracket contains exactly one root asymptotically; pick the one
        # closest to the asymptotic estimate if the scan resolves more.
        estimate = mode * spacing
        best = sign_change[
            np.argmin(np.abs(0.5 * (grid[sign_change] + grid[sign_change + 1]) - estimate))
        ]

        eigenvalues[mode - 1] = brentq(
            lambda k: float(_neumann_eigenvalue_residual(k, s_a, s_b, order)),
            grid[best], grid[best + 1], xtol=1e-15, rtol=8.9e-16, maxiter=200
        )

    return eigenvalues


def _neumann_eigenfunctions(s, kappa, s_a, weight_power):
    """Evaluate the zero-flux eigenfunctions phi_n(s).

    Parameters
    ----------
    s : np.ndarray
        Linear radius values, shape (n_points,).
    kappa : np.ndarray
        Eigenvalues, shape (n_modes,).
    s_a : float
        Lower interval bound of the linear radius (where phi' = 0 is imposed).
    weight_power : int
        Geometry weight power p.

    Returns
    -------
    np.ndarray
        Eigenfunction values, shape (n_modes, n_points).
    """
    s = np.asarray(s, dtype=float)
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))

    if weight_power == 0:
        # cos(kappa (s - s_a)); the shift makes phi'(s_a) = 0 explicit.
        return np.cos(kappa[:, None] * (s[None, :] - s_a))

    nu = 0.5 * (weight_power - 1)
    order = nu + 1.0

    argument = kappa[:, None] * s[None, :]
    boundary = kappa * s_a

    values = (
        yv(order, boundary)[:, None] * jv(nu, argument)
        - jv(order, boundary)[:, None] * yv(nu, argument)
    )
    return values * s[None, :] ** (-nu)


class DispersionExpansion:
    """Eigenfunction expansion of the pure dispersion problem.

    Holds the eigenvalues, the expansion coefficients of the initial profile
    and the decay rates, so that the solution can be evaluated at arbitrary
    coordinates and times.

    Attributes
    ----------
    geometry : ColumnGeometry
        Column geometry.
    col_dispersion : float
        Dispersion coefficient D.
    kappa : np.ndarray
        Eigenvalues, kappa[0] = 0 (the conserved mean).
    coefficients : np.ndarray
        Expansion coefficients a_n.
    decay_rates : np.ndarray
        Decay rates D m^2 kappa_n^2 of the modes.
    initial_condition_error : float
        Maximum deviation of the truncated expansion from the initial profile
        at t = 0, evaluated on a uniform test grid. This is the worst case
        accuracy of the expansion, since all modes are undamped at t = 0.
    """

    def __init__(self, geometry, col_dispersion, init_profile=None,
                 n_modes=N_EIGENMODES,
                 n_gauss=QUADRATURE_POINTS_PER_SUBINTERVAL, n_sub=None):
        if init_profile is None:
            init_profile = gaussian_bump()

        self.geometry = geometry
        self.col_dispersion = float(col_dispersion)
        self.n_modes = int(n_modes)

        power = geometry.weight_power
        if geometry.slope == 0.0:
            # Constant cross section: use the transport coordinate itself, so
            # that s = x, m = 1 and the weight is s^0 = 1.
            s_a, s_b = 0.0, geometry.bed_length
            slope = 1.0
        else:
            s_a = min(geometry.s_0, geometry.s_1)
            s_b = max(geometry.s_0, geometry.s_1)
            slope = geometry.slope

        self._s_a = s_a
        self._s_b = s_b

        if n_sub is None:
            n_sub = max(64, QUADRATURE_SUBINTERVALS_PER_MODE * self.n_modes)

        points, weights = _composite_gauss_legendre(s_a, s_b, n_sub, n_gauss)
        geometry_weight = points ** power if power else np.ones_like(points)

        if geometry.slope == 0.0:
            xi_points = points / geometry.bed_length
        else:
            xi_points = geometry.normalized_coordinate(points)
        c_0 = np.asarray(init_profile(xi_points), dtype=float)

        kappa = np.concatenate((
            [0.0],
            _neumann_eigenvalues(s_a, s_b, power, self.n_modes)
        ))

        phi = np.vstack((
            np.ones((1, points.size)),
            _neumann_eigenfunctions(points, kappa[1:], s_a, power)
        ))

        measure = weights * geometry_weight
        norms = np.einsum('np,p,np->n', phi, measure, phi, optimize=True)
        projections = np.einsum('np,p,p->n', phi, measure, c_0, optimize=True)

        self.kappa = kappa
        self.coefficients = projections / norms
        self.decay_rates = self.col_dispersion * slope ** 2 * kappa ** 2

        xi_test = np.linspace(0.0, 1.0, 501)
        self.initial_condition_error = float(np.max(np.abs(
            self(xi_test, 0.0) - np.asarray(init_profile(xi_test), dtype=float)
        )))

    def __call__(self, xi, t):
        """Evaluate the expansion at normalized coordinates xi and time t.

        Parameters
        ----------
        xi : array_like
            Normalized coordinates x / L in [0, 1].
        t : float
            Time.

        Returns
        -------
        np.ndarray
            Concentration at xi, same shape as xi.
        """
        xi = np.asarray(xi, dtype=float)
        flat = xi.reshape(-1)

        if self.geometry.slope == 0.0:
            s = flat * self.geometry.bed_length
        else:
            s = self.geometry.radius(flat)

        phi = np.vstack((
            np.ones((1, s.size)),
            _neumann_eigenfunctions(s, self.kappa[1:], self._s_a,
                                    self.geometry.weight_power)
        ))
        amplitudes = self.coefficients * np.exp(-self.decay_rates * t)
        return (amplitudes @ phi).reshape(xi.shape)


def dispersion_solution(xi, t, geometry, col_dispersion, init_profile=None,
                        n_modes=N_EIGENMODES):
    """Exact solution of the pure dispersion problem w c_t = D (w c_x)_x.

    Convenience wrapper around DispersionExpansion; build the expansion
    explicitly when evaluating at many times.

    Parameters
    ----------
    xi : array_like
        Normalized coordinates x / L in [0, 1].
    t : float
        Time.
    geometry : ColumnGeometry
        Column geometry.
    col_dispersion : float
        Dispersion coefficient D.
    init_profile : callable
        Initial profile c_0(xi), defaults to the benchmark's Gaussian bump.
    n_modes : int
        Number of eigenmodes.

    Returns
    -------
    np.ndarray
        Concentration at xi, same shape as xi.
    """
    expansion = DispersionExpansion(
        geometry, col_dispersion, init_profile=init_profile, n_modes=n_modes
    )
    return expansion(xi, t)


# %% Dispatch


def reversed_flow_solution(solution_at, xi, t):
    """Evaluate a solution of the mirrored problem at the original coordinates.

    Flow towards decreasing x is the mirror image of flow towards increasing x on
    the geometry with swapped ends, so a solution of the mirrored problem
    (built with ColumnGeometry.mirrored and the mirrored initial profile) is
    evaluated at 1 - xi.

    Parameters
    ----------
    solution_at : callable
        Solution of the mirrored problem, (xi, t) -> concentration.
    xi : array_like
        Normalized coordinates x / L in [0, 1] of the original problem.
    t : float
        Time.

    Returns
    -------
    np.ndarray
        Concentration at xi, same shape as xi.
    """
    xi = np.asarray(xi, dtype=float)
    return solution_at(1.0 - xi, t)


def mirrored_profile(init_profile):
    """Return the initial profile of the mirrored problem, c_0(1 - xi)."""
    def profile(xi):
        return init_profile(1.0 - np.asarray(xi, dtype=float))

    return profile


def analytical_solution(xi, t, geometry, flux_coefficient=0.0,
                        col_dispersion=0.0, init_profile=None,
                        n_modes=N_EIGENMODES, towards_increasing_x=True):
    """Return the analytical solution of the pure transport benchmark.

    Parameters
    ----------
    xi : array_like
        Normalized coordinates x / L in [0, 1].
    t : float
        Time.
    geometry : ColumnGeometry
        Column geometry.
    flux_coefficient : float
        Interstitial volumetric flux q; 0 disables advection.
    col_dispersion : float
        Dispersion coefficient D; 0 disables dispersion.
    init_profile : callable
        Initial profile c_0(xi), defaults to the benchmark's Gaussian bump.
    n_modes : int
        Number of eigenmodes of the pure dispersion expansion.
    towards_increasing_x : bool
        Whether the flow runs towards increasing x (increasing index), see
        flows_towards_increasing_x. Only relevant if advection is active.

    Returns
    -------
    np.ndarray
        Concentration at xi, same shape as xi.

    Raises
    ------
    NotImplementedError
        If both advection and dispersion are active, see the module docstring.
    """
    advection = flux_coefficient != 0.0
    dispersion = col_dispersion != 0.0

    if advection and not towards_increasing_x:
        # Flow towards decreasing x is the mirror image of the same problem on the
        # geometry with swapped ends.
        if init_profile is None:
            init_profile = gaussian_bump()
        return reversed_flow_solution(
            lambda eta, time: analytical_solution(
                eta, time, geometry.mirrored(),
                flux_coefficient=flux_coefficient,
                col_dispersion=col_dispersion,
                init_profile=mirrored_profile(init_profile),
                n_modes=n_modes, towards_increasing_x=True
            ),
            xi, t
        )

    if advection and dispersion:
        raise NotImplementedError(
            "No analytical solution is available for combined advection and "
            "dispersion in a variable cross section column: the Danckwerts "
            "eigenexpansion is conditioned like exp(Pe) and cannot be summed "
            "in double precision for this benchmark (Pe = 140). See the "
            "module docstring of src/analytical.py."
        )

    if not (advection or dispersion):
        raise ValueError(
            "At least one of advection (flux_coefficient) and dispersion "
            "(col_dispersion) must be active."
        )

    if advection:
        return advection_solution(
            xi, t, geometry, flux_coefficient, init_profile=init_profile
        )

    return dispersion_solution(
        xi, t, geometry, col_dispersion, init_profile=init_profile,
        n_modes=n_modes
    )


def analytical_solution_from_config(xi, t, config, unit_id='001',
                                    init_profile=None, n_modes=N_EIGENMODES):
    """Return the analytical solution for a CADET configuration.

    All model parameters (geometry, flow rate, porosity, dispersion) are read
    from the configuration, so that the reference cannot drift away from the
    simulated setting.

    Parameters
    ----------
    xi : array_like
        Normalized coordinates x / L in [0, 1].
    t : float
        Time.
    config : dict
        Full CADET configuration.
    unit_id : string
        Unit ID of the column.
    init_profile : callable
        Initial profile c_0(xi), defaults to the benchmark's Gaussian bump.
    n_modes : int
        Number of eigenmodes of the pure dispersion expansion.

    Returns
    -------
    np.ndarray
        Concentration at xi, same shape as xi.
    """
    parameters = transport_parameters_from_config(config, unit_id=unit_id)
    return analytical_solution(
        xi, t,
        parameters['geometry'],
        flux_coefficient=(
            parameters['flux_coefficient'] if parameters['flow_rate'] else 0.0
        ),
        col_dispersion=parameters['col_dispersion'],
        init_profile=init_profile,
        n_modes=n_modes,
        towards_increasing_x=flows_towards_increasing_x(
            parameters['geometry'].kind, parameters['forward_flow']
        ),
    )


# %% Reference file generation


# Settings of scripts/column_geometries.py for which an analytical solution is
# available, mapping the setting name to the model_kwargs of
# setting_Col1D_pureTransport_1comp_benchmark1.get_model.
#
# The combined advection-dispersion settings (radialDPFR/frustumDPFR) and the
# particle settings (radialLRMP/frustumLRMP) are deliberately absent: no
# analytical solution is available for them, see the module docstring.
REFERENCE_SETTINGS = {
    'radialAdvDPFR_1comp_benchmark1': {
        'column_geometry': 'RADIAL_FLOW_CYLINDER_SHELL',
        'advection': True, 'dispersion': False,
    },
    'radialDispDPFR_1comp_benchmark1': {
        'column_geometry': 'RADIAL_FLOW_CYLINDER_SHELL',
        'advection': False, 'dispersion': True,
    },
    'frustumAdvDPFR_1comp_benchmark1': {
        'column_geometry': 'AXIAL_FLOW_FRUSTUM',
        'advection': True, 'dispersion': False,
    },
    'frustumDispDPFR_1comp_benchmark1': {
        'column_geometry': 'AXIAL_FLOW_FRUSTUM',
        'advection': False, 'dispersion': True,
    },
}


def reference_file_name(setting_name):
    """Return the file name of the analytical reference of a setting."""
    return 'analytical_' + setting_name + '.h5'


def reference_grid(poly_deg=REFERENCE_POLYDEG, n_elem=REFERENCE_NELEM):
    """Return the normalized LGL node coordinates of the reference grid.

    The reference is stored as a DG solution, i.e. as nodal values at the LGL
    nodes of a uniform element grid, with element interfaces duplicated as in
    the CADET-Core DG state vector.

    Parameters
    ----------
    poly_deg : int
        Polynomial degree of the reference representation.
    n_elem : int
        Number of uniform elements.

    Returns
    -------
    np.ndarray
        Node coordinates xi in [0, 1], shape (n_elem * (poly_deg + 1),).
    """
    from src.utility.convergence import LGL_NodesWeights

    nodes, _ = LGL_NodesWeights(poly_deg)
    h = 1.0 / n_elem
    offsets = np.arange(n_elem)[:, None] * h
    return (offsets + 0.5 * (nodes[None, :] + 1.0) * h).reshape(-1)


def reference_representation_error(solution, poly_deg=REFERENCE_POLYDEG,
                                   n_elem=REFERENCE_NELEM, n_test=7):
    """Return the interpolation error of the stored reference representation.

    The reference h5 files hold nodal values of the analytical solution, so the
    error machinery sees the degree poly_deg interpolant of the analytical
    solution rather than the solution itself. This function measures the
    resulting error by comparing the interpolant against the analytical
    solution at n_test additional Gauss points per element, i.e. exactly the
    kind of points at which convergence.calculate_bulk_convergence_errors
    evaluates the reference.

    Parameters
    ----------
    solution : callable
        Analytical solution xi (np.array) -> concentration (np.array).
    poly_deg : int
        Polynomial degree of the reference representation.
    n_elem : int
        Number of uniform elements.
    n_test : int
        Number of test points per element.

    Returns
    -------
    float
        Maximum absolute deviation of the interpolant from the solution.
    """
    from src.utility.convergence import (
        LGL_NodesWeights, barycentric_weights, interpolate_uniform_dg_values
    )

    nodal = np.asarray(solution(reference_grid(poly_deg, n_elem)), dtype=float)

    nodes, _ = LGL_NodesWeights(poly_deg)
    bary = barycentric_weights(poly_deg)

    test_nodes, _ = np.polynomial.legendre.leggauss(n_test)
    h = 1.0 / n_elem
    offsets = np.arange(n_elem)[:, None] * h
    xi_test = (
        offsets + 0.5 * (test_nodes[None, :] + 1.0) * h
    ).reshape(-1)

    interpolated = interpolate_uniform_dg_values(
        nodal[:, None], xi_test, 1.0, poly_deg, n_elem, nodes, bary
    ).reshape(-1)

    return float(np.max(np.abs(interpolated - np.asarray(solution(xi_test)))))


def write_reference_h5(file_name, unit_config, times, values,
                       poly_deg=REFERENCE_POLYDEG, n_elem=REFERENCE_NELEM,
                       meta=None):
    """Write an analytical reference in the layout of a CADET simulation output.

    The file contains the fields that the bulk error machinery of
    src.utility.convergence reads from a reference: the column geometry (to
    reconstruct the geometry weight of the FV degrees of freedom), BED_LENGTH,
    the discretization (to determine the reference's representation) and
    SOLUTION_BULK.

    Parameters
    ----------
    file_name : string
        Path of the h5 file to write.
    unit_config : dict
        Column unit configuration of the simulated setting, i.e.
        input/model/unit_XXX, from which the geometry fields are copied.
    times : array_like
        Solution times of the reference, shape (n_times,).
    values : array_like
        Reference solution, shape (n_times, n_elem * (poly_deg + 1), n_comp).
    poly_deg : int
        Polynomial degree of the reference representation.
    n_elem : int
        Number of uniform elements.
    meta : dict
        Additional provenance information, written to input/meta.

    Returns
    -------
    string
        The file name.
    """
    import h5py

    values = np.asarray(values, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    expected = (times.size, n_elem * (poly_deg + 1))
    if values.shape[:2] != expected:
        raise ValueError(
            f"Expected reference values of shape {expected} + (n_comp,), "
            f"got {values.shape}."
        )

    bed_length = _scalar(_lookup(unit_config, 'BED_LENGTH'))

    os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)

    # Geometry fields that CADET-Core (and _get_fv_bulk_geometry_weighting)
    # needs to reconstruct the cross section area of the column.
    geometry_fields = ['GEOMETRY', 'BED_LENGTH', 'CYLINDER_HEIGHT',
                       'CROSS_SECTION_AREA', 'CROSS_SECTION_AREA_INNER',
                       'CROSS_SECTION_AREA_OUTER',
                       'CROSS_SECTION_AREA_LARGE_END',
                       'CROSS_SECTION_AREA_SMALL_END']
    copied_fields = geometry_fields + ['UNIT_TYPE', 'NCOMP', 'COL_POROSITY',
                                       'COL_DISPERSION', 'FORWARD_FLOW']

    with h5py.File(file_name, 'w') as h5:
        unit = h5.create_group('input/model/unit_001')
        for field in copied_fields:
            value = _lookup(unit_config, field)
            if value is None:
                continue
            if isinstance(value, str):
                unit.create_dataset(field, data=np.bytes_(value))
            else:
                unit.create_dataset(field, data=np.asarray(value))

        disc = h5.create_group('input/model/unit_001/discretization')
        disc.create_dataset('SPATIAL_METHOD', data=np.bytes_('DG'))
        disc.create_dataset('POLYDEG', data=np.int32(poly_deg))
        disc.create_dataset('NELEM', data=np.int32(n_elem))

        info = h5.create_group('input/meta')
        info.create_dataset(
            'NAME', data=np.bytes_(os.path.basename(file_name))
        )
        for key, value in (meta or {}).items():
            if isinstance(value, str):
                info.create_dataset(key, data=np.bytes_(value))
            else:
                info.create_dataset(key, data=np.asarray(value))

        h5.create_dataset('output/solution/SOLUTION_TIMES', data=times)
        h5.create_dataset(
            'output/solution/unit_001/SOLUTION_BULK', data=values
        )
        h5.create_dataset(
            'output/coordinates/unit_001/AXIAL_COORDINATES',
            data=bed_length * reference_grid(poly_deg, n_elem)
        )

    return file_name


def generate_reference(setting_name, model_kwargs, output_dir=REFERENCE_DIR,
                       poly_deg=REFERENCE_POLYDEG, n_elem=REFERENCE_NELEM,
                       n_modes=N_EIGENMODES, verbose=True):
    """Write the analytical reference of one pure-transport setting.

    The CADET configuration of the setting is built with the very same model
    builder that the convergence study uses, and all model parameters (geometry,
    flow rate, porosity, dispersion coefficient, solution times, initial
    profile) are taken from it. The reference therefore cannot drift away from
    the simulated setting.

    Parameters
    ----------
    setting_name : string
        Name of the setting, used for the reference file name.
    model_kwargs : dict
        Keyword arguments of
        setting_Col1D_pureTransport_1comp_benchmark1.get_model, i.e. the same
        arguments that the benchmark configuration passes.
    output_dir : string
        Directory to write the reference to.
    poly_deg, n_elem : int
        Representation of the written reference, see write_reference_h5.
    n_modes : int
        Number of eigenmodes of the pure dispersion expansion.
    verbose : bool
        Print the accuracy diagnostics of the generated reference.

    Returns
    -------
    string
        Name of the written file.
    """
    from src.benchmark_models import (
        setting_Col1D_pureTransport_1comp_benchmark1 as setting
    )

    kwargs = dict(model_kwargs)
    kwargs.pop('spatial_method_bulk', None)
    config = setting.get_model(spatial_method_bulk=3, **kwargs)

    unit_config = config['input']['model']['unit_001']
    parameters = transport_parameters_from_config(config)
    towards_increasing_x = flows_towards_increasing_x(
        parameters['geometry'].kind, parameters['forward_flow']
    )

    init_profile = setting.get_initial_profile(**kwargs)

    times = _lookup(config['input']['solver'], 'user_solution_times_unit_state')
    if times is None:
        raise ValueError(
            f"Setting {setting_name} does not define "
            "user_solution_times_unit_state, so the solution times of the "
            "bulk solution (and hence of the reference) are unknown."
        )
    times = np.atleast_1d(np.asarray(times, dtype=float))

    advection = parameters['flow_rate'] != 0.0
    dispersion = parameters['col_dispersion'] != 0.0

    if advection and dispersion:
        raise NotImplementedError(
            f"No analytical solution is available for {setting_name} "
            "(combined advection and dispersion), see the module docstring of "
            "src/analytical.py."
        )

    if dispersion:
        expansion = DispersionExpansion(
            parameters['geometry'], parameters['col_dispersion'],
            init_profile=init_profile, n_modes=n_modes
        )
        def solution_at(t):
            return lambda xi: expansion(xi, t)
        diagnostics = {
            'N_EIGENMODES': np.int32(n_modes),
            'EIGENEXPANSION_INITIAL_CONDITION_ERROR':
                expansion.initial_condition_error,
        }
        kind = 'pure dispersion eigenfunction expansion'
    else:
        def solution_at(t):
            return lambda xi: analytical_solution(
                xi, t, parameters['geometry'],
                flux_coefficient=parameters['flux_coefficient'],
                init_profile=init_profile,
                towards_increasing_x=towards_increasing_x
            )
        diagnostics = {}
        kind = ('pure advection characteristics, flow towards '
                + ('increasing' if towards_increasing_x else 'decreasing')
                + ' x')

    xi_nodes = reference_grid(poly_deg, n_elem)
    values = np.stack(
        [np.asarray(solution_at(t)(xi_nodes), dtype=float) for t in times],
        axis=0
    )[:, :, None]

    representation_error = max(
        reference_representation_error(solution_at(t), poly_deg, n_elem)
        for t in times
    )

    file_name = os.path.join(output_dir, reference_file_name(setting_name))
    write_reference_h5(
        file_name, unit_config, times, values,
        poly_deg=poly_deg, n_elem=n_elem,
        meta={
            'ANALYTICAL_SOLUTION': kind,
            'SETTING': setting_name,
            'GEOMETRY': _string(_lookup(unit_config, 'GEOMETRY')),
            'FLUX_COEFFICIENT': parameters['flux_coefficient'] if advection else 0.0,
            'FLOW_TOWARDS_INCREASING_X': np.int32(towards_increasing_x),
            'COL_DISPERSION': parameters['col_dispersion'],
            'REPRESENTATION_ERROR': representation_error,
            **diagnostics,
        }
    )

    if verbose:
        print(f"{setting_name}: {kind}")
        print(f"    times                 {times}")
        print(f"    representation error  {representation_error:.2e} "
              f"(DG P{poly_deg} Z{n_elem})")
        for key, value in diagnostics.items():
            print(f"    {key.lower():21s} {value}")
        print(f"    -> {file_name}")

    return file_name


def generate_geometry_references(output_dir=REFERENCE_DIR, verbose=True,
                                 user_solution_times_unit_state=None,
                                 model_kwargs=None, **kwargs):
    """Write the analytical references of all analytically available settings.

    Parameters
    ----------
    output_dir : string
        Directory to write the references to.
    verbose : bool
        Print the accuracy diagnostics of the generated references.
    user_solution_times_unit_state : array_like
        Solution times of the bulk solution, which must be the ones of the
        convergence study, see DEFAULT_SOLUTION_TIMES.
    model_kwargs : dict
        Additional keyword arguments of the model builder, applied to every
        setting (e.g. non-default initial profile parameters).
    kwargs : dict
        Forwarded to generate_reference.

    Returns
    -------
    dict
        Setting name -> written file name.
    """
    if user_solution_times_unit_state is None:
        user_solution_times_unit_state = list(DEFAULT_SOLUTION_TIMES)

    shared = dict(model_kwargs or {})
    shared['user_solution_times_unit_state'] = user_solution_times_unit_state

    return {
        setting_name: generate_reference(
            setting_name, {**setting_kwargs, **shared},
            output_dir=output_dir, verbose=verbose, **kwargs
        )
        for setting_name, setting_kwargs in REFERENCE_SETTINGS.items()
    }


def load_reference(setting_name, reference_dir=REFERENCE_DIR):
    """Load an analytical reference as a CADET object, or return None.

    Intended for the ref_files argument of
    bench_func.run_convergence_analysis: settings without an analytical
    solution (and missing reference files) simply yield None, which makes the
    convergence analysis fall back to the self-convergence reference.

    Parameters
    ----------
    setting_name : string
        Name of the setting.
    reference_dir : string
        Directory holding the reference files. None disables the analytical
        references altogether.

    Returns
    -------
    Cadet object or None
    """
    from src.utility.convergence import get_simulation

    if reference_dir is None:
        return None

    file_name = os.path.join(reference_dir, reference_file_name(setting_name))
    if not os.path.exists(file_name):
        return None
    return get_simulation(file_name)


if __name__ == '__main__':
    generate_geometry_references()
