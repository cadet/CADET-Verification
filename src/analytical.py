import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import brentq

# =============================================================================
# Geometry
# =============================================================================

def get_column_geometry_configuration(geometry: str):
    """
    Geometry parameters corresponding to the CADET benchmark setup.
    """

    velocity = 0.000575       # interstitial velocity [m/s]
    porosity = 0.37
    Q = 6.0e-05               # volumetric flow rate [m^3/s]

    axial_flow_cross_section_area = (
        Q / velocity / porosity
    )

    axial_flow_radius = np.sqrt(
        axial_flow_cross_section_area / np.pi
    )

    if geometry == "AXIAL_FLOW_CYLINDER":
        return {
            "cross_section_area": axial_flow_cross_section_area,
            "col_length": 0.014,
        }

    elif geometry == "RADIAL_FLOW_CYLINDER_SHELL":
        H = 0.25

        r_outer = (
            axial_flow_cross_section_area
            / (2.0 * np.pi * H)
        )

        r_inner = r_outer - 0.014

        return {
            "cross_section_area": axial_flow_cross_section_area,
            "col_length": H,
            "col_radius_outer": r_outer,
            "col_radius_inner": r_inner,
        }

    elif geometry == "AXIAL_FLOW_FRUSTUM":
        return {
            "cross_section_area": axial_flow_cross_section_area,
            "col_radius_outer": axial_flow_radius,
            "col_radius_inner": 0.75 * axial_flow_radius,
            "col_radius_large_end": axial_flow_radius,
            "col_radius_small_end": 0.75 * axial_flow_radius,
            "col_length": 0.014,
        }

    else:
        raise ValueError(
            f"Unknown geometry: {geometry}"
        )


# =============================================================================
# Initial condition
# =============================================================================

def gaussian_bump(
    x,
    length,
    center=0.5,
    stddev=0.05,
    amplitude=1.0,
):
    """
    Gaussian bump used by the CADET benchmark.

    The Gaussian parameters center and stddev are in normalized
    coordinates x / length.
    """

    xi = np.asarray(x) / length

    return amplitude * np.exp(
        -(xi - center) ** 2
        / (2.0 * stddev ** 2)
    )


# =============================================================================
# Public interface
# =============================================================================

def analytical_solution(
    x,
    t,
    geometry="AXIAL_FLOW_CYLINDER",
    advection=True,
    dispersion=True,
    col_dispersion=5.75e-8,
    porosity=0.37,
    flow_rate=6.0e-5,
    init_center=0.5,
    init_stddev=0.05,
    init_amplitude=1.0,
    n_reference=4096,
):
    """
    Analytical/reference solution for the CADET pure-transport benchmark.

    Parameters
    ----------
    x : array_like
        Spatial coordinate.

        AXIAL_FLOW_CYLINDER:
            x in [0, L]

        AXIAL_FLOW_FRUSTUM:
            x in [0, L]

        RADIAL_FLOW_CYLINDER_SHELL:
            x is radial coordinate in [r_inner, r_outer].

    t : float or array_like
        Time [s].

    geometry : str
        "AXIAL_FLOW_CYLINDER"
        "AXIAL_FLOW_FRUSTUM"
        "RADIAL_FLOW_CYLINDER_SHELL"

    advection : bool
        Enable advection.

    dispersion : bool
        Enable dispersion.

    col_dispersion : float
        Dispersion coefficient [m^2/s].

    porosity : float
        Column porosity.

    flow_rate : float
        Volumetric flow rate [m^3/s].

    init_center : float
        Initial Gaussian center in normalized coordinates.

    init_stddev : float
        Initial Gaussian standard deviation in normalized coordinates.

    init_amplitude : float
        Initial Gaussian amplitude.

    n_reference : int
        Number of cells used for variable-geometry reference solution.

    Returns
    -------
    ndarray
        If x and t are scalar: scalar.

        If t is scalar and x is an array:
            shape x.shape

        If x is scalar and t is an array:
            shape t.shape

        If both are arrays:
            shape (len(t), len(x))
    """

    geometry = geometry.upper()

    if not (advection or dispersion):
        raise ValueError(
            "At least one of advection and dispersion "
            "must be enabled."
        )

    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)

    x_scalar = x_arr.ndim == 0
    t_scalar = t_arr.ndim == 0

    x_flat = x_arr.reshape(-1)
    t_flat = t_arr.reshape(-1)

    geom = get_column_geometry_configuration(
        geometry
    )

    # -------------------------------------------------------------------------
    # Constant-area cylinder
    # -------------------------------------------------------------------------

    if geometry == "AXIAL_FLOW_CYLINDER":

        L = geom["col_length"]

        A = geom["cross_section_area"]

        velocity = (
            flow_rate
            / (porosity * A)
            if advection
            else 0.0
        )

        D = (
            col_dispersion
            if dispersion
            else 0.0
        )

        result = _cylinder_solution(
            x_flat,
            t_flat,
            L=L,
            velocity=velocity,
            D=D,
            center=init_center,
            stddev=init_stddev,
            amplitude=init_amplitude,
        )

    # -------------------------------------------------------------------------
    # Frustum
    # -------------------------------------------------------------------------

    elif geometry == "AXIAL_FLOW_FRUSTUM":

        L = geom["col_length"]

        r_large = geom["col_radius_large_end"]
        r_small = geom["col_radius_small_end"]

        if not dispersion:

            result = _frustum_advection_solution(
                x_flat,
                t_flat,
                L=L,
                r_large=r_large,
                r_small=r_small,
                Q=flow_rate,
                porosity=porosity,
                center=init_center,
                stddev=init_stddev,
                amplitude=init_amplitude,
            )

        else:

            result = _variable_geometry_solution(
                x_flat,
                t_flat,
                geometry=geometry,
                Q=flow_rate if advection else 0.0,
                D=col_dispersion,
                porosity=porosity,
                center=init_center,
                stddev=init_stddev,
                amplitude=init_amplitude,
                n_reference=n_reference,
                L=L,
                r_large=r_large,
                r_small=r_small,
            )

    # -------------------------------------------------------------------------
    # Radial shell
    # -------------------------------------------------------------------------

    elif geometry == "RADIAL_FLOW_CYLINDER_SHELL":

        r_inner = geom["col_radius_inner"]
        r_outer = geom["col_radius_outer"]
        H = geom["col_length"]

        if not dispersion:

            result = _radial_advection_solution(
                x_flat,
                t_flat,
                r_inner=r_inner,
                r_outer=r_outer,
                H=H,
                Q=flow_rate,
                porosity=porosity,
                center=init_center,
                stddev=init_stddev,
                amplitude=init_amplitude,
            )

        else:

            result = _variable_geometry_solution(
                x_flat,
                t_flat,
                geometry=geometry,
                Q=flow_rate if advection else 0.0,
                D=col_dispersion,
                porosity=porosity,
                center=init_center,
                stddev=init_stddev,
                amplitude=init_amplitude,
                n_reference=n_reference,
                r_inner=r_inner,
                r_outer=r_outer,
                H=H,
            )

    else:
        raise ValueError(
            f"Unknown geometry: {geometry}"
        )

    # -------------------------------------------------------------------------
    # Restore user requested shape
    # -------------------------------------------------------------------------

    if x_scalar and t_scalar:
        return float(result[0, 0])

    if t_scalar:
        return result[0].reshape(x_arr.shape)

    if x_scalar:
        return result[:, 0].reshape(t_arr.shape)

    return result.reshape(
        t_arr.shape + x_arr.shape
    )


# =============================================================================
# Constant-area axial cylinder
# =============================================================================

def _cylinder_solution(
    x,
    t,
    L,
    velocity,
    D,
    center,
    stddev,
    amplitude,
):
    """
    Closed-form Gaussian solution for constant-area
    advection-diffusion.

    c(x,t) =
        A sigma0 / sigma(t)
        exp[-(x-x0-vt)^2/(2 sigma(t)^2)]

    where

        sigma(t)^2 = sigma0^2 + 2 D t.

    The Gaussian initial condition is assumed to be sufficiently
    far from the boundaries that the whole-line solution is valid.
    """

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)

    sigma0 = stddev * L
    x0 = center * L

    X = x[None, :]
    T = t[:, None]

    # ------------------------------------------------------------------
    # Pure advection
    # ------------------------------------------------------------------

    if D == 0.0:

        result = amplitude * np.exp(
            -(
                X
                - x0
                - velocity * T
            ) ** 2
            / (2.0 * sigma0**2)
        )

    # ------------------------------------------------------------------
    # Dispersion, with or without advection
    # ------------------------------------------------------------------

    else:

        sigma2 = (
            sigma0**2
            + 2.0 * D * T
        )

        sigma = np.sqrt(sigma2)

        result = (
            amplitude
            * sigma0
            / sigma
            * np.exp(
                -(
                    X
                    - x0
                    - velocity * T
                ) ** 2
                / (2.0 * sigma2)
            )
        )

    # ------------------------------------------------------------------
    # Apply physical domain mask.
    #
    # IMPORTANT:
    # np.broadcast_to() is required here because boolean indexing
    # does not perform the same broadcasting as arithmetic operations.
    # ------------------------------------------------------------------

    physical = np.broadcast_to(
        (X >= 0.0) & (X <= L),
        result.shape,
    )

    result = np.where(
        physical,
        result,
        0.0,
    )

    return result

# =============================================================================
# Frustum geometry
# =============================================================================

def _frustum_radius(
    x,
    L,
    r_large,
    r_small,
):
    """
    Radius at axial coordinate x.
    """

    return (
        r_large
        + (r_small - r_large)
        * np.asarray(x)
        / L
    )


def _frustum_area(
    x,
    L,
    r_large,
    r_small,
):
    r = _frustum_radius(
        x,
        L,
        r_large,
        r_small,
    )

    return np.pi * r**2


def _frustum_volume_coordinate(
    x,
    L,
    r_large,
    r_small,
):
    """
    V(x) = integral_0^x A(s) ds.
    """

    x = np.asarray(x)

    b = (
        r_small - r_large
    ) / L

    return np.pi * (
        r_large**2 * x
        + r_large * b * x**2
        + b**2 * x**3 / 3.0
    )


# =============================================================================
# Frustum pure advection
# =============================================================================

def _frustum_advection_solution(
    x,
    t,
    L,
    r_large,
    r_small,
    Q,
    porosity,
    center,
    stddev,
    amplitude,
):
    """
    Exact characteristic solution for pure advection in a frustum.

    Q is constant and

        v(x) = Q / (epsilon A(x)).

    Introducing the volume coordinate

        V(x) = integral A(x) dx

    gives

        epsilon/Q * dV/dt = 1.

    Therefore the characteristic is

        V(x(t)) =
            V(x0) + Q t / epsilon.
    """

    x = np.asarray(x)
    t = np.asarray(t)

    result = np.zeros(
        (len(t), len(x))
    )

    total_volume = _frustum_volume_coordinate(
        L,
        L,
        r_large,
        r_small,
    )

    for it, ti in enumerate(t):

        V_target = (
            _frustum_volume_coordinate(
                x,
                L,
                r_large,
                r_small,
            )
            - Q * ti / porosity
        )

        valid = (
            (V_target >= 0.0)
            & (V_target <= total_volume)
        )

        x0 = np.zeros_like(x)

        # Invert V(x0) numerically.
        valid_indices = np.flatnonzero(valid)

        for j in valid_indices:

            target = V_target[j]

            # Bisection is robust and this is only used for the
            # pure-advection analytical reference.
            lo = 0.0
            hi = L

            for _ in range(60):

                mid = 0.5 * (lo + hi)

                Vmid = _frustum_volume_coordinate(
                    mid,
                    L,
                    r_large,
                    r_small,
                )

                if Vmid < target:
                    lo = mid
                else:
                    hi = mid

            x0[j] = 0.5 * (lo + hi)

        c0 = gaussian_bump(
            x0,
            L,
            center=center,
            stddev=stddev,
            amplitude=amplitude,
        )

        result[it] = np.where(
            valid,
            c0,
            0.0,
        )

    return result


# =============================================================================
# Radial shell pure advection
# =============================================================================

def _radial_advection_solution(
    r,
    t,
    r_inner,
    r_outer,
    H,
    Q,
    porosity,
    center,
    stddev,
    amplitude,
):
    """
    Exact characteristic solution for pure radial advection.

    v(r) =
        Q / (epsilon 2 pi r H)

    hence

        r(t)^2 =
            r0^2 + Q t/(epsilon pi H).
    """

    r = np.asarray(r)
    t = np.asarray(t)

    result = np.zeros(
        (len(t), len(r))
    )

    dr = (
        r_outer - r_inner
    )

    for it, ti in enumerate(t):

        r0_squared = (
            r**2
            - Q * ti
            / (porosity * np.pi * H)
        )

        valid = (
            r0_squared >= r_inner**2
        )

        r0 = np.sqrt(
            np.maximum(
                r0_squared,
                r_inner**2,
            )
        )

        xi0 = (
            r0 - r_inner
        ) / dr

        result[it] = np.where(
            valid,
            amplitude * np.exp(
                -(xi0 - center)**2
                / (2.0 * stddev**2)
            ),
            0.0,
        )

    return result


# =============================================================================
# Bernoulli function for Scharfetter-Gummel flux
# =============================================================================

def _bernoulli(Pe):
    """
    Bernoulli function

        B(Pe) = Pe / (exp(Pe) - 1)

    evaluated stably for small and large Pe.
    """

    Pe = np.asarray(Pe)

    result = np.empty_like(
        Pe,
        dtype=float,
    )

    small = np.abs(Pe) < 1.0e-6
    moderate = (
        ~small
        & (Pe < 50.0)
    )
    large_positive = Pe >= 50.0

    # Taylor expansion around zero:
    #
    # B(x) = 1 - x/2 + x^2/12 - x^4/720 + ...
    x = Pe[small]

    result[small] = (
        1.0
        - x / 2.0
        + x**2 / 12.0
        - x**4 / 720.0
    )

    # Normal evaluation.
    x = Pe[moderate]

    result[moderate] = (
        x / np.expm1(x)
    )

    # For large positive x:
    #
    # B(x) ~ x exp(-x)
    #
    # which is small enough that this approximation is sufficient.
    x = Pe[large_positive]

    result[large_positive] = (
        x * np.exp(-x)
    )

    return result


# =============================================================================
# Sparse variable-geometry reference solver
# =============================================================================

def _variable_geometry_solution(
    x,
    t,
    geometry,
    Q,
    D,
    porosity,
    center,
    stddev,
    amplitude,
    n_reference=4096,
    **kwargs,
):
    """
    Sparse reference solution for variable-area transport.

    The governing equation is

        epsilon A(x) dc/dt
          =
        -d/dx[
            Q c - epsilon A D dc/dx
        ].

    The spatial discretization is finite-volume.

    For the combined advection-dispersion case the face flux is
    evaluated using the Scharfetter-Gummel exponential-fitting formula.

    The resulting semi-discrete system is

        dc/dt = L c,

    and the solution is evaluated as

        c(t) = exp(t L) c(0)

    using scipy.sparse.linalg.expm_multiply.

    This avoids time-stepping error in the reference calculation.
    """

    x = np.asarray(x)
    t = np.asarray(t)

    # -------------------------------------------------------------------------
    # Geometry-specific domain and area
    # -------------------------------------------------------------------------

    if geometry == "AXIAL_FLOW_FRUSTUM":

        xmin = 0.0
        xmax = kwargs["L"]

        L = kwargs["L"]
        r_large = kwargs["r_large"]
        r_small = kwargs["r_small"]

        def area(xx):
            return _frustum_area(
                xx,
                L,
                r_large,
                r_small,
            )

        def initial_condition(xx):
            return gaussian_bump(
                xx,
                L,
                center=center,
                stddev=stddev,
                amplitude=amplitude,
            )

    elif geometry == "RADIAL_FLOW_CYLINDER_SHELL":

        xmin = kwargs["r_inner"]
        xmax = kwargs["r_outer"]

        r_inner = kwargs["r_inner"]
        r_outer = kwargs["r_outer"]
        H = kwargs["H"]

        def area(rr):
            return (
                2.0
                * np.pi
                * rr
                * H
            )

        def initial_condition(rr):

            xi = (
                rr - r_inner
            ) / (
                r_outer - r_inner
            )

            return amplitude * np.exp(
                -(xi - center)**2
                / (2.0 * stddev**2)
            )

    else:
        raise ValueError(
            "Variable-geometry reference solver "
            f"does not support {geometry}."
        )

    # -------------------------------------------------------------------------
    # Cell-centered finite-volume grid
    # -------------------------------------------------------------------------

    n = int(n_reference)

    if n < 16:
        raise ValueError(
            "n_reference must be at least 16."
        )

    dx = (
        xmax - xmin
    ) / n

    # Cell centers.
    xc = (
        xmin
        + (np.arange(n) + 0.5) * dx
    )

    # Face coordinates.
    xf = (
        xmin
        + np.arange(n + 1) * dx
    )

    A_cell = area(xc)
    A_face = area(xf)

    # -------------------------------------------------------------------------
    # Cell-volume weighting
    #
    # The governing equation contains epsilon*A*c_t.
    #
    # For a uniform grid:
    #
    #   epsilon A_i dc_i/dt
    #       = -(F_{i+1/2} - F_{i-1/2}) / dx
    #
    # -------------------------------------------------------------------------

    mass = (
        porosity
        * A_cell
        * dx
    )

    # -------------------------------------------------------------------------
    # Build sparse operator.
    #
    # Each interior face contributes a 2x2 block.
    # -------------------------------------------------------------------------

    lower = np.zeros(n - 1)
    diagonal = np.zeros(n)
    upper = np.zeros(n - 1)

    # -------------------------------------------------------------------------
    # Scharfetter-Gummel face coefficients
    #
    # Flux:
    #
    # F = Q c_L
    #     - epsilon A D dc/dx
    #
    # can be represented in exponential-fitting form as
    #
    # F =
    #   epsilon A D / dx
    #   [B(-Pe)c_L - B(Pe)c_R]
    #
    # where
    #
    # Pe = Q dx / (epsilon A D).
    #
    # For D=0 this smoothly approaches the upwind advective flux.
    # -------------------------------------------------------------------------

    if D > 0.0:

        Pe = (
            Q * dx
            / (
                porosity
                * A_face
                * D
            )
        )

        Bp = _bernoulli(Pe)
        Bm = _bernoulli(-Pe)

        diffusion_factor = (
            porosity
            * A_face
            * D
            / dx
        )

        coeff_left = (
            diffusion_factor
            * Bm
        )

        coeff_right = (
            diffusion_factor
            * Bp
        )

    else:

        # Pure advection.
        #
        # Positive Q:
        #
        # F = Q c_L
        #
        # Negative Q:
        #
        # F = Q c_R
        #
        if Q >= 0.0:

            coeff_left = np.full(
                n + 1,
                Q,
                dtype=float,
            )

            coeff_right = np.zeros(
                n + 1
            )

        else:

            coeff_left = np.zeros(
                n + 1
            )

            coeff_right = np.full(
                n + 1,
                Q,
                dtype=float,
            )

    # -------------------------------------------------------------------------
    # Interior faces
    # -------------------------------------------------------------------------

    for f in range(1, n):

        i = f - 1
        j = f

        a = coeff_left[f]
        b = coeff_right[f]

        # Flux:
        #
        # F_f = a*c_i - b*c_j
        #
        # dc_i/dt contains -F_f/m_i
        # dc_j/dt contains +F_f/m_j

        diagonal[i] -= a / mass[i]
        upper[i] += b / mass[i]

        lower[i] += a / mass[j]
        diagonal[j] -= b / mass[j]

    # -------------------------------------------------------------------------
    # Boundary conditions
    #
    # CADET benchmark:
    #
    #   inlet concentration = 0
    #   outlet = convective outflow
    #
    # For pure dispersion the benchmark has zero-flux boundaries.
    #
    # For combined transport the left boundary has c_in = 0 and the
    # right boundary is outflow.
    # -------------------------------------------------------------------------

    if Q != 0.0:

        if Q > 0.0:

            # Inlet face:
            #
            # c_in = 0.
            #
            # Therefore its flux is zero.
            #
            # No matrix contribution is required.

            pass

            # Outlet face:
            #
            # For positive flow, F = Q*c_N at the outlet.
            #
            # The boundary flux leaves the last cell.
            #
            if D > 0.0:

                # SG with a zero diffusive gradient at the outlet.
                #
                # The exponential-fitting formulation reduces to the
                # appropriate outflow flux when the exterior state is
                # taken equal to the last interior state.

                diagonal[-1] -= (
                    Q / mass[-1]
                )

            else:

                diagonal[-1] -= (
                    Q / mass[-1]
                )

        else:

            raise NotImplementedError(
                "Negative flow is not implemented."
            )

    else:

        # Pure diffusion:
        #
        # zero diffusive flux at both boundaries.
        #
        # No boundary matrix contribution is needed.

        pass

    # -------------------------------------------------------------------------
    # Sparse tridiagonal operator
    # -------------------------------------------------------------------------

    Lmat = diags(
        diagonals=[
            lower,
            diagonal,
            upper,
        ],
        offsets=[
            -1,
            0,
            1,
        ],
        shape=(n, n),
        format="csc",
    )

    # -------------------------------------------------------------------------
    # Initial condition
    # -------------------------------------------------------------------------

    c0 = initial_condition(xc)

    # -------------------------------------------------------------------------
    # Evaluate exp(t L)c0.
    #
    # scipy.sparse.linalg.expm_multiply is the correct import.
    # -------------------------------------------------------------------------

    if len(t) == 0:
        return np.empty(
            (0, len(x))
        )

    c_reference = np.empty(
        (len(t), len(x))
    )

    # Evaluate all requested times independently.
    #
    # This is slightly more expensive than a single start/stop call for
    # equally spaced times, but works for arbitrary t arrays.
    #
    # For a convergence study with np.linspace(), use the optimized
    # branch below.

    if (
        len(t) > 1
        and np.allclose(
            np.diff(t),
            np.diff(t)[0],
        )
    ):

        # Equidistant output times.
        #
        # expm_multiply supports a start/stop/num interface and avoids
        # repeating the Krylov setup unnecessarily.

        states = expm_multiply(
            Lmat,
            c0,
            start=float(t[0]),
            stop=float(t[-1]),
            num=len(t),
        )

        c_reference[:, :] = states

    else:

        for k, tk in enumerate(t):

            if tk == 0.0:

                c_reference[k] = c0

            else:

                c_reference[k] = expm_multiply(
                    Lmat * float(tk),
                    c0,
                )

    # -------------------------------------------------------------------------
    # Interpolate cell-centered reference solution to requested coordinates.
    # -------------------------------------------------------------------------

    result = np.empty(
        (len(t), len(x))
    )

    for k in range(len(t)):

        result[k] = np.interp(
            x,
            xc,
            c_reference[k],
            left=0.0,
            right=0.0,
        )

    return result


###############################################################################
###############################################################################
###############################################################################



import numpy as np

# geom = get_column_geometry_configuration(
#     "RADIAL_FLOW_CYLINDER_SHELL"
# )
# r = np.linspace(
#     geom["col_radius_inner"],
#     geom["col_radius_outer"],
#     500,
# )

x = np.linspace(0.0, 0.014, 500)
t = np.linspace(0.0, 30.0, 601)

# c = analytical_solution(
#     x,
#     t,
#     geometry="AXIAL_FLOW_FRUSTUM", # AXIAL_FLOW_CYLINDER # AXIAL_FLOW_CYLINDER
#     advection=True,
#     dispersion=False,
# )

# from matplotlib import pyplot as plt

# plt.plot(x, c[25, :], label=f"t = {t[25]} s")
# plt.show()


# import h5py
# import os
# os.makedirs("data/CADET-Verification_reference", exist_ok=True)

# with h5py.File("data/CADET-Verification_reference/analytical_frustumAdvDPFR_x_c_t1.25s.h5", "w") as f:

#     # Input
#     f.create_dataset(
#         "input/model/unit_001/col_length",
#         data=0.014
#     )

#     # Time
#     f.create_dataset(
#         "output/solution/USER_SOLUTION_TIMES",
#         data=np.array([t[0], t[25]])
#     )

#     # Axial coordinates
#     f.create_dataset(
#         "output/coordinates/unit_001/AXIAL_COORDINATES",
#         data=x
#     )

#     # Concentration solution
#     c_export = np.stack([
#         c[25, :],
#         c[-1, :],
#     ], axis=0)[:, :, np.newaxis]
#     f.create_dataset(
#         "output/solution/unit_001/SOLUTION_BULK",
#         data=c_export
#     )



from src.utility.convergence import get_simulation
file1 = r"C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core\chromatography\frustumAdvDPFR_1comp_benchmark1_DG_P4Z4096.h5"
file2 = r"C:\Users\jmbr\software\CADET-Verification\data\CADET-Verification_reference\analytical_frustumAdvDPFR_x_c_t1.25s.h5"


sim1 = get_simulation(file1)
sim2 = get_simulation(file2)

solution1 = sim1.root.output.solution.unit_001.solution_bulk
solution2 = sim2.root.output.solution.unit_001.solution_bulk
ref_coordinates = sim2.root.output.unit_001.axial_coordinates

print(solution1.shape)
print(solution2.shape)

print(np.max(np.abs(solution1 - solution2)))




