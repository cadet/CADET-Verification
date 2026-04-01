# -*- coding: utf-8 -*-
"""
Analytical solutions for 1D linear chromatography models (LRM, LRMP, GRM).

Uses Laplace domain transfer functions with numerical inverse Laplace
transform (de Hoog algorithm via mpmath) to compute outlet concentration
profiles for columns with Danckwerts boundary conditions, linear binding,
and rectangular pulse inlet.

References
----------
- Qamar et al., "Analytical solutions and moment analysis of general rate
  model for linear liquid chromatography", Chem. Eng. Sci. 107 (2014) 192-205
- Miyabe, "Moment equations for chromatography based on Langmuir type
  reaction kinetics", J. Chromatogr. A 1218 (2011) 6378-6393
- Javeed et al., "Efficient and accurate numerical simulation of nonlinear
  chromatographic processes", Comput. Chem. Eng. 35 (2011) 2294-2305

"""

import numpy as np
from mpmath import mp, mpf, sqrt, exp, tanh


def _danckwerts_transfer_function(s, u, D_ax, L, phi_s):
    """Column transfer function with Danckwerts boundary conditions.

    Computes C_out(s) / C_in(s) for a 1D column with axial dispersion
    and a lumped sink term phi(s).

    The governing equation in Laplace domain is:
        D_ax * C'' - u * C' - phi(s) * C = 0

    Parameters
    ----------
    s : mpf
        Laplace variable.
    u : mpf
        Interstitial velocity.
    D_ax : mpf
        Axial dispersion coefficient.
    L : mpf
        Column length.
    phi_s : mpf
        Lumped Laplace-domain sink/source term (depends on model).

    Returns
    -------
    mpf
        Transfer function value G(s) = C_out(s) / C_in(s).
    """
    Pe = u * L / D_ax
    lam = sqrt(Pe**2 / 4 + phi_s * L**2 / D_ax)

    # Numerically stable form using exponential shift
    G = 2 * lam * exp(Pe / 2) / (
        (lam + Pe / 2) * exp(lam) + (lam - Pe / 2) * exp(-lam)
    )

    return G


def _rectangular_pulse_inlet(s, c_in, t_inj):
    """Laplace transform of a rectangular pulse inlet.

    c_in(t) = c_in for 0 <= t < t_inj, 0 otherwise.

    Parameters
    ----------
    s : mpf
        Laplace variable.
    c_in : mpf
        Inlet concentration during injection.
    t_inj : mpf
        Injection duration.

    Returns
    -------
    mpf
        C_in(s) = c_in * (1 - exp(-s * t_inj)) / s
    """
    return c_in * (1 - exp(-s * t_inj)) / s


def _phi_LRM(s, epsilon_t, k_a, k_d):
    """Lumped sink term for the LRM (Lumped Rate Model, total porosity).

    Governing equation:
        dc/dt = -u/epsilon_t * dc/dz + D_ax * d²c/dz² - F * dq/dt
        dq/dt = k_a * c - k_d * q

    where F = (1 - epsilon_t) / epsilon_t

    phi(s) = s + F * s * k_a / (s + k_d)
           = s * (1 + F * k_a / (s + k_d))
    """
    F = (1 - epsilon_t) / epsilon_t
    return s * (1 + F * k_a / (s + k_d))


def _phi_LRMP(s, epsilon_c, epsilon_p, k_f, R_p, k_a, k_d):
    """Lumped sink term for the LRMP (Lumped Rate Model with Pores).

    Includes film mass transfer resistance. Particles are homogeneous
    (no pore diffusion), with linear binding.

    Particle phase (well-mixed):
        epsilon_p * dc_p/dt + (1-epsilon_p) * dq/dt = 3*k_f/R_p * (c - c_p)
        dq/dt = k_a * c_p - k_d * q

    In Laplace domain with a_p(s) = s*(epsilon_p + (1-epsilon_p)*k_a/(s+k_d)):
        phi(s) = s + (1-eps_c)/eps_c * a_p * 3*k_f/R_p / (a_p + 3*k_f/R_p)
    """
    Fp = (1 - epsilon_c) / epsilon_c
    a_p = s * (epsilon_p + (1 - epsilon_p) * k_a / (s + k_d))
    j_coeff = a_p * 3 * k_f / R_p / (a_p + 3 * k_f / R_p)
    return s + Fp * j_coeff


def _phi_GRM(s, epsilon_c, epsilon_p, k_f, R_p, D_p, k_a, k_d,
             D_s=0.0):
    """Lumped sink term for the GRM (General Rate Model).

    Includes film mass transfer, pore diffusion, and optional surface
    diffusion. Spherical particles with linear binding.

    The particle equation (spherical coordinates):
        epsilon_p * dc_p/dt + (1-epsilon_p) * dq/dt
            = 1/r² * d/dr(r² * (D_p * dc_p/dr + (1-epsilon_p)/epsilon_p * D_s * dq/dr))

    For linear binding (q = k_a/k_d * c_p at equilibrium, or kinetic):
        dq/dt = k_a * c_p - k_d * q

    In Laplace domain, effective diffusivity and reaction term:
        D_eff(s) = D_p + (1-epsilon_p) * D_s * k_a / (s + k_d)
        alpha(s) = epsilon_p * s + (1-epsilon_p) * k_a * s / (s + k_d)

    Particle solution (spherical Bessel):
        C_p(r,s) = A * sinh(gamma*r) / r
        where gamma = sqrt(alpha / D_eff)

    Flux at particle surface:
        j = k_f * (C - C_p(R_p)) coupled with D_eff * dC_p/dr|_{R_p} = k_f * (C - C_p(R_p))

    Effective flux coefficient:
        3/R_p * k_f * (gamma*R_p*coth(gamma*R_p) - 1) * D_eff / R_p
        / (k_f + (gamma*R_p*coth(gamma*R_p) - 1) * D_eff / R_p)
    """
    Fp = (1 - epsilon_c) / epsilon_c
    D_eff = D_p + (1 - epsilon_p) * D_s * k_a / (s + k_d)
    alpha = s * (epsilon_p + (1 - epsilon_p) * k_a / (s + k_d))

    gamma = sqrt(alpha / D_eff)
    gR = gamma * R_p

    # Bi_p: modified Biot number for particle
    # coth(gR) - 1/gR = (gR*cosh(gR) - sinh(gR)) / (gR*sinh(gR))
    if abs(gR) < mpf('1e-10'):
        # Taylor expansion for small gR: coth(x) - 1/x ≈ x/3
        Omega = D_eff * gR / (3 * R_p)
    else:
        Omega = D_eff * (gR / tanh(gR) - 1) / R_p

    j_coeff = 3 / R_p * k_f * Omega / (k_f + Omega)

    return s + Fp * j_coeff


def compute_analytical_outlet(
        model_type, params, solution_times, dps=50, method='dehoog'):
    """Compute analytical outlet concentration via inverse Laplace transform.

    Parameters
    ----------
    model_type : str
        One of 'LRM', 'LRMP', 'GRM'.
    params : dict
        Model parameters. Required keys depend on model_type:
        - All: 'velocity', 'col_dispersion', 'col_length',
               'ka', 'kd', 'c_in', 't_inj'
        - LRM: 'total_porosity'
        - LRMP: 'col_porosity', 'par_porosity', 'film_diffusion', 'par_radius'
        - GRM: same as LRMP plus 'pore_diffusion',
               optional 'surface_diffusion'
    solution_times : array_like
        Times at which to evaluate the solution.
    dps : int
        Decimal places for mpmath precision. Default 50.
    method : str
        Inverse Laplace method: 'dehoog', 'talbot', or 'stehfest'.

    Returns
    -------
    numpy.ndarray
        Outlet concentration at each solution time.
    """
    mp.dps = dps

    u = mpf(str(params['velocity']))
    D_ax = mpf(str(params['col_dispersion']))
    L = mpf(str(params['col_length']))
    k_a = mpf(str(params['ka']))
    k_d = mpf(str(params['kd']))
    c_in = mpf(str(params.get('c_in', 1.0)))
    t_inj = mpf(str(params['t_inj']))

    if model_type == 'LRM':
        eps_t = mpf(str(params['total_porosity']))

        def C_out_laplace(s):
            phi = _phi_LRM(s, eps_t, k_a, k_d)
            G = _danckwerts_transfer_function(s, u, D_ax, L, phi)
            return _rectangular_pulse_inlet(s, c_in, t_inj) * G

    elif model_type == 'LRMP':
        eps_c = mpf(str(params['col_porosity']))
        eps_p = mpf(str(params['par_porosity']))
        k_f = mpf(str(params['film_diffusion']))
        R_p = mpf(str(params['par_radius']))

        def C_out_laplace(s):
            phi = _phi_LRMP(s, eps_c, eps_p, k_f, R_p, k_a, k_d)
            G = _danckwerts_transfer_function(s, u, D_ax, L, phi)
            return _rectangular_pulse_inlet(s, c_in, t_inj) * G

    elif model_type == 'GRM':
        eps_c = mpf(str(params['col_porosity']))
        eps_p = mpf(str(params['par_porosity']))
        k_f = mpf(str(params['film_diffusion']))
        R_p = mpf(str(params['par_radius']))
        D_p = mpf(str(params['pore_diffusion']))
        D_s = mpf(str(params.get('surface_diffusion', 0.0)))

        def C_out_laplace(s):
            phi = _phi_GRM(s, eps_c, eps_p, k_f, R_p, D_p, k_a, k_d, D_s)
            G = _danckwerts_transfer_function(s, u, D_ax, L, phi)
            return _rectangular_pulse_inlet(s, c_in, t_inj) * G

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Compute inverse Laplace transform at each time point
    result = np.zeros(len(solution_times))

    for i, t in enumerate(solution_times):
        if t <= 0:
            result[i] = 0.0
        else:
            t_mp = mpf(str(t))
            result[i] = float(
                mp.invertlaplace(C_out_laplace, t_mp, method=method)
            )

    return result


def get_LRM_analytical_reference(solution_times, dps=50):
    """Analytical reference for the LRM benchmark (setting_Col1D_linLRM_1comp_benchmark1).

    Parameters matching setting_Col1D_linLRM_1comp_benchmark1.py:
    - col_dispersion = 0.0001
    - col_length = 1.0
    - total_porosity = 0.6
    - velocity = 0.03333...
    - ka = 1.0, kd = 1.0 (kinetic)
    - Pulse inlet: c=1.0 from t=0 to t=10
    """
    params = {
        'velocity': 1.0 / 30.0,
        'col_dispersion': 1e-4,
        'col_length': 1.0,
        'total_porosity': 0.6,
        'ka': 1.0,
        'kd': 1.0,
        'c_in': 1.0,
        't_inj': 10.0,
    }
    return compute_analytical_outlet('LRM', params, solution_times, dps=dps)


def get_LRMP_analytical_reference(solution_times, dps=50):
    """Analytical reference for the LRMP benchmark (setting_Col1D_lin_1comp with HOMOGENEOUS_PARTICLE).

    Parameters matching setting_Col1D_lin_1comp_benchmark1.py with particle_type='HOMOGENEOUS_PARTICLE':
    - col_dispersion = 5.75e-08
    - col_length = 0.014
    - col_porosity = 0.37
    - velocity = 0.000575
    - film_diffusion = 6.9e-06
    - par_radius = 4.5e-05
    - par_porosity = 0.75
    - ka = 3.55, kd = 0.1 (kinetic)
    - Pulse inlet: c=1.0 from t=0 to t=10
    """
    params = {
        'velocity': 5.75e-4,
        'col_dispersion': 5.75e-8,
        'col_length': 0.014,
        'col_porosity': 0.37,
        'par_porosity': 0.75,
        'film_diffusion': 6.9e-6,
        'par_radius': 4.5e-5,
        'ka': 3.55,
        'kd': 0.1,
        'c_in': 1.0,
        't_inj': 10.0,
    }
    return compute_analytical_outlet('LRMP', params, solution_times, dps=dps)


def get_GRM_analytical_reference(solution_times, surface_diffusion=0.0, dps=50):
    """Analytical reference for the GRM benchmark (setting_Col1D_lin_1comp with GENERAL_RATE_PARTICLE).

    Parameters matching setting_Col1D_lin_1comp_benchmark1.py with particle_type='GENERAL_RATE_PARTICLE':
    - col_dispersion = 5.75e-08
    - col_length = 0.014
    - col_porosity = 0.37
    - velocity = 0.000575
    - film_diffusion = 6.9e-06
    - par_radius = 4.5e-05
    - par_porosity = 0.75
    - pore_diffusion = 6.07e-11
    - ka = 3.55, kd = 0.1 (kinetic)
    - Pulse inlet: c=1.0 from t=0 to t=10
    """
    params = {
        'velocity': 5.75e-4,
        'col_dispersion': 5.75e-8,
        'col_length': 0.014,
        'col_porosity': 0.37,
        'par_porosity': 0.75,
        'film_diffusion': 6.9e-6,
        'par_radius': 4.5e-5,
        'pore_diffusion': 6.07e-11,
        'surface_diffusion': surface_diffusion,
        'ka': 3.55,
        'kd': 0.1,
        'c_in': 1.0,
        't_inj': 10.0,
    }
    return compute_analytical_outlet('GRM', params, solution_times, dps=dps)
