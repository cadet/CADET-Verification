r"""
PTC vs. NLEQ-RES robustness comparison on a Hill-kinetics activation
cascade -- a reaction network DESIGNED BY CONSTRUCTION (not fit to a
published data set) so that Newton's basin of attraction is provably
limited near the equilibrium, unlike the SMA benchmark (sma_robustness.py,
where a correctly implemented Newton is robust everywhere) or the
pH-buffer model (ph_buffer_robustness.py, where the Jacobian is singular
everywhere and Newton is bad everywhere, not just far away).

Chemical context
-----------------
Hill kinetics describe cooperative binding/catalysis: the reaction rate
grows with the effector concentration but saturates once binding sites
are occupied (classic examples: O2 binding to hemoglobin, allosteric
enzymes, cooperative transcription-factor binding). Here four species
:math:`X_1,\dots,X_4` form a feedforward activation cascade, the
standard motif for enzyme/signalling cascades and gene-expression
cascades: :math:`X_1` is supplied at a constant rate, each :math:`X_i`
catalytically activates production of :math:`X_{i+1}` following Hill
kinetics, and every species is cleared by its own saturating
(Michaelis-Menten) degradation:

.. math::
    R_0:\ \varnothing \to X_1, \qquad
    R_{2i-1}:\ X_i \to \varnothing, \qquad
    R_{2i}:\ X_i \xrightarrow{\text{cat.}} X_i + X_{i+1}

with stoichiometric matrix (rows :math:`X_1..X_4`, columns
:math:`R_0,\dots,R_7`)

.. math::
    S = \begin{pmatrix}
    1 & -1 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & -1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & -1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 1 & -1
    \end{pmatrix}

and rate vector (Hill functions for activation and degradation)

.. math::
    v(x) = \begin{pmatrix}
    V_0 \\
    D_1 x_1^{m_1}/(L_1^{m_1}+x_1^{m_1}) \\
    V_2 x_1^{n_2}/(K_2^{n_2}+x_1^{n_2}) \\
    D_2 x_2^{m_2}/(L_2^{m_2}+x_2^{m_2}) \\
    V_3 x_2^{n_3}/(K_3^{n_3}+x_2^{n_3}) \\
    D_3 x_3^{m_3}/(L_3^{m_3}+x_3^{m_3}) \\
    V_4 x_3^{n_4}/(K_4^{n_4}+x_3^{n_4}) \\
    D_4 x_4^{m_4}/(L_4^{m_4}+x_4^{m_4})
    \end{pmatrix}.

The steady state solves :math:`F(x) = S\,v(x) = 0`, implemented directly
(not via S) in F(x)/J(x) below.

Why Newton struggles far from x_ref
-------------------------------------
Every entry of :math:`v` saturates, so :math:`F` stays bounded as
:math:`\|x\|\to\infty` while :math:`J = S\,\partial v/\partial x \to 0`
(Hill-function derivatives vanish away from half-saturation). Newton's
implied step :math:`J^{-1}F` then blows up and overshoots for distant
starting points, whereas PTC's :math:`(I-tJ)\Delta x = F` with
SER-adaptive :math:`t` stays regularized regardless of how small
:math:`J` is. The cascade is feedforward, so :math:`J` is triangular and
its eigenvalues are exactly its (negative) diagonal entries -- a stable
node by construction, unlike an earlier cyclic variant that produced a
saddle. Rate constants satisfy :math:`D_i>V_i`, which guarantees
compute_x_ref() can find the equilibrium by simple sequential bisection.
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


N = 4
V0 = 6.0                                 # species 1: constant input rate
V = np.array([0.0, 10.0, 14.0, 11.0])    # activation rate (species 2-4)
K = np.array([1.0, 3.0, 5.0, 3.5])       # activation half-saturation
HILL_N = np.array([1, 4, 4, 4])          # Hill coefficient (activation)
D = np.array([9.0, 12.0, 16.0, 13.0])    # max degradation rate (D_i > V_i everywhere)
L = np.array([3.0, 2.5, 4.0, 3.0])       # degradation half-saturation
HILL_M = np.array([3, 3, 3, 3])          # Hill coefficient (degradation)

STOFFE = ['species1', 'species2', 'species3', 'species4']


def F(x):
    # Failed trial iterates during the robustness sweep can be very far
    # from x_ref (that is the point of the test); the resulting overflow
    # in intermediate powers is expected and harmless (Python floats
    # saturate to inf, which downstream isfinite() checks catch), so it
    # is suppressed here rather than left to spam the console.
    with np.errstate(over='ignore', invalid='ignore'):
        prod = np.empty(N)
        prod[0] = V0
        for i in range(1, N):
            xm = x[i - 1]
            prod[i] = V[i] * xm ** HILL_N[i] / (K[i] ** HILL_N[i] + xm ** HILL_N[i])
        cons = D * x ** HILL_M / (L ** HILL_M + x ** HILL_M)
        return prod - cons


def J(x):
    with np.errstate(over='ignore', invalid='ignore'):
        Jm = np.zeros((N, N))
        dcons_dx = D * HILL_M * L ** HILL_M * x ** (HILL_M - 1) / (L ** HILL_M + x ** HILL_M) ** 2
        for i in range(N):
            Jm[i, i] -= dcons_dx[i]
        for i in range(1, N):
            xm = x[i - 1]
            dprod = (V[i] * HILL_N[i] * K[i] ** HILL_N[i] * xm ** (HILL_N[i] - 1)
                     / (K[i] ** HILL_N[i] + xm ** HILL_N[i]) ** 2)
            Jm[i, i - 1] += dprod
        return Jm


def compute_x_ref():
    """Solve the feedforward cascade sequentially (each step a monotonic
    scalar equation), guaranteed solvable since D_i > V_i for all i."""
    x_ref = np.zeros(N)
    x_ref[0] = brentq(
        lambda x1: V0 - D[0] * x1 ** HILL_M[0] / (L[0] ** HILL_M[0] + x1 ** HILL_M[0]),
        1e-8, 1e8, xtol=1e-14, rtol=1e-14,
    )
    for i in range(1, N):
        prod_i = V[i] * x_ref[i - 1] ** HILL_N[i] / (K[i] ** HILL_N[i] + x_ref[i - 1] ** HILL_N[i])
        x_ref[i] = brentq(
            lambda xi, p=prod_i, i=i: p - D[i] * xi ** HILL_M[i] / (L[i] ** HILL_M[i] + xi ** HILL_M[i]),
            1e-8, 1e8, xtol=1e-14, rtol=1e-14,
        )
    return x_ref


def ptc_solve(x0, t0=1.0, max_iter=200, eps=1e-12, allow_initial_increase=True):
    """PTC with SER step-size control (matches the already-validated
    ptc_algorithm.py logic used for the SMA and pH-buffer models)."""
    x = np.asarray(x0, dtype=float).copy()
    t = t0
    try:
        res = la.norm(F(x))
    except Exception:
        return x, np.inf
    if not np.isfinite(res):
        return x, np.inf

    for i in range(max_iter):
        if res <= eps:
            break
        try:
            dx = la.solve(np.eye(N) - t * J(x), F(x))
        except la.LinAlgError:
            break
        if not np.all(np.isfinite(dx)):
            break

        x1 = x + t * dx
        if not np.all(np.isfinite(x1)):
            break
        try:
            res1 = la.norm(F(x1))
        except Exception:
            break
        if not np.isfinite(res1):
            break

        if res1 >= res and not (allow_initial_increase and i < 3):
            break

        try:
            t1 = t * la.norm(F(x)) / la.norm(F(x1))
        except Exception:
            break
        if not np.isfinite(t1) or t1 <= 0.0:
            break

        x, res, t = x1, res1, t1

    return x, res


def nleq_res_solve(x0, max_iter=100, res_tol=1e-10, damping=1.0, min_damping=1e-4,
                    max_stall_iter=200):
    """NLEQ-RES ported from CADET-Core's adaptiveTrustRegionNewtonMethod()
    (see atrn_res.py for the same algorithm applied to the SMA model, with
    notes on the corrected mu formula)."""
    x = np.asarray(x0, dtype=float).copy()

    mu = 0.0
    last_residual_norm = 0.0
    try:
        last_residual = F(x)
    except Exception:
        return x, np.inf
    residual_norm = la.norm(last_residual)
    if not np.isfinite(residual_norm):
        return x, np.inf

    for k_iter in range(max_iter):
        if residual_norm <= res_tol:
            return x, residual_norm

        try:
            dx = la.solve(J(x), last_residual)
        except la.LinAlgError:
            return x, residual_norm
        if not np.all(np.isfinite(dx)):
            return x, residual_norm

        if k_iter > 0:
            mu *= last_residual_norm / residual_norm
            damping = min(1.0, mu)
        last_residual_norm = residual_norm

        stalled = True
        for _inner in range(max_stall_iter):
            if damping < min_damping:
                stalled = False
                break

            trial_point = x - damping * dx
            try:
                residual_mem = F(trial_point)
            except Exception:
                residual_mem = None

            if residual_mem is None or not np.all(np.isfinite(residual_mem)):
                trial_norm = np.inf
                theta = np.inf
                mu = 0.0
            else:
                trial_norm = la.norm(residual_mem)
                theta = trial_norm / last_residual_norm
                factor = 1.0 - damping
                correction_norm = la.norm(residual_mem - factor * last_residual)
                mu = (np.inf if correction_norm == 0.0
                      else 0.5 * last_residual_norm * damping ** 2 / correction_norm)

            if theta >= 1.0:
                damping = min(mu, 0.5 * damping)
                continue

            damping_new = min(1.0, mu)
            if damping_new >= 4.0 * damping:
                damping = damping_new
                continue

            stalled = False
            break

        if damping < min_damping or stalled:
            return x, residual_norm

        x = trial_point
        residual_norm = trial_norm
        last_residual = residual_mem

    return x, residual_norm


def generate_points(x_ref, n_samples, r_min, r_max, rng):
    """Points on a spherical shell, scaled per-component by the
    equilibrium's own magnitude (so a given radius perturbs each species
    roughly proportionally, as in the other two robustness scripts)."""
    dirs = rng.standard_normal((N, n_samples))
    dirs /= la.norm(dirs, axis=0)
    rad = (rng.uniform(r_min ** N, r_max ** N, n_samples)) ** (1.0 / N)
    ellipse = np.diag(x_ref)
    disp = ellipse @ (dirs * rad[None, :])
    return x_ref[:, None] + disp


def run_sweep(x_ref, n_samples=400, r_min0=0.0, r_max0=0.05, stepsize=0.3,
              max_distance=15, normmin=1e-4, max_iter=200, seed=13):
    rng = np.random.default_rng(seed)
    r_min, r_max = r_min0, r_max0

    results = []
    for step in range(1, max_distance + 1):
        r_min += stepsize
        r_max += stepsize

        pts = generate_points(x_ref, n_samples, r_min, r_max, rng)
        n_counter = 0
        n_ptc = 0
        n_newton = 0

        for i in range(n_samples):
            x0 = pts[:, i]
            if np.any(x0 < 0):
                continue
            n_counter += 1

            _, res_ptc = ptc_solve(x0, max_iter=max_iter)
            if res_ptc < normmin:
                n_ptc += 1

            _, res_new = nleq_res_solve(x0, max_iter=max_iter)
            if res_new < normmin:
                n_newton += 1

        rate_ptc = n_ptc / n_counter if n_counter else float('nan')
        rate_newton = n_newton / n_counter if n_counter else float('nan')
        results.append((r_min, n_counter, rate_ptc, rate_newton))
        print(f"step {step}: r_min={r_min:.2f} n={n_counter} "
              f"PTC={rate_ptc:.3f} Newton={rate_newton:.3f}")

    return pd.DataFrame(results, columns=['r_min', 'n_counter', 'PTC', 'Newton'])


def plot_results(df, save_path='cascade_reaction_robustness.png'):
    plt.figure()
    plt.plot(df['r_min'], df['PTC'], marker='o', label='PTC')
    plt.plot(df['r_min'], df['Newton'], marker='o', label='NLEQ-RES (Newton)')
    plt.xlabel('Innerer Radius der sphärischen Schale (Abstand zu x_ref)')
    plt.ylabel('Konvergenzrate')
    plt.title('Saturating Hill-Kaskade: PTC vs. NLEQ-RES')
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot gespeichert: {save_path}")


if __name__ == "__main__":
    x_ref = compute_x_ref()
    print("x_ref =", x_ref)
    print("F(x_ref) =", F(x_ref))
    print("cond(J(x_ref)) =", np.linalg.cond(J(x_ref)))
    print("eig(J(x_ref)) =", np.linalg.eigvals(J(x_ref)), "(all real & negative by construction)")
    print()
    df = run_sweep(x_ref)
    df.to_csv('Data_Cascade_Robustheit.csv', index=False)
    plot_results(df)
