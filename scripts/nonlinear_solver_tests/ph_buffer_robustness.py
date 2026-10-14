"""
PTC vs. NLEQ-RES robustness comparison on the pH-buffer reaction model
(thesis "Beispiel 3", implemented in PTC_example3.py).

Unlike the SMA binding model used elsewhere in this folder, this system has
a GENUINELY singular Jacobian everywhere (rank 10 of 19, from real mass-
and charge-conservation laws encoded in the stoichiometric matrix `sm`,
which has only 10 columns) -- not merely a numerically severe but
technically full-rank conditioning problem. This is the textbook "dynamic
invariant" scenario motivating PTC over plain/damped Newton (Deuflhard,
2011, Sec. 2.1; see also Bemerkung 2.6).

PTC solves (I - t*J)*dx = F: adding the identity regularizes the singular
J for any t, by construction.

NLEQ-RES (the adaptive trust-region damped Newton this thesis compares
against, ported from CADET-Core's AdaptiveTrustRegionNewton.hpp) solves
J*dx = F directly: this linear system is singular at every point, so the
solve either raises LinAlgError or is numerically meaningless.
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PTC_example3 import jacobi as _ph_jacobi, fkt_wert as _ph_fkt_wert


K_H = np.array([
    2.4e-8, 2.29e-6, 1.31e-6, 8.47e-6, 1.02e-4,
    3.28e-4, 1.86e-7, 4.6e-3, 1.75e-2, 1.38e-1,
])
K_Z = np.ones(10)
SM = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])
ARGS = [K_H, K_Z, SM]
REL_STOFFE = [
    'Dap+', 'Dap2+', 'Dea+', 'Tris+', 'Imi+',
    'Bts+', 'Pip+', 'Pip2+', 'Ace', 'Lac',
]

STOFFE = [
    'Dap', 'Dap+', 'Dap2+', 'Dea', 'Dea+', 'Tris', 'Tris+', 'Imi', 'Imi+',
    'Bts', 'Bts+', 'Pip', 'Pip+', 'Pip2+', 'Ace-', 'Ace', 'Lac-', 'Lac', 'H+',
]

X0_INITIAL = np.array([
    10., 0, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 0, 0, 10, 0, 10, 10 ** -0.5,
])


def F(x):
    return _ph_fkt_wert(ARGS, x, REL_STOFFE)


def J(x):
    return _ph_jacobi(ARGS, x, REL_STOFFE)


def compute_x_ref(x0=X0_INITIAL, t0=1.0, max_iter=60, eps=1e-13):
    """Refine the thesis's ~4-digit equilibrium to full precision via PTC/SER."""
    x = np.asarray(x0, dtype=float).copy()
    t = t0
    res = la.norm(F(x))
    for i in range(max_iter):
        I = np.eye(len(x))
        dx = la.solve(I - t * J(x), F(x))
        x1 = x + t * dx
        res1 = la.norm(F(x1))
        if res1 >= res and i >= 3:
            break
        t = t * la.norm(F(x)) / la.norm(F(x1))
        x, res = x1, res1
        if res < eps:
            break
    return x, res


def ptc_solve(x0, t0=1.0, max_iter=100, eps=1e-10, allow_initial_increase=True):
    """
    PTC with SER step-size control, matching the already-validated
    ptc_algorithm.py logic: no artificial abort on transient negative
    components, tolerate residual increase in the first few iterations.
    Returns (x_final, final_residual_norm).
    """
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
            I = np.eye(len(x))
            dx = la.solve(I - t * J(x), F(x))
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


def nleq_res_solve(x0, max_iter=100, res_tol=2e-2, damping=1.0, min_damping=1e-4,
                    max_stall_iter=200):
    """
    NLEQ-RES (residual based adaptive trust-region damped Newton), ported
    from CADET-Core's adaptiveTrustRegionNewtonMethod() -- see atrn_res.py
    for the same algorithm applied to the SMA model, with notes on the
    corrected mu formula. Returns (x_final, final_residual_norm).
    """
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
            Jx = J(x)
            dx = la.solve(Jx, last_residual)
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

                if correction_norm == 0.0:
                    mu = np.inf
                else:
                    mu = 0.5 * last_residual_norm * damping ** 2 / correction_norm

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


def generate_points(x_ref, n_samples, r_min, r_max, rng, floor=1e-2):
    """
    Points on a spherical shell around x_ref, radius in [r_min, r_max],
    scaled per-component by max(|x_ref_i|, floor) so a given radius
    perturbs each species roughly proportionally to its own natural
    scale (mirrors the sphToEllipse construction used for the SMA model,
    generalized to this model's much wider range of magnitudes).
    """
    n = len(x_ref)
    dirs = rng.standard_normal((n, n_samples))
    dirs /= la.norm(dirs, axis=0)
    rad = (rng.uniform(r_min ** n, r_max ** n, n_samples)) ** (1.0 / n)
    ellipse = np.maximum(np.abs(x_ref), floor)
    disp = ellipse[:, None] * dirs * rad[None, :]
    return x_ref[:, None] + disp


def run_sweep(x_ref, n_samples=300, r_min0=0.0, r_max0=0.1, stepsize=0.5,
              max_distance=20, normmin=0.02, max_iter=100, seed=0):
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

    return pd.DataFrame(
        results, columns=['r_min', 'n_counter', 'PTC', 'Newton']
    )


def plot_results(df, save_path='ph_buffer_robustness.png'):
    plt.figure()
    plt.plot(df['r_min'], df['PTC'], marker='o', label='PTC')
    plt.plot(df['r_min'], df['Newton'], marker='o', label='NLEQ-RES (Newton)')
    plt.xlabel('Innerer Radius der sphärischen Schale (Abstand zu x_ref)')
    plt.ylabel('Konvergenzrate')
    plt.title('pH-Puffer-Modell: PTC vs. NLEQ-RES (singuläre Jacobi-Matrix)')
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot gespeichert: {save_path}")


if __name__ == "__main__":
    x_ref, res = compute_x_ref()
    print("x_ref =", x_ref)
    print("residual at x_ref =", res)
    print("rank(J(x_ref)) =", np.linalg.matrix_rank(J(x_ref)), "/", len(x_ref))
    print()
    df = run_sweep(x_ref, n_samples=1000, stepsize=0.3, max_distance=15)
    df.to_csv('Data_pH_Buffer_Robustheit.csv', index=False)
    plot_results(df)
