
import numpy as np
import matplotlib.pyplot as plt


def run_implicit_euler(
    F,
    JF,
    x0,
    h,
    tol=1e-8,
    inner_tol=1e-6,
    step_tol=1e-12,
    innerstep_tol=1e-12,
    max_steps=5000,
    max_inner_iter=100,
    verbose=False,
):
    x = np.array(x0, dtype=float, copy=True)

    n_steps = 0
    n_lgs = 0
    terminated = False
    reason = None

    # Initial residual
    try:
        Fx = F(x)
        res = np.linalg.norm(Fx)

        if verbose:
            history = {
                "values": [x.copy()],
                "residuals": [res],
                "inner_iters": [],
            }
        else:
            history = None

    except Exception:
        reason = "unphysical"

        if verbose:
            return x, n_steps, n_lgs, terminated, reason, history
        else:
            return x, terminated, reason

    # ------------------------------------------------------
    # Outer implicit Euler iterations
    # ------------------------------------------------------
    while n_steps < max_steps:

        if res <= tol:
            terminated = True
            reason = "residual_tol"
            break

        x_old = x.copy()

        # Implicit Euler equation:
        #
        # G(z) = z - x_old - h * F(z)
        #
        # JG(z) = I - h * JF(z)

        def G(z):
            return z - x_old - h * F(z)

        def JG(z):
            n = len(z)
            return np.eye(n) - h * JF(z)

        # --------------------------------------------------
        # Inner Newton solve
        # --------------------------------------------------
        (
            x_new,
            inner_iters,
            inner_lgs,
            inner_terminated,
            inner_reason,
            _,
        ) = run_newton(
            G,
            JG,
            x,
            tol=inner_tol,
            step_tol=innerstep_tol,
            max_iter=max_inner_iter,
            verbose=True,
        )

        n_lgs += inner_lgs

        if not inner_terminated:
            terminated = False
            reason = inner_reason
            break

        # --------------------------------------------------
        # Outer step
        # --------------------------------------------------
        outer_step = np.linalg.norm(x_new - x)

        x = x_new
        n_steps += 1

        # Update residual
        try:
            Fx = F(x)
            res = np.linalg.norm(Fx)

        except Exception:
            reason = "unphysical"
            break

        if verbose:
            history["values"].append(x.copy())
            history["residuals"].append(res)
            history["inner_iters"].append(inner_iters)

        # --------------------------------------------------
        # Outer stopping criteria
        # --------------------------------------------------
        if res < tol:
            terminated = True
            reason = "residual_tol"
            break

        if outer_step < step_tol:
            terminated = True
            reason = "step_tol"
            break

    if not terminated and reason is None:
        reason = "max_iter"

    if verbose:
        return x, n_steps, n_lgs, terminated, reason, history
    else:
        return x, terminated, reason


def run_newton(
    F,
    JF,
    x0,
    tol=1e-8,
    step_tol=1e-12,
    max_iter=500,
    verbose=False,
):
    x = np.array(x0, dtype=float, copy=True)

    n_iters = 0
    n_lgs = 0
    terminated = False
    reason = None

    if verbose:
        history = {
            "values": [x.copy()],
            "residuals": [],
            "steps": [],
        }
    else:
        history = None

    while n_iters < max_iter:

        # --------------------------------------------------
        # Residual
        # --------------------------------------------------
        try:
            Fx = F(x)
        except Exception:
            reason = "unphysical"
            break

        res = np.linalg.norm(Fx)

        if verbose:
            history["residuals"].append(res)

        # --------------------------------------------------
        # Stopping criterion 1
        # --------------------------------------------------
        if res < tol:
            terminated = True
            reason = "residual_tol"
            break

        # --------------------------------------------------
        # Jacobian
        # --------------------------------------------------
        try:
            JFx = JF(x)
        except Exception:
            reason = "unphysical"
            break

        # --------------------------------------------------
        # Newton step
        # --------------------------------------------------
        try:
            # Julia: s = JFx \ (-Fx)
            s = np.linalg.solve(JFx, -Fx)
            n_lgs += 1

        except np.linalg.LinAlgError:
            reason = "linear_solve_failed"
            break

        # --------------------------------------------------
        # Update step
        # --------------------------------------------------
        step = np.linalg.norm(s)

        x = x + s
        n_iters += 1

        if verbose:
            history["values"].append(x.copy())
            history["steps"].append(step)

        # --------------------------------------------------
        # Stopping criterion 2
        # --------------------------------------------------
        if step < step_tol:
            terminated = True
            reason = "step_tol"
            break

    if not terminated and reason is None:
        reason = "max_iter"

    if verbose:
        return x, n_iters, n_lgs, terminated, reason, history
    else:
        return x, terminated, reason


def run_ptc_posF(
    F,
    JF,
    x0,
    delta0=1e-2,
    tol=1e-8,
    step_tol=1e-12,
    max_iter=500,
    verbose=False,
):
    x = np.array(x0, dtype=float, copy=True)
    delta = delta0

    n_iters = 0
    n_lgs = 0
    terminated = False
    reason = None

    if verbose:
        history = {
            "values": [],
            "residuals": [],
            "timesteps": [],
        }
    else:
        history = None

    # Initial residual
    try:
        Fx = F(x)
        res = np.linalg.norm(Fx)

        if verbose:
            history["values"].append(x.copy())
            history["residuals"].append(res)
            history["timesteps"].append(delta)

    except Exception:
        reason = "unphysical"

        if verbose:
            return x, n_iters, n_lgs, terminated, reason, history
        else:
            return x, terminated, reason

    while n_iters < max_iter and res > tol:

        # --------------------------------------------------
        # Jacobian
        # --------------------------------------------------
        try:
            JFx = JF(x)
        except Exception:
            reason = "unphysical"
            break

        # --------------------------------------------------
        # PTC step
        #
        # Julia:
        # s = ((1/delta)*I - JFx) \ Fx
        #
        # Note the sign convention from the original code.
        # --------------------------------------------------
        try:
            A = (1.0 / delta) * np.eye(len(x)) - JFx
            s = np.linalg.solve(A, Fx)
            n_lgs += 1

        except np.linalg.LinAlgError:
            reason = "linear_solve_failed"
            break

        step = np.linalg.norm(s)

        # --------------------------------------------------
        # Evaluate new residual
        # --------------------------------------------------
        try:
            Fx_new = F(x + s)
        except Exception:
            reason = "unphysical"
            break

        res_new = np.linalg.norm(Fx_new)

        # Update timestep
        delta = delta * res / res_new

        x = x + s
        Fx = Fx_new
        res = res_new

        n_iters += 1

        if verbose:
            history["values"].append(x.copy())
            history["residuals"].append(res)
            history["timesteps"].append(delta)

        # --------------------------------------------------
        # Step stopping criterion
        # --------------------------------------------------
        if step < step_tol:
            terminated = True
            reason = "step_tol"
            break

    if res <= tol:
        terminated = True
        reason = "residual_tol"

    if not terminated and reason is None:
        reason = "max_iter"

    if verbose:
        return x, n_iters, n_lgs, terminated, reason, history
    else:
        return x, terminated, reason


# ---------------------------------------------------------------------------
# Generate points using ellipsoidal scaling.
# Points with a negative component are discarded and NOT resampled.
# ---------------------------------------------------------------------------
def generate_points_paper(
    x_ref,
    n_samples,
    r_min,
    r_max,
    ellipse,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    x_ref = np.asarray(x_ref, dtype=float)
    ellipse = np.asarray(ellipse, dtype=float)

    d = len(x_ref)

    # Julia:
    # dirs = randn(rng, d, nSamples)
    dirs = rng.standard_normal((d, n_samples))

    # Normalize each column
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)

    # Uniform distribution in volume of d-dimensional spherical shell
    u = rng.random(n_samples)

    rad = (
        u * (r_max**d - r_min**d) + r_min**d
    ) ** (1.0 / d)

    # Julia:
    # disp = ellipse .* (dirs .* rad')
    disp = ellipse[:, None] * dirs * rad[None, :]

    points = []

    for i in range(n_samples):
        x = x_ref + disp[:, i]

        # Julia: all(>=(0), x)
        if np.all(x >= 0):
            points.append(x.copy())

    return points


# ---------------------------------------------------------------------------
# Robustness test exactly following the Julia implementation.
# ---------------------------------------------------------------------------
def robustness_test_paper(
    F,
    JF,
    x_ref,
    n_samples=1000,
    n_min=100,
    r_min0=0.0,
    r_max0=0.1,
    stepsize=0.1,
    normmin=0.05,
    max_distance=300,
    abs_tol=1e-2,
    ellipse=None,
    delta0_ptc=1e-1,
    delta0_ie=1e-1,
    tol=1e-10,
    max_iter=50,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if ellipse is None:
        ellipse = np.array([1000.0, 10.0, 10.0, 10.0])

    r_min = r_min0
    r_max = r_max0

    r_mins = []
    rate_ptc = []
    rate_newton = []
    rate_implicit_euler = []

    print(f"Maximal-Distanz: {max_distance}")

    for step in range(1, max_distance + 1):

        r_min += stepsize
        r_max += stepsize

        points = generate_points_paper(
            x_ref,
            n_samples,
            r_min,
            r_max,
            ellipse,
            rng=rng,
        )

        n_counter = len(points)

        if n_counter < n_min:
            break

        n_ptc = 0
        n_new = 0
        n_ie = 0

        for x0 in points:

            # --------------------------------------------------
            # 1. Pseudo-Transient Continuation
            # --------------------------------------------------
            try:
                x_ptc = run_ptc_posF(
                    F,
                    JF,
                    x0,
                    delta0=delta0_ptc,
                    tol=tol,
                    max_iter=max_iter,
                    verbose=False,
                )[0]

                ptc_ok = (
                    np.max(np.abs(x_ptc - x_ref)) < abs_tol
                )

            except Exception as e:
                # Julia only catches DomainError here.
                #
                # In Python we treat numerical/domain failures as
                # unsuccessful points. Other exceptions are re-raised.
                if isinstance(e, (ValueError, FloatingPointError)):
                    ptc_ok = False
                else:
                    raise

            if ptc_ok:
                n_ptc += 1

            # --------------------------------------------------
            # 2. Standard Newton
            # --------------------------------------------------
            try:
                x_new = run_newton(
                    F,
                    JF,
                    x0,
                    tol=tol,
                    max_iter=max_iter,
                    verbose=False,
                )[0]

                new_ok = (
                    np.max(np.abs(x_new - x_ref)) < abs_tol
                )

            except Exception as e:
                if isinstance(e, (ValueError, FloatingPointError)):
                    new_ok = False
                else:
                    raise

            if new_ok:
                n_new += 1

            # --------------------------------------------------
            # 3. Implicit Euler
            # --------------------------------------------------
            try:
                x_ie = run_implicit_euler(
                    F,
                    JF,
                    x0,
                    1e-1,
                    tol=tol,
                    max_steps=max_iter * 100,
                    verbose=False,
                )[0]

                ie_ok = (
                    np.max(np.abs(x_ie - x_ref)) < abs_tol
                )

            except Exception as e:
                if isinstance(e, (ValueError, FloatingPointError)):
                    ie_ok = False
                else:
                    raise

            if ie_ok:
                n_ie += 1

        r_mins.append(r_min)
        rate_ptc.append(n_ptc / n_counter)
        rate_newton.append(n_new / n_counter)
        rate_implicit_euler.append(n_ie / n_counter)

        print(
            f"step {step}: "
            f"r_min={r_min}, "
            f"nCounter={n_counter}, "
            f"PTC={rate_ptc[-1]}, "
            f"Newton={rate_newton[-1]}, "
            f"ImplicitEuler={rate_implicit_euler[-1]}"
        )

    return (
        r_mins,
        rate_ptc,
        rate_newton,
        rate_implicit_euler,
    )


def run_paper_reproduction():

    ka = np.array([
        0.0,
        35.5,
        1.59,
        7.7,
    ])

    kd = np.array([
        0.0,
        1000.0,
        1000.0,
        1000.0,
    ])

    nu = np.array([
        0.0,
        4.7,
        5.29,
        3.7,
    ])

    sigma = np.array([
        0.0,
        11.83,
        10.6,
        10.0,
    ])

    Lambda = 1200.0

    yCp = np.array([
        58.377002519964755,
        0.002935229673204726,
        0.01506102366722263,
        0.13523701213590386,
    ])

    F = lambda x: F_lew(
        x, ka, kd, nu, sigma, Lambda, yCp
    )

    JF = lambda x: JF_lew(
        x, ka, kd, nu, sigma, Lambda, yCp
    )

    # --------------------------------------------------
    # Preconditioning
    # --------------------------------------------------
    F_ref = F(yCp)

    scales = np.maximum(np.abs(F_ref), 1.0)

    V = np.diag(1.0 / scales)

    VF = lambda x: V @ F(x)
    VJF = lambda x: V @ JF(x)

    # --------------------------------------------------
    # Find equilibrium
    # --------------------------------------------------
    x_eq = run_newton(
        VF,
        VJF,
        yCp,
        tol=1e-14,
    )[0]

    print(
        "||VF(x_eq)|| =",
        np.linalg.norm(VF(x_eq))
    )

    print(
        "||F(x_eq)|| =",
        np.linalg.norm(F(x_eq))
    )

    # --------------------------------------------------
    # Robustness test
    # --------------------------------------------------
    (
        r_mins,
        rate_ptc,
        rate_newton,
        rate_euler,
    ) = robustness_test_paper(
        F,
        JF,
        x_eq,
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure()

    plt.plot(
        r_mins,
        rate_ptc,
        label="PTC",
        marker="o",
    )

    plt.plot(
        r_mins,
        rate_newton,
        label="NEW",
        marker="o",
    )

    plt.plot(
        r_mins,
        rate_euler,
        label="Euler",
        marker="o",
    )

    plt.xlabel("Innerer Radius der sphärischen Schale")
    plt.ylabel("Konvergenzrate")
    plt.title("Vergleich der beiden Verfahren")
    plt.legend()
    plt.show()

    return (
        r_mins,
        rate_ptc,
        rate_newton,
        plt,
    )


def F_lew(x, ka, kd, nu, sigma, Lambda, yCp):

    n = len(x)

    q0bar = x[0] - np.dot(sigma, x)

    c0 = yCp[0]

    fw = np.zeros_like(x, dtype=float)

    fw[0] = (
        -x[0]
        + Lambda
        - np.dot(nu, x)
    )

    for i in range(1, n):
        fw[i] = (
            -kd[i] * x[i] * c0**nu[i]
            + ka[i] * yCp[i] * q0bar**nu[i]
        )

    return fw


def JF_lew(x, ka, kd, nu, sigma, Lambda, yCp):

    n = len(x)

    q0bar = x[0] - np.dot(sigma, x)

    c0 = yCp[0]

    # Julia:
    #
    # ext_sigma[1] = 1
    # ext_sigma[2:end] = -sigma[2:end]
    #
    ext_sigma = np.empty(n)
    ext_sigma[0] = 1.0
    ext_sigma[1:] = -sigma[1:]

    J = np.zeros((n, n))

    J[0, 0] = -1.0
    J[0, 1:] = -nu[1:]

    for i in range(1, n):

        J[i, :] = (
            ka[i]
            * yCp[i]
            * nu[i]
            * q0bar**(nu[i] - 1)
            * ext_sigma
        )

        J[i, i] -= kd[i] * c0**nu[i]

    return J


run_paper_reproduction()