import numpy as np
import numpy.linalg as la

from smaBinding import fkt_wert, jacobi


def nleq_res(F, J, x0, eps=1e-10, lambda_min=1e-8, theta_max=0.5,
             maxiter=30, lambda0=1.0, scaling=None):
    """
    NLEQ-RES nonlinear equation solver.

    Solves
        F(x) = 0

    using a damped Newton method with residual-based step-size control
    following the NLEQ-RES algorithm.

    Parameters
    ----------
    F : callable
        Residual function F(x) -> array_like.
    J : callable
        Jacobian function J(x) -> 2-D array_like.
    x0 : array_like
        Initial iterate.
    eps : float, optional
        Required residual accuracy.
    lambda_min : float, optional
        Minimum allowed damping factor. If the predicted damping falls
        below this value, convergence is considered failed.
    theta_max : float, optional
        Threshold for switching to a quasi-Newton method. Since no QNRES
        implementation is supplied here, the switch is currently treated
        as acceptance of the full Newton step.
    maxiter : int, optional
        Maximum number of iterations.
    lambda0 : float, optional
        Initial damping factor, normally 1.
    scaling : array_like or None, optional
        Constant diagonal scaling vector D such that the norm used is
        ||D^-1 F||_2. If None, the ordinary 2-norm is used.

    Returns
    -------
    x : ndarray
        Approximate solution.
    residual_norm : float
        Norm of F(x).
    info : dict
        Diagnostic information.

    Raises
    ------
    ValueError
        If convergence fails.
    """

    x = np.asarray(x0, dtype=float).copy()

    if scaling is not None:
        scaling = np.asarray(scaling, dtype=float)

        if np.any(scaling <= 0):
            raise ValueError("Scaling muss strikt positiv sein.")

    def scaled_norm(v):
        """Scaled Euclidean norm."""
        v = np.asarray(v, dtype=float)

        if scaling is None:
            return la.norm(v, 2)

        return la.norm(v / scaling, 2)

    def safe_F(x):
        value = np.asarray(F(x), dtype=float)

        if value.ndim != 1:
            value = value.ravel()

        if not np.all(np.isfinite(value)):
            raise ValueError("Residuum enthält NaN oder Inf.")

        return value

    # Initial evaluation
    Fx = safe_F(x)
    fnorm = scaled_norm(Fx)

    if fnorm <= eps:
        return x, fnorm, {
            "converged": True,
            "iterations": 0,
            "reason": "initial residual below tolerance",
        }

    # Initial damping
    lam = float(lambda0)

    if not (0.0 < lam <= 1.0):
        raise ValueError("lambda0 muss im Bereich (0, 1] liegen.")

    previous_norm = None
    previous_lambda = lam

    for k in range(maxiter):

        # --------------------------------------------------------------
        # 1. Step k: convergence test
        # --------------------------------------------------------------
        if fnorm <= eps:
            return x, fnorm, {
                "converged": True,
                "iterations": k,
                "reason": "residual tolerance reached",
            }

        # Evaluate Jacobian
        Jx = np.asarray(J(x), dtype=float)

        if Jx.ndim != 2:
            raise ValueError("Jacobian muss eine 2-D Matrix sein.")

        if not np.all(np.isfinite(Jx)):
            raise ValueError("Jacobian enthält NaN oder Inf.")

        # Solve
        try:
            dx = la.solve(Jx, -Fx)
        except la.LinAlgError as exc:
            raise ValueError(
                f"Newton-System bei Iteration {k} singulär/"
                "schlecht konditioniert."
            ) from exc

        if not np.all(np.isfinite(dx)):
            raise ValueError("Newton-Schritt enthält NaN oder Inf.")

        # --------------------------------------------------------------
        # Prediction of damping factor for k > 0
        #
        # μ_k = ||F(x_{k-1})|| / ||F(x_k)|| * λ_{k-1}
        # λ_k = min(1, μ_k)
        # --------------------------------------------------------------
        if k > 0 and previous_norm is not None:
            if fnorm <= 0.0:
                lam = 1.0
            else:
                mu_prediction = (
                    previous_norm / fnorm
                ) * previous_lambda

                lam = min(1.0, mu_prediction)

        # Regularity test
        if lam < lambda_min:
            raise ValueError(
                f"NLEQ-RES Konvergenzfehler bei Iteration {k}: "
                f"lambda={lam:.3e} < lambda_min={lambda_min:.3e}"
            )

        # --------------------------------------------------------------
        # 2./3. Trial iterate and monitoring
        #
        # The damping factor can be repeatedly reduced. Therefore this
        # part is implemented as an inner loop.
        # --------------------------------------------------------------
        while True:

            # Trial iterate
            x_trial = x + lam * dx

            try:
                F_trial = safe_F(x_trial)
            except Exception:
                # Treat invalid function evaluations as an unsuccessful
                # trial step.
                theta = np.inf
                mu = 0.0
            else:
                trial_norm = scaled_norm(F_trial)

                # ------------------------------------------------------
                # Monitoring quantity
                #
                # Θ_k =
                # ||F(x_{k+1})|| / ||F(x_k)||
                # ------------------------------------------------------
                if fnorm == 0.0:
                    theta = 0.0
                else:
                    theta = trial_norm / fnorm

                # ------------------------------------------------------
                # μ_k =
                #   1 / (2 ||F(x_k)|| λ_k^2)
                #   * ||F(x_{k+1}) - (1-λ_k)F(x_k)||
                # ------------------------------------------------------
                if fnorm == 0.0 or lam == 0.0:
                    mu = 0.0
                else:
                    correction = (
                        F_trial - (1.0 - lam) * Fx
                    )

                    mu = (
                        scaled_norm(correction)
                        / (2.0 * fnorm * lam ** 2)
                    )

                if not np.isfinite(theta) or not np.isfinite(mu):
                    theta = np.inf
                    mu = 0.0

            # ----------------------------------------------------------
            # If Θ_k >= 1:
            #
            # λ'_k = min(μ_k, λ_k / 2)
            #
            # Then repeat the regularity test.
            # ----------------------------------------------------------
            if theta >= 1.0:

                new_lam = min(mu, 0.5 * lam)

                if new_lam < lambda_min:
                    raise ValueError(
                        f"NLEQ-RES Konvergenzfehler bei Iteration {k}: "
                        f"reduzierte Dämpfung={new_lam:.3e}"
                    )

                lam = new_lam
                continue

            # ----------------------------------------------------------
            # Otherwise:
            #
            # λ'_k = min(1, μ_k)
            # ----------------------------------------------------------
            new_lam = min(1.0, mu)

            # ----------------------------------------------------------
            # QNRES switch condition:
            #
            # if λ'_k = λ_k = 1 and Θ_k < Θmax
            #
            # The original algorithm switches to QNRES here. Since no
            # QNRES implementation is supplied, use the accepted Newton
            # step. This is equivalent to continuing with full Newton
            # steps rather than silently implementing a different method.
            # ----------------------------------------------------------
            if (
                new_lam == lam == 1.0
                and theta < theta_max
            ):
                # Accept full Newton step.
                pass

            # ----------------------------------------------------------
            # If λ'_k >= 4 λ_k:
            #
            # replace λ_k and repeat trial evaluation.
            # ----------------------------------------------------------
            if new_lam >= 4.0 * lam:
                lam = new_lam

                if lam < lambda_min:
                    raise ValueError(
                        f"NLEQ-RES Konvergenzfehler bei Iteration {k}: "
                        f"lambda={lam:.3e}"
                    )

                continue

            # ----------------------------------------------------------
            # Accept x_{k+1}
            # ----------------------------------------------------------
            x_new = x_trial
            Fx_new = F_trial
            fnorm_new = trial_norm

            break

        # --------------------------------------------------------------
        # Store quantities for next iteration
        # --------------------------------------------------------------
        previous_norm = fnorm
        previous_lambda = lam

        x = x_new
        Fx = Fx_new
        fnorm = fnorm_new

        # Physical/numerical sanity
        if not np.all(np.isfinite(x)) or not np.isfinite(fnorm):
            raise ValueError(
                f"Ungültiger Iterationswert bei Iteration {k}."
            )

    raise ValueError(
        f"NLEQ-RES erreichte nach {maxiter} Iterationen "
        f"keine Konvergenz. Residuum: {fnorm:.6e}"
    )


def main(x, maxiter=100):
    """
    NLEQ-RES-basierter Robustheitstest.

    Parameters
    ----------
    x : array_like
        Startpunkt für die Iteration (4-dimensionaler Vektor).

    maxiter : int, optional
        Maximale Anzahl an Iterationen (default: 100).

    Returns
    -------
    float
        Norm des Residuums an der approximierten Lösung.

    Raises
    ------
    ValueError
        Wenn keine Konvergenz erreicht wird oder der Startpunkt
        physikalisch nicht sinnvoll ist.
    """

    # Vorgegebene Konstanten für das Problem
    yCp = np.array([
        5.8377002519964755e+01,
        2.9352296732047269e-03,
        1.5061023667222226e-02,
        1.3523701213590386e-01
    ])

    kA = np.array([0.0, 35.5, 1.59, 7.7])
    kD = np.array([0.0, 1000.0, 1000.0, 1000.0])
    nu = np.array([0.0, 4.7, 5.29, 3.7])
    sigma = np.array([0.0, 11.83, 10.6, 10.0])
    Lambda = 1.2e3

    args = [yCp, kA, kD, nu, sigma, Lambda]

    # Startpunkt
    x0 = np.asarray(x, dtype=float).copy()

    # Prüfe Dimension
    if x0.ndim != 1 or len(x0) != 4:
        raise ValueError(
            "Der Startpunkt muss ein 4-dimensionaler Vektor sein."
        )

    # Prüfe physikalische Plausibilität
    if np.any(x0 < 0):
        raise ValueError("Negativer Startpunkt")

    # --------------------------------------------------------------
    # Wrapper für das konkrete Gleichungssystem
    # --------------------------------------------------------------
    def residual(x_trial):
        try:
            return np.asarray(
                fkt_wert(args, x_trial),
                dtype=float
            )
        except Exception as exc:
            raise ValueError(
                "Fehler bei der Berechnung des Residuums."
            ) from exc

    def jac(x_trial):
        try:
            return np.asarray(
                jacobi(args, x_trial),
                dtype=float
            )
        except Exception as exc:
            raise ValueError(
                "Fehler bei der Berechnung der Jacobi-Matrix."
            ) from exc

    # --------------------------------------------------------------
    # NLEQ-RES
    # --------------------------------------------------------------
    try:
        solution, final_res, info = nleq_res(
            F=residual,
            J=jac,
            x0=x0,
            eps=1e-10,
            lambda_min=1e-8,
            theta_max=0.5,
            maxiter=maxiter,
            lambda0=1.0
        )

    except Exception as exc:
        raise ValueError(
            f"NLEQ-RES konnte keine Konvergenz erreichen: {exc}"
        ) from exc

    # --------------------------------------------------------------
    # Finale Plausibilitätsprüfung
    # --------------------------------------------------------------
    if not np.all(np.isfinite(solution)):
        raise ValueError(
            "NLEQ-RES lieferte eine ungültige Lösung."
        )

    if np.any(solution < 0):
        raise ValueError(
            "NLEQ-RES lieferte eine physikalisch ungültige Lösung "
            "(negative Komponente)."
        )

    if not np.isfinite(final_res):
        raise ValueError(
            "NLEQ-RES lieferte ein ungültiges Residuum."
        )

    if final_res > 1e-10:
        raise ValueError(
            f"NLEQ-RES-Konvergenz nicht ausreichend: "
            f"Residuum = {final_res:.6e}"
        )

    return final_res