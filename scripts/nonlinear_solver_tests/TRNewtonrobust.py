"""
TRNewtonrobust.py - Trust Region Newton-Löser für Robustheitstests
Verwendet scipy.optimize.root für robuste Konvergenz
"""

import numpy as np
from numpy import linalg as la
from scipy.optimize import root

from smaBinding import fkt_wert, jacobi

def main(x):
    """
    Trust Region Newton-Löser mit scipy.optimize.root für Robustheitstests.

    Verwendet verschiedene robuste Methoden von scipy.optimize.root,
    ähnlich dem NLEQ-Paket.

    Parameters:
    -----------
    x : array_like
        Startpunkt für die Iteration (4-dimensionaler Vektor)

    Returns:
    --------
    float
        Norm des Residuums an der approximierten Lösung,
        oder wirft ValueError wenn keine Konvergenz
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
    x0 = np.array(x, dtype=float)

    # Prüfe ob Startpunkt physikalisch sinnvoll ist
    if np.any(x0 < 0):
        raise ValueError("Negativer Startpunkt")

    # Verwende scipy.optimize.root mit trust region Methode
    # Dies entspricht am ehesten einem NLEQ-ähnlichen Verhalten
    methods = ['lm', 'hybr', 'df-sane']  # Levenberg-Marquardt zuerst

    for method in methods:
        try:
            # Wrapper-Funktion für scipy
            def residual(x_trial):
                try:
                    return fkt_wert(args, x_trial)
                except:
                    return np.ones(len(x_trial)) * 1e10

            # Jacobi-Funktion für Methoden, die sie unterstützen
            def jac(x_trial):
                try:
                    return jacobi(args, x_trial)
                except:
                    return np.eye(len(x_trial)) * 1e-10

            # Rufe scipy.optimize.root auf
            if method in ['lm', 'hybr']:
                result = root(residual, x0, method=method, jac=jac,
                            options={'maxiter': 100, 'xtol': 1e-10})
            else:
                result = root(residual, x0, method=method,
                            options={'maxiter': 100})

            # Prüfe ob Lösung erfolgreich war
            if result.success:
                # Berechne finales Residuum
                final_res = la.norm(residual(result.x))

                # Prüfe ob Lösung physikalisch sinnvoll ist
                if np.all(result.x >= 0) and not np.isnan(final_res) and not np.isinf(final_res):
                    return final_res

        except Exception:
            continue

    # Wenn alle Methoden fehlschlagen, werfe ValueError
    raise ValueError("Keine Konvergenz mit allen Methoden")
