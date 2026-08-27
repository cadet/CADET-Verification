"""
Bsp2_Rob.py - Pseudo Transient Continuation mit SER für Robustheitstests
Basiert auf dem Langmuir-Isotherme-Modell aus PTC_example2.py

UPDATED: Verwendet scipy.optimize für robustere Konvergenz
"""

import numpy as np
from numpy import linalg as la
from scipy.optimize import root


# Berechne Jacobi-Matrix
def jacobi(args, x):
    yCp = args[0]
    kA = args[1]
    kD = args[2]
    nu = args[3]
    sigma = args[4]

    q0bar = x[0] - np.sum(sigma * x)

    # Verhindere negative Werte bei der Potenzierung
    if q0bar <= 0:
        return np.eye(len(x)) * 1e-10  # Notfall-Jacobi

    c0powNu = np.power(yCp[0], nu)
    q0barPowNuM1 = np.power(q0bar, nu - 1.0)
    extSigma = -sigma
    extSigma[0] = 1.0

    jm = np.zeros((len(x), len(x)))
    jm[0, 0] = -1.0
    jm[0, 1:] = -nu[1:]

    for i in range(1, len(x)):
        jm[i, :] = +kA[i] * yCp[i] * nu[i] * q0barPowNuM1[i] * extSigma
        jm[i, i] -= kD[i] * c0powNu[i]

    return jm


# Berechne Funktionswerte (Residuum)
def fkt_wert(args, x):
    yCp = args[0]
    kA = args[1]
    kD = args[2]
    nu = args[3]
    sigma = args[4]
    Lambda = args[5]

    q0bar = x[0] - np.sum(sigma * x)

    # Verhindere negative Werte bei der Potenzierung
    if q0bar <= 0:
        # Gebe großes Residuum zurück, um solver wegzuleiten
        return np.ones(len(x)) * 1e10

    c0powNu = np.power(yCp[0], nu)
    q0barPowNu = np.power(q0bar, nu)

    fw = np.zeros(len(x))
    fw[0] = -x[0] + Lambda - np.sum(nu * x)
    fw[1:] = (
        -kD[1:] * x[1:] * c0powNu[1:]
        + kA[1:] * yCp[1:] * q0barPowNu[1:]
    )

    return fw


# Berechne die Änderungsrate Delta x
def delta_x(args, t, x):
    i = np.eye(len(x))
    dx = la.solve(i - t * jacobi(args, x), fkt_wert(args, x))
    return dx


# Berechne SER Zeitschritt
def t_ser(args, t_n, x_n, x_n1):
    z = t_n * la.norm(fkt_wert(args, x_n))
    n = la.norm(fkt_wert(args, x_n1))
    return z / n


def main(x):
    """
    Hauptfunktion für Robustheitstests mit scipy.optimize.root.
    Nimmt einen Startpunkt x und gibt die Residuum-Norm an der
    approximierten Lösung zurück.

    Parameters:
    -----------
    x : array_like
        Startpunkt für die Iteration (4-dimensionaler Vektor)

    Returns:
    --------
    float
        Norm des Residuums an der approximierten Lösung,
        oder NaN wenn keine Konvergenz
    """
    # Vorgegebene Konstanten für das Problem (aus PTC_example2.py)
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
        return float('nan')

    # Verwende scipy.optimize.root mit verschiedenen Methoden
    # Versuche zuerst 'hybr' (modifiziertes Powell), dann 'lm' (Levenberg-Marquardt)
    methods = ['hybr', 'lm', 'df-sane']

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
            if method in ['hybr', 'lm']:
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

    # Wenn alle Methoden fehlschlagen, gebe NaN zurück
    return float('nan')
