import numpy as np
import numpy.linalg as la
import pandas as pd

from smaBinding import fkt_wert, jacobi

def delta_x(k, t, x):
    """
    Berechnet die PTC-Änderungsrate Δx aus

        (I - t * J(x)) Δx = F(x)

    Parameters
    ----------
    k : object
        Benutzerdefinierte Parameter des Problems.
    t : float
        Aktuelle PTC-Zeitschrittweite.
    x : ndarray
        Aktueller Zustand.

    Returns
    -------
    ndarray
        Änderungsrate Δx.
    """
    x = np.asarray(x, dtype=float)

    I = np.eye(len(x))
    J = np.asarray(jacobi(k, x), dtype=float)
    F = np.asarray(fkt_wert(k, x), dtype=float)

    A = I - t * J

    return la.solve(A, F)


def t_opt(k, t_n, x_n, x_n1, dx):
    """
    Berechnet die optimale nächste PTC-Zeitschrittweite.

    Entsprechend Gleichung (2.3.3) bzw. dem im Paper
    angegebenen Python-Code.
    """
    F_n = np.asarray(fkt_wert(k, x_n), dtype=float)
    F_n1 = np.asarray(fkt_wert(k, x_n1), dtype=float)

    z = abs(np.dot(dx, F_n - dx))
    n = 2.0 * la.norm(dx) * la.norm(F_n1 - dx)

    if not np.isfinite(z) or not np.isfinite(n) or n <= 0.0:
        return np.nan

    t_n1 = (z / n) * t_n

    return t_n1


def t_ser(k, t_n, x_n, x_n1):
    """
    Berechnet die nächste PTC-Zeitschrittweite mit
    der Switched Evolution Relaxation (SER).
    """
    F_n = np.asarray(fkt_wert(k, x_n), dtype=float)
    F_n1 = np.asarray(fkt_wert(k, x_n1), dtype=float)

    numerator = t_n * la.norm(F_n)
    denominator = la.norm(F_n1)

    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan

    if denominator <= 0.0:
        # Residuum ist bereits praktisch null.
        return t_n

    t_n1 = numerator / denominator

    return t_n1


def ptc_solver(
    k,
    x0,
    stoffe,
    t_n=1.0,
    max_iter=30,
    opt=False,
    eps=1e-10,
    allow_initial_increase=True,
):
    """
    Löst ein nichtlineares Gleichungssystem mit
    Pseudo-Transient Continuation (PTC).

    Das Verfahren basiert auf der im Paper beschriebenen
    Implementierung.

    Parameters
    ----------
    k : object
        Benutzerdefinierte Parameter des Problems.

    x0 : array_like
        Anfangszustand.

    stoffe : list
        Namen der Zustandsgrößen. Wird für die DataFrame-Spalten
        verwendet.

    t_n : float, default=1.0
        Initiale PTC-Zeitschrittweite.

    max_iter : int, default=100
        Maximale Anzahl von PTC-Iterationen.

    opt : bool, default=True
        True  -> optimale Zeitschrittweitenstrategie t_opt
        False -> SER-Heuristik t_ser

    eps : float, default=1e-10
        Abbruchschwelle für die Residuum-Norm.

    allow_initial_increase : bool, default=True
        Wenn True, wird eine Verschlechterung des Residuums in
        den ersten drei Iterationen toleriert, entsprechend der
        modifizierten Abbruchbedingung aus dem Paper.

    Returns
    -------
    pandas.DataFrame or None
        Iterationshistorie. None, wenn bereits der Startzustand
        ungültig ist oder ein grundlegender Fehler auftritt.
    """

    # ---------------------------------------------------------
    # Eingaben validieren
    # ---------------------------------------------------------

    x_n = np.asarray(x0, dtype=float).copy()

    if x_n.ndim != 1:
        return None

    if len(x_n) != len(stoffe):
        return None

    if not np.all(np.isfinite(x_n)):
        return None

    if np.any(x_n < 0.0):
        return None

    if not np.isfinite(t_n) or t_n <= 0.0:
        return None

    if max_iter < 0:
        return None

    if eps <= 0.0 or not np.isfinite(eps):
        return None

    # ---------------------------------------------------------
    # Hilfsfunktion für das Residuum
    # ---------------------------------------------------------

    def residual_norm(x):
        try:
            F = np.asarray(fkt_wert(k, x), dtype=float)

            if F.shape != x.shape:
                return np.inf

            if not np.all(np.isfinite(F)):
                return np.inf

            return float(la.norm(F))

        except Exception:
            return np.inf

    # ---------------------------------------------------------
    # Initialisierung
    # ---------------------------------------------------------

    res_n = residual_norm(x_n)

    if not np.isfinite(res_n):
        return None

    x_list = [x_n.copy()]
    t_list = [float(t_n)]
    res_list = [res_n]

    # Falls der Startpunkt bereits Lösung ist
    if res_n <= eps:
        df = pd.DataFrame(x_list, columns=stoffe)
        df["Residuum"] = res_list
        df["Zeitschrittweite"] = t_list
        df.index.name = "Iterationen"
        return df

    # ---------------------------------------------------------
    # PTC Iterationen
    # ---------------------------------------------------------

    for i in range(max_iter):

        # -----------------------------------------------------
        # Δx bestimmen
        # -----------------------------------------------------

        try:
            dx = delta_x(k, t_n, x_n)
            dx = np.asarray(dx, dtype=float)
        except Exception:
            break

        if dx.shape != x_n.shape or not np.all(np.isfinite(dx)):
            break

        # -----------------------------------------------------
        # Neuen Zustand berechnen
        #
        # x_(n+1) = x_n + t_n * Δx
        # -----------------------------------------------------

        x_n1 = x_n + t_n * dx

        if not np.all(np.isfinite(x_n1)):
            break

        # Physikalische Bedingung:
        # Konzentrationen/Zustände dürfen nicht negativ werden.
        if np.any(x_n1 < 0.0):
            break

        # -----------------------------------------------------
        # Neues Residuum
        # -----------------------------------------------------

        res_n1 = residual_norm(x_n1)

        if not np.isfinite(res_n1):
            break

        # -----------------------------------------------------
        # Modifizierte Abbruchbedingung
        #
        # In den ersten drei Iterationen darf das Residuum
        # steigen.
        # -----------------------------------------------------

        residual_increased = res_n1 >= res_n

        if residual_increased and (
            not allow_initial_increase or i >= 3
        ):
            # print("Abbruch: Keine Residual-Reduktion")  # Kommentiert für weniger Output
            break

        # -----------------------------------------------------
        # Nächste Zeitschrittweite bestimmen
        # -----------------------------------------------------

        try:
            if opt:
                t_n1 = t_opt(
                    k,
                    t_n,
                    x_n,
                    x_n1,
                    dx,
                )
            else:
                t_n1 = t_ser(
                    k,
                    t_n,
                    x_n,
                    x_n1,
                )
        except Exception:
            break

        # -----------------------------------------------------
        # Zeitschrittweite validieren
        # -----------------------------------------------------

        if not np.isfinite(t_n1) or t_n1 <= 0.0:
            break

        # -----------------------------------------------------
        # Ergebnisse speichern
        # -----------------------------------------------------

        x_list.append(x_n1.copy())
        res_list.append(res_n1)
        t_list.append(float(t_n1))

        # -----------------------------------------------------
        # Aktuellen Zustand übernehmen
        # -----------------------------------------------------

        x_n = x_n1
        res_n = res_n1
        t_n = t_n1

        # -----------------------------------------------------
        # Konvergenz
        # -----------------------------------------------------

        if res_n <= eps:
            break

    # ---------------------------------------------------------
    # DataFrame erzeugen
    # ---------------------------------------------------------

    df = pd.DataFrame(
        data=np.asarray(x_list),
        columns=stoffe,
    )

    df["Residuum"] = res_list
    df["Zeitschrittweite"] = t_list

    df.index.name = "Iterationen"

    return df

def main(
    x,
    t_n=1.0,
    max_iter=100,
    opt=True,
    eps=1e-10,
):
    """
    Hauptfunktion für Robustheitstests mit PTC.

    Nimmt einen Startpunkt x und gibt die Residuum-Norm an der
    approximierten Lösung zurück.

    Parameters
    ----------
    x : array_like
        Startpunkt für die Iteration (4-dimensionaler Vektor).

    t_n : float, default=1.0
        Initiale PTC-Zeitschrittweite.

    max_iter : int, default=100
        Maximale Anzahl an PTC-Iterationen.

    opt : bool, default=True
        Zeitschrittweitenstrategie:
            True  -> optimale Strategie t_opt
            False -> SER-Heuristik t_ser

    eps : float, default=1e-10
        Abbruchschwelle für das Residuum.

    Returns
    -------
    float
        Norm des Residuums an der approximierten Lösung,
        oder NaN wenn keine gültige Konvergenz erreicht wurde.
    """

    # Vorgegebene Konstanten für das Problem
    # (aus PTC_example2.py)

    yCp = np.array([
        5.8377002519964755e+01,
        2.9352296732047269e-03,
        1.5061023667222226e-02,
        1.3523701213590386e-01
    ])

    kA = np.array([
        0.0,
        35.5,
        1.59,
        7.7
    ])

    kD = np.array([
        0.0,
        1000.0,
        1000.0,
        1000.0
    ])

    nu = np.array([
        0.0,
        4.7,
        5.29,
        3.7
    ])

    sigma = np.array([
        0.0,
        11.83,
        10.6,
        10.0
    ])

    Lambda = 1.2e3

    args = [
        yCp,
        kA,
        kD,
        nu,
        sigma,
        Lambda,
    ]

    # ---------------------------------------------------------
    # Startpunkt
    # ---------------------------------------------------------

    x0 = np.asarray(x, dtype=float)

    # Physikalische Plausibilität
    if x0.ndim != 1:
        return float("nan")

    if len(x0) != 4:
        return float("nan")

    if not np.all(np.isfinite(x0)):
        return float("nan")

    if np.any(x0 < 0.0):
        return float("nan")

    # ---------------------------------------------------------
    # Namen der Zustandsgrößen
    #
    # Diese Namen ggf. an die ursprüngliche PTC_example2.py
    # anpassen.
    # ---------------------------------------------------------

    stoffe = [
        "Stoff 1",
        "Stoff 2",
        "Stoff 3",
        "Stoff 4",
    ]

    # ---------------------------------------------------------
    # PTC ausführen
    # ---------------------------------------------------------

    try:
        result = ptc_solver(
            k=args,
            x0=x0,
            stoffe=stoffe,
            t_n=t_n,
            max_iter=max_iter,
            opt=opt,
            eps=eps,
            allow_initial_increase=True,
        )
    except Exception:
        return float("nan")

    # ---------------------------------------------------------
    # Prüfen, ob PTC eine Lösung geliefert hat
    # ---------------------------------------------------------

    if result is None or len(result) == 0:
        return float("nan")

    # Letzten Zustand auslesen
    x_final = result[stoffe].iloc[-1].to_numpy(dtype=float)

    # Physikalische Plausibilität
    if not np.all(np.isfinite(x_final)):
        return float("nan")

    if np.any(x_final < 0.0):
        return float("nan")

    # ---------------------------------------------------------
    # Finales Residuum unabhängig vom gespeicherten Wert
    # noch einmal berechnen
    # ---------------------------------------------------------

    try:
        final_res = la.norm(
            np.asarray(
                fkt_wert(args, x_final),
                dtype=float,
            )
        )
    except Exception:
        return float("nan")

    if not np.isfinite(final_res):
        return float("nan")

    # Nur die Residuum-Norm zurückgeben
    return float(final_res)