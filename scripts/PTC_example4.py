import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Importiere zu testendes Verfahren und Vergleichsverfahren
import Bsp2_Rob
import SmaTest_Rob

# =============================================================================
# KONFIGURATION: Wähle Run-Modus
# =============================================================================
# "QUICK_TEST" - Schneller Test in wenigen Minuten (wenige Samples/Iterationen)
# "FULL_RUN"   - Vollständiger Run wie im Paper (dauert sehr lange!)
RUN_MODE = "QUICK_TEST"  # <-- Ändere dies zu "FULL_RUN" für den vollständigen Run
# =============================================================================

# Erzeuge Sphäre mit gleichverteilten Punkten
def points(nDim, nSamples, r_min, r_max):
    # Richtung ziehen
    dirs = np.random.normal(size=(nDim, nSamples))

    norms = np.linalg.norm(dirs, axis=0)
    dirs = dirs / norms

    # Radius ziehen
    rad = np.power(
        np.random.uniform(
            low=r_min**nDim,
            high=r_max**nDim,
            size=(nSamples,)
        ),
        1.0 / nDim
    )

    return dirs * rad


# Zeige df
def plot_df(df):
    # Nur plotten wenn Daten vorhanden sind
    if len(df) == 0:
        print("Warnung: Keine Daten zum Plotten vorhanden")
        return
    # print(df)
    df.plot()
    plt.show()
    return


# Schreibe df als csv in das LaTeX-Verzeichnis
def write_df(df, path, file):
    import os
    os.makedirs(path, exist_ok=True)
    with open(path + file, "w") as tf:
        tf.write(df.to_csv())
    return


# Berechne Robustheit
def calc_rob(args, liste1, liste2):
    nSamples = args[0]
    nMin = args[1]
    nDim = args[2]
    r_min = args[3]
    r_max = args[4]
    normmin = args[5]
    stepsize = args[6]
    maxDistance = args[7]
    x_ref = args[8]

    df_g = pd.DataFrame(columns=liste1)
    df_t = pd.DataFrame(columns=liste2)

    # Zeit-Tracking
    start_time = time.time()

    # Vergrößere Abstand von x_ref
    for step in range(0, maxDistance):
        # Zähle wie oft die Verfahren konvergieren
        counterPTC = 0
        counterNEW = 0

        # Fortschrittsanzeige
        if step > 0:
            elapsed = time.time() - start_time
            avg_time_per_step = elapsed / step
            remaining_steps = maxDistance - step
            eta_seconds = avg_time_per_step * remaining_steps
            eta_minutes = eta_seconds / 60
            print(f"Schritt {step}/{maxDistance} | "
                  f"Elapsed: {elapsed/60:.1f}min | "
                  f"ETA: {eta_minutes:.1f}min")
        else:
            print(f"Schritt {step}/{maxDistance}")

        # Zähle legale Punkte
        nCounter = nSamples
        nNegative = 0  # Debug: Zähle negative Punkte
        nNaN = 0       # Debug: Zähle NaN-Ergebnisse

        # Vergrößere Abstand
        r_min = r_min + stepsize
        r_max = r_max + stepsize

        # Erzeuge Punkte und transformiere in eine Ellipse
        sphToEllipse = np.diag([1000, 10, 10, 10])

        pts = points(nDim, nSamples, r_min, r_max)
        pts = np.dot(sphToEllipse, pts)

        # Für alle erzeugten Punkte ...
        for i in range(0, nSamples):
            x = []
            skip = False

            # Für jeden Punkt gehe von x_ref aus und erhalte neuen Punkt
            for j in range(0, nDim):
                x_j = x_ref[j] + pts[j, i]

                if x_j < 0:
                    skip = True

                x.append(x_j)

            # Entferne negative Punkte
            if skip:
                nCounter = nCounter - 1
                nNegative += 1

            else:
                # Ruft Bsp2-Verfahren auf und gibt Residual-Norm zurück
                norm1 = Bsp2_Rob.main(x)

                if norm1 < normmin:
                    counterPTC = counterPTC + 1

                # Ruft Vergleichsverfahren auf und gibt Residual-Norm zurück
                try:
                    norm2 = SmaTest_Rob.smaTestRob(x)
                except ValueError:
                    norm2 = float("nan")

                if norm2 < normmin:
                    counterNEW = counterNEW + 1

                if np.isnan(norm1) and np.isnan(norm2):
                    nCounter = nCounter - 1
                    nNaN += 1

                if nCounter < nMin:
                    print(f"  Abbruch bei Punkt {i+1}/{nSamples}: Nicht mehr genügend legale Punkte")
                    print(f"  nCounter={nCounter}, nMin={nMin}, nNegative={nNegative}, nNaN={nNaN}")
                    break

        # Prüfe ob genügend legale Punkte vorhanden sind
        if nCounter < nMin:
            print(f"Abbruch nach {step} Iterationen: Nicht mehr genügend legale Punkte")
            print(f"  Finale Statistik: nCounter={nCounter}, nMin={nMin}")
            print(f"  Negative Punkte: {nNegative}/{nSamples} ({100*nNegative/nSamples:.1f}%)")
            print(f"  NaN Ergebnisse: {nNaN}/{nSamples} ({100*nNaN/nSamples:.1f}%)")
            break

        # Berechne Anzahl der konvergierten legalen Punkte
        # Verhindere Division durch Null
        if nCounter > 0:
            wert1 = counterPTC / nCounter
            wert2 = counterNEW / nCounter
        else:
            print(f"Warnung in Iteration {step}: Keine legalen Punkte gefunden, überspringe")
            continue

        # Erzeuge df mit Prozent der konvergierten Punkte
        # ... für Ausgabe als Plot
        df_g.loc[step] = [wert1, wert2]

        # ... und für Ausgabe als Tabelle
        df_t.loc[step] = [wert1, wert2, nCounter, r_min]
        df_t.index.name = "Iterationen"

    return df_g, df_t


def main():
    # Parameter-Sets für verschiedene Run-Modi
    if RUN_MODE == "QUICK_TEST":
        print("=" * 70)
        print("QUICK TEST MODE - Schneller Test in wenigen Minuten")
        print("=" * 70)
        nSamples = 1000      # Reduziert von 100000
        maxDistance = 30     # Reduziert von 301
        stepsize = 0.2       # Kleinere Schritte als vorher (von 0.1)
        print(f"Samples pro Iteration: {nSamples}")
        print(f"Anzahl Iterationen: {maxDistance}")
        print(f"Schrittweite: {stepsize}")
        print(f"Geschätzte Laufzeit: 3-8 Minuten")
        print("=" * 70)
    elif RUN_MODE == "FULL_RUN":
        print("=" * 70)
        print("FULL RUN MODE - Vollständiger Run (WARNUNG: Dauert sehr lange!)")
        print("=" * 70)
        nSamples = 100000    # Original-Wert aus dem Paper
        maxDistance = 301    # Original-Wert aus dem Paper
        stepsize = 0.1       # Original-Wert aus dem Paper
        print(f"Samples pro Iteration: {nSamples}")
        print(f"Anzahl Iterationen: {maxDistance}")
        print(f"Schrittweite: {stepsize}")
        print(f"Geschätzte Laufzeit: Mehrere Stunden!")
        print("=" * 70)
    else:
        raise ValueError(f"Unbekannter RUN_MODE: {RUN_MODE}")

    # Gemeinsame Parameter
    nMin = nSamples / 100
    nDim = 4
    r_min = 0
    r_max = 0.1
    normmin = 0.02

    x_ref = np.array([
        1.0485785488181000e+03,
        1.1604726694141368e+01,
        1.1469542586742687e+01,
        9.7852311988018670e+00
    ])

    args = [
        nSamples,
        nMin,
        nDim,
        r_min,
        r_max,
        normmin,
        stepsize,
        maxDistance,
        x_ref
    ]

    liste1 = ["Bsp2", "smaTest"]
    liste2 = ["Bsp2", "smaTest", "Punkte", "rmin"]

    path = "../ba_nv_latex/"

    # Unterschiedliche Dateinamen für verschiedene Modi
    if RUN_MODE == "QUICK_TEST":
        file = "Data_Robustheit_QuickTest.csv"
    else:
        file = "Data_Robustheit.csv"

    print(f"\nStarte Robustheitstest...")
    start_time = time.time()

    df_g, df_t = calc_rob(args, liste1, liste2)

    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Test abgeschlossen in {elapsed_time/60:.2f} Minuten")
    print(f"Ergebnisse gespeichert in: {path}{file}")
    print(f"{'=' * 70}\n")

    plot_df(df_g)
    write_df(df_t, path, file)

    return


if __name__ == "__main__":
    main()