import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importiere zu testendes Verfahren und Vergleichsverfahren
import Bsp2_Rob
import SmaTest_Rob

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
    # print(df)
    df.plot()
    plt.show()
    return


# Schreibe df als csv in das LaTeX-Verzeichnis
def write_df(df, path, file):
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

    # Vergrößere Abstand von x_ref
    for step in range(0, maxDistance):
        # Zähle wie oft die Verfahren konvergieren
        counterPTC = 0
        counterNEW = 0

        print(step)

        # Zähle legale Punkte
        nCounter = nSamples

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

                if nCounter < nMin:
                    print("Abbruch: Nicht mehr genügend legale Punkte")
                    break

        # Berechne Anzahl der konvergierten legalen Punkte
        wert1 = counterPTC / nCounter
        wert2 = counterNEW / nCounter

        # Erzeuge df mit Prozent der konvergierten Punkte
        # ... für Ausgabe als Plot
        df_g.loc[step] = [wert1, wert2]

        # ... und für Ausgabe als Tabelle
        df_t.loc[step] = [wert1, wert2, nCounter, r_min]
        df_t.index.name = "Iterationen"

    return df_g, df_t


def main():
    # Parameter
    nSamples = 100000  # Typical: 100000
    nMin = nSamples / 100
    nDim = 4
    r_min = 0
    r_max = 0.1
    normmin = 0.02
    stepsize = 0.1
    maxDistance = 301

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
    file = "Data_Robustheit.csv"

    df_g, df_t = calc_rob(args, liste1, liste2)

    plot_df(df_g)
    write_df(df_t, path, file)

    return


if __name__ == "__main__":
    main()