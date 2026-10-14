import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from numpy import linalg as la

# Berechne Jacobi-Matrix
def jacobi(args, x):
    yCp = args[0]
    kA = args[1]
    kD = args[2]
    nu = args[3]
    sigma = args[4]
    # Lambda = args[5]

    q0bar = x[0] - np.sum(sigma * x)
    c0powNu = np.power(yCp[0], nu)
    # q0barPowNu = np.power(q0bar, nu)
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


# Berechne Funktionswerte
def fkt_wert(args, x):
    yCp = args[0]
    kA = args[1]
    kD = args[2]
    nu = args[3]
    sigma = args[4]
    Lambda = args[5]

    q0bar = x[0] - np.sum(sigma * x)
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


# Berechne optimalen Zeitschritt
def t_opt(args, t_n, x_n, x_n1, dx):
    z = abs(np.dot(dx, fkt_wert(args, x_n) - dx))
    n = 2 * la.norm(dx) * la.norm(fkt_wert(args, x_n1) - dx)

    while z < n:
        z = 4 * z

    return t_n * (z / n)


# Berechne SER Zeitschritt
def t_ser(args, t_n, x_n, x_n1):
    z = t_n * la.norm(fkt_wert(args, x_n))
    n = la.norm(fkt_wert(args, x_n1))
    return z / n


# Zeige df
def plot_df(df):
    print(df)
    df.plot()
    plt.show()
    return


# Schreibe df als csv in das LaTeX-Verzeichnis
def write_df(df, path, file):
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, file), 'w') as tf:
        tf.write(df.to_csv())


# Berechne Beispiel
def calc_bsp(args, x0, t_n, stoffe, max_iter, opt, eps):
    x_n = x0
    x_list = [x0]
    t_list = [t_n]
    res_list = [la.norm(fkt_wert(args, x0))]
    i = 0

    while (
        i in range(0, max_iter)
        and la.norm(fkt_wert(args, x_n)) > eps
    ):
        dx = delta_x(args, t_n, x_n)
        x_n1 = x_n + t_n * dx

        # Modifizierte Abbruchbedingung, verzichtet auf eine Reduktion
        # des Residuums in den ersten Iterationen
        if la.norm(fkt_wert(args, x_n1)) >= la.norm(fkt_wert(args, x_n)):
            if i < 3:
                res_list = np.vstack(
                    (res_list, la.norm(fkt_wert(args, x_n1)))
                )
                x_list = np.vstack((x_list, x_n1))

                if opt:
                    t_n1 = t_opt(args, t_n, x_n, x_n1, dx)
                else:
                    t_n1 = t_ser(args, t_n, x_n, x_n1)

                t_list = np.vstack((t_list, t_n1))
                x_n = x_n1
                t_n = t_n1

            else:
                print('Abbruch: Keine Residual-Reduktion')
                break

        if opt:
            t_n1 = t_opt(args, t_n, x_n, x_n1, dx)
        else:
            t_n1 = t_ser(args, t_n, x_n, x_n1)

        t_list = np.vstack((t_list, t_n1))
        res_list = np.vstack(
            (res_list, la.norm(fkt_wert(args, x_n1)))
        )
        x_list = np.vstack((x_list, x_n1))

        t_n = t_n1
        x_n = x_n1
        i = i + 1

    df = pd.DataFrame(data=x_list, columns=stoffe)
    df['Residuum'] = res_list
    df['Zeitschrittweite'] = t_list
    df.index.name = 'Iterationen'

    return df


def main():
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

    # Anfänglicher Zustand
    x0 = np.array([7e+02, 4e+00, 1.4e+01, 5e+00])

    # Anfänglich gewählter Zeitschritt
    t_n = 1

    stoffe = ['$C_a$', '$C_b$', '$C_c$', '$C_d$']
    max_iter = 50
    eps = 1e-10

    for opt in [True, False]:
        path = "../ba_nv_latex/"

        if opt:
            print("Beispiel: 2 OPT")
            file = 'Data_Bsp2.csv'
            df = calc_bsp(
                args, x0, t_n, stoffe, max_iter, opt, eps
            )
            df_o = df.loc[:, 'Residuum']

        else:
            print("\nBeispiel: 2 SER")
            file = 'Data_Bsp2SER.csv'
            df = calc_bsp(
                args, x0, t_n, stoffe, max_iter, opt, eps
            )
            df_s = df.loc[:, 'Residuum']

        plot_df(df)
        write_df(df, path, file)

    file = 'Data_Bsp2RES.csv'
    dfr = pd.merge(
        df_o,
        df_s,
        on='Iterationen',
        how='outer'
    )

    dfr.columns = ['OPT', 'SER']
    plot_df(dfr)
    write_df(dfr, path, file)

    return


if __name__ == "__main__":
    main()