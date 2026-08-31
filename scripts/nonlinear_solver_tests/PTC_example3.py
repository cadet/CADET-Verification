import math as m
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as la

# Berechne Jacobi-Matrix
def jacobi(args, x, rel_stoffe):
    k_h = args[0]
    k_z = args[1]
    sm = args[2]

    jm = np.zeros((len(rel_stoffe), len(x)))

    jm[0, 0] = -k_z[0] * x[18]
    jm[0, 1] = k_h[0]
    jm[0, 18] = -k_z[0] * x[0]

    jm[1, 1] = -k_z[1] * x[18]
    jm[1, 2] = k_h[1]
    jm[1, 18] = -k_z[1] * x[1]

    jm[2, 3] = -k_z[2] * x[18]
    jm[2, 4] = k_h[2]
    jm[2, 18] = -k_z[2] * x[3]

    jm[3, 5] = -k_z[3] * x[18]
    jm[3, 6] = k_h[3]
    jm[3, 18] = -k_z[3] * x[5]

    jm[4, 7] = -k_z[4] * x[18]
    jm[4, 8] = k_h[4]
    jm[4, 18] = -k_z[4] * x[7]

    jm[5, 9] = -k_z[5] * x[18]
    jm[5, 10] = k_h[5]
    jm[5, 18] = -k_z[5] * x[9]

    jm[6, 11] = -k_z[6] * x[18]
    jm[6, 12] = k_h[6]
    jm[6, 18] = -k_z[6] * x[11]

    jm[7, 12] = -k_z[7] * x[18]
    jm[7, 13] = k_h[7]
    jm[7, 18] = -k_z[7] * x[12]

    jm[8, 14] = -k_z[8] * x[18]
    jm[8, 15] = k_h[8]
    jm[8, 18] = -k_z[8] * x[14]

    jm[9, 16] = -k_z[9] * x[18]
    jm[9, 17] = k_h[9]
    jm[9, 18] = -k_z[9] * x[16]

    jm = np.matmul(sm, jm)

    return jm


# Berechne Funktionswerte
def fkt_wert(args, x, rel_stoffe):
    k_h = args[0]
    k_z = args[1]
    sm = args[2]

    fw = np.zeros(len(rel_stoffe))

    fw[0] = k_h[0] * x[1] - k_z[0] * x[0] * x[18]
    fw[1] = k_h[1] * x[2] - k_z[1] * x[1] * x[18]
    fw[2] = k_h[2] * x[4] - k_z[2] * x[3] * x[18]
    fw[3] = k_h[3] * x[6] - k_z[3] * x[5] * x[18]
    fw[4] = k_h[4] * x[8] - k_z[4] * x[7] * x[18]
    fw[5] = k_h[5] * x[10] - k_z[5] * x[9] * x[18]
    fw[6] = k_h[6] * x[12] - k_z[6] * x[11] * x[18]
    fw[7] = k_h[7] * x[13] - k_z[7] * x[12] * x[18]
    fw[8] = k_h[8] * x[15] - k_z[8] * x[14] * x[18]
    fw[9] = k_h[9] * x[17] - k_z[9] * x[16] * x[18]

    fw = np.matmul(sm, fw)

    return fw


# Berechne Änderungsrate mit Vorkonditionierung
def delta_x(args, t, x, rel_stoffe):
    i = np.eye(len(x))

    jm = jacobi(args, x, rel_stoffe)
    am = i - t * jm
    tm = t_matrix(am)

    ta = np.matmul(tm, am)
    tat = np.matmul(ta, tm)
    tat_inv = np.linalg.solve(tat, i)
    tf = np.matmul(tm, fkt_wert(args, x, rel_stoffe))

    y = np.matmul(tat_inv, tf)
    dx = np.matmul(tm, y)

    return dx


# Berechne optimalen Zeitschritt
def t_opt(args, t_n, x_n, x_n1, dx, rel_stoffe):
    z = abs(
        np.matmul(
            np.transpose(dx),
            fkt_wert(args, x_n, rel_stoffe) - dx
        )
    )

    n = 2 * la.norm(dx) * la.norm(
        fkt_wert(args, x_n1, rel_stoffe) - dx
    )

    return (z / n) * t_n


# Berechne SER Zeitschritt
def t_ser(args, t_n, x_n, x_n1, rel_stoffe):
    z = t_n * la.norm(fkt_wert(args, x_n, rel_stoffe))
    n = la.norm(fkt_wert(args, x_n1, rel_stoffe))

    return z / n


# Berechne D Matrix zur Vorkonditionierung
def d_matrix(am):
    dm = np.zeros((len(am), len(am)))

    for i in range(0, len(am)):
        dm[i, i] = 1 / max(abs(am[i, :]))

    return dm


# Berechne T Matrix zur Vorkonditionierung
def t_matrix(am):
    tm = np.zeros((len(am), len(am)))

    for i in range(0, len(am)):
        tm[i, i] = m.sqrt(1 / max(abs(am[i, :])))

    return tm


# Zeige df
def plot_df(df, stoffe):
    pd.set_option('display.max_columns', 19)
    print(df)

    df.plot()
    plt.xlabel('Iterationen')
    plt.ylabel('Konzentrationen')
    plt.legend(stoffe)
    plt.show()

    return


# Schreibe df als csv in das LaTeX-Verzeichnis
def write_df(df, path, file):
    # Gesamttabelle
    df.to_csv(path + file, encoding='utf-8')

    a = ['$Dap$', '$Dap+$', '$Dap2+$',
         '$Pip$', '$Pip+$', '$Pip2+$']

    b = ['$Dea$', '$Dea+$', '$Tris$',
         '$Tris+$', '$Imi$', '$Imi+$']

    c = ['$Bts$', '$Bts+$', '$Ace−$',
         '$Ace$', '$Lac−$', '$Lac$']

    d = ['Residuum', 'Zeitschrittweite']

    df.loc[:, a].to_csv(path + file[:-4] + 'a.csv', encoding='utf-8')
    df.loc[:, b].to_csv(path + file[:-4] + 'b.csv', encoding='utf-8')
    df.loc[:, c].to_csv(path + file[:-4] + 'c.csv', encoding='utf-8')
    df.loc[:, d].to_csv(path + file[:-4] + 'd.csv', encoding='utf-8')


# Berechne Beispiel
def calc_bsp(args, x0, t_n, stoffe, rel_stoffe, max_iter, opt, eps):
    x_n = x0
    x_list = [x0]
    t_list = [t_n]
    res_list = [la.norm(fkt_wert(args, x_n, rel_stoffe))]
    i = 0

    while (
        i in range(0, max_iter)
        and la.norm(fkt_wert(args, x_n, rel_stoffe)) > eps
    ):
        dx = delta_x(args, t_n, x_n, rel_stoffe)
        x_n1 = x_n + t_n * dx

        if (
            la.norm(fkt_wert(args, x_n1, rel_stoffe))
            > la.norm(fkt_wert(args, x_n, rel_stoffe))
        ):
            print('Abbruch: Keine Residual-Reduktion')
            break

        if opt:
            t_n1 = t_opt(
                args, t_n, x_n, x_n1, dx, rel_stoffe
            )
        else:
            t_n1 = t_ser(
                args, t_n, x_n, x_n1, rel_stoffe
            )

        if t_n1 < t_n:
            print('Abbruch: Maximaler Zeitschritt erreicht')
            break

        t_list = np.vstack((t_list, t_n1))
        res_list = np.vstack(
            (
                res_list,
                la.norm(fkt_wert(args, x_n1, rel_stoffe))
            )
        )
        x_list = np.vstack((x_list, x_n1))

        x_n = x_n1
        t_n = t_n1
        i = i + 1

    df = pd.DataFrame(data=x_list, columns=stoffe)
    df['Residuum'] = res_list
    df['Zeitschrittweite'] = t_list
    df.index.name = 'Iterationen'

    return df


def main():
    # Vorgegebene Konstanten für das Problem
    k_h = np.array([
        2.4e-8, 2.29e-6, 1.31e-6, 8.47e-6, 1.02e-4,
        3.28e-4, 1.86e-7, 4.6e-3, 1.75e-2, 1.38e-1
    ])

    k_z = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ])

    sm = np.array([
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
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    args = [k_h, k_z, sm]

    # Anfängliche Konzentrationen
    x0 = np.array([
        10, 0, 0, 10, 0,
        10, 0, 10, 0, 10,
        0, 10, 0, 0, 0,
        10, 0, 10, 10**-0.5
    ])

    # Anfangs gewählter Zeitschritt
    t_n = 1

    stoffe = [
        '$Dap$', '$Dap+$', '$Dap2+$',
        '$Dea$', '$Dea+$',
        '$Tris$', '$Tris+$',
        '$Imi$', '$Imi+$',
        '$Bts$', '$Bts+$',
        '$Pip$', '$Pip+$', '$Pip2+$',
        '$Ace−$', '$Ace$',
        '$Lac−$', '$Lac$',
        '$H+$'
    ]

    rel_stoffe = [
        'Dap+', 'Dap2+', 'Dea+', 'Tris+', 'Imi+',
        'Bts+', 'Pip+', 'Pip2+', 'Ace', 'Lac'
    ]

    max_iter = 50
    eps = 1e-10

    for opt in [True, False]:
        path = "../ba_nv_latex/"

        if opt:
            print("Beispiel: 3 OPT")
            file = 'Data_Bsp3.csv'

            df = calc_bsp(
                args, x0, t_n, stoffe, rel_stoffe,
                max_iter, opt, eps
            )

            dfo = df.loc[:, 'Residuum']

        else:
            print("\nBeispiel: 3 SER")
            file = 'Data_Bsp3SER.csv'

            df = calc_bsp(
                args, x0, t_n, stoffe, rel_stoffe,
                max_iter, opt, eps
            )

            dfs = df.loc[:, 'Residuum']

        plot_df(df, stoffe)
        write_df(df, path, file)


if __name__ == "__main__":
    main()