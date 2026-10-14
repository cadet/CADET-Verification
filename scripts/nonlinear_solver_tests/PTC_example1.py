import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from numpy import linalg as la

# Berechne Funktionswerte
def fkt_wert(k, x):
    return np.array(
        [-k[0] * x[0] * x[1] + k[1] * x[2],
         -k[0] * x[0] * x[1] + k[1] * x[2],
         k[0] * x[0] * x[1] - k[1] * x[2]]
    )

# Berechne Jacobi-Matrix
def jacobi(k, x):
    return np.array(
        [[-k[0] * x[1], -k[0] * x[0], k[1]],
         [-k[0] * x[1], -k[0] * x[0], k[1]],
         [k[0] * x[1], k[0] * x[0], -k[1]]]
    )

# Berechne die Änderungsrate Delta x
def delta_x(k, t, x):
    i = np.eye(len(x))
    dx = la.solve(i - t * jacobi(k, x), fkt_wert(k, x))
    return dx

# Berechne optimalen Zeitschritt
def t_opt(k, t_n, x_n, x_n1, dx):
    z = abs(np.dot(dx, fkt_wert(k, x_n) - dx))
    n = 2 * la.norm(dx) * la.norm(fkt_wert(k, x_n1) - dx)
    return (z / n) * t_n

# Berechne SER Zeitschritt
def t_ser(k, t_n, x_n, x_n1):
    z = t_n * la.norm(fkt_wert(k, x_n))
    n = la.norm(fkt_wert(k, x_n1))
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
def calc_bsp(k, x0, t_n, stoffe, max_iter, opt, eps):
    x_n = x0
    x_list = [x0]
    t_list = [t_n]
    res_list = [la.norm(fkt_wert(k, x0))]
    i = 0

    while i in range(0, max_iter) and la.norm(fkt_wert(k, x_n)) > eps:
        dx = delta_x(k, t_n, x_n)
        x_n1 = x_n + t_n * dx

        if la.norm(fkt_wert(k, x_n1)) >= la.norm(fkt_wert(k, x_n)):
            print('Abbruch: Keine Residual-Reduktion')
            break

        if opt:
            t_n1 = t_opt(k, t_n, x_n, x_n1, dx)
        else:
            t_n1 = t_ser(k, t_n, x_n, x_n1)

        t_list = np.vstack((t_list, t_n1))
        res_list = np.vstack((res_list, la.norm(fkt_wert(k, x_n1))))
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
    # Reaktionskonstanten
    k = np.array([0.2, 0.3])

    # Anfängliche Konzentrationen
    x0 = np.array([0.25, 1.25, 0.75])

    # Anfänglich gewählter Zeitschritt
    t_n = 1

    eps = 1e-10

    max_iter = 50

    stoffe = ['$C_a$', '$C_b$', '$C_c$']
    stoffe2 = ['$C_A$', '$C_B$', '$C_C$']

    for opt in [True, False]:
        path = "../ba_nv_latex/"

        if opt:
            file = 'Data_Bsp1OPT.csv'
            print("Beispiel: 1 OPT")
            df1 = calc_bsp(k, x0, t_n, stoffe, max_iter, opt, eps)
            dfo = df1.loc[:, 'Residuum']
            write_df(df1, path, file)
        else:
            file = 'Data_Bsp1SER.csv'
            print("**n Beispiel: 1 SER")
            df2 = calc_bsp(k, x0, t_n, stoffe2, max_iter, opt, eps)
            dfs = df2.loc[:, 'Residuum']
            write_df(df2, path, file)

    file = 'Data_Bsp1.csv'
    dfm = pd.merge(df1, df2, on='Iterationen', how='outer')
    write_df(dfm, path, file)
    plot_df(dfm)

    file = 'Data_Bsp1RES.csv'
    dfr = pd.merge(dfo, dfs, on='Iterationen', how='outer')
    dfr.columns = ['OPT', 'SER']
    plot_df(dfr)
    write_df(dfr, path, file)

    return


if __name__ == "__main__":
    main()