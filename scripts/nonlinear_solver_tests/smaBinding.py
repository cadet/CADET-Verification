import numpy as np

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