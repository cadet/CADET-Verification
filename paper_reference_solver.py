# -*- coding: utf-8 -*-
"""
Independent reference solver for the *paper's exact equations* (ChromOps, Meyer et al. 2026):

  dc/dt   = -v dc/dz + Dax d2c/dz2 - (1-eps)*eps_p/eps * kMT * (c - cp)      (Eq. 1)
  dcp/dt  = kMT*(c - cp) - dq/dt                                             (Eq. 2)
  dq_i/dt = ka_bar*[cp_i - (1/keq_i)*(cp_s/(Lam - sum (nu+sig) q_j))^nu_i q_i]  (Eq. 4)
  dq_s/dt = -sum nu_i dq_i/dt                                                (Eq. 5)

Purpose: determine where the paper's own model puts the protein peaks (esp. A),
to separate genuine model differences from digitization error in proteinA.csv.
First-order upwind FV in space, BDF with sparse Jacobian in time.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse

# --- parameters (mol/m^3, m, s) ---
N = 100
L = 0.01
v = 7.51e-5
Dax = 1.5e-9
eps = 0.37
eps_p = 0.66
phi = (1.0 - eps) * eps_p / eps          # bulk-side film factor of Eq. 1
kMT = np.array([1.39] + [0.139] * 6)
Lam = 324.7
ka_bar = 10.0
keq = np.array([1.0e5, 1.0e5, 3.0e4, 5.0e2, 5.0, 1.0e6])
nu = np.array([22.0, 22.0, 21.0, 20.0, 10.0, 23.0])
sig = np.full(6, 3.0)

dz = L / N
Ncomp = 7  # 0 = salt

# inlet program
section_times = [0.0, 360.0, 720.0, 1620.0, 2340.0, 9540.0, 10260.0]
salt_start = [40.0, 40.0, 40.0, 240.0, 240.0, 1040.0]
salt_end = [40.0, 40.0, 240.0, 240.0, 640.0, 1040.0]
cfeed = np.array([1.381, 0.03046, 0.3087, 0.06092, 0.1665, 0.08326])

def cin(t, k):
    """inlet concentrations during section k"""
    t0, t1 = section_times[k], section_times[k + 1]
    out = np.zeros(Ncomp)
    out[0] = salt_start[k] + (salt_end[k] - salt_start[k]) * (t - t0) / (t1 - t0)
    if k == 0:
        out[1:] = cfeed
    return out

# state layout: y = [c(N,7), cp(N,7), q(N,7)] flattened C-order
def unpack(y):
    z = y.reshape(3, N, Ncomp)
    return z[0], z[1], z[2]

def rhs(t, y, k):
    c, cp, q = unpack(y)
    ci = cin(t, k)

    # SMA driving-force kinetics (paper Eq. 4/5)
    Q = Lam - q[:, 1:] @ (nu + sig)
    Q = np.maximum(Q, 1e-3)
    cps = np.maximum(cp[:, 0], 1e-12)
    ratio = (cps[:, None] / Q[:, None]) ** nu[None, :]
    dq = ka_bar * (cp[:, 1:] - (1.0 / keq)[None, :] * ratio * q[:, 1:])
    dq0 = -(dq @ nu)
    dq_full = np.concatenate([dq0[:, None], dq], axis=1)

    # film transfer
    film = kMT[None, :] * (c - cp)
    dcp = film - dq_full

    # transport: upwind convection + central dispersion, Danckwerts-ish
    dc = np.empty_like(c)
    dc[0] = v / dz * (ci - c[0])
    dc[1:] = v / dz * (c[:-1] - c[1:])
    # dispersion with zero-flux ends
    lap = np.zeros_like(c)
    lap[1:-1] = (c[2:] - 2 * c[1:-1] + c[:-2])
    lap[0] = (c[1] - c[0])
    lap[-1] = (c[-2] - c[-1])
    dc += Dax / dz**2 * lap
    dc -= phi * film

    return np.concatenate([dc.ravel(), dcp.ravel(), dq_full.ravel()])

# sparsity pattern
def sparsity():
    n = 3 * N * Ncomp
    rows, cols = [], []
    def idx(block, j, i):
        return block * N * Ncomp + j * Ncomp + i
    for j in range(N):
        for i in range(Ncomp):
            # dc: depends on c(j-1,i), c(j,i), c(j+1,i), cp(j,i)
            for jj in (j - 1, j, j + 1):
                if 0 <= jj < N:
                    rows.append(idx(0, j, i)); cols.append(idx(0, jj, i))
            rows.append(idx(0, j, i)); cols.append(idx(1, j, i))
            # dcp: depends on c(j,i), cp(j,:) and q(j,:) via dq (Q couples all)
            rows.append(idx(1, j, i)); cols.append(idx(0, j, i))
            for ii in range(Ncomp):
                rows.append(idx(1, j, i)); cols.append(idx(1, j, ii))
                rows.append(idx(1, j, i)); cols.append(idx(2, j, ii))
            # dq: depends on cp(j,:), q(j,:)
            for ii in range(Ncomp):
                rows.append(idx(2, j, i)); cols.append(idx(1, j, ii))
                rows.append(idx(2, j, i)); cols.append(idx(2, j, ii))
    n_ = 3 * N * Ncomp
    return sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_, n_)).tocsr()

S = sparsity()

# initial conditions
c0 = np.zeros((N, Ncomp)); c0[:, 0] = 40.0
cp0 = np.zeros((N, Ncomp)); cp0[:, 0] = 40.0
q0 = np.zeros((N, Ncomp)); q0[:, 0] = Lam
y = np.concatenate([c0.ravel(), cp0.ravel(), q0.ravel()])

t_all, outlet_all = [], []
for k in range(len(section_times) - 1):
    t0, t1 = section_times[k], section_times[k + 1]
    teval = np.arange(t0, t1, 4.0)
    sol = solve_ivp(rhs, (t0, t1), y, method='BDF', args=(k,),
                    jac_sparsity=S, t_eval=teval, rtol=1e-5, atol=1e-8)
    if not sol.success:
        raise RuntimeError(f'section {k}: {sol.message}')
    y = sol.y[:, -1].copy()
    # need value at t1 too; re-evaluate via last state next section start
    c_out = sol.y.reshape(3, N, Ncomp, -1)[0, -1, :, :]  # outlet cell
    t_all.append(sol.t)
    outlet_all.append(c_out.T)
    print(f'section {k} done, nsteps={sol.t.size}')

t = np.concatenate(t_all)
outlet = np.vstack(outlet_all)
np.savez('paper_reference_solution.npz', t=t, outlet=outlet)

names = ['salt', 'A', 'B', 'C', 'D', 'E', 'F']
w = 11.67
print('\ncomponent  peak_t   peak_OD   FWHM_lo  FWHM_hi')
for i in range(1, 7):
    od = w * outlet[:, i]
    j = od.argmax()
    half = od > od[j] / 2
    print(f'{names[i]:>3}  {t[j]:8.0f}  {od[j]:8.3f}  {t[half].min():8.0f} {t[half].max():8.0f}')
