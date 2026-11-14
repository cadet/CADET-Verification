# -*- coding: utf-8 -*-
"""
Independent solver for the *exact* equations of Section 4.2 of

    Meyer et al., 2026, Computers and Chemical Engineering,
    "ChromOps.jl: High-order simulation and discrete forward sensitivity
     analysis for chromatography models"

i.e. Eqs. (1), (2), (4) and (5) with the paper's own driving-force binding
kinetics, the parameters of Tables 3 and 4 and the inlet program of
Eqs. (35)-(40). Depends only on numpy/scipy -- no CADET, no Julia.

Purpose
-------
This script exists to isolate a single modeling question, and it is written to
be run by someone who has neither CADET nor this repository. The only thing that
changes between its two modes is the available-sites ("free ligand") term in the
denominator of Eq. (4):

    --denominator paper_eq4    qbar = Lambda - sum_j (nu_j + sigma_j) q_j
                               Eq. (4) exactly as printed.

    --denominator bound_salt   qbar = Lambda - sum_j nu_j q_j
                               which is the bound-salt concentration q_s that
                               Eq. (5) already integrates as a state variable.

`paper_eq4` puts the six peaks 253-569 s early and up to 18 % low relative to
the published Figure 1; `bound_salt` reproduces every peak height to within
0.5 % and every peak time to a single uniform offset. See Meyer2026_fig1.md.

Numerics
--------
Space: cell-centred finite volumes; upwind convection with the numerical-
    dispersion correction Dax_eff = Dax - v*dz/2, so the *effective* dispersion
    equals the physical Dax of Table 3. Zero dispersive flux at both ends, with
    the inlet concentration entering through the convective flux -- the same
    boundary split the paper describes for its own operators in Eqs. (12)-(14).
Time:  BDF (scipy) with an analytic sparse Jacobian. The analytic Jacobian is
    not an optimization but a requirement: the desorption coefficient
    (ka_bar/keq_i) * (c_ps/qbar)^nu_i spans more than 20 orders of magnitude
    over the load, and a finite-difference Jacobian fails outright.

Units: mol/m^3, m, s throughout (Table 3's mol/L are multiplied by 1000).

Usage
-----
    python Meyer2026_fig1_paper_equations.py
    python Meyer2026_fig1_paper_equations.py --denominator bound_salt --plot
    python Meyer2026_fig1_paper_equations.py --n 800 --check-jacobian
"""
import argparse
import os

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
DIGITIZED = os.path.join(HERE, 'Meyer2026_fig1_digitized')

# --- Table 3: model parameters ---------------------------------------------
L = 0.01                                    # column length, 10 mm
V_INT = 7.51e-5                             # interstitial velocity, 7.51e-3 cm/s
DAX = 1.5e-9                                # axial dispersion, 1.5e-5 cm^2/s
EPS = 0.37                                  # bulk (interstitial) porosity
EPS_P = 0.66                                # particle porosity
LAMBDA = 324.7                              # ionic capacity, 0.3247 mol/L
K_MT = np.array([1.39] + [0.139] * 6)       # lumped mass transfer, salt first

# --- Table 4: SMA isotherm parameters (components A..F) --------------------
NU = np.array([22.0, 22.0, 21.0, 20.0, 10.0, 23.0])
SIGMA = np.full(6, 3.0)
K_EQ = np.array([1.0e5, 1.0e5, 3.0e4, 5.0e2, 5.0, 1.0e6])
KA_BAR = 10.0

# --- Eqs. (35)-(40): inlet program -----------------------------------------
DELTA_T = [360.0, 360.0, 900.0, 720.0, 7200.0, 720.0]
SECTIONS = np.concatenate(([0.0], np.cumsum(DELTA_T)))
SALT_START = np.array([0.04, 0.04, 0.04, 0.24, 0.24, 1.04]) * 1000.0
SALT_END = np.array([0.04, 0.04, 0.24, 0.24, 0.64, 1.04]) * 1000.0
C_FEED = 1e-5 * np.array([138.1, 3.046, 30.87, 6.092, 16.65, 8.326]) * 1000.0

W_EXT = 1.167e4 / 1000.0   # extinction coefficient, AU L/(mol cm) -> per mol/m^3
COMPONENTS = 'ABCDEF'
NC = 7                     # salt + six proteins
PHI = (1.0 - EPS) * EPS_P / EPS    # bulk-side film factor of Eq. (1)


class PaperModel:
    """Semi-discretization of Eqs. (1), (2), (4), (5) on N finite volumes."""

    def __init__(self, n_cells, denominator='paper_eq4', ka_scale=1.0):
        if denominator not in ('paper_eq4', 'bound_salt'):
            raise ValueError(f"unknown denominator {denominator!r}")
        self.n = n_cells
        self.denominator = denominator
        self.ka_bar = KA_BAR * ka_scale
        self.shield = (NU + SIGMA) if denominator == 'paper_eq4' else NU.copy()

        self.dz = L / n_cells
        self.dax_eff = DAX - V_INT * self.dz / 2.0
        if self.dax_eff <= 0.0:
            raise ValueError(
                f"n={n_cells} is too coarse: upwind numerical dispersion "
                f"{V_INT * self.dz / 2:.3g} exceeds Dax={DAX:.3g} m^2/s. "
                f"Use n > {int(np.ceil(V_INT * L / (2 * DAX)))}.")
        self.q_floor = 1e-6 * LAMBDA
        self.floor_hits = 0
        self._build_constant_jacobian()
        self._build_binding_pattern()

    # -- indexing: block b in {c, cp, q}, cell i, component j ---------------
    def _idx(self, b, i, j):
        return b * self.n * NC + i * NC + j

    def initial_state(self):
        """Column pre-equilibrated with the 0.04 mol/L salt of Eq. (39).

        The paper does not state its initial condition; this is the only choice
        consistent with the first phase of the salt program, and it puts the
        ion exchanger fully in the salt form, q_s(0) = Lambda.
        """
        c = np.zeros((self.n, NC)); c[:, 0] = SALT_START[0]
        cp = np.zeros((self.n, NC)); cp[:, 0] = SALT_START[0]
        q = np.zeros((self.n, NC)); q[:, 0] = LAMBDA
        return np.concatenate([c.ravel(), cp.ravel(), q.ravel()])

    @staticmethod
    def inlet(t, k):
        out = np.zeros(NC)
        out[0] = SALT_START[k] + (SALT_END[k] - SALT_START[k]) \
            * (t - SECTIONS[k]) / (SECTIONS[k + 1] - SECTIONS[k])
        if k == 0:
            out[1:] = C_FEED
        return out

    # -- Eq. (4): binding, plus the pieces the Jacobian needs ---------------
    def _binding(self, cp, q):
        qbar = LAMBDA - q[:, 1:] @ self.shield
        if qbar.min() < self.q_floor:
            self.floor_hits += 1
        qbar = np.maximum(qbar, self.q_floor)
        cps = np.maximum(cp[:, 0], 1e-30)
        # K_i = (1/keq_i) * (c_ps/qbar)^nu_i, in logs to survive nu_i ~ 23
        lg = NU[None, :] * (np.log(cps)[:, None] - np.log(qbar)[:, None]) - np.log(K_EQ)[None, :]
        K = np.exp(np.minimum(lg, 300.0))
        R = K * q[:, 1:]
        return self.ka_bar * (cp[:, 1:] - R), qbar, cps, K, R

    def rhs(self, t, y, k):
        z = y.reshape(3, self.n, NC)
        c, cp, q = z[0], z[1], z[2]
        dq, *_ = self._binding(cp, q)

        dq_full = np.empty((self.n, NC))
        dq_full[:, 0] = -(dq @ NU)          # Eq. (5)
        dq_full[:, 1:] = dq

        film = K_MT[None, :] * (c - cp)
        dcp = film - dq_full                # Eq. (2)

        dc = np.empty_like(c)               # Eq. (1)
        dc[0] = V_INT / self.dz * (self.inlet(t, k) - c[0])
        dc[1:] = V_INT / self.dz * (c[:-1] - c[1:])
        lap = np.empty_like(c)
        lap[1:-1] = c[2:] - 2 * c[1:-1] + c[:-2]
        lap[0] = c[1] - c[0]                # zero dispersive flux, inlet
        lap[-1] = c[-2] - c[-1]             # zero dispersive flux, outlet
        dc += self.dax_eff / self.dz**2 * lap - PHI * film
        return np.concatenate([dc.ravel(), dcp.ravel(), dq_full.ravel()])

    # -- analytic Jacobian ---------------------------------------------------
    def _build_constant_jacobian(self):
        n, dz = self.n, self.dz
        conv, disp = V_INT / dz, self.dax_eff / dz**2
        rows, cols, vals = [], [], []
        for i in range(n):
            for j in range(NC):
                r = self._idx(0, i, j)
                rows.append(r); cols.append(self._idx(0, i, j)); vals.append(-conv)
                if i > 0:
                    rows.append(r); cols.append(self._idx(0, i - 1, j)); vals.append(conv)
                if 0 < i < n - 1:
                    rows += [r, r, r]
                    cols += [self._idx(0, i, j), self._idx(0, i - 1, j), self._idx(0, i + 1, j)]
                    vals += [-2 * disp, disp, disp]
                elif i == 0:
                    rows += [r, r]; cols += [self._idx(0, 0, j), self._idx(0, 1, j)]
                    vals += [-disp, disp]
                else:
                    rows += [r, r]; cols += [self._idx(0, n - 1, j), self._idx(0, n - 2, j)]
                    vals += [-disp, disp]
                # film term in Eq. (1) and Eq. (2)
                rows += [r, r]; cols += [self._idx(0, i, j), self._idx(1, i, j)]
                vals += [-PHI * K_MT[j], PHI * K_MT[j]]
                rp = self._idx(1, i, j)
                rows += [rp, rp]; cols += [self._idx(0, i, j), self._idx(1, i, j)]
                vals += [K_MT[j], -K_MT[j]]
        size = 3 * n * NC
        self._j_const = sparse.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()

    def _build_binding_pattern(self):
        rows, cols = [], []
        for i in range(self.n):
            for j in range(NC):
                for block in (1, 2):        # d(cp)/dt row, then d(q)/dt row
                    r = self._idx(block, i, j)
                    for m in range(NC):
                        rows += [r, r]
                        cols += [self._idx(1, i, m), self._idx(2, i, m)]
        self._j_rows, self._j_cols = np.array(rows), np.array(cols)

    def jac(self, t, y, k):
        n = self.n
        z = y.reshape(3, n, NC)
        _, qbar, cps, K, R = self._binding(z[1], z[2])
        six = np.arange(6)

        d_cp = np.zeros((n, NC, NC))
        d_q = np.zeros((n, NC, NC))
        d_cp[:, 1:, 0] = -self.ka_bar * NU[None, :] * R / cps[:, None]
        d_cp[:, six + 1, six + 1] += self.ka_bar
        blk = -self.ka_bar * (R * NU[None, :] / qbar[:, None])[:, :, None] * self.shield[None, None, :]
        blk[:, six, six] += -self.ka_bar * K
        d_q[:, 1:, 1:] = blk
        # salt row of Eq. (5): dq_s = -sum_i nu_i dq_i
        d_cp[:, 0, :] = -np.einsum('i,nij->nj', NU, d_cp[:, 1:, :])
        d_q[:, 0, :] = -np.einsum('i,nij->nj', NU, d_q[:, 1:, :])

        # layout must match _build_binding_pattern: (cell, comp, block, m, cp|q)
        stack = np.empty((n, NC, 2, NC, 2))
        stack[:, :, 0, :, 0] = -d_cp
        stack[:, :, 0, :, 1] = -d_q
        stack[:, :, 1, :, 0] = d_cp
        stack[:, :, 1, :, 1] = d_q
        size = 3 * n * NC
        j_bind = sparse.coo_matrix((stack.ravel(), (self._j_rows, self._j_cols)),
                                   shape=(size, size)).tocsr()
        return self._j_const + j_bind

    def check_jacobian(self, n_cols=40, seed=0):
        """Central differences on random columns of a loaded, non-trivial state."""
        rng = np.random.default_rng(seed)
        n = self.n
        c = rng.uniform(0.01, 1.0, (n, NC)); c[:, 0] = rng.uniform(40, 600, n)
        cp = c * rng.uniform(0.5, 1.0, (n, NC))
        q = np.zeros((n, NC))
        q[:, 1:] = rng.uniform(0.0, 1.5, (n, 6))
        q[:, 0] = LAMBDA - q[:, 1:] @ NU
        y = np.concatenate([c.ravel(), cp.ravel(), q.ravel()])
        ja = self.jac(100.0, y, 0).toarray()
        err = scale = 0.0
        for j in rng.choice(y.size, n_cols, replace=False):
            h = 1e-7 * max(abs(y[j]), 1.0)
            yp, ym = y.copy(), y.copy()
            yp[j] += h; ym[j] -= h
            fd = (self.rhs(100.0, yp, 0) - self.rhs(100.0, ym, 0)) / (2 * h)
            err = max(err, np.abs(fd - ja[:, j]).max())
            scale = max(scale, np.abs(fd).max())
        return err / scale

    # -- integration ---------------------------------------------------------
    def solve(self, out_dt=4.0, chunk=600.0, rtol=1e-7, atol=1e-9, verbose=True):
        """Integrate over the whole program, keeping only the outlet cell.

        Sections are integrated in chunks so that peak memory stays independent
        of the grid size (`solve_ivp` retains the full state at every requested
        output time, which is ~250 MB per section at n=800 otherwise).
        """
        y = self.initial_state()
        times, outlet = [], []
        for k in range(len(SECTIONS) - 1):
            t0, t1 = SECTIONS[k], SECTIONS[k + 1]
            edges = np.append(np.arange(t0, t1, chunk), t1)
            for a, b in zip(edges[:-1], edges[1:]):
                t_eval = np.arange(a, b + 1e-9, out_dt)
                sol = solve_ivp(self.rhs, (a, b), y, method='BDF', args=(k,),
                                jac=self.jac, t_eval=t_eval, rtol=rtol, atol=atol)
                if not sol.success:
                    raise RuntimeError(f'section {k} [{a:.0f}, {b:.0f}] s: {sol.message}')
                y = sol.y[:, -1].copy()
                times.append(sol.t)
                outlet.append(sol.y.reshape(3, self.n, NC, -1)[0, -1, :, :].T)
            if verbose:
                print(f'  section {k + 1}/{len(SECTIONS) - 1} done (t = {t1:.0f} s)', flush=True)
        t = np.concatenate(times)
        keep = np.concatenate(([True], np.diff(t) > 1e-9))   # chunk edges repeat
        return t[keep], np.vstack(outlet)[keep]


def load_digitized():
    """Curves read from Figure 1; returns {} if they are not available."""
    if not os.path.isdir(DIGITIZED):
        return {}
    out = {}
    for name in list(COMPONENTS) + ['salt']:
        path = os.path.join(DIGITIZED, f'{name}.csv')
        if os.path.isfile(path):
            out[name] = np.genfromtxt(path, delimiter=',', skip_header=1)
    return out


def report(t, outlet, ref):
    od = W_EXT * outlet[:, 1:]
    print(f'\n{"comp":>5} {"t_peak":>9} {"OD_peak":>9} {"area":>10}', end='')
    if ref:
        print(f' | {"t_fig":>8} {"dt":>7} {"OD_fig":>8} {"height err":>11}', end='')
    print()
    shifts = []
    for i, name in enumerate(COMPONENTS):
        j = od[:, i].argmax()
        line = f'{name:>5} {t[j]:9.0f} {od[j, i]:9.4f} {np.trapz(od[:, i], t):10.1f}'
        if name in ref:
            r = ref[name]
            tf, hf = r[r[:, 1].argmax(), 0], r[:, 1].max()
            shifts.append(tf - t[j])
            line += f' | {tf:8.0f} {tf - t[j]:+7.0f} {hf:8.4f} {(od[j, i] / hf - 1) * 100:+10.2f} %'
        print(line)
    if shifts:
        print(f'\npeak-time offset vs Figure 1: mean {np.mean(shifts):+.0f} s, '
              f'spread {max(shifts) - min(shifts):.0f} s '
              f'(a uniform offset shows up as a small spread)')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=400, help='number of finite volumes (default 400)')
    p.add_argument('--denominator', default='paper_eq4', choices=('paper_eq4', 'bound_salt'),
                   help="available-sites term of Eq. (4) (default: as printed)")
    p.add_argument('--ka-scale', type=float, default=1.0,
                   help='multiplier on ka_bar; a large value approaches rapid equilibrium')
    p.add_argument('--out-dt', type=float, default=4.0, help='output interval in s')
    p.add_argument('--check-jacobian', action='store_true')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--save', default=None, help='write the outlet to this .npz')
    args = p.parse_args()

    model = PaperModel(args.n, args.denominator, args.ka_scale)
    print(f"Paper's Eqs. (1),(2),(4),(5); denominator = {args.denominator}, "
          f"n = {args.n}, ka_bar = {model.ka_bar:g} 1/s")
    print(f'  effective axial dispersion {model.dax_eff:.4g} + numerical '
          f'{V_INT * model.dz / 2:.4g} = {DAX:.4g} m^2/s (Table 3)')
    if args.check_jacobian:
        print(f'  analytic Jacobian vs central differences: '
              f'{model.check_jacobian():.2e} relative')

    t, outlet = model.solve(out_dt=args.out_dt)
    if model.floor_hits:
        print(f'  WARNING: available-sites floor was hit {model.floor_hits} times')

    ref = load_digitized()
    report(t, outlet, ref)

    if args.save:
        np.savez(args.save, t=t, outlet=outlet)
        print(f'\nsaved {args.save}')

    if args.plot:
        import matplotlib.pyplot as plt
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, name in enumerate(COMPONENTS):
            if name in ref:
                ax.plot(ref[name][:, 0], ref[name][:, 1], color=colors[i], lw=4, alpha=0.3)
            ax.plot(t, W_EXT * outlet[:, i + 1], color=colors[i], lw=1.2, label=name)
        ax.plot([], [], color='0.4', lw=4, alpha=0.4, label='Figure 1')
        ax.set_xlabel('time (s)'); ax.set_ylabel('optical density (AU/cm)')
        ax.set_title(f"paper's equations, denominator = {args.denominator}")
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        name = os.path.join(HERE, f'Meyer2026_fig1_paper_equations_{args.denominator}.png')
        fig.savefig(name, dpi=130)
        print(f'saved {name}')


if __name__ == '__main__':
    main()
