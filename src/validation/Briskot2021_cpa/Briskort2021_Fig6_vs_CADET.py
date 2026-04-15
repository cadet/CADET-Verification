"""
Comparison of CADET, GoSilico, and experimental data (Fig. 6).
Signed absolute and relative errors at the measured points.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Briskot2021_fig6 import run_batch_equilibrium

# ─── Data from Briskot2021_fig6.py ───────────────────────────────────
PH_VALUES = [5.5, 6.0, 6.5, 7.0]
PH_IM_MAP = {
    5.5: [20, 95, 145, 220],
    6.0: [20, 70, 95, 120],
    6.5: [20, 45, 70, 95],
    7.0: [20, 45, 95],
}

EXPERIMENTAL_DATA = {
    5.5: {
        20:  {'c': [1.743e-2,3.848e-2,5.561e-2,8.868e-2,9.259e-2],
              'q': [3.345,3.467,3.178,3.558,3.315]},
        95:  {'c': [7.214e-3,2.345e-2,4.449e-2,6.523e-2,6.673e-2,8.086e-2,8.176e-2,1.007e-1,1.025e-1],
              'q': [1.685,1.959,1.959,2.096,2.036,2.081,2.036,2.127,2.005]},
        145: {'c': [2.705e-2,3.667e-2,6.854e-2,7.275e-2,8.417e-2,1.314e-1,1.32e-1,1.007e-1],
              'q': [0.4975,0.5888,0.7868,1.198,0.802,1.015,1.0,0.5431]},
        220: {'c': [1.285e-2,1.942e-2,5.289e-2,8.546e-2,1.312e-1,1.377e-1,1.425e-1],
              'q': [0.07107,0.07107,0.07107,0.07107,0.132,0.132,0.1624]},
    },
    6.0: {
        20:  {'c': [4.633e-3,3.041e-2,5.444e-2,5.82e-2],
              'q': [3.837,3.658,3.97,3.822]},
        70:  {'c': [4.923e-3,1.911e-2,3.388e-2,5.618e-2,6.197e-2,7.905e-2,8.108e-2],
              'q': [1.564,1.817,2.069,2.158,2.366,2.589,2.455]},
        95:  {'c': [2.317e-2,2.635e-2,4.517e-2,5.097e-2,6.515e-2,6.921e-2,8.6e-2,8.832e-2,8.919e-2,1.181e-1],
              'q': [1.163,1.178,1.282,1.223,1.446,1.416,1.416,1.609,1.49,1.55]},
        120: {'c': [3.185e-2,3.793e-2,6.892e-2,7.181e-2,8.369e-2,8.977e-2,1.219e-1,1.251e-1,9.295e-2],
              'q': [0.198,0.2277,0.4802,0.4505,0.4505,0.5396,0.9703,0.8515,0.2871]},
    },
    6.5: {
        20:  {'c': [4.5e-3,1.26e-2,1.92e-2,5.28e-2,5.46e-2],
              'q': [2.316,3.571,3.357,3.969,3.847]},
        45:  {'c': [6e-4,4.5e-3,1.05e-2,3.63e-2,6.36e-2,6.72e-2,7.86e-2],
              'q': [0.9541,1.719,1.888,2.286,2.439,2.224,2.393]},
        70:  {'c': [1.26e-2,2.28e-2,4.11e-2,6.09e-2,7.83e-2,7.98e-2,9.96e-2,1.035e-1],
              'q': [0.6633,0.8929,1.23,1.352,1.413,1.306,1.49,1.413]},
        95:  {'c': [6.63e-2,7.26e-2,7.5e-2,7.86e-2,8.28e-2,8.97e-2,9.18e-2,1.014e-1,1.293e-1,1.344e-1,1.404e-1,1.437e-1],
              'q': [0.4184,0.5255,0.6173,0.6173,0.4184,0.4949,0.4184,0.4184,0.602,0.8469,0.4949,0.5561]},
    },
    7.0: {
        20:  {'c': [8.806e-4,2.935e-4,1.937e-2,5.46e-2,5.607e-2,6.164e-2],
              'q': [1.03,2.275,3.04,3.265,3.19,3.64]},
        45:  {'c': [1.82e-2,2.73e-2,4.99e-2,6.869e-2,8.513e-2,8.718e-2,1.115e-1],
              'q': [0.55,0.79,1.12,1.24,1.39,1.24,1.495]},
        95:  {'c': [7.867e-2,8.307e-2,1.221e-1,1.391e-1],
              'q': [0.085,0.265,0.235,0.34]},
    },
}

GO_SILICO_DATA = {
    5.5: {
        20:  {'c': [1.743e-2,3.848e-2,5.561e-2,8.868e-2,9.259e-2],
              'q': [3.551,3.551,3.551,3.551,3.564]},
        95:  {'c': [7.214e-3,2.345e-2,4.449e-2,6.523e-2,6.673e-2,8.086e-2,8.176e-2,1.007e-1,1.025e-1],
              'q': [1.859,2.11,2.216,2.269,2.269,2.295,2.295,2.322,2.322]},
        145: {'c': [2.705e-2,3.667e-2,6.854e-2,7.275e-2,8.417e-2,1.314e-1,1.32e-1,1.007e-1],
              'q': [3.392e-1,4.317e-1,6.167e-1,6.3e-1,6.828e-1,8.15e-1,8.414e-1]},
        220: {'c': [1.285e-2,1.942e-2,5.289e-2,8.546e-2,1.312e-1,1.377e-1,1.425e-1],
              'q': [2.203e-2,3.524e-2,2.203e-2,3.524e-2,3.524e-2,4.846e-2,3.524e-2]},
    },
    6.0: {
        20:  {'c': [4.633e-3,3.041e-2,5.444e-2,5.82e-2],
              'q': [3.403,3.403,3.403,3.415]},
        70:  {'c': [4.923e-3,1.911e-2,3.388e-2,5.618e-2,6.197e-2,7.905e-2,8.108e-2],
              'q': [1.788,2.068,2.195,2.22,2.233,2.271,2.284]},
        95:  {'c': [2.317e-2,2.635e-2,4.517e-2,5.097e-2,6.515e-2,6.921e-2,8.6e-2,8.832e-2,8.919e-2,1.181e-1],
              'q': [6.695e-1,7.331e-1,9.237e-1,9.619e-1,1.153,1.191,1.267,1.28,1.356,1.381]},
        120: {'c': [3.185e-2,3.793e-2,6.892e-2,7.181e-2,8.369e-2,8.977e-2,1.219e-1,1.251e-1,9.295e-2],
              'q': [1.864e-1,2.246e-1,2.246e-1,3.644e-1,3.771e-1,4.153e-1,4.153e-1,4.407e-1]},
    },
    6.5: {
        20:  {'c': [4.5e-3,1.26e-2,1.92e-2,5.28e-2,5.46e-2],
              'q': [3.197,3.224,3.224,3.263,3.263]},
        45:  {'c': [6e-4,4.5e-3,1.05e-2,3.63e-2,6.36e-2,6.72e-2,7.86e-2],
              'q': [2.145,2.263,2.447,2.487,2.5,2.513]},
        70:  {'c': [1.26e-2,2.28e-2,4.11e-2,6.09e-2,7.83e-2,7.98e-2,9.96e-2,1.035e-1],
              'q': [6.579e-1,8.947e-1,1.079,1.224,1.303,1.303,1.355,1.395]},
        95:  {'c': [6.63e-2,7.26e-2,7.5e-2,7.86e-2,8.28e-2,8.97e-2,9.18e-2,1.014e-1,1.293e-1,1.344e-1,1.404e-1,1.437e-1],
              'q': [1.974e-1,1.842e-1,2.632e-1,2.895e-1,2.895e-1,2.763e-1,3.026e-1,2.895e-1,3.026e-1,3.289e-1,3.684e-1,3.684e-1]},
    },
    7.0: {
        20:  {'c': [8.806e-4,2.935e-4,1.937e-2,5.46e-2,5.607e-2,6.164e-2],
              'q': [2.849,2.978,3.017,3.017,3.004]},
        45:  {'c': [1.82e-2,2.73e-2,4.99e-2,6.869e-2,8.513e-2,8.718e-2,1.115e-1],
              'q': [1.181,1.349,1.491,1.634,1.672,1.685,1.75]},
        95:  {'c': [7.867e-2,8.307e-2,1.221e-1,1.391e-1],
              'q': [1.724e-2,6.897e-2,5.603e-2]},
    },
}

# Consistent colors for each ionic strength
IM_COLORS = {20: '#e41a1c', 45: '#ff7f00', 70: '#4daf4a',
             95: '#377eb8', 120: '#984ea3', 145: '#a65628', 220: '#999999'}
IM_MARKERS = {20: 'o', 45: 's', 70: 'D', 95: '^', 120: 'v', 145: 'P', 220: 'X'}


def get_cadet_q(c_list, pH, Im):
    """CADET simulation (CSTR batch equilibrium, as in Briskot2021_fig6.py)."""
    q_vals = []
    for c in c_list:
        try:
            _, q_eq = run_batch_equilibrium(float(c), float(pH), int(Im))
            q_vals.append(q_eq)
        except Exception as ex:
            print(f'  CADET error at c={c}, pH={pH}, Im={Im}: {ex}')
            q_vals.append(float('nan'))
    return q_vals


def collect_errors(pH, Im, exp_data, gs_data, cadet_func):
    """Return (c, q_exp, q_cadet, q_gs) for points with all three values available."""
    c_exp  = np.array(exp_data['c'])
    q_exp  = np.array(exp_data['q'])

    if Im not in gs_data:
        return [], [], [], []

    # GoSilico may have fewer points; zip truncates automatically
    c_gs = gs_data[Im]['c']
    q_gs = list(zip(c_gs, gs_data[Im]['q']))   # (c, q) pairs

    # Include only points available in both experimental and GoSilico data
    # GoSilico uses the same c values as the experimental data (same measured points)
    results = []
    gs_idx = 0
    for c_e, q_e in zip(c_exp, q_exp):
        if gs_idx >= len(q_gs):
            break
        c_g, q_g = q_gs[gs_idx]
        gs_idx += 1
        q_c = cadet_func([c_e], pH, Im)[0]
        results.append((c_e, q_e, q_c, q_g))

    if not results:
        return [], [], [], []
    arr = np.array(results)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]



def _collect_all(pH):
    """Return error arrays for each ionic strength."""
    data = {}
    for Im in PH_IM_MAP[pH]:
        if Im not in EXPERIMENTAL_DATA[pH]:
            continue
        c_pts, q_exp, q_cadet, q_gs = collect_errors(
            pH, Im, EXPERIMENTAL_DATA[pH][Im], GO_SILICO_DATA[pH], get_cadet_q)
        if len(c_pts) == 0:
            continue
        data[Im] = dict(
            c=c_pts, q_exp=q_exp, q_cadet=q_cadet, q_gs=q_gs,
            abs_c=q_cadet - q_exp, abs_g=q_gs - q_exp,
            rel_c=(q_cadet - q_exp) / q_exp * 100,
            rel_g=(q_gs    - q_exp) / q_exp * 100,
        )
    return data


def R2(exp, mod):
    exp, mod = np.array(exp), np.array(mod)
    ss_res = np.sum((exp - mod)**2)
    ss_tot = np.sum((exp - np.mean(exp))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _format_ax(ax, xlabel, ylabel, title, zero_line=False):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    if zero_line:
        ax.axhline(0, color='k', lw=1.0)


def plot_isotherms(results_by_ph):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    fig.suptitle('Figure 6 Briskot et. data and simulation vs CADET', fontsize=13, fontweight='bold')

    for idx, pH in enumerate(PH_VALUES):
        ax = axes[idx // 2][idx % 2]
        data = results_by_ph[pH]
        q_exp_c, q_mod_c, q_exp_g, q_mod_g = [], [], [], []

        for Im, d in data.items():
            col, mk = IM_COLORS[Im], IM_MARKERS[Im]
            ax.scatter(d['c'], d['q_exp'],   color=col, marker=mk, s=55,
                       zorder=5, label=f'{Im} mM', edgecolors='k', linewidths=0.5)
            ax.plot(d['c'], d['q_cadet'], color=col, ls='-',  lw=1.6, zorder=4)
            ax.plot(d['c'], d['q_gs'],    color=col, ls='--', lw=1.6, zorder=3)
            q_exp_c.extend(d['q_exp']); q_mod_c.extend(d['q_cadet'])
            q_exp_g.extend(d['q_exp']); q_mod_g.extend(d['q_gs'])

        r2_c = R2(q_exp_c, q_mod_c)
        r2_g = R2(q_exp_g, q_mod_g)

        _format_ax(ax, r'$c$ [mol m$^{-3}$]', r'$q$ [mol m$^{-3}$]', f'pH {pH}')
        ax.set_ylim(bottom=0)

        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        extras = [
            Line2D([0],[0], color='k', ls='-',  lw=1.6, label='CADET'),
            Line2D([0],[0], color='k', ls='--', lw=1.6, label='GoSilico'),
            Line2D([0],[0], color='k', ls='none', marker='o', ms=6,
                   markeredgecolor='k', markerfacecolor='none', label='Experiment'),
        ]
        ax.legend(handles=list(seen.values()) + extras,
                  labels=list(seen.keys()) + ['CADET','GoSilico','Experiment'],
                  fontsize=7, loc='lower right', ncol=2)
        ax.text(0.02, 0.97, f'R²(CADET)={r2_c:.3f}\nR²(GoSilico)={r2_g:.3f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'fig1_isotherms.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'Saved: {out}')


def plot_abs_errors(results_by_ph):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    fig.suptitle('Absolute error', fontsize=13, fontweight='bold')

    for idx, pH in enumerate(PH_VALUES):
        ax = axes[idx // 2][idx % 2]
        data = results_by_ph[pH]

        for k, (Im, d) in enumerate(data.items()):
            col = IM_COLORS[Im]
            offset = 0.0008 * (k - len(data) / 2)
            c = d['c'] + offset

            ax.scatter(c, d['abs_c'], color=col, marker='o', s=55,
                       zorder=5, label=f'{Im} mM', edgecolors='none')
            ax.scatter(c, d['abs_g'], color=col, marker='^', s=55,
                       zorder=5, edgecolors='none', alpha=0.55)
            for ci, ae, ag in zip(c, d['abs_c'], d['abs_g']):
                ax.vlines(ci, 0, ae, color=col, lw=0.8, alpha=0.35)
                ax.vlines(ci, 0, ag, color=col, lw=0.8, ls=':', alpha=0.35)

        _format_ax(ax,
                   r'$c$ [mol m$^{-3}$]',
                   r'absolute error [mol m$^{-3}$]',
                   f'pH {pH}', zero_line=True)

        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        extras = [
            Line2D([0],[0], color='k', marker='o', ls='none', ms=7, label='CADET'),
            Line2D([0],[0], color='k', marker='^', ls='none', ms=7, alpha=0.55, label='GoSilico'),
        ]
        ax.legend(handles=list(seen.values()) + extras,
                  labels=list(seen.keys()) + ['CADET','GoSilico'],
                  fontsize=7, loc='best', ncol=2)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'fig2_abs_errors.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'Saved: {out}')


def plot_rel_errors(results_by_ph):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    fig.suptitle('Relative Error',
                 fontsize=13, fontweight='bold')

    for idx, pH in enumerate(PH_VALUES):
        ax = axes[idx // 2][idx % 2]
        data = results_by_ph[pH]

        for k, (Im, d) in enumerate(data.items()):
            col = IM_COLORS[Im]
            offset = 0.0008 * (k - len(data) / 2)
            c = d['c'] + offset

            ax.scatter(c, d['rel_c'], color=col, marker='o', s=55,
                       zorder=5, label=f'{Im} mM', edgecolors='none')
            ax.scatter(c, d['rel_g'], color=col, marker='^', s=55,
                       zorder=5, edgecolors='none', alpha=0.55)
            for ci, re, rg in zip(c, d['rel_c'], d['rel_g']):
                ax.vlines(ci, 0, re, color=col, lw=0.8, alpha=0.35)
                ax.vlines(ci, 0, rg, color=col, lw=0.8, ls=':', alpha=0.35)

        _format_ax(ax,
                   r'$c$ [mol m$^{-3}$]',
                   r'relative error [%]',
                   f'pH {pH}', zero_line=True)

        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        extras = [
            Line2D([0],[0], color='k', marker='o', ls='none', ms=7, label='CADET'),
            Line2D([0],[0], color='k', marker='^', ls='none', ms=7, alpha=0.55, label='GoSilico'),
        ]
        ax.legend(handles=list(seen.values()) + extras,
                  labels=list(seen.keys()) + ['CADET','GoSilico'],
                  fontsize=7, loc='best', ncol=2)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'fig3_rel_errors.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)

    print('Computing isotherms...')
    results_by_ph = {pH: _collect_all(pH) for pH in PH_VALUES}

    plot_isotherms(results_by_ph)
    plot_abs_errors(results_by_ph)
    plot_rel_errors(results_by_ph)
    plt.show()
