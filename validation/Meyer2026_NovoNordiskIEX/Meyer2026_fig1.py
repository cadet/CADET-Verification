# -*- coding: utf-8 -*-
"""
Reproduction of Fig. 1 from:

    Meyer et al., 2026, Computers and Chemical Engineering,
    "ChromOps.jl: High-order simulation and discrete forward sensitivity
     analysis for chromatography models", Section 4.2: "Ion-exchange
     chromatography with six components".

Runs the CADET setup defined in
`src/benchmark_models/setting_COL1D_NovoNordiskIEXbenchmark.py` under both
documented readings of the SMA available-sites term, overlays the curves read
from Figure 1, and prints the validation metrics. Further explanation on model
and parameter selection, and on the two inconsistencies between the paper's
equations and its Figure 1, is provided under Meyer2026_fig1.md.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from cadet import Cadet

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from src.benchmark_models.setting_COL1D_NovoNordiskIEXbenchmark import get_model  # noqa: E402

CADET_PATH = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"

DIGITIZED = os.path.join(HERE, 'Meyer2026_fig1_digitized')
COMPONENTS = 'ABCDEF'
W_EXT = 1.167e4 / 1000.0   # extinction coefficient, AU L/(mol cm) -> per mol/m^3
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']

N_ELEM = 256          # FV cells; the solution is grid converged (256 == 1024)
SPATIAL_METHOD = 0    # 0 = FV/WENO3

# ---------------------------------------------------------------------------
# The published chromatogram is uniformly later than the stated inlet program
# (Eqs. 35-40) plus the column hold-up implied by Table 3 can produce, by
# 216 s. Salt is unretained here (q_s is constant while no protein is bound),
# so it is an isotherm-independent clock: Table 3 gives an outlet delay of
# (L/v_int)*(1 + (1-eps)*eps_p/eps) = 283 s, whereas Figure 1 shows 499 s. The
# gap cannot be hold-up -- closing it would need a non-flowing accessible
# volume fraction above 1.0 -- and the paper reports no dead volume or
# equilibration phase that would account for it. See Meyer2026_fig1.md.
#
# It is therefore applied to the *reference data only*, never to the
# simulation: the CADET input stays exactly on the paper's stated program and
# contains no fitted quantity.
FIGURE_TIME_OFFSET = 216.0
# ---------------------------------------------------------------------------


def load_digitized():
    return {name: np.genfromtxt(os.path.join(DIGITIZED, f'{name}.csv'),
                                delimiter=',', skip_header=1)
            for name in list(COMPONENTS) + ['salt']}


def run(sim, shielding):
    sim.root = get_model(spatial_method_bulk=SPATIAL_METHOD, axNElem=N_ELEM,
                         shielding=shielding, idas_reftol=1e-8)
    sim.filename = os.path.join(HERE, f'Meyer2026_fig1_{shielding}.h5')
    sim.save()
    result = sim.run_simulation()
    if result.return_code != 0:
        raise RuntimeError(f'CADET failed ({shielding}): {result.error_message}\n{result.log}')
    sim.load_from_file()
    return (np.asarray(sim.root.output.solution.solution_times),
            np.asarray(sim.root.output.solution.unit_000.solution_outlet))


def metrics(t, outlet, ref, label):
    """Peak and full-curve deviations; reference data carries the offset."""
    print(f'\n{label}')
    print(f'  {"comp":>4} {"t_peak":>8} {"t_fig":>8} {"dt":>7} '
          f'{"OD_peak":>8} {"OD_fig":>8} {"height":>8} {"area":>7} {"NRMSE":>7}')
    nrmse_all, dt_all = [], []
    for i, name in enumerate(COMPONENTS):
        od = W_EXT * outlet[:, i + 1]
        r = ref[name]
        t_ref = r[:, 0] - FIGURE_TIME_OFFSET
        j, jf = od.argmax(), r[:, 1].argmax()
        dt = r[jf, 0] - FIGURE_TIME_OFFSET - t[j]
        h_err = od[j] / r[jf, 1] - 1.0
        a_err = np.trapz(od, t) / np.trapz(r[:, 1], t_ref) - 1.0
        nrmse = np.sqrt(np.mean((np.interp(t_ref, t, od) - r[:, 1])**2)) / r[jf, 1]
        nrmse_all.append(nrmse); dt_all.append(dt)
        print(f'  {name:>4} {t[j]:8.0f} {r[jf, 0] - FIGURE_TIME_OFFSET:8.0f} {dt:+7.0f} '
              f'{od[j]:8.4f} {r[jf, 1]:8.4f} {h_err:+7.2%} {a_err:+7.2%} {nrmse:7.2%}')
    print(f'  mean NRMSE {np.mean(nrmse_all):.2%}, '
          f'max |dt_peak| {max(abs(d) for d in dt_all):.0f} s')
    return np.mean(nrmse_all)


def main():
    ref = load_digitized()
    sim = Cadet(install_path=CADET_PATH)

    print('=' * 76)
    print(f'Reference data offset by -{FIGURE_TIME_OFFSET:.0f} s (see header and .md);')
    print('the simulation runs on the paper\'s stated inlet program unchanged.')

    results = {}
    for shielding, label in [
            ('paper_eq4', 'A. Eq. (4) as printed: qbar = Lambda - sum_j (nu_j + sigma_j) q_j'),
            ('bound_salt', 'B. qbar = Lambda - sum_j nu_j q_j  (= the bound salt q_s of Eq. 5)')]:
        t, outlet = run(sim, shielding)
        results[shielding] = (t, outlet)
        metrics(t, outlet, ref, label)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    titles = {'paper_eq4': 'Eq. (4) as printed:  $\\bar q = \\Lambda - \\sum_j (\\nu_j+\\sigma_j) q_j$',
              'bound_salt': 'bound salt:  $\\bar q = \\Lambda - \\sum_j \\nu_j q_j = q_s$'}
    for ax, shielding in zip(axes, ('paper_eq4', 'bound_salt')):
        t, outlet = results[shielding]
        for i, name in enumerate(COMPONENTS):
            ax.plot(ref[name][:, 0] - FIGURE_TIME_OFFSET, ref[name][:, 1],
                    color=COLORS[i], lw=4, alpha=0.30)
            ax.plot(t, W_EXT * outlet[:, i + 1], color=COLORS[i], lw=1.2, label=name)
        ax.plot([], [], color='0.4', lw=4, alpha=0.4,
                label=f'Fig. 1 (shifted $-${FIGURE_TIME_OFFSET:.0f} s)')
        ax.set_ylabel('optical density (AU/cm)')
        ax.set_title(titles[shielding], fontsize=10)
        ax.set_xlim(0, 10300)
        ax.legend(ncol=4, fontsize=8)
    axes[1].set_xlabel('time (s)')
    fig.suptitle('CADET (thin) vs. Meyer et al. (2026) Figure 1 (thick, faded)', fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, 'Meyer2026_fig1_comparison.png')
    fig.savefig(out, dpi=130)
    print(f'\nsaved {out}')

    # The salt trace is the isotherm-independent clock behind FIGURE_TIME_OFFSET.
    t, outlet = results['paper_eq4']
    fig2, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ref['salt'][:, 0], ref['salt'][:, 1], color='0.4', lw=4, alpha=0.4,
            label='Fig. 1, as published')
    ax.plot(ref['salt'][:, 0] - FIGURE_TIME_OFFSET, ref['salt'][:, 1], 'k--', lw=1,
            label=f'Fig. 1, shifted $-${FIGURE_TIME_OFFSET:.0f} s')
    ax.plot(t, outlet[:, 0] / 1000.0, color='#0072B2', lw=1.4, label='CADET outlet')
    ax.set_xlabel('time (s)'); ax.set_ylabel('salt (mol/L)')
    ax.set_title('Salt is unretained, so it dates the chromatogram independently '
                 'of the isotherm', fontsize=10)
    ax.legend(fontsize=8)
    fig2.tight_layout()
    out2 = os.path.join(HERE, 'Meyer2026_fig1_salt_offset.png')
    fig2.savefig(out2, dpi=130)
    print(f'saved {out2}')


if __name__ == '__main__':
    main()
