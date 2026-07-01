# -*- coding: utf-8 -*-
"""
Validation tests: compare CADET-Core simulations against experimental data.
"""

import os
import tempfile
import numpy as np
from pathlib import Path

from cadet import Cadet

import src.benchmark_models.setting_membraneRadialFlowEffioEtAl as setting_effio

_reference_data_path_ = str(
    Path(__file__).resolve().parent.parent / 'data' / 'validation'
)

F_M3_PER_S = 5.0e-8   # 3 mL/min in m^3/s
F_ML_PER_S = F_M3_PER_S * 1e6  # 0.05 mL/s


def _run_simulation(model_config, cadet_path):
    """Run a CADET simulation and return (times, outlet)."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        fname = tmp.name

    try:
        sim = Cadet(install_path=cadet_path)
        sim.root.input = model_config['input']
        sim.filename = fname
        sim.save()
        data = sim.run_simulation()
        if not data.return_code == 0:
            raise RuntimeError(f"CADET simulation failed:\n{data.error_message}")
        sim.load()

        times = np.array(sim.root.output.solution.solution_times)
        outlet = np.array(sim.root.output.solution.unit_001.solution_outlet)
        if outlet.ndim == 2:
            outlet = outlet[:, 0]
    finally:
        if os.path.exists(fname):
            os.remove(fname)

    return times, outlet


def _load_csv(filename):
    data = np.loadtxt(os.path.join(_reference_data_path_, filename), delimiter=',')
    idx = np.argsort(data[:, 0])
    return data[idx, 0], data[idx, 1]


def _plot_effio(dextran, nacl, output_path):
    """Two-panel comparison against Effio et al. (2016), Fig. 3."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    panels = [
        (ax1, dextran, 'effio_dextran_meas.csv', 'effio_dextran_sim.csv',
         '(a) Dextran T2000', 3.5),
        (ax2, nacl, 'effio_nacl_meas.csv', 'effio_nacl_sim.csv',
         '(b) NaCl', 6.0),
    ]

    for ax, (times, outlet), meas_file, sim_file, title, xmax in panels:
        meas_V, meas_y = _load_csv(meas_file)
        sim_V, sim_y = _load_csv(sim_file)

        cadet_n = outlet / outlet.max()
        V_cadet = times * F_ML_PER_S

        ax.plot(V_cadet, cadet_n, '-', color='#d62728', linewidth=1.8,
                label='CADET-Core DG', zorder=3)
        ax.plot(sim_V, sim_y / sim_y.max(), '-', color='#1f77b4',
                linewidth=1.5, label='Effio et al. (sim.)', zorder=2)
        ax.plot(meas_V, meas_y / meas_y.max(), 'o', color='#2ca02c',
                markersize=4, label='Effio et al. (exp.)', zorder=4)
        ax.set_xlabel('Elution volume [mL]', fontsize=13)
        ax.set_ylabel('Normalized outlet signal [–]', fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, xmax)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, which='major', alpha=0.3)

    fig.tight_layout()
    out_file = os.path.join(output_path, 'membraneRadialFlowEffioEtAl.png')
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Validation plot saved: {out_file}")


def validation_tests(output_path, cadet_path):
    """Run all validation tests and save comparison plots to *output_path*."""
    os.makedirs(output_path, exist_ok=True)

    print("Running validation: membraneRadialFlowEffioEtAl (Effio et al. 2016)")
    print("  Fig. 3a: Dextran T2000 (pure radial transport)")
    dextran = _run_simulation(setting_effio.get_model_dextran(), cadet_path)
    print("  Fig. 3b: NaCl (radial LRMP)")
    nacl = _run_simulation(setting_effio.get_model_nacl(), cadet_path)

    _plot_effio(dextran, nacl, output_path)
