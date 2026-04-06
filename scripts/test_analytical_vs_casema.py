#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate Python analytical solutions against CASEMA semi-analytic references.

For each model (LRM, LRMP, GRM), the test reads the CASEMA reference HDF5
file, extracts its model parameters and solution times, computes the Python
analytical solution with the *same* parameters, and compares the outlet
concentrations.

Note: The CASEMA reference files use different benchmark settings than the
CADET-Verification Python-generated references (different injection times,
time ranges, and for LRMP different model parameters). This test validates
the Python inverse-Laplace solver itself, not the stored reference files.

Usage
-----
    python scripts/test_analytical_vs_casema.py
"""

import sys
from pathlib import Path

import numpy as np
import h5py

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utility.analytical_solutions import compute_analytical_outlet

_casema_dir = Path(__file__).resolve().parent.parent / 'data' / 'CASEMA_reference'


def _read_casema_reference(filepath):
    """Read CASEMA reference: solution times, outlet, and model parameters."""
    with h5py.File(filepath, 'r') as f:
        times = f['output/solution/SOLUTION_TIMES'][:]
        outlet = f['output/solution/unit_001/SOLUTION_OUTLET'][:]

        section_times = f['input/solver/sections/SECTION_TIMES'][:]
        c_in = float(f['input/model/unit_000/sec_000/CONST_COEFF'][0])
        t_inj = float(section_times[1])

        u1 = f['input/model/unit_001']
        unit_type = u1['UNIT_TYPE'][()].decode()

        params = {
            'velocity': float(u1['VELOCITY'][()]),
            'col_length': float(u1['COL_LENGTH'][()]),
            'c_in': c_in,
            't_inj': t_inj,
        }

        # COL_DISPERSION may be scalar or array
        col_disp = u1['COL_DISPERSION'][()]
        params['col_dispersion'] = (
            float(col_disp) if np.ndim(col_disp) == 0 else float(col_disp[0])
        )

        params['ka'] = float(u1['adsorption/LIN_KA'][0])
        params['kd'] = float(u1['adsorption/LIN_KD'][0])

        if 'WITHOUT_PORES' in unit_type:
            model_type = 'LRM'
            params['total_porosity'] = float(u1['TOTAL_POROSITY'][()])
        elif 'LUMPED_RATE_MODEL_WITH_PORES' in unit_type:
            model_type = 'LRMP'
            params['col_porosity'] = float(u1['COL_POROSITY'][()])
            params['par_porosity'] = float(u1['PAR_POROSITY'][()])
            params['par_radius'] = float(u1['PAR_RADIUS'][()])
            film_diff = u1['FILM_DIFFUSION'][()]
            params['film_diffusion'] = (
                float(film_diff) if np.ndim(film_diff) == 0
                else float(film_diff[0])
            )
        elif 'GENERAL_RATE_MODEL' in unit_type:
            model_type = 'GRM'
            params['col_porosity'] = float(u1['COL_POROSITY'][()])
            params['par_porosity'] = float(u1['PAR_POROSITY'][()])
            params['par_radius'] = float(u1['PAR_RADIUS'][()])
            film_diff = u1['FILM_DIFFUSION'][()]
            params['film_diffusion'] = (
                float(film_diff) if np.ndim(film_diff) == 0
                else float(film_diff[0])
            )
            params['pore_diffusion'] = float(u1['PAR_DIFFUSION'][()])
            params['surface_diffusion'] = float(u1['PAR_SURFDIFFUSION'][()])
        else:
            raise ValueError(f"Unknown unit type: {unit_type}")

        # Read CASEMA accuracy metadata
        casema_error = None
        if 'meta' in f and 'ERROR' in f['meta']:
            casema_error = f['meta']['ERROR'][()].decode()

    return model_type, params, times, outlet, casema_error


def validate_model(name, filepath, dps=50):
    """Compute Python analytical solution with CASEMA's parameters and compare."""
    print(f"\n{'='*60}")
    print(f"Validating: {name}")
    print(f"{'='*60}")

    model_type, params, casema_times, casema_outlet, casema_error = (
        _read_casema_reference(filepath)
    )

    print(f"  Model type:    {model_type}")
    print(f"  Time range:    {casema_times[0]:.2f} .. {casema_times[-1]:.2f}"
          f" ({len(casema_times)} points)")
    print(f"  Key params:    velocity={params['velocity']:.6e},"
          f" col_disp={params['col_dispersion']:.6e},"
          f" t_inj={params['t_inj']:.1f}")
    if casema_error:
        print(f"  CASEMA error:  {casema_error}")

    # Compute Python analytical at the same time points
    print(f"  Computing Python analytical ({dps} dps) ...")
    python_outlet = compute_analytical_outlet(
        model_type, params, casema_times, dps=dps
    )

    # Compare — only at points where the CASEMA signal is physically
    # meaningful (positive concentration, above noise floor).
    # Some CASEMA files exhibit numerical artifacts (large negative
    # values) at late times; these are excluded.
    max_casema = np.max(casema_outlet)  # peak positive value
    significant = (
        (casema_outlet > 1e-4 * max_casema)
        & (python_outlet >= 0.0)
    )
    n_significant = np.sum(significant)

    abs_err = np.abs(python_outlet - casema_outlet)

    if n_significant > 0:
        # Normalize by peak value to avoid division-by-near-zero
        norm_err = abs_err[significant] / max_casema
        max_norm_err = np.max(norm_err)
        mean_norm_err = np.mean(norm_err)
        max_abs_err = np.max(abs_err[significant])
    else:
        max_norm_err = 0.0
        mean_norm_err = 0.0
        max_abs_err = 0.0

    # Report any CASEMA artifacts
    n_negative = np.sum(casema_outlet < -1e-10 * max_casema)
    if n_negative > 0:
        print(f"  WARNING: CASEMA has {n_negative} unphysical negative"
              f" values (excluded from comparison)")

    print(f"  Max |CASEMA outlet|:    {max_casema:.8e}")
    print(f"  Significant points:     {n_significant}")
    print(f"  Max absolute error:     {max_abs_err:.8e}")
    print(f"  Max peak-normalized err:{max_norm_err:.8e}")
    print(f"  Mean peak-normalized err:{mean_norm_err:.8e}")

    # Show worst significant point
    if n_significant > 0:
        sig_indices = np.where(significant)[0]
        worst_sig = sig_indices[np.argmax(abs_err[significant])]
        print(f"  Worst point:  t={casema_times[worst_sig]:.2f},"
              f" python={python_outlet[worst_sig]:.6e},"
              f" casema={casema_outlet[worst_sig]:.6e}")

    return max_norm_err, mean_norm_err


def main():
    print("Validating Python analytical solutions against CASEMA references")
    print("=" * 60)

    results = []

    for name, filename in [
        ('LRM (dynLin, 1 comp)', 'ref_LRM_dynLin_1comp_benchmark1.h5'),
        ('LRMP (dynLin, 1 comp)', 'ref_LRMP_dynLin_1comp_benchmark1.h5'),
        ('GRM (dynLin, 1 comp)', 'ref_GRM_dynLin_1comp_benchmark1.h5'),
    ]:
        filepath = _casema_dir / filename
        if not filepath.exists():
            print(f"\n  SKIPPED {name}: {filepath} not found")
            continue
        max_rel, mean_rel = validate_model(name, filepath)
        results.append((name, max_rel, mean_rel))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Python analytical vs CASEMA accuracy")
    print(f"{'='*60}")
    for name, max_norm, mean_norm in results:
        print(f"  {name}:")
        print(f"    max peak-normalized error  = {max_norm:.2e}")
        print(f"    mean peak-normalized error = {mean_norm:.2e}")

    print("\nNote: discrepancies may arise from differences in the")
    print("inverse Laplace transform implementation or model formulation")
    print("conventions between the Python solver and CASEMA.")


if __name__ == '__main__':
    main()
