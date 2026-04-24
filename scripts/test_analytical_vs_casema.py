#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate CADET-Verification analytical references against CASEMA references.

Both reference sets should use the same benchmark settings (from
setting_Col1D_linLRM_1comp_benchmark1 and setting_Col1D_lin_1comp_benchmark1).

The test compares the stored reference solutions at overlapping time points
to ensure consistency.

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

_data_dir = Path(__file__).resolve().parent.parent / 'data'
_casema_dir = _data_dir / 'CASEMA_reference'
_verification_dir = _data_dir / 'CADET-Verification_reference' / 'chromatography'


def _read_reference(filepath):
    """Read reference solution: times and outlet concentration."""
    with h5py.File(filepath, 'r') as f:
        times = f['output/solution/SOLUTION_TIMES'][:]
        outlet = f['output/solution/unit_001/SOLUTION_OUTLET'][:]
    return times, outlet


def validate_pair(name, casema_file, verification_file):
    """Compare CASEMA and CADET-Verification references at overlapping times."""
    print(f"\n{'='*60}")
    print(f"Validating: {name}")
    print(f"{'='*60}")

    casema_times, casema_outlet = _read_reference(casema_file)
    verif_times, verif_outlet = _read_reference(verification_file)

    print(f"  CASEMA:        {len(casema_times)} points,"
          f" t=[{casema_times[0]:.2f}, {casema_times[-1]:.2f}]")
    print(f"  Verification:  {len(verif_times)} points,"
          f" t=[{verif_times[0]:.2f}, {verif_times[-1]:.2f}]")

    # Find overlapping time points
    casema_set = set(np.round(casema_times, 10))
    overlap_mask = np.array([np.round(t, 10) in casema_set for t in verif_times])
    n_overlap = np.sum(overlap_mask)

    if n_overlap == 0:
        print("  No overlapping time points found!")
        return None, None

    # Build matching arrays
    verif_at_overlap = verif_outlet[overlap_mask]
    verif_times_overlap = verif_times[overlap_mask]

    # For each overlapping verification time, find the CASEMA index
    casema_idx = np.searchsorted(casema_times, verif_times_overlap)
    casema_at_overlap = casema_outlet[casema_idx]

    abs_err = np.abs(verif_at_overlap - casema_at_overlap)
    max_val = max(np.max(np.abs(casema_at_overlap)),
                  np.max(np.abs(verif_at_overlap)), 1e-30)

    # Only compare at significant signal levels
    significant = np.abs(casema_at_overlap) > 1e-4 * max_val
    n_significant = np.sum(significant)

    if n_significant > 0:
        norm_err = abs_err[significant] / max_val
        max_norm_err = np.max(norm_err)
        mean_norm_err = np.mean(norm_err)
        max_abs_err = np.max(abs_err[significant])
    else:
        max_norm_err = 0.0
        mean_norm_err = 0.0
        max_abs_err = 0.0

    print(f"  Overlapping points:     {n_overlap}")
    print(f"  Significant points:     {n_significant}")
    print(f"  Max absolute error:     {max_abs_err:.8e}")
    print(f"  Max peak-normalized err:{max_norm_err:.8e}")
    print(f"  Mean peak-normalized err:{mean_norm_err:.8e}")

    passed = max_norm_err < 1e-6
    print(f"  Result: {'PASSED' if passed else 'FAILED'}"
          f" (threshold: 1e-6 peak-normalized)")

    return max_norm_err, passed


def main():
    print("Comparing CADET-Verification references against CASEMA references")
    print("=" * 60)

    results = []

    for name, filename in [
        ('LRM (dynLin, 1 comp)', 'LRM_dynLin_1comp_benchmark1.h5'),
        ('LRMP (dynLin, 1 comp)', 'LRMP_dynLin_1comp_benchmark1.h5'),
        ('GRM (dynLin, 1 comp)', 'GRM_dynLin_1comp_benchmark1.h5'),
    ]:
        casema_file = _casema_dir / filename
        verif_file = _verification_dir / filename

        if not casema_file.exists():
            print(f"\n  SKIPPED {name}: {casema_file} not found")
            continue
        if not verif_file.exists():
            print(f"\n  SKIPPED {name}: {verif_file} not found")
            continue

        max_err, passed = validate_pair(name, casema_file, verif_file)
        if max_err is not None:
            results.append((name, max_err, passed))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, max_err, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: max peak-normalized error = {max_err:.2e}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll validations passed.")
    else:
        print("\nSome validations FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    main()
