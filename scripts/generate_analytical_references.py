#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate analytical reference solutions for 1D linear chromatography models.

This script computes the analytical outlet concentration for the LRM, LRMP,
and GRM benchmark settings using Laplace-domain transfer functions with
numerical inverse Laplace transform (de Hoog algorithm via mpmath).

The results are stored as HDF5 files in data/CADET-Core_reference/chromatography/
for use as static convergence references.

Usage
-----
    python scripts/generate_analytical_references.py

"""

import sys
import os
import time
from pathlib import Path

import numpy as np
import h5py

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utility.analytical_solutions import (
    get_LRM_analytical_reference,
    get_LRMP_analytical_reference,
    get_GRM_analytical_reference,
)


def save_analytical_reference(filepath, solution_times, outlet_concentration):
    """Save analytical reference solution in CADET-compatible HDF5 format."""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset(
            '/output/solution/unit_001/SOLUTION_OUTLET',
            data=outlet_concentration
        )
        f.create_dataset(
            '/output/solution/SOLUTION_TIMES',
            data=solution_times
        )
    print(f"  Saved to {filepath}")


def main():
    output_dir = Path(__file__).resolve().parent.parent / 'data' / 'CADET-Core_reference' / 'chromatography'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Solution times matching the benchmark settings
    solution_times = np.linspace(0.0, 1500.0, 1500 * 4 + 1)
    dps = 30

    print(f"Computing analytical references ({len(solution_times)} time points, {dps} dps)...")
    print()

    # LRM
    print("1/4: LRM (dynLin, 1 comp) ...")
    t0 = time.time()
    lrm_ref = get_LRM_analytical_reference(solution_times, dps=dps)
    print(f"     Computed in {time.time() - t0:.1f}s, max={np.max(lrm_ref):.6e}")
    save_analytical_reference(
        output_dir / 'ref_LRM_dynLin_1comp_benchmark1.h5',
        solution_times, lrm_ref
    )

    # LRMP
    print("2/4: LRMP (dynLin, 1 comp) ...")
    t0 = time.time()
    lrmp_ref = get_LRMP_analytical_reference(solution_times, dps=dps)
    print(f"     Computed in {time.time() - t0:.1f}s, max={np.max(lrmp_ref):.6e}")
    save_analytical_reference(
        output_dir / 'ref_LRMP_dynLin_1comp_benchmark1.h5',
        solution_times, lrmp_ref
    )

    # GRM (no surface diffusion)
    print("3/4: GRM (dynLin, 1 comp, no surface diffusion) ...")
    t0 = time.time()
    grm_ref = get_GRM_analytical_reference(solution_times, surface_diffusion=0.0, dps=dps)
    print(f"     Computed in {time.time() - t0:.1f}s, max={np.max(grm_ref):.6e}")
    save_analytical_reference(
        output_dir / 'ref_GRM_dynLin_1comp_benchmark1.h5',
        solution_times, grm_ref
    )

    # GRM with surface diffusion
    print("4/4: GRMsd (dynLin, 1 comp, surface diffusion=5e-11) ...")
    t0 = time.time()
    grmsd_ref = get_GRM_analytical_reference(solution_times, surface_diffusion=5e-11, dps=dps)
    print(f"     Computed in {time.time() - t0:.1f}s, max={np.max(grmsd_ref):.6e}")
    save_analytical_reference(
        output_dir / 'ref_GRMsd_dynLin_1comp_benchmark1.h5',
        solution_times, grmsd_ref
    )

    print()
    print("Done. Reference files saved to:", output_dir)


if __name__ == '__main__':
    main()
