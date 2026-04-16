#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate analytical reference solutions for 1D linear chromatography models.

This script computes the analytical outlet concentration for the LRM, LRMP,
and GRM benchmark settings using Laplace-domain transfer functions with
numerical inverse Laplace transform (de Hoog algorithm via mpmath).

The results are stored as HDF5 files in data/CADET-Verification_reference/chromatography/
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
    LRM_BENCHMARK_PARAMS,
    LRMP_BENCHMARK_PARAMS,
    GRM_BENCHMARK_PARAMS,
)


def save_analytical_reference(filepath, solution_times, outlet_concentration,
                              model_type, params):
    """Save analytical reference solution in CADET-compatible HDF5 format.

    Includes an ``input`` group with model parameters following the
    CADET-Core input file convention (only the parameters used by the
    analytical solver).
    """
    with h5py.File(filepath, 'w') as f:
        # Output group
        f.create_dataset(
            '/output/solution/unit_001/SOLUTION_OUTLET',
            data=outlet_concentration
        )
        f.create_dataset(
            '/output/solution/SOLUTION_TIMES',
            data=solution_times
        )

        # Input group – model parameters in CADET-Core format
        u1 = f.create_group('/input/model/unit_001')
        u1.create_dataset('COL_DISPERSION', data=params['col_dispersion'])
        u1.create_dataset('COL_LENGTH', data=params['col_length'])
        u1.create_dataset('VELOCITY', data=params['velocity'])
        u1.create_dataset('NCOMP', data=1)
        u1.create_dataset('ADSORPTION_MODEL', data='LINEAR')

        ads = u1.create_group('adsorption')
        ads.create_dataset('IS_KINETIC', data=1)
        ads.create_dataset('LIN_KA', data=[params['ka']])
        ads.create_dataset('LIN_KD', data=[params['kd']])

        if model_type == 'LRM':
            u1.create_dataset('UNIT_TYPE',
                              data='LUMPED_RATE_MODEL_WITHOUT_PORES')
            u1.create_dataset('TOTAL_POROSITY', data=params['total_porosity'])
        elif model_type == 'LRMP':
            u1.create_dataset('UNIT_TYPE',
                              data='LUMPED_RATE_MODEL_WITH_PORES')
            u1.create_dataset('COL_POROSITY', data=params['col_porosity'])
            u1.create_dataset('PAR_POROSITY', data=params['par_porosity'])
            u1.create_dataset('FILM_DIFFUSION', data=[params['film_diffusion']])
            u1.create_dataset('PAR_RADIUS', data=params['par_radius'])
        elif model_type == 'GRM':
            u1.create_dataset('UNIT_TYPE',
                              data='GENERAL_RATE_MODEL')
            u1.create_dataset('COL_POROSITY', data=params['col_porosity'])
            u1.create_dataset('PAR_POROSITY', data=params['par_porosity'])
            u1.create_dataset('FILM_DIFFUSION', data=params['film_diffusion'])
            u1.create_dataset('PAR_RADIUS', data=params['par_radius'])
            u1.create_dataset('PAR_DIFFUSION', data=params['pore_diffusion'])
            u1.create_dataset('PAR_SURFDIFFUSION',
                              data=params.get('surface_diffusion', 0.0))

        # Inlet definition
        u0 = f.create_group('/input/model/unit_000')
        u0.create_dataset('UNIT_TYPE', data='INLET')
        u0.create_dataset('NCOMP', data=1)
        u0.create_dataset('INLET_TYPE', data='PIECEWISE_CUBIC_POLY')
        sec0 = u0.create_group('sec_000')
        sec0.create_dataset('CONST_COEFF', data=[params.get('c_in', 1.0)])
        sec1 = u0.create_group('sec_001')
        sec1.create_dataset('CONST_COEFF', data=[0.0])

        # Section times
        solver = f.create_group('/input/solver')
        solver.create_dataset('USER_SOLUTION_TIMES', data=solution_times)
        sections = solver.create_group('sections')
        sections.create_dataset('NSEC', data=2)
        sections.create_dataset(
            'SECTION_TIMES',
            data=[0.0, params['t_inj'], float(solution_times[-1])]
        )

    print(f"  Saved to {filepath}")


def main():
    output_dir = Path(__file__).resolve().parent.parent / 'data' / 'CADET-Verification_reference' / 'chromatography'
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
        solution_times, lrm_ref, 'LRM', LRM_BENCHMARK_PARAMS
    )

    # LRMP
    print("2/4: LRMP (dynLin, 1 comp) ...")
    t0 = time.time()
    lrmp_ref = get_LRMP_analytical_reference(solution_times, dps=dps)
    print(f"     Computed in {time.time() - t0:.1f}s, max={np.max(lrmp_ref):.6e}")
    save_analytical_reference(
        output_dir / 'ref_LRMP_dynLin_1comp_benchmark1.h5',
        solution_times, lrmp_ref, 'LRMP', LRMP_BENCHMARK_PARAMS
    )

    # GRM (no surface diffusion)
    print("3/4: GRM (dynLin, 1 comp, no surface diffusion) ...")
    t0 = time.time()
    grm_ref = get_GRM_analytical_reference(solution_times, surface_diffusion=0.0, dps=dps)
    print(f"     Computed in {time.time() - t0:.1f}s, max={np.max(grm_ref):.6e}")
    save_analytical_reference(
        output_dir / 'ref_GRM_dynLin_1comp_benchmark1.h5',
        solution_times, grm_ref, 'GRM', GRM_BENCHMARK_PARAMS
    )

    # GRM with surface diffusion
    print("4/4: GRMsd (dynLin, 1 comp, surface diffusion=5e-11) ...")
    t0 = time.time()
    grmsd_params = dict(GRM_BENCHMARK_PARAMS, surface_diffusion=5e-11)
    grmsd_ref = get_GRM_analytical_reference(solution_times, surface_diffusion=5e-11, dps=dps)
    print(f"     Computed in {time.time() - t0:.1f}s, max={np.max(grmsd_ref):.6e}")
    save_analytical_reference(
        output_dir / 'ref_GRMsd_dynLin_1comp_benchmark1.h5',
        solution_times, grmsd_ref, 'GRM', grmsd_params
    )

    print()
    print("Done. Reference files saved to:", output_dir)


if __name__ == '__main__':
    main()
