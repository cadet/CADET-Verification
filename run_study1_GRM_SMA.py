#!/usr/bin/env python3
"""
Runner for Study 1 GRM SMA: CGL vs LGL.

Usage:
    python run_study1_GRM_SMA.py [--node-type CGL|LGL] [--polydeg N] [--n-repeats N] [--small]

If --node-type and --polydeg are given, runs only that single config.
Otherwise, runs all 10 configs (CGL/LGL x P1-P5).
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.study1_cgl_vs_lgl_GRM_SMA import study1_GRM_SMA_tests


def main():
    parser = argparse.ArgumentParser(description='Study 1 GRM SMA: CGL vs LGL')
    parser.add_argument('--node-type', type=str, default=None,
                        help='Node type: CGL or LGL (default: both)')
    parser.add_argument('--polydeg', type=int, default=None,
                        help='Polynomial degree (default: 1-5)')
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='Number of reruns for min wall time (default: 3)')
    parser.add_argument('--small', action='store_true',
                        help='Use reduced discretization levels')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--output-path', type=str,
                        default='output/test_cadet-core/radialDG',
                        help='Output directory')
    parser.add_argument('--cadet-path', type=str,
                        default=os.path.expanduser('~/local-fix'),
                        help='Path to CADET install')
    args = parser.parse_args()

    node_types = [args.node_type] if args.node_type else None
    polydegs = [args.polydeg] if args.polydeg else None

    print(f"=== Study 1 GRM SMA: {args.node_type or 'CGL+LGL'} "
          f"P{args.polydeg or '1-5'} ===")
    print(f"  n_repeats: {args.n_repeats}")
    print(f"  n_jobs: {args.n_jobs}")
    print(f"  output: {args.output_path}")
    print(f"  cadet: {args.cadet_path}")
    print()

    study1_GRM_SMA_tests(
        n_jobs=args.n_jobs,
        small_test=args.small,
        output_path=args.output_path,
        cadet_path=args.cadet_path,
        polydegs=polydegs,
        node_types=node_types,
        n_repeats=args.n_repeats,
    )

    print(f"\n=== Done: {args.node_type or 'CGL+LGL'} "
          f"P{args.polydeg or '1-5'} ===")


if __name__ == '__main__':
    main()
