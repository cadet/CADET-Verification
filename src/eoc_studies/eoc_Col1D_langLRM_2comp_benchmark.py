import sys
import os
import numpy as np
import pandas as pd
from cadet import Cadet
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from setting_Col1D_langLRM_2comp_benchmark1 import get_model

# ── helpers ────────────────────────────────────────────────────────────────────
def load_from_h5(spatial_method, refinement, output_path):
    sim = Cadet()
    sim.filename = f"{output_path}/temp_method{spatial_method}_ref{refinement}.h5"
    sim.load_from_file()
    outlet = sim.root.output.solution.unit_001.solution_outlet
    times  = sim.root.output.solution.solution_times
    return times, outlet

def L_inf_error(t_sol, solution, t_ref, reference):
    interp_func = interp1d(t_ref, reference, axis=0)
    ref_interp  = interp_func(t_sol)
    diff = solution - ref_interp
    return np.max(np.abs(diff))

def compute_eoc(errors, refinements, spatial_method):
    eoc_values = [None]
    for i in range(1, len(errors)):
        n_prev = 8 * refinements[i-1] * (spatial_method + 1)
        n_curr = 8 * refinements[i]   * (spatial_method + 1)
        eoc = np.log(errors[i] / errors[i-1]) / np.log(n_prev / n_curr)
        eoc_values.append(eoc)
    return eoc_values

def build_eoc_table(spatial_method, refinements, t_ref, ref_solution, output_path):
    method_label = "FV" if spatial_method == 0 else f"DG P{spatial_method}"
    errors = []
    for n_ref in refinements:
        t_, solution = load_from_h5(spatial_method, n_ref, output_path)
        err = L_inf_error(t_, solution, t_ref, ref_solution)
        errors.append(err)

    eoc_values = compute_eoc(errors, refinements, spatial_method)

    return pd.DataFrame({
        "Method":      method_label,
        "Refinement":  refinements,
        "N elements":  [8 * r for r in refinements],
        "L-inf error": errors,
        "EOC":         [f"{e:.2f}" if e is not None else "-" for e in eoc_values]
    })

# ── parallel worker ────────────────────────────────────────────────────────────
def run_single_simulation(method_bulk, n_ref, output_path):
    """Runs one simulation. Each call is independent — safe to parallelize."""
    try:
        sim = Cadet()
        sim.root = get_model(spatial_method_bulk=method_bulk, refinement=n_ref)
        sim.filename = f"{output_path}/temp_method{method_bulk}_ref{n_ref}.h5"
        sim.save()
        sim.run_simulation()
        sim.load_from_file()
    except Exception as e:
        print(f"  FAILED: method={method_bulk}, ref={n_ref} -> {e}")
        raise

# ── main study ─────────────────────────────────────────────────────────────────
def eoc_display_testing(ref, output_path, n_jobs, small_test):
    os.makedirs(output_path, exist_ok=True)

    # ── 1. Reference simulation (must finish before others) ───────────────────
    print(f"Running reference simulation (method=5, ref={ref})...")
    sim = Cadet()
    sim.root = get_model(spatial_method_bulk=5, refinement=ref)
    sim.filename = f"{output_path}/temp_method5_ref{ref}.h5"
    sim.save()
    sim.run_simulation()
    sim.load_from_file()
    print("  reference done.")

    # ── 2. Build list of all (method, refinement) combinations ────────────────
    refinements    = [1, 2, 4, 8, 16] if small_test else [1, 2, 4, 8, 16, 32, 64, 128]
    spatial_methods = [(0, "FV"), (3, "DG P3")]

    all_runs = [
        (method_bulk, n_ref)
        for method_bulk, _ in spatial_methods
        for n_ref in refinements
    ]

    # ── 3. Run all simulations in parallel ────────────────────────────────────
    print(f"\nStarting {len(all_runs)} simulations with n_jobs={n_jobs}...")
    Parallel(n_jobs=n_jobs)(
        delayed(run_single_simulation)(method_bulk, n_ref, output_path)
        for method_bulk, n_ref in all_runs
    )
    print("All simulations done.")

    # ── 4. Load reference and build EOC tables ────────────────────────────────
    print("\nLoading reference solution...")
    t_ref, ref_solution = load_from_h5(5, ref, output_path)

    print("\nBuilding FV table...")
    table_FV = build_eoc_table(0, refinements, t_ref, ref_solution, output_path)
    print("\n=== FV EOC Table ===")
    print(table_FV.to_string(index=False))

    print("\nBuilding DG P3 table...")
    table_dg = build_eoc_table(3, refinements, t_ref, ref_solution, output_path)
    print("\n=== DG P3 EOC Table ===")
    print(table_dg.to_string(index=False))

    # ── 5. Save tables to disk ────────────────────────────────────────────────
    table_FV.to_csv(f"{output_path}/eoc_FV.csv",    index=False)
    table_dg.to_csv(f"{output_path}/eoc_DG_P3.csv", index=False)
    print(f"\nEOC tables saved to {output_path}")

# ── entry point ────────────────────────────────────────────────────────────────
def mainFunc(n_jobs, cadet_path, small_test, output_path):
    Cadet.cadet_path = cadet_path
    ref_refinement = 32 if small_test else 64
    eoc_display_testing(ref_refinement, output_path, n_jobs, small_test)

# mainFunc(n_jobs=-1,
#          cadet_path = r"(...)",
#          small_test = True,
#          output_path = "" # todo
#          )