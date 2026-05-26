import sys
sys.path.append(r"C:\Users\Matthias\software\CADET-Verification\src\benchmark_models")
import numpy as np
import pandas as pd
from cadet import Cadet
from scipy.interpolate import interp1d

# ── helpers ────────────────────────────────────────────────────────────────────
def load_from_h5(spatial_method, refinement):
    sim = Cadet()
    sim.filename = f"temp_method{spatial_method}_ref{refinement}.h5"
    sim.load_from_file()
    outlet = sim.root.output.solution.unit_001.solution_outlet
    times  = sim.root.output.solution.solution_times
    return times, outlet

def L_inf_error(t_sol, solution, t_ref, reference):
    interp_func = interp1d(t_ref, reference, axis=0)
    ref_interp  = interp_func(t_sol)
    diff = solution - ref_interp
    return np.max(np.abs(diff))
    #return np.max(np.abs(solution - reference))

def compute_eoc(errors, refinements, spatial_method):
    eoc_values = [None]
    for i in range(1, len(errors)):
        n_prev = 8 * refinements[i-1] * (spatial_method + 1)
        n_curr = 8 * refinements[i]   * (spatial_method + 1)
        eoc = np.log(errors[i] / errors[i-1]) / np.log(n_prev / n_curr)
        eoc_values.append(eoc)
    return eoc_values

def build_eoc_table(spatial_method, refinements, t_ref, ref_solution):
    method_label = "FV" if spatial_method == 0 else f"DG P{spatial_method}"
    errors = []
    for ref in refinements:
        t_, solution = load_from_h5(spatial_method, ref)
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
def quick_dimension_check():
    t_ref, ref_solution = load_from_h5(5, 32)
    t_1, sol_1 = load_from_h5(0, 1)

    print(f"t_ref: start={t_ref[0]}, end={t_ref[-1]}, n={len(t_ref)}")
    print(f"t_sol: start={t_1[0]},   end={t_1[-1]},   n={len(t_1)}")

def another_check():
    t_ref, ref_solution = load_from_h5(5, 32)
    t_1, sol_1 = load_from_h5(0, 1)

    print("t_ref first 5:", t_ref[:5])
    print("t_sol first 5:", t_1[:5])

# self comparison test - MUST give 0
def test():
    t1, sol1 = load_from_h5(0, 16)
    t2, sol2 = load_from_h5(0, 16)

    print("shapes:", sol1.shape, sol2.shape)
    print("times identical:", np.allclose(t1, t2))
    print("solutions identical:", np.allclose(sol1, sol2))
    print("L_inf:", L_inf_error(t1, sol1, t2, sol2))

def eoc_display_testing():
    # for ref in [32]:
    #     print(f"Running reference ref={ref}...")
    #     sim = Cadet()
    #     sim.root = get_model(spatial_method_bulk=5, refinement=ref)
    #     sim.filename = f"temp_method5_ref{ref}.h5" 
    #     sim.save()
    #     sim.run_simulation()
    #     print(f"  simulation done, saving...")
    #     sim.load_from_file()
    #     print(f"  loaded successfully")
    refinements   = [1, 2, 4, 8, 16, 32]
    ref_refinement = 32

    print("Loading reference solution...")
    t_ref, ref_solution = load_from_h5(5, ref_refinement)

    print("\nBuilding FV table...")
    table_FV = build_eoc_table(0, refinements, t_ref, ref_solution)
    print("\n=== FV EOC Table ===")
    print(table_FV.to_string(index=False))

    print("\nBuilding DG P3 table...")
    table_dg = build_eoc_table(3, refinements, t_ref, ref_solution)
    print("\n=== DG P3 EOC Table ===")
    print(table_dg.to_string(index=False))
# ── main ───────────────────────────────────────────────────────────────────────

def mainFunc(
        n_jobs,
        cadet_path,
        small_test,
        output_path):
    
    Cadet.cadet_path = cadet_path

    # quick_dimension_check()
    # another_check()
    # test()
    eoc_display_testing()
    
    import matplotlib.pyplot as plt
    
    t_ref, ref_solution = load_from_h5(5, 32)  # P5 ref=64 as reference
    t_1, sol_1 = load_from_h5(0, 16)           # FV ref=16 as test
    
    print(np.max(np.abs(ref_solution - sol_1)))
    error = np.abs(ref_solution - sol_1)
    
    idx = np.unravel_index(np.argmax(error), error.shape)
    
    print("Maximum error:", error[idx])
    print("Zeitindex:", idx[0])
    print("Komponente:", idx[1])
    print("Zeitpunkt:", t_1[idx[0]])
    
    plt.figure()
    plt.plot(t_ref, ref_solution[:, 0], label="ref (P5 ref=32) comp1")
    plt.plot(t_1,   sol_1[:, 0],        label="test (FV ref=16) comp1", linestyle="--")
    plt.plot(t_ref, ref_solution[:, 1], label="ref (P5 ref=32) comp2")
    plt.plot(t_1,   sol_1[:, 1],        label="test (FV ref=16) comp2", linestyle="--")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("concentration")
    plt.title("Reference vs Test solution")
    plt.show()

mainFunc(n_jobs=-1,
         cadet_path = r"C:\Users\jmbr\Desktop\CADET_compiled\master5_fixParCoords_783967a\aRELEASE\bin\cadet-cli.exe",
         small_test = True,
         output_path = "" # todo
         )