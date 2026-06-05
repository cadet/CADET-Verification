"""

This script trains GPR surrogates on more or less sparse (and polluted) Langmuir data,
runs a CADET simulation with GPR binding, and compares to a mechanistic simulation.

"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from cadet import Cadet

import src.benchmark_models.setting_Col1D_langLRM_2comp_benchmark1 as langmuir_2Comp_setting
import src.benchmark_models.settings_data_driven_bnd as langmuir_1Comp_setting
from src.training_gpr import train_gpr_for_cadet
from src.training_ann import train_ann_for_cadet


def multi_component_langmuir_equilibrium(
    cp: np.ndarray,
    keq: np.ndarray,
    qmax: np.ndarray,
) -> np.ndarray:
    cp = np.asarray(cp, dtype=float)
    keq = np.asarray(keq, dtype=float)
    qmax = np.asarray(qmax, dtype=float)

    if cp.ndim == 1:
        # Single-component Langmuir
        denominator = 1.0 + keq * cp
    else:
        # Multi-component Langmuir
        denominator = 1.0 + np.sum(keq * cp, axis=1, keepdims=True)

    return qmax * keq * cp / denominator


def get_langmuir_parameters(model_setting:Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    model = model_setting(spatial_method_bulk=0)
    binding_model = model.input.model.unit_001.particle_type_000.adsorption_model

    if binding_model != "MULTI_COMPONENT_LANGMUIR":
        raise ValueError(f"Binding model is {binding_model}, expected MULTI_COMPONENT_LANGMUIR")
    
    ka = model.input.model.unit_001.particle_type_000.adsorption.mcl_ka
    kd = model.input.model.unit_001.particle_type_000.adsorption.mcl_kd
    qmax = model.input.model.unit_001.particle_type_000.adsorption.mcl_qmax
    
    return np.array(ka), np.array(kd), np.array(qmax)


def run_hybrid_sim_analysis(
        binding_model: str["GPR", "ANN"],
        cadet_path: str, output_dir: str,
        c_sample: np.ndarray, cp: np.ndarray, cs: np.ndarray, kernel: Optional[str],
        get_model: callable,
        add_noise: bool = True,       # only for GPR
        hidden_nodes: int = 75,       # only for ANN
        epochs: int = 2000,           # only for ANN
        patience: int = 500,          # only for ANN
        ) -> tuple[Cadet, Cadet]:

    # Step 1: plot isotherm training data
    if cp.ndim == 2:

        nComp = cp.shape[1]

        # Plot the isotherm cs over cp for the first component.
        # That is, x and z axis are the cp of the two components and the y axis is the cs of the first component
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(cp[:, 0].reshape(len(c_sample), len(c_sample)),
                        cp[:, 1].reshape(len(c_sample), len(c_sample)),
                        cs[:, 0].reshape(len(c_sample), len(c_sample)),
                        cmap='viridis')
        ax.set_xlabel("$c^p_1$")
        ax.set_ylabel("$c^p_2$")
        ax.set_zlabel("$c^s_1$")

        # Plot the isotherm cs over cp for the second component.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(cp[:, 0].reshape(len(c_sample), len(c_sample)),
                        cp[:, 1].reshape(len(c_sample), len(c_sample)),
                        cs[:, 1].reshape(len(c_sample), len(c_sample)),
                        cmap='viridis')
        ax.set_xlabel("$c^p_1$")
        ax.set_ylabel("$c^p_2$")
        ax.set_zlabel("$c^s_2$")

    elif cp.ndim == 1:
        
        nComp = 1

        plt.figure()
        plt.plot(cp, cs)
        plt.xlabel("$c^p$")
        plt.ylabel("$c^s$")
        plt.title("Isotherm")

    # Step 2: mechanistic simulation
    simMechanistic = Cadet()
    simMechanistic.install_path = cadet_path
    if nComp == 1:
        simMechanistic.filename = str(output_dir + "/Col1D_GRM_lang_1comp_benchmark1.h5")
    elif nComp == 2:
        simMechanistic.filename = str(output_dir + "/Col1D_LRM_lang_2comp_benchmark1.h5")
    simMechanistic.root = get_model(spatial_method_bulk=0, write_solution_bulk=1, write_solution_solid=1)
    simMechanistic.save()
    return_data = simMechanistic.run_simulation()
    if return_data.return_code != 0:
        raise RuntimeError(f"CADET simulation failed with return code {return_data.return_code}, error message {return_data.error_message} and log {return_data.log}")
    else:
        simMechanistic.save()

    # plot outlet of both components over time
    time = simMechanistic.root.output.solution.solution_times
    if nComp == 1:
        outlet_cp = simMechanistic.root.output.solution.unit_001.solution_outlet[:]
        plt.figure()
        plt.plot(time, outlet_cp, label="Mechanistic", color='green')
        plt.xlabel("Time")
        plt.ylabel("Outlet Concentration")
        plt.legend()
    elif nComp == 2:
        outlet_cp_1 = simMechanistic.root.output.solution.unit_001.solution_outlet[:, 0]
        outlet_cp_2 = simMechanistic.root.output.solution.unit_001.solution_outlet[:, 1]
        plt.figure()
        plt.plot(time, outlet_cp_1, label="Mech comp 1", color='green')
        plt.plot(time, outlet_cp_2, label="Mech comp 2", color='red')
        plt.xlabel("Time")
        plt.ylabel("Outlet Concentration")
        plt.legend()

    # Step 3: train data-driven surrogate
    if binding_model == "GPR":
        training_results = train_gpr_for_cadet(
            cp, cs,
            kernel=kernel, optimization_restarts=1, add_noise=add_noise,
        )
    elif binding_model == "ANN":
        training_results = train_ann_for_cadet(cp, cs, hidden_nodes=hidden_nodes, epochs=epochs, patience=patience)
    else:
        raise ValueError(f"Invalid data-driven binding model {binding_model}, expected 'GPR' or 'ANN'")

    # print(training_results)

    # Step 4: run hybrid simulation
    simHybrid = Cadet()
    simHybrid.install_path = cadet_path
    if nComp == 1:
        simHybrid.filename = str(output_dir + "/Col1D_GRM_langGPR_1comp_benchmark1.h5") if binding_model == "GPR" else str(output_dir + "/Col1D_GRM_langANN_1comp_benchmark1.h5")
    elif nComp == 2:
        simHybrid.filename = str(output_dir + "/Col1D_LRM_langGPR_2comp_benchmark1.h5") if binding_model == "GPR" else str(output_dir + "/Col1D_LRM_langANN_2comp_benchmark1.h5")
    
    simHybrid.root = get_model(spatial_method_bulk=0, write_solution_bulk=1, write_solution_solid=1)
    simHybrid.root.input.model.unit_001.particle_type_000.adsorption_model = "GAUSSIAN_PROCESS_REGRESSION" if binding_model == "GPR" else "NEURAL_NETWORK"
    if binding_model == "GPR":
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.CP_NDIM = nComp
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.CS_NDIM = nComp
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.CP_VALS = cp
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.CS_VALS = cs
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.GPR_KKIN = [1000.0] * nComp
    elif binding_model == "ANN":
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.NN_KKIN = [1000.0] * nComp
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.NLAYERS = 2
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.NNODES = hidden_nodes
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.POROSITY_FACTOR = 1.0 #/ (1.0-epsilon_p)

    for key in training_results:
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption[key] = training_results[key]
    simHybrid.save()
    return_data = simHybrid.run_simulation()
    if return_data.return_code != 0:
        raise RuntimeError(f"CADET simulation failed with return code {return_data.return_code}, error message {return_data.error_message} and log {return_data.log}")
    else:
        simHybrid.save()

    # Step 5: plot hybrid simulation results and compare to mechanistic simulation
    if nComp == 1:
        outlet_cp_hybrid = simHybrid.root.output.solution.unit_001.solution_outlet[:]
        plt.plot(time, outlet_cp_hybrid, label="Hybrid", linestyle='dashed', color='blue')
        plt.xlabel("Time")
        plt.ylabel("Outlet Concentration")
        plt.legend()
        plt.savefig(output_dir + "/Col1D_GRM_langGPR_1comp.png")

        # plot the difference and print max abs error between mechanistic and hybrid simulation for both components
        plt.figure()
        plt.plot(time, outlet_cp - outlet_cp_hybrid, label="Error comp 1")
        plt.xlabel("Time")
        plt.ylabel("Difference in Outlet Concentration")
        plt.legend()

        abs_max_diff_cp = np.max(np.abs(outlet_cp - outlet_cp_hybrid))
        plt.text(0.05, 0.95, f"Max abs error:\nComp 1: {abs_max_diff_cp:.3e}",
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # relative error of that max abs deviation, i.e. max abs error divided by max outlet concentration in mechanistic simulation
        abs_max_relative_error_cp = abs_max_diff_cp / np.max(outlet_cp)
        plt.text(0.05, 0.75, f"Rel. error of max. deviation:\nComp 1: {abs_max_relative_error_cp:.3e}",
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(output_dir + "/error_Col1D_GRM_langGPR_1comp.png")


    elif nComp == 2:
        outlet_cp_1_hybrid = simHybrid.root.output.solution.unit_001.solution_outlet[:, 0]
        outlet_cp_2_hybrid = simHybrid.root.output.solution.unit_001.solution_outlet[:, 1]
        # plt.figure()
        plt.plot(time, outlet_cp_1_hybrid, label="Hybrid comp 1", linestyle='dashed', color='blue')
        plt.plot(time, outlet_cp_2_hybrid, label="Hybrid comp 2", linestyle='dashed', color='orange')
        plt.xlabel("Time")
        plt.ylabel("Outlet Concentration")
        plt.legend()
        plt.savefig(output_dir + "/Col1D_LRM_langGPR_2comp.png")

        # plot the difference and print max abs error between mechanistic and hybrid simulation for both components
        plt.figure()
        plt.plot(time, outlet_cp_1 - outlet_cp_1_hybrid, label="Error comp 1")
        plt.plot(time, outlet_cp_2 - outlet_cp_2_hybrid, label="Error comp 2")
        plt.xlabel("Time")
        plt.ylabel("Difference in Outlet Concentration")
        plt.legend()

        abs_max_diff_cp_1 = np.max(np.abs(outlet_cp_1 - outlet_cp_1_hybrid))
        abs_max_diff_cp_2 = np.max(np.abs(outlet_cp_2 - outlet_cp_2_hybrid))
        plt.text(0.05, 0.95, f"Max abs error:\nComp 1: {abs_max_diff_cp_1:.3e}\nComp 2: {abs_max_diff_cp_2:.3e}",
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # relative error of that max abs deviation, i.e. max abs error divided by max outlet concentration in mechanistic simulation
        abs_max_relative_error_cp_1 = abs_max_diff_cp_1 / np.max(outlet_cp_1)
        abs_max_relative_error_cp_2 = abs_max_diff_cp_2 / np.max(outlet_cp_2)
        plt.text(0.05, 0.75, f"Rel. error of max. deviation:\nComp 1: {abs_max_relative_error_cp_1:.3e}\nComp 2: {abs_max_relative_error_cp_2:.3e}",
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(output_dir + "/error_Col1D_LRM_langGPR_2comp.png")

    return simMechanistic, simHybrid


#######################################################################################
# Langmuir 2Comp binding isotherm GPR analysis
#######################################################################################

def binding_train_GPR_langmuir2Comp(cadet_path: str, output_dir: str):

    refinement = 4
    NTRAIN = 10 # 50 with refinement 1 gets us xe-2 max. abs. error for both components
    cpMax = 10.0 # seen as max c^l for unrefined model

    # sample the cp space and compute the corresponding equilibrium loading
    Ka, Kd, Qmax = get_langmuir_parameters(langmuir_2Comp_setting.get_model)
    print(f"Ka: {Ka}, Kd: {Kd}, Qmax: {Qmax}")
    c_sample = np.linspace(0, cpMax, NTRAIN + 1)
    cp = np.array([[c1_i, c2_j] for c1_i in c_sample for c2_j in c_sample])
    cs = multi_component_langmuir_equilibrium(cp=cp, keq=Ka/Kd, qmax=Qmax)

    simMechanistic1, simHybrid1 = run_hybrid_sim_analysis(
        "GPR",
        cadet_path, output_dir,
        c_sample, cp, cs, kernel="MLP",
        get_model=partial(langmuir_2Comp_setting.get_model, refinement=refinement),
        )
    

#######################################################################################
# 1comp Langmuir binding isotherm GPR analysis
#######################################################################################

def binding_train_GPR_langmuir1Comp(cadet_path: str, output_dir: str):

    NTRAIN = 10

    Ka, Kd, Qmax = 2.0, 1.0, 20.0
    cpMax = 10.0

    cp = np.linspace(0.0, cpMax, NTRAIN + 1)
    cs = multi_component_langmuir_equilibrium(cp=cp, keq=Ka/Kd, qmax=Qmax)

    simMechanistic1, simHybrid1 = run_hybrid_sim_analysis(
        "GPR",
        cadet_path, output_dir,
        cp, cp, cs, kernel="MLP",
        get_model=partial(
            langmuir_1Comp_setting.get_model,
                file_name="testitest.h5",
                mode="MCL",
                loading=np.linspace(0.0, 50.0, 50 + 1),
                column_key="favorable_lysozyme",
                keq=Ka/Kd,
                qm=Qmax * (1.0 - 0.75), # mechanistic model shenanigans
                add_noise=True, # for deterministic data sets
            )
        )

#######################################################################################
# 1comp Langmuir binding isotherm ANN analysis
#######################################################################################

def binding_train_ANN_langmuir1Comp(cadet_path: str, output_dir: str):

    NTRAIN = 10

    Ka, Kd, Qmax = 2.0, 1.0, 20.0
    cpMax = 10.0

    cp = np.linspace(0.0, cpMax, NTRAIN + 1)
    cs = multi_component_langmuir_equilibrium(cp=cp, keq=Ka/Kd, qmax=Qmax)

    simMechanistic1, simHybrid1 = run_hybrid_sim_analysis(
        "ANN",
        cadet_path, output_dir,
        cp, cp, cs, kernel=None,
        get_model=partial(
            langmuir_1Comp_setting.get_model,
            file_name="testitest.h5",
            mode="MCL",
            loading=np.linspace(0.0, 50.0, 50 + 1),
            column_key="favorable_lysozyme",
            keq=Ka/Kd,
            qm=Qmax * (1.0 - 0.75),
        ),
        hidden_nodes=8,
        epochs=2000,
        patience=500
    )
