"""

This script trains GPR surrogates on more or less sparse (and polluted) Langmuir data,
runs a CADET simulation with GPR binding, and compares to a mechanistic simulation.

"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cadet import Cadet

import src.benchmark_models.setting_Col1D_langLRM_2comp_benchmark1 as langmuir_2Comp_setting
import src.benchmark_models.settings_data_driven_bnd as langmuir_1Comp_setting
from src.surrogate_models import ANNSurrogate, BaseSurrogate, GPRSurrogate
from src.training_gpr import train_gpr_for_cadet
from src.training_spline import train_spline_for_cadet


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
        get_model: callable,
        training_results: Optional[Dict] = None,            # dictionary with training_results in cadet format
        kernel: Optional[str] = None,                       # only for GPR
        add_noise: bool = True,                             # only for GPR
        cp: np.ndarray=None, cs: np.ndarray=None,           # only for GPR
        normalization_factor: Optional[List[float]] = None, # only for ANN
        hidden_nodes: int = 10,                             # only for ANN
        n_layers: int = 2,                                  # only for ANN
        epochs: int = 500,                                  # only for ANN
        patience: int = 50,                                 # only for ANN
        max_retries: int = 2,                               # only for ANN
        acceptance_threshold: float = 10.0,                 # only for ANN
        training_strategy: Literal[                         # only for ANN
            "random_split",
            "k_fold",
            "leave_one_out",
            "none",
        ] = "random_split",
        validation_split: float = 0.2,                      # only for ANN
        plot_training_curves: Optional[str] = None,         # only for ANN
        ) -> tuple[Cadet, Cadet]:

    # Step 1: mechanistic simulation
    simMechanistic = Cadet()
    simMechanistic.install_path = cadet_path
    simMechanistic.root = get_model(spatial_method_bulk=0, write_solution_bulk=1, write_solution_solid=1)
    nComp = simMechanistic.root.input.model.unit_001.ncomp

    if nComp == 1:
        simMechanistic.filename = str(output_dir + "/Col1D_GRM_lang_1comp_benchmark1.h5")
    elif nComp == 2:
        simMechanistic.filename = str(output_dir + "/Col1D_LRM_lang_2comp_benchmark1.h5")
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
        fig_outlet = plt.figure()
        plt.plot(time, outlet_cp, label="Mechanistic", color='green')
        plt.xlabel("Time")
        plt.ylabel("Outlet Concentration")
        plt.legend()
    elif nComp == 2:
        outlet_cp_1 = simMechanistic.root.output.solution.unit_001.solution_outlet[:, 0]
        outlet_cp_2 = simMechanistic.root.output.solution.unit_001.solution_outlet[:, 1]
        fig_outlet = plt.figure()
        plt.plot(time, outlet_cp_1, label="Mech comp 1", color='green')
        plt.plot(time, outlet_cp_2, label="Mech comp 2", color='red')
        plt.xlabel("Time")
        plt.ylabel("Outlet Concentration")
        plt.legend()

    # Step 2: train data-driven surrogate
    if binding_model == "GPR" and training_results is None:
        training_results = train_gpr_for_cadet(
            cp, cs,
            kernel=kernel, optimization_restarts=1, add_noise=add_noise
        )

    elif binding_model == "ANN" and training_results is None:
        from src.training_ann import train_ann_for_cadet

        training_results = train_ann_for_cadet(
            cp, cs, hidden_nodes=hidden_nodes, n_layers=n_layers,
            normalization_factor=normalization_factor, epochs=epochs, patience=patience,
            max_retries=max_retries, acceptance_threshold=acceptance_threshold, training_strategy=training_strategy, validation_split=validation_split,
            plot_training_curves=plot_training_curves
            )

    # Step 3: run hybrid simulation
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
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.NLAYERS = n_layers
        simHybrid.root.input.model.unit_001.particle_type_000.adsorption.NNODES = hidden_nodes
        for component in range(nComp):
            simHybrid.root.input.model.unit_001.particle_type_000.adsorption[f"bound_state_{component:03d}"].POROSITY_FACTOR = 1.0 #/ (1.0-epsilon_p)

    for key in training_results:
        if binding_model == "ANN":
            simHybrid.root.input.model.unit_001.particle_type_000.adsorption[key].update(training_results[key])
        elif binding_model == "GPR":
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
        plt.figure(fig_outlet.number)
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
        
        plt.savefig(output_dir + "/error_Col1D_GRM_langGPR_1comp.png" if binding_model == "GPR" else output_dir + "/error_Col1D_GRM_langANN_1comp.png")

    elif nComp == 2:
        outlet_cp_1_hybrid = simHybrid.root.output.solution.unit_001.solution_outlet[:, 0]
        outlet_cp_2_hybrid = simHybrid.root.output.solution.unit_001.solution_outlet[:, 1]
        plt.figure(fig_outlet.number)
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
        
        plt.savefig(output_dir + "/error_Col1D_LRM_langGPR_2comp.png" if binding_model == "GPR" else output_dir + "/error_Col1D_LRM_langANN_2comp.png")

    return simMechanistic, simHybrid
    
def _error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the standard surrogate error metrics."""

    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAX_ERROR": float(np.max(np.abs(y_true - y_pred))),
    }


def _reshape_structured_surface(cp: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a 2D structured grid for surface plotting."""

    cp = np.asarray(cp, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)

    if cp.ndim != 2 or cp.shape[1] != 2:
        raise ValueError("Surface plotting requires a 2D input grid.")
    if cp.shape[0] != values.shape[0]:
        raise ValueError("Input coordinates and values must contain the same number of samples.")

    axis_0 = np.unique(cp[:, 0])
    axis_1 = np.unique(cp[:, 1])
    if axis_0.size * axis_1.size != cp.shape[0]:
        raise ValueError("Surface plotting requires a full Cartesian product grid.")

    grid = np.empty((axis_0.size, axis_1.size), dtype=float)
    index_0 = {float(value): idx for idx, value in enumerate(axis_0)}
    index_1 = {float(value): idx for idx, value in enumerate(axis_1)}

    for sample, value in zip(cp, values):
        i = index_0[float(sample[0])]
        j = index_1[float(sample[1])]
        grid[i, j] = value

    X, Y = np.meshgrid(axis_0, axis_1, indexing="ij")
    return X, Y, grid


def plot_surface_comparison(cp, cs_true, cs_pred, output_dir, isotherm_model="langmuir", surrogate_model="surrogate", stateIdx=None):
    
    cp = np.asarray(cp)
    cs_true = np.asarray(cs_true)
    cs_pred = np.asarray(cs_pred)

    if cp.ndim != 2 or cs_true.ndim != 1 or cs_pred.ndim != 1:
        raise ValueError(f"cp shape {cp.shape}, cs_true shape {cs_true.shape} and cs_pred shape {cs_pred.shape} are not compatible with surface plotting, expected cp.ndim == 2, cs_true.ndim == 1 and cs_pred.ndim == 1")
    if cs_true.shape[0] != len(cs_pred) or cs_true.shape[0] != cp.shape[0]:
        raise ValueError(f"cs_true shape {cs_true.shape} and cs_pred shape {cs_pred.shape} are not compatible, expected same number of samples, and compatible with cp shape {cp.shape}")
    if cp.shape[1] != 2:
        raise ValueError(f"cp shape {cp.shape} is not compatible with surface plotting, needs cp.shape[1] == 2")

    X, Y, Z_true = _reshape_structured_surface(cp, cs_true)
    _, _, Z_pred = _reshape_structured_surface(cp, cs_pred)
    Z_err = np.abs(Z_true - Z_pred)

    # -------------------------
    # TRUE model surface
    # -------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_true, cmap='viridis')
    ax.set_title(f"True Langmuir - state {stateIdx}")
    ax.set_xlabel("$c^p_1$")
    ax.set_ylabel("$c^p_2$")
    ax.set_zlabel("$c^s$")
    save_path = output_dir + f"/True_{isotherm_model}1Comp_isotherm_comp_{stateIdx}.png"
    plt.savefig(save_path)
    plt.close(fig)

    # -------------------------
    # ANN surface
    # -------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_pred, cmap='viridis')
    ax.set_title(f"{surrogate_model} prediction - state {stateIdx}")
    ax.set_xlabel("$c^p_1$")
    ax.set_ylabel("$c^p_2$")
    ax.set_zlabel("$c^s$")
    save_path = output_dir + f"/{surrogate_model}_{isotherm_model}1Comp_isotherm_comp_{stateIdx}.png"
    plt.savefig(save_path)
    plt.close(fig)

    # -------------------------
    # ERROR surface
    # -------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_err, cmap='inferno')
    ax.set_title(f"Absolute error - state {stateIdx}")
    ax.set_xlabel("$c^p_1$")
    ax.set_ylabel("$c^p_2$")
    ax.set_zlabel("|error|")
    save_path = output_dir + f"/{surrogate_model}_2Comp{isotherm_model}_error_comp_{stateIdx}.png"
    plt.savefig(save_path)
    plt.close(fig)

def isotherm_comparison_1D(cp, cs_true, cs_pred, output_dir, isotherm_model="langmuir", surrogate_model="surrogate"):

    metrics = _error_metrics(np.asarray(cs_true).reshape(-1), np.asarray(cs_pred).reshape(-1))

    plt.figure()
    plt.plot(cp, cs_true, label=f"True {isotherm_model}", color='green')
    plt.plot(cp, cs_pred, label=f"{surrogate_model} prediction", linestyle='dashed', color='blue')
    plt.xlabel("cp")
    plt.ylabel("cs")
    plt.title(f"Isotherm comparison")
    plt.text(0.25, 0.15, f"MSE: {metrics['MSE']:.3e}\nRMSE: {metrics['RMSE']:.3e}\nMAE: {metrics['MAE']:.3e}\nR2: {metrics['R2']:.3f}\nMax abs error: {metrics['MAX_ERROR']:.3e}",
            transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.savefig(output_dir + f"/{surrogate_model}_{isotherm_model}1Comp_isotherm_comparison.png")
    plt.legend()

    metrics = {
        "bound_state_000": {
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "MAX_ERROR": metrics["MAX_ERROR"],
        }
    }
    
    return metrics

def evaluate_surrogate_vs_isotherm(
        cp,
        cs_true,
        surrogate: BaseSurrogate | None = None,
        output_path: str = "",
        isotherm_model="langmuir",
        surrogate_model: Optional[str] = None,
        training_results: Optional[Dict[str, Any]] = None,
        **kwargs
        ):
    """Evaluate a surrogate against the exact isotherm and generate plots."""

    cp_array = np.asarray(cp, dtype=float)
    cs_true_array = np.asarray(cs_true, dtype=float)
    cs_true_was_1d = cs_true_array.ndim == 1
    if cs_true_was_1d:
        cs_true_array = cs_true_array.reshape(-1, 1)

    if cp_array.ndim == 1:
        n_inputs = 1
    elif cp_array.ndim == 2:
        n_inputs = cp_array.shape[1]
    else:
        raise ValueError(
            f"cp shape {cp_array.shape} is not compatible with evaluation, expected cp.ndim == 1 or cp.ndim == 2"
        )

    if surrogate is None:
        if surrogate_model == "ANN":
            surrogate = ANNSurrogate.from_training_results(
                training_results or {},
                hidden_nodes=kwargs.get("hidden_nodes"),
                n_layers=kwargs.get("n_layers"),
                normalization_factor=kwargs.get("normalization_factor"),
            )
        elif surrogate_model == "GPR":
            surrogate = GPRSurrogate.from_training_results(
                training_results or {},
                cp=cp_array,
                cs=cs_true_array,
                kernel_name=kwargs["kernel_name"],
            )
        elif surrogate_model == "Spline":
            if isinstance(training_results, dict) and "model" in training_results:
                surrogate = training_results["model"]
            else:
                surrogate = train_spline_for_cadet(cp_array, cs_true_array)["model"]
        else:
            raise ValueError("A surrogate object must be provided for evaluation.")

    cs_pred_array = np.asarray(surrogate.predict(cp_array), dtype=float)
    if cs_pred_array.ndim == 1:
        cs_pred_array = cs_pred_array.reshape(-1, 1)
    if cs_pred_array.shape != cs_true_array.shape:
        raise ValueError(
            f"Surrogate prediction shape {cs_pred_array.shape} does not match reference shape {cs_true_array.shape}."
        )

    metrics: Dict[str, Dict[str, float]] = {}
    surrogate_label = getattr(surrogate, "name", surrogate_model or "surrogate")

    for i in range(cs_true_array.shape[1]):
        y_true = cs_true_array[:, i]
        y_pred = cs_pred_array[:, i]
        metrics[f"bound_state_{i:03d}"] = _error_metrics(y_true, y_pred)

        if n_inputs == 1:
            cp_plot = cp_array.reshape(-1)
            isotherm_comparison_1D(
                cp=cp_plot,
                cs_true=y_true,
                cs_pred=y_pred,
                output_dir=output_path,
                surrogate_model=surrogate_label,
                isotherm_model=isotherm_model,
            )
        elif n_inputs == 2:
            plot_surface_comparison(
                cp=cp_array,
                cs_true=y_true,
                cs_pred=y_pred,
                output_dir=output_path,
                isotherm_model=isotherm_model,
                surrogate_model=surrogate_label,
                stateIdx=i,
            )

    cs_pred = cs_pred_array.reshape(-1) if cs_true_was_1d else cs_pred_array
    return cs_pred, metrics, [surrogate]


def _build_ann_surrogate(
    training_results: Dict[str, Any],
    *,
    hidden_nodes: int,
    n_layers: int,
    normalization_factor: Optional[List[float]],
) -> ANNSurrogate:
    """Rebuild an ANN surrogate for evaluation."""

    return ANNSurrogate.from_training_results(
        training_results,
        hidden_nodes=hidden_nodes,
        n_layers=n_layers,
        normalization_factor=normalization_factor,
    )


def _build_gpr_surrogate(
    training_results: Dict[str, Any],
    *,
    cp: np.ndarray,
    cs: np.ndarray,
    kernel_name: str,
) -> GPRSurrogate:
    """Rebuild a GPR surrogate for evaluation."""

    return GPRSurrogate.from_training_results(
        training_results,
        cp=cp,
        cs=cs,
        kernel_name=kernel_name,
    )

#######################################################################################
# 1comp Langmuir binding isotherm GPR analysis
#######################################################################################

def binding_train_GPR_langmuir1Comp(cadet_path: str, output_dir: str):

    NTRAIN = 10
    kernel = "MLP"
    optimization_restarts = 10
    add_noise = True # for mechanistic data sets, to avoid overfitting and numerical issues in GPR training, especially with small data sets and more complex kernels

    Ka, Kd, Qmax = 2.0, 1.0, 20.0
    cpMax = 10.0

    cp = np.linspace(0.0, cpMax, NTRAIN)
    cs = multi_component_langmuir_equilibrium(cp=cp, keq=Ka/Kd, qmax=Qmax)

    output_dir=str(output_dir) + f"/langmuir1comp/GPR"
    
    os.makedirs(output_dir, exist_ok=True)

    training_results = train_gpr_for_cadet(
                cp, cs,
                kernel=kernel,
                optimization_restarts=optimization_restarts,
                add_noise=add_noise,
                )

    surrogate = _build_gpr_surrogate(training_results, cp=cp, cs=cs, kernel_name=kernel)

    cs_pred, metrics, models = evaluate_surrogate_vs_isotherm(
        cp=cp,
        cs_true=cs,
        surrogate=surrogate,
        output_path=output_dir,
        isotherm_model="langmuir"
    )

    print("\n=== Isotherm approximation metrics ===")
    for k, v in metrics.items():
        print(k, v)

    simMechanistic1, simHybrid1 = run_hybrid_sim_analysis(
        "GPR",
        cadet_path, output_dir,
        kernel="MLP",
        get_model=partial(
            langmuir_1Comp_setting.get_model,
                file_name="testitest.h5",
                mode="MCL",
                loading=np.linspace(0.0, 50.0, 50 + 1),
                column_key="favorable_lysozyme",
                keq=Ka/Kd,
                qm=Qmax * (1.0 - 0.75), # mechanistic model shenanigans
                add_noise=True, # for deterministic data sets
            ),
            training_results=training_results,
            cp=cp, cs=cs
        )

#######################################################################################
# Langmuir 2Comp binding isotherm GPR analysis
#######################################################################################

def binding_train_GPR_langmuir2Comp(cadet_path: str, output_dir: str):

    refinement = 4
    NTRAIN = 10
    kernel = "MLP"
    optimization_restarts = 10
    add_noise = True # for mechanistic data sets, to avoid overfitting and numerical issues in GPR training, especially with small data sets and more complex kernels

    cpMax = 10.0 # seen as max c^l for unrefined model
    # sample the cp space and compute the corresponding equilibrium loading
    Ka, Kd, Qmax = get_langmuir_parameters(langmuir_2Comp_setting.get_model)
    print(f"Ka: {Ka}, Kd: {Kd}, Qmax: {Qmax}")
    c_sample = np.linspace(0, cpMax, NTRAIN)
    cp = np.array([[c1_i, c2_j] for c1_i in c_sample for c2_j in c_sample])
    cs = multi_component_langmuir_equilibrium(cp=cp, keq=Ka/Kd, qmax=Qmax)

    output_dir = str(output_dir) + f"/langmuir2comp/GPR"
    os.makedirs(output_dir, exist_ok=True)

    training_results = train_gpr_for_cadet(
                cp, cs,
                kernel=kernel,
                optimization_restarts=optimization_restarts,
                add_noise=add_noise,
                )

    surrogate = _build_gpr_surrogate(training_results, cp=cp, cs=cs, kernel_name=kernel)

    cs_pred, metrics, models = evaluate_surrogate_vs_isotherm(
        cp=cp,
        cs_true=cs,
        surrogate=surrogate,
        output_path=output_dir,
        isotherm_model="langmuir"
    )

    simMechanistic1, simHybrid1 = run_hybrid_sim_analysis(
        "GPR",
        cadet_path, output_dir,
        kernel="MLP",
        get_model=partial(langmuir_2Comp_setting.get_model, refinement=refinement),
        training_results=training_results,
        cp=cp, cs=cs,
        )

#######################################################################################
# 1comp Langmuir binding isotherm ANN analysis
#######################################################################################

def binding_train_ANN_langmuir1Comp(cadet_path: str, output_dir: str):

    NTRAIN = 150
    hidden_nodes = 4
    n_layers = 2
    epochs = 200 # maximum number of training epochs
    patience = 50 # stop training if validation loss does not improve for `patience` consecutive epochs
    normalization_factor = [1.0] # [Ka/Kd], [1.0] # None
    acceptance_threshold = 0.01 # stops retries if threshold is achieved
    max_retries = 1
    training_strategy = "random_split" #  "none", "random_split", "leave_one_out", "k_fold"
    validation_split = 0.2 # for random split

    # sample the cp space and compute the corresponding equilibrium loading
    Ka, Kd, Qmax = 2.0, 1.0, 20.0
    cpMax = 10.0
    print(f"Ka: {Ka}, Kd: {Kd}, Qmax: {Qmax}")
    cp = np.linspace(0.0, cpMax, NTRAIN)
    cs = multi_component_langmuir_equilibrium(cp=cp, keq=Ka/Kd, qmax=Qmax)

    plot_training_curves=str(output_dir) + f"/langmuir1comp/epochs{epochs}_strategy_{training_strategy}" + f"/nLayer{n_layers}_nNodes{hidden_nodes}" + f"/nTrain{NTRAIN}"
    
    os.makedirs(plot_training_curves, exist_ok=True)

    from src.training_ann import train_ann_for_cadet

    training_results = train_ann_for_cadet(
                cp, cs, hidden_nodes=hidden_nodes, n_layers=n_layers,
                normalization_factor=normalization_factor, epochs=epochs, patience=patience,
                max_retries=max_retries, acceptance_threshold=acceptance_threshold, training_strategy=training_strategy, validation_split=validation_split,
                plot_training_curves=plot_training_curves
                )

    surrogate = _build_ann_surrogate(
        training_results,
        hidden_nodes=hidden_nodes,
        n_layers=n_layers,
        normalization_factor=normalization_factor,
    )

    cs_pred, metrics, models = evaluate_surrogate_vs_isotherm(
        cp=cp,
        cs_true=cs,
        surrogate=surrogate,
        output_path=plot_training_curves,
        isotherm_model="langmuir"
    )

    print("\n=== Summary metrics ===")
    for k, v in metrics.items():
        print(k, v)

    simMechanistic1, simHybrid1 = run_hybrid_sim_analysis(
        "ANN",
        cadet_path, plot_training_curves,
        get_model=partial(
            langmuir_1Comp_setting.get_model,
                file_name="testitest.h5",
                mode="MCL",
                loading=np.linspace(0.0, 50.0, 50 + 1),
                column_key="favorable_lysozyme",
                keq=Ka/Kd,
                qm=Qmax * (1.0 - 0.75), # mechanistic model shenanigans
                add_noise=True, # for deterministic data sets
            ),
        training_results=training_results,
        normalization_factor=normalization_factor,
        n_layers=n_layers,
        hidden_nodes=hidden_nodes,
    )

#######################################################################################
# Langmuir 2Comp binding isotherm ANN analysis
#######################################################################################

def binding_train_ANN_langmuir2Comp(cadet_path: str, output_dir: str):

    refinement = 4

    NTRAIN = 100
    cpMax = 10.0 # seen as max c^l for unrefined model
    hidden_nodes = 4
    n_layers = 2
    epochs = 150 # maximum number of training epochs
    patience = 50 # stop training if validation loss does not improve for `patience` consecutive epochs
    normalization_factor = [1.0, 1.0] # Ka/Kd, [1.0, 1.0] # None
    acceptance_threshold = 10.0 # stops retries if threshold is achieved
    max_retries = 1
    training_strategy = "random_split" #  "none", "random_split", "leave_one_out", "k_fold"
    validation_split = 0.2

    # sample the cp space and compute the corresponding equilibrium loading
    Ka, Kd, Qmax = get_langmuir_parameters(langmuir_2Comp_setting.get_model)
    print(f"Ka: {Ka}, Kd: {Kd}, Qmax: {Qmax}")
    c_sample = np.linspace(0, cpMax, NTRAIN)
    cp = np.array([[c1_i, c2_j] for c1_i in c_sample for c2_j in c_sample])
    cs = multi_component_langmuir_equilibrium(cp=cp, keq=Ka/Kd, qmax=Qmax)

    plot_training_curves=str(output_dir) + f"/langmuir2comp/epochs{epochs}_strategy_{training_strategy}" + f"/nLayer{n_layers}_nNodes{hidden_nodes}" + f"/nTrain{NTRAIN}"

    os.makedirs(plot_training_curves, exist_ok=True)

    from src.training_ann import train_ann_for_cadet

    training_results = train_ann_for_cadet(
                cp, cs, hidden_nodes=hidden_nodes, n_layers=n_layers,
                normalization_factor=normalization_factor, epochs=epochs, patience=patience,
                max_retries=max_retries, acceptance_threshold=acceptance_threshold, training_strategy=training_strategy, validation_split=validation_split,
                plot_training_curves=plot_training_curves
                )

    surrogate = _build_ann_surrogate(
        training_results,
        hidden_nodes=hidden_nodes,
        n_layers=n_layers,
        normalization_factor=normalization_factor,
    )

    cs_pred, metrics, models = evaluate_surrogate_vs_isotherm(
        cp=cp,
        cs_true=cs,
        surrogate=surrogate,
        output_path=plot_training_curves,
        isotherm_model="langmuir"
    )

    print("\n=== Summary metrics ===")
    for k, v in metrics.items():
        print(k, v)

    simMechanistic1, simHybrid1 = run_hybrid_sim_analysis(
        "ANN",
        cadet_path, plot_training_curves,
        get_model=partial(langmuir_2Comp_setting.get_model, refinement=refinement, idas_reftol=1e-4),
        training_results=training_results,
        normalization_factor=normalization_factor,
        n_layers=n_layers,
        hidden_nodes=hidden_nodes,
    )

#######################################################################################
# Langmuir 2Comp binding isotherm Spline analysis
#######################################################################################

def binding_train_Spline_langmuir2Comp(cadet_path: str, output_dir: str):
    NTRAIN = 10
    cpMax = 10.0

    Ka, Kd, Qmax = get_langmuir_parameters(langmuir_2Comp_setting.get_model)
    print(f"Ka: {Ka}, Kd: {Kd}, Qmax: {Qmax}")
    c_sample = np.linspace(0, cpMax, NTRAIN)
    cp = np.array([[c1_i, c2_j] for c1_i in c_sample for c2_j in c_sample])
    cs = multi_component_langmuir_equilibrium(cp=cp, keq=Ka/Kd, qmax=Qmax)

    output_dir = str(output_dir) + f"/langmuir2comp/Spline"
    os.makedirs(output_dir, exist_ok=True)

    training_results = train_spline_for_cadet(cp, cs)
    surrogate = training_results["model"]

    cs_pred, metrics, models = evaluate_surrogate_vs_isotherm(
        cp=cp,
        cs_true=cs,
        surrogate=surrogate,
        output_path=output_dir,
        isotherm_model="langmuir",
    )

    print("\n=== Summary metrics ===")
    for k, v in metrics.items():
        print(k, v)

    return cs_pred, metrics, models


###

# oldModel = Cadet()
# oldModel.install_path = r"C:\Users\jmbr\Desktop\CADET_compiled\branch_ANN_mulComp\aRELEASE"
# oldModel.filename = r"C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core\chromatography\binding\epochs250_strategy_random_split\old_Col1D_LRM_langANN_2comp_benchmark1.h5"
# oldModel.load_from_file()
# oldModel.run_simulation()
# oldOutlet = oldModel.root.output.solution.unit_001.solution_outlet
# times = oldModel.root.output.solution.solution_times
# plt.plot(times, oldOutlet[:, 0], label="Old ANN comp 1", linestyle='solid', color='blue')
# plt.plot(times, oldOutlet[:, 1], label="Old ANN comp 2", linestyle='solid', color='blue')

# newModel = Cadet()
# newModel.install_path = r"C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE"
# newModel.filename = r"C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core\chromatography\binding\epochs250_strategy_random_split\new_Col1D_LRM_langANN_2comp_benchmark1.h5"
# newModel.load_from_file()
# newModel.run_simulation()
# newOutlet = newModel.root.output.solution.unit_001.solution_outlet
# times = newModel.root.output.solution.solution_times
# plt.plot(times, newOutlet[:, 0], label="New ANN comp 1", linestyle='dashed', color='red')
# plt.plot(times, newOutlet[:, 1], label="New ANN comp 2", linestyle='dashed', color='red')
# plt.show()

# print(max(abs(oldOutlet[:, 0] - newOutlet[:, 0])))
# print(max(abs(oldOutlet[:, 1] - newOutlet[:, 1])))
