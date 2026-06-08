"""Train notebook-style ANN/GPR surrogates, run CADET, and compare to references.

This script reproduces two reference notebook simulation cases:
1. GPR adsorption with the shallow MLP kernel case stored in
   Input Files/GPR/Shallow/Number of Training Data/MLP/GPR_Shallow_100.h5
2. ANN adsorption with the 2 hidden layers / 4 neurons case stored in
   Input Files/ANN/model_ANN_check_4_neurons_1pKin_native.h5

For each case the workflow is:
1. Train the surrogate model
2. Copy the corresponding reference CADET H5 file
3. Replace only the adsorption payload in that copied file
4. Run CADET on the copied file
5. Compare the new outlet profile against the notebook reference output

The ANN reference sparse training knots are not stored in the workspace.
To make the reproduction executable, the ANN case is retrained against the
saved notebook ANN model response from the matching Keras H5 file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import platform
import shutil
from typing import Iterable
import warnings

import h5py
import numpy as np


# Reduce non-actionable TensorFlow startup noise while keeping errors visible.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Disable oneDNN custom op reordering for reproducible CPU numerics and to
# avoid the informational startup message.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# GPy/paramz occasionally emits overflow warnings during restart exploration.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"overflow encountered in expm1",
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = WORKSPACE_ROOT / "data" / "CADET-Core_reference" / "binding"

GPR_REFERENCE_1 = "Col1D_GRM_GPR_Shallow_MLP_7.h5"
ANN_REFERENCE_1 = "AP_2_layer_4_nodes_shallow.h5"


@dataclass
class CaseResult:
    case_name: str
    output_h5: Path
    outlet_rmse: float
    outlet_max_abs: float
    extra_metrics: dict[str, float]


def _ensure_inputs_exist(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(f" - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{joined}")


def _ensure_runtime_dependencies() -> None:
    missing = []
    for module in ("cadet", "GPy", "tensorflow", "scipy", "sklearn"):
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(module)

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing runtime dependencies: {joined}. "
            "Install them in the active environment before running this script."
        )


def _load_gpr_reference_training_data(reference_h5: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(reference_h5, "r") as handle:
        training = handle["/input/model/unit_001/particle_type_000/adsorption"]
        cp = np.asarray(training["CP_VALS"][()], dtype=float)
        cs = np.asarray(training["CS_VALS"][()], dtype=float)
        params = np.asarray(training["TRAINED_PARAMS"][()], dtype=float)
    return cp, cs, params


def _load_reference_outlet(reference_h5: Path) -> np.ndarray:
    with h5py.File(reference_h5, "r") as handle:
        return np.asarray(handle["/output/solution/unit_001/SOLUTION_OUTLET"][()], dtype=float).reshape(-1)


def _copy_reference_file(reference_h5: Path, output_h5: Path) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference_h5, output_h5)


def get_cadet_path():
    install_path = None
    
    executable = 'cadet-cli'
    if install_path is None:
        try:
            if platform.system() == 'Windows':
                executable += '.exe'
            executable_path = Path(shutil.which(executable))
        except TypeError:
            raise FileNotFoundError(
                "CADET could not be found. Please set an install path"
            )
        install_path = executable_path.parent.parent
    
    install_path = Path(install_path)
    cadet_bin_path = install_path
    
    if cadet_bin_path.exists():
        return cadet_bin_path
        
    else:
        raise FileNotFoundError(
            "CADET could not be found. Please check the path"
        )


def _run_cadet(output_h5: Path, cadet_path: str | None = None):
    try:
        from cadet import Cadet  # type: ignore
    except ImportError as exc:  # pragma: no cover - external dependency
        raise ImportError(
            "pycadet is required to run CADET simulations from this script."
        ) from exc

    model = Cadet()
    if cadet_path is not None:
        model.install_path = cadet_path
    model.filename = str(output_h5)
    return_data = model.run_simulation()
    model.load_from_file()
    if return_data.return_code != 0:
        raise RuntimeError(f"CADET simulation failed with return code {return_data.return_code}, error message {return_data.error_message} and log {return_data.log}")
    else:
        model.save()

    return model


def _compare_outlets(candidate_h5: Path, reference_h5: Path) -> tuple[float, float]:
    cand = _load_reference_outlet(candidate_h5)
    ref = _load_reference_outlet(reference_h5)
    length = min(len(cand), len(ref))
    if length == 0:
        raise ValueError("No outlet data found for comparison.")
    diff = cand[:length] - ref[:length]
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    return rmse, max_abs


def _load_keras_weights(keras_h5: Path) -> list[np.ndarray]:
    with h5py.File(keras_h5, "r") as handle:
        return [
            np.asarray(handle["/model_weights/dense/dense/kernel:0"][()], dtype=float),
            np.asarray(handle["/model_weights/dense/dense/bias:0"][()], dtype=float),
            np.asarray(handle["/model_weights/dense_1/dense_1/kernel:0"][()], dtype=float),
            np.asarray(handle["/model_weights/dense_1/dense_1/bias:0"][()], dtype=float),
            np.asarray(handle["/model_weights/dense_2/dense_2/kernel:0"][()], dtype=float),
            np.asarray(handle["/model_weights/dense_2/dense_2/bias:0"][()], dtype=float),
        ]


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(x))


def _reference_ann_predict(c: np.ndarray, *, norm_factor: float, sim_h5: Path) -> np.ndarray:
    k0, b0, k1, b1, k2, b2 = _load_weights(sim_h5)
    x = (np.asarray(c, dtype=float).reshape(-1, 1) * norm_factor)
    h0 = _elu(x @ k0 + b0)
    h1 = _elu(h0 @ k1 + b1)
    y = np.maximum(h1 @ k2 + b2, 0.0)
    return y.reshape(-1)


def _overwrite_gpr_payload(output_h5: Path, trained_params: np.ndarray) -> None:
    with h5py.File(output_h5, "r+") as handle:
        handle["/input/model/unit_001/particle_type_000/adsorption/TRAINED_PARAMS"][...] = trained_params


def _overwrite_ann_payload(output_h5: Path, trained_weights: list[np.ndarray]) -> None:
    with h5py.File(output_h5, "r+") as handle:
        base = "/input/model/unit_001/particle_type_000/adsorption"
        dataset_names = [
            f"{base}/layer_0/KERNEL",
            f"{base}/layer_0/BIAS",
            f"{base}/layer_1/KERNEL",
            f"{base}/layer_1/BIAS",
            f"{base}/layer_2/KERNEL",
            f"{base}/layer_2/BIAS",
        ]
        for name, value in zip(dataset_names, trained_weights, strict=True):
            handle[name][...] = value


def run_gpr_case(out_dir: Path, *, optimization_restarts: int) -> CaseResult:
    from gpr_training import train_gpr_for_cadet

    cp, cs, ref_params = _load_gpr_reference_training_data(INPUT_DIR / GPR_REFERENCE_1)

    trained = train_gpr_for_cadet(
        cp,
        cs,
        kernel="MLP",
        optimization_restarts=optimization_restarts,
        prediction_max=float(np.max(cp)),
        prediction_min=float(0.0),
    )

    output_h5 = out_dir / GPR_REFERENCE_1
    _copy_reference_file(INPUT_DIR /  GPR_REFERENCE_1, output_h5)
    _overwrite_gpr_payload(output_h5, trained.params_for_cadet)
# Reproduction summary
# ====================
# Case: GPR_MLP_Shallow_100
# Output H5: C:\Users\jmbr\software\CADET-Verification\data\CADET-Core_reference\binding\Reproduced\Col1D_GRM_GPR_Shallow_MLP_7.h5
# Outlet RMSE: 1.415323e-02
# Outlet max abs diff: 2.938733e-02
# param_rmse: 3.490356e+02
# langmuir_nrmse: 1.110019e+01

#     from src.benchmark_models.settings_data_driven_bnd import get_model

#     model = get_model(
#         file_name=str(output_h5),
#         mode="GPR", cp=cp, cs=cs, loading=cs, column_key="favorable_lysozyme", keq=1.0, qm=1.0,
#         gpr_params=trained.params_for_cadet, gpr_kernel="MLP",
#         ann_weights=None, ann_norm_factor=None, ann_poros_factor=None, ann_impl=None
#         )
#     model.save()
# Reproduction summary
# ====================
# Case: GPR_MLP_Shallow_100
# Output H5: C:\Users\jmbr\software\CADET-Verification\data\CADET-Core_reference\binding\Reproduced\Col1D_GRM_GPR_Shallow_MLP_7.h5
# Outlet RMSE: 1.024235e-05
# Outlet max abs diff: 3.252847e-04
# param_rmse: 3.490412e+02
# langmuir_nrmse: 1.110019e+01

    cadet_path = r"C:\Users\jmbr\Desktop\CADET_compiled\branch_GPRmulComp\aRELEASE" # get_cadet_path()

    _run_cadet(output_h5, cadet_path)

    outlet_rmse, outlet_max_abs = _compare_outlets(output_h5, INPUT_DIR /  GPR_REFERENCE_1)

    param_rmse = float(np.sqrt(np.mean((trained.params_for_cadet - ref_params) ** 2)))

    return CaseResult(
        case_name="GPR_MLP_Shallow_100",
        output_h5=output_h5,
        outlet_rmse=outlet_rmse,
        outlet_max_abs=outlet_max_abs,
        extra_metrics={
            "param_rmse": param_rmse,
            "langmuir_nrmse": trained.langmuir_nrmse,
        },
    )


def run_ann_case(
    out_dir: Path,
    *,
    ann_epochs: int,
    ann_training_points: int,
    ann_seed: int,
) -> CaseResult:
    from ann_training import train_ann_for_cadet

    with h5py.File(INPUT_DIR / ANN_REFERENCE_1, "r") as handle:
        norm_factor = float(handle["/input/model/unit_001/particle_type_000/adsorption/NORM_FACTOR"][()])
        feed_concentration = float(handle["/input/model/unit_000/sec_000/CONST_COEFF"][0])

    dense_c = np.linspace(0.0, feed_concentration, 1000)
    dense_q = _reference_ann_predict(dense_c, norm_factor=norm_factor, sim_h5=INPUT_DIR / ANN_REFERENCE_1)

    train_idx = np.linspace(0, len(dense_c) - 1, ann_training_points, dtype=int)
    train_c = dense_c[train_idx]
    train_q = dense_q[train_idx]

    trained = train_ann_for_cadet(
        train_c,
        train_q,
        keq=norm_factor,
        qm=float(np.max(dense_q)),
        dense_c_data=dense_c,
        hidden_nodes=4,
        epochs=ann_epochs,
        patience=max(100, ann_epochs // 4),
        max_retries=3,
        acceptance_threshold=1e-4,
        validation_split=0.2,
        verbose=0,
        random_seed=ann_seed,
    )

    output_h5 = out_dir / "reproduced_ann_2layers_4neurons_native.h5"
    _copy_reference_file(INPUT_DIR / ANN_REFERENCE_1, output_h5)
    trained_weights = [
        trained.weights_for_cadet.layer_0_kernel,
        trained.weights_for_cadet.layer_0_bias,
        trained.weights_for_cadet.layer_1_kernel,
        trained.weights_for_cadet.layer_1_bias,
        trained.weights_for_cadet.layer_2_kernel,
        trained.weights_for_cadet.layer_2_bias,
    ]
    _overwrite_ann_payload(output_h5, trained_weights)
    _run_cadet(output_h5)

    outlet_rmse, outlet_max_abs = _compare_outlets(output_h5, INPUT_DIR / ANN_REFERENCE_1)

    ann_prediction_rmse = float(np.sqrt(np.mean((trained.prediction_q - dense_q) ** 2)))

    return CaseResult(
        case_name="ANN_2Layers_4Neurons_Native",
        output_h5=output_h5,
        outlet_rmse=outlet_rmse,
        outlet_max_abs=outlet_max_abs,
        extra_metrics={
            "ann_prediction_rmse": ann_prediction_rmse,
            "integral_criterion": trained.integral_criterion,
            "mae": trained.mae,
            "mse": trained.mse,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ANN/GPR, run CADET, and compare to notebook H5 references.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=INPUT_DIR / "Reproduced",
        help="Directory for reproduced CADET H5 files.",
    )
    parser.add_argument(
        "--gpr-restarts",
        type=int,
        default=5,
        help="Number of GPR optimization restarts.",
    )
    parser.add_argument(
        "--ann-epochs",
        type=int,
        default=1500,
        help="Maximum ANN training epochs.",
    )
    parser.add_argument(
        "--ann-training-points",
        type=int,
        default=100,
        help="Number of dense ANN reference points used for retraining.",
    )
    parser.add_argument(
        "--ann-seed",
        type=int,
        default=0,
        help="Random seed for ANN retraining.",
    )

    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (WORKSPACE_ROOT / args.out_dir).resolve()

    _ensure_inputs_exist([INPUT_DIR / GPR_REFERENCE_1, INPUT_DIR / ANN_REFERENCE_1])
    _ensure_runtime_dependencies()

    results = [
        run_gpr_case(out_dir, optimization_restarts=args.gpr_restarts),
        # run_ann_case(
        #     out_dir,
        #     ann_epochs=args.ann_epochs,
        #     ann_training_points=args.ann_training_points,
        #     ann_seed=args.ann_seed,
        # ),
    ]

    print("\nReproduction summary")
    print("====================")
    for result in results:
        print(f"Case: {result.case_name}")
        print(f"Output H5: {result.output_h5}")
        print(f"Outlet RMSE: {result.outlet_rmse:.6e}")
        print(f"Outlet max abs diff: {result.outlet_max_abs:.6e}")
        for key, value in result.extra_metrics.items():
            print(f"{key}: {value:.6e}")
        print("")


if __name__ == "__main__":
    main()
