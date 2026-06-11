"""Train Gaussian Process Regression (GPR) models for CADET binding models.

This module helps you go from raw binding training pairs to a CADET configuration:
- Input: concentration samples ``cp`` and loading samples ``cs``.
- Model: GPy-based GPR with selectable kernel (RBF, MLP, or +Linear variants).
- Output: hyper-parameters expected by CADET GPR configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


KernelName = Literal["RBF", "MLP", "RBF_Linear", "MLP_Linear"]

def _get_gpy() -> Any:
    """Import GPy lazily so the module remains importable without it."""

    try:
        import GPy  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on external env
        raise ImportError(
            "GPy is required for GPR training. Install 'GPy' in the active environment."
        ) from exc

    return GPy

def _build_kernel(gpy: Any, name: KernelName, input_dim: int) -> Any:
    """Construct a supported kernel from a short kernel name."""

    if name == "RBF":
        return gpy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=1.0)
    if name == "MLP":
        return gpy.kern.MLP(input_dim=input_dim, variance=1.0, weight_variance=1.0, bias_variance=1.0)
    if name == "RBF_Linear":
        return gpy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=1.0) + gpy.kern.Linear(input_dim=input_dim)
    if name == "MLP_Linear":
        return gpy.kern.MLP(input_dim=input_dim, variance=1.0, weight_variance=1.0, bias_variance=1.0) + gpy.kern.Linear(input_dim=input_dim)
    raise ValueError(f"Unsupported kernel: {name}")

def _pack_cadet_params(model: Any, kernel_name: KernelName, bndSuffix:str) -> np.ndarray:
    """Pack GPy hyperparameters to the 7-value CADET expected ordering."""

    if kernel_name == "RBF":
        return {
                "KERNEL": "RBF",
                f"MLP_WEIGHT_VARIANCE_BND_{bndSuffix}": 0.0,
                f"MLP_BIAS_VARIANCE_BND_{bndSuffix}": 0.0,
                f"MLP_VARIANCE_BND_{bndSuffix}": 0.0,
                f"RBF_VARIANCE_BND_{bndSuffix}": model.rbf.flattened_parameters[0][0],
                f"RBF_LENGTHSCALE_BND_{bndSuffix}": model.rbf.flattened_parameters[1][0] ** 2,
                f"LINEAR_VARIANCE_BND_{bndSuffix}": 0.0,
                f"GAUSSIAN_NOISE_VARIANCE_BND_{bndSuffix}": model.Gaussian_noise.flattened_parameters[0][0],
            }

    if kernel_name == "MLP":
        return {
            "KERNEL": "MLP",
            f"MLP_WEIGHT_VARIANCE_BND_{bndSuffix}": model.mlp.flattened_parameters[1][0],
            f"MLP_BIAS_VARIANCE_BND_{bndSuffix}": model.mlp.flattened_parameters[2][0],
            f"MLP_VARIANCE_BND_{bndSuffix}": model.mlp.flattened_parameters[0][0],
            f"RBF_VARIANCE_BND_{bndSuffix}": 0.0,
            f"RBF_LENGTHSCALE_BND_{bndSuffix}": 0.0,
            f"LINEAR_VARIANCE_BND_{bndSuffix}": 0.0,
            f"GAUSSIAN_NOISE_VARIANCE_BND_{bndSuffix}": model.Gaussian_noise.flattened_parameters[0][0],
        }

    if kernel_name == "RBF_Linear":
            return {
                "KERNEL": "RBF_Linear",
                f"MLP_WEIGHT_VARIANCE_BND_{bndSuffix}":0.0,
                f"MLP_BIAS_VARIANCE_BND_{bndSuffix}":0.0,
                f"MLP_VARIANCE_BND_{bndSuffix}":0.0,
                f"RBF_VARIANCE_BND_{bndSuffix}": model.sum.rbf.flattened_parameters[0][0],
                f"RBF_LENGTHSCALE_BND_{bndSuffix}": model.sum.rbf.flattened_parameters[1][0] ** 2,
                f"LINEAR_VARIANCE_BND_{bndSuffix}": model.sum.linear.flattened_parameters[0][0],
                f"GAUSSIAN_NOISE_VARIANCE_BND_{bndSuffix}": model.Gaussian_noise.flattened_parameters[0][0],
            }

    if kernel_name == "MLP_Linear":
        return {
            "KERNEL": "MLP_Linear",
            f"MLP_WEIGHT_VARIANCE_BND_{bndSuffix}": model.sum.mlp.flattened_parameters[1][0],
            f"MLP_BIAS_VARIANCE_BND_{bndSuffix}": model.sum.mlp.flattened_parameters[2][0],
            f"MLP_VARIANCE_BND_{bndSuffix}": model.sum.mlp.flattened_parameters[0][0],
            f"RBF_VARIANCE_BND_{bndSuffix}": 0.0,
            f"RBF_LENGTHSCALE_BND_{bndSuffix}": 0.0,
            f"LINEAR_VARIANCE_BND_{bndSuffix}": model.sum.linear.flattened_parameters[0][0],
            f"GAUSSIAN_NOISE_VARIANCE_BND_{bndSuffix}": model.Gaussian_noise.flattened_parameters[0][0],
        }

    raise ValueError(f"Unsupported kernel: {kernel_name}")

import numpy as np

def to_gpy_array(X):
    """
    Convert a NumPy array into GPy-compatible 2D format.

    Rules:
    - 1D -> (N, 1)
    - 2D -> unchanged
    - 3D+ -> error
    """

    X = np.asarray(X)

    if X.ndim > 2:
        raise ValueError(f"Input must be 1D or 2D array, got shape {X.shape}")

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X

def rebuild_gpr_from_cadet_weights(cp, cs_comp, weights, kernel_name, boundIdx=0):

    gpy = _get_gpy()

    suffix = f"{boundIdx:03d}"
    
    kernel = _build_kernel(gpy, kernel_name, input_dim=cp.shape[1])
    
    model = gpy.models.GPRegression(cp, cs_comp, kernel)

    if kernel_name == "RBF":
        model.rbf.variance = weights[f"RBF_VARIANCE_BND_{suffix}"]
        model.rbf.lengthscale = np.sqrt(
            weights[f"RBF_LENGTHSCALE_BND_{suffix}"]
        )

    elif kernel_name == "MLP":
        model.mlp.variance = weights[f"MLP_VARIANCE_BND_{suffix}"]
        model.mlp.weight_variance = weights[f"MLP_WEIGHT_VARIANCE_BND_{suffix}"]
        model.mlp.bias_variance = weights[f"MLP_BIAS_VARIANCE_BND_{suffix}"]

    elif kernel_name == "RBF_Linear":
        model.sum.rbf.variance = weights[f"RBF_VARIANCE_BND_{suffix}"]
        model.sum.rbf.lengthscale = np.sqrt(
            weights[f"RBF_LENGTHSCALE_BND_{suffix}"]
        )
        model.sum.linear.variances = weights[f"LINEAR_VARIANCE_BND_{suffix}"]

    elif kernel_name == "MLP_Linear":
        model.sum.mlp.variance = weights[f"MLP_VARIANCE_BND_{suffix}"]
        model.sum.mlp.weight_variance = weights[f"MLP_WEIGHT_VARIANCE_BND_{suffix}"]
        model.sum.mlp.bias_variance = weights[f"MLP_BIAS_VARIANCE_BND_{suffix}"]
        model.sum.linear.variances = weights[f"LINEAR_VARIANCE_BND_{suffix}"]

    else:
        raise ValueError(f"Unsupported kernel: {kernel_name}")

    model.Gaussian_noise.variance = (
        weights[f"GAUSSIAN_NOISE_VARIANCE_BND_{suffix}"]
    )

    return model

def train_gpr_for_cadet(
    cp: np.ndarray,
    cs: np.ndarray,
    *,
    kernel: KernelName = "RBF",
    optimization_restarts: int = 30,
    add_noise: bool = True
):
    """Train per-component GPR models and return CADET-ready hyperparameter vectors."""

    cp = np.asarray(cp, dtype=float)
    cs = np.asarray(cs, dtype=float)

    if cs.ndim == 1:
        cs = cs.reshape(-1, 1)
    elif cs.ndim != 2:
        raise ValueError(f"cs must be 1D or 2D, got {cs.shape}")

    N, nBound = cs.shape

    if cp.ndim == 1:
        cp = cp.reshape(-1, 1)
        cp_mode = "shared"
    elif cp.ndim == 2:
        if cp.shape[0] != N:
            raise ValueError(f"cp and cs must have same number of rows: {cp.shape}, {cs.shape}")

        if cp.shape[1] == 1:
            cp_mode = "shared"
        elif cp.shape[1] == nBound:
            cp_mode = "per_output"
        else:
            raise ValueError(
                f"cp has shape {cp.shape} but cs has shape {cs.shape}. "
                "cp must have either 1 column or match cs columns."
            )
    else:
        raise ValueError(f"cp must be 1D or 2D, got {cp.shape}")

    kernel_params_for_cadet = {}

    for i in range(nBound):

        gpy = _get_gpy()

        # select input
        if cp_mode == "per_output":
            X = cp[:, [i]]   # (N,1) per output
        else:
            X = cp           # shared input

        y = cs[:, [i]]       # always (N,1)

        gpr = gpy.models.GPRegression(X, y, _build_kernel(gpy, kernel, input_dim=X.shape[1]))

        if add_noise:
            gpr.Gaussian_noise.variance = 1e-6
            gpr.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-2)

        gpr.optimize(messages=True)
        gpr.optimize_restarts(num_restarts=optimization_restarts)

        kernel_params_for_cadet.update(
            _pack_cadet_params(gpr, kernel, bndSuffix=f"{i:03d}")
        )

    return kernel_params_for_cadet
