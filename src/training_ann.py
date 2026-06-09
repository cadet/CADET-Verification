"""Train feed-forward ANN surrogates for CADET binding models.

This module helps you go from raw binding training pairs to a CADET configuration:
- Input: concentration samples ``cp`` and loading samples ``cs``.
- Model: a 2-hidden-layer Keras network (default width 75, activations ELU/ELU/ReLU).
- Output: hyper-parameters expected by CADET ANN configuration.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold, LeaveOneOut
from typing import Literal
tf.get_logger().setLevel("ERROR")

def _build_model(input_dim: int, hidden_nodes: int = 16, n_layers: int = 2) -> Any:

    if n_layers == 1:
        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            layers.Dense(hidden_nodes, activation="elu", kernel_initializer="he_uniform", bias_initializer="zeros"),
            layers.Dense(1, activation=None),
        ])
    elif n_layers == 2:
        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            layers.Dense(hidden_nodes, activation="elu", kernel_initializer="he_uniform", bias_initializer="zeros"),
            layers.Dense(hidden_nodes, activation="elu", kernel_initializer="he_uniform", bias_initializer="zeros"),
            layers.Dense(1, activation=None),
        ])
    else:
        raise ValueError("Unsupported number of layers: {}".format(n_layers))
    
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=["mae"],
    )
    return model


def _to_cadet_weights(model):
    weights = {}

    dense_idx = 0

    for layer in model.layers:

        if not hasattr(layer, "get_weights"):
            continue

        params = layer.get_weights()

        if len(params) == 0:
            continue

        kernel, bias = params

        weights[f"layer_{dense_idx}"] = {
            "KERNEL": np.asarray(kernel),
            "BIAS": np.asarray(bias),
        }

        dense_idx += 1

    return weights

def _get_splits(X, y, strategy: str, val_ratio: float = 0.2):
    """Return list of (train_idx, val_idx) splits."""

    n = len(X)
    idx = np.arange(n)

    if strategy == "random_split":
        np.random.shuffle(idx)
        split = int(n * (1 - val_ratio))
        return [(idx[:split], idx[split:])]

    elif strategy == "k_fold":
        k = int(1 / val_ratio) if val_ratio > 0 else 5
        kf = KFold(n_splits=k, shuffle=True)
        return [(train_idx, val_idx) for train_idx, val_idx in kf.split(X)]

    elif strategy == "leave_one_out":
        loo = LeaveOneOut()
        return [(train_idx, val_idx) for train_idx, val_idx in loo.split(X)]

    elif strategy == "none":
        return [(idx, idx)]  # train on all, validate on all

    else:
        raise ValueError(f"Unknown training_strategy: {strategy}")

def _train_single_ann(
    X_norm: np.ndarray,
    y: np.ndarray,
    *,
    hidden_nodes: int,
    epochs: int,
    patience: int,
    max_retries: int,
    acceptance_threshold: float,
    verbose: int,
    validation_split: float,
    training_strategy: Literal[
        "random_split",
        "k_fold",
        "leave_one_out",
        "none",
    ] = "random_split",
    n_layers: int = 2,
    plot_training_curves: bool = False,
) -> tuple[dict, float]:
    """Train one ANN with flexible validation strategy."""

    X_norm = np.asarray(X_norm, dtype=float)
    y = np.asarray(y, dtype=float)

    splits = _get_splits(X_norm, y, training_strategy, val_ratio=validation_split)

    best_weights = None
    best_criterion = float("inf")

    for _ in range(max_retries):

        print(f"Training attempt {_ + 1}/{max_retries} with strategy '{training_strategy}'...")
        split_criteria = []
        best_fold_weights: Optional[dict] = None
        best_fold_criterion = float("inf")

        # -------------------------------------------------
        # train + evaluate over all splits
        # -------------------------------------------------
        for train_idx, val_idx in splits:

            model = _build_model(input_dim=X_norm.shape[1], hidden_nodes=hidden_nodes, n_layers=n_layers)

            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )

            X_train, y_train = X_norm[train_idx], y[train_idx]
            X_val, y_val = X_norm[val_idx], y[val_idx]

            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                verbose=verbose,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
            )

            if plot_training_curves:
                plt.figure()
                plt.semilogy(history.history["loss"], label="train")
                plt.semilogy(history.history["val_loss"], label="validation")
                plt.xlabel("Epoch")
                plt.ylabel("MSE Loss")
                plt.legend()
                plt.grid(True)
                print("Stopped after", len(history.history["loss"]), "epochs")
                print("Best val_loss =", np.min(history.history["val_loss"]))

            pred_val = model.predict(X_val, verbose=0).reshape(-1)
            fold_criterion = float(np.mean((y_val - pred_val) ** 2))
            split_criteria.append(fold_criterion)

            if fold_criterion < best_fold_criterion:
                best_fold_criterion = fold_criterion
                best_fold_weights = _to_cadet_weights(model)

        criterion = float(np.mean(split_criteria))

        if criterion < best_criterion:
            best_criterion = criterion
            best_weights = best_fold_weights

        if criterion <= acceptance_threshold:
            break

        tf.keras.backend.clear_session()

    if best_weights is None:
        raise RuntimeError("ANN training did not produce a valid model.")

    return best_weights, best_criterion

def train_ann_for_cadet(
    cp: np.ndarray,
    cs: np.ndarray,
    *,
    hidden_nodes: int = 16,
    epochs: int = 500,
    patience: int = 50,
    acceptance_threshold: float = 100.0,
    validation_split: float = 0.2,
    verbose: int = 0,
    random_seed: Optional[int] = None,
    max_retries: int = 2,
    training_strategy: Literal[
        "random_split",
        "k_fold",
        "leave_one_out",
        "none",
    ] = "random_split",
    n_layers: int = 2,
    plot_training_curves: bool = False,
    normalization_factor: Optional[List[float]] = None,
) -> dict:

    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)

    cp = np.asarray(cp, dtype=float)
    cs = np.asarray(cs, dtype=float)

    if cs.ndim == 1:
        cs = cs.reshape(-1, 1)

    N, nBound = cs.shape

    if cp.ndim == 1:
        cp = cp.reshape(-1, 1)

    if cp.shape[0] != N:
        raise ValueError("cp and cs must have same number of rows")

    results = {}

    for i in range(nBound):

        y = cs[:, i]
        X = cp

        if normalization_factor is not None:
            norm_factor = normalization_factor[i]
        else:
            norm_factor = 1.0 / (np.max(X, axis=0) + 1e-12)

        X_norm = X / norm_factor

        weights, _ = _train_single_ann(
            X_norm,
            y,
            hidden_nodes=hidden_nodes,
            n_layers=n_layers,
            max_retries=max_retries,
            acceptance_threshold=acceptance_threshold,
            verbose=verbose,
            epochs=epochs,
            patience=patience,
            training_strategy=training_strategy,
            validation_split=validation_split,
            plot_training_curves=plot_training_curves
        )

        results[f"bound_state_{i:03d}"] = {
            "NORM_FACTOR": norm_factor,
            "layer_0": {
                "KERNEL": weights["layer_0"]["KERNEL"],
                "BIAS": weights["layer_0"]["BIAS"],
            },
            "layer_1": {
                "KERNEL": weights["layer_1"]["KERNEL"],
                "BIAS": weights["layer_1"]["BIAS"],
            }
        }

        if n_layers == 2:
            results[f"bound_state_{i:03d}"]["layer_2"] = {
                "KERNEL": weights["layer_2"]["KERNEL"],
                "BIAS": weights["layer_2"]["BIAS"],
            }

    return results
