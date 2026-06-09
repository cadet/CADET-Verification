"""Train feed-forward ANN surrogates for CADET binding models.

This module helps you go from raw binding training pairs to a CADET configuration:
- Input: concentration samples ``cp`` and loading samples ``cs``.
- Model: a 2-hidden-layer Keras network (default width 75, activations ELU/ELU/ReLU).
- Output: hyper-parameters expected by CADET ANN configuration.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.get_logger().setLevel("ERROR")

from src.benchmark_models.settings_data_driven_bnd import AnnWeights
from src.training_gpr import fit_langmuir, langmuir_isotherm


def _custom_loss(lambda_train: float = 500000.0):
    def loss(y_true: Any, y_pred: Any) -> Any:
        mse_term = tf.reduce_mean(tf.square(y_true - y_pred))
        l1_term = tf.reduce_mean(tf.abs(y_true - y_pred))
        return mse_term + lambda_train * l1_term
    return loss


def _build_model(input_dim: int, hidden_nodes: int = 75) -> Any:
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(hidden_nodes, activation="elu", kernel_initializer="random_uniform", bias_initializer="zeros"),
        layers.Dense(hidden_nodes, activation="elu", kernel_initializer="random_uniform", bias_initializer="zeros"),
        layers.Dense(1, activation=None),  # no output activation — CADET applies none either
    ])
    model.compile(
        loss=_custom_loss(),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=["mae", "mse"],
    )
    return model


def _to_cadet_weights(model: Any) -> AnnWeights:
    k0, b0, k1, b1, k2, b2 = model.get_weights()
    return AnnWeights(
        layer_0_kernel=np.asarray(k0),
        layer_0_bias=np.asarray(b0),
        layer_1_kernel=np.asarray(k1),
        layer_1_bias=np.asarray(b1),
        layer_2_kernel=np.asarray(k2),
        layer_2_bias=np.asarray(b2),
    )

def _train_single_ann(
    X_norm: np.ndarray,
    y: np.ndarray,
    keq: float,
    qm: float,
    *,
    hidden_nodes: int,
    epochs: int,
    patience: int,
    max_retries: int,
    acceptance_threshold: float,
    validation_split: float,
    verbose: int,
) -> tuple[AnnWeights, float]:
    """Train one component's ANN, retrying until the integral criterion passes.

    The integral criterion measures how well the ANN reproduces the smooth
    Langmuir reference curve on a dense grid — not just at the training points.
    This drives retries because ANN training is non-deterministic (random init).
    """
    dense_norm = np.linspace(X_norm.min(), X_norm.max(), 1000)
    dense_cp = dense_norm / keq
    ref_cs = langmuir_isotherm(dense_cp, keq, qm)

    best_weights: Optional[AnnWeights] = None
    best_criterion = float("inf")

    for _ in range(max_retries):
        model = _build_model(input_dim=1, hidden_nodes=hidden_nodes)
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
        model.fit(
            X_norm, y,
            epochs=epochs,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[early_stop],
        )

        pred_dense = model.predict(dense_norm.reshape(-1, 1), verbose=0).reshape(-1)
        criterion = float(np.trapz((ref_cs - pred_dense) ** 2, dense_cp))

        if criterion < best_criterion:
            best_criterion = criterion
            best_weights = _to_cadet_weights(model)

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
    hidden_nodes: int = 75,
    epochs: int = 2000,
    patience: int = 500,
    max_retries: int = 5,
    acceptance_threshold: float = 100.0,
    validation_split: float = 0.2,
    verbose: int = 0,
    random_seed: Optional[int] = None,
    mechanistic_reference: Optional[callable] = None,
) -> dict:
    """Train per-component ANN models and return CADET-ready weight dicts.

    Each bound state (cs column) gets its own ANN.
    All ANNs are trained on the FULL cp feature matrix.
    """

    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)

    cp = np.asarray(cp, dtype=float)
    cs = np.asarray(cs, dtype=float)

    # -------------------------
    # normalize cs (outputs)
    # -------------------------
    if cs.ndim == 1:
        cs = cs.reshape(-1, 1)
    elif cs.ndim != 2:
        raise ValueError(f"cs must be 1D or 2D, got {cs.shape}")

    N, nBound = cs.shape

    # -------------------------
    # normalize cp (inputs)
    # -------------------------
    if cp.ndim == 1:
        cp = cp.reshape(-1, 1)
    elif cp.ndim != 2:
        raise ValueError(f"cp must be 1D or 2D, got {cp.shape}")

    if cp.shape[0] != N:
        raise ValueError(f"cp and cs must have same number of rows: {cp.shape}, {cs.shape}")

    # -------------------------
    # train per bound state ANN
    # -------------------------
    results = {}

    X_raw = cp

    for i in range(nBound):
        y = cs[:, i]

        # fit Langmuir scaling per bound state
        keq, qm = fit_langmuir(X_raw.reshape(-1, cp.shape[1]), y)

        # IMPORTANT: normalization must broadcast correctly
        X_norm = X_raw * keq

        weights, _ = _train_single_ann(
            X_norm,
            y,
            keq,
            qm,
            hidden_nodes=hidden_nodes,
            epochs=epochs,
            patience=patience,
            max_retries=max_retries,
            acceptance_threshold=acceptance_threshold,
            validation_split=validation_split,
            verbose=verbose,
        )

        results[f"bound_state_{i:03d}"] = {
            "NORM_FACTOR": keq,
            "layer_0": {
                "KERNEL": weights.layer_0_kernel,
                "BIAS": weights.layer_0_bias,
            },
            "layer_1": {
                "KERNEL": weights.layer_1_kernel,
                "BIAS": weights.layer_1_bias,
            },
            "layer_2": {
                "KERNEL": weights.layer_2_kernel,
                "BIAS": weights.layer_2_bias,
            },
        }

    return results
