"""Lightweight surrogate wrappers used by the Python evaluation workflow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from collections.abc import Sequence as ABCSequence
from typing import Any, Optional, Sequence

import numpy as np

from src.training_gpr import rebuild_gpr_from_cadet_weights, to_gpy_array, train_gpr_for_cadet


def _get_training_ann_module() -> Any:
    """Import the ANN training module lazily to avoid hard TensorFlow coupling."""

    from src import training_ann

    return training_ann


def _as_2d_array(X: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return ``X`` as a floating-point 2D array."""

    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}.")
    return array


def _stack_outputs(outputs: list[np.ndarray]) -> np.ndarray:
    """Stack one prediction vector per output state into ``(N, M)``."""

    if not outputs:
        raise RuntimeError("No surrogate outputs are available.")

    stacked = np.column_stack([np.asarray(output, dtype=float).reshape(-1) for output in outputs])
    if stacked.ndim == 1:
        stacked = stacked.reshape(-1, 1)
    return stacked


class BaseSurrogate(ABC):
    """Common prediction interface for surrogate models."""

    name: str = "surrogate"

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> BaseSurrogate:
        """Fit the surrogate on training data and return ``self``."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict equilibrium loadings for the given concentrations."""


@dataclass
class ANNSurrogate(BaseSurrogate):
    """Wrapper around the existing CADET ANN training/rebuild workflow."""

    hidden_nodes: int = 16
    n_layers: int = 2
    normalization_factor: Optional[Sequence[Any]] = None
    train_kwargs: dict[str, Any] = field(default_factory=dict)
    models: list[Any] = field(default_factory=list)

    name: str = "ANN"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ANNSurrogate:
        """Train the ANN surrogate using the existing CADET training helper."""

        training_ann = _get_training_ann_module()
        training_results = training_ann.train_ann_for_cadet(
            X,
            Y,
            hidden_nodes=self.hidden_nodes,
            n_layers=self.n_layers,
            normalization_factor=self.normalization_factor,
            **self.train_kwargs,
        )

        surrogate = self.from_training_results(
            training_results,
            hidden_nodes=self.hidden_nodes,
            n_layers=self.n_layers,
            normalization_factor=self.normalization_factor,
        )
        self.models = surrogate.models
        self.normalization_factor = surrogate.normalization_factor
        return self

    @classmethod
    def from_training_results(
        cls,
        training_results: dict[str, Any],
        *,
        hidden_nodes: Optional[int] = None,
        n_layers: Optional[int] = None,
        normalization_factor: Optional[Sequence[Any]] = None,
    ) -> ANNSurrogate:
        """Rebuild an ANN surrogate from CADET-ready weights."""

        training_ann = _get_training_ann_module()
        state_keys = sorted(key for key in training_results if key.startswith("bound_state_"))
        if not state_keys:
            raise ValueError("training_results does not contain any ANN bound states.")

        states = [training_results[key] for key in state_keys]
        first_kernel = np.asarray(states[0]["layer_0"]["KERNEL"])
        inferred_hidden_nodes = int(first_kernel.shape[1])
        inferred_n_layers = len([key for key in states[0] if key.startswith("layer_")]) - 1

        models = []
        factors = []
        for state in states:
            state_hidden_nodes = hidden_nodes if hidden_nodes is not None else int(np.asarray(state["layer_0"]["KERNEL"]).shape[1])
            state_n_layers = n_layers if n_layers is not None else len([key for key in state if key.startswith("layer_")]) - 1
            state_input_dim = int(np.asarray(state["layer_0"]["KERNEL"]).shape[0])
            model = training_ann.rebuild_ann_from_cadet_weights(
                state,
                input_dim=state_input_dim,
                hidden_nodes=state_hidden_nodes,
                n_layers=state_n_layers,
            )
            models.append(model)
            factors.append(state.get("NORM_FACTOR"))

        return cls(
            hidden_nodes=hidden_nodes if hidden_nodes is not None else inferred_hidden_nodes,
            n_layers=n_layers if n_layers is not None else inferred_n_layers,
            normalization_factor=normalization_factor if normalization_factor is not None else factors,
            models=models,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict equilibrium loadings for all output states."""

        if not self.models:
            raise RuntimeError("ANNSurrogate has not been fitted.")

        X_array = _as_2d_array(X)
        outputs = []

        for index, model in enumerate(self.models):
            if self.normalization_factor is None:
                norm_factor: Any = 1.0
            elif isinstance(self.normalization_factor, ABCSequence) and not isinstance(self.normalization_factor, np.ndarray):
                norm_factor = self.normalization_factor[index]
            else:
                norm_factor = self.normalization_factor

            X_norm = X_array / norm_factor
            predictions = model.predict(X_norm, verbose=0).reshape(-1)
            outputs.append(predictions)

        return _stack_outputs(outputs)


@dataclass
class GPRSurrogate(BaseSurrogate):
    """Wrapper around the existing CADET GPR training/rebuild workflow."""

    kernel_name: str = "RBF"
    optimization_restarts: int = 30
    add_noise: bool = True
    models: list[Any] = field(default_factory=list)

    name: str = "GPR"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> GPRSurrogate:
        """Train the GPR surrogate using the existing CADET training helper."""

        training_results = train_gpr_for_cadet(
            X,
            Y,
            kernel=self.kernel_name,
            optimization_restarts=self.optimization_restarts,
            add_noise=self.add_noise,
        )

        surrogate = self.from_training_results(
            training_results,
            cp=X,
            cs=Y,
            kernel_name=self.kernel_name,
        )
        self.models = surrogate.models
        return self

    @staticmethod
    def _bound_indices(training_results: dict[str, Any]) -> list[int]:
        indices = {
            int(match.group(1))
            for key in training_results
            for match in [re.search(r"_BND_(\d{3})$", key)]
            if match is not None
        }
        return sorted(indices)

    @classmethod
    def from_training_results(
        cls,
        training_results: dict[str, Any],
        *,
        cp: np.ndarray,
        cs: np.ndarray,
        kernel_name: str,
    ) -> GPRSurrogate:
        """Rebuild a GPR surrogate from CADET-ready hyperparameters."""

        cp_array = to_gpy_array(cp)
        cs_array = np.asarray(cs, dtype=float)
        if cs_array.ndim == 1:
            cs_array = cs_array.reshape(-1, 1)

        models = []
        for bound_idx in cls._bound_indices(training_results):
            model = rebuild_gpr_from_cadet_weights(
                cp=cp_array,
                cs_comp=to_gpy_array(cs_array[:, [bound_idx]]),
                weights=training_results,
                kernel_name=kernel_name,
                boundIdx=bound_idx,
            )
            models.append(model)

        return cls(kernel_name=kernel_name, models=models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict equilibrium loadings for all output states."""

        if not self.models:
            raise RuntimeError("GPRSurrogate has not been fitted.")

        X_array = to_gpy_array(X)
        outputs = []
        for model in self.models:
            mean, _ = model.predict(X_array)
            outputs.append(mean.reshape(-1))

        return _stack_outputs(outputs)
