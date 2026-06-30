"""Structured-grid spline surrogates for adsorption equilibrium data."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.surrogate_models import BaseSurrogate


def _as_2d_array(X: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return ``X`` as a floating-point 2D array."""

    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}.")
    return array


@dataclass(frozen=True)
class StructuredGridInfo:
    """Information recovered from a Cartesian product training grid."""

    axes: tuple[np.ndarray, ...]
    grid_shape: tuple[int, ...]


def _recover_structured_grid(X: np.ndarray) -> StructuredGridInfo:
    """Recover a Cartesian grid description from sample coordinates."""

    X = _as_2d_array(X)
    axes = tuple(np.unique(X[:, dim]) for dim in range(X.shape[1]))
    grid_shape = tuple(len(axis) for axis in axes)

    if np.prod(grid_shape, dtype=int) != X.shape[0]:
        raise ValueError(
            "Training samples do not form a Cartesian product grid: "
            f"expected {np.prod(grid_shape, dtype=int)} samples, got {X.shape[0]}."
        )

    seen = np.zeros(grid_shape, dtype=bool)
    for row in X:
        indices = []
        for axis, value in zip(axes, row):
            matches = np.flatnonzero(np.isclose(axis, value, rtol=1e-12, atol=1e-12))
            if matches.size != 1:
                raise ValueError("Training samples do not lie on a unique structured grid.")
            indices.append(int(matches[0]))

        index_tuple = tuple(indices)
        if seen[index_tuple]:
            raise ValueError("Training samples contain duplicate grid points.")
        seen[index_tuple] = True

    if not np.all(seen):
        raise ValueError("Training samples do not cover the full Cartesian grid.")

    return StructuredGridInfo(axes=axes, grid_shape=grid_shape)


def _reshape_outputs_on_grid(Y: np.ndarray, grid: StructuredGridInfo, X: np.ndarray) -> list[np.ndarray]:
    """Reshape each output column onto the recovered structured grid."""

    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError(f"Expected Y to be 1D or 2D, got shape {Y.shape}.")
    if Y.shape[0] != X.shape[0]:
        raise ValueError("X and Y must contain the same number of samples.")

    reshaped_outputs = [np.empty(grid.grid_shape, dtype=float) for _ in range(Y.shape[1])]

    axis_maps = []
    for axis in grid.axes:
        axis_maps.append({float(value): idx for idx, value in enumerate(axis)})

    for sample_idx, row in enumerate(X):
        indices = tuple(axis_maps[dim][float(value)] for dim, value in enumerate(row))
        for output_idx in range(Y.shape[1]):
            reshaped_outputs[output_idx][indices] = Y[sample_idx, output_idx]

    return reshaped_outputs


@dataclass
class SplineSurrogate(BaseSurrogate):
    """Multivariate spline surrogate based on structured-grid interpolation."""

    backend: str = "RegularGridInterpolator"
    bounds_error: bool = False
    fill_value: float | None = None
    interpolators: list[Any] = field(default_factory=list)
    grid: StructuredGridInfo | None = None

    name: str = "Spline"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> SplineSurrogate:
        """Fit one interpolator per output column on a structured grid."""

        X_array = _as_2d_array(X)
        Y_array = np.asarray(Y, dtype=float)
        if Y_array.ndim == 1:
            Y_array = Y_array.reshape(-1, 1)

        self.grid = _recover_structured_grid(X_array)
        reshaped_outputs = _reshape_outputs_on_grid(Y_array, self.grid, X_array)

        if self.backend != "RegularGridInterpolator":
            raise ValueError(f"Unsupported spline backend: {self.backend}")

        self.interpolators = [
            RegularGridInterpolator(
                self.grid.axes,
                reshaped,
                bounds_error=self.bounds_error,
                fill_value=self.fill_value,
            )
            for reshaped in reshaped_outputs
        ]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the fitted interpolators at the given points."""

        if not self.interpolators or self.grid is None:
            raise RuntimeError("SplineSurrogate has not been fitted.")

        X_array = _as_2d_array(X)
        outputs = [np.asarray(interpolator(X_array), dtype=float).reshape(-1) for interpolator in self.interpolators]
        return np.column_stack(outputs)


def train_spline_for_cadet(
    cp: np.ndarray,
    cs: np.ndarray,
    *,
    backend: str = "RegularGridInterpolator",
    bounds_error: bool = False,
    fill_value: float | None = None,
) -> dict[str, Any]:
    """Train a spline surrogate and return a CADET-like result dictionary."""

    start = time.perf_counter()
    model = SplineSurrogate(
        backend=backend,
        bounds_error=bounds_error,
        fill_value=fill_value,
    ).fit(cp, cs)
    training_time = time.perf_counter() - start

    cs_array = np.asarray(cs, dtype=float)
    if cs_array.ndim == 1:
        cs_array = cs_array.reshape(-1, 1)

    metadata = {
        "backend": backend,
        "bounds_error": bounds_error,
        "fill_value": fill_value,
        "n_inputs": int(len(model.grid.axes) if model.grid is not None else 0),
        "n_outputs": int(cs_array.shape[1]),
        "grid_shape": model.grid.grid_shape if model.grid is not None else (),
    }

    return {
        "model": model,
        "training_time": training_time,
        "metadata": metadata,
    }
