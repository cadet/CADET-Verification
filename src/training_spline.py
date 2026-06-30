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


def build_spline_surrogate(training_results, backend="RegularGridInterpolator", bounds_error=False, fill_value=None):

    """Build a spline surrogate from training data."""
    
    cp = np.column_stack([training_results[key] for key in training_results if key.startswith("CP_VALS_COMP_")])
    cs = np.column_stack([training_results[key] for key in training_results if "_BND_" in key])

    return SplineSurrogate(
        backend=backend,
        bounds_error=bounds_error,
        fill_value=fill_value,
    ).fit(cp, cs)


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

    cp = np.asarray(cp, dtype=float)
    cs = np.asarray(cs, dtype=float)

    result = { }

    # CP slices (per-component, used by build_spline_surrogate on the Python side)
    if cp.ndim == 1:
        result["CP_VALS_COMP_000"] = cp
    else:
        for comp in range(cp.shape[1]):
            result[f"CP_VALS_COMP_{comp:03d}"] = cp[:, comp]

    # CS slices (per-component, used by build_spline_surrogate on the Python side)
    if cs.ndim == 1:
        result["CS_VALS_COMP_000_BND_000"] = cs

    elif cs.ndim == 2:
        # Assume one boundary per component, columns = components
        for comp in range(cs.shape[1]):
            result[f"CS_VALS_COMP_{comp:03d}_BND_000"] = cs[:, comp]

    elif cs.ndim == 3:
        # Shape: (n_points, n_comp, n_bound)
        for comp in range(cs.shape[1]):
            for bnd in range(cs.shape[2]):
                result[f"CS_VALS_COMP_{comp:03d}_BND_{bnd:03d}"] = cs[:, comp, bnd]

    else:
        raise ValueError(f"Unsupported cs shape {cs.shape}")

    # For multicomponent data, also emit the combined interleaved CP_VALS / CS_VALS arrays.
    # The C++ SplineBinding competitive mode reads these keys and performs multilinear
    # interpolation on the structured Cartesian grid, which exactly mirrors
    # RegularGridInterpolator(method='linear').  The combined format is:
    #   CP_VALS[sample * nComp + comp]  = pore-phase concentration of component comp for sample
    #   CS_VALS[sample * nBound + bnd]  = solid-phase concentration of bound state bnd for sample
    # When CP_VALS is present the C++ ignores the per-component CP_VALS_COMP_* keys, so
    # including both sets of keys is safe.
    if cp.ndim == 2 and cp.shape[1] > 1:
        # cp shape: (n_samples, n_comp) → ravel to [c0_s0, c1_s0, c0_s1, c1_s1, ...]
        result["CP_VALS"] = cp.ravel()

        if cs.ndim == 2:
            # cs shape: (n_samples, n_bound_total) → ravel to [q0_s0, q1_s0, q0_s1, q1_s1, ...]
            result["CS_VALS"] = cs.ravel()
        elif cs.ndim == 3:
            # cs shape: (n_samples, n_comp, n_bound_per_comp)
            # Flatten inner two dims in C-order: bound states are ordered comp0/bnd0, comp0/bnd1, ..., comp1/bnd0, ...
            result["CS_VALS"] = cs.reshape(cs.shape[0], -1).ravel()

    return result
