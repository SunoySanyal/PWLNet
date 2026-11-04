"""Piecewise-linear function data used by manual Pyomo formulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pwlnet.formulations.pyomo_pwl.breakpoints import (
    get_tanh_breakpoints,
    validate_breakpoints,
)
from pwlnet.utils.typing import ArrayLike

__all__ = [
    "PiecewiseLinearData",
    "build_piecewise_linear_data",
    "build_tanh_pwl_data",
    "extend_piecewise_linear_data",
]


@dataclass(frozen=True)
class PiecewiseLinearData:
    """Immutable univariate piecewise-linear approximation data.

    Attributes
    ----------
    breakpoints : numpy.ndarray
        Strictly increasing x-coordinates.
    values : numpy.ndarray
        Function values at each breakpoint.
    slopes : numpy.ndarray
        Segment slopes, one per interval.
    intercepts : numpy.ndarray
        Segment intercepts, one per interval.
    name : str
        Human-readable function name.
    """

    breakpoints: np.ndarray
    values: np.ndarray
    slopes: np.ndarray
    intercepts: np.ndarray
    name: str = "piecewise_linear"

    @property
    def n_breakpoints(self) -> int:
        """Number of breakpoints."""
        return int(self.breakpoints.size)

    @property
    def n_segments(self) -> int:
        """Number of linear segments."""
        return int(self.slopes.size)

    @property
    def domain(self) -> tuple[float, float]:
        """Closed approximation domain."""
        return (float(self.breakpoints[0]), float(self.breakpoints[-1]))

    def segment_index(self, x: ArrayLike) -> np.ndarray:
        """Return segment indices for values clipped to the domain."""
        arr = np.asarray(x, dtype=float)
        clipped = np.clip(arr, self.breakpoints[0], self.breakpoints[-1])
        idx = np.searchsorted(self.breakpoints, clipped, side="right") - 1
        return np.clip(idx, 0, self.n_segments - 1)

    def evaluate(self, x: ArrayLike, *, clip: bool = True) -> np.ndarray:
        """Evaluate the PWL function.

        Parameters
        ----------
        x : array_like
            Points where the approximation is evaluated.
        clip : bool, optional
            If true, values outside the domain are clipped to the nearest
            endpoint. If false, out-of-domain values raise ``ValueError``.

        Returns
        -------
        numpy.ndarray
            Approximation values with the same shape as ``x``.
        """
        arr = np.asarray(x, dtype=float)
        if not clip and (
            np.any(arr < self.breakpoints[0]) or np.any(arr > self.breakpoints[-1])
        ):
            raise ValueError(
                f"Cannot evaluate {self.name} outside domain {self.domain}."
            )
        eval_x = np.clip(arr, self.breakpoints[0], self.breakpoints[-1]) if clip else arr
        idx = self.segment_index(eval_x)
        return self.slopes[idx] * eval_x + self.intercepts[idx]

    def max_abs_error(
        self,
        fn,
        *,
        n_grid: int = 10001,
    ) -> float:
        """Estimate max absolute approximation error on a uniform grid."""
        grid = np.linspace(self.breakpoints[0], self.breakpoints[-1], int(n_grid))
        return float(np.max(np.abs(self.evaluate(grid) - fn(grid))))


def build_piecewise_linear_data(
    breakpoints: ArrayLike,
    values: ArrayLike,
    *,
    name: str = "piecewise_linear",
) -> PiecewiseLinearData:
    """Build segment slopes/intercepts from breakpoint and value vectors."""
    x = validate_breakpoints(breakpoints, name=f"{name} breakpoints")
    y = np.asarray(values, dtype=float).ravel()
    if y.shape != x.shape:
        raise ValueError(
            f"{name} values must have shape {x.shape}, got {y.shape}."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name} values must contain only finite values.")
    dx = np.diff(x)
    slopes = np.diff(y) / dx
    intercepts = y[:-1] - slopes * x[:-1]
    return PiecewiseLinearData(
        breakpoints=x,
        values=y.copy(),
        slopes=slopes,
        intercepts=intercepts,
        name=name,
    )


def extend_piecewise_linear_data(
    pwl: PiecewiseLinearData,
    lower: float,
    upper: float,
) -> PiecewiseLinearData:
    """Extend a PWL function with flat segments to cover ``[lower, upper]``.

    Pyomo PWL-activation blocks constrain the pre-activation variable to
    the breakpoint domain. When interval bound propagation shows the
    pre-activation can leave that domain, clamping the variable would
    make feasible operating points infeasible. This helper instead
    extends the function with constant (saturation) segments holding the
    endpoint values, matching the clipping behavior of
    :meth:`PiecewiseLinearData.evaluate`.

    Parameters
    ----------
    pwl : PiecewiseLinearData
        Base approximation data.
    lower, upper : float
        Required closed domain. Extension happens only on the sides
        that actually exceed the current domain.

    Returns
    -------
    PiecewiseLinearData
        The same data when ``[lower, upper]`` is already covered,
        otherwise a copy with flat end segments appended.
    """
    lo, hi = float(lower), float(upper)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(
            f"Cannot extend {pwl.name} to a non-finite domain [{lo}, {hi}]."
        )
    x = pwl.breakpoints
    y = pwl.values
    if lo >= x[0] and hi <= x[-1]:
        return pwl
    if lo < x[0]:
        x = np.concatenate(([lo], x))
        y = np.concatenate(([y[0]], y))
    if hi > x[-1]:
        x = np.concatenate((x, [hi]))
        y = np.concatenate((y, [y[-1]]))
    return build_piecewise_linear_data(x, y, name=f"{pwl.name}_extended")


def build_tanh_pwl_data(
    pieces: int | ArrayLike = 5,
    *,
    require_tanh_domain: bool = True,
) -> PiecewiseLinearData:
    """Build PWL approximation data for ``tanh``."""
    breakpoints = get_tanh_breakpoints(
        pieces,
        require_tanh_domain=require_tanh_domain,
    )
    return build_piecewise_linear_data(
        breakpoints,
        np.tanh(breakpoints),
        name=f"tanh_pwl_{breakpoints.size - 1}_pieces",
    )
