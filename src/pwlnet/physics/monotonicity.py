"""Monotonicity physics constraints.

Monotonicity (``dy_out/dx_in`` should keep a fixed sign) cannot be
written as a residual of ``(x, y)`` alone -- it needs the surrogate's
input-output sensitivity. It is therefore represented as a lightweight
:class:`MonotonicityConstraint` specification that the physics-informed
trainer consumes via automatic differentiation, and that NumPy code can
check against a finite-difference Jacobian.

This module is pure NumPy and imports no deep-learning backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pwlnet.utils.typing import ArrayLike

__all__ = ["MonotonicityConstraint", "finite_difference_violation"]


@dataclass(frozen=True)
class MonotonicityConstraint:
    """Require output ``output_index`` to be monotone in input ``input_index``.

    Attributes
    ----------
    input_index : int
        Index of the input the sensitivity is taken with respect to.
    output_index : int
        Index of the output that must be monotone.
    sign : int
        ``+1`` for non-decreasing, ``-1`` for non-increasing.
    weight : float
        Per-constraint penalty weight used by the trainer. Default 1.0.
    name : str
        Identifier. Defaults to ``"mono[out<-in]"``.
    """

    input_index: int
    output_index: int
    sign: int = 1
    weight: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if int(self.sign) not in (-1, 1):
            raise ValueError(f"sign must be +1 or -1, got {self.sign!r}.")
        object.__setattr__(self, "input_index", int(self.input_index))
        object.__setattr__(self, "output_index", int(self.output_index))
        object.__setattr__(self, "sign", int(self.sign))
        object.__setattr__(self, "weight", float(self.weight))
        if not self.name:
            arrow = "up" if self.sign > 0 else "down"
            object.__setattr__(
                self, "name", f"mono_{arrow}[y{self.output_index}<-x{self.input_index}]"
            )


def finite_difference_violation(
    predict_fn,
    x: ArrayLike,
    constraint: MonotonicityConstraint,
    *,
    delta: float = 1e-3,
) -> np.ndarray:
    """Per-sample monotonicity violation via a forward finite difference.

    For each row of ``x`` the input ``constraint.input_index`` is nudged
    by ``delta`` and the signed sensitivity ``sign * (y(x+δ) - y(x))`` is
    formed; the violation is ``max(0, -signed_sensitivity / delta)`` so a
    wrong-sign slope is positive.

    Parameters
    ----------
    predict_fn : callable
        Maps an array ``(n, n_inputs)`` to outputs ``(n, n_outputs)`` in
        ORIGINAL units (e.g. ``NetworkDefinition.predict``).
    x : array_like
        Points to check, shape ``(n, n_inputs)``.
    constraint : MonotonicityConstraint
    delta : float, optional
        Finite-difference step in the input's own units. Default ``1e-3``.

    Returns
    -------
    numpy.ndarray
        Non-negative violations, shape ``(n,)``.
    """
    xa = np.atleast_2d(np.asarray(x, dtype=float)).copy()
    base = np.asarray(predict_fn(xa), dtype=float)
    bumped = xa.copy()
    bumped[:, constraint.input_index] += float(delta)
    nxt = np.asarray(predict_fn(bumped), dtype=float)
    slope = (nxt[:, constraint.output_index] - base[:, constraint.output_index]) / float(delta)
    return np.maximum(0.0, -constraint.sign * slope)
