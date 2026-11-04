"""Output-bound and non-negativity physics residuals (affine, inequality).

These encode known physical limits on the surrogate outputs -- e.g.
mass fractions in ``[0, 1]`` -- as inequality residuals (``r <= 0``
desired) so they can be added as soft penalties during training or as
linear constraints later.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pwlnet.physics.algebraic import AffineResidual

__all__ = [
    "upper_bound_residual",
    "lower_bound_residual",
    "nonnegativity_residual",
    "box_residual",
]


def _select(n_outputs: int, output_indices: Sequence[int] | None) -> list[int]:
    return list(range(n_outputs)) if output_indices is None else [int(j) for j in output_indices]


def upper_bound_residual(
    n_outputs: int,
    upper: float | Sequence[float],
    output_indices: Sequence[int] | None = None,
    *,
    name: str = "upper_bound",
) -> AffineResidual:
    """Residual ``y[idx] - upper <= 0`` (one row per selected output)."""
    idx = _select(n_outputs, output_indices)
    ub = np.broadcast_to(np.asarray(upper, dtype=float).reshape(-1), (len(idx),))
    w_out = np.zeros((len(idx), n_outputs))
    for row, j in enumerate(idx):
        w_out[row, j] = 1.0
    return AffineResidual(None, w_out, -ub, kind="inequality", name=name)


def lower_bound_residual(
    n_outputs: int,
    lower: float | Sequence[float],
    output_indices: Sequence[int] | None = None,
    *,
    name: str = "lower_bound",
) -> AffineResidual:
    """Residual ``lower - y[idx] <= 0`` (one row per selected output)."""
    idx = _select(n_outputs, output_indices)
    lb = np.broadcast_to(np.asarray(lower, dtype=float).reshape(-1), (len(idx),))
    w_out = np.zeros((len(idx), n_outputs))
    for row, j in enumerate(idx):
        w_out[row, j] = -1.0
    return AffineResidual(None, w_out, lb, kind="inequality", name=name)


def nonnegativity_residual(
    n_outputs: int,
    output_indices: Sequence[int] | None = None,
    *,
    name: str = "nonnegativity",
) -> AffineResidual:
    """Residual ``-y[idx] <= 0`` (selected outputs must be non-negative)."""
    return lower_bound_residual(n_outputs, 0.0, output_indices, name=name)


def box_residual(
    n_outputs: int,
    lower: float | Sequence[float],
    upper: float | Sequence[float],
    output_indices: Sequence[int] | None = None,
    *,
    name: str = "box",
) -> AffineResidual:
    """Two-sided ``lower <= y[idx] <= upper`` as a stacked inequality residual."""
    idx = _select(n_outputs, output_indices)
    lb = np.broadcast_to(np.asarray(lower, dtype=float).reshape(-1), (len(idx),))
    ub = np.broadcast_to(np.asarray(upper, dtype=float).reshape(-1), (len(idx),))
    rows = 2 * len(idx)
    w_out = np.zeros((rows, n_outputs))
    b = np.zeros(rows)
    for k, j in enumerate(idx):
        w_out[k, j] = 1.0          # y - ub <= 0
        b[k] = -ub[k]
        w_out[len(idx) + k, j] = -1.0  # lb - y <= 0
        b[len(idx) + k] = lb[k]
    return AffineResidual(None, w_out, b, kind="inequality", name=name)
