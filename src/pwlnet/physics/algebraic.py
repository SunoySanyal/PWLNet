"""Affine algebraic physics residuals.

``AffineResidual`` is the workhorse residual: ``r = x @ W_in^T +
y @ W_out^T + b``. Mass balances, sum-to-one closure, linear bounds and
most first-target physics conditions are special cases. Either weight
matrix may be ``None`` (no dependence on inputs / outputs).
"""

from __future__ import annotations

import numpy as np

from pwlnet.physics.residuals import PhysicsResidual
from pwlnet.utils.typing import ArrayLike

__all__ = ["AffineResidual", "linear_residual"]


def _as_matrix(value: ArrayLike | None, n_res: int | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}.")
    if n_res is not None and arr.shape[0] != n_res:
        raise ValueError(
            f"{name} has {arr.shape[0]} rows but {n_res} residuals were expected."
        )
    return arr


class AffineResidual(PhysicsResidual):
    """Affine residual ``r = x @ W_in^T + y @ W_out^T + b``.

    Parameters
    ----------
    w_in : array_like or None
        Input coefficients, shape ``(n_residuals, n_inputs)``. ``None``
        means the residual does not use the inputs.
    w_out : array_like or None
        Output coefficients, shape ``(n_residuals, n_outputs)``.
        ``None`` means the residual does not use the outputs.
    b : array_like or float, optional
        Constant term, shape ``(n_residuals,)`` (a scalar is broadcast).
        Default ``0``.
    kind : {"equality", "inequality"}, optional
        Residual kind. Default ``"equality"``.
    name : str, optional
        Identifier. Default ``"affine"``.

    Raises
    ------
    ValueError
        If both ``w_in`` and ``w_out`` are ``None``, or the row counts of
        ``w_in``, ``w_out`` and ``b`` are inconsistent.
    """

    is_affine = True

    def __init__(
        self,
        w_in: ArrayLike | None,
        w_out: ArrayLike | None,
        b: ArrayLike | float = 0.0,
        *,
        kind: str = "equality",
        name: str = "affine",
    ) -> None:
        super().__init__(name=name, kind=kind)
        w_in = _as_matrix(w_in, None, "w_in")
        w_out = _as_matrix(w_out, None, "w_out")
        if w_in is None and w_out is None:
            raise ValueError("AffineResidual needs at least one of w_in, w_out.")
        n_res = w_in.shape[0] if w_in is not None else w_out.shape[0]
        if w_in is not None and w_out is not None and w_in.shape[0] != w_out.shape[0]:
            raise ValueError(
                f"w_in has {w_in.shape[0]} rows but w_out has {w_out.shape[0]}."
            )
        b_arr = np.asarray(b, dtype=float).reshape(-1)
        if b_arr.size == 1:
            b_arr = np.full(n_res, float(b_arr[0]))
        if b_arr.shape[0] != n_res:
            raise ValueError(f"b has {b_arr.shape[0]} entries but {n_res} residuals.")
        self.w_in = w_in
        self.w_out = w_out
        self.b = b_arr

    @property
    def n_residuals(self) -> int:
        return int(self.b.shape[0])

    def affine_matrices(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        return self.w_in, self.w_out, self.b

    def residual(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        out = np.broadcast_to(self.b, (1, self.n_residuals)).astype(float).copy()
        if self.w_in is not None:
            xa = np.atleast_2d(np.asarray(x, dtype=float))
            if xa.shape[1] != self.w_in.shape[1]:
                raise ValueError(
                    f"x has {xa.shape[1]} inputs but residual expects {self.w_in.shape[1]}."
                )
            out = out + xa @ self.w_in.T
        if self.w_out is not None:
            ya = np.atleast_2d(np.asarray(y, dtype=float))
            if ya.shape[1] != self.w_out.shape[1]:
                raise ValueError(
                    f"y has {ya.shape[1]} outputs but residual expects {self.w_out.shape[1]}."
                )
            out = out + ya @ self.w_out.T
        return out


def linear_residual(
    *,
    n_inputs: int,
    n_outputs: int,
    input_coeffs: dict[int, float] | None = None,
    output_coeffs: dict[int, float] | None = None,
    const: float = 0.0,
    kind: str = "equality",
    name: str = "linear",
) -> AffineResidual:
    """Build a single affine residual from sparse coefficient maps.

    Convenience for one scalar condition such as
    ``coeff_a*x[a] + coeff_p*y[p] + const (== or <=) 0``.

    Parameters
    ----------
    n_inputs, n_outputs : int
        Surrogate input/output widths.
    input_coeffs, output_coeffs : dict of int to float or None
        Sparse coefficients keyed by input/output index.
    const : float, optional
        Constant term.
    kind : {"equality", "inequality"}, optional
    name : str, optional

    Returns
    -------
    AffineResidual
        A residual with ``n_residuals == 1``.
    """
    w_in = None
    if input_coeffs:
        w_in = np.zeros((1, n_inputs))
        for idx, c in input_coeffs.items():
            w_in[0, int(idx)] = float(c)
    w_out = None
    if output_coeffs:
        w_out = np.zeros((1, n_outputs))
        for idx, c in output_coeffs.items():
            w_out[0, int(idx)] = float(c)
    if w_in is None and w_out is None:
        raise ValueError("linear_residual needs input_coeffs or output_coeffs.")
    return AffineResidual(w_in, w_out, float(const), kind=kind, name=name)
