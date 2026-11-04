"""Physics-residual loss terms for gradient training.

A :class:`PhysicsLoss` wraps a :class:`pwlnet.physics.PhysicsResidual`
with a reduction (``mse``/``mae``) and a weight. It produces both a NumPy
value (for reporting) and -- for affine residuals -- a differentiable
Keras penalty used by
:class:`pwlnet.training.physics_informed.PhysicsInformedTrainer`.

This module is import-time free of Keras: the Keras penalty receives the
``keras.ops`` namespace from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pwlnet.physics.residuals import PhysicsResidual

__all__ = ["PhysicsLoss", "SUPPORTED_PHYSICS_REDUCTIONS"]

SUPPORTED_PHYSICS_REDUCTIONS = ("mse", "mae")


@dataclass
class PhysicsLoss:
    """A weighted physics-residual penalty.

    Parameters
    ----------
    residual : PhysicsResidual
        The physical-consistency condition to penalize.
    loss : {"mse", "mae"}, optional
        Reduction of the effective residual (``r`` for equality,
        ``max(r, 0)`` for inequality). Default ``"mse"``.
    weight : float, optional
        Multiplier in the total training objective. Default ``1.0``.
    name : str, optional
        Identifier; defaults to the residual's name.
    """

    residual: PhysicsResidual
    loss: str = "mse"
    weight: float = 1.0
    name: str = ""
    _: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        red = str(self.loss).strip().lower()
        if red not in SUPPORTED_PHYSICS_REDUCTIONS:
            raise ValueError(
                f"loss must be one of {SUPPORTED_PHYSICS_REDUCTIONS}, got {self.loss!r}."
            )
        self.loss = red
        self.weight = float(self.weight)
        if not self.name:
            self.name = self.residual.name

    # ------------------------------------------------------------------
    def _effective(self, r: np.ndarray) -> np.ndarray:
        return r if self.residual.kind == "equality" else np.maximum(r, 0.0)

    def numpy_value(self, x: Any, y: Any) -> float:
        """Reduced penalty (unweighted) on NumPy arrays."""
        eff = self._effective(np.atleast_2d(np.asarray(self.residual.residual(x, y), float)))
        if eff.size == 0:
            return 0.0
        return float(np.mean(eff**2) if self.loss == "mse" else np.mean(np.abs(eff)))

    def weighted_numpy_value(self, x: Any, y: Any) -> float:
        """``weight * numpy_value``."""
        return self.weight * self.numpy_value(x, y)

    # ------------------------------------------------------------------
    def keras_term(self, x_orig: Any, y_orig: Any, *, ops: Any) -> Any:
        """Differentiable unweighted penalty for an affine residual.

        Parameters
        ----------
        x_orig, y_orig : tensor
            Inputs/outputs in ORIGINAL units (Keras/TF tensors).
        ops : module
            The ``keras.ops`` namespace (passed in so this module needs
            no Keras at import time).

        Returns
        -------
        tensor
            Scalar penalty tensor.

        Raises
        ------
        TypeError
            If the residual is not affine (only affine residuals support
            the gradient-training penalty in this phase).
        """
        if not self.residual.is_affine:
            raise TypeError(
                f"PhysicsLoss.keras_term requires an affine residual; "
                f"{type(self.residual).__name__} is not affine."
            )
        w_in, w_out, b = self.residual.affine_matrices()
        dtype = getattr(y_orig, "dtype", "float32")
        r = ops.convert_to_tensor(np.asarray(b, dtype=np.float32))
        r = ops.cast(r, dtype)
        if w_in is not None:
            win = ops.cast(ops.convert_to_tensor(np.asarray(w_in, np.float32)), dtype)
            r = r + ops.matmul(x_orig, ops.transpose(win))
        if w_out is not None:
            wout = ops.cast(ops.convert_to_tensor(np.asarray(w_out, np.float32)), dtype)
            r = r + ops.matmul(y_orig, ops.transpose(wout))
        eff = r if self.residual.kind == "equality" else ops.maximum(r, 0.0)
        return ops.mean(ops.square(eff)) if self.loss == "mse" else ops.mean(ops.abs(eff))
