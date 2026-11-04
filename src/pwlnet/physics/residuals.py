"""Physics-residual framework for PWLNet.

A :class:`PhysicsResidual` is a reusable object representing a physical
consistency condition ``r(x, y)`` on a surrogate that maps inputs ``x``
to outputs ``y`` (both in ORIGINAL units). The same object is meant to
be used in several backends:

* **Gradient training** -- as a differentiable soft penalty added to the
  Keras loss (see :mod:`pwlnet.objectives.physics_loss` and
  :class:`pwlnet.training.physics_informed.PhysicsInformedTrainer`).
* **Reporting / checking** -- evaluated on NumPy arrays.
* **Optimization** (later phases) -- emitted as Pyomo expressions or
  SOSX-node constraints.

To keep that reuse tractable the first-target residuals are **affine**:
``r = x @ W_in^T + y @ W_out^T + b``. Affine residuals expose their
coefficient matrices via :meth:`affine_matrices`, so a single description
drives the NumPy evaluation, the differentiable Keras penalty, and (in
later phases) a linear Pyomo constraint. The architecture document calls
these the "affine algebraic residuals" that are the first implementation
target.

This module is pure NumPy and must not import keras/tensorflow/pyomo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pwlnet.utils.typing import ArrayLike

__all__ = ["PhysicsResidual", "RESIDUAL_KINDS"]

#: A residual is an equality condition (``r == 0`` desired) or an
#: inequality condition (``r <= 0`` desired; only positive ``r`` is a
#: violation).
RESIDUAL_KINDS = ("equality", "inequality")


class PhysicsResidual(ABC):
    """Abstract physical-consistency condition ``r(x, y)``.

    Parameters
    ----------
    name : str
        Human-readable identifier used in reports.
    kind : {"equality", "inequality"}
        ``"equality"`` penalizes ``|r|``; ``"inequality"`` penalizes only
        the positive part ``max(r, 0)`` (the condition is ``r <= 0``).

    Notes
    -----
    Subclasses implement :meth:`residual`. Affine subclasses also set
    :attr:`is_affine` and implement :meth:`affine_matrices`, which lets
    the objective layer build a differentiable penalty without a
    backend-specific reimplementation of the residual.
    """

    is_affine: bool = False

    def __init__(self, name: str, kind: str = "equality") -> None:
        kind = str(kind).strip().lower()
        if kind not in RESIDUAL_KINDS:
            raise ValueError(f"kind must be one of {RESIDUAL_KINDS}, got {kind!r}.")
        self.name = str(name)
        self.kind = kind

    # ------------------------------------------------------------------
    @abstractmethod
    def residual(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Signed residual values, shape ``(n_samples, n_residuals)``.

        ``x`` and ``y`` are in ORIGINAL units, shape ``(n_samples,
        n_inputs)`` / ``(n_samples, n_outputs)`` (1D arrays are treated
        as a single sample).
        """

    @property
    @abstractmethod
    def n_residuals(self) -> int:
        """Number of scalar residual components produced per sample."""

    # ------------------------------------------------------------------
    def affine_matrices(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        """Return ``(W_in, W_out, b)`` for an affine residual.

        ``W_in`` has shape ``(n_residuals, n_inputs)`` (or ``None`` when
        the residual does not depend on inputs), ``W_out`` has shape
        ``(n_residuals, n_outputs)`` (or ``None``), and ``b`` has shape
        ``(n_residuals,)``. The residual is
        ``x @ W_in^T + y @ W_out^T + b``.

        Raises
        ------
        NotImplementedError
            For non-affine residuals.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is not affine; affine_matrices() is unavailable."
        )

    # ------------------------------------------------------------------
    def violation(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Non-negative violation array.

        ``|r|`` for equality residuals, ``max(r, 0)`` for inequality
        residuals. Shape ``(n_samples, n_residuals)``.
        """
        r = np.atleast_2d(np.asarray(self.residual(x, y), dtype=float))
        return np.abs(r) if self.kind == "equality" else np.maximum(r, 0.0)

    def evaluate(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Alias for :meth:`residual` (signed values), as a 2D array."""
        return np.atleast_2d(np.asarray(self.residual(x, y), dtype=float))

    def max_violation(self, x: ArrayLike, y: ArrayLike) -> float:
        """Largest single violation over all samples and components."""
        v = self.violation(x, y)
        return float(np.max(v)) if v.size else 0.0

    def mean_violation(self, x: ArrayLike, y: ArrayLike) -> float:
        """Mean violation over all samples and components."""
        v = self.violation(x, y)
        return float(np.mean(v)) if v.size else 0.0

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, kind={self.kind!r}, "
            f"n_residuals={self.n_residuals})"
        )
