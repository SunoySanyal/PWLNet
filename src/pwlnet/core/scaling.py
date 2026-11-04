"""Offset/factor scaling transforms for network inputs and outputs.

The scaled representation used throughout PWLNet is

.. math:: x_\\text{scaled} = (x - \\text{offset}) / \\text{factor}

which covers standardization (mean/std), min-max scaling, and the
identity transform. The same convention matches OMLT's
``OffsetScaling``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.utils.typing import ArrayLike
from pwlnet.utils.validation import as_2d_array

__all__ = ["OffsetScaler"]


class OffsetScaler:
    """Per-feature affine scaler: ``x_scaled = (x - offset) / factor``.

    Parameters
    ----------
    offset : numpy.ndarray
        Per-feature offsets, shape ``(n_features,)``.
    factor : numpy.ndarray
        Per-feature scale factors, shape ``(n_features,)``. Must be
        nonzero; positive factors preserve bound ordering.
    method : str, optional
        Name of the fitting method that produced this scaler (purely
        informational; one of ``"standard"``, ``"minmax"``, ``"none"``
        or ``"manual"``).

    Attributes
    ----------
    offset : numpy.ndarray
        Per-feature offsets, shape ``(n_features,)``.
    factor : numpy.ndarray
        Per-feature factors, shape ``(n_features,)``.
    """

    def __init__(self, offset: ArrayLike, factor: ArrayLike, method: str = "manual") -> None:
        self.offset: np.ndarray = np.atleast_1d(np.asarray(offset, dtype=float)).ravel()
        self.factor: np.ndarray = np.atleast_1d(np.asarray(factor, dtype=float)).ravel()
        self.method: str = method
        if self.offset.shape != self.factor.shape:
            raise ValueError(
                f"offset shape {self.offset.shape} != factor shape {self.factor.shape}."
            )
        if np.any(self.factor == 0.0):
            raise ValueError("Scaling factors must be nonzero.")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, data: ArrayLike, method: str = "standard") -> "OffsetScaler":
        """Fit a scaler to data.

        Parameters
        ----------
        data : array_like
            Data of shape ``(n_samples, n_features)`` (a 1D array is
            treated as a single sample).
        method : {"standard", "minmax", "none"}, optional
            * ``"standard"`` -- offset = mean, factor = std (population
              std, ``ddof=0``); features with zero std get factor 1.
            * ``"minmax"`` -- offset = min, factor = max - min (scaled
              data lies in [0, 1]); constant features get factor 1.
            * ``"none"`` -- offset = 0, factor = 1 (identity).

        Returns
        -------
        OffsetScaler
            The fitted scaler.

        Raises
        ------
        ValueError
            If the method name is unknown.
        """
        arr = as_2d_array(data, name="data")
        if method == "standard":
            offset = arr.mean(axis=0)
            factor = arr.std(axis=0)
            factor = np.where(factor == 0.0, 1.0, factor)
        elif method == "minmax":
            offset = arr.min(axis=0)
            factor = arr.max(axis=0) - offset
            factor = np.where(factor == 0.0, 1.0, factor)
        elif method == "none":
            offset = np.zeros(arr.shape[1])
            factor = np.ones(arr.shape[1])
        else:
            raise ValueError(
                f"Unknown scaling method '{method}'. Use 'standard', 'minmax' or 'none'."
            )
        return cls(offset=offset, factor=factor, method=method)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        """Number of features this scaler applies to."""
        return int(self.offset.shape[0])

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale data: ``(x - offset) / factor``.

        Parameters
        ----------
        x : array_like
            Array of shape ``(n_samples, n_features)`` or
            ``(n_features,)``.

        Returns
        -------
        numpy.ndarray
            Scaled array with the same shape as the input.
        """
        arr = np.asarray(x, dtype=float)
        return (arr - self.offset) / self.factor

    def inverse_transform(self, x: ArrayLike) -> np.ndarray:
        """Undo scaling: ``x * factor + offset``.

        Parameters
        ----------
        x : array_like
            Scaled array of shape ``(n_samples, n_features)`` or
            ``(n_features,)``.

        Returns
        -------
        numpy.ndarray
            Array in the original units, same shape as the input.
        """
        arr = np.asarray(x, dtype=float)
        return arr * self.factor + self.offset

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Keys ``"offset"``, ``"factor"`` (lists of floats) and
            ``"method"``.
        """
        return {
            "offset": self.offset.tolist(),
            "factor": self.factor.tolist(),
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OffsetScaler":
        """Reconstruct a scaler from :meth:`to_dict` output.

        Parameters
        ----------
        d : dict
            Dictionary with keys ``"offset"`` and ``"factor"`` (and
            optionally ``"method"``).

        Returns
        -------
        OffsetScaler
            The reconstructed scaler.
        """
        return cls(
            offset=np.asarray(d["offset"], dtype=float),
            factor=np.asarray(d["factor"], dtype=float),
            method=d.get("method", "manual"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OffsetScaler):
            return NotImplemented
        return (
            np.array_equal(self.offset, other.offset)
            and np.array_equal(self.factor, other.factor)
        )

    def __repr__(self) -> str:
        return (
            f"OffsetScaler(n_features={self.n_features}, method='{self.method}')"
        )
