"""Abstract base class for steady-state process benchmarks.

A *process* in PWLNet is a (possibly expensive, possibly implicit)
steady-state input/output map used to generate surrogate-training data
and to benchmark downstream optimization. Concrete processes (e.g. the
Williams-Otto reactor in :mod:`pwlnet.processes.williams_otto`)
implement :meth:`BaseProcess.simulate` for a single operating point and
inherit input sampling and dataset generation.

This module is framework-free: numpy only (pandas is imported lazily
inside :meth:`BaseProcess.to_dataframe`).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from pwlnet.utils.exceptions import PWLNetError
from pwlnet.utils.logging import get_logger
from pwlnet.utils.typing import ArrayLike, BoundsList

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["BaseProcess"]

_logger = get_logger(__name__)

_SAMPLING_METHODS = ("grid", "random", "lhs")


class BaseProcess(ABC):
    """Abstract steady-state process: a deterministic map ``inputs -> outputs``.

    Subclasses must provide :attr:`name`, :attr:`input_names`,
    :attr:`output_names` and :meth:`simulate`; they inherit
    :meth:`sample_inputs`, :meth:`generate_dataset` and
    :meth:`to_dataframe`.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier of the process (e.g. ``"williams_otto"``)."""

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """Ordered names of the process inputs (decision variables)."""

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Ordered names of the process outputs."""

    @abstractmethod
    def simulate(self, inputs: ArrayLike) -> np.ndarray:
        """Simulate a single operating point.

        Parameters
        ----------
        inputs : array_like, shape (n_inputs,)
            One operating point, ordered as :attr:`input_names`.

        Returns
        -------
        numpy.ndarray, shape (n_outputs,)
            Steady-state outputs ordered as :attr:`output_names`.

        Raises
        ------
        pwlnet.utils.exceptions.PWLNetError
            If the simulation does not converge at this point.
        """

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def n_inputs(self) -> int:
        """Number of process inputs."""
        return len(self.input_names)

    @property
    def n_outputs(self) -> int:
        """Number of process outputs."""
        return len(self.output_names)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_inputs(
        self,
        n: int,
        ranges: BoundsList,
        method: str = "lhs",
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample input points inside a box.

        Parameters
        ----------
        n : int
            Target number of samples. For ``method="grid"`` the actual
            count is ``ceil(n ** (1/d)) ** d`` (the smallest full
            tensor-product grid with at least ``n`` points), returned in
            row-major order so consecutive points are neighbours (useful
            for warm-started continuation).
        ranges : list of (float, float)
            Per-input ``(low, high)`` bounds, one pair per input.
        method : {"grid", "random", "lhs"}, optional
            Sampling design. ``"lhs"`` (default) uses a Latin hypercube
            (:class:`scipy.stats.qmc.LatinHypercube`), ``"random"`` is
            i.i.d. uniform, ``"grid"`` a full tensor-product grid.
        seed : int or None, optional
            Random seed for ``"random"`` and ``"lhs"`` (ignored for
            ``"grid"``).

        Returns
        -------
        numpy.ndarray, shape (n_points, d)
            Sampled input points.

        Raises
        ------
        ValueError
            On unknown ``method``, empty ``ranges``, ``n < 1``, or a
            range with ``low > high``.
        """
        if method not in _SAMPLING_METHODS:
            raise ValueError(
                f"Unknown sampling method '{method}'; expected one of {_SAMPLING_METHODS}."
            )
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}.")
        bounds = np.asarray(ranges, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0:
            raise ValueError(f"ranges must be a non-empty list of (low, high) pairs, got {ranges!r}.")
        lows, highs = bounds[:, 0], bounds[:, 1]
        if np.any(lows > highs):
            raise ValueError(f"Each range must satisfy low <= high, got {ranges!r}.")
        d = bounds.shape[0]

        if method == "grid":
            n_per_dim = max(2, math.ceil(n ** (1.0 / d))) if n > 1 else 1
            axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in bounds]
            mesh = np.meshgrid(*axes, indexing="ij")
            return np.column_stack([m.ravel() for m in mesh])

        if method == "random":
            rng = np.random.default_rng(seed)
            return rng.uniform(lows, highs, size=(n, d))

        # method == "lhs"
        from scipy.stats import qmc  # deferred: scipy may still be installing

        sampler = qmc.LatinHypercube(d=d, seed=seed) # type: ignore
        unit = sampler.random(n)
        return qmc.scale(unit, lows, highs)

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------
    def generate_dataset(
        self,
        n: int,
        ranges: BoundsList,
        method: str = "lhs",
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample inputs and simulate each point.

        Points where :meth:`simulate` raises
        :class:`~pwlnet.utils.exceptions.PWLNetError` are skipped with a
        logged warning.

        Parameters
        ----------
        n : int
            Target number of samples (see :meth:`sample_inputs`).
        ranges : list of (float, float)
            Per-input ``(low, high)`` bounds.
        method : {"grid", "random", "lhs"}, optional
            Sampling design (default ``"lhs"``).
        seed : int or None, optional
            Random seed.

        Returns
        -------
        X : numpy.ndarray, shape (n_converged, n_inputs)
        Y : numpy.ndarray, shape (n_converged, n_outputs)
        """
        points = self.sample_inputs(n, ranges, method=method, seed=seed)
        kept_x: list[np.ndarray] = []
        kept_y: list[np.ndarray] = []
        for point in points:
            try:
                outputs = self.simulate(point)
            except PWLNetError as exc:
                _logger.warning(
                    "%s: simulation failed at %s; skipping point (%s).",
                    self.name,
                    np.array2string(point, precision=4),
                    exc,
                )
                continue
            kept_x.append(np.asarray(point, dtype=float))
            kept_y.append(np.asarray(outputs, dtype=float))
        if not kept_x:
            return (
                np.empty((0, self.n_inputs)),
                np.empty((0, self.n_outputs)),
            )
        return np.vstack(kept_x), np.vstack(kept_y)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def to_dataframe(self, X: ArrayLike, Y: ArrayLike) -> "pd.DataFrame":
        """Combine an ``(X, Y)`` dataset into a named pandas DataFrame.

        Parameters
        ----------
        X : array_like, shape (n, n_inputs)
            Input samples (columns named per :attr:`input_names`).
        Y : array_like, shape (n, n_outputs)
            Output samples (columns named per :attr:`output_names`).

        Returns
        -------
        pandas.DataFrame
            ``n`` rows with columns ``input_names + output_names``.
        """
        import pandas as pd  # deferred: keep base module importable without pandas

        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(Y, dtype=float)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.n_inputs:
            raise ValueError(
                f"X must have shape (n, {self.n_inputs}), got {x_arr.shape}."
            )
        if y_arr.ndim != 2 or y_arr.shape[1] != self.n_outputs:
            raise ValueError(
                f"Y must have shape (n, {self.n_outputs}), got {y_arr.shape}."
            )
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}."
            )
        data = np.hstack([x_arr, y_arr])
        return pd.DataFrame(data, columns=list(self.input_names) + list(self.output_names))
