"""Breakpoint utilities for manual piecewise-linear tanh formulations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pwlnet.utils.typing import ArrayLike

__all__ = [
    "STANDARD_TANH_BREAKPOINTS",
    "get_tanh_breakpoints",
    "validate_breakpoints",
]


STANDARD_TANH_BREAKPOINTS: dict[int, tuple[float, ...]] = {
    3: (-4.0, -1.0, 1.0, 4.0),
    5: (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0),
    7: (-4.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 4.0),
}
"""Standard odd-piece tanh breakpoint sets from the SOSX paper."""


def validate_breakpoints(
    breakpoints: ArrayLike,
    *,
    name: str = "breakpoints",
    require_tanh_domain: bool = False,
) -> np.ndarray:
    """Validate and normalize a 1D strictly increasing breakpoint array.

    Parameters
    ----------
    breakpoints : array_like
        Candidate breakpoints. A valid set has at least two finite,
        strictly increasing values.
    name : str, optional
        Name used in error messages.
    require_tanh_domain : bool, optional
        If true, require the first and last breakpoints to be ``-4`` and
        ``4``. This matches the tanh approximation domain used for the
        Phase-2 PWL formulations.

    Returns
    -------
    numpy.ndarray
        Copy of the validated breakpoints as a float vector.
    """
    arr = np.asarray(breakpoints, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    if require_tanh_domain and (arr[0] != -4.0 or arr[-1] != 4.0):
        raise ValueError(f"{name} must span the tanh PWL domain [-4, 4].")
    return arr.copy()


def get_tanh_breakpoints(
    pieces: int | Iterable[float] = 5,
    *,
    require_tanh_domain: bool = True,
) -> np.ndarray:
    """Return standard or custom breakpoints for approximating tanh.

    Parameters
    ----------
    pieces : int or iterable of float, optional
        If an integer, it must be one of ``3``, ``5``, or ``7`` and
        returns the standard SOSX-paper breakpoint set. If an iterable,
        it is validated and returned as a custom breakpoint vector.
    require_tanh_domain : bool, optional
        Require custom breakpoints to span ``[-4, 4]``. Standard sets
        always satisfy this.

    Returns
    -------
    numpy.ndarray
        Breakpoint vector of length ``n_pieces + 1``.
    """
    if isinstance(pieces, int):
        try:
            return np.array(STANDARD_TANH_BREAKPOINTS[pieces], dtype=float)
        except KeyError as exc:
            raise ValueError(
                "Standard tanh breakpoints are available only for "
                f"3, 5, or 7 pieces; got {pieces}."
            ) from exc
    return validate_breakpoints(
        pieces,
        name="custom tanh breakpoints",
        require_tanh_domain=require_tanh_domain,
    )
