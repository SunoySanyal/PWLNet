"""Array validation helpers shared across PWLNet modules."""

from __future__ import annotations

import numpy as np

from pwlnet.utils.typing import ArrayLike

__all__ = ["as_2d_array", "check_shape", "check_shapes"]


def as_2d_array(x: ArrayLike, name: str = "x", dtype: type | np.dtype = float) -> np.ndarray:
    """Coerce input to a 2D float array of shape ``(n_samples, n_features)``.

    A 1D array of shape ``(d,)`` is promoted to a single-sample batch of
    shape ``(1, d)``. Scalars and arrays with more than 2 dimensions are
    rejected.

    Parameters
    ----------
    x : array_like
        Input data, shape ``(n, d)`` or ``(d,)``.
    name : str, optional
        Name used in error messages.
    dtype : dtype, optional
        Target dtype (default ``float``).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, d)``.

    Raises
    ------
    ValueError
        If ``x`` is 0-dimensional or has more than 2 dimensions.
    """
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 1D or 2D array, got ndim={arr.ndim} with shape {arr.shape}."
        )
    return arr


def check_shape(arr: np.ndarray, expected: tuple[int | None, ...], name: str = "array") -> None:
    """Check that an array has the expected shape.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to check.
    expected : tuple of int or None
        Expected shape; ``None`` entries are wildcards that match any
        size along that axis.
    name : str, optional
        Name used in error messages.

    Raises
    ------
    ValueError
        If the number of dimensions or any non-wildcard axis size does
        not match.
    """
    if arr.ndim != len(expected):
        raise ValueError(
            f"{name} must have {len(expected)} dimension(s), got shape {arr.shape}."
        )
    for axis, (actual, want) in enumerate(zip(arr.shape, expected)):
        if want is not None and actual != want:
            raise ValueError(
                f"{name} has shape {arr.shape}, expected size {want} along axis {axis}."
            )


def check_shapes(**specs: tuple[np.ndarray, tuple[int | None, ...]]) -> None:
    """Check several arrays against expected shapes at once.

    Parameters
    ----------
    **specs
        Keyword arguments mapping a name to ``(array, expected_shape)``,
        where ``expected_shape`` follows the :func:`check_shape`
        convention (``None`` entries are wildcards).

    Raises
    ------
    ValueError
        If any array fails its shape check.

    Examples
    --------
    >>> import numpy as np
    >>> W = np.zeros((3, 2)); b = np.zeros(2)
    >>> check_shapes(weights=(W, (3, 2)), biases=(b, (2,)))
    """
    for name, (arr, expected) in specs.items():
        check_shape(arr, expected, name=name)
