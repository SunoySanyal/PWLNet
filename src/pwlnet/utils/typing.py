"""Common type aliases used across PWLNet."""

from __future__ import annotations

import os
from typing import Sequence, Union

import numpy as np

#: Anything convertible to a numpy array of floats.
ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], float, int]

#: A scalar lower/upper bound pair.
Bounds = tuple[float, float]

#: Per-feature bounds, one ``(lb, ub)`` pair per input dimension.
BoundsList = list[Bounds]

#: A filesystem path.
PathLike = Union[str, os.PathLike]

__all__ = ["ArrayLike", "Bounds", "BoundsList", "PathLike"]
