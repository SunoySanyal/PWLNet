"""Activation functions and metadata for PWLNet networks.

Pure-numpy activation functions plus a registry (:data:`ACTIVATIONS`)
that records, per activation, whether it is piecewise linear (MILP
representable) or smooth (NLP representable) and its Keras layer name.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "relu",
    "tanh",
    "identity",
    "ActivationInfo",
    "ACTIVATIONS",
    "get_activation",
]


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit, ``max(x, 0)`` elementwise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of any shape.

    Returns
    -------
    numpy.ndarray
        Array of the same shape with negative entries set to zero.
    """
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent, elementwise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of any shape.

    Returns
    -------
    numpy.ndarray
        ``np.tanh(x)``, same shape as the input.
    """
    return np.tanh(np.asarray(x, dtype=float))


def identity(x: np.ndarray) -> np.ndarray:
    """Identity (linear) activation, returns the input unchanged.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of any shape.

    Returns
    -------
    numpy.ndarray
        ``x`` as a float array, same shape.
    """
    return np.asarray(x, dtype=float)


@dataclass(frozen=True)
class ActivationInfo:
    """Metadata describing one activation function.

    Attributes
    ----------
    name : str
        Canonical PWLNet name (``"relu"``, ``"tanh"``, ``"linear"``).
    fn : callable
        Vectorized numpy implementation, ``fn(x) -> ndarray``.
    is_piecewise_linear : bool
        True if the function is piecewise linear (exactly MILP
        representable): relu and linear/identity.
    smooth : bool
        True if the function is smooth (C-infinity, NLP representable):
        tanh and linear/identity.
    keras_name : str
        The activation string Keras uses for this function.
    """

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    is_piecewise_linear: bool
    smooth: bool
    keras_name: str

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the activation function elementwise."""
        return self.fn(x)


_RELU_INFO = ActivationInfo(
    name="relu", fn=relu, is_piecewise_linear=True, smooth=False, keras_name="relu"
)
_TANH_INFO = ActivationInfo(
    name="tanh", fn=tanh, is_piecewise_linear=False, smooth=True, keras_name="tanh"
)
_LINEAR_INFO = ActivationInfo(
    name="linear", fn=identity, is_piecewise_linear=True, smooth=True, keras_name="linear"
)

#: Registry of supported activations keyed by canonical name. ``"identity"``
#: is accepted as an alias for ``"linear"``.
ACTIVATIONS: dict[str, ActivationInfo] = {
    "relu": _RELU_INFO,
    "tanh": _TANH_INFO,
    "linear": _LINEAR_INFO,
    "identity": _LINEAR_INFO,
}


def get_activation(name: str) -> ActivationInfo:
    """Look up activation metadata by name.

    Parameters
    ----------
    name : str
        One of ``"relu"``, ``"tanh"``, ``"linear"`` (alias
        ``"identity"``). Case-insensitive.

    Returns
    -------
    ActivationInfo
        The registry entry.

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    key = name.strip().lower()
    try:
        return ACTIVATIONS[key]
    except KeyError:
        raise KeyError(
            f"Unknown activation '{name}'. Supported: {sorted(set(ACTIVATIONS))}."
        ) from None
