"""Dense layer representation for PWLNet networks.

Pure-numpy layer used by :class:`pwlnet.core.network.NetworkDefinition`.
Weights follow the Keras convention: ``shape (n_in, n_out)`` with the
forward pass ``y = act(x @ W + b)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pwlnet.core.activations import get_activation
from pwlnet.utils.exceptions import NetworkValidationError
from pwlnet.utils.typing import ArrayLike

__all__ = ["DenseLayer"]

_ALLOWED_ACTIVATIONS = ("relu", "tanh", "linear")


@dataclass
class DenseLayer:
    """A fully-connected layer ``y = activation(x @ weights + biases)``.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight matrix of shape ``(n_in, n_out)`` (Keras kernel
        convention).
    biases : numpy.ndarray
        Bias vector of shape ``(n_out,)``.
    activation : str
        One of ``"relu"``, ``"tanh"``, ``"linear"`` (``"identity"`` is
        normalized to ``"linear"``).

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If shapes are inconsistent or the activation is unknown.
    """

    weights: np.ndarray
    biases: np.ndarray
    activation: str = field(default="linear")

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.biases = np.asarray(self.biases, dtype=float).ravel()
        act = str(self.activation).strip().lower()
        if act == "identity":
            act = "linear"
        if act not in _ALLOWED_ACTIVATIONS:
            raise NetworkValidationError(
                f"Unknown activation '{self.activation}'. "
                f"Supported: {list(_ALLOWED_ACTIVATIONS)}."
            )
        self.activation = act
        if self.weights.ndim != 2:
            raise NetworkValidationError(
                f"weights must be 2D (n_in, n_out), got shape {self.weights.shape}."
            )
        if self.biases.shape != (self.weights.shape[1],):
            raise NetworkValidationError(
                f"biases shape {self.biases.shape} does not match n_out="
                f"{self.weights.shape[1]} from weights shape {self.weights.shape}."
            )

    # ------------------------------------------------------------------
    @property
    def n_inputs(self) -> int:
        """Number of layer inputs (rows of the weight matrix)."""
        return int(self.weights.shape[0])

    @property
    def n_outputs(self) -> int:
        """Number of layer outputs / neurons (columns of the weight matrix)."""
        return int(self.weights.shape[1])

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (weights + biases)."""
        return int(self.weights.size + self.biases.size)

    # ------------------------------------------------------------------
    def pre_activation(self, x: ArrayLike) -> np.ndarray:
        """Affine part of the layer, ``x @ W + b``.

        Parameters
        ----------
        x : array_like
            Input of shape ``(n_samples, n_in)`` or ``(n_in,)``.

        Returns
        -------
        numpy.ndarray
            Pre-activation values, shape ``(n_samples, n_out)`` or
            ``(n_out,)`` matching the input's dimensionality.
        """
        arr = np.asarray(x, dtype=float)
        return arr @ self.weights + self.biases

    def forward(self, x: ArrayLike) -> np.ndarray:
        """Full layer output, ``activation(x @ W + b)``.

        Parameters
        ----------
        x : array_like
            Input of shape ``(n_samples, n_in)`` or ``(n_in,)``.

        Returns
        -------
        numpy.ndarray
            Post-activation values, shape ``(n_samples, n_out)`` or
            ``(n_out,)`` matching the input's dimensionality.
        """
        return get_activation(self.activation)(self.pre_activation(x))
