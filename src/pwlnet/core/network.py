"""Internal network representation for PWLNet.

:class:`NetworkDefinition` is the framework-independent core object of
the package: a stack of :class:`pwlnet.core.layers.DenseLayer` objects
with optional input/output :class:`pwlnet.core.scaling.OffsetScaler`
transforms and input bounds in ORIGINAL (unscaled) units.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from pwlnet.core.layers import DenseLayer
from pwlnet.core.scaling import OffsetScaler
from pwlnet.utils.exceptions import NetworkValidationError
from pwlnet.utils.typing import ArrayLike, BoundsList, PathLike
from pwlnet.utils.validation import as_2d_array

__all__ = ["NetworkDefinition"]

_FORMAT_VERSION = 1


class NetworkDefinition:
    """Framework-independent feed-forward dense network.

    The forward pass through the SCALED network is
    ``h_{k+1} = act_k(h_k @ W_k + b_k)``; :meth:`predict` additionally
    applies input scaling before, and inverse output scaling after, the
    forward pass.

    Parameters
    ----------
    layers : list of DenseLayer
        Layers in order; the last layer produces the (scaled) network
        outputs.
    input_scaler : OffsetScaler or None, optional
        Scaler applied to raw inputs before the forward pass.
    output_scaler : OffsetScaler or None, optional
        Scaler whose ``inverse_transform`` maps scaled network outputs
        back to original units.
    input_bounds : list of (float, float) or None, optional
        Per-input ``(lb, ub)`` bounds in ORIGINAL (unscaled) space.

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If the configuration is structurally invalid (see
        :meth:`validate`).
    """

    def __init__(
        self,
        layers: list[DenseLayer],
        input_scaler: OffsetScaler | None = None,
        output_scaler: OffsetScaler | None = None,
        input_bounds: BoundsList | None = None,
    ) -> None:
        self.layers: list[DenseLayer] = list(layers)
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.input_bounds: BoundsList | None = (
            [(float(lb), float(ub)) for lb, ub in input_bounds]
            if input_bounds is not None
            else None
        )
        self.validate()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_inputs(self) -> int:
        """Number of network inputs."""
        return self.layers[0].n_inputs

    @property
    def n_outputs(self) -> int:
        """Number of network outputs."""
        return self.layers[-1].n_outputs

    @property
    def n_layers(self) -> int:
        """Total number of dense layers (hidden + output)."""
        return len(self.layers)

    @property
    def hidden_layer_sizes(self) -> list[int]:
        """Widths of the hidden layers (all layers except the last)."""
        return [layer.n_outputs for layer in self.layers[:-1]]

    @property
    def activations(self) -> list[str]:
        """Activation name of each layer, in order."""
        return [layer.activation for layer in self.layers]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Check structural consistency of the network.

        Raises
        ------
        pwlnet.utils.exceptions.NetworkValidationError
            If the network has no layers, consecutive layer shapes do
            not chain, a scaler's feature count does not match the
            network, or input bounds are malformed (wrong length or
            ``lb > ub``).
        """
        if not self.layers:
            raise NetworkValidationError("NetworkDefinition requires at least one layer.")
        for k, layer in enumerate(self.layers):
            if not isinstance(layer, DenseLayer):
                raise NetworkValidationError(
                    f"Layer {k} is {type(layer).__name__}, expected DenseLayer."
                )
        for k in range(len(self.layers) - 1):
            n_out = self.layers[k].n_outputs
            n_in = self.layers[k + 1].n_inputs
            if n_out != n_in:
                raise NetworkValidationError(
                    f"Layer {k} has n_outputs={n_out} but layer {k + 1} expects "
                    f"n_inputs={n_in}."
                )
        if self.input_scaler is not None and self.input_scaler.n_features != self.n_inputs:
            raise NetworkValidationError(
                f"input_scaler has {self.input_scaler.n_features} features but the "
                f"network has {self.n_inputs} inputs."
            )
        if self.output_scaler is not None and self.output_scaler.n_features != self.n_outputs:
            raise NetworkValidationError(
                f"output_scaler has {self.output_scaler.n_features} features but the "
                f"network has {self.n_outputs} outputs."
            )
        if self.input_bounds is not None:
            if len(self.input_bounds) != self.n_inputs:
                raise NetworkValidationError(
                    f"input_bounds has {len(self.input_bounds)} entries but the "
                    f"network has {self.n_inputs} inputs."
                )
            for i, (lb, ub) in enumerate(self.input_bounds):
                if not (np.isfinite(lb) and np.isfinite(ub)):
                    raise NetworkValidationError(
                        f"input_bounds[{i}] = ({lb}, {ub}) must be finite."
                    )
                if lb > ub:
                    raise NetworkValidationError(
                        f"input_bounds[{i}] has lb={lb} > ub={ub}."
                    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def forward_scaled(self, X_scaled: ArrayLike) -> np.ndarray:
        """Forward pass through the raw network (no scalers applied).

        Parameters
        ----------
        X_scaled : array_like
            Inputs in SCALED space, shape ``(n_samples, n_inputs)`` or
            ``(n_inputs,)``.

        Returns
        -------
        numpy.ndarray
            Outputs in SCALED space, shape ``(n_samples, n_outputs)``
            or ``(n_outputs,)`` matching the input's dimensionality.
        """
        x_in = np.asarray(X_scaled, dtype=float)
        squeeze = x_in.ndim == 1
        h = as_2d_array(x_in, name="X_scaled")
        if h.shape[1] != self.n_inputs:
            raise NetworkValidationError(
                f"X_scaled has {h.shape[1]} features, expected {self.n_inputs}."
            )
        for layer in self.layers:
            h = layer.forward(h)
        return h[0] if squeeze else h

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Evaluate the network in ORIGINAL units.

        Applies input scaling (if any), the forward pass, then inverse
        output scaling (if any).

        Parameters
        ----------
        X : array_like
            Inputs in original units, shape ``(n_samples, n_inputs)``
            or ``(n_inputs,)``.

        Returns
        -------
        numpy.ndarray
            Outputs in original units, shape ``(n_samples, n_outputs)``
            or ``(n_outputs,)`` matching the input's dimensionality.
        """
        x_in = np.asarray(X, dtype=float)
        squeeze = x_in.ndim == 1
        arr = as_2d_array(x_in, name="X")
        if arr.shape[1] != self.n_inputs:
            raise NetworkValidationError(
                f"X has {arr.shape[1]} features, expected {self.n_inputs}."
            )
        if self.input_scaler is not None:
            arr = self.input_scaler.transform(arr)
        y = self.forward_scaled(arr)
        if self.output_scaler is not None:
            y = self.output_scaler.inverse_transform(y)
        return y[0] if squeeze else y

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------
    def scaled_input_bounds(self) -> BoundsList:
        """Input bounds transformed into SCALED space.

        Returns
        -------
        list of (float, float)
            Per-input bounds after applying the input scaler. If there
            is no input scaler the original bounds are returned. Bound
            ordering is preserved even for negative scaling factors.

        Raises
        ------
        pwlnet.utils.exceptions.NetworkValidationError
            If the network has no input bounds.
        """
        if self.input_bounds is None:
            raise NetworkValidationError(
                "Network has no input_bounds; set them before requesting "
                "scaled_input_bounds()."
            )
        lb = np.array([b[0] for b in self.input_bounds], dtype=float)
        ub = np.array([b[1] for b in self.input_bounds], dtype=float)
        if self.input_scaler is None:
            return [(float(l), float(u)) for l, u in zip(lb, ub)]
        a = self.input_scaler.transform(lb)
        b = self.input_scaler.transform(ub)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        return [(float(l), float(u)) for l, u in zip(lo, hi)]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize the full network to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Round-trippable representation (see :meth:`from_dict`).
        """
        return {
            "format_version": _FORMAT_VERSION,
            "layers": [
                {
                    "weights": layer.weights.tolist(),
                    "biases": layer.biases.tolist(),
                    "activation": layer.activation,
                }
                for layer in self.layers
            ],
            "input_scaler": self.input_scaler.to_dict() if self.input_scaler else None,
            "output_scaler": self.output_scaler.to_dict() if self.output_scaler else None,
            "input_bounds": (
                [[lb, ub] for lb, ub in self.input_bounds]
                if self.input_bounds is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NetworkDefinition":
        """Reconstruct a network from :meth:`to_dict` output.

        Parameters
        ----------
        d : dict
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        NetworkDefinition
            The reconstructed network.
        """
        layers = [
            DenseLayer(
                weights=np.asarray(spec["weights"], dtype=float),
                biases=np.asarray(spec["biases"], dtype=float),
                activation=spec["activation"],
            )
            for spec in d["layers"]
        ]
        input_scaler = (
            OffsetScaler.from_dict(d["input_scaler"]) if d.get("input_scaler") else None
        )
        output_scaler = (
            OffsetScaler.from_dict(d["output_scaler"]) if d.get("output_scaler") else None
        )
        raw_bounds = d.get("input_bounds")
        input_bounds = (
            [(float(lb), float(ub)) for lb, ub in raw_bounds]
            if raw_bounds is not None
            else None
        )
        return cls(
            layers=layers,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            input_bounds=input_bounds,
        )

    def save(self, path: PathLike) -> None:
        """Save the network to a JSON file (exact float round-trip).

        Parameters
        ----------
        path : str or os.PathLike
            Destination file path (conventionally ``*.json``).
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: PathLike) -> "NetworkDefinition":
        """Load a network previously written by :meth:`save`.

        Parameters
        ----------
        path : str or os.PathLike
            Source file path.

        Returns
        -------
        NetworkDefinition
            The loaded network; weights, biases, scalers and bounds are
            bit-exact with the saved network.
        """
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        sizes = [self.n_inputs] + [layer.n_outputs for layer in self.layers]
        arch = " -> ".join(str(s) for s in sizes)
        return (
            f"NetworkDefinition({arch}, activations={self.activations}, "
            f"input_scaler={'yes' if self.input_scaler else 'no'}, "
            f"output_scaler={'yes' if self.output_scaler else 'no'})"
        )
