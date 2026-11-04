"""Shared result contract for all PWLNet pruning methods.

Every pruning function consumes a :class:`pwlnet.core.network.NetworkDefinition`
and returns a :class:`PruneOutcome` whose ``.network`` is a NEW network
(the input network is never mutated). This module is pure numpy and must
stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pwlnet.core.network import NetworkDefinition

__all__ = ["PruneOutcome"]


@dataclass
class PruneOutcome:
    """Result of a single pruning operation.

    Pruning functions take a :class:`NetworkDefinition` and return a
    ``PruneOutcome`` whose :attr:`network` is a brand-new
    :class:`NetworkDefinition` (deep-copied layers; the input network's
    arrays are never mutated). The remaining fields record what changed
    so downstream code can report and (where applicable) trust exactness.

    Parameters
    ----------
    network : NetworkDefinition
        The NEW pruned network. The input network is never mutated.
    method : str
        Identifier of the pruning method, e.g. ``"dead_relu"`` or
        ``"weight_magnitude"``.
    removed_neurons : dict[int, list[int]], optional
        Mapping ``layer_index -> list of ORIGINAL neuron indices removed``
        from that hidden layer. Indices refer to the columns of the
        layer's weight matrix in the INPUT network.
    activation_changes : dict[int, str], optional
        Mapping ``layer_index -> new activation`` for layers whose
        activation was replaced (e.g. ``"relu" -> "linear"``).
    n_zeroed_weights : int, optional
        Number of individual weight entries set to zero (magnitude /
        sparsification methods).
    exact : bool, optional
        ``True`` iff the operation is function-preserving over the input
        box to roughly ``1e-9``.
    max_abs_output_change : float or None, optional
        Estimate of the maximum absolute change in any network output
        over the input box, in ORIGINAL output units (sampled and/or
        bound-based). ``None`` if unknown.
    notes : list[str], optional
        Free-form human-readable notes about the operation.

    Attributes
    ----------
    n_removed_neurons : int
        Total number of neurons removed across all layers (read-only
        property derived from :attr:`removed_neurons`).
    """

    network: NetworkDefinition
    method: str
    removed_neurons: dict[int, list[int]] = field(default_factory=dict)
    activation_changes: dict[int, str] = field(default_factory=dict)
    n_zeroed_weights: int = 0
    exact: bool = False
    max_abs_output_change: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_removed_neurons(self) -> int:
        """Total number of removed neurons across all layers."""
        return sum(len(v) for v in self.removed_neurons.values())

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of this outcome.

        Returns
        -------
        dict
            Keys ``"method"``, ``"n_removed_neurons"``,
            ``"removed_neurons"`` (layer index keys as strings),
            ``"activation_changes"`` (layer index keys as strings),
            ``"n_zeroed_weights"``, ``"exact"``,
            ``"max_abs_output_change"`` and ``"notes"``. All values are
            JSON-serializable (no numpy types, integer dict keys are
            stringified).
        """
        return {
            "method": self.method,
            "n_removed_neurons": int(self.n_removed_neurons),
            "removed_neurons": {
                str(layer_index): [int(j) for j in neurons]
                for layer_index, neurons in self.removed_neurons.items()
            },
            "activation_changes": {
                str(layer_index): str(act)
                for layer_index, act in self.activation_changes.items()
            },
            "n_zeroed_weights": int(self.n_zeroed_weights),
            "exact": bool(self.exact),
            "max_abs_output_change": (
                None
                if self.max_abs_output_change is None
                else float(self.max_abs_output_change)
            ),
            "notes": list(self.notes),
        }
