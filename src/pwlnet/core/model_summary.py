"""Model summaries and formulation size estimates.

Reports parameter/neuron counts per layer and rough estimates of how
large the resulting Pyomo formulation will be for a given formulation
type. The estimates are documented heuristics, not exact counts of any
particular OMLT version.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pwlnet.core.network import NetworkDefinition
from pwlnet.utils.exceptions import FormulationError

__all__ = ["ModelSummary", "summarize", "estimate_formulation_size"]

_SUPPORTED_FORMULATIONS = ("relu_bigm", "tanh_exact")


@dataclass
class ModelSummary:
    """Structural summary of a :class:`NetworkDefinition`.

    Attributes
    ----------
    n_inputs : int
        Number of network inputs.
    n_outputs : int
        Number of network outputs.
    n_layers : int
        Total number of dense layers (hidden + output).
    n_parameters : int
        Total number of weights and biases.
    n_neurons : int
        Total number of HIDDEN neurons (output layer excluded).
    layer_table : list of dict
        One row per layer with keys ``"layer"``, ``"n_inputs"``,
        ``"n_outputs"``, ``"activation"``, ``"n_parameters"``.
    """

    n_inputs: int
    n_outputs: int
    n_layers: int
    n_parameters: int
    n_neurons: int
    layer_table: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"NetworkDefinition: {self.n_inputs} inputs -> {self.n_outputs} outputs, "
            f"{self.n_layers} layers, {self.n_neurons} hidden neurons, "
            f"{self.n_parameters} parameters",
            f"{'layer':>5} {'n_in':>6} {'n_out':>6} {'activation':>10} {'params':>8}",
        ]
        for row in self.layer_table:
            lines.append(
                f"{row['layer']:>5} {row['n_inputs']:>6} {row['n_outputs']:>6} "
                f"{row['activation']:>10} {row['n_parameters']:>8}"
            )
        return "\n".join(lines)


def summarize(network: NetworkDefinition) -> ModelSummary:
    """Build a structural summary of a network.

    Parameters
    ----------
    network : NetworkDefinition
        The network to summarize.

    Returns
    -------
    ModelSummary
        Parameter counts, hidden-neuron count, and a per-layer table.
    """
    layer_table = [
        {
            "layer": k,
            "n_inputs": layer.n_inputs,
            "n_outputs": layer.n_outputs,
            "activation": layer.activation,
            "n_parameters": layer.n_parameters,
        }
        for k, layer in enumerate(network.layers)
    ]
    return ModelSummary(
        n_inputs=network.n_inputs,
        n_outputs=network.n_outputs,
        n_layers=network.n_layers,
        n_parameters=sum(row["n_parameters"] for row in layer_table),
        n_neurons=sum(network.hidden_layer_sizes),
        layer_table=layer_table,
    )


def estimate_formulation_size(network: NetworkDefinition, formulation: str) -> dict[str, int]:
    """Estimate the size of an optimization formulation of the network.

    The counts are documented heuristics for planning and reporting;
    exact numbers depend on the formulation backend (OMLT version,
    presolve, etc.).

    Estimation rules (``H`` = total hidden neurons, ``d`` = n_inputs,
    ``m`` = n_outputs):

    * ``"relu_bigm"`` (big-M MILP, OMLT ReLUBigMFormulation style):
        - continuous: ``d`` input vars + 2 per hidden neuron
          (pre-activation ``zhat`` and post-activation ``z``) + ``m``
          output vars -> ``d + 2H + m``.
        - binary: 1 per hidden neuron -> ``H``.
        - constraints: 1 affine equality per hidden neuron (defines
          ``zhat``) + 4 big-M inequalities per hidden neuron
          (``z >= 0``, ``z >= zhat``, ``z <= zhat - lb (1 - sigma)``,
          ``z <= ub sigma``) + ``m`` output equalities -> ``5H + m``.
    * ``"tanh_exact"`` (smooth NLP, OMLT full-space style):
        - continuous: ``d + 2H + m`` (same variable layout, no
          binaries).
        - binary: 0.
        - constraints: 1 affine equality + 1 nonlinear equality
          ``z = tanh(zhat)`` per hidden neuron + ``m`` output
          equalities -> ``2H + m``.

    Parameters
    ----------
    network : NetworkDefinition
        The network to be embedded.
    formulation : {"relu_bigm", "tanh_exact"}
        Formulation type.

    Returns
    -------
    dict
        Keys ``"n_continuous_vars"``, ``"n_binary_vars"``,
        ``"n_constraints"`` (all int).

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If the formulation name is unknown, or if ``"relu_bigm"`` is
        requested for a network containing tanh layers (big-M is only
        exact for piecewise-linear activations).
    """
    if formulation not in _SUPPORTED_FORMULATIONS:
        raise FormulationError(
            f"Unknown formulation '{formulation}'. "
            f"Supported: {list(_SUPPORTED_FORMULATIONS)}."
        )

    n_hidden = sum(network.hidden_layer_sizes)
    d = network.n_inputs
    m = network.n_outputs

    if formulation == "relu_bigm":
        hidden_acts = {layer.activation for layer in network.layers[:-1]}
        if "tanh" in hidden_acts:
            raise FormulationError(
                "relu_bigm requested but the network has tanh hidden layers; "
                "big-M is only exact for piecewise-linear activations. "
                "Use 'tanh_exact' or a PWL-tanh formulation instead."
            )
        return {
            "n_continuous_vars": d + 2 * n_hidden + m,
            "n_binary_vars": n_hidden,
            "n_constraints": 5 * n_hidden + m,
        }

    # tanh_exact
    return {
        "n_continuous_vars": d + 2 * n_hidden + m,
        "n_binary_vars": 0,
        "n_constraints": 2 * n_hidden + m,
    }
