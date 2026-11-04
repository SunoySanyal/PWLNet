"""Pruning and embedding-size reports for trained networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pwlnet.core.bounds import propagate_bounds
from pwlnet.core.model_summary import estimate_formulation_size, summarize
from pwlnet.core.network import NetworkDefinition
from pwlnet.utils.exceptions import FormulationError, NetworkValidationError

__all__ = ["PruneReport", "build_prune_report"]


@dataclass(frozen=True)
class PruneReport:
    """Compact report of network size and pruning opportunities."""

    n_inputs: int
    n_outputs: int
    n_layers: int
    n_hidden_neurons: int
    n_parameters: int
    formulation_estimates: dict[str, dict[str, int]] = field(default_factory=dict)
    relu_state_counts: dict[str, int] = field(default_factory=dict)
    saturated_tanh_neurons: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "n_layers": self.n_layers,
            "n_hidden_neurons": self.n_hidden_neurons,
            "n_parameters": self.n_parameters,
            "formulation_estimates": self.formulation_estimates,
            "relu_state_counts": self.relu_state_counts,
            "saturated_tanh_neurons": self.saturated_tanh_neurons,
            "notes": list(self.notes),
        }


def build_prune_report(
    network: NetworkDefinition,
    formulations: tuple[str, ...] = ("relu_bigm", "tanh_exact"),
    tanh_saturation_abs: float = 3.0,
) -> PruneReport:
    """Build a pruning/readiness report from structure and interval bounds.

    Parameters
    ----------
    network : NetworkDefinition
        Network to inspect.
    formulations : tuple of str, optional
        Formulation estimates to attempt. Incompatible formulations are
        skipped with a note instead of raising.
    tanh_saturation_abs : float, optional
        Pre-activation magnitude beyond which a tanh neuron is reported
        as saturated throughout the input box.

    Returns
    -------
    PruneReport
        Summary suitable for README/notebook reports and quick tests.
    """
    summary = summarize(network)
    notes: list[str] = []
    estimates: dict[str, dict[str, int]] = {}
    for formulation in formulations:
        try:
            estimates[formulation] = estimate_formulation_size(network, formulation)
        except FormulationError as exc:
            notes.append(f"Skipped {formulation}: {exc}")

    relu_counts = {"dead": 0, "always_active": 0, "uncertain": 0}
    saturated_tanh = 0
    if network.input_bounds is None:
        notes.append("Input bounds unavailable; activation-state pruning report skipped.")
    else:
        try:
            bounds = propagate_bounds(network)
        except NetworkValidationError as exc:
            notes.append(f"Bound propagation failed: {exc}")
        else:
            for layer, (lb, ub) in zip(
                network.layers[:-1], bounds.pre_activation_bounds[:-1]
            ):
                if layer.activation == "relu":
                    relu_counts["dead"] += int((ub <= 0.0).sum())
                    relu_counts["always_active"] += int((lb >= 0.0).sum())
                    relu_counts["uncertain"] += int(((lb < 0.0) & (ub > 0.0)).sum())
                elif layer.activation == "tanh":
                    saturated_tanh += int(
                        ((ub <= -tanh_saturation_abs) | (lb >= tanh_saturation_abs)).sum()
                    )

    return PruneReport(
        n_inputs=summary.n_inputs,
        n_outputs=summary.n_outputs,
        n_layers=summary.n_layers,
        n_hidden_neurons=summary.n_neurons,
        n_parameters=summary.n_parameters,
        formulation_estimates=estimates,
        relu_state_counts=relu_counts,
        saturated_tanh_neurons=saturated_tanh,
        notes=notes,
    )
