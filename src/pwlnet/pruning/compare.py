"""Before/after comparison reporting for PWLNet pruning (pure numpy).

This is the Phase-3 validation table: it tabulates how a network's
structural and formulation-level size changes from BEFORE pruning to
AFTER pruning -- hidden-neuron counts, parameter counts, nonzero
parameter counts, and per-formulation continuous/binary variable and
constraint counts. Solve *time* is measured in the notebook (it depends
on the solver), not here.

The functions never raise for an incompatible formulation (e.g. asking
for ``relu_bigm`` on a network with tanh hidden layers): the offending
formulation is simply skipped and recorded as a note.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pwlnet.core.model_summary import estimate_formulation_size, summarize
from pwlnet.core.network import NetworkDefinition
from pwlnet.utils.exceptions import FormulationError
from pwlnet.utils.logging import get_logger

__all__ = [
    "count_nonzero_parameters",
    "formulation_size",
    "NetworkComparison",
    "compare_networks",
]

_logger = get_logger(__name__)

# Metric keys reported per formulation, in display order.
_FORMULATION_METRICS = ("n_continuous_vars", "n_binary_vars", "n_constraints")


def count_nonzero_parameters(network: NetworkDefinition) -> int:
    """Count the nonzero weights and biases of a network.

    Counting *nonzero* (rather than total) parameters lets magnitude /
    sparsification pruning show up in the comparison table even when no
    neurons are structurally removed: zeroing a weight with
    :func:`pwlnet.pruning.structure.zero_small_weights` leaves the array
    shape unchanged but reduces this count.

    Parameters
    ----------
    network : NetworkDefinition
        Network to inspect (never mutated).

    Returns
    -------
    int
        Total number of weight and bias entries whose value is not
        exactly ``0.0``, summed across all layers.
    """
    total = 0
    for layer in network.layers:
        total += int(np.count_nonzero(layer.weights))
        total += int(np.count_nonzero(layer.biases))
    return total


def formulation_size(network: NetworkDefinition, formulation: str) -> dict[str, int]:
    """Estimate a formulation's size, returning ``{}`` instead of raising.

    Thin, non-raising wrapper over
    :func:`pwlnet.core.model_summary.estimate_formulation_size`. If the
    formulation is unknown or incompatible with the network (for example
    ``"relu_bigm"`` requested for a network that still has tanh hidden
    layers), the underlying :class:`~pwlnet.utils.exceptions.FormulationError`
    is caught, logged at INFO level, and an empty dict is returned so the
    comparison table can simply omit that column.

    Parameters
    ----------
    network : NetworkDefinition
        Network to be embedded.
    formulation : str
        Formulation name (e.g. ``"relu_bigm"`` or ``"tanh_exact"``).

    Returns
    -------
    dict
        ``{"n_continuous_vars": ..., "n_binary_vars": ...,
        "n_constraints": ...}`` on success, or ``{}`` if the formulation
        is unknown or incompatible with this network.
    """
    try:
        return estimate_formulation_size(network, formulation)
    except FormulationError as exc:
        _logger.info("Formulation '%s' unavailable for this network: %s", formulation, exc)
        return {}


def _network_row(
    network: NetworkDefinition, formulations: tuple[str, ...]
) -> dict[str, Any]:
    """Build the structural + formulation metric row for one network.

    Parameters
    ----------
    network : NetworkDefinition
        Network to measure.
    formulations : tuple of str
        Formulation names to estimate (incompatible ones are skipped).

    Returns
    -------
    dict
        Keys ``"n_hidden_neurons"``, ``"n_parameters"``,
        ``"n_nonzero_parameters"`` plus, for every COMPATIBLE
        formulation ``f`` and metric ``k``, ``f"{f}.{k}"``.
    """
    summary = summarize(network)
    row: dict[str, Any] = {
        "n_hidden_neurons": int(summary.n_neurons),
        "n_parameters": int(summary.n_parameters),
        "n_nonzero_parameters": count_nonzero_parameters(network),
    }
    for formulation in formulations:
        size = formulation_size(network, formulation)
        for metric in _FORMULATION_METRICS:
            if metric in size:
                row[f"{formulation}.{metric}"] = int(size[metric])
    return row


@dataclass
class NetworkComparison:
    """Before/after size comparison of two networks (Phase-3 table).

    Holds one metric row for the ``before`` network, one for the
    ``after`` network, and the per-metric deltas (``after - before``).
    Structural metrics (``n_hidden_neurons``, ``n_parameters``,
    ``n_nonzero_parameters``) are always present; per-formulation metrics
    (``"{formulation}.{metric}"``) are present only for formulations
    compatible with BOTH networks.

    Parameters
    ----------
    before : dict
        Metric row for the network before pruning (see
        :func:`compare_networks`).
    after : dict
        Metric row for the network after pruning.
    deltas : dict
        ``after[k] - before[k]`` for every key present in both rows.
    formulations : tuple of str
        The formulation names that were requested.
    notes : list of str, optional
        Human-readable notes, e.g. which formulations were skipped for
        which network because they were incompatible.
    """

    before: dict[str, Any]
    after: dict[str, Any]
    deltas: dict[str, Any]
    formulations: tuple[str, ...]
    notes: list[str] = field(default_factory=list)

    # Display order / labels for the structural metrics common to all nets.
    _STRUCTURAL_KEYS = (
        "n_hidden_neurons",
        "n_parameters",
        "n_nonzero_parameters",
    )

    def _ordered_keys(self) -> list[str]:
        """Return all metric keys in a stable display order.

        Structural metrics first, then each formulation's metrics in the
        order the formulations were requested. Only keys actually present
        in either row are returned.
        """
        keys: list[str] = [k for k in self._STRUCTURAL_KEYS]
        for formulation in self.formulations:
            for metric in _FORMULATION_METRICS:
                key = f"{formulation}.{metric}"
                if key in self.before or key in self.after:
                    keys.append(key)
        return keys

    def to_rows(self) -> list[dict[str, Any]]:
        """Return the comparison as a list of per-metric row dicts.

        Each row is JSON-friendly with keys ``"metric"``, ``"before"``,
        ``"after"`` and ``"delta"``. Missing values (a formulation
        incompatible with one network) are reported as ``None``.

        Returns
        -------
        list of dict
            One row per metric, in display order.
        """
        rows: list[dict[str, Any]] = []
        for key in self._ordered_keys():
            before = self.before.get(key)
            after = self.after.get(key)
            delta = self.deltas.get(key)
            rows.append(
                {
                    "metric": key,
                    "before": None if before is None else int(before),
                    "after": None if after is None else int(after),
                    "delta": None if delta is None else int(delta),
                }
            )
        return rows

    def to_markdown(self) -> str:
        """Render the comparison as a GitHub-flavoured Markdown table.

        Columns: ``Metric | Before | After | Delta``. Deltas are
        signed (``+`` prefix for positive). Missing values render as an
        empty cell. Any :attr:`notes` are appended as a bullet list under
        the table.

        Returns
        -------
        str
            Markdown source suitable for display in a notebook.
        """
        header = "| Metric | Before | After | Delta |"
        sep = "| --- | ---: | ---: | ---: |"
        lines = [header, sep]
        for row in self.to_rows():
            before = "" if row["before"] is None else str(row["before"])
            after = "" if row["after"] is None else str(row["after"])
            if row["delta"] is None:
                delta = ""
            else:
                delta = f"{row['delta']:+d}"
            lines.append(f"| {row['metric']} | {before} | {after} | {delta} |")
        text = "\n".join(lines)
        if self.notes:
            note_lines = "\n".join(f"- {note}" for note in self.notes)
            text = f"{text}\n\n{note_lines}"
        return text

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the comparison.

        Returns
        -------
        dict
            Keys ``"before"``, ``"after"``, ``"deltas"``,
            ``"formulations"`` and ``"notes"``.
        """
        return {
            "before": dict(self.before),
            "after": dict(self.after),
            "deltas": dict(self.deltas),
            "formulations": list(self.formulations),
            "notes": list(self.notes),
        }


def compare_networks(
    before: NetworkDefinition,
    after: NetworkDefinition,
    formulations: tuple[str, ...] = ("relu_bigm", "tanh_exact"),
) -> NetworkComparison:
    """Compare two networks' structural and formulation-level sizes.

    Builds the Phase-3 before/after validation table: for each network it
    records the hidden-neuron count, total parameter count, nonzero
    parameter count, and -- for every requested formulation compatible
    with that network -- the estimated continuous-variable, binary-variable
    and constraint counts. Deltas are ``after - before`` for every metric
    present in BOTH rows.

    A formulation incompatible with one network (e.g. ``"relu_bigm"`` on a
    net with tanh hidden layers) is skipped for that network with a note;
    its delta is therefore omitted (reported as ``None`` by
    :meth:`NetworkComparison.to_rows`).

    Parameters
    ----------
    before : NetworkDefinition
        Network before pruning (never mutated).
    after : NetworkDefinition
        Network after pruning (never mutated).
    formulations : tuple of str, optional
        Formulation names to estimate. Defaults to
        ``("relu_bigm", "tanh_exact")``.

    Returns
    -------
    NetworkComparison
        The before row, after row, deltas, and any notes.
    """
    formulations = tuple(formulations)
    before_row = _network_row(before, formulations)
    after_row = _network_row(after, formulations)

    deltas: dict[str, Any] = {}
    for key, before_value in before_row.items():
        if key in after_row:
            deltas[key] = int(after_row[key]) - int(before_value)

    notes: list[str] = []
    for formulation in formulations:
        in_before = any(
            f"{formulation}.{m}" in before_row for m in _FORMULATION_METRICS
        )
        in_after = any(
            f"{formulation}.{m}" in after_row for m in _FORMULATION_METRICS
        )
        if in_before and not in_after:
            notes.append(
                f"Formulation '{formulation}' is incompatible with the AFTER "
                f"network; its columns are omitted."
            )
        elif in_after and not in_before:
            notes.append(
                f"Formulation '{formulation}' is incompatible with the BEFORE "
                f"network; its columns are omitted."
            )
        elif not in_before and not in_after:
            notes.append(
                f"Formulation '{formulation}' is incompatible with both "
                f"networks; its columns are omitted."
            )

    return NetworkComparison(
        before=before_row,
        after=after_row,
        deltas=deltas,
        formulations=formulations,
        notes=notes,
    )
