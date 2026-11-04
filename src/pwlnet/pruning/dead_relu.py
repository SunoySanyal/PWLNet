"""Bound-based ReLU pruning for PWLNet (pure numpy).

This module uses interval bound propagation (``pwlnet.core.bounds``) to
identify ReLU neurons whose activation pattern is *constant* over the
network's input box, then removes or linearizes them **exactly** (the
pruned network reproduces the original to floating-point round-off).

Two structural facts about a ReLU neuron with scaled pre-activation
interval ``[lb, ub]`` over the input box drive everything here:

* **Dead** (``ub <= 0``): ``relu(z) = 0`` for every reachable ``z``, so
  the neuron is the constant ``0`` and can be folded away with
  :func:`pwlnet.pruning.structure.remove_hidden_neurons`
  (``constant_outputs=0``). This shrinks the hidden width and, for the
  ``relu_bigm`` formulation, removes one binary variable per neuron.
* **Always-active** (``lb >= 0``): ``relu(z) = z`` for every reachable
  ``z``, so the neuron is exactly linear. ``DenseLayer`` carries a
  single per-layer activation, so we can only flip a layer to
  ``"linear"`` when *all* of its neurons are always-active; that layer
  is then fused into its linear neighbours via
  :func:`pwlnet.pruning.structure.fuse_linear_layers`. Either way the
  ReLU binaries for that layer disappear.

Bound propagation here is *conservative*: if a neuron is flagged dead or
always-active it provably is so over the box (the real interval is a
subset of the propagated one), so every change in this module is exact.
The converse does not hold -- conservative bounds may miss neurons that
are in fact constant -- so this is sound but not complete.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

import numpy as np

from pwlnet.core.bounds import propagate_bounds
from pwlnet.core.network import NetworkDefinition
from pwlnet.pruning import structure
from pwlnet.pruning.base import PruneOutcome
from pwlnet.utils.exceptions import NetworkValidationError

__all__ = [
    "prune_dead_relus",
    "simplify_always_active_relus",
    "prune_relus",
]


def _require_bounds(network: NetworkDefinition) -> None:
    """Raise if the network has no input bounds.

    Parameters
    ----------
    network : NetworkDefinition
        Network to check.

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If ``network.input_bounds`` is ``None``.
    """
    if network.input_bounds is None:
        raise NetworkValidationError(
            "Bound-based ReLU pruning requires network.input_bounds (in "
            "original units); none are set."
        )


def _hidden_relu_indices(network: NetworkDefinition) -> list[int]:
    """Indices of hidden layers whose activation is ``"relu"``.

    Parameters
    ----------
    network : NetworkDefinition
        Network to inspect.

    Returns
    -------
    list of int
        Layer indices ``k`` with ``0 <= k < n_layers - 1`` (hidden) and
        ``layers[k].activation == "relu"``.
    """
    n_layers = network.n_layers
    return [
        k
        for k in range(n_layers - 1)
        if network.layers[k].activation == "relu"
    ]


def prune_dead_relus(network: NetworkDefinition) -> PruneOutcome:
    """Remove dead ReLU neurons (constant ``0`` over the input box).

    A hidden ReLU neuron ``j`` in layer ``k`` is *dead* when its scaled
    pre-activation upper bound is ``<= 0`` over the input box; then
    ``relu`` outputs the constant ``0`` everywhere reachable, so the
    neuron contributes nothing and is folded away exactly with
    :func:`pwlnet.pruning.structure.remove_hidden_neurons` using a
    constant output of ``0``.

    If *every* neuron in a hidden layer is dead, removing them all would
    leave a zero-width layer (structurally invalid). Rather than perform
    the more delicate fold-and-drop of an entire layer, that single layer
    is left untouched and a note is recorded -- the rest of the network is
    still pruned. (Such a fully-dead layer means the network output is
    constant downstream; it is rare and flagged for the caller.)

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated). Must have ``input_bounds`` set.

    Returns
    -------
    PruneOutcome
        ``method="dead_relu"``. ``removed_neurons`` maps each affected
        hidden layer index to the ORIGINAL indices removed.
        ``exact=True`` and ``max_abs_output_change=0.0`` (the removal is
        function-preserving over the box). If nothing was removed the
        returned ``network`` is a deep copy.

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If the network has no input bounds.
    """
    _require_bounds(network)
    bounds = propagate_bounds(network)

    current = structure.copy_network(network)
    removed: dict[int, list[int]] = {}
    notes: list[str] = []

    for k in _hidden_relu_indices(network):
        # pre_activation_bounds[k] corresponds to layers[k]; ub <= 0 means
        # relu is pinned to 0 over the whole box -> dead neuron.
        pre_ub = bounds.pre_activation_bounds[k][1]
        dead = np.flatnonzero(pre_ub <= 0.0)
        if dead.size == 0:
            continue

        n_neurons = network.layers[k].n_outputs
        if dead.size == n_neurons:
            # Removing every neuron would produce a 0-width layer, which is
            # structurally invalid. Leave this layer untouched (correctness
            # over completeness) and tell the caller.
            notes.append(
                f"layer {k}: all {n_neurons} relu neurons are dead "
                f"(constant 0); left untouched to avoid an empty layer."
            )
            continue

        dead_list = [int(j) for j in dead]
        # Folding constant 0 is a no-op on the downstream bias, but we pass
        # it explicitly to document the dead-neuron semantics.
        current = structure.remove_hidden_neurons(
            current, k, dead_list, constant_outputs=0.0
        )
        removed[k] = dead_list

    return PruneOutcome(
        network=current,
        method="dead_relu",
        removed_neurons=removed,
        exact=True,
        max_abs_output_change=0.0,
        notes=notes,
    )


def simplify_always_active_relus(network: NetworkDefinition) -> PruneOutcome:
    """Linearize ReLU layers that are always-active over the input box.

    A hidden ReLU neuron ``j`` in layer ``k`` is *always-active* when its
    scaled pre-activation lower bound is ``>= 0`` over the input box; then
    ``relu(z) = z`` everywhere reachable, so the neuron is exactly linear.

    Because :class:`pwlnet.core.layers.DenseLayer` carries a single
    activation for the whole layer, a per-neuron activation change is not
    possible. Therefore:

    * If **all** neurons in a hidden ReLU layer are always-active, the
      layer is switched to ``"linear"`` with
      :func:`pwlnet.pruning.structure.set_layer_activation`. After all
      such layers are switched, consecutive linear layers are fused with
      :func:`pwlnet.pruning.structure.fuse_linear_layers`. This is exact
      and removes that layer's ReLU binaries from the ``relu_bigm``
      formulation.
    * If only **some** neurons are always-active, the layer's activation
      cannot be changed without breaking the still-nonlinear neurons, so
      the layer is left as ReLU and a note records the count (no change).

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated). Must have ``input_bounds`` set.

    Returns
    -------
    PruneOutcome
        ``method="always_active_relu"``. ``activation_changes`` maps each
        relinearized layer index (in the INPUT network's indexing) to
        ``"linear"``. ``exact=True`` and ``max_abs_output_change=0.0``.
        No neurons are removed by this step (fusion only merges layers),
        so ``removed_neurons`` is empty.

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If the network has no input bounds.
    """
    _require_bounds(network)
    bounds = propagate_bounds(network)

    current = structure.copy_network(network)
    activation_changes: dict[int, str] = {}
    notes: list[str] = []
    any_changed = False

    for k in _hidden_relu_indices(network):
        # pre_activation_bounds[k] corresponds to layers[k]; lb >= 0 means
        # relu(z) = z over the whole box -> the neuron is exactly linear.
        pre_lb = bounds.pre_activation_bounds[k][0]
        always = np.flatnonzero(pre_lb >= 0.0)
        if always.size == 0:
            continue

        n_neurons = network.layers[k].n_outputs
        if always.size == n_neurons:
            # Whole layer is linear: flip its activation. We operate on
            # 'current' but the layer index is stable here because this
            # step never removes neurons or layers before fusing.
            current = structure.set_layer_activation(current, k, "linear")
            activation_changes[k] = "linear"
            any_changed = True
        else:
            notes.append(
                f"layer {k}: {int(always.size)}/{n_neurons} relu neurons "
                f"are always-active but the rest are not; activation left "
                f"as relu (per-layer activation cannot be split)."
            )

    if any_changed:
        # Fuse the freshly-linearized layers into their linear neighbours
        # (including the final linear output layer). Exact composition.
        current = structure.fuse_linear_layers(current)

    return PruneOutcome(
        network=current,
        method="always_active_relu",
        activation_changes=activation_changes,
        exact=True,
        max_abs_output_change=0.0,
        notes=notes,
    )


def prune_relus(
    network: NetworkDefinition,
    dead: bool = True,
    always_active: bool = True,
) -> PruneOutcome:
    """Apply dead-neuron removal then always-active linearization.

    Convenience wrapper that runs :func:`prune_dead_relus` (if ``dead``)
    and then :func:`simplify_always_active_relus` (if ``always_active``)
    on its result, merging the two outcomes. Both steps are exact, so the
    combined outcome is exact too.

    Note that ``simplify_always_active_relus`` is re-evaluated against the
    network produced by the dead-neuron step (its bounds are recomputed
    there), so its ``activation_changes`` layer indices refer to that
    intermediate network. Since the dead step never adds or reorders
    layers (it only removes neurons within hidden layers), the hidden
    layer indices are unchanged between the two steps.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated). Must have ``input_bounds`` set.
    dead : bool, optional
        If ``True`` (default), remove dead ReLU neurons.
    always_active : bool, optional
        If ``True`` (default), linearize fully always-active ReLU layers.

    Returns
    -------
    PruneOutcome
        ``method="relu_prune"`` combining ``removed_neurons`` and
        ``activation_changes`` from both steps. ``exact=True`` and
        ``max_abs_output_change=0.0``.

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If the network has no input bounds.
    """
    _require_bounds(network)

    current = structure.copy_network(network)
    removed: dict[int, list[int]] = {}
    activation_changes: dict[int, str] = {}
    notes: list[str] = []

    if dead:
        dead_outcome = prune_dead_relus(current)
        current = dead_outcome.network
        for layer_index, neurons in dead_outcome.removed_neurons.items():
            removed[layer_index] = list(neurons)
        notes.extend(dead_outcome.notes)

    if always_active:
        aa_outcome = simplify_always_active_relus(current)
        current = aa_outcome.network
        activation_changes.update(aa_outcome.activation_changes)
        notes.extend(aa_outcome.notes)

    return PruneOutcome(
        network=current,
        method="relu_prune",
        removed_neurons=removed,
        activation_changes=activation_changes,
        exact=True,
        max_abs_output_change=0.0,
        notes=notes,
    )
