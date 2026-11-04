"""Weight-magnitude pruning for PWLNet networks (pure numpy).

This module implements unstructured magnitude pruning: small-|weight|
entries are zeroed (sparsified) and, optionally, any HIDDEN neuron that
becomes structurally trivial as a result is removed EXACTLY by folding
its (now-constant or now-irrelevant) contribution into the downstream
bias via :func:`pwlnet.pruning.structure.remove_hidden_neurons`.

Two kinds of trivial hidden neuron are detected after zeroing:

* **Dead-input neuron** -- its entire incoming weight column
  ``W_k[:, j]`` is zero, so its pre-activation collapses to the constant
  ``b_k[j]`` and its post-activation collapses to the constant
  ``act_k(b_k[j])`` over the whole input box. Removing it with
  ``constant_outputs = act_k(b_k[j])`` is exact GIVEN the zeros.
* **Dead-output neuron** -- its entire outgoing weight row
  ``W_{k+1}[j, :]`` is zero, so it contributes nothing to the next
  layer's pre-activation regardless of its value. Removing it with
  ``constant_outputs = 0`` is exact GIVEN the zeros (the fold adds
  ``0 * row_j = 0``).

The zeroing step itself generally changes the network function, so the
overall :class:`~pwlnet.pruning.base.PruneOutcome` is reported with
``exact=False`` whenever at least one weight was zeroed (and ``True``
only in the degenerate case where nothing was zeroed). The cleanup
removals are exact RELATIVE to the zeroed network, not relative to the
original.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

import numpy as np

from pwlnet.core.activations import get_activation
from pwlnet.core.layers import DenseLayer
from pwlnet.core.network import NetworkDefinition
from pwlnet.pruning.base import PruneOutcome
from pwlnet.pruning.structure import remove_hidden_neurons, zero_small_weights

__all__ = ["prune_weights_magnitude", "neuron_constant_value"]

# Number of Monte-Carlo samples used to estimate the sampled
# max-abs-output-change versus the original network.
_N_SAMPLE = 512


def neuron_constant_value(layer: DenseLayer, j: int) -> float:
    """Constant post-activation value of a dead-input neuron.

    If neuron ``j`` of ``layer`` has an all-zero incoming weight column
    (``layer.weights[:, j] == 0``), its pre-activation over any input is
    the constant bias ``layer.biases[j]`` and its post-activation is the
    constant ``activation(layer.biases[j])``. This helper returns that
    value (it does NOT check that the column is actually zero -- callers
    are responsible for that).

    Parameters
    ----------
    layer : DenseLayer
        The layer the neuron belongs to.
    j : int
        Neuron (column) index within ``layer``.

    Returns
    -------
    float
        ``activation(layer.biases[j])`` as a Python float.
    """
    act = get_activation(layer.activation)
    return float(act(np.asarray(layer.biases[j], dtype=float)))


def _sample_inputs(network: NetworkDefinition, n_samples: int, rng) -> np.ndarray | None:
    """Draw inputs uniformly from the network's input box (original units).

    Parameters
    ----------
    network : NetworkDefinition
        Network whose ``input_bounds`` define the box.
    n_samples : int
        Number of points to draw.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(n_samples, n_inputs)`` drawn uniformly from the
        box, or ``None`` if the network has no ``input_bounds``.
    """
    if network.input_bounds is None:
        return None
    lb = np.array([b[0] for b in network.input_bounds], dtype=float)
    ub = np.array([b[1] for b in network.input_bounds], dtype=float)
    return lb + (ub - lb) * rng.random((n_samples, network.n_inputs))


def _sampled_max_abs_output_change(
    original: NetworkDefinition,
    pruned: NetworkDefinition,
    n_samples: int,
    rng,
) -> float | None:
    """Estimate the max abs output change between two networks over the box.

    Parameters
    ----------
    original : NetworkDefinition
        Reference network (defines the input box).
    pruned : NetworkDefinition
        Network to compare against ``original``.
    n_samples : int
        Number of Monte-Carlo points.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    float or None
        The maximum absolute difference of any output component over the
        sampled points, in ORIGINAL output units, or ``None`` if the
        network has no ``input_bounds`` (so no box to sample).
    """
    X = _sample_inputs(original, n_samples, rng)
    if X is None:
        return None
    y0 = np.atleast_2d(original.predict(X))
    y1 = np.atleast_2d(pruned.predict(X))
    return float(np.max(np.abs(y0 - y1)))


def _find_trivial_hidden_neurons(
    network: NetworkDefinition,
) -> dict[int, tuple[list[int], list[float]]]:
    """Find hidden neurons made trivial by zeroed weights.

    Scans every HIDDEN layer ``k`` (``0 <= k < n_layers - 1``) for neurons
    whose removal is exact given the current (already-zeroed) weights:

    * dead-output (outgoing row ``W_{k+1}[j, :]`` all zero) -> constant
      ``0.0`` (its value is irrelevant downstream);
    * dead-input (incoming column ``W_k[:, j]`` all zero) -> constant
      ``activation(b_k[j])``.

    Dead-output is preferred when a neuron is both, since a zero constant
    is unconditionally exact there.

    Parameters
    ----------
    network : NetworkDefinition
        The (already-zeroed) network to scan; not mutated.

    Returns
    -------
    dict[int, tuple[list[int], list[float]]]
        Mapping ``layer_index -> (neuron_indices, constant_outputs)`` for
        the trivial neurons in that hidden layer (positionally aligned).
        Layers with no trivial neurons are omitted.
    """
    result: dict[int, tuple[list[int], list[float]]] = {}
    n_layers = network.n_layers
    for k in range(n_layers - 1):  # hidden layers only
        this_layer = network.layers[k]
        next_layer = network.layers[k + 1]
        indices: list[int] = []
        consts: list[float] = []
        for j in range(this_layer.n_outputs):
            outgoing_dead = not np.any(next_layer.weights[j, :])
            incoming_dead = not np.any(this_layer.weights[:, j])
            if outgoing_dead:
                # Contribution is 0 regardless of the neuron's value.
                indices.append(j)
                consts.append(0.0)
            elif incoming_dead:
                # Pre-activation is the constant bias -> constant post-act.
                indices.append(j)
                consts.append(neuron_constant_value(this_layer, j))
        if indices:
            result[k] = (indices, consts)
    return result


def prune_weights_magnitude(
    network: NetworkDefinition,
    threshold: float | None = None,
    fraction: float | None = None,
    per_layer: bool = True,
    cleanup: bool = True,
) -> PruneOutcome:
    """Magnitude-prune a network's weights, optionally cleaning up neurons.

    Exactly one of ``threshold`` or ``fraction`` must be supplied. The
    smallest-magnitude weights are set to zero via
    :func:`pwlnet.pruning.structure.zero_small_weights` (weights only;
    biases are never zeroed here). When ``cleanup`` is ``True``, any
    HIDDEN neuron rendered structurally trivial by the zeroing -- a fully
    disconnected incoming column or outgoing row -- is removed EXACTLY by
    folding its constant contribution into the next layer's bias.

    Parameters
    ----------
    network : NetworkDefinition
        Source network. Never mutated; the outcome's ``network`` is new.
    threshold : float or None, optional
        Absolute magnitude cutoff (every weight with ``abs < threshold``
        is zeroed). Mutually exclusive with ``fraction``.
    fraction : float or None, optional
        Zero the smallest ``fraction`` of weight magnitudes (in ``[0, 1]``).
        Mutually exclusive with ``threshold``.
    per_layer : bool, optional
        Compute the ``fraction`` quantile per layer (default) or globally.
        Ignored when ``threshold`` is used.
    cleanup : bool, optional
        If ``True`` (default), remove hidden neurons that became
        structurally trivial after zeroing.

    Returns
    -------
    PruneOutcome
        Outcome with ``method="weight_magnitude"``, the zeroed/cleaned
        network, ``n_zeroed_weights`` set, ``removed_neurons`` recording
        the cleanup removals (original-layer indices), ``exact`` ``False``
        whenever any weight was zeroed (the cleanup removals are exact
        only relative to the zeroed network), and ``max_abs_output_change``
        a sampled estimate versus the original network over the input box
        (``None`` -- with an explanatory note -- if the network has no
        ``input_bounds``).

    Raises
    ------
    ValueError
        If neither or both of ``threshold``/``fraction`` are provided, or
        if they are out of range (propagated from ``zero_small_weights``).
    """
    rng = np.random.default_rng(0)
    notes: list[str] = []

    # Step 1: zero the smallest-magnitude weights (weights only).
    zeroed_net, n_zeroed = zero_small_weights(
        network,
        threshold=threshold,
        fraction=fraction,
        per_layer=per_layer,
        include_biases=False,
    )

    removed_neurons: dict[int, list[int]] = {}
    working = zeroed_net

    # Step 2: optional exact cleanup of structurally-trivial hidden neurons.
    if cleanup:
        trivial = _find_trivial_hidden_neurons(working)
        # Remove from the deepest hidden layer first so that the
        # original-layer indices we record stay valid (removing neurons
        # from layer k does not renumber layers; we remove all flagged
        # neurons of a layer in one call, so within-layer indices are the
        # original ones).
        for k in sorted(trivial.keys()):
            indices, consts = trivial[k]
            working = remove_hidden_neurons(
                working, k, indices, constant_outputs=consts
            )
            removed_neurons[k] = list(indices)
        n_removed = sum(len(v) for v in removed_neurons.values())
        if n_removed:
            notes.append(
                f"cleanup removed {n_removed} structurally-trivial hidden "
                f"neuron(s) exactly (relative to the zeroed network)."
            )

    # Step 3: sampled output-change estimate vs the ORIGINAL network.
    max_change = _sampled_max_abs_output_change(network, working, _N_SAMPLE, rng)
    if max_change is None:
        notes.append(
            "network has no input_bounds; max_abs_output_change not "
            "estimated (no input box to sample)."
        )

    notes.append(
        f"zeroed {n_zeroed} weight(s); zeroing changes the function so the "
        f"overall outcome is exact only if nothing was zeroed."
    )

    return PruneOutcome(
        network=working,
        method="weight_magnitude",
        removed_neurons=removed_neurons,
        activation_changes={},
        n_zeroed_weights=int(n_zeroed),
        exact=(n_zeroed == 0),
        max_abs_output_change=max_change,
        notes=notes,
    )
