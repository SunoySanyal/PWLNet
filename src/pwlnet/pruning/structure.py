"""Structural primitives for PWLNet pruning (pure numpy).

These functions operate directly on :class:`NetworkDefinition` objects,
returning NEW validated networks without ever mutating the input. They
are the foundation every pruning method builds on:

* :func:`copy_network` -- deep copy.
* :func:`remove_hidden_neurons` -- exact constant-folding removal of
  hidden neurons (columns) into the next layer's bias.
* :func:`set_layer_activation` -- replace a single layer's activation.
* :func:`fuse_linear_layers` -- collapse runs of consecutive linear
  layers into one, preserving the function exactly.
* :func:`zero_small_weights` -- magnitude sparsification.

Index-math invariants used throughout (so callers fold constants
correctly):

* A layer's weight matrix has shape ``(n_in, n_out)`` (Keras kernel
  convention); the forward pass is ``y = act(x @ W + b)``.
* "Neuron j of layer k" == column ``j`` of ``layers[k].weights`` and
  entry ``j`` of ``layers[k].biases``. Its post-activation value feeds
  the NEXT layer as input feature ``j``, i.e. ROW ``j`` of
  ``layers[k+1].weights``.
* Removing neuron ``j`` from hidden layer ``k`` whose post-activation is
  the constant ``c_j`` over the input box is exact iff you first add
  ``c_j * layers[k+1].weights[j, :]`` to ``layers[k+1].biases`` (fold the
  constant contribution into the downstream bias), THEN delete column
  ``j`` of ``layers[k].weights`` + entry ``j`` of ``layers[k].biases``
  AND row ``j`` of ``layers[k+1].weights``.
* "Hidden layer" means index ``< n_layers - 1``. The last layer
  (``n_layers - 1``) is the output layer and may never be removed or
  have its rows folded as if hidden.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

import numpy as np

from pwlnet.core.layers import DenseLayer
from pwlnet.core.network import NetworkDefinition
from pwlnet.utils.exceptions import NetworkValidationError

__all__ = [
    "copy_network",
    "remove_hidden_neurons",
    "set_layer_activation",
    "fuse_linear_layers",
    "zero_small_weights",
]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _copy_layers(network: NetworkDefinition) -> list[DenseLayer]:
    """Deep-copy a network's layers into fresh ``DenseLayer`` objects.

    Parameters
    ----------
    network : NetworkDefinition
        Source network.

    Returns
    -------
    list of DenseLayer
        New layers with copied (not shared) weight/bias arrays.
    """
    return [
        DenseLayer(
            weights=np.array(layer.weights, dtype=float, copy=True),
            biases=np.array(layer.biases, dtype=float, copy=True),
            activation=layer.activation,
        )
        for layer in network.layers
    ]


def _build(network: NetworkDefinition, layers: list[DenseLayer]) -> NetworkDefinition:
    """Construct a new network reusing the source scalers and bounds.

    Parameters
    ----------
    network : NetworkDefinition
        Source network providing ``input_scaler``, ``output_scaler`` and
        ``input_bounds`` (the scalers are reused by reference, the bounds
        are deep-copied by ``NetworkDefinition.__init__``).
    layers : list of DenseLayer
        Layers for the new network.

    Returns
    -------
    NetworkDefinition
        A validated network.
    """
    return NetworkDefinition(
        layers=layers,
        input_scaler=network.input_scaler,
        output_scaler=network.output_scaler,
        input_bounds=network.input_bounds,
    )


# ----------------------------------------------------------------------
# Public primitives
# ----------------------------------------------------------------------
def copy_network(network: NetworkDefinition) -> NetworkDefinition:
    """Deep-copy a network.

    The returned network has brand-new :class:`DenseLayer` objects with
    copied weight/bias arrays. The input/output scalers are reused by
    reference (they are immutable in practice), and the input bounds are
    copied by the :class:`NetworkDefinition` constructor.

    Parameters
    ----------
    network : NetworkDefinition
        Network to copy.

    Returns
    -------
    NetworkDefinition
        An independent, validated copy. Mutating its layer arrays does
        not affect ``network``.
    """
    return _build(network, _copy_layers(network))


def remove_hidden_neurons(
    network: NetworkDefinition,
    layer_index: int,
    neuron_indices,
    constant_outputs=None,
) -> NetworkDefinition:
    """Remove neurons from a hidden layer by folding their constants.

    Each removed neuron ``j`` is assumed to take a CONSTANT post-activation
    value ``c_j`` over the input box (default ``0.0``). The removal is
    exact (function-preserving over the box) iff each removed neuron's
    true post-activation really equals ``c_j`` there.

    The fold (see module invariants) is:

    1. For every removed neuron ``j``, add
       ``c_j * layers[layer_index + 1].weights[j, :]`` to
       ``layers[layer_index + 1].biases``.
    2. Delete column ``j`` of ``layers[layer_index].weights`` and entry
       ``j`` of ``layers[layer_index].biases``.
    3. Delete row ``j`` of ``layers[layer_index + 1].weights``.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated).
    layer_index : int
        Index of the HIDDEN layer to prune. Must satisfy
        ``0 <= layer_index < n_layers - 1`` (the output layer cannot be
        pruned this way).
    neuron_indices : array_like of int
        Original neuron (column) indices to remove from
        ``layers[layer_index]``. An empty list is a no-op (returns a deep
        copy). Indices must be unique and in range.
    constant_outputs : array_like of float or None, optional
        Constant post-activation value ``c_j`` for each removed neuron,
        in the SAME ORDER as ``neuron_indices``. If ``None``, all are
        ``0.0``. A scalar is broadcast to all removed neurons.

    Returns
    -------
    NetworkDefinition
        A new validated network with the neurons removed and their
        constant contributions folded into the next layer's bias.

    Raises
    ------
    ValueError
        If ``layer_index`` is out of range or is the output layer, if any
        neuron index is out of range or duplicated, or if
        ``constant_outputs`` has the wrong length.
    """
    n_layers = network.n_layers
    if not (0 <= layer_index < n_layers - 1):
        raise ValueError(
            f"layer_index={layer_index} is not a hidden layer; expected "
            f"0 <= layer_index < {n_layers - 1} (n_layers - 1). The output "
            f"layer cannot be pruned with remove_hidden_neurons."
        )

    idx = np.asarray(list(neuron_indices), dtype=int).ravel()
    layers = _copy_layers(network)

    # Empty removal -> deep copy no-op.
    if idx.size == 0:
        return _build(network, layers)

    n_neurons = layers[layer_index].n_outputs
    if np.any(idx < 0) or np.any(idx >= n_neurons):
        raise ValueError(
            f"neuron_indices out of range for layer {layer_index} with "
            f"{n_neurons} neurons: {idx.tolist()}."
        )
    if np.unique(idx).size != idx.size:
        raise ValueError(f"neuron_indices contains duplicates: {idx.tolist()}.")

    if constant_outputs is None:
        consts = np.zeros(idx.size, dtype=float)
    else:
        consts = np.atleast_1d(np.asarray(constant_outputs, dtype=float)).ravel()
        if consts.size == 1 and idx.size > 1:
            consts = np.full(idx.size, float(consts[0]))
        if consts.size != idx.size:
            raise ValueError(
                f"constant_outputs has {consts.size} entries but "
                f"{idx.size} neurons are being removed."
            )

    this_layer = layers[layer_index]
    next_layer = layers[layer_index + 1]

    # Step 1: fold each removed neuron's constant contribution into the
    # downstream bias. next_layer.weights has shape (n_in=n_neurons, n_out).
    # Row j of next_layer.weights is the outgoing weight vector of neuron j.
    # Summing c_j * row_j over removed neurons gives the bias correction.
    bias_correction = consts @ next_layer.weights[idx, :]
    next_layer.biases = next_layer.biases + bias_correction

    # Steps 2 & 3: delete column j of this layer's weights + bias entry j,
    # and row j of the next layer's weights.
    this_layer.weights = np.delete(this_layer.weights, idx, axis=1)
    this_layer.biases = np.delete(this_layer.biases, idx, axis=0)
    next_layer.weights = np.delete(next_layer.weights, idx, axis=0)

    return _build(network, layers)


def set_layer_activation(
    network: NetworkDefinition, layer_index: int, activation: str
) -> NetworkDefinition:
    """Return a copy of ``network`` with one layer's activation replaced.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated).
    layer_index : int
        Index of the layer whose activation to change
        (``0 <= layer_index < n_layers``).
    activation : str
        New activation name, one of ``"relu"``, ``"tanh"``, ``"linear"``
        (``"identity"`` is normalized to ``"linear"`` by
        :class:`DenseLayer`).

    Returns
    -------
    NetworkDefinition
        A new validated network with the updated activation.

    Raises
    ------
    ValueError
        If ``layer_index`` is out of range.
    NetworkValidationError
        If ``activation`` is not a supported activation name.
    """
    n_layers = network.n_layers
    if not (0 <= layer_index < n_layers):
        raise ValueError(
            f"layer_index={layer_index} out of range for a network with "
            f"{n_layers} layers."
        )
    layers = _copy_layers(network)
    # DenseLayer.__post_init__ validates / normalizes the activation name
    # and raises NetworkValidationError on an unknown one.
    layers[layer_index] = DenseLayer(
        weights=layers[layer_index].weights,
        biases=layers[layer_index].biases,
        activation=activation,
    )
    return _build(network, layers)


def fuse_linear_layers(network: NetworkDefinition) -> NetworkDefinition:
    """Fuse runs of consecutive ``linear`` layers into single layers.

    Any maximal run of two or more consecutive layers that are ALL
    ``"linear"`` is collapsed into one equivalent linear layer. For a run
    ``A`` then ``B`` (forward pass ``(x @ W_a + b_a) @ W_b + b_b``), the
    fused layer is ``W = W_a @ W_b``, ``b = b_a @ W_b + b_b``, which
    preserves the function exactly. Non-linear layers and isolated linear
    layers are left untouched. At least one layer always remains.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated).

    Returns
    -------
    NetworkDefinition
        A new validated network whose ``predict`` matches the input's to
        within floating-point round-off (~1e-9), with consecutive linear
        layers merged.
    """
    src = _copy_layers(network)
    fused: list[DenseLayer] = []
    i = 0
    n = len(src)
    while i < n:
        layer = src[i]
        if layer.activation != "linear":
            fused.append(layer)
            i += 1
            continue
        # Start a run of consecutive linear layers at i.
        w = layer.weights
        b = layer.biases
        j = i + 1
        while j < n and src[j].activation == "linear":
            # Compose A (current accumulated) into B = src[j].
            # (x @ W + b) @ W_b + b_b  ->  W' = W @ W_b, b' = b @ W_b + b_b
            w_b = src[j].weights
            b_b = src[j].biases
            b = b @ w_b + b_b
            w = w @ w_b
            j += 1
        fused.append(DenseLayer(weights=w, biases=b, activation="linear"))
        i = j
    return _build(network, fused)


def zero_small_weights(
    network: NetworkDefinition,
    threshold: float | None = None,
    fraction: float | None = None,
    per_layer: bool = True,
    include_biases: bool = False,
):
    """Zero out small-magnitude weights (and optionally biases).

    Exactly one of ``threshold`` or ``fraction`` must be supplied.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated).
    threshold : float or None, optional
        Absolute cutoff: every entry with ``abs(value) < threshold`` is
        set to ``0``. Must be ``>= 0``.
    fraction : float or None, optional
        Zero the smallest ``fraction`` of entries by magnitude (between
        ``0`` and ``1`` inclusive). Applied per layer when ``per_layer``
        is ``True``, otherwise globally across all included entries.
        The cutoff is the ``fraction``-quantile of the magnitudes, so
        ties at the boundary may push the zeroed count slightly above or
        below ``fraction * N``.
    per_layer : bool, optional
        If ``True`` (default), the ``fraction`` quantile is computed
        independently for each layer; if ``False`` it is computed over
        the pooled magnitudes of all included entries.
    include_biases : bool, optional
        If ``True``, biases are eligible for zeroing as well; otherwise
        only weight matrices are considered.

    Returns
    -------
    tuple of (NetworkDefinition, int)
        The new validated network and the number of entries set to zero.

    Raises
    ------
    ValueError
        If neither or both of ``threshold``/``fraction`` are given, if
        ``threshold < 0``, or if ``fraction`` is outside ``[0, 1]``.
    """
    if (threshold is None) == (fraction is None):
        raise ValueError(
            "Provide exactly one of 'threshold' or 'fraction' "
            "(got threshold=%r, fraction=%r)." % (threshold, fraction)
        )
    if threshold is not None and threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}.")
    if fraction is not None and not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")

    layers = _copy_layers(network)

    # Collect the arrays eligible for zeroing, in order, so the global
    # path can pool their magnitudes.
    arrays: list[np.ndarray] = []
    for layer in layers:
        arrays.append(layer.weights)
        if include_biases:
            arrays.append(layer.biases)

    n_zeroed = 0

    def _apply_threshold(arr: np.ndarray, cut: float) -> int:
        mask = np.abs(arr) < cut
        count = int(mask.sum())
        arr[mask] = 0.0
        return count

    if threshold is not None:
        for arr in arrays:
            n_zeroed += _apply_threshold(arr, float(threshold))
        return _build(network, layers), n_zeroed

    # fraction path
    if per_layer:
        for arr in arrays:
            if arr.size == 0:
                continue
            cut = float(np.quantile(np.abs(arr), fraction))
            # Zero entries with magnitude <= cut so that requesting
            # fraction zeros approximately that fraction (quantile is the
            # value at the fraction-th position).
            mask = np.abs(arr) <= cut
            n_zeroed += int(mask.sum())
            arr[mask] = 0.0
    else:
        all_mags = np.concatenate(
            [np.abs(arr).ravel() for arr in arrays if arr.size > 0]
        )
        if all_mags.size > 0:
            cut = float(np.quantile(all_mags, fraction))
            for arr in arrays:
                if arr.size == 0:
                    continue
                mask = np.abs(arr) <= cut
                n_zeroed += int(mask.sum())
                arr[mask] = 0.0

    return _build(network, layers), n_zeroed
