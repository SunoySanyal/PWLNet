"""Group-lasso sparsity tooling for PWLNet pruning.

Group lasso encourages whole *neurons* (output units of a Dense layer) to
go to zero by penalizing the L2 norm of each neuron's weight group rather
than individual weights. This module provides:

* :func:`neuron_group_norms` -- pure-numpy inspection of how close each
  hidden neuron's full weight group (incoming column + outgoing row) is to
  zero, i.e. which neurons group-lasso training has driven toward zero.
* :func:`group_lasso_penalty` -- a pure-numpy reporting helper that sums
  the per-neuron group L2 norms over a stack of weight matrices.
* :func:`make_group_lasso_regularizer` -- a Keras ``Regularizer`` (lazy
  ``import keras``) applying column-group L2 to a single Dense kernel,
  suitable as a ``kernel_regularizer`` during training.
* :func:`prune_by_group_norm` -- remove hidden neurons whose group norm is
  below a threshold, returning a :class:`PruneOutcome`.

The numpy helpers and :func:`prune_by_group_norm` are pure numpy. Keras is
imported lazily *inside* :func:`make_group_lasso_regularizer` so importing
this module never requires TensorFlow/Keras (per the architectural rules,
``import keras`` -- Keras 3 standalone, never ``from tensorflow import
keras``).

Group definition
----------------
For hidden layer ``k`` (``0 <= k < n_layers - 1``), neuron ``j``'s weight
group is its INCOMING column ``W_k[:, j]`` (shape ``(n_in_k,)``)
concatenated with its OUTGOING row ``W_{k+1}[j, :]`` (shape
``(n_out_{k+1},)``); the group L2 norm is the Euclidean norm of that
concatenation. A small group norm means the neuron both reads little from
the previous layer and contributes little to the next, so removing it is a
small (approximate) perturbation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.core.network import NetworkDefinition
from pwlnet.pruning.base import PruneOutcome
from pwlnet.pruning.structure import remove_hidden_neurons
from pwlnet.utils.logging import get_logger

__all__ = [
    "neuron_group_norms",
    "group_lasso_penalty",
    "make_group_lasso_regularizer",
    "prune_by_group_norm",
]

_LOGGER = get_logger("pruning.group_lasso")

#: Number of samples drawn from the input box to estimate the sampled max
#: output change in :func:`prune_by_group_norm`.
_SAMPLE_POINTS = 512


def neuron_group_norms(network: NetworkDefinition) -> dict[int, np.ndarray]:
    """Per-neuron group L2 norm for each hidden layer (pure numpy).

    For each hidden layer ``k`` (every layer except the output layer), the
    group norm of neuron ``j`` is the Euclidean norm of its weight group:
    the incoming column ``W_k[:, j]`` concatenated with the outgoing row
    ``W_{k+1}[j, :]``. Neurons with small group norms are the ones
    group-lasso training has driven toward zero.

    Parameters
    ----------
    network : NetworkDefinition
        Network to inspect (never mutated).

    Returns
    -------
    dict[int, numpy.ndarray]
        Mapping ``layer_index -> array`` of shape ``(n_neurons,)`` giving
        the group L2 norm of each neuron in that hidden layer. Layer
        indices run ``0 .. n_layers - 2`` (the output layer has no
        downstream layer and is excluded). Empty dict for a single-layer
        (output-only) network.
    """
    norms: dict[int, np.ndarray] = {}
    n_layers = network.n_layers
    for k in range(n_layers - 1):
        this_w = network.layers[k].weights  # (n_in_k, n_neurons)
        next_w = network.layers[k + 1].weights  # (n_neurons, n_out_{k+1})
        n_neurons = this_w.shape[1]
        # Incoming contribution: sum of squares down each column of W_k.
        incoming_sq = np.sum(this_w**2, axis=0)  # (n_neurons,)
        # Outgoing contribution: sum of squares across each row of W_{k+1}.
        outgoing_sq = np.sum(next_w**2, axis=1)  # (n_neurons,)
        norms[k] = np.sqrt(incoming_sq + outgoing_sq).astype(float)
        # Defensive: shapes must line up (validated network guarantees this).
        if norms[k].shape[0] != n_neurons:  # pragma: no cover - validate() guards
            raise ValueError(
                f"group-norm shape mismatch at layer {k}: "
                f"{norms[k].shape[0]} != {n_neurons}."
            )
    return norms


def group_lasso_penalty(weight_matrices) -> float:
    """Sum of per-neuron (column) group L2 norms over weight matrices.

    This is the classic group-lasso penalty for a stack of Dense kernels,
    treating each OUTPUT unit (column) of every matrix as one group:
    ``sum_matrices sum_j ||W[:, j]||_2``. It is a pure-numpy reporting
    helper (e.g. to log the realized penalty), independent of the
    incoming+outgoing group used by :func:`neuron_group_norms`.

    Parameters
    ----------
    weight_matrices : iterable of array_like
        Weight matrices, each of shape ``(n_in, n_out)``. A single 2D
        array is also accepted and treated as a one-element stack.

    Returns
    -------
    float
        The summed group-lasso penalty (``0.0`` if there are no matrices
        or all are empty).
    """
    mats = weight_matrices
    arr = np.asarray(mats, dtype=float) if isinstance(mats, np.ndarray) else None
    if arr is not None and arr.ndim == 2:
        mats = [arr]
    total = 0.0
    for w in mats:
        w_arr = np.asarray(w, dtype=float)
        if w_arr.ndim != 2:
            raise ValueError(
                f"each weight matrix must be 2D (n_in, n_out), got shape "
                f"{w_arr.shape}."
            )
        if w_arr.size == 0:
            continue
        # Column-group L2: norm down each column, then sum over columns.
        total += float(np.sum(np.sqrt(np.sum(w_arr**2, axis=0))))
    return total


def make_group_lasso_regularizer(strength: float) -> Any:
    """Build a Keras group-lasso ``kernel_regularizer`` (lazy keras import).

    The returned regularizer applies column-group L2 to a Dense kernel:
    ``strength * sum_j ||W[:, j]||_2`` where column ``j`` is the incoming
    weight vector of output unit ``j``. Used as a ``kernel_regularizer`` it
    pushes whole OUTPUT UNITS (neurons) of that layer to zero, producing
    the structured sparsity that :func:`prune_by_group_norm` then removes.

    Parameters
    ----------
    strength : float
        Non-negative regularization strength (the ``lambda`` multiplier).

    Returns
    -------
    keras.regularizers.Regularizer
        A regularizer instance with a ``get_config`` for serialization.

    Raises
    ------
    ValueError
        If ``strength`` is negative.
    pwlnet.utils.exceptions.TrainingError
        If Keras cannot be imported (raised lazily on this call).
    """
    strength = float(strength)
    if strength < 0:
        raise ValueError(f"strength must be >= 0, got {strength}.")

    from pwlnet.training.initialization import import_keras  # lazy keras path

    keras = import_keras()

    class GroupLassoRegularizer(keras.regularizers.Regularizer):
        """Column-group L2 (group lasso) over a Dense kernel's output units.

        ``__call__(W)`` returns ``strength * sum_j sqrt(sum_i W[i, j]**2)``,
        i.e. the sum over output units ``j`` of the L2 norm of that unit's
        incoming weight column.
        """

        def __init__(self, strength: float) -> None:
            self.strength = float(strength)

        def __call__(self, weights):
            # sum_j ||W[:, j]||_2 : norm reduces axis 0 (inputs), sum over
            # axis 1 (output units / neurons). Uses keras.ops so it stays
            # backend-agnostic and differentiable inside the graph.
            col_norms = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(weights), axis=0)
            )
            return self.strength * keras.ops.sum(col_norms)

        def get_config(self) -> dict[str, Any]:
            return {"strength": self.strength}

    return GroupLassoRegularizer(strength)


def prune_by_group_norm(
    network: NetworkDefinition, threshold: float
) -> PruneOutcome:
    """Remove hidden neurons whose group norm is below ``threshold``.

    Neurons with a small :func:`neuron_group_norms` value contribute little
    to the network, so they are removed and treated as a constant ``0``
    post-activation contribution (folded into the next layer's bias via
    :func:`pwlnet.pruning.structure.remove_hidden_neurons`). This is an
    APPROXIMATION (``exact=False``): unless the group norm is literally
    zero, dropping the neuron perturbs the function slightly.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated).
    threshold : float
        Non-negative cutoff. A neuron in hidden layer ``k`` is removed iff
        its group norm is strictly less than ``threshold``.

    Returns
    -------
    PruneOutcome
        Outcome with ``method="group_lasso"``, the pruned ``network``,
        ``removed_neurons`` keyed by hidden-layer index (ORIGINAL neuron
        indices), ``exact=False`` and, when the input network carries
        ``input_bounds``, a sampled ``max_abs_output_change`` estimate in
        ORIGINAL output units.

    Raises
    ------
    ValueError
        If ``threshold`` is negative.
    """
    threshold = float(threshold)
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}.")

    norms = neuron_group_norms(network)
    removed: dict[int, list[int]] = {}
    for layer_index, layer_norms in norms.items():
        below = np.nonzero(layer_norms < threshold)[0]
        if below.size:
            removed[layer_index] = [int(j) for j in below]

    pruned = network
    for layer_index, neuron_indices in removed.items():
        # constant_outputs defaults to 0.0: treat the low-norm neuron's
        # post-activation as ~0 over the box (the approximation).
        pruned = remove_hidden_neurons(pruned, layer_index, neuron_indices)

    n_removed = sum(len(v) for v in removed.values())
    min_removed_norm = (
        min(float(norms[k][j]) for k, js in removed.items() for j in js)
        if n_removed
        else None
    )
    max_removed_norm = (
        max(float(norms[k][j]) for k, js in removed.items() for j in js)
        if n_removed
        else None
    )

    max_change = _sampled_max_output_change(network, pruned)

    notes = [
        f"threshold={threshold:.6g}",
        f"removed {n_removed} neuron(s) with group norm < threshold",
    ]
    if n_removed:
        notes.append(
            f"removed group-norm range [{min_removed_norm:.6g}, "
            f"{max_removed_norm:.6g}]"
        )
    if max_change is None:
        notes.append(
            "max_abs_output_change unavailable (network has no input_bounds)"
        )

    return PruneOutcome(
        network=pruned,
        method="group_lasso",
        removed_neurons=removed,
        activation_changes={},
        n_zeroed_weights=0,
        exact=False,
        max_abs_output_change=max_change,
        notes=notes,
    )


def _sampled_max_output_change(
    original: NetworkDefinition, pruned: NetworkDefinition
) -> float | None:
    """Sampled max abs output change over the input box (ORIGINAL units).

    Draws uniform samples from the original network's input box and returns
    the max absolute difference between the two networks' predictions, or
    ``None`` if the network has no ``input_bounds`` to sample from.
    """
    if original.input_bounds is None:
        return None
    lb = np.array([b[0] for b in original.input_bounds], dtype=float)
    ub = np.array([b[1] for b in original.input_bounds], dtype=float)
    rng = np.random.default_rng(0)
    samples = rng.uniform(lb, ub, size=(_SAMPLE_POINTS, lb.size))
    diff = np.abs(original.predict(samples) - pruned.predict(samples))
    return float(np.max(diff))
