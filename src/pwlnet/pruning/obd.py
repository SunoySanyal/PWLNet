"""Optimal Brain Damage (OBD) style saliency pruning.

This module implements weight-saliency pruning following LeCun, Denker &
Solla, "Optimal Brain Damage" (NeurIPS 1989). The saliency of a weight is

    s_i = 0.5 * H_ii * w_i^2,

where ``H_ii`` is the ``i``-th diagonal entry of the loss Hessian with
respect to the network weights and ``w_i`` is the weight value. Weights
with the smallest saliency are the cheapest to remove (smallest predicted
increase in the loss under the OBD diagonal/quadratic assumption).

Computing the exact Hessian diagonal is expensive, so this module uses the
**empirical-Fisher / Gauss-Newton diagonal approximation**:

    H_ii ~= mean_n ( d L_n / d w_i )^2,

i.e. the mean over the ``n`` training samples of the squared per-sample
gradient of the loss with respect to each weight. This is the standard
positive-semidefinite surrogate used in practice; it requires only first
derivatives (one backward pass per sample) and is exactly the diagonal of
the empirical Fisher information for an MSE loss.

The Keras model is reconstructed in SCALED space (matching
:class:`pwlnet.training.keras_trainer.KerasTrainer`), so saliencies are
computed in the same space the network was trained in. Keras and
TensorFlow are imported lazily inside the functions; importing this module
never requires either.

Pruning never mutates the input network: :func:`optimal_brain_damage`
returns a :class:`~pwlnet.pruning.base.PruneOutcome` whose ``.network`` is
a new :class:`~pwlnet.core.network.NetworkDefinition`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.core.network import NetworkDefinition
from pwlnet.pruning.base import PruneOutcome
from pwlnet.pruning.structure import copy_network, remove_hidden_neurons
from pwlnet.training.initialization import import_keras
from pwlnet.utils.exceptions import PWLNetError
from pwlnet.utils.logging import get_logger
from pwlnet.utils.typing import ArrayLike
from pwlnet.utils.validation import as_2d_array

__all__ = ["obd_saliencies", "optimal_brain_damage"]

_LOGGER = get_logger("pruning.obd")

_LOSSES = ("mse", "mae")


def _import_tf() -> Any:
    """Import and return the ``tensorflow`` module (lazy).

    Returns
    -------
    module
        The imported ``tensorflow`` module.

    Raises
    ------
    pwlnet.utils.exceptions.PWLNetError
        If TensorFlow cannot be imported.
    """
    try:
        import tensorflow as tf  # noqa: PLC0415 -- deliberate lazy import
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PWLNetError(
            "TensorFlow is required for OBD saliency pruning but could not be "
            "imported. Install the training extras (e.g. `pip install "
            "tensorflow`) and retry. "
            f"Original import error: {exc}"
        ) from exc
    return tf


def _build_keras_model(network: NetworkDefinition, keras: Any) -> Any:
    """Build a Keras ``Sequential`` mirroring ``network`` in scaled space.

    The model operates entirely in SCALED space (no scalers inside the
    graph); each :class:`~pwlnet.core.layers.DenseLayer` becomes a Keras
    ``Dense`` layer with the same activation and weights are copied in.

    Parameters
    ----------
    network : NetworkDefinition
        Source network.
    keras : module
        The imported ``keras`` module.

    Returns
    -------
    keras.Sequential
        A built model whose weights equal ``network``'s layer arrays. The
        trainable-variable order is ``[W_0, b_0, W_1, b_1, ...]`` matching
        the network's layer order.
    """
    model = keras.Sequential(name="pwlnet_obd")
    model.add(keras.layers.Input(shape=(network.n_inputs,)))
    for layer in network.layers:
        model.add(
            keras.layers.Dense(
                layer.n_outputs,
                activation=layer.activation,
            )
        )
    # Force variable creation, then copy weights in layer order.
    model.build((None, network.n_inputs))
    for keras_layer, core_layer in zip(model.layers, network.layers):
        keras_layer.set_weights(
            [
                np.asarray(core_layer.weights, dtype=np.float32),
                np.asarray(core_layer.biases, dtype=np.float32),
            ]
        )
    return model


def _scaled_targets(network: NetworkDefinition, y: np.ndarray) -> np.ndarray:
    """Scale targets into the network's output (scaled) space.

    Parameters
    ----------
    network : NetworkDefinition
        Source network carrying the optional output scaler.
    y : numpy.ndarray
        Targets in ORIGINAL units, shape ``(n_samples, n_outputs)``.

    Returns
    -------
    numpy.ndarray
        Targets in SCALED space (identity if there is no output scaler).
    """
    if network.output_scaler is not None:
        return np.asarray(network.output_scaler.transform(y), dtype=float)
    return y


def _scaled_inputs(network: NetworkDefinition, X: np.ndarray) -> np.ndarray:
    """Scale inputs into the network's input (scaled) space.

    Parameters
    ----------
    network : NetworkDefinition
        Source network carrying the optional input scaler.
    X : numpy.ndarray
        Inputs in ORIGINAL units, shape ``(n_samples, n_inputs)``.

    Returns
    -------
    numpy.ndarray
        Inputs in SCALED space (identity if there is no input scaler).
    """
    if network.input_scaler is not None:
        return np.asarray(network.input_scaler.transform(X), dtype=float)
    return X


def obd_saliencies(
    network: NetworkDefinition,
    X: ArrayLike,
    y: ArrayLike,
    loss: str = "mse",
) -> dict[int, np.ndarray]:
    """Compute per-weight OBD saliencies for each layer's weight matrix.

    A Keras model mirroring ``network`` is built in SCALED space and the
    diagonal Hessian is approximated by the empirical-Fisher /
    Gauss-Newton estimate

        H_ii ~= mean_n ( d L_n / d w_i )^2,

    the mean over samples of the squared per-sample loss gradient. The
    saliency of weight ``i`` is then ``s_i = 0.5 * H_ii * w_i^2``.

    Only the weight matrices (not the biases) receive saliencies, matching
    the OBD parameters that :func:`optimal_brain_damage` prunes.

    Parameters
    ----------
    network : NetworkDefinition
        Network whose weights are scored (never mutated).
    X : array_like
        Inputs in ORIGINAL units, shape ``(n_samples, n_inputs)``.
    y : array_like
        Targets in ORIGINAL units, shape ``(n_samples, n_outputs)`` or
        ``(n_samples,)``.
    loss : {"mse", "mae"}, optional
        Per-sample loss used for the gradient. ``"mse"`` (default) uses
        the mean squared error over outputs; ``"mae"`` the mean absolute
        error.

    Returns
    -------
    dict[int, numpy.ndarray]
        Mapping ``layer_index -> saliency array`` with the same shape as
        ``network.layers[layer_index].weights``. All entries are ``>= 0``.

    Raises
    ------
    pwlnet.utils.exceptions.PWLNetError
        If Keras/TensorFlow cannot be imported.
    ValueError
        If ``loss`` is not supported or ``X``/``y`` shapes are
        inconsistent with the network.
    """
    loss_name = str(loss).strip().lower()
    if loss_name not in _LOSSES:
        raise ValueError(f"loss must be one of {list(_LOSSES)}, got {loss!r}.")

    keras = import_keras()
    tf = _import_tf()

    X_arr = as_2d_array(X, name="X")
    y_in = np.asarray(y, dtype=float)
    if y_in.ndim == 1:
        y_in = y_in.reshape(-1, 1)
    y_arr = as_2d_array(y_in, name="y")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    if X_arr.shape[0] == 0:
        raise ValueError("Cannot compute saliencies on an empty dataset.")
    if X_arr.shape[1] != network.n_inputs:
        raise ValueError(
            f"X has {X_arr.shape[1]} features, expected {network.n_inputs}."
        )
    if y_arr.shape[1] != network.n_outputs:
        raise ValueError(
            f"y has {y_arr.shape[1]} outputs, expected {network.n_outputs}."
        )

    X_scaled = _scaled_inputs(network, X_arr).astype(np.float32)
    y_scaled = _scaled_targets(network, y_arr).astype(np.float32)

    model = _build_keras_model(network, keras)
    # Trainable order is [W_0, b_0, W_1, b_1, ...]; weight matrices are the
    # even-indexed variables. We accumulate H_ii only for those.
    weight_vars = [v for i, v in enumerate(model.trainable_variables) if i % 2 == 0]

    X_t = tf.constant(X_scaled)
    y_t = tf.constant(y_scaled)

    def per_sample_loss(preds: Any, targets: Any) -> Any:
        """Per-sample loss vector, shape (batch,)."""
        if loss_name == "mse":
            return tf.reduce_mean(tf.square(preds - targets), axis=1)
        return tf.reduce_mean(tf.abs(preds - targets), axis=1)

    # Accumulate the sum over samples of squared per-sample gradients for
    # each weight matrix, then divide by n to get the mean (empirical
    # Fisher diagonal). We use tape.jacobian to obtain per-sample gradients
    # in one pass per batch: jac[i] = d L_i / d W has shape (batch, *W).
    sq_grad_sums = [np.zeros(v.shape, dtype=np.float64) for v in weight_vars]
    n_samples = int(X_scaled.shape[0])

    batch_size = 256
    for start in range(0, n_samples, batch_size):
        xb = X_t[start : start + batch_size]
        yb = y_t[start : start + batch_size]
        with tf.GradientTape() as tape:
            preds = model(xb, training=False)
            losses = per_sample_loss(preds, yb)  # (batch,)
        # jacobian of the per-sample loss vector w.r.t. each weight matrix:
        # result[k] has shape (batch, *weight_vars[k].shape).
        jac = tape.jacobian(losses, weight_vars)
        for k, jk in enumerate(jac):
            if jk is None:
                continue
            sq_grad_sums[k] += np.sum(
                np.square(np.asarray(jk, dtype=np.float64)), axis=0
            )

    saliencies: dict[int, np.ndarray] = {}
    for layer_index, (var, sq_sum) in enumerate(zip(weight_vars, sq_grad_sums)):
        h_diag = sq_sum / float(n_samples)  # mean over samples => H_ii
        w = np.asarray(network.layers[layer_index].weights, dtype=np.float64)
        saliencies[layer_index] = 0.5 * h_diag * np.square(w)

    return saliencies


def optimal_brain_damage(
    network: NetworkDefinition,
    X: ArrayLike,
    y: ArrayLike,
    fraction: float = 0.2,
    cleanup: bool = True,
    loss: str = "mse",
) -> PruneOutcome:
    """Prune the lowest-saliency weights via Optimal Brain Damage.

    Saliencies are computed with :func:`obd_saliencies` (empirical-Fisher
    diagonal Hessian, ``s_i = 0.5 * H_ii * w_i^2``). The smallest
    ``fraction`` of weights by saliency are set to zero. When ``cleanup``
    is ``True``, any HIDDEN neuron left fully disconnected -- all incoming
    weights zero (it can only contribute a constant) or all outgoing
    weights zero (it cannot influence the output) -- is removed EXACTLY
    via :func:`~pwlnet.pruning.structure.remove_hidden_neurons`, folding
    its constant post-activation contribution into the downstream bias.

    The zeroing itself is NOT function-preserving, so ``exact=False`` on
    the returned outcome; only the subsequent neuron cleanup is exact.

    Parameters
    ----------
    network : NetworkDefinition
        Network to prune (never mutated).
    X : array_like
        Inputs in ORIGINAL units used to estimate the Hessian diagonal.
    y : array_like
        Targets in ORIGINAL units.
    fraction : float, optional
        Fraction of weight entries (across all weight matrices, pooled)
        to zero, lowest saliency first. In ``[0, 1]``. Default ``0.2``.
    cleanup : bool, optional
        If ``True`` (default), remove hidden neurons left fully
        disconnected after zeroing, exactly folding their constant
        contributions downstream.
    loss : {"mse", "mae"}, optional
        Loss used for the saliency gradients. Default ``"mse"``.

    Returns
    -------
    PruneOutcome
        Outcome whose ``.network`` is the new pruned network.
        ``exact`` is ``False``. ``n_zeroed_weights`` counts the entries
        set to zero. ``removed_neurons`` records any neurons removed by
        cleanup. ``max_abs_output_change`` is a sampled estimate over the
        input box when ``network.input_bounds`` is available, else
        ``None``.

    Raises
    ------
    ValueError
        If ``fraction`` is outside ``[0, 1]``.
    pwlnet.utils.exceptions.PWLNetError
        If Keras/TensorFlow cannot be imported.
    """
    if not (0.0 <= float(fraction) <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")
    fraction = float(fraction)

    saliencies = obd_saliencies(network, X, y, loss=loss)

    # Work on a deep copy; we mutate its weight arrays in place to zero
    # the lowest-saliency entries (masking driven by saliency, not
    # magnitude).
    pruned = copy_network(network)

    # Pool every weight-matrix saliency to pick a single global threshold.
    pooled = np.concatenate(
        [saliencies[k].ravel() for k in sorted(saliencies)]
    )
    n_zeroed = 0
    notes: list[str] = []
    if pooled.size > 0 and fraction > 0.0:
        # The fraction-quantile of saliencies: entries <= cut are zeroed.
        cut = float(np.quantile(pooled, fraction))
        for layer_index in sorted(saliencies):
            sal = saliencies[layer_index]
            mask = sal <= cut
            n_zeroed += int(mask.sum())
            w = pruned.layers[layer_index].weights
            w[mask] = 0.0
    notes.append(
        f"OBD empirical-Fisher diagonal Hessian; loss={loss}; "
        f"zeroed {n_zeroed} weights at fraction={fraction}."
    )

    removed_neurons: dict[int, list[int]] = {}
    if cleanup:
        pruned, removed_neurons = _cleanup_disconnected(pruned)
        if removed_neurons:
            total = sum(len(v) for v in removed_neurons.values())
            notes.append(
                f"Cleanup removed {total} fully-disconnected hidden "
                f"neuron(s) exactly."
            )

    max_change = _sampled_max_change(network, pruned)

    return PruneOutcome(
        network=pruned,
        method="optimal_brain_damage",
        removed_neurons=removed_neurons,
        activation_changes={},
        n_zeroed_weights=int(n_zeroed),
        exact=False,
        max_abs_output_change=max_change,
        notes=notes,
    )


def _cleanup_disconnected(
    network: NetworkDefinition,
) -> tuple[NetworkDefinition, dict[int, list[int]]]:
    """Remove hidden neurons that are fully disconnected after zeroing.

    A hidden neuron is removed exactly via
    :func:`~pwlnet.pruning.structure.remove_hidden_neurons` if either:

    * all its INCOMING weights are zero (column of ``layers[k].weights``)
      -- its pre-activation is the constant bias, so its post-activation
      is the constant ``act(bias)`` over the whole box; or
    * all its OUTGOING weights are zero (row of ``layers[k+1].weights``)
      -- it cannot influence any downstream unit, so it can be removed
      with constant contribution ``0``.

    Each removal folds the neuron's constant post-activation value into
    the downstream bias, so the network function is preserved exactly.

    Parameters
    ----------
    network : NetworkDefinition
        Network whose hidden neurons are scanned (a fresh copy is
        returned; the input is not mutated beyond what its caller already
        owns).

    Returns
    -------
    tuple of (NetworkDefinition, dict[int, list[int]])
        The cleaned network and a mapping ``layer_index -> removed
        original neuron indices``.
    """
    from pwlnet.core.activations import get_activation

    current = network
    removed: dict[int, list[int]] = {}
    n_hidden = current.n_layers - 1  # output layer is the last
    for k in range(n_hidden):
        layer = current.layers[k]
        next_layer = current.layers[k + 1]
        w_in = layer.weights  # (n_in, n_neurons)
        w_out = next_layer.weights  # (n_neurons, n_out)
        act = get_activation(layer.activation)

        dead_in = np.all(w_in == 0.0, axis=0)  # no incoming signal
        dead_out = np.all(w_out == 0.0, axis=1)  # no outgoing signal

        idx: list[int] = []
        consts: list[float] = []
        for j in range(layer.n_outputs):
            if dead_in[j]:
                # pre-activation is the constant bias -> constant post-act.
                c = float(act(np.asarray([layer.biases[j]], dtype=float))[0])
                idx.append(j)
                consts.append(c)
            elif dead_out[j]:
                # neuron has no path to the output; removing it with any
                # constant (0) is exact because its row of W_{k+1} is zero.
                idx.append(j)
                consts.append(0.0)

        if idx:
            current = remove_hidden_neurons(
                current, k, idx, constant_outputs=consts
            )
            removed[k] = [int(j) for j in idx]

    return current, removed


def _sampled_max_change(
    original: NetworkDefinition,
    pruned: NetworkDefinition,
    n_samples: int = 2048,
    seed: int = 0,
) -> float | None:
    """Estimate the max abs output change over the input box by sampling.

    Parameters
    ----------
    original : NetworkDefinition
        The unpruned network.
    pruned : NetworkDefinition
        The pruned network.
    n_samples : int, optional
        Number of uniform samples drawn from the input box. Default
        ``2048``.
    seed : int, optional
        RNG seed for reproducibility. Default ``0``.

    Returns
    -------
    float or None
        Max absolute difference between the two networks' outputs over
        the sampled points, in ORIGINAL output units, or ``None`` if the
        network has no ``input_bounds`` to sample from.
    """
    if original.input_bounds is None:
        return None
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in original.input_bounds], dtype=float)
    ub = np.array([b[1] for b in original.input_bounds], dtype=float)
    samples = rng.uniform(lb, ub, size=(n_samples, original.n_inputs))
    y0 = original.predict(samples)
    y1 = pruned.predict(samples)
    return float(np.max(np.abs(y0 - y1)))
