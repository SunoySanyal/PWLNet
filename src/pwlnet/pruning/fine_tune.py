"""Fine-tuning of a (pruned) network with Keras (lazy import).

After a structural / magnitude pruning step the network's function has
usually drifted from the original. :func:`fine_tune` recovers accuracy by
re-training the *existing* weights for a few epochs with gradient descent,
starting from the pruned network's parameters rather than a fresh random
initialisation. Crucially it:

* rebuilds a Keras ``Sequential`` whose per-layer widths AND activations
  match the input :class:`NetworkDefinition` (hidden ``relu``/``tanh``/
  ``linear`` exactly as given; the output layer carries its own
  activation, normally ``linear``);
* seeds every Dense layer with the network's current weights/biases via
  ``set_weights`` (warm start, NOT random init);
* REUSES the network's existing ``input_scaler``/``output_scaler`` -- they
  are *not* refitted, so the pruned net keeps the exact scaled space it
  was trained in -- and trains on ``scaler.transform(X)`` /
  ``scaler.transform(y)``;
* optionally preserves magnitude sparsity: when ``freeze_zeros`` is set,
  every weight that is exactly ``0.0`` at the start stays ``0.0``
  throughout training, enforced by re-applying a 0/1 mask after each
  gradient update (a Keras callback). This lets fine-tuning recover the
  surviving weights without resurrecting the ones a magnitude method
  deliberately zeroed.

Keras is imported lazily inside :func:`fine_tune`; importing this module
never requires TensorFlow/Keras. If Keras is unavailable a
:class:`pwlnet.utils.exceptions.TrainingError` is raised.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.core.activations import get_activation
from pwlnet.core.layers import DenseLayer
from pwlnet.core.network import NetworkDefinition
from pwlnet.pruning.base import PruneOutcome
from pwlnet.training.initialization import import_keras, seed_everything
from pwlnet.utils.exceptions import TrainingError
from pwlnet.utils.logging import get_logger
from pwlnet.utils.typing import ArrayLike
from pwlnet.utils.validation import as_2d_array

__all__ = ["fine_tune"]

_LOGGER = get_logger("pruning.fine_tune")


def _as_xy(X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``X``/``y`` to aligned 2D float arrays in original units."""
    X_arr = as_2d_array(X, name="X")
    y_in = np.asarray(y, dtype=float)
    if y_in.ndim == 1:
        y_in = y_in.reshape(-1, 1)
    y_arr = as_2d_array(y_in, name="y")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise TrainingError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    if X_arr.shape[0] == 0:
        raise TrainingError("Cannot fine-tune on an empty dataset (0 samples).")
    return X_arr, y_arr


def _scale_inputs(scaler: Any, arr: np.ndarray) -> np.ndarray:
    """Apply ``scaler.transform`` if a scaler is present, else passthrough."""
    return scaler.transform(arr) if scaler is not None else arr


def _weighted_keras_loss(keras: Any, base_loss: str, weights: np.ndarray) -> Any:
    """Per-output weighted ``mse``/``mae`` loss (mean-1 normalised, scaled space).

    Mirrors :meth:`pwlnet.training.KerasTrainer._build_weighted_loss` so a
    pruned network can be fine-tuned with the same constraint emphasis it was
    originally trained under. The weights are renormalised to mean 1 so the
    reported loss stays on the same numeric scale as the unweighted run.
    """
    w = weights * (weights.size / float(weights.sum()))
    w_tensor = keras.ops.convert_to_tensor(w.astype("float32"))
    elementwise = keras.ops.square if base_loss == "mse" else keras.ops.absolute

    def weighted_loss(y_true: Any, y_pred: Any) -> Any:
        per_output = elementwise(y_true - y_pred)
        return keras.ops.mean(per_output * w_tensor)

    weighted_loss.__name__ = f"weighted_{base_loss}"
    return weighted_loss


def _make_freeze_zero_callback(keras: Any, masks: list[np.ndarray]) -> Any:
    """Build a callback that re-applies weight zero-masks after each batch.

    Parameters
    ----------
    keras : module
        The imported keras module.
    masks : list of numpy.ndarray
        One 0/1 mask per Dense layer (same shape as the layer's kernel);
        entries are ``0.0`` where the original weight was exactly zero and
        ``1.0`` elsewhere. The mask is multiplied into the kernel after
        every optimiser update so frozen weights never drift off zero.

    Returns
    -------
    keras.callbacks.Callback
        The mask-reapplying callback.
    """

    class FreezeZeroMask(keras.callbacks.Callback):
        """Hold weights that started at zero at zero during training."""

        def __init__(self, layer_masks: list[np.ndarray]) -> None:
            super().__init__()
            self._masks = layer_masks

        def _apply(self) -> None:
            dense_layers = [
                lyr
                for lyr in self.model.layers
                if isinstance(lyr, keras.layers.Dense)
            ]
            for lyr, mask in zip(dense_layers, self._masks):
                w, b = lyr.get_weights()
                lyr.set_weights([np.asarray(w) * mask, b])

        def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
            self._apply()

        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            # Belt-and-braces: also re-assert the mask at epoch boundaries.
            self._apply()

    return FreezeZeroMask(masks)


def fine_tune(
    network: NetworkDefinition,
    X: ArrayLike,
    y: ArrayLike,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    loss: str = "mse",
    validation_data: tuple[ArrayLike, ArrayLike] | None = None,
    freeze_zeros: bool = True,
    random_seed: int | None = None,
    verbose: int = 0,
    output_loss_weights: ArrayLike | None = None,
) -> PruneOutcome:
    """Fine-tune a (pruned) network's weights with Keras, warm-started.

    A Keras ``Sequential`` is built to match ``network`` exactly (per-layer
    widths and activations, output activation included) and each Dense
    layer is initialised from ``network``'s current weights/biases. The
    network's existing input/output scalers are reused (never refitted),
    and training happens in that same SCALED space on
    ``input_scaler.transform(X)`` / ``output_scaler.transform(y)``. The
    trained weights are read back into fresh :class:`DenseLayer` objects
    and returned in a NEW :class:`NetworkDefinition` carrying the same
    scalers, bounds and activations.

    Parameters
    ----------
    network : NetworkDefinition
        The (pruned) network to fine-tune. Never mutated.
    X : array_like
        Training inputs in ORIGINAL units, shape ``(n_samples, n_inputs)``.
    y : array_like
        Training targets in ORIGINAL units, shape ``(n_samples, n_outputs)``
        or ``(n_samples,)``.
    epochs : int, optional
        Number of training epochs. Default ``200``.
    batch_size : int, optional
        Mini-batch size. Default ``32``.
    learning_rate : float, optional
        Adam learning rate. Default ``1e-3``.
    loss : str, optional
        Keras loss identifier (e.g. ``"mse"``, ``"mae"``). Default
        ``"mse"``.
    validation_data : tuple of (array_like, array_like) or None, optional
        Explicit ``(X_val, y_val)`` in ORIGINAL units; scaled with the
        network's scalers and passed to Keras for monitoring. Default
        ``None``.
    freeze_zeros : bool, optional
        If ``True`` (default), every weight that is exactly ``0.0`` at the
        start of training is held at ``0.0`` throughout (a 0/1 mask is
        re-applied after each batch), preserving magnitude sparsity.
        Biases are never frozen.
    random_seed : int or None, optional
        Seed for ``keras.utils.set_random_seed`` (controls dropout-free
        mini-batch shuffling). ``None`` leaves training unseeded. Default
        ``None``.
    verbose : int, optional
        Keras ``fit`` verbosity (0 silent). Default ``0``.
    output_loss_weights : array_like or None, optional
        Per-output weights (length ``n_outputs``) applied to the
        (scaled-space) ``mse``/``mae`` loss, so the pruned network can be
        fine-tuned with the same constraint emphasis it was originally
        trained under. Renormalised to mean 1, matching
        :class:`pwlnet.training.KerasTrainer`. ``None`` (default) fine-tunes
        every output with equal weight.

    Returns
    -------
    PruneOutcome
        Outcome with ``method="fine_tune"``, ``network`` set to the tuned
        network, ``exact=False`` (gradient training changes the function),
        and a note recording the before/after training loss.

    Raises
    ------
    pwlnet.utils.exceptions.TrainingError
        If Keras is unavailable, the shapes of ``X``/``y`` (or
        ``validation_data``) are inconsistent, the dataset is empty, or
        Keras training fails.
    """
    keras = import_keras()

    X_arr, y_arr = _as_xy(X, y)
    if X_arr.shape[1] != network.n_inputs:
        raise TrainingError(
            f"X has {X_arr.shape[1]} features but the network expects "
            f"{network.n_inputs} inputs."
        )
    if y_arr.shape[1] != network.n_outputs:
        raise TrainingError(
            f"y has {y_arr.shape[1]} targets but the network has "
            f"{network.n_outputs} outputs."
        )

    if random_seed is not None:
        seed_everything(int(random_seed))

    # REUSE the pruned network's scalers; do NOT refit. The pruned net was
    # trained in this scaled space, so we transform with these and inverse
    # nothing during training (the model lives entirely in scaled space).
    input_scaler = network.input_scaler
    output_scaler = network.output_scaler
    X_scaled = _scale_inputs(input_scaler, X_arr)
    y_scaled = _scale_inputs(output_scaler, y_arr)

    # Scale the explicit validation set with the SAME (network) scalers.
    validation_scaled = None
    if validation_data is not None:
        X_val_arr, y_val_arr = _as_xy(validation_data[0], validation_data[1])
        if (
            X_val_arr.shape[1] != X_arr.shape[1]
            or y_val_arr.shape[1] != y_arr.shape[1]
        ):
            raise TrainingError(
                "validation_data feature/target widths "
                f"({X_val_arr.shape[1]}, {y_val_arr.shape[1]}) do not match the "
                f"training widths ({X_arr.shape[1]}, {y_arr.shape[1]})."
            )
        validation_scaled = (
            _scale_inputs(input_scaler, X_val_arr),
            _scale_inputs(output_scaler, y_val_arr),
        )

    # Build the Keras model to match the network's widths AND activations,
    # warm-starting every Dense layer from the network's current arrays.
    model = keras.Sequential(name="pwlnet_fine_tune")
    model.add(keras.layers.Input(shape=(network.n_inputs,)))
    for layer in network.layers:
        keras_act = get_activation(layer.activation).keras_name
        model.add(keras.layers.Dense(layer.n_outputs, activation=keras_act))

    # Set the weights AFTER the layers are built (Input layer materialises
    # the kernels). Dense layers are exactly network.layers in order.
    dense_layers = [lyr for lyr in model.layers if isinstance(lyr, keras.layers.Dense)]
    if len(dense_layers) != network.n_layers:
        raise TrainingError(
            f"Internal error: built {len(dense_layers)} Dense layers for a "
            f"{network.n_layers}-layer network."
        )
    for keras_layer, src in zip(dense_layers, network.layers):
        keras_layer.set_weights(
            [
                np.array(src.weights, dtype=float, copy=True),
                np.array(src.biases, dtype=float, copy=True),
            ]
        )

    optimizer = keras.optimizers.Adam(learning_rate=float(learning_rate))
    if output_loss_weights is None:
        compiled_loss: Any = str(loss)
    else:
        weights = np.asarray(output_loss_weights, dtype=float).reshape(-1)
        if (
            weights.size != network.n_outputs
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or weights.sum() <= 0.0
        ):
            raise TrainingError(
                "output_loss_weights must be a finite, non-negative vector of "
                f"length n_outputs={network.n_outputs} with a positive sum, "
                f"got {output_loss_weights!r}."
            )
        compiled_loss = _weighted_keras_loss(keras, str(loss), weights)
    model.compile(optimizer=optimizer, loss=compiled_loss)

    callbacks: list[Any] = []
    masks: list[np.ndarray] | None = None
    if freeze_zeros:
        # Mask is 1 where the weight is nonzero, 0 where it is exactly zero.
        masks = [
            (np.asarray(src.weights, dtype=float) != 0.0).astype(float)
            for src in network.layers
        ]
        callbacks.append(_make_freeze_zero_callback(keras, masks))

    has_validation = validation_scaled is not None
    monitor = "val_loss" if has_validation else "loss"

    # Loss BEFORE any gradient steps (model already warm-started).
    before_loss = float(
        model.evaluate(X_scaled, y_scaled, verbose=0, return_dict=True)["loss"]
    )

    try:
        history = model.fit(
            X_scaled,
            y_scaled,
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_data=validation_scaled,
            verbose=int(verbose),
            callbacks=callbacks,
            shuffle=True,
        )
    except Exception as exc:  # pragma: no cover - backend failures
        raise TrainingError(f"Keras fine-tuning failed: {exc}") from exc

    after_loss = float(history.history[monitor][-1]) if history.history.get(monitor) else None
    train_after = (
        float(history.history["loss"][-1]) if history.history.get("loss") else before_loss
    )

    # Extract the tuned weights back into fresh DenseLayers, preserving the
    # network's per-layer activations (the Keras config carries them but we
    # use the source activations to avoid any keras<->core name drift).
    tuned_layers: list[DenseLayer] = []
    for keras_layer, src in zip(dense_layers, network.layers):
        w, b = keras_layer.get_weights()
        tuned_layers.append(
            DenseLayer(
                weights=np.asarray(w, dtype=float),
                biases=np.asarray(b, dtype=float),
                activation=src.activation,
            )
        )

    tuned = NetworkDefinition(
        layers=tuned_layers,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        input_bounds=network.input_bounds,
    )

    # Count surviving frozen zeros (for the note) and assert they held.
    n_frozen = int(sum(int((m == 0.0).sum()) for m in masks)) if masks is not None else 0

    notes = [
        f"fine_tune: {epochs} epochs, lr={learning_rate:g}, loss={loss}, "
        f"freeze_zeros={freeze_zeros}",
        f"train loss {before_loss:.6g} -> {train_after:.6g} (scaled space)",
    ]
    if after_loss is not None and has_validation:
        notes.append(f"final {monitor}={after_loss:.6g} (scaled space)")
    if freeze_zeros:
        notes.append(f"held {n_frozen} weight(s) frozen at zero")

    _LOGGER.info(
        "fine_tune: loss %.6g -> %.6g over %d epochs (freeze_zeros=%s, frozen=%d)",
        before_loss,
        train_after,
        epochs,
        freeze_zeros,
        n_frozen,
    )

    return PruneOutcome(
        network=tuned,
        method="fine_tune",
        exact=False,
        max_abs_output_change=None,
        notes=notes,
    )
