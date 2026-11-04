"""Keras-based trainer that produces a core :class:`NetworkDefinition`.

The trainer fits :class:`pwlnet.core.scaling.OffsetScaler` objects to the
training inputs and targets, trains a Keras ``Sequential`` model entirely
in SCALED space, then extracts the trained weights into pure-numpy
:class:`pwlnet.core.layers.DenseLayer` objects. The returned
:class:`pwlnet.core.network.NetworkDefinition` carries the scalers, so
``NetworkDefinition.predict`` (input scaling -> forward pass -> inverse
output scaling) reproduces the Keras model's inverse-scaled predictions
to within float32 round-off (~1e-5).

With ``activation="pwl_tanh"`` the hidden layers are trained through the
SAME piecewise-linear tanh the MILP embedding builds, and are exported
labelled ``"tanh"`` so the embedding still builds its PWL block -- only
now the weights were fitted through that very block, so block and
network agree. The numpy reference for such a network is
:func:`pwlnet.formulations.pyomo_pwl.evaluate_tanh_pwl_network` (same
``breakpoints``), NOT ``NetworkDefinition.predict``, which reads the
exported layers with exact tanh. See :mod:`pwlnet.training.pwl_activation`.

Keras is imported lazily inside methods; importing this module never
requires TensorFlow/Keras.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.core.activations import get_activation
from pwlnet.core.layers import DenseLayer
from pwlnet.core.network import NetworkDefinition
from pwlnet.core.scaling import OffsetScaler
from pwlnet.formulations.pyomo_pwl.breakpoints import get_tanh_breakpoints
from pwlnet.formulations.pyomo_pwl.tanh_pwl_sos2 import evaluate_tanh_pwl_network
from pwlnet.training.callbacks import make_early_stopping, make_epoch_logger
from pwlnet.training.initialization import (
    get_initializer,
    import_keras,
    layer_seeds,
    seed_everything,
)
from pwlnet.training.pwl_activation import PWL_TANH, make_pwl_tanh_activation
from pwlnet.utils.exceptions import TrainingError
from pwlnet.utils.logging import get_logger
from pwlnet.utils.typing import ArrayLike
from pwlnet.utils.validation import as_2d_array

__all__ = ["KerasTrainer"]

_LOGGER = get_logger("training.keras_trainer")

_HIDDEN_ACTIVATIONS = ("relu", "tanh", PWL_TANH)

#: Activations trained in Keras under a different name than the one the
#: exported DenseLayer carries. ``pwl_tanh`` exports as ``"tanh"`` so the
#: PWL embedding builds its block for the layer it was trained through.
_EXPORT_ACTIVATIONS = {PWL_TANH: "tanh"}

#: Default breakpoint count for ``activation="pwl_tanh"``.
_DEFAULT_BREAKPOINTS = 5
_LOSSES = ("mse", "mae")
_SCALING_METHODS = ("standard", "minmax", "none")

#: Absolute-per-unit-scale tolerance for the keras <-> numpy equivalence check.
_EQUIVALENCE_TOL = 1e-5

#: Number of training samples used for the post-fit equivalence check.
_EQUIVALENCE_SAMPLES = 256


class KerasTrainer:
    """Train a dense feed-forward Keras network and export it to PWLNet core.

    Parameters
    ----------
    hidden_layers : list of int
        Widths of the hidden layers, e.g. ``[32, 32]``. Must be
        non-empty with positive entries.
    activation : {"relu", "tanh", "pwl_tanh"}, optional
        Activation for all hidden layers (the output layer is always
        linear). Default ``"relu"``. ``"pwl_tanh"`` trains through the
        piecewise-linear tanh of ``breakpoints`` and exports the hidden
        layers labelled ``"tanh"``, so a PWL embedding reproduces the
        trained model instead of approximating it; see
        :mod:`pwlnet.training.pwl_activation`.
    breakpoints : int or array_like or None, optional
        PWL-tanh breakpoints, only valid with
        ``activation="pwl_tanh"``: a count (``3``, ``5``, or ``7`` pick
        the standard SOSX-paper sets) or an explicit breakpoint vector
        spanning ``[-4, 4]``. Default None, which means 5 for
        ``"pwl_tanh"`` and is rejected as a stray option for the other
        activations. Pass the same value to the embedding.
    loss : {"mse", "mae"}, optional
        Training loss. Default ``"mse"``.
    optimizer : str or keras optimizer, optional
        Keras optimizer identifier (e.g. ``"adam"``, ``"sgd"``) or an
        already-built optimizer instance. Default ``"adam"``.
    learning_rate : float, optional
        Learning rate applied to the optimizer. Default ``1e-3``.
    epochs : int, optional
        Maximum number of training epochs. Default ``200``.
    batch_size : int, optional
        Mini-batch size. Default ``32``.
    validation_split : float, optional
        Fraction of the training data held out for validation, in
        ``[0, 1)``. Default ``0.1``.
    scale_inputs : bool, optional
        Fit an :class:`OffsetScaler` on ``X`` and train on scaled
        inputs. Default True.
    scale_outputs : bool, optional
        Fit an :class:`OffsetScaler` on ``y`` and train on scaled
        targets. Default True.
    scaling_method : {"standard", "minmax", "none"}, optional
        Method passed to :meth:`OffsetScaler.fit`. Default
        ``"standard"``.
    random_seed : int or None, optional
        Seed for ``keras.utils.set_random_seed`` plus deterministic
        per-layer initializer seeds. ``None`` leaves training unseeded.
    verbose : int, optional
        Keras ``fit`` verbosity (0 silent, 1 progress bar, 2 one line
        per epoch). Default 0.
    early_stopping_patience : int or None, optional
        If set, attach an early-stopping callback with this patience
        (monitoring ``val_loss`` when ``validation_split > 0``, else
        ``loss``) that restores the best weights. Default None
        (disabled).
    output_loss_weights : array_like or None, optional
        Per-output weights applied to the (scaled-space) ``mse``/``mae``
        loss, length ``n_outputs``. Use this to penalise some target
        columns more heavily than others (e.g. constraint outputs that
        the optimiser will treat as hard limits). The weights are
        renormalised to mean 1 so the reported loss stays on the same
        scale as the unweighted run. ``None`` (default) trains every
        output with equal weight.

    Attributes
    ----------
    model_ : keras.Model
        The trained Keras model (set by :meth:`fit`).
    history_ : dict of str to list of float
        Per-epoch training metrics from ``keras.Model.fit``.
    network_ : NetworkDefinition
        The exported network returned by :meth:`fit`.
    input_scaler_ : OffsetScaler or None
        Scaler fitted on ``X`` (None if ``scale_inputs=False``).
    output_scaler_ : OffsetScaler or None
        Scaler fitted on ``y`` (None if ``scale_outputs=False``).
    equivalence_error_ : float
        Max absolute difference between the numpy reference reading of
        ``network_`` and the inverse-scaled Keras predictions on (a
        subsample of) the training data. The reference is
        ``network_.predict`` for ``"relu"``/``"tanh"``, and
        ``evaluate_tanh_pwl_network(..., breakpoints=self.breakpoints)``
        for ``"pwl_tanh"`` -- for which ``network_.predict`` instead
        reads the exported layers with EXACT tanh and therefore differs
        by the PWL approximation error, by construction.
    """

    def __init__(
        self,
        hidden_layers: list[int],
        activation: str = "relu",
        breakpoints: int | ArrayLike | None = None,
        loss: str = "mse",
        optimizer: Any = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        validation_split: float = 0.1,
        scale_inputs: bool = True,
        scale_outputs: bool = True,
        scaling_method: str = "standard",
        random_seed: int | None = None,
        verbose: int = 0,
        early_stopping_patience: int | None = None,
        output_loss_weights: ArrayLike | None = None,
    ) -> None:
        self.hidden_layers = [int(width) for width in hidden_layers]
        if not self.hidden_layers or any(w <= 0 for w in self.hidden_layers):
            raise ValueError(
                f"hidden_layers must be a non-empty list of positive ints, "
                f"got {hidden_layers!r}."
            )
        act = str(activation).strip().lower()
        if act not in _HIDDEN_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(_HIDDEN_ACTIVATIONS)}, got {activation!r}."
            )
        self.activation = act
        if act == PWL_TANH:
            # Validate now so a bad breakpoint set fails before training.
            self.breakpoints: int | ArrayLike | None = (
                _DEFAULT_BREAKPOINTS if breakpoints is None else breakpoints
            )
            get_tanh_breakpoints(self.breakpoints)
        else:
            if breakpoints is not None:
                raise ValueError(
                    f"breakpoints is only valid with activation='{PWL_TANH}', "
                    f"got activation={activation!r}. Leave it unset rather than "
                    "have it silently ignored."
                )
            self.breakpoints = None
        loss_name = str(loss).strip().lower()
        if loss_name not in _LOSSES:
            raise ValueError(f"loss must be one of {list(_LOSSES)}, got {loss!r}.")
        self.loss = loss_name
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        if not 0.0 <= float(validation_split) < 1.0:
            raise ValueError(
                f"validation_split must be in [0, 1), got {validation_split!r}."
            )
        self.validation_split = float(validation_split)
        self.scale_inputs = bool(scale_inputs)
        self.scale_outputs = bool(scale_outputs)
        method = str(scaling_method).strip().lower()
        if method not in _SCALING_METHODS:
            raise ValueError(
                f"scaling_method must be one of {list(_SCALING_METHODS)}, "
                f"got {scaling_method!r}."
            )
        self.scaling_method = method
        self.random_seed = None if random_seed is None else int(random_seed)
        self.verbose = int(verbose)
        if early_stopping_patience is not None and int(early_stopping_patience) < 1:
            raise ValueError(
                f"early_stopping_patience must be a positive int or None, "
                f"got {early_stopping_patience!r}."
            )
        self.early_stopping_patience = (
            None if early_stopping_patience is None else int(early_stopping_patience)
        )
        if output_loss_weights is None:
            self.output_loss_weights: np.ndarray | None = None
        else:
            weights = np.asarray(output_loss_weights, dtype=float).reshape(-1)
            if (
                weights.size == 0
                or not np.all(np.isfinite(weights))
                or np.any(weights < 0.0)
                or weights.sum() <= 0.0
            ):
                raise ValueError(
                    "output_loss_weights must be a non-empty vector of "
                    "finite, non-negative weights with a positive sum, "
                    f"got {output_loss_weights!r}."
                )
            self.output_loss_weights = weights

        # Set by fit().
        self.model_: Any = None
        self.history_: dict[str, list[float]] | None = None
        self.network_: NetworkDefinition | None = None
        self.input_scaler_: OffsetScaler | None = None
        self.output_scaler_: OffsetScaler | None = None
        self.equivalence_error_: float | None = None

    @property
    def export_activation(self) -> str:
        """Activation name the exported hidden ``DenseLayer``s carry.

        Equal to :attr:`activation` except for ``"pwl_tanh"``, which
        exports as ``"tanh"``: the core representation has no separate
        PWL activation, and labelling it ``"tanh"`` is what makes the
        PWL embedding build a block for a layer trained through that
        very block.
        """
        return _EXPORT_ACTIVATIONS.get(self.activation, self.activation)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> NetworkDefinition:
        """Train on ``(X, y)`` and export the model as a NetworkDefinition.

        Parameters
        ----------
        X : array_like
            Inputs in ORIGINAL units, shape ``(n_samples, n_inputs)``
            (a 1D array is treated as a single sample).
        y : array_like
            Targets in ORIGINAL units, shape ``(n_samples, n_outputs)``
            or ``(n_samples,)`` (promoted to one output column).
        validation_data : tuple of (array_like, array_like) or None, optional
            Explicit ``(X_val, y_val)`` validation set in ORIGINAL
            units. When provided it is scaled with the scalers fitted on
            the training data and passed to Keras directly, overriding
            ``validation_split`` (Keras forbids using both). This makes
            the validation split fixed and known to the caller, so the
            same partition can be scored after training. Default
            ``None`` (fall back to the internal ``validation_split``).

        Returns
        -------
        NetworkDefinition
            Pure-numpy network carrying the fitted input/output scalers
            and per-input bounds taken from the training-data min/max.
            ``NetworkDefinition.predict(X)`` matches the Keras model's
            inverse-scaled predictions to ~1e-5.

        Raises
        ------
        pwlnet.utils.exceptions.TrainingError
            If Keras is unavailable, the shapes of ``X`` and ``y`` (or
            of ``validation_data``) are inconsistent, or Keras training
            fails.
        """
        keras = import_keras()

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
            raise TrainingError("Cannot train on an empty dataset (0 samples).")

        if self.random_seed is not None:
            seed_everything(self.random_seed)

        # Fit scalers in ORIGINAL space; the Keras model lives fully in
        # SCALED space.
        self.input_scaler_ = (
            OffsetScaler.fit(X_arr, method=self.scaling_method)
            if self.scale_inputs
            else None
        )
        self.output_scaler_ = (
            OffsetScaler.fit(y_arr, method=self.scaling_method)
            if self.scale_outputs
            else None
        )
        X_scaled = (
            self.input_scaler_.transform(X_arr) if self.input_scaler_ is not None else X_arr
        )
        y_scaled = (
            self.output_scaler_.transform(y_arr)
            if self.output_scaler_ is not None
            else y_arr
        )

        # Scale an explicit validation set with the TRAINING scalers so
        # it lives in the same scaled space as the model.
        validation_scaled = None
        if validation_data is not None:
            X_val_arr = as_2d_array(validation_data[0], name="X_val")
            y_val_in = np.asarray(validation_data[1], dtype=float)
            if y_val_in.ndim == 1:
                y_val_in = y_val_in.reshape(-1, 1)
            y_val_arr = as_2d_array(y_val_in, name="y_val")
            if X_val_arr.shape[0] != y_val_arr.shape[0]:
                raise TrainingError(
                    f"validation X has {X_val_arr.shape[0]} samples but "
                    f"validation y has {y_val_arr.shape[0]}."
                )
            if X_val_arr.shape[1] != X_arr.shape[1] or y_val_arr.shape[1] != y_arr.shape[1]:
                raise TrainingError(
                    "validation_data feature/target widths "
                    f"({X_val_arr.shape[1]}, {y_val_arr.shape[1]}) do not match "
                    f"the training widths ({X_arr.shape[1]}, {y_arr.shape[1]})."
                )
            X_val_scaled = (
                self.input_scaler_.transform(X_val_arr)
                if self.input_scaler_ is not None
                else X_val_arr
            )
            y_val_scaled = (
                self.output_scaler_.transform(y_val_arr)
                if self.output_scaler_ is not None
                else y_val_arr
            )
            validation_scaled = (X_val_scaled, y_val_scaled)

        model = self._build_model(keras, X_arr.shape[1], y_arr.shape[1])

        callbacks: list[Any] = [
            make_epoch_logger(log_every=max(1, self.epochs // 10), logger=_LOGGER)
        ]
        # Keras records val_loss when either an explicit validation set
        # or a positive internal split is present.
        has_validation = validation_scaled is not None or self.validation_split > 0.0
        if self.early_stopping_patience is not None:
            monitor = "val_loss" if has_validation else "loss"
            callbacks.append(
                make_early_stopping(
                    patience=self.early_stopping_patience, monitor=monitor
                )
            )

        # Keras forbids passing both validation_data and a positive
        # validation_split; the explicit set takes precedence.
        fit_validation_split = 0.0 if validation_scaled is not None else self.validation_split

        try:
            history = model.fit(
                X_scaled,
                y_scaled,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=fit_validation_split,
                validation_data=validation_scaled,
                verbose=self.verbose,
                callbacks=callbacks,
                shuffle=True,
            )
        except Exception as exc:  # pragma: no cover - backend failures
            raise TrainingError(f"Keras training failed: {exc}") from exc

        self.model_ = model
        self.history_ = {
            key: [float(v) for v in values] for key, values in history.history.items()
        }

        input_bounds = [
            (float(lo), float(hi))
            for lo, hi in zip(X_arr.min(axis=0), X_arr.max(axis=0))
        ]
        network = NetworkDefinition(
            layers=self._extract_layers(keras, model),
            input_scaler=self.input_scaler_,
            output_scaler=self.output_scaler_,
            input_bounds=input_bounds,
        )
        self.network_ = network
        self.equivalence_error_ = self._check_equivalence(network, X_arr)
        return network

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_model(self, keras: Any, n_inputs: int, n_outputs: int) -> Any:
        """Build and compile the Keras ``Sequential`` model in scaled space."""
        seeds = layer_seeds(self.random_seed, len(self.hidden_layers) + 1)
        keras_hidden_act: Any = (
            make_pwl_tanh_activation(self.breakpoints)
            if self.activation == PWL_TANH
            else get_activation(self.activation).keras_name
        )

        model = keras.Sequential(name="pwlnet_dense")
        model.add(keras.layers.Input(shape=(n_inputs,)))
        for width, seed in zip(self.hidden_layers, seeds[:-1]):
            model.add(
                keras.layers.Dense(
                    width,
                    activation=keras_hidden_act,
                    kernel_initializer=get_initializer("glorot_uniform", seed),
                    bias_initializer="zeros",
                )
            )
        model.add(
            keras.layers.Dense(
                n_outputs,
                activation="linear",
                kernel_initializer=get_initializer("glorot_uniform", seeds[-1]),
                bias_initializer="zeros",
            )
        )

        optimizer = (
            keras.optimizers.get(self.optimizer)
            if isinstance(self.optimizer, str)
            else self.optimizer
        )
        try:
            optimizer.learning_rate = self.learning_rate
        except AttributeError:  # pragma: no cover - exotic optimizer objects
            _LOGGER.warning(
                "Could not set learning_rate on optimizer %r; using its default.",
                self.optimizer,
            )
        loss = (
            self.loss
            if self.output_loss_weights is None
            else self._build_weighted_loss(keras, n_outputs)
        )
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def _build_weighted_loss(self, keras: Any, n_outputs: int) -> Any:
        """Build a per-output weighted ``mse``/``mae`` loss (scaled space).

        The weights are renormalised to mean 1 so the weighted mean over
        outputs keeps the loss on the same numeric scale as the
        unweighted ``mse``/``mae`` (the loss curves remain comparable).
        """
        if self.output_loss_weights.size != n_outputs:
            raise TrainingError(
                f"output_loss_weights has length {self.output_loss_weights.size} "
                f"but the network has {n_outputs} outputs."
            )
        weights = self.output_loss_weights * (
            self.output_loss_weights.size / float(self.output_loss_weights.sum())
        )
        w_tensor = keras.ops.convert_to_tensor(weights, dtype="float32")
        elementwise = keras.ops.square if self.loss == "mse" else keras.ops.absolute

        def weighted_loss(y_true: Any, y_pred: Any) -> Any:
            per_output = elementwise(y_true - y_pred)
            return keras.ops.mean(per_output * w_tensor)

        weighted_loss.__name__ = f"weighted_{self.loss}"
        return weighted_loss

    def _extract_layers(self, keras: Any, model: Any) -> list[DenseLayer]:
        """Copy trained Keras Dense layers into core DenseLayer objects.

        The activation label comes from :attr:`export_activation` rather
        than from the Keras layer config: a callable activation (as
        ``"pwl_tanh"`` uses) has no name the core representation would
        accept, and the mapping to ``"tanh"`` is deliberate.
        """
        dense = [
            layer for layer in model.layers if isinstance(layer, keras.layers.Dense)
        ]
        if not dense:
            raise TrainingError("No Dense layers found in the trained Keras model.")
        layers: list[DenseLayer] = []
        for index, layer in enumerate(dense):
            weights, biases = layer.get_weights()
            is_output = index == len(dense) - 1
            layers.append(
                DenseLayer(
                    weights=np.asarray(weights, dtype=float),
                    biases=np.asarray(biases, dtype=float),
                    activation="linear" if is_output else self.export_activation,
                )
            )
        return layers

    def _check_equivalence(
        self, network: NetworkDefinition, X_arr: np.ndarray
    ) -> float:
        """Compare the exported network against the Keras model.

        Returns the max absolute difference on (a subsample of) the
        training data; logs a warning if it exceeds the documented
        ~1e-5 tolerance (relative to the prediction magnitude). For
        ``"pwl_tanh"`` the exported network is read through the PWL
        (:func:`evaluate_tanh_pwl_network`), which is the function the
        Keras model was actually trained with -- reading it with exact
        tanh would report the PWL approximation error instead.
        """
        X_check = X_arr[: min(X_arr.shape[0], _EQUIVALENCE_SAMPLES)]
        X_keras = (
            self.input_scaler_.transform(X_check)
            if self.input_scaler_ is not None
            else X_check
        )
        y_keras = np.asarray(self.model_.predict(X_keras, verbose=0), dtype=float)
        if self.output_scaler_ is not None:
            y_keras = self.output_scaler_.inverse_transform(y_keras)
        y_network = (
            evaluate_tanh_pwl_network(network, X_check, breakpoints=self.breakpoints)
            if self.activation == PWL_TANH
            else network.predict(X_check)
        )
        diff = float(np.max(np.abs(y_network - y_keras)))
        scale = max(1.0, float(np.max(np.abs(y_keras))))
        if diff > _EQUIVALENCE_TOL * scale:
            _LOGGER.warning(
                "NetworkDefinition/Keras prediction mismatch: max abs diff %.3e "
                "exceeds tolerance %.3e.",
                diff,
                _EQUIVALENCE_TOL * scale,
            )
        return diff

    def __repr__(self) -> str:
        pwl = "" if self.breakpoints is None else f"breakpoints={self.breakpoints}, "
        return (
            f"KerasTrainer(hidden_layers={self.hidden_layers}, "
            f"activation='{self.activation}', {pwl}loss='{self.loss}', "
            f"epochs={self.epochs}, fitted={self.network_ is not None})"
        )
