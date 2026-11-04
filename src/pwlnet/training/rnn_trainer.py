"""Keras trainer for a one-step ReLU RNN, exported to a RecurrentCell.

Trains a vanilla Elman recurrent network -- a single ``keras.SimpleRNN``
layer with ``activation="relu"`` (NO LSTM/GRU/tanh/sigmoid, so the cell
stays piecewise linear and MILP-exact) followed by a linear ``Dense``
readout applied at every step. The trained weights are extracted into a
pure-numpy :class:`pwlnet.recurrent.cell.RecurrentCell`, which embeds and
unrolls without any Keras dependency.

The model is trained entirely in SCALED space (exogenous inputs scaled by
an :class:`OffsetScaler`, targets scaled by another); both scalers are
folded into the exported cell so that ``cell`` consumes original-unit
controls and produces original-unit outputs.

Keras is imported lazily so importing this module never requires
TensorFlow/Keras.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.core.scaling import OffsetScaler
from pwlnet.recurrent.cell import RecurrentCell
from pwlnet.training.callbacks import make_early_stopping, make_epoch_logger
from pwlnet.training.initialization import (
    get_initializer,
    import_keras,
    layer_seeds,
    seed_everything,
)
from pwlnet.utils.exceptions import TrainingError
from pwlnet.utils.logging import get_logger

__all__ = ["RNNTrainer"]

_LOGGER = get_logger("training.rnn_trainer")

_LOSSES = ("mse", "mae")
_SCALING_METHODS = ("standard", "minmax", "none")

#: Per-unit-scale tolerance for the keras <-> numpy rollout equivalence check.
_EQUIVALENCE_TOL = 1e-4


class RNNTrainer:
    """Train a ReLU ``SimpleRNN`` and export a :class:`RecurrentCell`.

    Parameters
    ----------
    hidden_units : int
        Width of the recurrent hidden state ``h`` (``SimpleRNN`` units).
    loss : {"mse", "mae"}, optional
        Training loss (scaled space). Default ``"mse"``.
    optimizer : str or keras optimizer, optional
        Keras optimizer identifier or instance. Default ``"adam"``.
    learning_rate : float, optional
        Learning rate applied to the optimizer. Default ``1e-3``.
    epochs : int, optional
        Maximum training epochs. Default ``200``.
    batch_size : int, optional
        Mini-batch size (in sequences). Default ``32``.
    validation_split : float, optional
        Fraction of sequences held out for validation, in ``[0, 1)``.
        Default ``0.1``.
    scale_inputs, scale_outputs : bool, optional
        Fit and apply :class:`OffsetScaler` transforms on the exogenous
        inputs / targets. Both default ``True``.
    scaling_method : {"standard", "minmax", "none"}, optional
        Method passed to :meth:`OffsetScaler.fit`. Default ``"standard"``.
    bounds_margin : float, optional
        Fractional padding added to the data-derived exogenous and hidden
        boxes (so the embedding box is a strict superset of the training
        range). Default ``0.0``.
    output_bounds : list of (float, float) or None, optional
        Physical ``(lb, ub)`` box for each observable output (original
        units). When given together with a positive ``bound_penalty_weight``
        the loss gains a soft penalty
        ``mean(relu(y - ub)^2 + relu(lb - y)^2)`` that pushes every
        predicted output back into its physical range -- e.g. keeping
        predicted concentrations non-negative and below their caps.
    bound_penalty_weight : float, optional
        Weight on the output-bound penalty. Keep it HIGH to hard-enforce
        physical limits on a small network. Default ``0.0`` (disabled).
    physics_residual : callable or None, optional
        ``physics_residual(tf, U_orig, Y_orig) -> residual_tensor``
        evaluated in ORIGINAL units over the predicted sequences
        (``U_orig``/``Y_orig`` have shape ``(batch, T, n_exog/n_out)``).
        Return any tensor whose mean square is the penalty -- typically a
        discretised process-model (mass/energy balance) residual, so the
        network learns dynamics consistent with known physics. Default
        ``None``.
    physics_weight : float, optional
        Weight on the ``physics_residual`` penalty. Default ``0.0``.
    random_seed : int or None, optional
        Seed for reproducible initialisation. Default ``None``.
    verbose : int, optional
        If positive, print periodic progress rows (and Keras ``fit``
        verbosity on the pure-data path). Default ``0``.
    early_stopping_patience : int or None, optional
        If set, attach early stopping (restoring best weights).

    Notes
    -----
    When either physics term is active, training switches from
    ``keras.Model.fit`` to a custom ``tf.GradientTape`` loop (mirroring
    :class:`pwlnet.training.PhysicsInformedTrainer`) so the data loss and
    each physics penalty are tracked separately in ``history_`` -- the
    keys ``data_loss``, ``bound_loss`` and ``physics_loss`` (plus their
    ``val_`` counterparts) are available for plotting.

    Attributes
    ----------
    model_ : keras.Model
        The trained Keras model (set by :meth:`fit`).
    history_ : dict
        Per-epoch metrics from ``keras.Model.fit``.
    cell_ : RecurrentCell
        The exported cell returned by :meth:`fit`.
    exog_scaler_, output_scaler_ : OffsetScaler or None
        The fitted scalers.
    equivalence_error_ : float
        Max abs difference between the cell's numpy rollout and the Keras
        model on a subsample of the training sequences.
    """

    def __init__(
        self,
        hidden_units: int,
        *,
        loss: str = "mse",
        optimizer: Any = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        validation_split: float = 0.1,
        scale_inputs: bool = True,
        scale_outputs: bool = True,
        scaling_method: str = "standard",
        bounds_margin: float = 0.0,
        output_bounds: list[tuple[float, float]] | None = None,
        bound_penalty_weight: float = 0.0,
        physics_residual: Any | None = None,
        physics_weight: float = 0.0,
        random_seed: int | None = None,
        verbose: int = 0,
        early_stopping_patience: int | None = None,
    ) -> None:
        self.hidden_units = int(hidden_units)
        if self.hidden_units <= 0:
            raise ValueError(f"hidden_units must be positive, got {hidden_units!r}.")
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
        self.bounds_margin = float(bounds_margin)
        self.output_bounds = (
            None
            if output_bounds is None
            else [(float(lb), float(ub)) for lb, ub in output_bounds]
        )
        self.bound_penalty_weight = float(bound_penalty_weight)
        self.physics_residual = physics_residual
        self.physics_weight = float(physics_weight)
        self.random_seed = None if random_seed is None else int(random_seed)
        self.verbose = int(verbose)
        if early_stopping_patience is not None and int(early_stopping_patience) < 1:
            raise ValueError(
                "early_stopping_patience must be a positive int or None, got "
                f"{early_stopping_patience!r}."
            )
        self.early_stopping_patience = (
            None if early_stopping_patience is None else int(early_stopping_patience)
        )

        # Set by fit().
        self.model_: Any = None
        self.history_: dict[str, list[float]] | None = None
        self.cell_: RecurrentCell | None = None
        self.exog_scaler_: OffsetScaler | None = None
        self.output_scaler_: OffsetScaler | None = None
        self.equivalence_error_: float | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        U: ArrayLike3D,
        Y: ArrayLike3D,
        validation_data: tuple[ArrayLike3D, ArrayLike3D] | None = None,
    ) -> RecurrentCell:
        """Train on sequence data and export a :class:`RecurrentCell`.

        Parameters
        ----------
        U : array_like
            Exogenous input sequences, shape ``(n_seq, T, n_exog)`` in
            ORIGINAL units.
        Y : array_like
            Target output sequences, shape ``(n_seq, T, n_out)`` (or
            ``(n_seq, T)`` for a single output) in ORIGINAL units;
            ``Y[s, t]`` is the observable after applying ``U[s, t]``.
        validation_data : tuple or None, optional
            Explicit ``(U_val, Y_val)`` validation sequences (original
            units), scaled with the training scalers and passed to Keras.

        Returns
        -------
        RecurrentCell
            The exported cell (also stored on ``self.cell_``).

        Raises
        ------
        pwlnet.utils.exceptions.TrainingError
            If Keras is unavailable, the data shapes are inconsistent, or
            training fails.
        """
        keras = import_keras()

        U_arr = _as_3d(U, name="U")
        Y_arr = _as_3d(Y, name="Y")
        if U_arr.shape[0] != Y_arr.shape[0] or U_arr.shape[1] != Y_arr.shape[1]:
            raise TrainingError(
                f"U {U_arr.shape} and Y {Y_arr.shape} must share (n_seq, T)."
            )
        if U_arr.shape[0] == 0:
            raise TrainingError("Cannot train on an empty dataset (0 sequences).")
        n_exog = U_arr.shape[2]
        n_out = Y_arr.shape[2]

        if self.random_seed is not None:
            seed_everything(self.random_seed)

        # Fit scalers in ORIGINAL space on the flattened (n_seq*T, features)
        # views, then scale the sequences for training.
        self.exog_scaler_ = (
            OffsetScaler.fit(_flatten(U_arr), method=self.scaling_method)
            if self.scale_inputs
            else None
        )
        self.output_scaler_ = (
            OffsetScaler.fit(_flatten(Y_arr), method=self.scaling_method)
            if self.scale_outputs
            else None
        )
        U_scaled = _scale_seq(U_arr, self.exog_scaler_)
        Y_scaled = _scale_seq(Y_arr, self.output_scaler_)

        validation_scaled = None
        if validation_data is not None:
            U_val = _as_3d(validation_data[0], name="U_val")
            Y_val = _as_3d(validation_data[1], name="Y_val")
            if U_val.shape[2] != n_exog or Y_val.shape[2] != n_out:
                raise TrainingError(
                    "validation_data feature widths do not match the training "
                    "widths."
                )
            validation_scaled = (
                _scale_seq(U_val, self.exog_scaler_),
                _scale_seq(Y_val, self.output_scaler_),
            )

        model = self._build_model(keras, n_exog, n_out)

        # Pure data loss -> keras.fit; with a physics/bound penalty we need
        # a custom loop that evaluates the penalty in original units.
        if self._physics_active():
            self.history_ = self._fit_with_physics(
                keras, model, U_scaled, Y_scaled, validation_scaled
            )
        else:
            self.history_ = self._fit_data_only(
                keras, model, U_scaled, Y_scaled, validation_scaled
            )
        self.model_ = model

        cell = self._export_cell(keras, model, U_arr, Y_arr)
        self.cell_ = cell
        self.equivalence_error_ = self._check_equivalence(cell, U_scaled, model)
        return cell

    def _physics_active(self) -> bool:
        """True if a bound or physics-residual penalty should be applied."""
        bound = self.output_bounds is not None and self.bound_penalty_weight > 0.0
        physics = self.physics_residual is not None and self.physics_weight > 0.0
        return bool(bound or physics)

    def _fit_data_only(
        self,
        keras: Any,
        model: Any,
        U_scaled: np.ndarray,
        Y_scaled: np.ndarray,
        validation_scaled: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, list[float]]:
        """Train with plain scaled-space data loss via ``keras.Model.fit``."""
        callbacks: list[Any] = [
            make_epoch_logger(log_every=max(1, self.epochs // 10), logger=_LOGGER)
        ]
        has_validation = validation_scaled is not None or self.validation_split > 0.0
        if self.early_stopping_patience is not None:
            monitor = "val_loss" if has_validation else "loss"
            callbacks.append(
                make_early_stopping(
                    patience=self.early_stopping_patience, monitor=monitor
                )
            )
        fit_validation_split = (
            0.0 if validation_scaled is not None else self.validation_split
        )
        try:
            history = model.fit(
                U_scaled,
                Y_scaled,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=fit_validation_split,
                validation_data=validation_scaled,
                verbose=self.verbose,
                callbacks=callbacks,
                shuffle=True,
            )
        except Exception as exc:  # pragma: no cover - backend failures
            raise TrainingError(f"Keras RNN training failed: {exc}") from exc
        return {key: [float(v) for v in vals] for key, vals in history.history.items()}

    def _fit_with_physics(
        self,
        keras: Any,
        model: Any,
        U_scaled: np.ndarray,
        Y_scaled: np.ndarray,
        validation_scaled: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, list[float]]:
        """Custom ``tf.GradientTape`` loop: data loss + physics/bound penalties.

        The data loss lives in SCALED space (matching the pure-data path);
        the penalties are evaluated in ORIGINAL units after unscaling the
        predictions/inputs inside the graph, so physical bounds and the
        process-model residual are expressed in the natural units of the
        problem. Best (lowest validation total-loss) weights are restored.
        """
        import tensorflow as tf

        if self.random_seed is not None:
            seed_everything(self.random_seed)

        # Carve a validation split when no explicit set was supplied.
        if validation_scaled is None and self.validation_split > 0.0:
            n_val = max(1, int(round(self.validation_split * U_scaled.shape[0])))
            U_val, Y_val = U_scaled[:n_val], Y_scaled[:n_val]
            U_tr, Y_tr = U_scaled[n_val:], Y_scaled[n_val:]
        elif validation_scaled is not None:
            U_tr, Y_tr = U_scaled, Y_scaled
            U_val, Y_val = validation_scaled
        else:
            U_tr, Y_tr = U_scaled, Y_scaled
            U_val, Y_val = U_scaled, Y_scaled

        optimizer = (
            keras.optimizers.get(self.optimizer)
            if isinstance(self.optimizer, str)
            else self.optimizer
        )
        try:
            optimizer.learning_rate = self.learning_rate
        except AttributeError:  # pragma: no cover
            pass

        c = self._physics_constants(tf)
        dataset = (
            tf.data.Dataset.from_tensor_slices(
                (U_tr.astype("float32"), Y_tr.astype("float32"))
            )
            .shuffle(len(U_tr), seed=self.random_seed, reshuffle_each_iteration=True)
            .batch(self.batch_size)
        )

        @tf.function
        def train_step(ub, yb):
            with tf.GradientTape() as tape:
                pred = model(ub, training=True)
                losses = self._loss_terms(tf, ub, yb, pred, c)
            grads = tape.gradient(losses["loss"], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return losses

        def evaluate(u_eval, y_eval) -> dict[str, float]:
            pred = model(tf.constant(u_eval, dtype=tf.float32), training=False)
            losses = self._loss_terms(
                tf,
                tf.constant(u_eval, dtype=tf.float32),
                tf.constant(y_eval, dtype=tf.float32),
                pred,
                c,
            )
            return {k: float(v.numpy()) for k, v in losses.items()}

        rows: list[dict[str, float]] = []
        best_val, best_weights, wait = np.inf, None, 0
        log_every = max(1, self.epochs // 10)
        for epoch in range(1, self.epochs + 1):
            batch_rows = [
                {k: float(v.numpy()) for k, v in train_step(ub, yb).items()}
                for ub, yb in dataset
            ]
            train_row = {
                key: float(np.mean([r[key] for r in batch_rows])) for key in batch_rows[0]
            }
            val_row = evaluate(U_val, Y_val)
            rows.append(
                {"epoch": float(epoch), **train_row, **{f"val_{k}": v for k, v in val_row.items()}}
            )
            if val_row["loss"] < best_val - 1e-9:
                best_val, best_weights, wait = val_row["loss"], model.get_weights(), 0
            else:
                wait += 1
            if self.verbose and (epoch == 1 or epoch % log_every == 0):
                print(
                    f"epoch {epoch:4d}: loss={train_row['loss']:.6g} "
                    f"data={train_row['data_loss']:.6g} val_loss={val_row['loss']:.6g}"
                )
            if (
                self.early_stopping_patience is not None
                and wait >= self.early_stopping_patience
            ):
                break

        if best_weights is not None:
            model.set_weights(best_weights)
        return {key: [float(r[key]) for r in rows] for key in rows[0]}

    def _physics_constants(self, tf: Any) -> dict[str, Any]:
        """Build the float32 tensors reused by the physics/bound loss terms."""
        x_offset = (
            self.exog_scaler_.offset.astype("float32")
            if self.exog_scaler_ is not None
            else None
        )
        x_factor = (
            self.exog_scaler_.factor.astype("float32")
            if self.exog_scaler_ is not None
            else None
        )
        y_offset = (
            self.output_scaler_.offset.astype("float32")
            if self.output_scaler_ is not None
            else None
        )
        y_factor = (
            self.output_scaler_.factor.astype("float32")
            if self.output_scaler_ is not None
            else None
        )
        const: dict[str, Any] = {
            "x_offset": None if x_offset is None else tf.constant(x_offset),
            "x_factor": None if x_factor is None else tf.constant(x_factor),
            "y_offset": None if y_offset is None else tf.constant(y_offset),
            "y_factor": None if y_factor is None else tf.constant(y_factor),
        }
        if self.output_bounds is not None:
            lb = np.array([b[0] for b in self.output_bounds], dtype="float32")
            ub = np.array([b[1] for b in self.output_bounds], dtype="float32")
            const["lb"] = tf.constant(lb)
            const["ub"] = tf.constant(ub)
        return const

    def _loss_terms(self, tf: Any, ub: Any, yb_true: Any, yb_pred: Any, c: dict) -> dict:
        """Compose the total loss = scaled data MSE + weighted penalties."""
        data_loss = tf.reduce_mean(tf.square(yb_pred - yb_true))
        terms: dict[str, Any] = {"data_loss": data_loss}
        total = data_loss

        # Unscale to original units for the physics-aware penalties.
        y_orig = yb_pred if c["y_factor"] is None else yb_pred * c["y_factor"] + c["y_offset"]
        u_orig = ub if c["x_factor"] is None else ub * c["x_factor"] + c["x_offset"]

        if "ub" in c and self.bound_penalty_weight > 0.0:
            over = tf.nn.relu(y_orig - c["ub"])
            under = tf.nn.relu(c["lb"] - y_orig)
            bound_loss = tf.reduce_mean(tf.square(over) + tf.square(under))
            terms["bound_loss"] = bound_loss
            total = total + self.bound_penalty_weight * bound_loss

        if self.physics_residual is not None and self.physics_weight > 0.0:
            residual = self.physics_residual(tf, u_orig, y_orig)
            physics_loss = tf.reduce_mean(tf.square(residual))
            terms["physics_loss"] = physics_loss
            total = total + self.physics_weight * physics_loss

        terms["loss"] = total
        return terms

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_model(self, keras: Any, n_exog: int, n_out: int) -> Any:
        """Build and compile the SimpleRNN(relu) + linear Dense readout model."""
        seeds = layer_seeds(self.random_seed, 2)
        model = keras.Sequential(name="pwlnet_relu_rnn")
        # (None, n_exog): variable-length sequences of exogenous inputs.
        model.add(keras.layers.Input(shape=(None, n_exog)))
        model.add(
            keras.layers.SimpleRNN(
                self.hidden_units,
                activation="relu",  # piecewise linear -> MILP-exact
                return_sequences=True,  # one hidden state per timestep
                kernel_initializer=get_initializer("glorot_uniform", seeds[0]),
                recurrent_initializer=get_initializer("orthogonal", seeds[0]),
                bias_initializer="zeros",
            )
        )
        # Linear readout applied at every timestep.
        model.add(
            keras.layers.TimeDistributed(
                keras.layers.Dense(
                    n_out,
                    activation="linear",
                    kernel_initializer=get_initializer("glorot_uniform", seeds[1]),
                    bias_initializer="zeros",
                )
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
            _LOGGER.warning("Could not set learning_rate on optimizer %r.", self.optimizer)
        model.compile(optimizer=optimizer, loss=self.loss)
        return model

    def _export_cell(
        self, keras: Any, model: Any, U_arr: np.ndarray, Y_arr: np.ndarray
    ) -> RecurrentCell:
        """Extract the trained weights and bound boxes into a RecurrentCell."""
        rnn_layer = None
        readout_w = readout_b = None
        for layer in model.layers:
            if isinstance(layer, keras.layers.SimpleRNN):
                rnn_layer = layer
            elif isinstance(layer, keras.layers.TimeDistributed):
                readout_w, readout_b = layer.get_weights()
            elif isinstance(layer, keras.layers.Dense):
                readout_w, readout_b = layer.get_weights()
        if rnn_layer is None:
            raise TrainingError("No SimpleRNN layer found in the trained model.")

        # SimpleRNN weights: [kernel (W_xh), recurrent_kernel (W_hh), bias].
        W_xh, W_hh, b_h = rnn_layer.get_weights()

        # Data-derived boxes (with optional margin). Exog in original units;
        # hidden box from the SCALED-space hidden states the RNN actually
        # visits on the training data (a reasonable reachable seed).
        exog_bounds = _padded_box(_flatten(U_arr), self.bounds_margin)
        hidden_states = self._hidden_states(keras, model, U_arr)
        hidden_init_bounds = _padded_box(hidden_states, self.bounds_margin)

        return RecurrentCell.from_elman(
            W_xh=W_xh,
            W_hh=W_hh,
            b_h=b_h,
            readout_weights=readout_w,
            readout_biases=readout_b,
            exog_scaler=self.exog_scaler_,
            output_scaler=self.output_scaler_,
            exog_bounds=exog_bounds,
            hidden_init_bounds=hidden_init_bounds,
        )

    def _hidden_states(self, keras: Any, model: Any, U_arr: np.ndarray) -> np.ndarray:
        """Return all hidden states the RNN visits on the training data.

        Builds a probe model that outputs the ``SimpleRNN`` sequence so the
        hidden box can be measured directly (these states are latent, so
        their range is otherwise unknown).
        """
        rnn_layer = next(
            layer for layer in model.layers if isinstance(layer, keras.layers.SimpleRNN)
        )
        probe = keras.Model(inputs=model.inputs, outputs=rnn_layer.output)
        U_scaled = _scale_seq(U_arr, self.exog_scaler_)
        h_seq = np.asarray(probe.predict(U_scaled, verbose=0), dtype=float)
        return h_seq.reshape(-1, h_seq.shape[-1])

    def _check_equivalence(
        self, cell: RecurrentCell, U_scaled: np.ndarray, model: Any
    ) -> float:
        """Max abs diff between the cell's numpy rollout and the Keras model."""
        n_check = min(U_scaled.shape[0], 16)
        # Unscale the exogenous sequences so the cell sees original units.
        U_check_scaled = U_scaled[:n_check]
        U_check = (
            self.exog_scaler_.inverse_transform(U_check_scaled.reshape(-1, U_check_scaled.shape[-1]))
            .reshape(U_check_scaled.shape)
            if self.exog_scaler_ is not None
            else U_check_scaled
        )
        y_keras = np.asarray(model.predict(U_check_scaled, verbose=0), dtype=float)
        if self.output_scaler_ is not None:
            flat = y_keras.reshape(-1, y_keras.shape[-1])
            y_keras = self.output_scaler_.inverse_transform(flat).reshape(y_keras.shape)
        max_diff = 0.0
        for s in range(n_check):
            _, y_cell = cell.rollout(np.zeros(cell.n_hidden), U_check[s])
            max_diff = max(max_diff, float(np.max(np.abs(y_cell - y_keras[s]))))
        scale = max(1.0, float(np.max(np.abs(y_keras))))
        if max_diff > _EQUIVALENCE_TOL * scale:
            _LOGGER.warning(
                "RecurrentCell/Keras rollout mismatch: max abs diff %.3e exceeds "
                "tolerance %.3e.",
                max_diff,
                _EQUIVALENCE_TOL * scale,
            )
        return max_diff

    def __repr__(self) -> str:
        return (
            f"RNNTrainer(hidden_units={self.hidden_units}, loss='{self.loss}', "
            f"epochs={self.epochs}, fitted={self.cell_ is not None})"
        )


# ----------------------------------------------------------------------
# Local typing alias + array helpers
# ----------------------------------------------------------------------
#: 3D sequence array (n_seq, T, features) or anything coercible to it.
ArrayLike3D = Any


def _as_3d(x: ArrayLike3D, name: str) -> np.ndarray:
    """Coerce ``x`` to a 3D ``(n_seq, T, features)`` float array.

    A 2D ``(n_seq, T)`` array is promoted to a single output feature.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise TrainingError(
            f"{name} must be 3D (n_seq, T, features) or 2D (n_seq, T), got shape "
            f"{arr.shape}."
        )
    return arr


def _flatten(arr3d: np.ndarray) -> np.ndarray:
    """Flatten ``(n_seq, T, F)`` to ``(n_seq*T, F)`` for scaler fitting."""
    return arr3d.reshape(-1, arr3d.shape[-1])


def _scale_seq(arr3d: np.ndarray, scaler: OffsetScaler | None) -> np.ndarray:
    """Apply a feature-wise scaler across a 3D sequence array."""
    if scaler is None:
        return arr3d
    flat = scaler.transform(_flatten(arr3d))
    return flat.reshape(arr3d.shape)


def _padded_box(flat: np.ndarray, margin: float) -> list[tuple[float, float]]:
    """Per-feature ``(min, max)`` box with fractional ``margin`` padding."""
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    pad = margin * (hi - lo) + 1e-9
    lo = lo - pad
    hi = hi + pad
    return [(float(l), float(u)) for l, u in zip(lo, hi)]
