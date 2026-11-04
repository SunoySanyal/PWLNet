"""Physics-informed Keras trainer producing core NetworkDefinitions.

The trainer owns the generic supervised-learning loop, scaling, and export.
Process physics is supplied by a small pluggable module that contributes
additional differentiable TensorFlow loss terms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pwlnet.core.layers import DenseLayer
from pwlnet.core.network import NetworkDefinition
from pwlnet.core.scaling import OffsetScaler
from pwlnet.physics import (
    PhysicsResidual,
    box_residual,
    linear_residual,
    lower_bound_residual,
    sum_to_one_residual,
    upper_bound_residual,
)
from pwlnet.training.initialization import import_keras, seed_everything
from pwlnet.utils.exceptions import TrainingError
from pwlnet.utils.typing import ArrayLike
from pwlnet.utils.validation import as_2d_array

__all__ = [
    "GenericAffineResidualPhysics",
    "PhysicsSpec",
    "PhysicsInformedTrainer",
    "PhysicsLoss",
    "affine_residual_value",
    "register_physics",
]

_HIDDEN_ACTIVATIONS = ("tanh", "relu")
_SCALING_METHODS = ("standard", "minmax", "none")
_PHYSICS_REGISTRY: dict[str, type["PhysicsLoss"]] = {}


class PhysicsLoss:
    """Base class for process-specific physics loss modules.

    Subclasses should set ``name``, optionally set ``input_dim`` and
    ``output_dim``, and implement :meth:`loss_terms`. Returned loss term
    names should not include ``"loss"`` or ``"data_loss"`` because those
    are managed by :class:`PhysicsInformedTrainer`.
    """

    name = "custom"
    input_dim: int | None = None
    output_dim: int | None = None
    default_weights: Mapping[str, float] = {}

    def validate_shapes(self, n_inputs: int, n_outputs: int) -> None:
        """Validate data dimensions for this physics module."""
        if self.input_dim is not None and n_inputs != self.input_dim:
            raise TrainingError(
                f"{self.name} physics expects X shape (n, {self.input_dim}) "
                f"but got (n, {n_inputs})."
            )
        if self.output_dim is not None and n_outputs != self.output_dim:
            raise TrainingError(
                f"{self.name} physics expects y shape (n, {self.output_dim}) "
                f"but got (n, {n_outputs})."
            )

    def build_constants(self, tf) -> dict[str, Any]:
        """Return TensorFlow constants reused by the training step."""
        return {}

    def loss_terms(self, tf, x_orig, y_orig, constants: Mapping[str, Any]) -> dict[str, Any]:
        """Return unweighted physics loss terms.

        ``x_orig`` and ``y_orig`` are tensors in original physical units.
        """
        raise NotImplementedError


def affine_residual_value(tf, residual_constants, x_orig, y_orig):
    """Evaluate an affine residual in TensorFlow.

    Affine residuals use ``r = x @ W_in.T + y @ W_out.T + b``. Inequality
    residuals represent ``r <= 0`` and are converted to positive violations
    with ``relu(r)`` before loss aggregation.
    """
    b = residual_constants["b"]
    n_samples = tf.shape(y_orig)[0]
    r = tf.broadcast_to(tf.reshape(b, (1, -1)), (n_samples, tf.shape(b)[0]))
    if residual_constants["w_in"] is not None:
        r = r + tf.linalg.matmul(x_orig, residual_constants["w_in"], transpose_b=True)
    if residual_constants["w_out"] is not None:
        r = r + tf.linalg.matmul(y_orig, residual_constants["w_out"], transpose_b=True)
    if residual_constants["kind"] == "inequality":
        r = tf.nn.relu(r)
    return r / residual_constants["scale"]


def piecewise_constant_residual_value(tf, residual_constants, x_orig, y_orig):
    """Evaluate an affine residual with a piecewise-constant input offset."""
    b = residual_constants["b"]
    n_samples = tf.shape(y_orig)[0]
    r = tf.broadcast_to(tf.reshape(b, (1, -1)), (n_samples, tf.shape(b)[0]))
    if residual_constants["w_in"] is not None:
        r = r + tf.linalg.matmul(x_orig, residual_constants["w_in"], transpose_b=True)
    if residual_constants["w_out"] is not None:
        r = r + tf.linalg.matmul(y_orig, residual_constants["w_out"], transpose_b=True)

    selector = (
        x_orig[:, residual_constants["input_index"]] * residual_constants["input_scale"]
        + residual_constants["input_offset"]
    )
    values = residual_constants["values"]
    piece = tf.fill(tf.shape(selector), values[-1])
    for breakpoint, value in reversed(
        list(zip(residual_constants["breakpoints"], values[:-1], strict=True))
    ):
        piece = tf.where(selector <= breakpoint, value, piece)
    r = r + tf.reshape(piece, (-1, 1))
    if residual_constants["kind"] == "inequality":
        r = tf.nn.relu(r)
    return r / residual_constants["scale"]


class GenericAffineResidualPhysics(PhysicsLoss):
    """Adapter from affine residual objects to trainer physics losses."""

    name = "generic_affine_residuals"

    def __init__(
        self,
        terms: Sequence[Mapping[str, Any]],
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        name: str | None = None,
    ) -> None:
        self.terms = [dict(term) for term in terms]
        if not self.terms:
            raise ValueError("GenericAffineResidualPhysics requires at least one term.")
        for term in self.terms:
            term_type = term.setdefault("type", "affine")
            if "name" not in term:
                raise ValueError("Each physics term needs a 'name'.")
            if term_type == "affine":
                if "residual" not in term:
                    raise ValueError("Each affine physics term needs a 'residual'.")
                residual = term["residual"]
                if not hasattr(residual, "affine_matrices"):
                    raise TypeError(f"Term {term['name']!r} is not an affine residual.")
            elif term_type != "piecewise_constant":
                raise ValueError(f"Unsupported physics term type {term_type!r}.")
            loss = str(term.get("loss", "mse")).lower()
            if loss not in {"mse", "mae"}:
                raise ValueError(f"Unsupported loss {loss!r}; expected 'mse' or 'mae'.")
            term["loss"] = loss
        self.input_dim = input_dim
        self.output_dim = output_dim
        if name is not None:
            self.name = str(name)
        self.default_weights = {str(term["name"]): 0.0 for term in self.terms}

    def build_constants(self, tf) -> dict[str, Any]:
        constants = {}
        for term in self.terms:
            if term["type"] == "affine":
                residual = term["residual"]
                w_in, w_out, b = residual.affine_matrices()
                constants[term["name"]] = {
                    "type": "affine",
                    "w_in": None if w_in is None else tf.constant(w_in, dtype=tf.float32),
                    "w_out": None if w_out is None else tf.constant(w_out, dtype=tf.float32),
                    "b": tf.constant(b, dtype=tf.float32),
                    "kind": residual.kind,
                    "loss": term["loss"],
                    "scale": tf.constant(term.get("scale", 1.0), dtype=tf.float32),
                }
            else:
                constants[term["name"]] = {
                    "type": "piecewise_constant",
                    "w_in": (
                        None
                        if term["w_in"] is None
                        else tf.constant(term["w_in"], dtype=tf.float32)
                    ),
                    "w_out": (
                        None
                        if term["w_out"] is None
                        else tf.constant(term["w_out"], dtype=tf.float32)
                    ),
                    "b": tf.constant(term["b"], dtype=tf.float32),
                    "kind": term["kind"],
                    "loss": term["loss"],
                    "input_index": int(term["input_index"]),
                    "input_scale": tf.constant(term["input_scale"], dtype=tf.float32),
                    "input_offset": tf.constant(term["input_offset"], dtype=tf.float32),
                    "breakpoints": tuple(
                        tf.constant(value, dtype=tf.float32)
                        for value in term["breakpoints"]
                    ),
                    "values": tuple(
                        tf.constant(value, dtype=tf.float32) for value in term["values"]
                    ),
                    "scale": tf.constant(term["scale"], dtype=tf.float32),
                }
        return constants

    def loss_terms(self, tf, x_orig, y_orig, constants: Mapping[str, Any]) -> dict[str, Any]:
        losses = {}
        for term in self.terms:
            term_name = term["name"]
            if constants[term_name]["type"] == "piecewise_constant":
                r = piecewise_constant_residual_value(
                    tf, constants[term_name], x_orig, y_orig
                )
            else:
                r = affine_residual_value(tf, constants[term_name], x_orig, y_orig)
            if constants[term_name]["loss"] == "mae":
                losses[f"{term_name}_loss"] = tf.reduce_mean(tf.abs(r))
            else:
                losses[f"{term_name}_loss"] = tf.reduce_mean(tf.square(r))
        return losses


class PhysicsSpec:
    """Name-based builder for affine physics-informed training losses.

    Users describe closure, bound, affine, and piecewise-constant backoff
    constraints with input/output variable names. The spec translates those
    names into differentiable loss terms for the trainer. Nonlinear name-based
    relations are not supported yet.
    """

    def __init__(
        self,
        *,
        input_names: Sequence[str],
        output_names: Sequence[str],
        name: str = "generic_affine_residuals",
    ) -> None:
        self.input_names = [str(v) for v in input_names]
        self.output_names = [str(v) for v in output_names]
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("input_names contains duplicate entries.")
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError("output_names contains duplicate entries.")
        self.name = str(name)
        self._input_index = {var: i for i, var in enumerate(self.input_names)}
        self._output_index = {var: i for i, var in enumerate(self.output_names)}
        self._terms: list[dict[str, Any]] = []

    def closure(
        self,
        variables: Sequence[str],
        *,
        total: float = 1.0,
        name: str = "closure",
        loss: str = "mse",
    ) -> "PhysicsSpec":
        """Add ``sum(outputs[variables]) == total`` as a physics loss."""
        indices = self._output_indices(variables)
        self._add_term(
            name=name,
            loss=loss,
            residual=sum_to_one_residual(
                n_outputs=len(self.output_names),
                output_indices=indices,
                total=total,
                name=name,
            ),
        )
        return self

    def upper_bounds(
        self,
        bounds: Mapping[str, float],
        *,
        name: str = "bounds",
        loss: str = "mse",
        violation_scale: Mapping[str, float] | float | None = None,
    ) -> "PhysicsSpec":
        """Add output upper bounds as ``output <= bound`` violations."""
        variables = list(bounds)
        self._add_term(
            name=name,
            loss=loss,
            residual=upper_bound_residual(
                n_outputs=len(self.output_names),
                upper=[float(bounds[var]) for var in variables],
                output_indices=self._output_indices(variables),
                name=name,
            ),
            scale=self._scales_for(variables, violation_scale),
        )
        return self

    def lower_bounds(
        self,
        bounds: Mapping[str, float],
        *,
        name: str = "lower_bounds",
        loss: str = "mse",
        violation_scale: Mapping[str, float] | float | None = None,
    ) -> "PhysicsSpec":
        """Add output lower bounds as ``output >= bound`` violations."""
        variables = list(bounds)
        self._add_term(
            name=name,
            loss=loss,
            residual=lower_bound_residual(
                n_outputs=len(self.output_names),
                lower=[float(bounds[var]) for var in variables],
                output_indices=self._output_indices(variables),
                name=name,
            ),
            scale=self._scales_for(variables, violation_scale),
        )
        return self

    def box_bounds(
        self,
        bounds: Mapping[str, tuple[float, float]],
        *,
        name: str = "bounds",
        loss: str = "mse",
        normalize: bool = False,
        violation_scale: Mapping[str, float] | float | None = None,
    ) -> "PhysicsSpec":
        """Add two-sided output bounds.

        When ``normalize=True``, each variable's upper and lower violations
        are divided by ``upper - lower``. ``violation_scale`` can override
        those scales explicitly.
        """
        variables = list(bounds)
        lower = [float(bounds[var][0]) for var in variables]
        upper = [float(bounds[var][1]) for var in variables]
        if violation_scale is None and normalize:
            scale = np.asarray(
                [max(hi - lo, 1e-9) for lo, hi in zip(lower, upper, strict=True)],
                dtype=float,
            )
        else:
            scale = self._scales_for(variables, violation_scale)
        scale = np.asarray(scale, dtype=float)
        self._add_term(
            name=name,
            loss=loss,
            residual=box_residual(
                n_outputs=len(self.output_names),
                lower=lower,
                upper=upper,
                output_indices=self._output_indices(variables),
                name=name,
            ),
            scale=np.r_[scale, scale],
        )
        return self

    def linear_relation(
        self,
        *,
        inputs: Mapping[str, float] | None = None,
        outputs: Mapping[str, float] | None = None,
        const: float = 0.0,
        sense: str = "==",
        name: str = "linear_relation",
        loss: str = "mse",
    ) -> "PhysicsSpec":
        """Add a named affine relation.

        The relation is interpreted as::

            sum(input_coeff[var] * input[var])
            + sum(output_coeff[var] * output[var])
            + const sense 0

        ``sense`` accepts ``"=="`` for equality and ``"<="`` for
        one-sided inequality violations. Use negative coefficients to express
        ``>=`` relations as ``<=``.
        """
        relation_kind = self._relation_kind(sense)
        self._add_term(
            name=name,
            loss=loss,
            residual=linear_residual(
                n_inputs=len(self.input_names),
                n_outputs=len(self.output_names),
                input_coeffs=self._input_coefficients(inputs or {}),
                output_coeffs=self._output_coefficients(outputs or {}),
                const=float(const),
                kind=relation_kind,
                name=name,
            ),
        )
        return self

    def backoff_relation(
        self,
        *,
        time_input: str,
        breakpoints: Sequence[float],
        backoffs: Sequence[float],
        inputs: Mapping[str, float] | None = None,
        outputs: Mapping[str, float] | None = None,
        limit: float = 0.0,
        time_scale: float = 1.0,
        time_offset: float = 0.0,
        name: str = "backoff",
        loss: str = "mse",
        violation_scale: float = 1.0,
    ) -> "PhysicsSpec":
        """Add ``linear(inputs, outputs) <= limit - backoff(time_input)``.

        ``breakpoints`` and ``backoffs`` define a piecewise-constant backoff
        curve. With breakpoints ``[30, 55]`` and backoffs ``[1.3, 0.7, 0.0]``,
        the first backoff applies up to 30, the second up to 55, and the last
        applies after 55. ``time_input`` may be transformed with
        ``time_input * time_scale + time_offset`` before breakpoint lookup.
        """
        if len(backoffs) != len(breakpoints) + 1:
            raise ValueError("backoffs must have exactly one more entry than breakpoints.")
        if float(violation_scale) <= 0.0:
            raise ValueError("violation_scale must be positive.")
        input_coeffs = self._input_coefficients(inputs or {})
        output_coeffs = self._output_coefficients(outputs or {})
        if not input_coeffs and not output_coeffs:
            raise ValueError("backoff_relation needs inputs or outputs coefficients.")
        w_in = self._coefficient_matrix(input_coeffs, len(self.input_names))
        w_out = self._coefficient_matrix(output_coeffs, len(self.output_names))
        self._add_term(
            name=name,
            loss=loss,
            term={
                "type": "piecewise_constant",
                "w_in": w_in,
                "w_out": w_out,
                "b": np.asarray([-float(limit)], dtype=float),
                "kind": "inequality",
                "input_index": self._input_index_for(time_input),
                "input_scale": float(time_scale),
                "input_offset": float(time_offset),
                "breakpoints": tuple(float(value) for value in breakpoints),
                "values": tuple(float(value) for value in backoffs),
                "scale": float(violation_scale),
            },
        )
        return self

    def monotonic_outputs(
        self,
        variables: Sequence[str],
        *,
        direction: str = "nondecreasing",
        name_prefix: str = "monotonic",
        loss: str = "mse",
        violation_scale: Mapping[str, float] | float | None = None,
    ) -> "PhysicsSpec":
        """Add transition monotonicity constraints for state outputs.

        This helper assumes the input vector contains the current state values
        in the same names as the output vector. For
        ``direction="nondecreasing"``, it adds ``current[var] - next[var] <= 0``.
        For ``"nonincreasing"``, it adds ``next[var] - current[var] <= 0``.
        ``violation_scale`` optionally normalizes each variable's violation.
        """
        sign = self._monotonic_sign(direction)
        scales = self._scales_for(variables, violation_scale)
        for variable, scale in zip(variables, scales, strict=True):
            if float(scale) <= 0.0:
                raise ValueError("monotonic violation scales must be positive.")
            coeff = sign / float(scale)
            self.linear_relation(
                inputs={str(variable): coeff},
                outputs={str(variable): -coeff},
                sense="<=",
                name=f"{name_prefix}_{variable}",
                loss=loss,
            )
        return self

    def term(
        self,
        name: str,
        residual: PhysicsResidual,
        *,
        loss: str = "mse",
    ) -> "PhysicsSpec":
        """Add a prebuilt affine residual term."""
        self._add_term(name=name, residual=residual, loss=loss)
        return self

    def build(self) -> GenericAffineResidualPhysics:
        """Return a trainer-compatible physics module."""
        return GenericAffineResidualPhysics(
            self._terms,
            input_dim=len(self.input_names),
            output_dim=len(self.output_names),
            name=self.name,
        )

    def _output_indices(self, variables: Sequence[str]) -> list[int]:
        indices = []
        for variable in variables:
            try:
                indices.append(self._output_index[str(variable)])
            except KeyError as exc:
                raise ValueError(
                    f"Unknown output variable {variable!r}; choices are {self.output_names}."
                ) from exc
        return indices

    def _input_coefficients(self, coefficients: Mapping[str, float]) -> dict[int, float]:
        indexed = {}
        for variable, coefficient in coefficients.items():
            try:
                indexed[self._input_index[str(variable)]] = float(coefficient)
            except KeyError as exc:
                raise ValueError(
                    f"Unknown input variable {variable!r}; choices are {self.input_names}."
                ) from exc
        return indexed

    def _input_index_for(self, variable: str) -> int:
        try:
            return self._input_index[str(variable)]
        except KeyError as exc:
            raise ValueError(
                f"Unknown input variable {variable!r}; choices are {self.input_names}."
            ) from exc

    def _output_coefficients(self, coefficients: Mapping[str, float]) -> dict[int, float]:
        indexed = {}
        for variable, coefficient in coefficients.items():
            try:
                indexed[self._output_index[str(variable)]] = float(coefficient)
            except KeyError as exc:
                raise ValueError(
                    f"Unknown output variable {variable!r}; choices are {self.output_names}."
                ) from exc
        return indexed

    def _scales_for(
        self,
        variables: Sequence[str],
        violation_scale: Mapping[str, float] | float | None,
    ) -> np.ndarray:
        if violation_scale is None:
            return np.ones(len(variables), dtype=float)
        if isinstance(violation_scale, Mapping):
            return np.asarray([float(violation_scale[var]) for var in variables], dtype=float)
        return np.full(len(variables), float(violation_scale), dtype=float)

    @staticmethod
    def _relation_kind(sense: str) -> str:
        normalized = str(sense).strip()
        if normalized in {"==", "=", "equality"}:
            return "equality"
        if normalized in {"<=", "le", "ineq", "inequality"}:
            return "inequality"
        raise ValueError("sense must be '==' or '<='.")

    @staticmethod
    def _monotonic_sign(direction: str) -> float:
        normalized = str(direction).strip().lower().replace("_", "-")
        if normalized in {"nondecreasing", "non-decreasing", "increasing", "up"}:
            return 1.0
        if normalized in {"nonincreasing", "non-increasing", "decreasing", "down"}:
            return -1.0
        raise ValueError("direction must be 'nondecreasing' or 'nonincreasing'.")

    @staticmethod
    def _coefficient_matrix(coefficients: Mapping[int, float], width: int) -> np.ndarray | None:
        if not coefficients:
            return None
        matrix = np.zeros((1, width), dtype=float)
        for index, coefficient in coefficients.items():
            matrix[0, int(index)] = float(coefficient)
        return matrix

    def _add_term(
        self,
        *,
        name: str,
        residual: PhysicsResidual | None = None,
        loss: str,
        scale: Any = 1.0,
        term: Mapping[str, Any] | None = None,
    ) -> None:
        term_name = str(name)
        if not term_name:
            raise ValueError("Physics term name cannot be empty.")
        if any(term["name"] == term_name for term in self._terms):
            raise ValueError(f"Duplicate physics term name {term_name!r}.")
        if term is None:
            self._terms.append(
                {"name": term_name, "loss": loss, "residual": residual, "scale": scale}
            )
        else:
            row = dict(term)
            row.update({"name": term_name, "loss": loss})
            self._terms.append(row)


def register_physics(name: str, physics_cls: type[PhysicsLoss]) -> None:
    """Register a physics module class for string-based lookup."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Physics registry name cannot be empty.")
    if not issubclass(physics_cls, PhysicsLoss):
        raise TypeError("physics_cls must be a PhysicsLoss subclass.")
    _PHYSICS_REGISTRY[key] = physics_cls


class PhysicsInformedTrainer:
    """Train a dense network with data loss plus a physics penalty.

    Parameters
    ----------
    hidden_layers : list of int
        Widths of the hidden layers.
    activation : {"tanh", "relu"}, optional
        Hidden-layer activation. Default ``"tanh"``.
    physics : str, PhysicsLoss, or PhysicsSpec
        Physics module name or instance. Process packages can register
        names with :func:`register_physics`; custom modules can subclass
        :class:`PhysicsLoss` and be passed directly. ``PhysicsSpec`` instances
        are translated to generic affine physics automatically.
    physics_weights : dict or None, optional
        Weights for physics terms. Recognized keys for built-in
        Williams-Otto are ``"closure"`` and ``"bounds"``. Missing keys
        default to ``0.0``.
    optimizer : str, optional
        Keras optimizer name. Default ``"adam"``.
    learning_rate : float, optional
        Optimizer learning rate.
    epochs : int, optional
        Maximum epochs.
    batch_size : int, optional
        Batch size.
    validation_split : float, optional
        Internal validation split when explicit validation data is not
        supplied to :meth:`fit`.
    scaling_method : {"standard", "minmax", "none"}, optional
        Input/output scaling method.
    random_seed : int or None, optional
        Random seed.
    verbose : int, optional
        If positive, print periodic progress rows.
    early_stopping_patience : int or None, optional
        Stop when validation composite loss stops improving.
    output_loss_weights : array_like or None, optional
        Per-output weights (length ``n_outputs``) applied to the
        (scaled-space) data MSE, so selected target columns -- e.g. the
        constraint mass fractions -- can be penalized more heavily. Weights are
        renormalized to mean 1 so the data-loss magnitude stays comparable to
        the unweighted run; the physics penalty terms are unaffected. ``None``
        (default) weights every output equally. Mirrors
        :class:`pwlnet.training.KerasTrainer`.
    """

    def __init__(
        self,
        hidden_layers: list[int],
        activation: str = "tanh",
        physics: str | PhysicsLoss | PhysicsSpec | None = None,
        physics_weights: dict[str, float] | None = None,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 500,
        batch_size: int = 32,
        validation_split: float = 0.1,
        scaling_method: str = "standard",
        random_seed: int | None = None,
        verbose: int = 0,
        early_stopping_patience: int | None = None,
        output_loss_weights: ArrayLike | None = None,
    ) -> None:
        self.hidden_layers = [int(w) for w in hidden_layers]
        if not self.hidden_layers or any(w <= 0 for w in self.hidden_layers):
            raise ValueError(f"hidden_layers must be positive ints, got {hidden_layers!r}.")
        act = str(activation).strip().lower()
        if act not in _HIDDEN_ACTIVATIONS:
            raise ValueError(f"activation must be one of {_HIDDEN_ACTIVATIONS}, got {activation!r}.")
        self.activation = act
        self.physics = self._resolve_physics(physics)
        weights = dict(self.physics.default_weights)
        weights.update(dict(physics_weights or {}))
        self.physics_weights = {str(k): float(v) for k, v in weights.items()}
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        if not 0.0 <= float(validation_split) < 1.0:
            raise ValueError(f"validation_split must be in [0, 1), got {validation_split!r}.")
        self.validation_split = float(validation_split)
        method = str(scaling_method).strip().lower()
        if method not in _SCALING_METHODS:
            raise ValueError(f"scaling_method must be one of {_SCALING_METHODS}, got {scaling_method!r}.")
        self.scaling_method = method
        self.random_seed = None if random_seed is None else int(random_seed)
        self.verbose = int(verbose)
        if early_stopping_patience is not None and int(early_stopping_patience) < 1:
            raise ValueError("early_stopping_patience must be a positive int or None.")
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
                    "output_loss_weights must be a non-empty vector of finite, "
                    "non-negative weights with a positive sum, got "
                    f"{output_loss_weights!r}."
                )
            self.output_loss_weights = weights
        self.model_: Any = None
        self.history_: dict[str, list[float]] | None = None
        self.network_: NetworkDefinition | None = None
        self.input_scaler_: OffsetScaler | None = None
        self.output_scaler_: OffsetScaler | None = None
        self.equivalence_error_: float | None = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> NetworkDefinition:
        """Train on original-unit data and return a NetworkDefinition."""
        keras = import_keras()
        import tensorflow as tf

        X_arr = as_2d_array(X, name="X")
        y_arr = as_2d_array(y, name="y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise TrainingError(f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}.")
        self.physics.validate_shapes(X_arr.shape[1], y_arr.shape[1])
        if (
            self.output_loss_weights is not None
            and self.output_loss_weights.size != y_arr.shape[1]
        ):
            raise TrainingError(
                f"output_loss_weights has length {self.output_loss_weights.size} "
                f"but the network has {y_arr.shape[1]} outputs."
            )

        X_train_arr, y_train_arr, validation_scaled = self._prepare_data(
            X_arr, y_arr, validation_data
        )
        if self.random_seed is not None:
            seed_everything(self.random_seed)

        model = self._build_model(keras, X_arr.shape[1], y_arr.shape[1])
        optimizer = self._build_optimizer(keras)
        dataset = (
            tf.data.Dataset.from_tensor_slices((X_train_arr.astype("float32"), y_train_arr.astype("float32")))
            .shuffle(len(X_train_arr), seed=self.random_seed, reshuffle_each_iteration=True)
            .batch(self.batch_size)
        )

        if validation_scaled is None:
            X_val_scaled, y_val_scaled = X_train_arr.astype("float32"), y_train_arr.astype("float32")
        else:
            X_val_scaled, y_val_scaled = validation_scaled

        constants = {
            "x_offset": tf.constant(self.input_scaler_.offset.astype("float32")),
            "x_factor": tf.constant(self.input_scaler_.factor.astype("float32")),
            "y_offset": tf.constant(self.output_scaler_.offset.astype("float32")),
            "y_factor": tf.constant(self.output_scaler_.factor.astype("float32")),
            "physics": self.physics.build_constants(tf),
            "data_weights": (
                None
                if self.output_loss_weights is None
                else tf.constant(
                    (
                        self.output_loss_weights
                        * (self.output_loss_weights.size / float(self.output_loss_weights.sum()))
                    ).astype("float32")
                )
            ),
        }

        @tf.function
        def train_step(xb, yb):
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                losses = self._loss_terms(tf, xb, yb, pred, constants)
            grads = tape.gradient(losses["loss"], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return losses

        def evaluate(x_eval, y_eval):
            pred = model(tf.constant(x_eval, dtype=tf.float32), training=False)
            losses = self._loss_terms(
                tf,
                tf.constant(x_eval, dtype=tf.float32),
                tf.constant(y_eval, dtype=tf.float32),
                pred,
                constants,
            )
            return {k: float(v.numpy()) for k, v in losses.items()}

        history_rows: list[dict[str, float]] = []
        best_val = np.inf
        best_weights = None
        wait = 0
        for epoch in range(1, self.epochs + 1):
            batch_rows = []
            for xb, yb in dataset:
                batch_rows.append({k: float(v.numpy()) for k, v in train_step(xb, yb).items()})
            train_row = {
                key: float(np.mean([row[key] for row in batch_rows]))
                for key in batch_rows[0]
            }
            val_row = evaluate(X_val_scaled, y_val_scaled)
            row = {"epoch": float(epoch), **train_row}
            row.update({f"val_{k}": v for k, v in val_row.items()})
            history_rows.append(row)
            val_loss = val_row["loss"]
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_weights = model.get_weights()
                wait = 0
            else:
                wait += 1
            if self.verbose and (epoch == 1 or epoch % max(1, self.epochs // 10) == 0):
                print(
                    f"epoch {epoch:4d}: loss={train_row['loss']:.6g} "
                    f"val_loss={val_loss:.6g}"
                )
            if self.early_stopping_patience is not None and wait >= self.early_stopping_patience:
                break

        if best_weights is not None:
            model.set_weights(best_weights)

        self.model_ = model
        self.history_ = self._history_dict(history_rows)
        input_bounds = [(float(lo), float(hi)) for lo, hi in zip(X_arr.min(axis=0), X_arr.max(axis=0))]
        self.network_ = self._export_network(model, input_bounds)
        self.equivalence_error_ = self._check_equivalence(X_arr)
        return self.network_

    def _prepare_data(self, X_arr, y_arr, validation_data):
        self.input_scaler_ = OffsetScaler.fit(X_arr, method=self.scaling_method)
        self.output_scaler_ = OffsetScaler.fit(y_arr, method=self.scaling_method)
        X_scaled = self.input_scaler_.transform(X_arr).astype("float32")
        y_scaled = self.output_scaler_.transform(y_arr).astype("float32")

        if validation_data is None:
            return X_scaled, y_scaled, None
        X_val = as_2d_array(validation_data[0], name="X_val")
        y_val = as_2d_array(validation_data[1], name="y_val")
        if X_val.shape[1] != X_arr.shape[1] or y_val.shape[1] != y_arr.shape[1]:
            raise TrainingError("validation_data widths do not match training data.")
        return (
            X_scaled,
            y_scaled,
            (
                self.input_scaler_.transform(X_val).astype("float32"),
                self.output_scaler_.transform(y_val).astype("float32"),
            ),
        )

    @staticmethod
    def _resolve_physics(physics: str | PhysicsLoss | PhysicsSpec | None) -> PhysicsLoss:
        if physics is None:
            raise ValueError(
                "physics must be provided as a registered name, a PhysicsSpec, "
                "or a PhysicsLoss instance."
            )
        if isinstance(physics, PhysicsLoss):
            return physics
        if isinstance(physics, PhysicsSpec):
            return physics.build()
        key = str(physics).strip().lower()
        try:
            physics_cls = _PHYSICS_REGISTRY[key]
        except KeyError as exc:
            choices = ", ".join(sorted(_PHYSICS_REGISTRY)) or "<none>"
            raise ValueError(
                f"Unknown physics module {physics!r}. Pass a PhysicsSpec, "
                f"a PhysicsLoss instance, or one of the registered names: {choices}."
            ) from exc
        return physics_cls()

    def _build_model(self, keras, n_inputs: int, n_outputs: int):
        model = keras.Sequential(name=f"pwlnet_{self.physics.name}_physics_informed")
        model.add(keras.layers.Input(shape=(n_inputs,)))
        for i, width in enumerate(self.hidden_layers):
            seed = None if self.random_seed is None else self.random_seed + 17 * (i + 1)
            model.add(
                keras.layers.Dense(
                    width,
                    activation=self.activation,
                    kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
                    bias_initializer="zeros",
                )
            )
        seed = None if self.random_seed is None else self.random_seed + 999
        model.add(
            keras.layers.Dense(
                n_outputs,
                activation="linear",
                kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
                bias_initializer="zeros",
            )
        )
        return model

    def _build_optimizer(self, keras):
        optimizer = keras.optimizers.get(self.optimizer)
        optimizer.learning_rate = self.learning_rate
        return optimizer

    def _loss_terms(self, tf, x_scaled, y_scaled_true, y_scaled_pred, c):
        squared_error = tf.square(y_scaled_pred - y_scaled_true)
        if c.get("data_weights") is not None:
            data_loss = tf.reduce_mean(squared_error * c["data_weights"])
        else:
            data_loss = tf.reduce_mean(squared_error)
        x_orig = x_scaled * c["x_factor"] + c["x_offset"]
        y_orig = y_scaled_pred * c["y_factor"] + c["y_offset"]
        physics_terms = self.physics.loss_terms(tf, x_orig, y_orig, c["physics"])
        total = data_loss
        for key, value in physics_terms.items():
            if key in {"loss", "data_loss"}:
                raise TrainingError(
                    "Physics modules cannot return reserved loss terms "
                    "'loss' or 'data_loss'."
                )
            weight_key = key[:-5] if key.endswith("_loss") else key
            total = total + self.physics_weights.get(weight_key, 0.0) * value
        return {
            "loss": total,
            "data_loss": data_loss,
            **physics_terms,
        }

    @staticmethod
    def _history_dict(rows: list[dict[str, float]]) -> dict[str, list[float]]:
        if not rows:
            return {}
        return {key: [float(row[key]) for row in rows] for key in rows[0]}

    def _export_network(self, model, input_bounds) -> NetworkDefinition:
        layers = []
        for layer in model.layers:
            weights = layer.get_weights() if hasattr(layer, "get_weights") else []
            if len(weights) != 2:
                continue
            W, b = weights
            layers.append(
                DenseLayer(
                    np.asarray(W, dtype=float),
                    np.asarray(b, dtype=float),
                    activation=layer.get_config().get("activation", "linear"),
                )
            )
        return NetworkDefinition(
            layers=layers,
            input_scaler=self.input_scaler_,
            output_scaler=self.output_scaler_,
            input_bounds=input_bounds,
        )

    def _check_equivalence(self, X_arr: np.ndarray) -> float:
        X_check = X_arr[: min(X_arr.shape[0], 256)]
        X_scaled = self.input_scaler_.transform(X_check)
        y_keras = np.asarray(self.model_.predict(X_scaled, verbose=0), dtype=float)
        y_keras = self.output_scaler_.inverse_transform(y_keras)
        return float(np.max(np.abs(self.network_.predict(X_check) - y_keras)))

    def __repr__(self) -> str:
        return (
            f"PhysicsInformedTrainer(hidden_layers={self.hidden_layers}, "
            f"activation={self.activation!r}, physics={self.physics.name!r}, "
            f"epochs={self.epochs}, fitted={self.network_ is not None})"
        )
