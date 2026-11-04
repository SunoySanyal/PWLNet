"""High-level PWLNet facade API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pwlnet.recurrent.cell import RecurrentCell

from pwlnet.core.bounds import BoundsResult, propagate_bounds
from pwlnet.core.model_summary import ModelSummary, summarize
from pwlnet.core.network import NetworkDefinition
from pwlnet.pruning.base import PruneOutcome
from pwlnet.pruning.prune import prune_network
from pwlnet.pruning.reports import PruneReport, build_prune_report
from pwlnet.training.keras_trainer import KerasTrainer
from pwlnet.training.physics_informed import PhysicsInformedTrainer, PhysicsLoss, PhysicsSpec
from pwlnet.utils.typing import ArrayLike, PathLike

__all__ = ["PWLNet"]


class PWLNet:
    """User-facing wrapper around a trained :class:`NetworkDefinition`.

    The facade intentionally stays thin: the internal representation is
    still the source of truth, while training, reports, and Pyomo/OMLT
    embedding are delegated to their specialist modules.
    """

    def __init__(self, network: NetworkDefinition, trainer: Any | None = None):
        self.network = network
        self.trainer = trainer

    @classmethod
    def train(
        cls,
        X: ArrayLike,
        y: ArrayLike,
        hidden_layers: list[int],
        activation: str = "relu",
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
        **trainer_options: Any,
    ) -> "PWLNet":
        """Train a Keras dense network and return a PWLNet facade.

        ``activation`` is ``"relu"``, ``"tanh"``, or ``"pwl_tanh"``.
        The last one trains through the piecewise-linear tanh that the
        MILP embedding builds (breakpoint set via the
        ``breakpoints=...`` trainer option, default 5) and exports the
        hidden layers as ``"tanh"``, so
        ``to_pyomo(formulation="tanh_pwl_sos2", backend="pyomo_pwl",
        breakpoints=...)`` reproduces the trained model rather than
        approximating it. Note
        that :meth:`predict` then reads the network with EXACT tanh; use
        :func:`pwlnet.formulations.pyomo_pwl.evaluate_tanh_pwl_network`
        with the same breakpoints for the reading the solver sees.

        ``validation_data`` is an optional explicit ``(X_val, y_val)``
        set forwarded to :meth:`KerasTrainer.fit`; when given it
        replaces the internal ``validation_split`` so the validation
        partition is fixed and can be scored after training.
        """
        trainer = KerasTrainer(
            hidden_layers=hidden_layers,
            activation=activation,
            **trainer_options,
        )
        return cls(trainer.fit(X, y, validation_data=validation_data), trainer=trainer)

    @classmethod
    def train_physics_informed(
        cls,
        X: ArrayLike,
        y: ArrayLike,
        hidden_layers: list[int],
        activation: str = "tanh",
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
        physics: str | PhysicsLoss | PhysicsSpec | None = None,
        physics_weights: dict[str, float] | None = None,
        **trainer_options: Any,
    ) -> "PWLNet":
        """Train with data loss plus physics penalties and return PWLNet.

        ``physics`` may be a registered module name, a high-level
        :class:`PhysicsSpec`, or a custom :class:`PhysicsLoss` instance.
        Process packages own process-specific physics modules.
        """
        trainer = PhysicsInformedTrainer(
            hidden_layers=hidden_layers,
            activation=activation,
            physics=physics,
            physics_weights=physics_weights,
            **trainer_options,
        )
        return cls(trainer.fit(X, y, validation_data=validation_data), trainer=trainer)

    @classmethod
    def train_sosx(
        cls,
        X: ArrayLike,
        y: ArrayLike,
        hidden_neurons: int,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
        *,
        initial_network: NetworkDefinition | None = None,
        W2: ArrayLike | None = None,
        incumbent: Any | None = None,
        **trainer_options: Any,
    ) -> "PWLNet":
        """Globally train a fixed-readout PWL-tanh network and return PWLNet.

        Wraps :class:`pwlnet.training.SosxTrainer`: a bounded,
        single-hidden-layer ``tanh`` network is trained under an MAE loss
        with the output weights fixed, so every branch-and-bound node is
        an LP. ``trainer_options`` are forwarded to the ``SosxTrainer``
        constructor (e.g. ``breakpoints``, ``readout``, ``solver``,
        ``solver_options``, ``max_nodes``, ``time_limit``,
        ``relative_gap``). The fitted :class:`SosxTrainer` is kept on the
        returned facade's ``trainer`` attribute, exposing the SOSX solve
        diagnostics (``result_``, ``training_mae_``, ``n_nodes_``, ...).
        """
        from pwlnet.training.sosx_trainer import SosxTrainer

        trainer = SosxTrainer(hidden_neurons=hidden_neurons, **trainer_options)
        network = trainer.fit(
            X,
            y,
            validation_data=validation_data,
            initial_network=initial_network,
            W2=W2,
            incumbent=incumbent,
        )
        return cls(network, trainer=trainer)

    @staticmethod
    def train_rnn(
        U: ArrayLike,
        Y: ArrayLike,
        hidden_units: int,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
        **trainer_options: Any,
    ) -> "RecurrentCell":
        """Train a one-step ReLU RNN and return a :class:`RecurrentCell`.

        Unlike the feed-forward trainers, this returns a
        :class:`pwlnet.recurrent.cell.RecurrentCell` rather than a
        ``PWLNet`` facade: the recurrent cell is its own self-contained
        object exposing the full pipeline -- numpy ``rollout``/``encode``,
        exact ``prune_dead_units``, ``recurrent_bounds``, and
        ``build_mpc``/``solve_mpc`` (which unroll it into a ``relu_bigm``
        MILP). See :class:`pwlnet.training.rnn_trainer.RNNTrainer` for the
        accepted ``trainer_options`` (e.g. ``epochs``, ``loss``,
        ``scaling_method``, ``bounds_margin``, ``random_seed``).

        Parameters
        ----------
        U : array_like
            Exogenous input sequences, shape ``(n_seq, T, n_exog)`` in
            ORIGINAL units.
        Y : array_like
            Target output sequences, shape ``(n_seq, T, n_out)`` (or
            ``(n_seq, T)`` for a single output) in ORIGINAL units.
        hidden_units : int
            Width of the recurrent hidden state.
        validation_data : tuple of (array_like, array_like) or None, optional
            Explicit ``(U_val, Y_val)`` validation sequences.
        **trainer_options
            Forwarded to :class:`RNNTrainer`.

        Returns
        -------
        RecurrentCell
            The exported cell; the fitted ``RNNTrainer`` is available on
            ``cell``-adjacent diagnostics via the trainer it was built
            from (kept by the caller if needed).
        """
        from pwlnet.training.rnn_trainer import RNNTrainer

        trainer = RNNTrainer(hidden_units=hidden_units, **trainer_options)
        return trainer.fit(U, Y, validation_data=validation_data)

    @classmethod
    def load(cls, path: PathLike) -> "PWLNet":
        """Load a saved network JSON file."""
        return cls(NetworkDefinition.load(path))

    def save(self, path: PathLike) -> None:
        """Save the wrapped network as JSON."""
        self.network.save(path)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Evaluate the wrapped network in original units."""
        return self.network.predict(X)

    def summary(self) -> ModelSummary:
        """Return a structural model summary."""
        return summarize(self.network)

    def bounds(self) -> BoundsResult:
        """Run interval bound propagation through the wrapped network."""
        return propagate_bounds(self.network)

    def prune_report(self) -> PruneReport:
        """Return size estimates and bound-derived pruning opportunities."""
        return build_prune_report(self.network)

    def prune(
        self,
        method: str = "dead_relu",
        *,
        fine_tune: bool = False,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
        return_outcome: bool = False,
        **kwargs: Any,
    ) -> "PWLNet | tuple[PWLNet, PruneOutcome]":
        """Prune the wrapped network and return a NEW :class:`PWLNet`.

        Dispatches to :func:`pwlnet.pruning.prune_network`, which maps
        ``method`` to one of the Phase-3 pruning entry points (see that
        function for the valid names and aliases, e.g. ``"dead_relu"``,
        ``"saturated_tanh"``, ``"weight_magnitude"``/``"magnitude"``,
        ``"neuron"``, ``"group_lasso"``, ``"obd"``). Extra keyword
        arguments (``fraction=``, ``threshold=``, ``abs_threshold=``,
        OBD's ``X=``/``y=``, ...) are forwarded to the selected method.

        The wrapped network is never mutated -- a brand-new network is
        wrapped in a fresh ``PWLNet`` (with ``trainer=None``, since the
        pruned network no longer corresponds to the original Keras model).

        Parameters
        ----------
        method : str, optional
            Pruning method name (default ``"dead_relu"``).
        fine_tune : bool, optional
            If ``True``, run :func:`pwlnet.pruning.fine_tune` on the pruned
            network to recover accuracy. Requires ``X`` and ``y``.
        X, y : array_like or None, optional
            Training data in ORIGINAL units, used for fine-tuning (and only
            consumed when ``fine_tune`` is ``True``; OBD takes its own
            ``X``/``y`` via ``**kwargs``).
        validation_data : tuple of (array_like, array_like) or None, optional
            Validation set forwarded to fine-tuning.
        return_outcome : bool, optional
            If ``True``, also return the :class:`PruneOutcome` (the
            fine-tuning outcome when ``fine_tune`` is set, else the pruning
            outcome).
        **kwargs
            Forwarded to the selected pruning method.

        Returns
        -------
        PWLNet or tuple of (PWLNet, PruneOutcome)
            The pruned network as a new ``PWLNet``; with the outcome too
            when ``return_outcome`` is ``True``.

        Raises
        ------
        ValueError
            If ``method`` is unknown, or ``fine_tune`` is requested without
            both ``X`` and ``y``.
        """
        outcome = prune_network(self.network, method, **kwargs)

        if fine_tune:
            if X is None or y is None:
                raise ValueError(
                    "fine_tune=True requires both X and y (training data in "
                    "original units)."
                )
            from pwlnet.pruning.fine_tune import fine_tune as _fine_tune

            outcome = _fine_tune(
                outcome.network, X, y, validation_data=validation_data
            )

        pruned = PWLNet(outcome.network, trainer=None)
        if return_outcome:
            return pruned, outcome
        return pruned

    def compress(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        *,
        fine_tune: bool = False,
        magnitude_fraction: float | None = None,
        neuron_fraction: float | None = None,
        return_outcome: bool = False,
        **fine_tune_kwargs: Any,
    ) -> "PWLNet | tuple[PWLNet, list[PruneOutcome]]":
        """Run the recommended compression pipeline and return a new PWLNet.

        The pipeline applies, in order:

        1. **Exact ReLU simplification** -- dead-neuron removal plus
           always-active linearization (:func:`pwlnet.pruning.prune_relus`),
           which is function-preserving over the input box. Skipped (with a
           note) if the network has no ``input_bounds``.
        2. **Optional magnitude pruning** -- when ``magnitude_fraction`` is
           given, zero the smallest fraction of weights and clean up any
           neuron rendered structurally trivial
           (:func:`pwlnet.pruning.prune_weights_magnitude`).
        3. **Optional neuron pruning** -- when ``neuron_fraction`` is given,
           drop the least-important neurons
           (:func:`pwlnet.pruning.prune_neurons`).
        4. **Optional fine-tuning** -- when ``fine_tune`` is set, recover
           accuracy on ``X``/``y`` (:func:`pwlnet.pruning.fine_tune`).

        Steps 2-4 are off by default, so the bare ``compress()`` call is the
        safe exact-only path.

        Parameters
        ----------
        X, y : array_like or None, optional
            Training data in ORIGINAL units (required if ``fine_tune`` is
            ``True``).
        fine_tune : bool, optional
            Append a fine-tuning step. Requires ``X`` and ``y``.
        magnitude_fraction : float or None, optional
            If set, run magnitude pruning with this ``fraction``.
        neuron_fraction : float or None, optional
            If set, run importance-based neuron pruning with this
            ``fraction``.
        return_outcome : bool, optional
            If ``True``, also return the list of :class:`PruneOutcome`
            objects produced by each step (in order).
        **fine_tune_kwargs
            Forwarded to :func:`pwlnet.pruning.fine_tune` (e.g. ``epochs``,
            ``validation_data``).

        Returns
        -------
        PWLNet or tuple of (PWLNet, list[PruneOutcome])
            The compressed network as a new ``PWLNet`` (and the per-step
            outcomes when ``return_outcome`` is ``True``).

        Raises
        ------
        ValueError
            If ``fine_tune`` is requested without both ``X`` and ``y``.
        """
        if fine_tune and (X is None or y is None):
            raise ValueError(
                "fine_tune=True requires both X and y (training data in "
                "original units)."
            )

        outcomes: list[PruneOutcome] = []
        current = self.network

        # Step 1: exact ReLU simplification (only when bounds are available
        # and the net actually has relu hidden layers).
        if current.input_bounds is not None and any(
            layer.activation == "relu" for layer in current.layers[:-1]
        ):
            relu_outcome = prune_network(current, "relus")
            outcomes.append(relu_outcome)
            current = relu_outcome.network

        # Step 2: optional magnitude pruning.
        if magnitude_fraction is not None:
            mag_outcome = prune_network(
                current, "weight_magnitude", fraction=magnitude_fraction
            )
            outcomes.append(mag_outcome)
            current = mag_outcome.network

        # Step 3: optional importance-based neuron pruning.
        if neuron_fraction is not None:
            neuron_outcome = prune_network(
                current, "neuron", fraction=neuron_fraction, X=X
            )
            outcomes.append(neuron_outcome)
            current = neuron_outcome.network

        # Step 4: optional fine-tuning.
        if fine_tune:
            from pwlnet.pruning.fine_tune import fine_tune as _fine_tune

            ft_outcome = _fine_tune(
                current,
                X,
                y,
                validation_data=fine_tune_kwargs.pop("validation_data", None),
                **fine_tune_kwargs,
            )
            outcomes.append(ft_outcome)
            current = ft_outcome.network

        compressed = PWLNet(current, trainer=None)
        if return_outcome:
            return compressed, outcomes
        return compressed

    def to_pyomo(
        self,
        formulation: str = "relu_bigm",
        backend: str = "omlt",
        model: Any | None = None,
        block_name: str = "surrogate",
        **formulation_options: Any,
    ) -> tuple[Any, Any]:
        """Build a Pyomo/OMLT block for the wrapped network.

        Returns ``(model, block)``. If ``model`` is ``None``, a new
        ``pyomo.environ.ConcreteModel`` is created.
        """
        backend_name = str(backend).strip().lower()
        if backend_name == "omlt":
            from pwlnet.formulations.omlt import build_omlt_block

            return build_omlt_block(
                self.network,
                formulation=formulation,
                model=model,
                block_name=block_name,
            )
        if backend_name == "pyomo_pwl":
            formulation_name = str(formulation).strip().lower()
            if formulation_name not in {"tanh_pwl_sos2", "tanh_pwl_bigm"}:
                raise ValueError(
                    "backend='pyomo_pwl' currently supports only "
                    "'tanh_pwl_sos2' and 'tanh_pwl_bigm'."
                )
            import pyomo.environ as pyo

            from pwlnet.formulations.pyomo_pwl import (
                build_tanh_pwl_bigm_block,
                build_tanh_pwl_sos2_block,
            )

            builder = (
                build_tanh_pwl_sos2_block
                if formulation_name == "tanh_pwl_sos2"
                else build_tanh_pwl_bigm_block
            )
            if model is None:
                model = pyo.ConcreteModel(name="pwlnet_pyomo_pwl")
            block = builder(
                model,
                self.network,
                block_name=block_name,
                breakpoints=formulation_options.get("breakpoints", 5),
            )
            return model, block
        else:
            raise ValueError(
                f"Unsupported backend '{backend}'. Use 'omlt' or 'pyomo_pwl'."
            )

    def __repr__(self) -> str:
        return f"PWLNet(network={self.network!r})"
