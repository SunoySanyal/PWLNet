"""Production SOSX-MAE neural-network trainer (fixed linear readout).

:class:`SosxTrainer` globally trains the input layer of a bounded,
single-hidden-layer PWL-tanh network under a mean-absolute-error loss,
keeping the output weights ``W2`` fixed. With ``W2`` fixed the only
bilinear product (trainable weight x trainable activation) disappears, so
every branch-and-bound node is a linear program solvable by HiGHS and the
:class:`pwlnet.branching.SosxBrancher` recovers the SOS2 segment
structure by branching on the per-sample convex-combination weights.

First-release guarantee
-----------------------
For a bounded single-hidden-layer PWL-tanh network with fixed output
weights and MAE loss, :meth:`SosxTrainer.fit` globally solves the stated
training formulation to the requested node, time, or optimality-gap
tolerance, returning a standard :class:`NetworkDefinition` usable by the
existing embedding and :func:`pwlnet.solve_sosx` APIs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.branching.sosx_brancher import SosxBrancher
from pwlnet.core.layers import DenseLayer
from pwlnet.core.network import NetworkDefinition
from pwlnet.core.scaling import OffsetScaler
from pwlnet.formulations.pyomo_pwl.piecewise_data import build_tanh_pwl_data
from pwlnet.formulations.sosx.sosx_branching_rules import split_window, split_window_Y
from pwlnet.formulations.sosx.sosx_solution import SosxResult
from pwlnet.formulations.sosx.sosx_violation import find_violations, find_violations_Y
from pwlnet.formulations.sosx.training_model import (
    build_sos2_mae_training_model,
    build_sosx_mae_training_model,
)
from pwlnet.formulations.sosx.training_relaxation import (
    Sos2TrainingRelaxation,
    SosxTrainingRelaxation,
)
from pwlnet.formulations.sosx.training_solution import (
    network_from_tracked_values,
    pwl_training_mae,
    track_training_variables,
    training_incumbent_from_network,
    training_incumbent_lam,
    training_incumbent_Y,
)
from pwlnet.formulations.sosx.weight_bounds import WeightBounds, resolve_weight_bounds
from pwlnet.solvers.solver_config import get_milp_solver
from pwlnet.utils.exceptions import TrainingError
from pwlnet.utils.logging import get_logger
from pwlnet.utils.typing import ArrayLike
from pwlnet.utils.validation import as_2d_array

__all__ = ["SosxTrainer"]

_LOGGER = get_logger("training.sosx_trainer")

_SUPPORTED_LOSSES = ("mae",)
_READOUT_STRATEGIES = ("keras", "random", "least_squares")
_FORMULATIONS = ("sosx", "sos2")
_SCALING_METHODS = ("standard", "minmax", "none")
_PERSISTENT_ALIASES = {
    None: "appsi_highs",
    "highs": "appsi_highs",
    "appsi_highs": "appsi_highs",
    "gurobi": "appsi_gurobi",
    "appsi_gurobi": "appsi_gurobi",
}


class SosxTrainer:
    """Globally train a fixed-readout PWL-tanh network with SOSX-MAE.

    Parameters
    ----------
    hidden_neurons : int
        Width of the single tanh hidden layer.
    breakpoints : int or array_like, optional
        PWL-tanh breakpoint count (or explicit breakpoints). Default 5.
    loss : {"mae"}, optional
        Training loss. Only ``"mae"`` is supported in this release.
    readout : {"keras", "random", "least_squares"} or array_like, optional
        How to obtain the fixed ``W2``. ``"keras"`` pretrains the full
        network with Keras and fixes its readout (and warm-starts from
        it); ``"random"`` draws a seeded readout; ``"least_squares"``
        fits ``W2`` to random hidden features; an array supplies ``W2``
        directly. Default ``"keras"``.
    solver : str or object or None, optional
        MILP/LP solver name or a configured Pyomo solver object. Default
        ``"appsi_highs"``.
    solver_options : dict or None, optional
        Forwarded to the solver factory. ``{"persistent": True}`` selects
        an APPSI persistent HiGHS/Gurobi interface.
    node_selection : {"best_bound", "depth_first"}, optional
        Open-node strategy. Default ``"best_bound"``.
    branching_rule : {"first", "most_violated"}, optional
        SOS-set selection rule. Default ``"first"``.
    max_nodes : int, optional
        Safety cap on solved nodes. Default 2000.
    time_limit : float or None, optional
        Brancher wall-clock limit in seconds.
    absolute_gap, relative_gap : float or None, optional
        Optimality-gap stopping tolerances.
    weight_bound, bias_bound, output_bias_bound : float or None, optional
        Per-group box half-widths for ``W1``/``b1``/``b2``. When omitted
        the bounds are derived to contain the warm-start network (falling
        back to a finite default).
    bounds : WeightBounds or None, optional
        Explicit box bounds; overrides the per-group scalars.
    scaling_method : {"standard", "minmax", "none"}, optional
        Scaler method for inputs/targets. Default ``"standard"``.
    scale_inputs, scale_outputs : bool, optional
        Whether to fit input/output scalers. Default ``True``.
    random_seed : int or None, optional
        Seed for the random/least-squares readout and the Keras pretrain.
    keras_options : dict or None, optional
        Extra options for the internal :class:`KerasTrainer` when
        ``readout="keras"`` (e.g. ``epochs``, ``batch_size``).
    formulation : {"sosx", "sos2"}, optional
        Training LP formulation. ``"sosx"`` (default) uses the Y-variable
        segment-fill formulation with locally ideal constraints and
        asymmetric branching from Koksal et al. (2026). ``"sos2"`` uses
        the original convex-combination (λ) formulation with symmetric
        bisection branching.
    parallel : bool, optional
        Solve the branch-and-bound node LPs across a pool of Ray actors
        instead of one at a time. Each worker rebuilds its own copy of the
        training relaxation, so the wide ``samples x neurons`` SOS-set tree
        fans out across processes/cores -- the tractability lever for more
        breakpoints/samples. Requires the optional ``ray`` dependency.
        Default ``False``.
    n_workers : int or None, optional
        Number of Ray actors when ``parallel=True``. ``None`` leaves one
        core free for the driver (``cpu_count - 1``).
    ray_init_kwargs : dict or None, optional
        Extra keyword arguments for ``ray.init`` (used only when this
        trainer has to start Ray itself).
    verbose : bool, optional
        Log brancher incumbent/infeasible events.

    Attributes
    ----------
    network_ : NetworkDefinition or None
        The trained network (set by :meth:`fit`).
    result_ : SosxResult or None
        The brancher report.
    training_mae_, validation_mae_ : float or None
        PWL-tanh MAE in scaled space on the training/validation sets.
    n_nodes_, wall_seconds_, solver_seconds_ : numbers
        Solve diagnostics from the brancher. Under ``parallel=True``
        ``solver_seconds_`` is the *summed* worker solve time (it can
        exceed ``wall_seconds_`` -- that gap is the parallel speedup).
    n_workers_ : int or None
        Number of Ray actors used (``None`` for a serial solve).
    optimality_gap_ : float or None
        Final relative optimality gap.
    W2_ : numpy.ndarray or None
        The fixed readout used.
    warm_network_ : NetworkDefinition or None
        The scaled-space warm-start network behind the chosen readout
        (``None`` for a cold ``random`` readout). Reuse it as
        ``W2=trainer.W2_, initial_network=trainer.warm_network_`` to refit
        the identical subproblem without re-pretraining.
    weight_bounds_ : WeightBounds or None
        The resolved box bounds.
    """

    def __init__(
        self,
        hidden_neurons: int,
        *,
        breakpoints: int | ArrayLike = 5,
        loss: str = "mae",
        readout: str | ArrayLike = "keras",
        solver: str | Any | None = "appsi_highs",
        solver_options: dict[str, Any] | None = None,
        node_selection: str = "best_bound",
        branching_rule: str = "first",
        max_nodes: int = 2000,
        time_limit: float | None = None,
        absolute_gap: float | None = None,
        relative_gap: float | None = None,
        weight_bound: float | None = None,
        bias_bound: float | None = None,
        output_bias_bound: float | None = None,
        bounds: WeightBounds | None = None,
        scaling_method: str = "standard",
        scale_inputs: bool = True,
        scale_outputs: bool = True,
        random_seed: int | None = None,
        keras_options: dict[str, Any] | None = None,
        formulation: str = "sosx",
        parallel: bool = False,
        n_workers: int | None = None,
        ray_init_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_neurons = int(hidden_neurons)
        if self.hidden_neurons <= 0:
            raise ValueError(
                f"hidden_neurons must be a positive int, got {hidden_neurons!r}."
            )
        self.breakpoints = breakpoints
        loss_name = str(loss).strip().lower()
        if loss_name not in _SUPPORTED_LOSSES:
            raise ValueError(
                f"loss must be one of {list(_SUPPORTED_LOSSES)} in this release, "
                f"got {loss!r}."
            )
        self.loss = loss_name

        if isinstance(readout, str):
            strategy = readout.strip().lower()
            if strategy not in _READOUT_STRATEGIES:
                raise ValueError(
                    f"readout must be one of {list(_READOUT_STRATEGIES)} or an "
                    f"array, got {readout!r}."
                )
            self.readout: str | np.ndarray = strategy
        else:
            self.readout = np.asarray(readout, dtype=float)

        self.solver = solver
        self.solver_options = dict(solver_options or {})
        self.node_selection = node_selection
        self.branching_rule = branching_rule
        self.max_nodes = int(max_nodes)
        self.time_limit = time_limit
        self.absolute_gap = absolute_gap
        self.relative_gap = relative_gap
        self.weight_bound = weight_bound
        self.bias_bound = bias_bound
        self.output_bias_bound = output_bias_bound
        self.bounds = bounds

        method = str(scaling_method).strip().lower()
        if method not in _SCALING_METHODS:
            raise ValueError(
                f"scaling_method must be one of {list(_SCALING_METHODS)}, "
                f"got {scaling_method!r}."
            )
        self.scaling_method = method
        self.scale_inputs = bool(scale_inputs)
        self.scale_outputs = bool(scale_outputs)
        self.random_seed = None if random_seed is None else int(random_seed)
        self.keras_options = dict(keras_options or {})
        form = str(formulation).strip().lower()
        if form not in _FORMULATIONS:
            raise ValueError(
                f"formulation must be one of {list(_FORMULATIONS)}, got {formulation!r}."
            )
        self.formulation = form
        self.parallel = bool(parallel)
        if n_workers is not None and int(n_workers) <= 0:
            raise ValueError(f"n_workers must be a positive int or None, got {n_workers!r}.")
        self.n_workers = None if n_workers is None else int(n_workers)
        self.ray_init_kwargs = dict(ray_init_kwargs or {})
        self.verbose = bool(verbose)

        # Set by fit().
        self.network_: NetworkDefinition | None = None
        self.result_: SosxResult | None = None
        self.training_mae_: float | None = None
        self.validation_mae_: float | None = None
        self.n_nodes_: int | None = None
        self.wall_seconds_: float | None = None
        self.solver_seconds_: float | None = None
        self.optimality_gap_: float | None = None
        self.n_workers_: int | None = None
        self.W2_: np.ndarray | None = None
        self.warm_network_: NetworkDefinition | None = None
        self.weight_bounds_: WeightBounds | None = None
        self.input_scaler_: OffsetScaler | None = None
        self.output_scaler_: OffsetScaler | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
        *,
        initial_network: NetworkDefinition | None = None,
        W2: ArrayLike | None = None,
        incumbent: SosxResult | Any | None = None,
    ) -> NetworkDefinition:
        """Globally train the input layer and return a NetworkDefinition.

        Parameters
        ----------
        X, y : array_like
            Training inputs/targets in ORIGINAL units.
        validation_data : tuple of (array_like, array_like) or None, optional
            Explicit ``(X_val, y_val)`` in original units, scored after
            training (scaled with the training scalers).
        initial_network : NetworkDefinition or None, optional
            A one-hidden-layer network used to warm-start the search (and,
            when ``readout`` does not otherwise pin ``W2``, to supply it).
        W2 : array_like or None, optional
            Explicit fixed readout, overriding ``readout``.
        incumbent : SosxResult or SosxTrainer-result or None, optional
            A previous SOSX incumbent (or any object with a ``lam`` /
            ``result_`` attribute) to revalidate and reuse.

        Returns
        -------
        NetworkDefinition
            The trained network, carrying the fitted scalers and per-input
            bounds.
        """
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

        # Fit scalers in ORIGINAL space; the training model lives in scaled space.
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
            self.input_scaler_.transform(X_arr)
            if self.input_scaler_ is not None
            else X_arr
        )
        y_scaled = (
            self.output_scaler_.transform(y_arr)
            if self.output_scaler_ is not None
            else y_arr
        )

        # Resolve the fixed readout W2 and (optionally) a warm-start network.
        fixed_W2, warm_network = self._resolve_readout(
            X_scaled, y_scaled, explicit_W2=W2, initial_network=initial_network
        )
        if fixed_W2.shape[0] != self.hidden_neurons:
            raise TrainingError(
                f"W2 has {fixed_W2.shape[0]} hidden rows but hidden_neurons="
                f"{self.hidden_neurons}."
            )
        self.W2_ = fixed_W2
        # The (scaled-space) warm-start net behind ``readout``/``initial_network``;
        # exposed so callers can reuse the same fixed readout + warm start across
        # several fits (e.g. a worker-count sweep) without re-pretraining.
        self.warm_network_ = warm_network

        # Resolve finite box bounds; widen to contain the warm-start network.
        self.weight_bounds_ = resolve_weight_bounds(
            self.bounds,
            weight_bound=self.weight_bound,
            bias_bound=self.bias_bound,
            output_bias_bound=self.output_bias_bound,
            cover=warm_network,
        )

        n_inputs = X_scaled.shape[1]
        hidden = self.hidden_neurons
        n_outputs = y_scaled.shape[1]

        if self.parallel:
            result = self._run_parallel(
                X_scaled, y_scaled, fixed_W2, warm_network, incumbent
            )
        else:
            result = self._run_serial(
                X_scaled, y_scaled, fixed_W2, warm_network, incumbent
            )
        if not result.found:
            raise TrainingError(
                f"SOSX training found no feasible incumbent (status={result.status!r}). "
                "Try widening the weight bounds, raising max_nodes/time_limit, or "
                "checking the fixed readout."
            )

        scaled_network = network_from_tracked_values(
            result.values,
            fixed_W2,
            n_inputs=n_inputs,
            hidden=hidden,
            n_outputs=n_outputs,
        )

        # Diagnostics (PWL MAE in scaled space, comparable to the objective).
        self.result_ = result
        self.training_mae_ = pwl_training_mae(
            scaled_network, X_scaled, y_scaled, breakpoints=self.breakpoints
        )
        self.validation_mae_ = self._validation_mae(
            scaled_network, validation_data, self.breakpoints
        )
        self.n_nodes_ = result.n_nodes
        self.wall_seconds_ = result.wall_seconds
        self.solver_seconds_ = result.solver_seconds
        self.optimality_gap_ = result.relative_gap

        input_bounds = [
            (float(lo), float(hi))
            for lo, hi in zip(X_arr.min(axis=0), X_arr.max(axis=0))
        ]
        network = NetworkDefinition(
            layers=scaled_network.layers,
            input_scaler=self.input_scaler_,
            output_scaler=self.output_scaler_,
            input_bounds=input_bounds,
        )
        self.network_ = network
        return network

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_readout(
        self,
        X_scaled: np.ndarray,
        y_scaled: np.ndarray,
        *,
        explicit_W2: ArrayLike | None,
        initial_network: NetworkDefinition | None,
    ) -> tuple[np.ndarray, NetworkDefinition | None]:
        """Return ``(W2, warm_network)`` in scaled space."""
        hidden = self.hidden_neurons
        n_outputs = y_scaled.shape[1]

        if explicit_W2 is not None:
            W2 = np.asarray(explicit_W2, dtype=float)
            if W2.ndim != 2 or W2.shape != (hidden, n_outputs):
                raise TrainingError(
                    f"Explicit W2 must have shape ({hidden}, {n_outputs}), "
                    f"got {W2.shape}."
                )
            return W2, initial_network

        if initial_network is not None:
            W2 = np.asarray(initial_network.layers[-1].weights, dtype=float)
            if W2.shape != (hidden, n_outputs):
                raise TrainingError(
                    f"initial_network readout has shape {W2.shape}, expected "
                    f"({hidden}, {n_outputs})."
                )
            return W2, initial_network

        if isinstance(self.readout, np.ndarray):
            W2 = self.readout
            if W2.shape != (hidden, n_outputs):
                raise TrainingError(
                    f"Supplied readout array has shape {W2.shape}, expected "
                    f"({hidden}, {n_outputs})."
                )
            return W2, None

        if self.readout == "keras":
            return self._keras_readout(X_scaled, y_scaled)
        if self.readout == "least_squares":
            return self._least_squares_readout(X_scaled, y_scaled)
        # "random"
        rng = np.random.default_rng(self.random_seed)
        W2 = rng.normal(0.0, 0.5, size=(hidden, n_outputs))
        return W2, None

    def _keras_readout(
        self, X_scaled: np.ndarray, y_scaled: np.ndarray
    ) -> tuple[np.ndarray, NetworkDefinition]:
        """Pretrain the full network with Keras (in scaled space) for ``W2``."""
        from pwlnet.training.keras_trainer import KerasTrainer

        options = dict(self.keras_options)
        # The trainer owns scaling; Keras must train in the same scaled space.
        options.update(scale_inputs=False, scale_outputs=False)
        options.setdefault("loss", self.loss)
        options.setdefault("epochs", 300)
        options.setdefault("verbose", 0)
        if self.random_seed is not None:
            options.setdefault("random_seed", self.random_seed)
        keras_net = KerasTrainer(
            hidden_layers=[self.hidden_neurons],
            activation="tanh",
            **options,
        ).fit(X_scaled, y_scaled)
        W2 = np.asarray(keras_net.layers[-1].weights, dtype=float)
        return W2, keras_net

    def _least_squares_readout(
        self, X_scaled: np.ndarray, y_scaled: np.ndarray
    ) -> tuple[np.ndarray, NetworkDefinition]:
        """Fit ``W2``/``b2`` to random hidden PWL features via least squares."""
        rng = np.random.default_rng(self.random_seed)
        n_inputs = X_scaled.shape[1]
        hidden = self.hidden_neurons
        pwl = build_tanh_pwl_data(self.breakpoints)
        W1 = rng.normal(0.0, 1.0, size=(n_inputs, hidden))
        b1 = rng.normal(0.0, 0.5, size=hidden)
        H = pwl.evaluate(X_scaled @ W1 + b1)
        design = np.hstack([H, np.ones((H.shape[0], 1))])
        coef, *_ = np.linalg.lstsq(design, y_scaled, rcond=None)
        W2 = coef[:hidden]
        b2 = coef[hidden]
        warm_network = NetworkDefinition(
            layers=[DenseLayer(W1, b1, "tanh"), DenseLayer(W2, b2, "linear")]
        )
        return W2, warm_network

    def _resolve_incumbent(
        self,
        training_model: Any,
        X_scaled: np.ndarray,
        warm_network: NetworkDefinition | None,
        incumbent: SosxResult | Any | None,
    ) -> SosxResult | dict[Any, Any] | None:
        """Pick the warm-start incumbent: explicit one first, else the network."""
        def _from_network(net: NetworkDefinition) -> SosxResult:
            if self.formulation == "sos2":
                return training_incumbent_lam(net, X_scaled, training_model.pwl)
            return training_incumbent_from_network(training_model, net, X_scaled)

        if incumbent is not None:
            if isinstance(incumbent, SosxResult):
                return incumbent
            if isinstance(incumbent, NetworkDefinition):
                return _from_network(incumbent)
            result = getattr(incumbent, "result_", None)
            if isinstance(result, SosxResult):
                return result
            if hasattr(incumbent, "lam"):
                return incumbent
            raise TrainingError(
                "incumbent must be a SosxResult, a NetworkDefinition, an object "
                "exposing 'lam', or a previous SosxTrainer; got "
                f"{type(incumbent).__name__}."
            )
        if warm_network is not None:
            return _from_network(warm_network)
        return None

    def _build_solver(self) -> tuple[Any, bool]:
        """Resolve the solver object and the persistent flag."""
        options = dict(self.solver_options)
        persistent = bool(options.pop("persistent", False))
        solver = self.solver
        if solver is None or isinstance(solver, str):
            name = solver
            if persistent:
                key = None if name is None else str(name).strip().lower()
                if key not in _PERSISTENT_ALIASES:
                    raise TrainingError(
                        "persistent SOSX training supports HiGHS or Gurobi through "
                        f"their APPSI interfaces, got solver={solver!r}."
                    )
                name = _PERSISTENT_ALIASES[key]
            return get_milp_solver(name, options=options), persistent
        if options:
            raise TrainingError(
                "solver_options can only configure a solver name/None; configure "
                "an explicitly supplied solver object before passing it."
            )
        return solver, persistent

    # ------------------------------------------------------------------
    # Branch-and-bound drivers (serial / Ray-parallel)
    # ------------------------------------------------------------------
    def _run_serial(
        self,
        X_scaled: np.ndarray,
        y_scaled: np.ndarray,
        fixed_W2: np.ndarray,
        warm_network: NetworkDefinition | None,
        incumbent: SosxResult | Any | None,
    ) -> SosxResult:
        """Single-process branch-and-bound on one reusable relaxation."""
        if self.formulation == "sos2":
            training_model = build_sos2_mae_training_model(
                X_scaled,
                y_scaled,
                fixed_W2,
                breakpoints=self.breakpoints,
                bounds=self.weight_bounds_,
            )
            relaxation = Sos2TrainingRelaxation(training_model)
            violation_fn = None   # default λ-based find_violations
            split_fn = None       # default λ-based split_window (bisection)
        else:
            training_model = build_sosx_mae_training_model(
                X_scaled,
                y_scaled,
                fixed_W2,
                breakpoints=self.breakpoints,
                bounds=self.weight_bounds_,
            )
            relaxation = SosxTrainingRelaxation(training_model)
            u_values = relaxation.u_values

            def _violation_fn(Y_values, windows, tol):
                return find_violations_Y(Y_values, u_values, windows, tol=tol)

            violation_fn = _violation_fn
            split_fn = split_window_Y

        track = track_training_variables(training_model)
        initial_incumbent = self._resolve_incumbent(
            training_model, X_scaled, warm_network, incumbent
        )
        solver_obj, persistent = self._build_solver()

        return SosxBrancher(
            relaxation,
            solver_obj,
            sense="minimize",
            track=track,
            node_selection=self.node_selection,
            branching_rule=self.branching_rule,
            max_nodes=self.max_nodes,
            time_limit=self.time_limit,
            absolute_gap=self.absolute_gap,
            relative_gap=self.relative_gap,
            persistent=persistent,
            initial_incumbent=initial_incumbent,
            violation_fn=violation_fn,
            split_fn=split_fn,
            verbose=self.verbose,
        ).solve()

    def _run_parallel(
        self,
        X_scaled: np.ndarray,
        y_scaled: np.ndarray,
        fixed_W2: np.ndarray,
        warm_network: NetworkDefinition | None,
        incumbent: SosxResult | Any | None,
    ) -> SosxResult:
        """Ray-parallel branch-and-bound: workers rebuild the relaxation.

        Each child node's LP is solved on a separate Ray actor, so the
        wide ``samples x neurons`` SOS-set tree -- which the serial solver
        walks one node at a time -- fans out across processes/cores. The
        global optimum is unchanged; only the explored-node count varies
        with worker timing.
        """
        from pwlnet.training.parallel_sosx import (
            ParallelSosxBrancher,
            SosxTrainingSpec,
            default_worker_count,
        )

        name, options, persistent = self._resolve_solver_name()
        spec = SosxTrainingSpec(
            X=X_scaled,
            y=y_scaled,
            W2=fixed_W2,
            breakpoints=self.breakpoints,
            bounds=self.weight_bounds_,
            solver_name=name,
            solver_options=options,
            persistent=persistent,
            formulation=self.formulation,
        )
        initial_lam = self._resolve_incumbent_lam(X_scaled, warm_network, incumbent)
        n_workers = (
            default_worker_count() if self.n_workers is None else self.n_workers
        )
        self.n_workers_ = int(n_workers)
        return ParallelSosxBrancher(
            spec,
            sense="minimize",
            n_workers=n_workers,
            node_selection=self.node_selection,
            branching_rule=self.branching_rule,
            max_nodes=self.max_nodes,
            time_limit=self.time_limit,
            absolute_gap=self.absolute_gap,
            relative_gap=self.relative_gap,
            initial_incumbent_lam=initial_lam,
            ray_init_kwargs=self.ray_init_kwargs,
            verbose=self.verbose,
        ).solve()

    def _resolve_solver_name(self) -> tuple[str | None, dict[str, Any], bool]:
        """Resolve ``(solver_name, options, persistent)`` for the workers.

        Parallel workers each rebuild the solver in their own process, so a
        prebuilt solver object cannot be shared -- only a name/None is
        accepted (the persistent flag is honoured per worker).
        """
        options = dict(self.solver_options)
        persistent = bool(options.pop("persistent", False))
        solver = self.solver
        if solver is not None and not isinstance(solver, str):
            raise TrainingError(
                "parallel=True requires solver to be a name string or None; it is "
                "rebuilt inside each Ray worker, so a prebuilt solver object cannot "
                "be reused. Pass solver='appsi_highs' (or another name) instead."
            )
        name = solver
        if persistent:
            key = None if name is None else str(name).strip().lower()
            if key not in _PERSISTENT_ALIASES:
                raise TrainingError(
                    "persistent SOSX training supports HiGHS or Gurobi through their "
                    f"APPSI interfaces, got solver={solver!r}."
                )
            name = _PERSISTENT_ALIASES[key]
        return name, options, persistent

    def _resolve_incumbent_lam(
        self,
        X_scaled: np.ndarray,
        warm_network: NetworkDefinition | None,
        incumbent: SosxResult | Any | None,
    ) -> dict[Any, np.ndarray] | None:
        """Pick the warm-start SOS-set weights for the parallel brancher.

        Mirrors :meth:`_resolve_incumbent` but returns just the ``lam``
        mapping (model-free), since the parallel driver never builds the
        Pyomo model on this process.
        """
        if incumbent is not None:
            source = incumbent
            if isinstance(source, NetworkDefinition):
                pwl = build_tanh_pwl_data(self.breakpoints)
                fn = training_incumbent_lam if self.formulation == "sos2" else training_incumbent_Y
                return fn(source, X_scaled, pwl).lam
            if not isinstance(source, SosxResult):
                result = getattr(source, "result_", None)
                if isinstance(result, SosxResult):
                    source = result
                elif not hasattr(source, "lam"):
                    raise TrainingError(
                        "incumbent must be a SosxResult, a NetworkDefinition, an "
                        "object exposing 'lam', or a previous SosxTrainer; got "
                        f"{type(incumbent).__name__}."
                    )
            return getattr(source, "lam", None)
        if warm_network is not None:
            pwl = build_tanh_pwl_data(self.breakpoints)
            fn = training_incumbent_lam if self.formulation == "sos2" else training_incumbent_Y
            return fn(warm_network, X_scaled, pwl).lam
        return None

    def _validation_mae(
        self,
        scaled_network: NetworkDefinition,
        validation_data: tuple[ArrayLike, ArrayLike] | None,
        breakpoints: int,
    ) -> float | None:
        if validation_data is None:
            return None
        X_val = as_2d_array(validation_data[0], name="X_val")
        y_val_in = np.asarray(validation_data[1], dtype=float)
        if y_val_in.ndim == 1:
            y_val_in = y_val_in.reshape(-1, 1)
        y_val = as_2d_array(y_val_in, name="y_val")
        X_val_scaled = (
            self.input_scaler_.transform(X_val)
            if self.input_scaler_ is not None
            else X_val
        )
        y_val_scaled = (
            self.output_scaler_.transform(y_val)
            if self.output_scaler_ is not None
            else y_val
        )
        return pwl_training_mae(
            scaled_network, X_val_scaled, y_val_scaled, breakpoints=breakpoints
        )

    def __repr__(self) -> str:
        readout = (
            "array" if isinstance(self.readout, np.ndarray) else self.readout
        )
        return (
            f"SosxTrainer(hidden_neurons={self.hidden_neurons}, "
            f"breakpoints={self.breakpoints}, loss='{self.loss}', "
            f"readout='{readout}', fitted={self.network_ is not None})"
        )
