"""Public SOSX solver API for generic Pyomo models and built-in processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pwlnet.branching.sosx_brancher import SosxBrancher
from pwlnet.branching.node import SosKey, Window
from pwlnet.core.network import NetworkDefinition
from pwlnet.formulations.sosx.sosx_node import SosxRelaxation
from pwlnet.formulations.sosx.sosx_solution import SosxResult
from pwlnet.utils.exceptions import FormulationError

__all__ = ["SosxSurrogateResult", "solve_sosx"]


@dataclass(frozen=True)
class SosxSurrogateResult:
    """Williams-Otto report returned by ``solve_sosx(network, ...)``."""

    solver_name: str
    status: str
    fb: float
    tr: float
    predicted_outputs: dict[str, float]
    surrogate_profit: float
    exact_outputs: dict[str, float]
    exact_profit: float
    exact_feasible: bool
    exact_optimum: dict[str, float]
    objective_value: float | None
    n_nodes: int
    solver_seconds: float
    wall_seconds: float
    best_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    node_selection: str
    branching_rule: str
    persistent: bool
    warm_started: bool
    lam: dict[Any, np.ndarray] | None
    n_workers: int | None = None
    operating_constraints: bool = True


# ---------------------------------------------------------------------------
# Picklable spec for parallel Williams-Otto surrogate optimisation
# ---------------------------------------------------------------------------

@dataclass
class _WilliamsOttoSpec:
    """Picklable build spec for parallel Williams-Otto downstream optimisation.

    Workers call :meth:`build_worker` to rebuild the full Pyomo model,
    SOS2 relaxation, solver, and track dict in-process.  The spec is
    shipped to every Ray actor via cloudpickle; :class:`NetworkDefinition`
    (numpy arrays) and :class:`~pwlnet.processes.williams_otto.model.\
WilliamsOttoParameters` (a plain dataclass) are both picklable.
    """

    network: Any  # NetworkDefinition
    params: Any   # WilliamsOttoParameters
    profit_formulation: str
    breakpoints: Any
    include_operating_constraints: bool
    block_name: str
    solver_name: str | None
    solver_options: dict[str, Any] = field(default_factory=dict)
    persistent: bool = False

    def root_windows(self) -> dict[SosKey, Window]:
        """Compute root windows analytically from network dimensions."""
        from pwlnet.formulations.pyomo_pwl.piecewise_data import build_tanh_pwl_data
        n_bp = build_tanh_pwl_data(self.breakpoints).n_breakpoints
        # The WO surrogate has one hidden tanh layer; hidden width is the
        # first dimension of the output-layer weight matrix.
        hidden = int(np.asarray(self.network.layers[-1].weights).shape[0])
        return {
            (self.block_name, 0, j): (0, n_bp - 1)
            for j in range(hidden)
        }

    def build_worker(self) -> tuple[Any, Any, bool, dict[str, Any]]:
        """Rebuild (relaxation, solver, persistent, track) inside a worker."""
        from pwlnet.formulations.sosx.sosx_node import SosxRelaxation
        from pwlnet.processes.williams_otto.model import OUTPUT_NAMES
        from pwlnet.processes.williams_otto.pyomo_model import (
            build_surrogate_optimization_model,
        )
        from pwlnet.solvers.solver_config import get_milp_solver

        model, block = build_surrogate_optimization_model(
            network=self.network,
            formulation="tanh_pwl_sos2",
            params=self.params,
            profit_formulation=self.profit_formulation,
            breakpoints=self.breakpoints,
            include_operating_constraints=self.include_operating_constraints,
            block_name=self.block_name,
        )
        relaxation = SosxRelaxation(model, [block])
        solver = get_milp_solver(self.solver_name, options=self.solver_options or None)
        output_keys = list(block.outputs.keys())
        track: dict[str, Any] = {"FB": model.FB, "TR": model.TR}
        for name, key in zip(OUTPUT_NAMES, output_keys):
            track[name] = block.outputs[key]
        return relaxation, solver, self.persistent, track


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_sosx(
    problem: Any,
    *,
    blocks: Any | None = None,
    solver: Any | str | None = None,
    solver_options: dict[str, Any] | None = None,
    sense: str = "maximize",
    track: dict[str, Any] | None = None,
    params: Any | None = None,
    profit_formulation: str = "mccormick",
    breakpoints: Any = 5,
    include_operating_constraints: bool = True,
    node_selection: str = "best_bound",
    branching_rule: str = "first",
    max_nodes: int = 2000,
    optimality_tolerance: float = 1e-7,
    violation_tolerance: float = 1e-6,
    time_limit: float | None = None,
    absolute_gap: float | None = None,
    relative_gap: float | None = None,
    parallel: bool = False,
    n_workers: int | None = None,
    ray_init_kwargs: dict[str, Any] | None = None,
    model_factory: Any | None = None,
    verbose: bool = False,
    tol: float | None = None,
    violation_tol: float | None = None,
) -> SosxResult | SosxSurrogateResult:
    """Solve a SOSX model or optimise a built-in Williams-Otto surrogate.

    ``problem`` supports two forms:

    * A Pyomo model containing one or more ``tanh_pwl_sos2`` blocks. The
      caller owns its variables, constraints, objective, sense, and tracked
      result expressions. A generic :class:`SosxResult` is returned.
    * A trained :class:`NetworkDefinition`. The Williams-Otto economic model
      is built, solved, and evaluated against the exact simulator. A
      :class:`SosxSurrogateResult` is returned.

    ``solver_options={"persistent": True}`` selects an APPSI persistent
    HiGHS/Gurobi interface. ``solver_options={"incumbent": previous_result}``
    revalidates a previous SOSX incumbent before the root search.

    ``parallel=True`` fans the node LP solves out over a pool of Ray actors.
    Each actor rebuilds its own copy of the Pyomo model in-process (so
    prebuilt solver objects cannot be shared; pass a solver name instead).
    For a :class:`NetworkDefinition` ``problem`` the model is rebuilt
    automatically. For a custom Pyomo model supply ``model_factory`` -- a
    picklable callable that returns ``(model, blocks, track_dict)``.

    The legacy names ``tol`` / ``violation_tol`` remain aliases for
    ``optimality_tolerance`` / ``violation_tolerance``.
    """
    if tol is not None:
        optimality_tolerance = float(tol)
    if violation_tol is not None:
        violation_tolerance = float(violation_tol)

    common = dict(
        solver=solver,
        solver_options=solver_options,
        node_selection=node_selection,
        branching_rule=branching_rule,
        max_nodes=max_nodes,
        optimality_tolerance=optimality_tolerance,
        violation_tolerance=violation_tolerance,
        time_limit=time_limit,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        parallel=parallel,
        n_workers=n_workers,
        ray_init_kwargs=ray_init_kwargs,
        verbose=verbose,
    )
    if isinstance(problem, NetworkDefinition):
        if blocks is not None or track is not None or sense != "maximize":
            raise FormulationError(
                "blocks, track, and a non-default sense apply only when problem "
                "is an already-built Pyomo model."
            )
        return _solve_williams_otto_network(
            problem,
            params=params,
            profit_formulation=profit_formulation,
            breakpoints=breakpoints,
            include_operating_constraints=include_operating_constraints,
            **common,
        )

    return _solve_pyomo_model(
        problem,
        blocks=blocks,
        sense=sense,
        track=track,
        model_factory=model_factory,
        **common,
    )


# ---------------------------------------------------------------------------
# Internal dispatch helpers
# ---------------------------------------------------------------------------

def _solve_pyomo_model(
    model: Any,
    *,
    blocks: Any | None,
    solver: Any | str | None,
    solver_options: dict[str, Any] | None,
    sense: str,
    track: dict[str, Any] | None,
    node_selection: str,
    branching_rule: str,
    max_nodes: int,
    optimality_tolerance: float,
    violation_tolerance: float,
    time_limit: float | None,
    absolute_gap: float | None,
    relative_gap: float | None,
    parallel: bool,
    n_workers: int | None,
    ray_init_kwargs: dict[str, Any] | None,
    model_factory: Any | None,
    verbose: bool,
) -> SosxResult:
    if parallel:
        return _solve_pyomo_model_parallel(
            model,
            blocks=blocks,
            solver=solver,
            solver_options=solver_options,
            sense=sense,
            track=track,
            model_factory=model_factory,
            node_selection=node_selection,
            branching_rule=branching_rule,
            max_nodes=max_nodes,
            optimality_tolerance=optimality_tolerance,
            violation_tolerance=violation_tolerance,
            time_limit=time_limit,
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
            n_workers=n_workers,
            ray_init_kwargs=ray_init_kwargs,
            verbose=verbose,
        )

    relaxation = SosxRelaxation(model, blocks)
    options = dict(solver_options or {})
    persistent = bool(options.pop("persistent", False))
    initial_incumbent = options.pop("incumbent", None)
    if solver is None or isinstance(solver, str):
        from pwlnet.solvers.solver_config import get_milp_solver

        solver_name = _persistent_solver_name(solver) if persistent else solver
        solver_obj = get_milp_solver(solver_name, options=options)
    else:
        if options:
            raise ValueError(
                "solver_options can only configure a solver name/None; configure "
                "an explicitly supplied solver object before passing it."
            )
        solver_obj = solver

    return SosxBrancher(
        relaxation,
        solver_obj,
        sense=sense,
        track=track,
        node_selection=node_selection,
        branching_rule=branching_rule,
        max_nodes=max_nodes,
        tol=optimality_tolerance,
        violation_tol=violation_tolerance,
        time_limit=time_limit,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        persistent=persistent,
        initial_incumbent=initial_incumbent,
        verbose=verbose,
    ).solve()


def _solve_pyomo_model_parallel(
    model: Any,
    *,
    blocks: Any | None,
    solver: Any | str | None,
    solver_options: dict[str, Any] | None,
    sense: str,
    track: dict[str, Any] | None,
    model_factory: Any | None,
    node_selection: str,
    branching_rule: str,
    max_nodes: int,
    optimality_tolerance: float,
    violation_tolerance: float,
    time_limit: float | None,
    absolute_gap: float | None,
    relative_gap: float | None,
    n_workers: int | None,
    ray_init_kwargs: dict[str, Any] | None,
    verbose: bool,
) -> SosxResult:
    """Parallel branch-and-bound over a generic Pyomo SOSX model.

    Requires ``model_factory`` -- a picklable callable that returns
    ``(model, blocks, track_dict)`` -- so each Ray worker can rebuild the
    model in-process. A lambda or module-level function works; a nested
    closure may not (depends on cloudpickle version).
    """
    if model_factory is None:
        raise FormulationError(
            "parallel=True for a Pyomo model requires model_factory: a picklable "
            "callable that returns (model, blocks, track_dict). Each Ray worker "
            "calls it to rebuild the model in-process, since Pyomo models cannot "
            "be serialised across processes. Example:\n"
            "  solve_sosx(model, parallel=True,\n"
            "             model_factory=lambda: build_my_model())"
        )

    from pwlnet.training.parallel_sosx import ParallelSosxBrancher

    options = dict(solver_options or {})
    persistent = bool(options.pop("persistent", False))
    initial_incumbent_lam = getattr(options.pop("incumbent", None), "lam", None)

    if solver is not None and not isinstance(solver, str):
        raise FormulationError(
            "parallel=True requires solver to be a name string or None; a "
            "prebuilt solver object cannot be sent across processes."
        )
    solver_name = solver
    if persistent:
        solver_name = _persistent_solver_name(solver_name)

    # Build the model once on the driver to discover root windows.
    tmp_model, tmp_blocks, tmp_track = model_factory()
    root_relaxation = SosxRelaxation(tmp_model, tmp_blocks)
    root_windows = root_relaxation.root_windows()

    spec = _GenericSosxSpec(
        model_factory=model_factory,
        root_windows_cache=root_windows,
        solver_name=solver_name,
        solver_options=options,
        persistent=persistent,
    )

    return ParallelSosxBrancher(
        spec,
        sense=sense,
        n_workers=n_workers,
        node_selection=node_selection,
        branching_rule=branching_rule,
        max_nodes=max_nodes,
        tol=optimality_tolerance,
        violation_tol=violation_tolerance,
        time_limit=time_limit,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        initial_incumbent_lam=initial_incumbent_lam,
        ray_init_kwargs=ray_init_kwargs,
        verbose=verbose,
    ).solve()


@dataclass
class _GenericSosxSpec:
    """Spec for parallel solving of any Pyomo SOSX model via a factory callable."""

    model_factory: Any  # Callable[[], tuple[model, blocks, track]]
    root_windows_cache: dict[SosKey, Window]
    solver_name: str | None
    solver_options: dict[str, Any] = field(default_factory=dict)
    persistent: bool = False

    def root_windows(self) -> dict[SosKey, Window]:
        return dict(self.root_windows_cache)

    def build_worker(self) -> tuple[Any, Any, bool, dict[str, Any]]:
        from pwlnet.formulations.sosx.sosx_node import SosxRelaxation
        from pwlnet.solvers.solver_config import get_milp_solver

        model, blocks, track = self.model_factory()
        relaxation = SosxRelaxation(model, blocks)
        solver = get_milp_solver(self.solver_name, options=self.solver_options or None)
        return relaxation, solver, self.persistent, track


def _solve_williams_otto_network(
    network: NetworkDefinition,
    *,
    solver: Any | str | None,
    solver_options: dict[str, Any] | None,
    params: Any | None,
    profit_formulation: str,
    breakpoints: Any,
    include_operating_constraints: bool,
    node_selection: str,
    branching_rule: str,
    max_nodes: int,
    optimality_tolerance: float,
    violation_tolerance: float,
    time_limit: float | None,
    absolute_gap: float | None,
    relative_gap: float | None,
    parallel: bool,
    n_workers: int | None,
    ray_init_kwargs: dict[str, Any] | None,
    verbose: bool,
) -> SosxSurrogateResult:
    from pwlnet.processes.williams_otto.model import (
        OUTPUT_NAMES,
        WilliamsOttoParameters,
    )
    from pwlnet.processes.williams_otto.pyomo_model import (
        EXACT_OPTIMUM,
        build_surrogate_optimization_model,
    )
    from pwlnet.processes.williams_otto.simulator import solve_steady_state

    p = params if params is not None else WilliamsOttoParameters()

    if parallel:
        result, n_workers_used = _solve_williams_otto_parallel(
            network=network,
            params=p,
            profit_formulation=profit_formulation,
            breakpoints=breakpoints,
            include_operating_constraints=include_operating_constraints,
            solver=solver,
            solver_options=solver_options,
            node_selection=node_selection,
            branching_rule=branching_rule,
            max_nodes=max_nodes,
            optimality_tolerance=optimality_tolerance,
            violation_tolerance=violation_tolerance,
            time_limit=time_limit,
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
            n_workers=n_workers,
            ray_init_kwargs=ray_init_kwargs,
            verbose=verbose,
        )
    else:
        n_workers_used = None
        model, block = build_surrogate_optimization_model(
            network=network,
            formulation="tanh_pwl_sos2",
            params=p,
            profit_formulation=profit_formulation,
            breakpoints=breakpoints,
            include_operating_constraints=include_operating_constraints,
        )
        output_keys = list(block.outputs.keys())
        tracked: dict[str, Any] = {"FB": model.FB, "TR": model.TR}
        for name, key in zip(OUTPUT_NAMES, output_keys):
            tracked[name] = block.outputs[key]

        result = _solve_pyomo_model(
            model,
            blocks=[block],
            solver=solver,
            solver_options=solver_options,
            sense="maximize",
            track=tracked,
            node_selection=node_selection,
            branching_rule=branching_rule,
            max_nodes=max_nodes,
            optimality_tolerance=optimality_tolerance,
            violation_tolerance=violation_tolerance,
            time_limit=time_limit,
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
            parallel=False,
            n_workers=None,
            ray_init_kwargs=None,
            model_factory=None,
            verbose=verbose,
        )

    if not result.found:
        raise FormulationError(
            f"SOSX found no SOS2-feasible incumbent (status={result.status!r})."
        )

    fb = float(np.clip(result.values["FB"], *p.fb_bounds))
    tr = float(np.clip(result.values["TR"], *p.tr_bounds))
    predicted_outputs = {name: float(result.values[name]) for name in OUTPUT_NAMES}
    surrogate_profit = float(
        (p.price_p * predicted_outputs["XP"] + p.price_e * predicted_outputs["XE"])
        * (p.fa + fb)
        - p.cost_a * p.fa
        - p.cost_b * fb
    )
    exact_state = solve_steady_state(fb, tr, params=p)
    exact_outputs = {
        name: float(value) for name, value in zip(OUTPUT_NAMES, exact_state.x)
    }
    exact_feasible = exact_outputs["XA"] <= p.xa_max and exact_outputs["XG"] <= p.xg_max

    solver_name = (
        type(solver).__name__ if solver is not None and not isinstance(solver, str)
        else solver or "appsi_highs"
    )
    return SosxSurrogateResult(
        solver_name=str(solver_name),
        status=result.status,
        fb=fb,
        tr=tr,
        predicted_outputs=predicted_outputs,
        surrogate_profit=surrogate_profit,
        exact_outputs=exact_outputs,
        exact_profit=float(exact_state.profit),
        exact_feasible=bool(exact_feasible),
        exact_optimum=dict(EXACT_OPTIMUM),
        objective_value=result.objective,
        n_nodes=result.n_nodes,
        solver_seconds=result.solver_seconds,
        wall_seconds=result.wall_seconds,
        best_bound=result.best_bound,
        absolute_gap=result.absolute_gap,
        relative_gap=result.relative_gap,
        node_selection=node_selection,
        branching_rule=branching_rule,
        persistent=result.persistent,
        warm_started=result.warm_started,
        lam=result.lam,
        n_workers=n_workers_used,
        operating_constraints=bool(include_operating_constraints),
    )


def _solve_williams_otto_parallel(
    *,
    network: NetworkDefinition,
    params: Any,
    profit_formulation: str,
    breakpoints: Any,
    include_operating_constraints: bool,
    solver: Any | str | None,
    solver_options: dict[str, Any] | None,
    node_selection: str,
    branching_rule: str,
    max_nodes: int,
    optimality_tolerance: float,
    violation_tolerance: float,
    time_limit: float | None,
    absolute_gap: float | None,
    relative_gap: float | None,
    n_workers: int | None,
    ray_init_kwargs: dict[str, Any] | None,
    verbose: bool,
) -> tuple[SosxResult, int]:
    """Run the WO surrogate optimisation over a Ray actor pool."""
    from pwlnet.training.parallel_sosx import ParallelSosxBrancher, default_worker_count

    options = dict(solver_options or {})
    persistent = bool(options.pop("persistent", False))
    initial_incumbent_lam = getattr(options.pop("incumbent", None), "lam", None)

    if solver is not None and not isinstance(solver, str):
        raise FormulationError(
            "parallel=True requires solver to be a name string or None; a "
            "prebuilt solver object cannot be sent across processes."
        )
    solver_name = solver
    if persistent:
        solver_name = _persistent_solver_name(solver_name)

    spec = _WilliamsOttoSpec(
        network=network,
        params=params,
        profit_formulation=profit_formulation,
        breakpoints=breakpoints,
        include_operating_constraints=include_operating_constraints,
        block_name="surrogate",
        solver_name=solver_name,
        solver_options=options,
        persistent=persistent,
    )
    resolved_workers = (
        default_worker_count() if n_workers is None else max(1, int(n_workers))
    )
    result = ParallelSosxBrancher(
        spec,
        sense="maximize",
        n_workers=resolved_workers,
        node_selection=node_selection,
        branching_rule=branching_rule,
        max_nodes=max_nodes,
        tol=optimality_tolerance,
        violation_tol=violation_tolerance,
        time_limit=time_limit,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        initial_incumbent_lam=initial_incumbent_lam,
        ray_init_kwargs=ray_init_kwargs,
        verbose=verbose,
    ).solve()
    return result, resolved_workers


def _persistent_solver_name(name: str | None) -> str:
    normalized = "appsi_highs" if name is None else str(name).strip().lower()
    aliases = {
        "highs": "appsi_highs",
        "appsi_highs": "appsi_highs",
        "gurobi": "appsi_gurobi",
        "gurobi_direct": "appsi_gurobi",
        "gurobi_persistent": "appsi_gurobi",
        "appsi_gurobi": "appsi_gurobi",
    }
    if normalized not in aliases:
        raise FormulationError(
            "persistent SOSX mode currently supports HiGHS or Gurobi through "
            f"their APPSI interfaces, got solver={name!r}."
        )
    return aliases[normalized]
