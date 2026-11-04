"""Ray-parallelised branch-and-bound for SOSX models.

The serial :class:`pwlnet.branching.SosxBrancher` solves one node
relaxation at a time on a single mutated Pyomo model. This module keeps
the *same* best-bound branch-and-bound logic but solves the open nodes
**in parallel** with a pool of Ray actors. Each actor rebuilds its own
copy of the relaxation from a picklable *spec* (numpy arrays + bounds +
solver name, never a live Pyomo model) and solves whatever breakpoint
windows the driver hands it.

The driver stays single-threaded: it owns the frontier, the incumbent,
and the branching decisions, so the search is identical to the serial
one -- only the LP solves fan out across processes and cores.

Spec protocol
-------------
Any spec passed to :class:`ParallelSosxBrancher` must be picklable
(shipped once to every worker) and expose:

``root_windows() -> dict[SosKey, Window]``
    The fully-open breakpoint window for every SOS set (seeds the root
    node without building the Pyomo model on the driver process).

``build_worker() -> tuple[relaxation, solver, persistent, track]``
    Called once per actor to build the Pyomo model and solver in-process.
    *track* is a ``dict[str, pyo.VarData]`` of named expressions to
    snapshot when a SOS2-feasible (incumbent-candidate) node is found.

``persistent : bool``
    Whether workers use an APPSI persistent solver instance.

:class:`SosxTrainingSpec` implements this protocol for the SOSX-MAE
neural-network training problem.  :class:`pwlnet.solvers.sosx_solver`
provides ``_WilliamsOttoSpec`` for downstream process optimisation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pwlnet.branching.branch_log import BranchLog
from pwlnet.branching.fathoming import can_fathom_by_bound, is_improving
from pwlnet.branching.incumbent import Incumbent
from pwlnet.branching.node import Node, SosKey, Window
from pwlnet.branching.node_selection import Frontier
from pwlnet.formulations.pyomo_pwl.piecewise_data import build_tanh_pwl_data
from pwlnet.formulations.sosx.sosx_branching_rules import (
    select_violation,
    split_window,
    split_window_at,
    split_window_Y,
)
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
from pwlnet.formulations.sosx.training_solution import track_training_variables
from pwlnet.formulations.sosx.weight_bounds import WeightBounds
from pwlnet.solvers.solver_config import get_milp_solver
from pwlnet.utils.exceptions import TrainingError
from pwlnet.utils.typing import ArrayLike

__all__ = ["SosxTrainingSpec", "ParallelSosxBrancher", "default_worker_count"]

_MAXIMIZE_NAMES = ("max", "maximize")
_MINIMIZE_NAMES = ("min", "minimize")


def default_worker_count(ray: Any | None = None) -> int:
    """Safe default actor count: leave one CPU for the driver/OS.

    When *ray* is a live, initialised Ray module the count is clamped to
    the cluster's reported CPU budget so we never advertise more logical
    workers than the machine provides.
    """
    if ray is not None and ray.is_initialized():
        try:
            cluster_cpus = int(ray.cluster_resources().get("CPU", os.cpu_count() or 2))
            return max(1, cluster_cpus - 1)
        except Exception:
            pass
    return max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------------------
# Worker result dataclass -- replaces the untyped 8-tuple
# ---------------------------------------------------------------------------

@dataclass
class _WorkerResult:
    """Compact, picklable outcome of one node solve on a Ray worker.

    Only SOS2-feasible (incumbent-candidate) nodes carry the full *lam*
    and *values* dicts.  A violating node returns only *branch_key* and
    *branch_weights* (one SOS set's weights) so the driver is not flooded
    with per-sample lambda tables as the worker count grows.
    """

    feasible: bool
    objective: float
    solve_seconds: float
    has_violation: bool
    branch_key: SosKey | None
    branch_k: int | None          # 1-indexed segment of the Y violation
    lam: dict[SosKey, np.ndarray] | None
    values: dict[str, float] | None


# ---------------------------------------------------------------------------
# Picklable build spec for SOSX-MAE training
# ---------------------------------------------------------------------------

@dataclass
class SosxTrainingSpec:
    """Everything a worker needs to rebuild the SOSX-MAE training relaxation.

    Holds only picklable data (numpy arrays, a :class:`WeightBounds`, the
    solver *name* and options) -- never a live Pyomo model or solver
    object. Ray serialises this once per actor; the actor then calls
    :meth:`build_worker` to construct its own model, relaxation and solver
    in-process.

    Parameters
    ----------
    X, y : numpy.ndarray
        Training inputs/targets in the model's (scaled) space.
    W2 : numpy.ndarray
        Fixed output weight matrix, shape ``(hidden, n_outputs)``.
    breakpoints : int or array_like
        PWL-tanh breakpoint count (or explicit breakpoints).
    bounds : WeightBounds
        Finite box bounds for ``W1``/``b1``/``b2``.
    solver_name : str or None
        Pyomo solver factory name resolved by each worker (an object is
        rejected -- it could not be rebuilt per process).
    solver_options : dict
        Options forwarded to the solver factory (``persistent`` already
        stripped out).
    persistent : bool
        Whether each worker reuses an APPSI persistent solver instance.
    """

    X: np.ndarray
    y: np.ndarray
    W2: np.ndarray
    breakpoints: int | ArrayLike
    bounds: WeightBounds
    solver_name: str | None
    solver_options: dict[str, Any] = field(default_factory=dict)
    persistent: bool = False
    formulation: str = "sosx"

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self.X).shape[0])

    @property
    def hidden(self) -> int:
        return int(np.asarray(self.W2).shape[0])

    @property
    def n_breakpoints(self) -> int:
        """Number of PWL breakpoints K."""
        return build_tanh_pwl_data(self.breakpoints).n_breakpoints

    @property
    def n_segments(self) -> int:
        """Number of PWL segments P = n_breakpoints - 1."""
        return self.n_breakpoints - 1

    def root_windows(self) -> dict[SosKey, Window]:
        """The fully-open window for every SOS set.

        Y-formulation (sosx): segment-indexed ``(1, P)``.
        λ-formulation (sos2): breakpoint-indexed ``(0, K-1)``.
        """
        if self.formulation == "sos2":
            hi = self.n_breakpoints - 1
            return {
                ("training", int(s), int(j)): (0, hi)
                for s in range(self.n_samples)
                for j in range(self.hidden)
            }
        P = self.n_segments
        return {
            ("training", int(s), int(j)): (1, P)
            for s in range(self.n_samples)
            for j in range(self.hidden)
        }

    def build_worker(self) -> tuple[Any, Any, bool, dict[str, Any]]:
        """Rebuild ``(relaxation, solver, persistent, track)`` inside a worker."""
        if self.formulation == "sos2":
            model = build_sos2_mae_training_model(
                self.X, self.y, self.W2, breakpoints=self.breakpoints, bounds=self.bounds
            )
            relaxation = Sos2TrainingRelaxation(model)
        else:
            model = build_sosx_mae_training_model(
                self.X, self.y, self.W2, breakpoints=self.breakpoints, bounds=self.bounds
            )
            relaxation = SosxTrainingRelaxation(model)
        solver = get_milp_solver(self.solver_name, options=self.solver_options or None)
        track = track_training_variables(model)
        return relaxation, solver, self.persistent, track


# ---------------------------------------------------------------------------
# Worker body (decorated with ray.remote lazily so importing never requires ray)
# ---------------------------------------------------------------------------

class _SosxNodeWorker:
    """A long-lived Ray actor holding one rebuilt relaxation + solver.

    Created once from a *spec*; thereafter each :meth:`solve` call applies
    a node's breakpoint windows to the worker's own Pyomo model and solves
    the resulting LP. Because the model is reused across calls, a
    persistent solver only pushes the changed lambda fixings between
    solves.

    Violation detection and branch-set selection happen in-process so only
    the minimal payload (see :class:`_WorkerResult`) is returned to the
    driver.
    """

    def __init__(
        self,
        spec: Any,
        branching_rule: str = "first",
        violation_tol: float = 1e-6,
    ) -> None:
        import pyomo.environ as pyo

        self._pyo = pyo
        self._branching_rule = branching_rule
        self._violation_tol = float(violation_tol)
        self._formulation = getattr(spec, "formulation", "sosx")
        self._relaxation, self._solver, self._persistent, self._track = (
            spec.build_worker()
        )
        # u_values only present for Y-formulation; None for SOS2 λ-formulation.
        self._u_values = getattr(self._relaxation, "u_values", None)

    def _select_branch(
        self, lam: dict, windows: dict
    ) -> tuple[SosKey, int] | None:
        """Find violations and return ``(branch_key, branch_k)`` or ``None``.

        For Y-formulation: ``branch_k`` is the 1-indexed violating segment.
        For SOS2 formulation: ``branch_k`` is the weighted-mean split index
        computed here so the driver needs no λ values to perform the split.
        """
        if self._formulation == "sos2":
            violations = find_violations(lam, windows, tol=self._violation_tol)
            if not violations:
                return None
            key, _ = select_violation(violations, self._branching_rule)
            lo, hi = windows[key]
            weights = lam[key][lo : hi + 1]
            idx = np.arange(lo, hi + 1)
            total = float(weights.sum())
            mean = float((weights * idx).sum() / total) if total > 0 else 0.5 * (lo + hi)
            r = max(lo + 1, min(hi - 1, int(round(mean))))
            return key, r
        else:
            violations = find_violations_Y(lam, self._u_values, windows, tol=self._violation_tol)
            if not violations:
                return None
            return select_violation(violations, self._branching_rule)

    def solve(self, windows: dict[SosKey, Window]) -> _WorkerResult:
        """Solve one node and return a compact :class:`_WorkerResult`.

        * infeasible → ``feasible=False``, all payload fields are ``None``;
        * SOS2-violating → ``has_violation=True`` with *branch_key* and
          *branch_k* for the chosen set;
        * SOS2-feasible → ``has_violation=False`` with full *lam* and *values*.
        """
        self._relaxation.apply_windows(windows)
        solution = self._relaxation.solve(self._solver, persistent=self._persistent)
        if not solution.feasible:
            return _WorkerResult(
                feasible=False,
                objective=solution.objective,
                solve_seconds=solution.solve_seconds,
                has_violation=False,
                branch_key=None,
                branch_k=None,
                lam=None,
                values=None,
            )

        branch = self._select_branch(solution.lam, windows)
        if branch is not None:
            branch_key, branch_k = branch
            return _WorkerResult(
                feasible=True,
                objective=solution.objective,
                solve_seconds=solution.solve_seconds,
                has_violation=True,
                branch_key=branch_key,
                branch_k=branch_k,
                lam=None,
                values=None,
            )

        values = {
            name: float(self._pyo.value(target)) for name, target in self._track.items()
        }
        return _WorkerResult(
            feasible=True,
            objective=solution.objective,
            solve_seconds=solution.solve_seconds,
            has_violation=False,
            branch_key=None,
            branch_k=None,
            lam=solution.lam,
            values=values,
        )


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------

class ParallelSosxBrancher:
    """Best-bound SOSX branch-and-bound that fans node solves out over Ray.

    Parameters
    ----------
    spec : SosxTrainingSpec or any compatible spec
        The picklable relaxation build spec shared with every worker.
        Must expose ``root_windows()``, ``build_worker()``, and
        ``persistent`` (see module docstring for the full protocol).
    sense : {"minimize", "maximize"}, optional
        Objective sense. Default ``"minimize"`` (training minimises MAE).
    n_workers : int or None, optional
        Number of Ray actors. ``None`` uses :func:`default_worker_count`.
    node_selection, branching_rule, max_nodes, tol, violation_tol,
    time_limit, absolute_gap, relative_gap, verbose
        Same meaning as on :class:`pwlnet.branching.SosxBrancher`.
    max_retries : int, optional
        How many times to re-queue a node whose Ray task raises before
        giving up and skipping it. Default ``2``.
    shutdown_ray : bool, optional
        When ``True`` *and* this instance started Ray itself, call
        ``ray.shutdown()`` after the solve completes. Default ``False``
        (Ray is left running so repeated fits reuse the cluster).
    initial_incumbent_lam : dict or None, optional
        SOS-set convex-combination weights from a previous incumbent /
        warm-start network. Revalidated on a worker before the search.
    ray_init_kwargs : dict or None, optional
        Extra keyword arguments for ``ray.init`` (only used when this
        brancher starts Ray itself).
    """

    def __init__(
        self,
        spec: Any,
        *,
        sense: str = "minimize",
        n_workers: int | None = None,
        node_selection: str = "best_bound",
        branching_rule: str = "first",
        max_nodes: int = 2000,
        tol: float = 1e-7,
        violation_tol: float = 1e-6,
        time_limit: float | None = None,
        absolute_gap: float | None = None,
        relative_gap: float | None = None,
        max_retries: int = 2,
        shutdown_ray: bool = False,
        initial_incumbent_lam: dict[SosKey, np.ndarray] | None = None,
        ray_init_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        self.maximize = _resolve_sense(sense)
        self.n_workers = (
            default_worker_count() if n_workers is None else max(1, int(n_workers))
        )
        self.node_selection = node_selection
        self.branching_rule = branching_rule
        self.max_nodes = int(max_nodes)
        self.tol = float(tol)
        self.violation_tol = float(violation_tol)
        self.time_limit = _optional_nonneg(time_limit, "time_limit", positive=True)
        self.absolute_gap = _optional_nonneg(absolute_gap, "absolute_gap")
        self.relative_gap = _optional_nonneg(relative_gap, "relative_gap")
        self.max_retries = max(0, int(max_retries))
        self.shutdown_ray = bool(shutdown_ray)
        self.initial_incumbent_lam = initial_incumbent_lam
        self.ray_init_kwargs = dict(ray_init_kwargs or {})
        self.log = BranchLog(verbose=verbose)
        self._next_seq = 1
        self._ray_owned = False

    # -- public entry point -------------------------------------------------

    def solve(self) -> SosxResult:
        """Run the parallel branch-and-bound loop and return the incumbent."""
        ray = _import_ray()
        spec = self.spec
        incumbent = Incumbent(maximize=self.maximize)
        frontier = Frontier(self.node_selection, maximize=self.maximize)
        root_bound = np.inf if self.maximize else -np.inf
        frontier.push(Node(bound=root_bound, seq=0, windows=spec.root_windows()))

        n_nodes = 0
        solver_seconds = 0.0
        start = time.perf_counter()
        stop_status: str | None = None
        self._next_seq = 1

        self._ray_owned = self._ensure_ray(ray)
        workers = self._spawn_workers(ray, spec)
        try:
            warm = self._revalidate_initial_incumbent(ray, workers[0], spec.root_windows())
            warm_started = warm is not None
            if warm is not None:
                objective, values, lam, secs = warm
                solver_seconds += secs
                n_nodes += 1
                incumbent.consider(
                    objective, values, lam, depth=0, seq=-1, tol=self.tol
                )
                self.log.node(-1, 0, "incumbent", objective)

            # ref -> (node, worker_index)
            inflight: dict[Any, tuple[Node, int]] = {}
            free: list[int] = list(range(self.n_workers))
            # node.seq -> retry count for failed tasks
            retry_counts: dict[int, int] = {}

            def dispatch() -> None:
                """Assign best open nodes to free workers, skipping fathomable ones."""
                while free and frontier:
                    if n_nodes + len(inflight) >= self.max_nodes:
                        return
                    node = frontier.pop()
                    if can_fathom_by_bound(
                        node.bound, incumbent.objective,
                        maximize=self.maximize, tol=self.tol,
                    ):
                        continue
                    worker_idx = free.pop()
                    ref = workers[worker_idx].solve.remote(node.windows)
                    inflight[ref] = (node, worker_idx)

            dispatch()
            while inflight:
                done, _ = ray.wait(list(inflight.keys()), num_returns=1)
                ref = done[0]
                node, worker_idx = inflight.pop(ref)
                free.append(worker_idx)

                try:
                    result: _WorkerResult = ray.get(ref)
                except Exception:
                    retries = retry_counts.get(node.seq, 0)
                    if retries < self.max_retries:
                        retry_counts[node.seq] = retries + 1
                        frontier.push(node)
                        self.log.node(node.seq, node.depth, "retried", None)
                    else:
                        self.log.node(node.seq, node.depth, "failed", None)
                    if stop_status is None:
                        dispatch()
                    continue

                solver_seconds += result.solve_seconds
                n_nodes += 1

                self._process_node(node, result, incumbent, frontier)

                if stop_status is None:
                    stop_status = self._stop_reason(
                        n_nodes, start, frontier, incumbent, inflight
                    )
                if stop_status is None:
                    dispatch()
                # When stop_status is set: drain already-running solves
                # (they may still improve the incumbent) but dispatch nothing new.

            wall_seconds = time.perf_counter() - start
            status = self._status(incumbent, stop_status)
            best_bound, abs_gap, rel_gap = self._gap(frontier, incumbent, inflight)
            return SosxResult(
                status=status,
                objective=incumbent.objective,
                values=incumbent.values,
                n_nodes=n_nodes,
                solver_seconds=solver_seconds,
                wall_seconds=wall_seconds,
                best_bound=best_bound,
                absolute_gap=abs_gap,
                relative_gap=rel_gap,
                incumbent_depth=incumbent.depth,
                lam=incumbent.lam,
                persistent=getattr(spec, "persistent", False),
                warm_started=warm_started,
            )
        finally:
            self._kill_workers(ray, workers)
            if self._ray_owned and self.shutdown_ray:
                try:
                    ray.shutdown()
                except Exception:
                    pass

    # -- worker lifecycle ---------------------------------------------------

    def _spawn_workers(self, ray: Any, spec: Any) -> list[Any]:
        WorkerActor = ray.remote(num_cpus=1)(_SosxNodeWorker)
        return [
            WorkerActor.remote(spec, self.branching_rule, self.violation_tol)
            for _ in range(self.n_workers)
        ]

    def _kill_workers(self, ray: Any, workers: list[Any]) -> None:
        """Release each actor's solver/model process (best-effort)."""
        for actor in workers:
            try:
                ray.kill(actor)
            except Exception:
                pass

    # -- node processing (mirrors the serial SosxBrancher) ------------------

    def _process_node(
        self,
        node: Node,
        result: _WorkerResult,
        incumbent: Incumbent,
        frontier: Frontier,
    ) -> None:
        """Apply one worker outcome to the incumbent and frontier."""
        if not result.feasible:
            self.log.node(node.seq, node.depth, "infeasible", None)
            return

        # The node's own optimum bounds its whole subtree.
        if not is_improving(
            result.objective, incumbent.objective,
            maximize=self.maximize, tol=self.tol,
        ):
            self.log.node(node.seq, node.depth, "fathomed", result.objective)
            return

        if not result.has_violation:
            incumbent.consider(
                result.objective,
                result.values or {},
                result.lam,
                depth=node.depth,
                seq=node.seq,
                tol=self.tol,
            )
            self.log.node(node.seq, node.depth, "incumbent", result.objective)
            return

        self.log.node(node.seq, node.depth, "feasible", result.objective)
        # Worker computed branch_key and branch_k; driver splits accordingly.
        formulation = getattr(self.spec, "formulation", "sosx")
        if formulation == "sos2":
            # branch_k is the weighted-mean split index computed in the worker.
            children = split_window_at(node.windows, result.branch_key, result.branch_k)
        else:
            children = split_window_Y(node.windows, (result.branch_key, result.branch_k))
        for child_windows in children:
            frontier.push(
                Node(
                    bound=result.objective,
                    seq=self._next_seq,
                    windows=child_windows,
                    depth=node.depth + 1,
                    parent=node.seq,
                )
            )
            self._next_seq += 1

    # -- warm start ---------------------------------------------------------

    def _revalidate_initial_incumbent(
        self,
        ray: Any,
        worker: Any,
        root_windows: dict[SosKey, Window],
    ) -> tuple[float, dict[str, float], dict[SosKey, np.ndarray], float] | None:
        """Solve the previous incumbent's tight windows on one worker.

        Y-formulation (sosx): pin each SOS set to window ``(k*, k*)`` where
        k* is the first segment not fully filled (Y[k] < u[k]).

        λ-formulation (sos2): pin each SOS set to window ``(left, right)``
        derived from the two-adjacent breakpoint support of λ.
        """
        val0 = self.initial_incumbent_lam
        if not isinstance(val0, dict):
            return None
        windows = dict(root_windows)
        if set(val0) != set(windows):
            return None

        formulation = getattr(self.spec, "formulation", "sosx")

        if formulation == "sos2":
            for key, raw in val0.items():
                lam = np.asarray(raw, dtype=float).ravel()
                support = np.flatnonzero(lam > self.violation_tol)
                if support.size == 0:
                    return None
                if support.size > 2 or (
                    support.size == 2 and int(support[1]) != int(support[0]) + 1
                ):
                    return None
                windows[key] = (int(support[0]), int(support[-1]))
        else:
            # Derive segment widths from spec (avoids building model on driver).
            pwl = build_tanh_pwl_data(self.spec.breakpoints)
            bp = pwl.breakpoints
            u = np.array([bp[k] - bp[k - 1] for k in range(1, len(bp))], dtype=float)
            for key, raw in val0.items():
                Y = np.asarray(raw, dtype=float).ravel()
                P = windows[key][1]  # P from root window (1, P)
                if Y.size != P:
                    return None
                k_star = P
                for k0 in range(P):
                    if Y[k0] < u[k0] - self.violation_tol:
                        k_star = k0 + 1  # convert to 1-indexed
                        break
                windows[key] = (k_star, k_star)

        try:
            result: _WorkerResult = ray.get(worker.solve.remote(windows))
        except Exception:
            return None
        if not result.feasible or result.has_violation:
            return None
        return result.objective, result.values, result.lam, result.solve_seconds

    # -- gap / status helpers -----------------------------------------------

    def _best_open_bound(
        self,
        frontier: Frontier,
        inflight: dict[Any, tuple[Node, int]],
    ) -> float | None:
        """Strongest objective bound among open *and* in-flight nodes.

        In-flight nodes have been popped from the frontier so a gap
        computation that ignored them could declare optimality early.
        Each carries its parent's relaxed objective as a valid subtree bound.
        """
        bounds: list[float] = []
        frontier_bound = frontier.best_bound()
        if frontier_bound is not None:
            bounds.append(float(frontier_bound))
        for node, _ in inflight.values():
            bounds.append(float(node.bound))
        if not bounds:
            return None
        return max(bounds) if self.maximize else min(bounds)

    def _gap(
        self,
        frontier: Frontier,
        incumbent: Incumbent,
        inflight: dict[Any, tuple[Node, int]],
    ) -> tuple[float | None, float | None, float | None]:
        if not incumbent.found:
            return self._best_open_bound(frontier, inflight), None, None
        best_bound = self._best_open_bound(frontier, inflight)
        if best_bound is None:
            best_bound = incumbent.objective
        if self.maximize:
            absolute = max(0.0, float(best_bound) - float(incumbent.objective))
        else:
            absolute = max(0.0, float(incumbent.objective) - float(best_bound))
        relative = absolute / max(abs(float(incumbent.objective)), 1e-12)
        return float(best_bound), absolute, relative

    def _gap_reached(
        self,
        frontier: Frontier,
        incumbent: Incumbent,
        inflight: dict[Any, tuple[Node, int]],
    ) -> bool:
        if self.absolute_gap is None and self.relative_gap is None:
            return False
        _, absolute, relative = self._gap(frontier, incumbent, inflight)
        if absolute is None:
            return False
        return (
            self.absolute_gap is not None and absolute <= self.absolute_gap
        ) or (
            self.relative_gap is not None and relative <= self.relative_gap
        )

    def _stop_reason(
        self,
        n_nodes: int,
        start: float,
        frontier: Frontier,
        incumbent: Incumbent,
        inflight: dict[Any, tuple[Node, int]],
    ) -> str | None:
        if n_nodes >= self.max_nodes:
            return "node_limit"
        if self.time_limit is not None and time.perf_counter() - start >= self.time_limit:
            return "time_limit"
        if self._gap_reached(frontier, incumbent, inflight):
            return "gap_limit"
        return None

    def _status(self, incumbent: Incumbent, stop_status: str | None) -> str:
        if stop_status is not None:
            return stop_status
        if not incumbent.found:
            return "infeasible"
        return "optimal"

    # -- Ray lifecycle ------------------------------------------------------

    def _ensure_ray(self, ray: Any) -> bool:
        """Start Ray if not yet running; return whether *this call* started it.

        Ray is left running after the solve so repeated fits in one
        process can reuse the cluster (startup costs several seconds).
        Pass ``shutdown_ray=True`` to the constructor for a clean teardown.
        """
        if ray.is_initialized():
            return False
        kwargs: dict[str, Any] = dict(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            configure_logging=False,
        )
        # Give Ray enough logical CPUs for every actor plus the driver.
        kwargs["num_cpus"] = max(self.n_workers + 1, os.cpu_count() or 1)
        kwargs.update(self.ray_init_kwargs)
        ray.init(**kwargs)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_ray() -> Any:
    try:
        import ray
    except ImportError as exc:
        raise TrainingError(
            "Parallel SOSX requires the optional 'ray' dependency. "
            "Install it with `pip install ray` or `pip install pwlnet[parallel]`."
        ) from exc
    return ray


def _resolve_sense(sense: str) -> bool:
    name = str(sense).strip().lower()
    if name in _MAXIMIZE_NAMES:
        return True
    if name in _MINIMIZE_NAMES:
        return False
    raise TrainingError(f"sense must be 'maximize' or 'minimize', got {sense!r}.")


def _optional_nonneg(
    value: float | None,
    name: str,
    *,
    positive: bool = False,
) -> float | None:
    if value is None:
        return None
    number = float(value)
    valid = number > 0.0 if positive else number >= 0.0
    if not np.isfinite(number) or not valid:
        qualifier = "positive" if positive else "nonnegative"
        raise TrainingError(f"{name} must be a finite {qualifier} number or None.")
    return number
