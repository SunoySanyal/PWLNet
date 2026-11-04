"""The SOSX branch-and-bound driver.

:class:`SosxBrancher` runs the loop sketched in the architecture
document (section 5): solve a node relaxation, fathom by bound, detect
the SOS2 violation, branch on the offending SOS set's breakpoint window,
and keep the best SOS2-feasible incumbent. The public solver entry point
lives in :mod:`pwlnet.solvers.sosx_solver`.

The brancher is parametrized by ``violation_fn`` and ``split_fn`` so the
same driver works for both:

- **Downstream optimization** (``SosxRelaxation``): λ-based violation
  detection (:func:`~pwlnet.formulations.sosx.find_violations`) and
  symmetric window bisection (:func:`~pwlnet.formulations.sosx.split_window`).

- **SOSX-MAE training** (``SosxTrainingRelaxation``): Y-variable violation
  detection (:func:`~pwlnet.formulations.sosx.find_violations_Y`) and the
  paper's asymmetric branching (:func:`~pwlnet.formulations.sosx.split_window_Y`).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from pwlnet.branching.branch_log import BranchLog
from pwlnet.branching.fathoming import can_fathom_by_bound, is_improving
from pwlnet.branching.incumbent import Incumbent
from pwlnet.branching.node import Node, SosKey, Window
from pwlnet.branching.node_selection import Frontier
from pwlnet.formulations.sosx.sosx_branching_rules import select_violation, split_window
from pwlnet.formulations.sosx.sosx_node import SosxRelaxation
from pwlnet.formulations.sosx.sosx_solution import SosxResult
from pwlnet.formulations.sosx.sosx_violation import find_violations
from pwlnet.utils.exceptions import FormulationError

__all__ = ["SosxBrancher"]

_MAXIMIZE_NAMES = ("max", "maximize")
_MINIMIZE_NAMES = ("min", "minimize")

# Violation function signature: (sos_values, windows, tol) → list[(SosKey, Any)]
ViolationFn = Callable[[dict, dict, float], list[tuple[SosKey, Any]]]
# Split function signature: (windows, violation) → (child_windows, child_windows)
SplitFn = Callable[[dict, Any], tuple[dict, dict]]


def _default_violation_fn(
    sos_values: dict[SosKey, np.ndarray],
    windows: dict[SosKey, Window],
    tol: float,
) -> list[tuple[SosKey, int]]:
    """λ-based violation detection (downstream optimization default)."""
    return find_violations(sos_values, windows, tol=tol)


def _default_split_fn(
    windows: dict[SosKey, Window],
    violation: tuple[SosKey, Any],
) -> tuple[dict, dict]:
    """λ-based symmetric window bisection (downstream optimization default).

    violation is (SosKey, spread); only the key is needed for split_window.
    """
    key = violation[0]
    # split_window needs the full lam dict; the violation tuple carries the
    # key but not lam. This is handled by wrapping in the brancher where lam
    # is in scope. When using the default path the brancher passes lam directly
    # via _split_with_lam which is set up in solve().
    raise NotImplementedError("Use _make_lam_split_fn to bind lam before calling.")


class SosxBrancher:
    """Branch-and-bound over the continuous SOS2 weights of a relaxation.

    Parameters
    ----------
    relaxation : SosxRelaxation
        The SOS2-dropped, reusable node relaxation.
    solver : object
        A Pyomo solver object (``solve(model)``), e.g. from
        :func:`pwlnet.solvers.get_milp_solver`.
    sense : {"maximize", "minimize"}, optional
        Must match the model's objective sense.
    track : dict[str, Any] or None, optional
        Model variables/expressions to snapshot at the incumbent,
        keyed by a readable name (e.g. ``{"FB": model.FB}``).
    node_selection : {"best_bound", "depth_first"}, optional
        Open-node strategy (see :class:`~pwlnet.branching.node_selection.Frontier`).
    branching_rule : {"first", "most_violated"}, optional
        How to choose the SOS set to split.
    max_nodes : int, optional
        Safety cap on solved nodes.
    tol : float, optional
        Objective tolerance for incumbent updates and bound fathoming.
    violation_tol : float, optional
        Weight threshold below which a lambda/Y value counts as zero.
    time_limit : float or None, optional
        Brancher wall-clock limit in seconds, checked between node solves.
    absolute_gap, relative_gap : float or None, optional
        Incumbent-to-best-open-bound stopping tolerances.
    persistent : bool, optional
        Reuse an APPSI solver instance and update only variables whose
        fixed state changed.
    violation_fn : callable or None, optional
        ``f(sos_values, windows, tol) -> list[(SosKey, Any)]``. Defaults to
        λ-based :func:`~pwlnet.formulations.sosx.find_violations` (for
        downstream optimization). Pass
        :func:`~pwlnet.formulations.sosx.find_violations_Y`-based wrapper
        for SOSX-MAE training.
    split_fn : callable or None, optional
        ``f(windows, violation) -> (child1, child2)``. Defaults to λ-based
        symmetric bisection (for downstream). Pass
        :func:`~pwlnet.formulations.sosx.split_window_Y` for training.
    verbose : bool, optional
        Log brancher incumbent/infeasible events.
    """

    def __init__(
        self,
        relaxation: SosxRelaxation,
        solver: Any,
        *,
        sense: str = "maximize",
        track: dict[str, Any] | None = None,
        node_selection: str = "best_bound",
        branching_rule: str = "first",
        max_nodes: int = 2000,
        tol: float = 1e-7,
        violation_tol: float = 1e-6,
        time_limit: float | None = None,
        absolute_gap: float | None = None,
        relative_gap: float | None = None,
        persistent: bool = False,
        initial_incumbent: SosxResult | dict[Any, Any] | None = None,
        violation_fn: ViolationFn | None = None,
        split_fn: SplitFn | None = None,
        verbose: bool = False,
    ):
        self.relaxation = relaxation
        self.solver = solver
        self.maximize = _resolve_sense(sense)
        self.track = dict(track or {})
        self.node_selection = node_selection
        self.branching_rule = branching_rule
        self.max_nodes = int(max_nodes)
        self.tol = float(tol)
        self.violation_tol = float(violation_tol)
        self.time_limit = _optional_nonnegative(time_limit, "time_limit", positive=True)
        self.absolute_gap = _optional_nonnegative(absolute_gap, "absolute_gap")
        self.relative_gap = _optional_nonnegative(relative_gap, "relative_gap")
        self.persistent = bool(persistent)
        self.initial_incumbent = initial_incumbent
        self._violation_fn = violation_fn   # None → use λ-based default in solve()
        self._split_fn = split_fn           # None → use λ-based split_window in solve()
        self.log = BranchLog(verbose=verbose)

    def _snapshot(self) -> dict[str, float]:
        import pyomo.environ as pyo

        return {name: float(pyo.value(target)) for name, target in self.track.items()}

    def solve(self) -> SosxResult:
        """Run the branch-and-bound loop and return the incumbent report."""
        relaxation = self.relaxation
        incumbent = Incumbent(maximize=self.maximize)
        frontier = Frontier(self.node_selection, maximize=self.maximize)
        root_bound = np.inf if self.maximize else -np.inf
        frontier.push(Node(bound=root_bound, seq=0, windows=relaxation.root_windows()))

        next_seq = 1
        n_nodes = 0
        solver_seconds = 0.0
        start = time.perf_counter()
        stop_status: str | None = None

        warm_solution = self._revalidate_initial_incumbent(relaxation)
        warm_started = warm_solution is not None
        if warm_solution is not None:
            solver_seconds += warm_solution.solve_seconds
            n_nodes += 1
            incumbent.consider(
                warm_solution.objective,
                self._snapshot(),
                warm_solution.lam,
                depth=0,
                seq=-1,
                tol=self.tol,
            )
            self.log.node(-1, 0, "incumbent", warm_solution.objective)

        while frontier:
            if n_nodes >= self.max_nodes:
                stop_status = "node_limit"
                break
            if self.time_limit is not None and time.perf_counter() - start >= self.time_limit:
                stop_status = "time_limit"
                break
            if self._gap_reached(frontier, incumbent):
                stop_status = "gap_limit"
                break

            node = frontier.pop()
            if can_fathom_by_bound(
                node.bound, incumbent.objective, maximize=self.maximize, tol=self.tol
            ):
                continue

            relaxation.apply_windows(node.windows)
            solution = relaxation.solve(self.solver, persistent=self.persistent)
            solver_seconds += solution.solve_seconds
            n_nodes += 1

            if not solution.feasible:
                self.log.node(node.seq, node.depth, "infeasible", None)
                continue

            if not is_improving(
                solution.objective, incumbent.objective, maximize=self.maximize, tol=self.tol
            ):
                self.log.node(node.seq, node.depth, "fathomed", solution.objective)
                continue

            violations = self._find_violations(solution.lam, node.windows)
            if not violations:
                incumbent.consider(
                    solution.objective,
                    self._snapshot(),
                    solution.lam,
                    depth=node.depth,
                    seq=node.seq,
                    tol=self.tol,
                )
                self.log.node(node.seq, node.depth, "incumbent", solution.objective)
                continue

            self.log.node(node.seq, node.depth, "feasible", solution.objective)
            violation = select_violation(violations, self.branching_rule)
            for child_windows in self._split(node.windows, violation, solution.lam):
                frontier.push(
                    Node(
                        bound=solution.objective,
                        seq=next_seq,
                        windows=child_windows,
                        depth=node.depth + 1,
                        parent=node.seq,
                    )
                )
                next_seq += 1

        wall_seconds = time.perf_counter() - start
        status = self._status(frontier, incumbent, stop_status)
        best_bound, abs_gap, rel_gap = self._gap(frontier, incumbent)
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
            persistent=self.persistent,
            warm_started=warm_started,
        )

    # ------------------------------------------------------------------
    # Violation / split dispatch
    # ------------------------------------------------------------------

    def _find_violations(
        self,
        sos_values: dict[SosKey, np.ndarray],
        windows: dict[SosKey, Window],
    ) -> list[tuple[SosKey, Any]]:
        if self._violation_fn is not None:
            return self._violation_fn(sos_values, windows, self.violation_tol)
        # Default: λ-based.
        return find_violations(sos_values, windows, tol=self.violation_tol)

    def _split(
        self,
        windows: dict[SosKey, Window],
        violation: tuple[SosKey, Any],
        lam: dict[SosKey, np.ndarray],
    ) -> tuple[dict, dict]:
        if self._split_fn is not None:
            # Y-based: violation = (SosKey, k); split_fn handles unpacking.
            return self._split_fn(windows, violation)
        # Default λ-based: violation = (SosKey, spread); only the key is needed.
        key = violation[0]
        return split_window(windows, key, lam)

    # ------------------------------------------------------------------
    # Warm-start revalidation
    # ------------------------------------------------------------------

    def _revalidate_initial_incumbent(
        self, relaxation: SosxRelaxation
    ) -> Any | None:
        """Solve the previous incumbent's windows on the current model.

        For λ-based incumbents: reads 2-adjacent support from λ arrays.
        For Y-based incumbents: finds the active segment k* per SOS set
        (first k where Y[k] < u[k]) and sets window (k*, k*).
        """
        if self.initial_incumbent is None:
            return None
        sos_values = getattr(self.initial_incumbent, "lam", self.initial_incumbent)
        if not isinstance(sos_values, dict):
            return None

        windows = relaxation.root_windows()
        if set(sos_values) != set(windows):
            return None

        # Detect whether we have Y values (training) or λ values (downstream)
        # by checking if the relaxation exposes u_values.
        u_values = getattr(relaxation, "u_values", None)

        if u_values is not None:
            # Y-based: find active segment k* for each SOS set.
            for key, Y in sos_values.items():
                Y = np.asarray(Y, dtype=float).ravel()
                u = u_values[key]
                P = len(u)
                if Y.size != P:
                    return None
                # Find first segment that is not fully filled → k*.
                k_star = P  # default: all full → k* = P
                for k0 in range(P):
                    if Y[k0] < u[k0] - self.violation_tol:
                        k_star = k0 + 1  # convert to 1-indexed
                        break
                k_star = max(1, min(k_star, P))
                windows[key] = (k_star, k_star)
        else:
            # λ-based: build tight windows from 2-adjacent support.
            for key, values in sos_values.items():
                weights = np.asarray(values, dtype=float).ravel()
                if weights.size != windows[key][1] + 1:
                    return None
                support = np.flatnonzero(weights > self.violation_tol)
                if support.size == 0:
                    return None
                if support.size > 2 or (
                    support.size == 2 and int(support[1]) != int(support[0]) + 1
                ):
                    return None
                windows[key] = (int(support[0]), int(support[-1]))

        relaxation.apply_windows(windows)
        solution = relaxation.solve(self.solver, persistent=self.persistent)
        if not solution.feasible:
            return None
        violations = self._find_violations(solution.lam, windows)
        return None if violations else solution

    # ------------------------------------------------------------------
    # Gap / status helpers
    # ------------------------------------------------------------------

    def _gap(
        self, frontier: Frontier, incumbent: Incumbent
    ) -> tuple[float | None, float | None, float | None]:
        if not incumbent.found:
            return frontier.best_bound(), None, None
        best_bound = frontier.best_bound()
        if best_bound is None:
            best_bound = incumbent.objective
        if self.maximize:
            absolute = max(0.0, float(best_bound) - float(incumbent.objective))
        else:
            absolute = max(0.0, float(incumbent.objective) - float(best_bound))
        relative = absolute / max(abs(float(incumbent.objective)), 1e-12)
        return float(best_bound), absolute, relative

    def _gap_reached(self, frontier: Frontier, incumbent: Incumbent) -> bool:
        if self.absolute_gap is None and self.relative_gap is None:
            return False
        _, absolute, relative = self._gap(frontier, incumbent)
        if absolute is None:
            return False
        return (
            self.absolute_gap is not None and absolute <= self.absolute_gap
        ) or (
            self.relative_gap is not None and relative <= self.relative_gap
        )

    def _status(
        self, frontier: Frontier, incumbent: Incumbent, stop_status: str | None
    ) -> str:
        if stop_status is not None:
            return stop_status
        if not incumbent.found:
            return "infeasible"
        return "optimal"


def _resolve_sense(sense: str) -> bool:
    name = str(sense).strip().lower()
    if name in _MAXIMIZE_NAMES:
        return True
    if name in _MINIMIZE_NAMES:
        return False
    raise FormulationError(
        f"sense must be 'maximize' or 'minimize', got {sense!r}."
    )


def _optional_nonnegative(
    value: float | None, name: str, *, positive: bool = False
) -> float | None:
    if value is None:
        return None
    number = float(value)
    valid = number > 0.0 if positive else number >= 0.0
    if not np.isfinite(number) or not valid:
        qualifier = "positive" if positive else "nonnegative"
        raise FormulationError(f"{name} must be a finite {qualifier} number or None.")
    return number
