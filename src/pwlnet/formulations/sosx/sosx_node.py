"""Build and re-solve SOSX node relaxations over an embedded tanh surrogate.

A :class:`SosxRelaxation` wraps a Pyomo model that already contains one
or more ``tanh_pwl_sos2`` surrogate blocks (see
:func:`pwlnet.formulations.pyomo_pwl.build_tanh_pwl_sos2_block`). On
construction it *drops* every SOS2 constraint, turning the model into a
continuous relaxation in the convex-combination weights ``lambda``. Each
branch-and-bound node is then realised by fixing the ``lambda`` weights
outside an allowed breakpoint window to zero -- no rebuild required, so
the same Pyomo model is reused for every node (the Phase-6 "reusable
model" strategy from the architecture's section 10).

Because the surrogate weights are constants, every pre-activation
``W.h + b`` stays linear at any depth, so a node is an LP (or a MILP when
the surrounding objective adds binaries) and is solvable by the default
open-source HiGHS backend -- the custom brancher, not the solver,
enforces the SOS2 segment structure.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from pwlnet.branching.node import SosKey, Window
from pwlnet.formulations.sosx.sosx_solution import NodeSolution
from pwlnet.utils.exceptions import FormulationError

__all__ = ["SosxRelaxation", "discover_sos2_blocks"]

_SOS2_NAME = re.compile(r"^layer_(\d+)_sos2$")
_OK_TERMINATIONS = ("optimal", "feasible")
_BAD_TERMINATIONS = ("infeasible", "unbounded", "error", "nosolution", "unknown", "interrupted")


def discover_sos2_blocks(model: Any) -> dict[str, Any]:
    """Find every block on ``model`` that carries SOS2 tanh activations.

    A block qualifies when it has at least one component named
    ``layer_<i>_sos2`` (the marker added by the SOS2 PWL-tanh builder).
    Returns a mapping of block name to block object.
    """
    import pyomo.environ as pyo

    found: dict[str, Any] = {}
    for block in model.component_objects(pyo.Block, descend_into=True):
        if any(_SOS2_NAME.match(comp.local_name) for comp in block.component_objects(pyo.SOSConstraint)):
            found[block.name] = block
    return found


def _tanh_layer_indices(block: Any) -> list[int]:
    """Indices ``i`` of tanh layers on ``block`` (those with an SOS2 set)."""
    import pyomo.environ as pyo

    indices: list[int] = []
    for comp in block.component_objects(pyo.SOSConstraint):
        match = _SOS2_NAME.match(comp.local_name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


class SosxRelaxation:
    """A reusable SOS2-dropped relaxation of an embedded tanh surrogate.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model containing one or more ``tanh_pwl_sos2`` surrogate blocks
        and exactly one active objective.
    blocks : sequence or None, optional
        Block objects or block names to treat as SOSX blocks. ``None``
        auto-discovers them with :func:`discover_sos2_blocks`.

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If no SOS2 surrogate block is found, or a named block has no
        SOS2 activations.
    """

    def __init__(self, model: Any, blocks: Any | None = None):
        import pyomo.environ as pyo

        self._pyo = pyo
        self.model = model

        resolved = self._resolve_blocks(model, blocks)
        if not resolved:
            raise FormulationError(
                "SosxRelaxation found no tanh_pwl_sos2 surrogate block on the "
                "model. Build one with formulation='tanh_pwl_sos2' first."
            )
        self.blocks = resolved
        self._changed_vars: list[Any] = []
        self._persistent_solver_id: int | None = None

        # Per SOS set: the ordered list of lambda VarData and the
        # breakpoint count K. Dropping SOS2 here makes the model a pure
        # continuous relaxation; the brancher re-imposes structure via
        # apply_windows.
        self.lambda_vars: dict[SosKey, list[Any]] = {}
        self._breakpoints: dict[SosKey, int] = {}
        for name, block in self.blocks.items():
            for layer_idx in _tanh_layer_indices(block):
                getattr(block, f"layer_{layer_idx}_sos2").deactivate()
                lam = getattr(block, f"layer_{layer_idx}_lambda")
                neurons = sorted({idx[0] for idx in lam})
                breakpoints = sorted({idx[1] for idx in lam})
                k = len(breakpoints)
                for j in neurons:
                    key: SosKey = (name, layer_idx, int(j))
                    self.lambda_vars[key] = [lam[j, b] for b in breakpoints]
                    self._breakpoints[key] = k

    def _resolve_blocks(self, model: Any, blocks: Any | None) -> dict[str, Any]:
        if blocks is None:
            return discover_sos2_blocks(model)
        resolved: dict[str, Any] = {}
        for item in blocks:
            block = model.component(item) if isinstance(item, str) else item
            if block is None:
                raise FormulationError(f"Model has no block named '{item}'.")
            if not _tanh_layer_indices(block):
                raise FormulationError(
                    f"Block '{getattr(block, 'name', item)}' has no SOS2 tanh "
                    "activations; it is not a SOSX block."
                )
            resolved[block.name] = block
        return resolved

    def sos_metadata(self) -> dict[SosKey, int]:
        """Breakpoint count ``K`` for every SOS set."""
        return dict(self._breakpoints)

    def root_windows(self) -> dict[SosKey, Window]:
        """The unrestricted ``(0, K-1)`` window for every SOS set (the root)."""
        return {key: (0, k - 1) for key, k in self._breakpoints.items()}

    def apply_windows(self, windows: dict[SosKey, Window]) -> list[Any]:
        """Fix lambdas outside each set's window to zero; unfix inside.

        Returns the list of VarData whose fixed-state changed, so a
        persistent solver interface can push just those updates.
        """
        changed: list[Any] = []
        for key, vars_ in self.lambda_vars.items():
            lo, hi = windows[key]
            for index, var in enumerate(vars_):
                should_fix = index < lo or index > hi
                if should_fix and not var.fixed:
                    var.fix(0.0)
                    changed.append(var)
                elif not should_fix and var.fixed:
                    var.unfix()
                    changed.append(var)
        self._changed_vars = changed
        return changed

    def read_lambdas(self) -> dict[SosKey, np.ndarray]:
        """Read the solved convex-combination weights for every SOS set."""
        pyo = self._pyo
        return {
            key: np.array([float(pyo.value(var)) for var in vars_])
            for key, vars_ in self.lambda_vars.items()
        }

    def _active_objective(self) -> Any:
        pyo = self._pyo
        for obj in self.model.component_data_objects(pyo.Objective, active=True):
            return obj
        raise FormulationError("SosxRelaxation model has no active objective.")

    def solve(self, solver: Any, *, persistent: bool = False) -> NodeSolution:
        """Solve the current node and return its relaxed objective and lambdas.

        Windows must already be applied with :meth:`apply_windows`. An
        infeasible node returns ``NodeSolution(feasible=False, ...)``
        rather than raising, so the brancher can simply skip it.
        """
        import time

        pyo = self._pyo
        t0 = time.perf_counter()
        if persistent:
            required = ("set_instance", "update_variables")
            missing = [name for name in required if not hasattr(solver, name)]
            if missing:
                raise FormulationError(
                    "persistent=True requires an APPSI-style solver exposing "
                    f"{required}; missing {missing} on {type(solver).__name__}."
                )
            try:
                if self._persistent_solver_id != id(solver):
                    if hasattr(solver, "config") and hasattr(solver.config, "load_solution"):
                        solver.config.load_solution = False
                    solver.set_instance(self.model)
                    self._persistent_solver_id = id(solver)
                elif self._changed_vars:
                    solver.update_variables(self._changed_vars)
                results = solver.solve(self.model)
            except (ValueError, RuntimeError):
                return NodeSolution(False, -np.inf, solve_seconds=time.perf_counter() - t0)
            solve_seconds = time.perf_counter() - t0
            if not _termination_ok(results):
                return NodeSolution(False, -np.inf, solve_seconds=solve_seconds)
            if hasattr(solver, "load_vars"):
                solver.load_vars()
        elif "appsi" in type(solver).__module__.lower():
            # APPSI wrappers (the default HiGHS path) auto-load the solution
            # on success and raise on an infeasible node.
            try:
                results = solver.solve(self.model)
            except (ValueError, RuntimeError):
                return NodeSolution(False, -np.inf, solve_seconds=time.perf_counter() - t0)
            solve_seconds = time.perf_counter() - t0
            if not _termination_ok(results):
                return NodeSolution(False, -np.inf, solve_seconds=solve_seconds)
        else:
            # Classic solvers (e.g. gurobi): defer the load so an infeasible
            # node neither warns nor loads a garbage/unbounded ray into the
            # model; load only once the status is good.
            results = solver.solve(self.model, load_solutions=False)
            solve_seconds = time.perf_counter() - t0
            if not _termination_ok(results):
                return NodeSolution(False, -np.inf, solve_seconds=solve_seconds)
            self.model.solutions.load_from(results)

        objective = float(pyo.value(self._active_objective()))
        return NodeSolution(
            feasible=True,
            objective=objective,
            lam=self.read_lambdas(),
            solve_seconds=solve_seconds,
        )


def _termination_ok(results: Any) -> bool:
    """Whether a Pyomo/APPSI results object reports an optimal/feasible solve.

    The check excludes the failure conditions first: ``"feasible"`` is a
    substring of ``"infeasible"`` (and Gurobi reports
    ``"infeasibleOrUnbounded"``), so a naive ``"feasible" in text`` would
    accept infeasible nodes.
    """
    condition = getattr(getattr(results, "solver", None), "termination_condition", None)
    if condition is None:
        condition = getattr(results, "termination_condition", None)
    text = str(condition).lower()
    if any(bad in text for bad in _BAD_TERMINATIONS):
        return False
    return any(good in text for good in _OK_TERMINATIONS)
