"""Reusable node relaxation for SOSX-MAE training (Y-variable formulation).

:class:`SosxTrainingRelaxation` implements the same relaxation protocol
as :class:`pwlnet.formulations.sosx.sosx_node.SosxRelaxation`
(``root_windows`` / ``apply_windows`` / ``solve``) so the *same*
:class:`pwlnet.branching.SosxBrancher` drives both fixed-network process
optimization and neural-network training.

Window semantics (Y-variable formulation)
------------------------------------------
A window ``(lo, hi)`` for SOS key ``(training, s, j)`` refers to *segment*
indices (1-indexed, 1 ≤ lo ≤ hi ≤ P = n_breakpoints - 1):

  - Segments 1 .. lo-1  are fixed to their maximum value u_k (fully filled).
  - Segments lo .. hi   are free within [0, u_k].
  - Segments hi+1 .. P  are fixed to 0.

This encodes the paper's two branches exactly:
  - Branch 1 (Y_k = u_k): window (lo, hi) → (k+1, hi)
  - Branch 2 (Y_{k+1:P}=0): window (lo, hi) → (lo, k)

A leaf node has lo = hi = k*: only one segment is free. Combined with the
locally ideal constraints already in the model, the LP solution at a leaf
satisfies SOS2 (at most one segment partially filled, all lower segments
full, all higher segments zero).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pwlnet.branching.node import SosKey, Window
from pwlnet.formulations.sosx.sosx_solution import NodeSolution
from pwlnet.formulations.sosx.training_model import Sos2TrainingModel, SosxTrainingModel
from pwlnet.utils.exceptions import FormulationError

__all__ = ["SosxTrainingRelaxation", "Sos2TrainingRelaxation"]

_OK_TERMINATIONS = ("optimal", "feasible")
_BAD_TERMINATIONS = (
    "infeasible",
    "unbounded",
    "error",
    "nosolution",
    "unknown",
    "interrupted",
)


class SosxTrainingRelaxation:
    """A reusable LP relaxation of the SOSX-MAE training model.

    Parameters
    ----------
    training_model : SosxTrainingModel
        The built training model (see
        :func:`pwlnet.formulations.sosx.build_sosx_mae_training_model`).
    """

    def __init__(self, training_model: SosxTrainingModel):
        import pyomo.environ as pyo

        self._pyo = pyo
        self.training_model = training_model
        self.model = training_model.model
        self.Y_vars = training_model.Y_vars
        self.u_values = training_model.u_values
        self._n_segments = training_model.n_segments
        self._changed_vars: list[Any] = []
        self._persistent_solver_id: int | None = None

    # -- relaxation protocol ------------------------------------------------

    def root_windows(self) -> dict[SosKey, Window]:
        """The unrestricted ``(1, P)`` segment window for every SOS set."""
        P = self._n_segments
        return {key: (1, P) for key in self.Y_vars}

    def apply_windows(self, windows: dict[SosKey, Window]) -> list[Any]:
        """Fix Y variables outside each set's window; unfix inside.

        For window ``(lo, hi)``:
          - Segments < lo: fix Y = u_k (full)
          - Segments > hi: fix Y = 0
          - Segments lo..hi: unfix (free within variable bounds)

        Returns the VarData whose fixed-state changed so a persistent
        solver can push just those updates.
        """
        changed: list[Any] = []
        for key, variables in self.Y_vars.items():
            lo, hi = windows[key]
            u = self.u_values[key]
            # variables is 0-indexed list; segment k (1-indexed) → variables[k-1]
            for idx, variable in enumerate(variables):
                seg = idx + 1  # convert to 1-indexed segment number
                if seg < lo:
                    # Force full (Y = u_k).
                    target = float(u[idx])
                    if not variable.fixed or variable.value != target:
                        variable.fix(target)
                        changed.append(variable)
                elif seg > hi:
                    # Force empty (Y = 0).
                    if not variable.fixed or variable.value != 0.0:
                        variable.fix(0.0)
                        changed.append(variable)
                else:
                    # Free within [0, u_k].
                    if variable.fixed:
                        variable.unfix()
                        changed.append(variable)
        self._changed_vars = changed
        return changed

    def read_Y(self) -> dict[SosKey, np.ndarray]:
        """Read the solved segment-fill values for every SOS set."""
        pyo = self._pyo
        return {
            key: np.array([float(pyo.value(v)) for v in variables], dtype=float)
            for key, variables in self.Y_vars.items()
        }

    def solve(self, solver: Any, *, persistent: bool = False) -> NodeSolution:
        """Solve the current node and return its objective and Y values.

        Windows must already be applied with :meth:`apply_windows`. An
        infeasible node returns ``NodeSolution(feasible=False, ...)``
        rather than raising, so the brancher simply skips it.
        """
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
                    if hasattr(solver, "config") and hasattr(
                        solver.config, "load_solution"
                    ):
                        solver.config.load_solution = False
                    solver.set_instance(self.model)
                    self._persistent_solver_id = id(solver)
                elif self._changed_vars:
                    solver.update_variables(self._changed_vars)
                results = solver.solve(self.model)
            except (ValueError, RuntimeError):
                return NodeSolution(False, np.inf, solve_seconds=time.perf_counter() - t0)
            solve_seconds = time.perf_counter() - t0
            if not _termination_ok(results):
                return NodeSolution(False, np.inf, solve_seconds=solve_seconds)
            if hasattr(solver, "load_vars"):
                solver.load_vars()
        elif "appsi" in type(solver).__module__.lower():
            try:
                results = solver.solve(self.model)
            except (ValueError, RuntimeError):
                return NodeSolution(False, np.inf, solve_seconds=time.perf_counter() - t0)
            solve_seconds = time.perf_counter() - t0
            if not _termination_ok(results):
                return NodeSolution(False, np.inf, solve_seconds=solve_seconds)
        else:
            results = solver.solve(self.model, load_solutions=False)
            solve_seconds = time.perf_counter() - t0
            if not _termination_ok(results):
                return NodeSolution(False, np.inf, solve_seconds=solve_seconds)
            self.model.solutions.load_from(results)

        objective = float(pyo.value(self.model.objective))
        # lam field carries Y values for training relaxation.
        return NodeSolution(
            feasible=True,
            objective=objective,
            lam=self.read_Y(),
            solve_seconds=solve_seconds,
        )


class Sos2TrainingRelaxation:
    """A reusable LP relaxation of the SOS2-MAE training model (λ formulation).

    Uses the original convex-combination (λ) formulation.  Windows are
    breakpoint-indexed ``(lo, hi)`` (0-indexed, 0 ≤ lo ≤ hi ≤ K-1): λ
    values outside the window are fixed to zero, those inside are free
    in [0, 1].  Violation detection and branching use the λ-based
    symmetric bisection from :func:`~pwlnet.formulations.sosx.find_violations`.

    Parameters
    ----------
    training_model : Sos2TrainingModel
        The built SOS2 training model.
    """

    def __init__(self, training_model: Sos2TrainingModel):
        import pyomo.environ as pyo

        self._pyo = pyo
        self.training_model = training_model
        self.model = training_model.model
        self.lambda_vars = training_model.lambda_vars
        self._n_breakpoints = training_model.n_breakpoints
        self._changed_vars: list[Any] = []
        self._persistent_solver_id: int | None = None

    def root_windows(self) -> dict[SosKey, Window]:
        """The unrestricted ``(0, K-1)`` window for every SOS set."""
        hi = self._n_breakpoints - 1
        return {key: (0, hi) for key in self.lambda_vars}

    def apply_windows(self, windows: dict[SosKey, Window]) -> list[Any]:
        """Fix λ values outside each window to zero; unfix inside."""
        changed: list[Any] = []
        for key, variables in self.lambda_vars.items():
            lo, hi = windows[key]
            for index, variable in enumerate(variables):
                should_fix = index < lo or index > hi
                if should_fix and not variable.fixed:
                    variable.fix(0.0)
                    changed.append(variable)
                elif not should_fix and variable.fixed:
                    variable.unfix()
                    changed.append(variable)
        self._changed_vars = changed
        return changed

    def read_lambdas(self) -> dict[SosKey, np.ndarray]:
        """Read the solved convex-combination weights for every SOS set."""
        pyo = self._pyo
        return {
            key: np.array([float(pyo.value(v)) for v in variables], dtype=float)
            for key, variables in self.lambda_vars.items()
        }

    def solve(self, solver: Any, *, persistent: bool = False) -> NodeSolution:
        """Solve the current node and return its objective and λ values."""
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
                    if hasattr(solver, "config") and hasattr(
                        solver.config, "load_solution"
                    ):
                        solver.config.load_solution = False
                    solver.set_instance(self.model)
                    self._persistent_solver_id = id(solver)
                elif self._changed_vars:
                    solver.update_variables(self._changed_vars)
                results = solver.solve(self.model)
            except (ValueError, RuntimeError):
                return NodeSolution(False, np.inf, solve_seconds=time.perf_counter() - t0)
            solve_seconds = time.perf_counter() - t0
            if not _termination_ok(results):
                return NodeSolution(False, np.inf, solve_seconds=solve_seconds)
            if hasattr(solver, "load_vars"):
                solver.load_vars()
        elif "appsi" in type(solver).__module__.lower():
            try:
                results = solver.solve(self.model)
            except (ValueError, RuntimeError):
                return NodeSolution(False, np.inf, solve_seconds=time.perf_counter() - t0)
            solve_seconds = time.perf_counter() - t0
            if not _termination_ok(results):
                return NodeSolution(False, np.inf, solve_seconds=solve_seconds)
        else:
            results = solver.solve(self.model, load_solutions=False)
            solve_seconds = time.perf_counter() - t0
            if not _termination_ok(results):
                return NodeSolution(False, np.inf, solve_seconds=solve_seconds)
            self.model.solutions.load_from(results)

        objective = float(pyo.value(self.model.objective))
        return NodeSolution(
            feasible=True,
            objective=objective,
            lam=self.read_lambdas(),
            solve_seconds=solve_seconds,
        )


def _termination_ok(results: Any) -> bool:
    """Whether a Pyomo/APPSI results object reports an optimal/feasible solve."""
    condition = getattr(getattr(results, "solver", None), "termination_condition", None)
    if condition is None:
        condition = getattr(results, "termination_condition", None)
    text = str(condition).lower()
    if any(bad in text for bad in _BAD_TERMINATIONS):
        return False
    return any(good in text for good in _OK_TERMINATIONS)
