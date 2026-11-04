"""Result containers for SOSX node solves and the overall optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pwlnet.branching.node import SosKey


@dataclass
class NodeSolution:
    """Outcome of solving one SOSX node relaxation.

    ``lam`` maps each SOS set to its solved convex-combination weights
    (used to detect violations and choose the branch); it is ``None``
    when the node was infeasible.
    """

    feasible: bool
    objective: float
    lam: dict[SosKey, np.ndarray] | None = None
    solve_seconds: float = 0.0


@dataclass
class SosxResult:
    """Final report from :func:`pwlnet.solvers.solve_sosx`.

    Attributes
    ----------
    status : str
        ``"optimal"`` when the tree was fully explored, ``"node_limit"``
        when ``max_nodes`` was hit first, ``"time_limit"`` when the
        brancher wall-clock limit was reached, ``"gap_limit"`` when a
        requested optimality gap was reached, or ``"infeasible"`` when
        the complete tree contains no SOS2-feasible point.
    objective : float or None
        Incumbent objective (``None`` if infeasible).
    values : dict[str, float]
        Snapshot of the user-tracked model quantities at the incumbent.
    n_nodes : int
        Number of node relaxations solved.
    solver_seconds : float
        Cumulative solver-call time (model build/update excluded).
    wall_seconds : float
        Total brancher wall time.
    best_bound : float or None
        Best objective bound remaining when the solve stopped.
    absolute_gap, relative_gap : float or None
        Final incumbent-to-bound optimality gaps.
    incumbent_depth : int or None
        Tree depth at which the incumbent was found.
    lam : dict or None
        Incumbent SOS-set convex-combination weights.
    warm_started : bool
        Whether a previous incumbent was successfully revalidated and used
        to seed this solve.
    """

    status: str
    objective: float | None
    values: dict[str, float] = field(default_factory=dict)
    n_nodes: int = 0
    solver_seconds: float = 0.0
    wall_seconds: float = 0.0
    best_bound: float | None = None
    absolute_gap: float | None = None
    relative_gap: float | None = None
    incumbent_depth: int | None = None
    lam: dict[SosKey, np.ndarray] | None = None
    persistent: bool = False
    warm_started: bool = False

    @property
    def found(self) -> bool:
        """Whether an SOS2-feasible incumbent was found."""
        return self.objective is not None
