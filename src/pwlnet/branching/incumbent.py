"""Best-so-far (incumbent) tracker for the SOSX branch-and-bound loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pwlnet.branching.fathoming import is_improving
from pwlnet.branching.node import SosKey


@dataclass
class Incumbent:
    """The best SOS2-feasible node found so far.

    ``values`` is a snapshot of the user-tracked model quantities (e.g.
    process inputs and surrogate outputs) at the incumbent point;
    ``lam`` holds the SOS-set convex-combination weights so the chosen
    breakpoints can be inspected after the solve.
    """

    maximize: bool = True
    objective: float | None = None
    values: dict[str, float] = field(default_factory=dict)
    lam: dict[SosKey, np.ndarray] | None = None
    depth: int | None = None
    seq: int | None = None
    n_updates: int = 0

    def consider(
        self,
        objective: float,
        values: dict[str, float],
        lam: dict[SosKey, Any] | None,
        *,
        depth: int | None = None,
        seq: int | None = None,
        tol: float = 1e-7,
    ) -> bool:
        """Replace the incumbent if ``objective`` strictly improves it.

        Returns ``True`` when the incumbent was updated.
        """
        if not is_improving(objective, self.objective, maximize=self.maximize, tol=tol):
            return False
        self.objective = objective
        self.values = dict(values)
        self.lam = lam
        self.depth = depth
        self.seq = seq
        self.n_updates += 1
        return True

    @property
    def found(self) -> bool:
        """Whether any SOS2-feasible point has been accepted."""
        return self.objective is not None
