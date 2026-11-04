"""Lightweight progress log for the SOSX branch-and-bound loop.

Records one entry per solved node and per incumbent update. Records are
kept in memory for post-hoc inspection (``records``) and, when
``verbose`` is set, emitted through the standard PWLNet logger so a run
can be followed live without coupling the engine to ``print``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pwlnet.utils.logging import get_logger

_logger = get_logger(__name__)


@dataclass
class BranchRecord:
    """One solved-node event in the branch-and-bound trace."""

    seq: int
    depth: int
    status: str                 # "incumbent" | "feasible" | "fathomed" | "infeasible"
    objective: float | None = None


@dataclass
class BranchLog:
    """Accumulates :class:`BranchRecord` events and optionally logs them."""

    verbose: bool = False
    records: list[BranchRecord] = field(default_factory=list)

    def node(self, seq: int, depth: int, status: str, objective: float | None) -> None:
        """Record a solved node and (if verbose) log a one-line summary."""
        self.records.append(BranchRecord(seq, depth, status, objective))
        if self.verbose and status in {"incumbent", "infeasible"}:
            obj = "      --" if objective is None else f"{objective:10.4f}"
            _logger.info("node %4d  depth %2d  %-10s  obj=%s", seq, depth, status, obj)

    @property
    def n_incumbents(self) -> int:
        """Number of incumbent updates recorded."""
        return sum(1 for r in self.records if r.status == "incumbent")
