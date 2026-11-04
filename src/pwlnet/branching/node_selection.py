"""Open-node selection strategies for the SOSX brancher.

The :class:`Frontier` holds the open nodes and decides which one is
explored next:

* ``"best_bound"`` -- pop the node with the most promising relaxed
  bound first (best-first search). Finds strong incumbents early and
  keeps the proven optimality gap small.
* ``"depth_first"`` -- pop the most recently created node (LIFO dive).
  Each solve is then a one-window neighbour of the previous one, which
  maximises warm-start basis reuse on persistent solvers (see the
  Phase-6 spike benchmark in ``experiments/phase6_sosx_compare.py``).
"""

from __future__ import annotations

import heapq
import itertools

from pwlnet.branching.node import Node
from pwlnet.utils.exceptions import FormulationError

STRATEGIES = ("best_bound", "depth_first")


class Frontier:
    """Container of open branch-and-bound nodes with a pop strategy."""

    def __init__(self, strategy: str = "best_bound", *, maximize: bool = True):
        if strategy not in STRATEGIES:
            raise FormulationError(
                f"Unknown node-selection strategy '{strategy}'. "
                f"Choose one of {list(STRATEGIES)}."
            )
        self.strategy = strategy
        self.maximize = maximize
        self._heap: list[tuple[float, int, Node]] = []
        self._stack: list[Node] = []
        self._counter = itertools.count()

    def push(self, node: Node) -> None:
        """Add an open node to the frontier."""
        if self.strategy == "depth_first":
            self._stack.append(node)
        else:
            # heapq is a min-heap; negate the bound for a maximization so
            # the largest upper bound pops first. The counter breaks ties
            # deterministically without comparing Node objects.
            key = -node.bound if self.maximize else node.bound
            heapq.heappush(self._heap, (key, next(self._counter), node))

    def pop(self) -> Node:
        """Remove and return the next node to explore."""
        if self.strategy == "depth_first":
            return self._stack.pop()
        return heapq.heappop(self._heap)[2]

    def best_bound(self) -> float | None:
        """Return the strongest objective bound among all open nodes."""
        if self.strategy == "depth_first":
            if not self._stack:
                return None
            bounds = (node.bound for node in self._stack)
            return max(bounds) if self.maximize else min(bounds)
        if not self._heap:
            return None
        return self._heap[0][2].bound

    def __len__(self) -> int:
        return len(self._stack) if self.strategy == "depth_first" else len(self._heap)

    def __bool__(self) -> bool:
        return len(self) > 0
