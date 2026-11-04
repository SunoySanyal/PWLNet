"""Branch-and-bound node and the SOS-set window types it branches over.

A SOSX node is fully described by, per SOS set, the inclusive window of
breakpoint indices its convex-combination weights are allowed to use.
Tightening a window towards width two (two adjacent breakpoints) drives
the relaxation back to an SOS2-feasible point; see
:mod:`pwlnet.branching.sosx_brancher`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Inclusive ``(lo, hi)`` window of allowed breakpoint indices for one SOS set.
Window = tuple[int, int]

#: Identifies one SOS set: ``(block_name, layer_index, neuron_index)``.
SosKey = tuple[str, int, int]


@dataclass
class Node:
    """A single open node in the SOSX branch-and-bound tree.

    Parameters
    ----------
    bound : float
        The inherited objective bound for this subtree -- the parent
        node's relaxed objective. For a maximization it is an upper
        bound (tightening windows can only shrink the feasible set), so
        the node can be fathomed when ``bound`` cannot beat the
        incumbent.
    seq : int
        Monotonic creation index, used as a stable tie-breaker.
    windows : dict[SosKey, Window]
        The allowed breakpoint-index window for every SOS set at this
        node.
    depth : int, optional
        Tree depth (root is 0).
    parent : int or None, optional
        ``seq`` of the parent node, for logging/diagnostics.
    """

    bound: float
    seq: int
    windows: dict[SosKey, Window] = field(compare=False)
    depth: int = 0
    parent: int | None = None
