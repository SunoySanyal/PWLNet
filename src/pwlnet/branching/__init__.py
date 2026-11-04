"""SOSX branch-and-bound engine: nodes, selection, fathoming, incumbents.

The engine drives the SOSX workflow over a model that embeds a tanh
surrogate as a ``tanh_pwl_sos2`` block: it relaxes the SOS2 constraints
and recovers the segment structure by branching on the continuous
convex-combination weights, keeping each node an LP/MILP. The public
:func:`pwlnet.solvers.solve_sosx` entry point wires these components to
solver backends.
"""

from pwlnet.branching.branch_log import BranchLog, BranchRecord
from pwlnet.branching.fathoming import can_fathom_by_bound, is_improving
from pwlnet.branching.incumbent import Incumbent
from pwlnet.branching.node import Node, SosKey, Window
from pwlnet.branching.node_selection import Frontier
from pwlnet.branching.sosx_brancher import SosxBrancher
from pwlnet.formulations.sosx.sosx_solution import NodeSolution, SosxResult

__all__ = [
    "BranchLog",
    "BranchRecord",
    "Frontier",
    "Incumbent",
    "Node",
    "NodeSolution",
    "SosKey",
    "SosxBrancher",
    "SosxResult",
    "Window",
    "can_fathom_by_bound",
    "is_improving",
]
