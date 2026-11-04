"""Fathoming predicates for the SOSX branch-and-bound loop.

A node (or its whole subtree) can be discarded when its objective bound
cannot strictly improve on the incumbent. Both predicates are sense-aware
through ``maximize`` and use the same ``tol`` the brancher uses for
incumbent updates, so a node is never fathomed against a value it could
have tied.
"""

from __future__ import annotations


def can_fathom_by_bound(
    bound: float, incumbent_objective: float | None, *, maximize: bool, tol: float
) -> bool:
    """Return ``True`` if a subtree's ``bound`` cannot beat the incumbent.

    For a maximization, ``bound`` is an upper bound on every descendant,
    so the subtree is hopeless once it sits at or below the incumbent.
    The minimization case is symmetric with a lower bound.
    """
    if incumbent_objective is None:
        return False
    if maximize:
        return bound <= incumbent_objective + tol
    return bound >= incumbent_objective - tol


def is_improving(
    objective: float, incumbent_objective: float | None, *, maximize: bool, tol: float
) -> bool:
    """Return ``True`` if ``objective`` strictly improves on the incumbent."""
    if incumbent_objective is None:
        return True
    if maximize:
        return objective > incumbent_objective + tol
    return objective < incumbent_objective - tol
