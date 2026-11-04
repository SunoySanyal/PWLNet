"""Detect SOS2 violations in a solved SOSX node relaxation.

A set of convex-combination weights is *SOS2-valid* when at most two of
them are nonzero and, if two, they sit on adjacent breakpoints. The node
relaxation drops the SOS2 constraints, so a solved node can place weight
on a wider, non-adjacent support; those sets are the branching
candidates.
"""

from __future__ import annotations

import numpy as np

from pwlnet.branching.node import SosKey, Window


def support_spread(weights: np.ndarray, window: Window, tol: float) -> int:
    """Index distance between the first and last nonzero weight in ``window``.

    Returns ``0`` for an SOS2-valid support (one weight, or two adjacent
    weights) and a positive value measuring how far the support violates
    adjacency otherwise.
    """
    lo, hi = window
    nz = [k for k in range(lo, hi + 1) if weights[k] > tol]
    if len(nz) <= 1:
        return 0
    if len(nz) == 2 and nz[1] == nz[0] + 1:
        return 0
    return nz[-1] - nz[0]


def find_violations(
    lam: dict[SosKey, np.ndarray],
    windows: dict[SosKey, Window],
    *,
    tol: float = 1e-6,
) -> list[tuple[SosKey, int]]:
    """Return ``(key, spread)`` for every SOS set whose support is not SOS2-valid.

    The list preserves ``lam`` iteration order, so ``[0]`` is the
    first-violation candidate; ``spread`` lets a rule pick the
    most-violated set instead.
    """
    violations: list[tuple[SosKey, int]] = []
    for key, weights in lam.items():
        spread = support_spread(weights, windows[key], tol)
        if spread > 0:
            violations.append((key, spread))
    return violations


def find_first_violation(
    lam: dict[SosKey, np.ndarray],
    windows: dict[SosKey, Window],
    *,
    tol: float = 1e-6,
) -> SosKey | None:
    """Return the first SOS set that violates SOS2 validity, or ``None``."""
    for key, weights in lam.items():
        if support_spread(weights, windows[key], tol) > 0:
            return key
    return None


# ---------------------------------------------------------------------------
# Y-variable violation detection (paper formulation, Koksal et al. 2026)
# ---------------------------------------------------------------------------

def find_violations_Y(
    Y_values: dict[SosKey, np.ndarray],
    u_values: dict[SosKey, np.ndarray],
    windows: dict[SosKey, Window],
    *,
    tol: float = 1e-6,
) -> list[tuple[SosKey, int]]:
    """Detect SOS2 violations in a Y-variable node solution.

    A Y solution is SOS2-valid when for each SOS set there exists at most
    one segment k* that is partially filled (0 < Y[k*] < u[k*]), all lower
    segments are fully filled (Y[k] = u[k] for k < k*), and all higher
    segments are empty (Y[k] = 0 for k > k*).

    A violation at segment k means ``Y[k] < u[k] - tol`` (not yet full)
    while ``Y[k+1] > tol`` (next segment is already partially used).

    Parameters
    ----------
    Y_values : dict[SosKey, numpy.ndarray]
        Per SOS key, array of Y values for segments 1..P (0-indexed
        internally, so Y_values[key][0] = Y_1, ..., Y_values[key][P-1] = Y_P).
    u_values : dict[SosKey, numpy.ndarray]
        Per SOS key, array of segment widths u_k (same indexing as Y_values).
    windows : dict[SosKey, Window]
        Current ``(lo, hi)`` segment-index window for each SOS set.
    tol : float, optional
        Numerical tolerance.

    Returns
    -------
    list of (SosKey, int)
        ``(key, k)`` for each SOS set with a violation; ``k`` is the
        1-indexed segment where ``Y[k] < u[k]`` and ``Y[k+1] > 0``.
        At most one violation is reported per SOS set (the leftmost one).
    """
    violations: list[tuple[SosKey, int]] = []
    for key, Y in Y_values.items():
        lo, hi = windows[key]
        u = u_values[key]
        # Scan segments lo..hi-1 (0-indexed: lo-1..hi-2).
        for k1 in range(lo - 1, hi - 1):  # k1 is 0-indexed; segment k = k1+1
            if Y[k1] < u[k1] - tol and Y[k1 + 1] > tol:
                violations.append((key, k1 + 1))  # return 1-indexed k
                break
    return violations
