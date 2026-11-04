"""Branching rules for SOSX: which SOS set to split and how.

Given the violations found by :mod:`pwlnet.formulations.sosx.sosx_violation`,
a *selection rule* picks the SOS set to branch on, and
:func:`split_window` splits that set's breakpoint window into two
strictly smaller children around the weighted-mean index. Repeatedly
narrowing the window drives every SOS set towards a width-two
(SOS2-valid) support.
"""

from __future__ import annotations

import numpy as np

from pwlnet.branching.node import SosKey, Window
from pwlnet.utils.exceptions import FormulationError

SELECTION_RULES = ("first", "most_violated")

# ---------------------------------------------------------------------------
# Y-variable branching (paper formulation, Koksal et al. 2026)
# ---------------------------------------------------------------------------


def split_window_Y(
    windows: dict[SosKey, Window],
    violation: tuple[SosKey, int],
) -> tuple[dict[SosKey, Window], dict[SosKey, Window]]:
    """Split windows according to the paper's asymmetric Y-branching rule.

    Given a violation ``(key, k)`` where segment k satisfies
    ``Y[k] < u[k]`` and ``Y[k+1] > 0``, two child nodes are produced:

    - **Branch 1** (``Y_k = u_k``): segment k is forced full; the free
      window advances to ``(k+1, hi)``.
    - **Branch 2** (``Y_{k+1:hi} = 0``): segments k+1..hi are zeroed;
      the free window shrinks to ``(lo, k)``.

    Both branches are strictly smaller, guaranteeing termination.

    Parameters
    ----------
    windows : dict[SosKey, Window]
        Current node windows (segment indices, 1-indexed).
    violation : (SosKey, int)
        A ``(key, k)`` pair from
        :func:`~pwlnet.formulations.sosx.sosx_violation.find_violations_Y`.

    Returns
    -------
    branch1_windows, branch2_windows : tuple of dict
        Child node windows for Branch 1 and Branch 2 respectively.
    """
    key, k = violation
    lo, hi = windows[key]
    if k < lo or k >= hi:
        raise FormulationError(
            f"Violation segment k={k} is outside window ({lo}, {hi}) for key {key}."
        )
    branch1 = dict(windows)
    branch1[key] = (k + 1, hi)   # Branch 1: segment k full → advance lo
    branch2 = dict(windows)
    branch2[key] = (lo, k)        # Branch 2: segments k+1..hi zeroed → shrink hi
    return branch1, branch2


def select_violation(
    violations: list[tuple[SosKey, int]], rule: str = "first"
) -> tuple[SosKey, int] | None:
    """Pick the violation entry to branch on from the violation list.

    Parameters
    ----------
    violations : list of (SosKey, int)
        ``(key, value)`` pairs — for λ-based detection ``value`` is the
        support spread; for Y-based detection it is the 1-indexed violating
        segment ``k``.
    rule : {"first", "most_violated"}, optional
        ``"first"`` takes the earliest violated set (cheap, the default);
        ``"most_violated"`` takes the set with the largest ``value``, which
        tends to tighten the relaxation faster per branch.

    Returns
    -------
    (SosKey, int) or None
        The full chosen violation entry, or ``None`` when there are none.
    """
    if not violations:
        return None
    if rule == "first":
        return violations[0]
    if rule == "most_violated":
        return max(violations, key=lambda item: item[1])
    raise FormulationError(
        f"Unknown branching rule '{rule}'. Choose one of {list(SELECTION_RULES)}."
    )


def split_window_at(
    windows: dict[SosKey, Window],
    key: SosKey,
    r: int,
) -> tuple[dict[SosKey, Window], dict[SosKey, Window]]:
    """Split ``key``'s window at a precomputed breakpoint index ``r``.

    Used by the parallel driver when the worker has already computed the
    weighted-mean split point so the driver does not need the λ array.
    """
    lo, hi = windows[key]
    left = dict(windows)
    left[key] = (lo, r)
    right = dict(windows)
    right[key] = (r, hi)
    return left, right


def split_window(
    windows: dict[SosKey, Window], key: SosKey, lam: dict[SosKey, np.ndarray]
) -> tuple[dict[SosKey, Window], dict[SosKey, Window]]:
    """Split ``key``'s window into two children around its weighted mean.

    The split index ``r`` is the weight-weighted mean breakpoint,
    clamped to ``[lo+1, hi-1]`` so both children are strictly smaller
    than the parent (guaranteeing termination). The left child keeps
    ``(lo, r)`` and the right child ``(r, hi)``; ``r`` is shared so the
    union still covers the parent window.
    """
    lo, hi = windows[key]
    idx = np.arange(lo, hi + 1)
    weights = lam[key][lo : hi + 1]
    total = float(weights.sum())
    mean = float((weights * idx).sum() / total) if total > 0 else 0.5 * (lo + hi)
    r = max(lo + 1, min(hi - 1, int(round(mean))))

    left = dict(windows)
    left[key] = (lo, r)
    right = dict(windows)
    right[key] = (r, hi)
    return left, right
