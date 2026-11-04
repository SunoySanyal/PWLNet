"""Van de Vusse CSTR benchmark model.

The benchmark follows the isothermal, constant-volume Van de Vusse
reaction network used in ``papers/VandeVusseProcess.pdf``:

    A -> B -> C,    2 A -> D

The manipulated input is the dilution rate ``D = F / V`` [1/min].  The
state/output vector is ordered as ``(CA, CB, CC, CD)`` in mol/L, where
``CB`` is the desired product concentration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pwlnet.utils.typing import ArrayLike

__all__ = [
    "DEFAULT_INITIAL_STATE",
    "CONTROL_BOUNDS",
    "CONTROL_NAMES",
    "DECISION_DT_MIN",
    "INPUT_NAMES",
    "OPERATING_POINT_1",
    "OPERATING_POINT_2",
    "OUTPUT_NAMES",
    "STATE_BOUNDS",
    "STATE_NAMES",
    "T_FINAL_MIN",
    "VanDeVusseParameters",
    "product_concentration",
    "rhs",
    "steady_state",
    "steady_state_residuals",
]

INPUT_NAMES: list[str] = ["D"]
CONTROL_NAMES: list[str] = INPUT_NAMES.copy()
STATE_NAMES: list[str] = ["CA", "CB", "CC", "CD"]
OUTPUT_NAMES: list[str] = STATE_NAMES.copy()

DEFAULT_INITIAL_STATE: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
STATE_BOUNDS: np.ndarray = np.array(
    [
        [0.0, 10.0],
        [0.0, 2.0],
        [0.0, 10.0],
        [0.0, 8.0],
    ],
    dtype=float,
)
CONTROL_BOUNDS: np.ndarray = np.array([[0.1, 4.0]], dtype=float)
DECISION_DT_MIN: float = 0.10
T_FINAL_MIN: float = 10.0


@dataclass(frozen=True)
class VanDeVusseParameters:
    """Paper parameter set for the isothermal Van de Vusse CSTR.

    Attributes
    ----------
    k1 : float
        First-order rate constant for ``A -> B`` [1/min].
    k2 : float
        First-order rate constant for ``B -> C`` [1/min].
    k3 : float
        Second-order rate constant for ``2 A -> D`` [L/(mol min)].
    caf : float
        Feed concentration of reactant A [mol/L].
    dilution_bounds : tuple of float
        Default sampling/control range for ``D = F / V`` [1/min].
    """

    k1: float = 5.0 / 6.0
    k2: float = 5.0 / 3.0
    k3: float = 1.0 / 6.0
    caf: float = 10.0
    dilution_bounds: tuple[float, float] = (0.1, 4.0)


OPERATING_POINT_1: dict[str, float] = {
    "D": 4.0 / 7.0,
    "CA": 3.0,
    "CB": 1.117,
    "CAf": 10.0,
}
"""Paper Set 1: inverse-response operating point."""

OPERATING_POINT_2: dict[str, float] = {
    "D": 2.8744,
    "CA": 6.0870,
    "CB": 1.117,
    "CAf": 10.0,
}
"""Paper Set 2: same ``CB`` as Set 1, without inverse response."""


def rhs(
    x: ArrayLike,
    dilution_rate: float,
    params: VanDeVusseParameters | None = None,
) -> np.ndarray:
    """Return dynamic concentration derivatives for one operating point.

    Parameters
    ----------
    x : array_like, shape (4,)
        Concentrations ``(CA, CB, CC, CD)`` [mol/L].
    dilution_rate : float
        Dilution rate ``D = F / V`` [1/min].
    params : VanDeVusseParameters or None, optional
        Kinetic/feed parameters.

    Returns
    -------
    numpy.ndarray, shape (4,)
        ``d(CA, CB, CC, CD) / dt`` in mol/(L min).
    """
    p = params if params is not None else VanDeVusseParameters()
    d = float(dilution_rate)
    ca, cb, cc, cd = np.asarray(x, dtype=float)
    return np.array(
        [
            d * (p.caf - ca) - p.k1 * ca - p.k3 * ca * ca,
            p.k1 * ca - (d + p.k2) * cb,
            p.k2 * cb - d * cc,
            0.5 * p.k3 * ca * ca - d * cd,
        ],
        dtype=float,
    )


def steady_state(
    dilution_rate: ArrayLike,
    params: VanDeVusseParameters | None = None,
) -> np.ndarray:
    """Analytic steady state for one or many dilution rates.

    The positive root of the A balance is used:

    ``k3*CA**2 + (D + k1)*CA - D*CAf = 0``.

    Parameters
    ----------
    dilution_rate : array_like
        Scalar or array of dilution rates ``D`` [1/min].
    params : VanDeVusseParameters or None, optional
        Kinetic/feed parameters.

    Returns
    -------
    numpy.ndarray
        Shape ``(4,)`` for a scalar input, otherwise ``dilution_rate.shape + (4,)``.
    """
    p = params if params is not None else VanDeVusseParameters()
    d = np.asarray(dilution_rate, dtype=float)
    if np.any(d <= 0.0):
        raise ValueError("dilution_rate must be positive.")

    disc = (d + p.k1) ** 2 + 4.0 * p.k3 * d * p.caf
    ca = (-(d + p.k1) + np.sqrt(disc)) / (2.0 * p.k3)
    cb = p.k1 * ca / (d + p.k2)
    cc = p.k2 * cb / d
    cd = 0.5 * p.k3 * ca * ca / d
    states = np.stack([ca, cb, cc, cd], axis=-1)
    if states.ndim == 1:
        return states
    return states


def steady_state_residuals(
    x: ArrayLike,
    dilution_rate: float,
    params: VanDeVusseParameters | None = None,
) -> np.ndarray:
    """Residuals of the steady-state balances."""
    return rhs(x, dilution_rate=dilution_rate, params=params)


def product_concentration(
    dilution_rate: ArrayLike,
    params: VanDeVusseParameters | None = None,
) -> np.ndarray:
    """Return the desired-product steady-state concentration ``CB``."""
    return steady_state(dilution_rate, params=params)[..., OUTPUT_NAMES.index("CB")]
