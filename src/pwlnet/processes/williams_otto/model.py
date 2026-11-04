"""Williams-Otto reactor: constants, kinetics, balances, profit.

Implements the steady-state Williams-Otto CSTR benchmark on a
mass-fraction basis exactly as specified in
``docs/williams_otto_spec.md`` (canonical Mendoza-2016 /
DDO-4-ChemEng parameter set, spec sections 1-4):

Reactions
    A + B -> C (k1),  B + C -> P + E (k2),  C + P -> G (k3)

Arrhenius kinetics (``T_R`` in degC, exponentials evaluated in Kelvin)
    ``k_i = a_i * exp(-b_i / (T_R + 273.15))``  [1/s]

Six steady-state component balances with mass-weighted stoichiometric
coefficients (spec section 2); their sum is identically
``FA + FB - F * sum(X)``, so ``sum(X) = 1`` is implied and is used only
as a consistency check, never as a seventh equation.

This module is pure numpy; the Pyomo counterpart lives in
:mod:`pwlnet.processes.williams_otto.pyomo_model`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from pwlnet.utils.typing import ArrayLike

__all__ = [
    "CELSIUS_TO_KELVIN",
    "DEFAULT_INITIAL_GUESS",
    "INPUT_NAMES",
    "OUTPUT_NAMES",
    "WilliamsOttoParameters",
    "profit",
    "rate_constants",
    "residuals",
]

#: Ordered process inputs: feed rate of B [kg/s] and reactor temperature [degC].
INPUT_NAMES: list[str] = ["FB", "TR"]

#: Ordered process outputs: reactor mass fractions.
OUTPUT_NAMES: list[str] = ["XA", "XB", "XC", "XE", "XG", "XP"]

#: Additive Celsius -> Kelvin conversion used inside the Arrhenius terms.
CELSIUS_TO_KELVIN: float = 273.15

#: Spec section 5 initial guess for the steady-state solve,
#: ``X0 = (XA, XB, XC, XE, XG, XP)``.
DEFAULT_INITIAL_GUESS: np.ndarray = np.array([0.1, 0.35, 0.02, 0.25, 0.1, 0.1])


@dataclass(frozen=True)
class WilliamsOttoParameters:
    """Canonical Williams-Otto parameter set (spec sections 1, 3, 4).

    All defaults are the verified "Mendoza-2016 / DDO-4-ChemEng" values;
    do not mix them with the Forbes-Marlin 3-input or Amrit dynamic
    variants (spec section 3 variant table). The alternative
    "Forbes-Marlin prices" (1143.38, 25.92, 76.23, 114.34) can be
    reproduced by overriding the four price/cost fields.

    Attributes
    ----------
    fa : float
        Fixed feed mass flow rate of pure A [kg/s].
    holdup : float
        Reactor mass holdup ``W`` [kg].
    a1, a2, a3 : float
        Arrhenius pre-exponential factors [1/s].
    b1, b2, b3 : float
        Arrhenius activation terms [K].
    price_p, price_e : float
        Selling prices of product P and by-product E [$ per kg].
    cost_a, cost_b : float
        Raw-material costs of feeds A and B [$ per kg].
    xa_max, xg_max : float
        Operating constraints ``XA <= xa_max``, ``XG <= xg_max``.
    fb_bounds : tuple of float
        Canonical decision range for ``FB`` [kg/s].
    tr_bounds : tuple of float
        Canonical decision range for ``T_R`` [degC].
    """

    fa: float = 1.8275
    holdup: float = 2105.2
    a1: float = 1.6599e6
    b1: float = 6666.7
    a2: float = 7.2117e8
    b2: float = 8333.3
    a3: float = 2.6745e12
    b3: float = 11111.0
    price_p: float = 1043.38
    price_e: float = 20.92
    cost_a: float = 79.23
    cost_b: float = 118.34
    xa_max: float = 0.12
    xg_max: float = 0.08
    fb_bounds: tuple[float, float] = (4.0, 7.0)
    tr_bounds: tuple[float, float] = (70.0, 100.0)


def rate_constants(
    tr: ArrayLike, params: WilliamsOttoParameters | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrhenius rate constants at reactor temperature ``tr``.

    Parameters
    ----------
    tr : array_like
        Reactor temperature ``T_R`` in **degrees Celsius** (scalar or
        array; converted to Kelvin internally).
    params : WilliamsOttoParameters or None, optional
        Parameter set (defaults to the canonical values).

    Returns
    -------
    tuple of numpy.ndarray
        ``(k1, k2, k3)`` in 1/s, with the shape of ``tr``.
    """
    p = params if params is not None else WilliamsOttoParameters()
    t_kelvin = np.asarray(tr, dtype=float) + CELSIUS_TO_KELVIN
    k1 = p.a1 * np.exp(-p.b1 / t_kelvin)
    k2 = p.a2 * np.exp(-p.b2 / t_kelvin)
    k3 = p.a3 * np.exp(-p.b3 / t_kelvin)
    return k1, k2, k3


def residuals(
    x: ArrayLike,
    fb: float,
    tr: float,
    params: WilliamsOttoParameters | None = None,
) -> np.ndarray:
    """Residuals of the six steady-state component mass balances.

    Implements spec section 2 exactly. With ``F = FA + FB`` and the
    mass-basis reaction rates ``r1 = k1*XA*XB*W``, ``r2 = k2*XB*XC*W``,
    ``r3 = k3*XC*XP*W`` (kg/s)::

        0 = FA - F*XA -     r1                      (A)
        0 = FB - F*XB -     r1 -     r2             (B)
        0 =    - F*XC + 2.0*r1 - 2.0*r2 -     r3    (C)
        0 =    - F*XE          + 2.0*r2             (E)
        0 =    - F*XG                   + 1.5*r3    (G)
        0 =    - F*XP          +     r2 - 0.5*r3    (P)

    Parameters
    ----------
    x : array_like, shape (6,)
        Mass fractions ``(XA, XB, XC, XE, XG, XP)``.
    fb : float
        Feed mass flow rate of B [kg/s].
    tr : float
        Reactor temperature ``T_R`` [degC].
    params : WilliamsOttoParameters or None, optional
        Parameter set (defaults to the canonical values).

    Returns
    -------
    numpy.ndarray, shape (6,)
        Balance residuals in kg/s, ordered ``(A, B, C, E, G, P)``. At a
        steady state all six vanish and ``sum(x) == 1`` follows.
    """
    p = params if params is not None else WilliamsOttoParameters()
    xa, xb, xc, xe, xg, xp = np.asarray(x, dtype=float)
    k1, k2, k3 = rate_constants(tr, p)
    f = p.fa + fb
    r1 = k1 * xa * xb * p.holdup
    r2 = k2 * xb * xc * p.holdup
    r3 = k3 * xc * xp * p.holdup
    return np.array(
        [
            p.fa - f * xa - r1,
            fb - f * xb - r1 - r2,
            -f * xc + 2.0 * r1 - 2.0 * r2 - r3,
            -f * xe + 2.0 * r2,
            -f * xg + 1.5 * r3,
            -f * xp + r2 - 0.5 * r3,
        ]
    )


def profit(
    x: ArrayLike | Mapping[str, float],
    fb: float,
    params: WilliamsOttoParameters | None = None,
) -> float:
    """Economic profit ``phi`` at an operating point [$ per s].

    Canonical "Mendoza-2016 prices" objective (spec section 4)::

        phi = (price_p*XP + price_e*XE) * (FA + FB) - cost_a*FA - cost_b*FB

    Parameters
    ----------
    x : array_like of shape (6,), or mapping
        Mass fractions, either ordered as :data:`OUTPUT_NAMES` or as a
        mapping with (at least) keys ``"XP"`` and ``"XE"``.
    fb : float
        Feed mass flow rate of B [kg/s].
    params : WilliamsOttoParameters or None, optional
        Parameter set (defaults to the canonical values).

    Returns
    -------
    float
        Profit in $/s.
    """
    p = params if params is not None else WilliamsOttoParameters()
    if isinstance(x, Mapping):
        x_p = float(x["XP"])
        x_e = float(x["XE"])
    else:
        arr = np.asarray(x, dtype=float)
        x_p = float(arr[OUTPUT_NAMES.index("XP")])
        x_e = float(arr[OUTPUT_NAMES.index("XE")])
    f = p.fa + fb
    return (p.price_p * x_p + p.price_e * x_e) * f - p.cost_a * p.fa - p.cost_b * fb
