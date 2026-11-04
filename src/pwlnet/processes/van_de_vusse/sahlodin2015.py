"""Sahlodin et al. (2015) nonisothermal Van de Vusse reactor.

This module follows the variable-volume jacketed reactor in
``papers/sahlodin2015.pdf``, subsection 4.2, equations (12)-(13) and Table 3.
The original paper states concentrations in ``mol/m^3``, temperatures in
degrees Celsius, volume in ``m^3``, time in hours, inlet flow in ``m^3/h``, and
cooling power in ``kJ/h``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pwlnet.utils.typing import ArrayLike

__all__ = [
    "SAHLODIN_CONTROL_BOUNDS",
    "SAHLODIN_CONTROL_NAMES",
    "SAHLODIN_DECISION_DT_H",
    "SAHLODIN_HORIZON_H",
    "SAHLODIN_INITIAL_STATE",
    "SAHLODIN_ON_SPEC_LIMITS",
    "SAHLODIN_OUTPUT_NAMES",
    "SAHLODIN_STATE_BOUNDS",
    "SAHLODIN_STATE_NAMES",
    "SahlodinVanDeVusseParameters",
    "heat_release",
    "outlet_flow",
    "reaction_rates",
    "rhs",
    "simulate_policy",
    "step_state",
]

SAHLODIN_STATE_NAMES: list[str] = ["CA", "CB", "CC", "CD", "T", "Tc", "V"]
SAHLODIN_OUTPUT_NAMES: list[str] = SAHLODIN_STATE_NAMES.copy()
SAHLODIN_CONTROL_NAMES: list[str] = ["Fin", "Pc"]

SAHLODIN_INITIAL_STATE: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 0.0, 108.0, 107.7, 1.0e-3],
    dtype=float,
)
"""Equation (13): ``CA, CB, CC, CD = 0``, ``T=108``, ``Tc=107.7``, ``V=1e-3``."""

SAHLODIN_STATE_BOUNDS: np.ndarray = np.array(
    [
        [0.0, 3000.0],
        [0.0, 2000.0],
        [0.0, 500.0],
        [0.0, 350.0],
        [0.0, 111.0],
        [0.0, 115.0],
        [0.0, 2.0],
    ],
    dtype=float,
)
"""State bounds from equation (P14), ordered as :data:`SAHLODIN_STATE_NAMES`."""

SAHLODIN_CONTROL_BOUNDS: np.ndarray = np.array(
    [
        [0.0, 40.0],
        [0.0, 4000.0],
    ],
    dtype=float,
)
"""Control bounds from subsection 4.2: ``Fin`` and ``Pc``."""

SAHLODIN_ON_SPEC_LIMITS: dict[str, float] = {"CC_max": 500.0, "CD_max": 350.0}
SAHLODIN_HORIZON_H: float = 0.8
SAHLODIN_DECISION_DT_H: float = 6.0 / 3600.0


@dataclass(frozen=True)
class SahlodinVanDeVusseParameters:
    """Table 3 parameters for the Sahlodin et al. Van de Vusse reactor."""

    k01: float = 1.29e12
    k02: float = 9.04e6
    e1: float = 9758.3
    e2: float = 8560.0
    delta_h_ab: float = 4.2
    delta_h_bc: float = -11.0
    delta_h_ad: float = -41.85
    alpha: float = 30.828
    beta: float = 86.688
    gamma: float = 0.1
    delta: float = 3.52e-4
    eta: float = 30.0
    tin: float = 104.9
    ca_in: float = 5.10e3


def reaction_rates(
    temperature_c: float,
    params: SahlodinVanDeVusseParameters | None = None,
) -> tuple[float, float]:
    """Return ``k1(T)`` and ``k2(T)`` from equation (12)."""
    p = params if params is not None else SahlodinVanDeVusseParameters()
    absolute_t = float(temperature_c) + 273.15
    k1 = p.k01 * np.exp(-p.e1 / absolute_t)
    k2 = p.k02 * np.exp(-p.e2 / absolute_t)
    return float(k1), float(k2)


def outlet_flow(
    volume: float,
    params: SahlodinVanDeVusseParameters | None = None,
) -> float:
    """Return ``Fout = eta * sqrt(V)`` from equation (12)."""
    p = params if params is not None else SahlodinVanDeVusseParameters()
    return float(p.eta * np.sqrt(max(float(volume), 0.0)))


def heat_release(
    x: ArrayLike,
    params: SahlodinVanDeVusseParameters | None = None,
) -> float:
    """Return the heat-release term ``h(t)`` from equation (12)."""
    p = params if params is not None else SahlodinVanDeVusseParameters()
    ca, cb, _cc, _cd, temp, _tc, _vol = np.asarray(x, dtype=float)
    k1, k2 = reaction_rates(float(temp), params=p)
    enthalpy_term = (
        k1 * (ca * p.delta_h_ab + cb * p.delta_h_bc)
        + k2 * ca * ca * p.delta_h_ad
    )
    return float(-p.delta * enthalpy_term)


def rhs(
    x: ArrayLike,
    u: ArrayLike,
    params: SahlodinVanDeVusseParameters | None = None,
) -> np.ndarray:
    """Return equation-(12) state derivatives for one state/control pair."""
    p = params if params is not None else SahlodinVanDeVusseParameters()
    ca, cb, cc, cd, temp, tc, vol = np.asarray(x, dtype=float)
    fin, pc = np.asarray(u, dtype=float)
    safe_vol = max(float(vol), 1e-9)
    k1, k2 = reaction_rates(float(temp), params=p)
    h = heat_release(x, params=p)
    fout = outlet_flow(float(vol), params=p)
    return np.array(
        [
            -k1 * ca - k2 * ca * ca + ((p.ca_in - ca) / safe_vol) * fin,
            k1 * (ca - cb) - (cb * fin) / safe_vol,
            k1 * cb - (cc * fin) / safe_vol,
            0.5 * k2 * ca * ca - (cd * fin) / safe_vol,
            h + p.alpha * (tc - temp) + fin * (p.tin - temp) / safe_vol,
            p.beta * (temp - tc) - p.gamma * pc,
            fin - fout,
        ],
        dtype=float,
    )


def step_state(
    x: ArrayLike,
    u: ArrayLike,
    dt_h: float = SAHLODIN_DECISION_DT_H,
    params: SahlodinVanDeVusseParameters | None = None,
    *,
    n_substeps: int = 1,
    clip: bool = True,
) -> np.ndarray:
    """Advance the Sahlodin plant one interval using RK4 integration."""
    if n_substeps < 1:
        raise ValueError(f"n_substeps must be >= 1, got {n_substeps}.")
    state = np.asarray(x, dtype=float).ravel().copy()
    control = np.asarray(u, dtype=float).ravel()
    if state.shape != (len(SAHLODIN_STATE_NAMES),):
        raise ValueError(
            f"x must have {len(SAHLODIN_STATE_NAMES)} entries, got shape {state.shape}."
        )
    if control.shape != (len(SAHLODIN_CONTROL_NAMES),):
        raise ValueError(
            f"u must have {len(SAHLODIN_CONTROL_NAMES)} entries, got shape {control.shape}."
        )
    dt = float(dt_h) / int(n_substeps)
    p = params if params is not None else SahlodinVanDeVusseParameters()
    for _ in range(int(n_substeps)):
        k1 = rhs(state, control, params=p)
        k2 = rhs(state + 0.5 * dt * k1, control, params=p)
        k3 = rhs(state + 0.5 * dt * k2, control, params=p)
        k4 = rhs(state + dt * k3, control, params=p)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if clip:
            state = np.clip(state, SAHLODIN_STATE_BOUNDS[:, 0], SAHLODIN_STATE_BOUNDS[:, 1])
    return state


def simulate_policy(
    control_profile: ArrayLike,
    params: SahlodinVanDeVusseParameters | None = None,
    x0: ArrayLike | None = None,
    *,
    dt_h: float = SAHLODIN_DECISION_DT_H,
    n_substeps: int = 1,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a piecewise-constant ``[Fin, Pc]`` profile on the paper plant."""
    profile = np.asarray(control_profile, dtype=float)
    if profile.ndim != 2 or profile.shape[1] != len(SAHLODIN_CONTROL_NAMES):
        raise ValueError(
            "control_profile must have shape (n_steps, 2) with columns [Fin, Pc]."
        )
    state = (
        np.asarray(x0, dtype=float).ravel()
        if x0 is not None
        else SAHLODIN_INITIAL_STATE.copy()
    )
    if state.shape != (len(SAHLODIN_STATE_NAMES),):
        raise ValueError(
            f"x0 must have {len(SAHLODIN_STATE_NAMES)} entries, got shape {state.shape}."
        )
    states = [state.copy()]
    p = params if params is not None else SahlodinVanDeVusseParameters()
    for control in profile:
        state = step_state(
            state,
            control,
            dt_h=dt_h,
            params=p,
            n_substeps=n_substeps,
            clip=clip,
        )
        states.append(state.copy())
    times = np.arange(len(states), dtype=float) * float(dt_h)
    return times, np.asarray(states, dtype=float)
