"""Dataset generation for Van de Vusse transition surrogates."""

from __future__ import annotations

from typing import Literal

import numpy as np

from pwlnet.processes.van_de_vusse.model import (
    CONTROL_BOUNDS,
    DECISION_DT_MIN,
    STATE_BOUNDS,
    T_FINAL_MIN,
    VanDeVusseParameters,
    steady_state,
)
from pwlnet.processes.van_de_vusse.simulator import (
    find_max_product_dilution,
    simulate_policy,
)

__all__ = [
    "make_transition_dataset",
    "random_control_profiles",
]


def random_control_profiles(
    rng: np.random.Generator,
    n_steps: int | None = None,
    *,
    params: VanDeVusseParameters | None = None,
    sigma_d: float = 0.25,
    jump_probability: float = 0.25,
    rate_limit: float = 0.35,
) -> np.ndarray:
    """Generate one bounded random-walk dilution-rate profile.

    The profile is centered around the exact steady-state ``CB`` optimum
    but occasionally receives larger jumps, so the transition data covers
    both local MPC behavior and inverse-response transients away from the
    optimum.
    """
    p = params if params is not None else VanDeVusseParameters()
    steps = int(round(T_FINAL_MIN / DECISION_DT_MIN)) if n_steps is None else int(n_steps)
    if steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    low, high = p.dilution_bounds
    d_nom = find_max_product_dilution(params=p).dilution_rate
    d = float(np.clip(d_nom + rng.normal(0.0, sigma_d), low, high))
    profile = np.empty(steps, dtype=float)
    for k in range(steps):
        if k > 0:
            if rng.random() < jump_probability:
                d += rng.normal(0.0, 2.0 * sigma_d)
            else:
                d += rng.normal(0.0, sigma_d)
            d = float(np.clip(d, profile[k - 1] - rate_limit, profile[k - 1] + rate_limit))
            d = float(np.clip(d, low, high))
        profile[k] = d
    return np.clip(profile, CONTROL_BOUNDS[0, 0], CONTROL_BOUNDS[0, 1])


def _initial_state(
    profile: np.ndarray,
    mode: Literal["profile", "inverse_response", "random"],
    rng: np.random.Generator,
    params: VanDeVusseParameters,
) -> np.ndarray:
    if mode == "profile":
        return steady_state(float(profile[0]), params=params)
    if mode == "inverse_response":
        return steady_state(4.0 / 7.0, params=params)
    if mode == "random":
        d0 = float(rng.uniform(params.dilution_bounds[0], params.dilution_bounds[1]))
        return steady_state(d0, params=params)
    raise ValueError("initial_state must be 'profile', 'inverse_response', or 'random'.")


def make_transition_dataset(
    n_trajectories: int = 160,
    rng: np.random.Generator | None = None,
    *,
    n_steps: int | None = None,
    params: VanDeVusseParameters | None = None,
    dt_min: float = DECISION_DT_MIN,
    target: Literal["delta", "next"] = "delta",
    initial_state: Literal["profile", "inverse_response", "random"] = "random",
    n_substeps: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one-step transition rows for Van de Vusse NMPC.

    Inputs are ordered ``[CA, CB, CC, CD, D]``. Targets are either the
    state change ``x[k+1] - x[k]`` (default) or the next state itself.
    """
    if n_trajectories < 1:
        raise ValueError(f"n_trajectories must be >= 1, got {n_trajectories}.")
    if target not in {"delta", "next"}:
        raise ValueError("target must be 'delta' or 'next'.")
    rng = np.random.default_rng(2479085) if rng is None else rng
    p = params if params is not None else VanDeVusseParameters()
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    for _ in range(int(n_trajectories)):
        profile = random_control_profiles(rng, n_steps=n_steps, params=p)
        x0 = _initial_state(profile, initial_state, rng, p)
        _, states = simulate_policy(
            profile,
            params=p,
            x0=x0,
            dt_min=dt_min,
            n_substeps=n_substeps,
        )
        for k, d in enumerate(profile):
            x_curr = np.clip(states[k], STATE_BOUNDS[:, 0], STATE_BOUNDS[:, 1])
            x_next = np.clip(states[k + 1], STATE_BOUNDS[:, 0], STATE_BOUNDS[:, 1])
            rows_x.append(np.r_[x_curr, float(d)])
            rows_y.append(x_next - x_curr if target == "delta" else x_next)
    return np.asarray(rows_x, dtype=float), np.asarray(rows_y, dtype=float)
