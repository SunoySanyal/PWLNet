"""Simulation and dataset helpers for the Van de Vusse CSTR benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from pwlnet.processes.van_de_vusse.model import (
    DECISION_DT_MIN,
    INPUT_NAMES,
    OUTPUT_NAMES,
    STATE_BOUNDS,
    VanDeVusseParameters,
    product_concentration,
    rhs,
    steady_state,
    steady_state_residuals,
)
from pwlnet.utils.typing import ArrayLike
from pwlnet.utils.validation import as_2d_array

__all__ = [
    "VanDeVusseSteadyState",
    "find_max_product_dilution",
    "generate_dataset",
    "simulate_points",
    "simulate_policy",
    "simulate_step_response",
    "solve_steady_state",
    "step_state",
]


@dataclass(frozen=True)
class VanDeVusseSteadyState:
    """Solved Van de Vusse steady state at one dilution rate."""

    dilution_rate: float
    x: np.ndarray
    residual_inf_norm: float

    @property
    def ca(self) -> float:
        """Reactant A concentration [mol/L]."""
        return float(self.x[OUTPUT_NAMES.index("CA")])

    @property
    def cb(self) -> float:
        """Desired product B concentration [mol/L]."""
        return float(self.x[OUTPUT_NAMES.index("CB")])

    def to_dict(self) -> dict[str, float]:
        """Return scalar fields keyed by input/output names."""
        row = {"D": self.dilution_rate}
        row.update({name: float(value) for name, value in zip(OUTPUT_NAMES, self.x)})
        row["residual_inf_norm"] = self.residual_inf_norm
        return row


def solve_steady_state(
    dilution_rate: float,
    params: VanDeVusseParameters | None = None,
) -> VanDeVusseSteadyState:
    """Evaluate the analytic steady state at ``D = F / V``."""
    x = steady_state(float(dilution_rate), params=params)
    residual = steady_state_residuals(x, float(dilution_rate), params=params)
    return VanDeVusseSteadyState(
        dilution_rate=float(dilution_rate),
        x=np.asarray(x, dtype=float),
        residual_inf_norm=float(np.max(np.abs(residual))),
    )


def simulate_points(
    inputs: ArrayLike,
    params: VanDeVusseParameters | None = None,
) -> np.ndarray:
    """Simulate many steady-state dilution-rate points."""
    X = as_2d_array(inputs, name="inputs")
    if X.shape[1] != 1:
        raise ValueError(f"inputs must have columns {INPUT_NAMES}, got shape {X.shape}.")
    return steady_state(X[:, 0], params=params)


def _sample_inputs(
    n_samples: int,
    method: Literal["lhs", "random", "grid"],
    params: VanDeVusseParameters,
    random_state: int | None,
) -> np.ndarray:
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    low, high = params.dilution_bounds
    method_name = str(method).lower()
    if method_name == "grid":
        return np.linspace(low, high, int(n_samples), dtype=float).reshape(-1, 1)
    if method_name == "lhs":
        rng = np.random.default_rng(random_state)
        edges = np.linspace(0.0, 1.0, int(n_samples) + 1)
        unit = edges[:-1] + rng.uniform(0.0, 1.0 / int(n_samples), size=int(n_samples))
        rng.shuffle(unit)
        return (low + unit * (high - low)).reshape(-1, 1)
    if method_name == "random":
        rng = np.random.default_rng(random_state)
        return rng.uniform(low, high, size=(int(n_samples), 1))
    raise ValueError("method must be one of 'lhs', 'random', or 'grid'.")


def generate_dataset(
    n_samples: int = 200,
    method: Literal["lhs", "random", "grid"] = "lhs",
    random_state: int | None = 0,
    params: VanDeVusseParameters | None = None,
    as_frame: bool = True,
) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
    """Generate a Van de Vusse steady-state supervised dataset."""
    p = params if params is not None else VanDeVusseParameters()
    inputs = _sample_inputs(
        n_samples=int(n_samples), method=method, params=p, random_state=random_state
    )
    outputs = simulate_points(inputs, params=p)
    residual_norms = np.array(
        [
            np.max(np.abs(steady_state_residuals(x, float(d), params=p)))
            for d, x in zip(inputs[:, 0], outputs)
        ],
        dtype=float,
    )
    frame = pd.DataFrame(
        np.column_stack([inputs, outputs, residual_norms]),
        columns=INPUT_NAMES + OUTPUT_NAMES + ["residual_inf_norm"],
    )
    if as_frame:
        return frame
    return frame[INPUT_NAMES].to_numpy(dtype=float), frame[OUTPUT_NAMES].to_numpy(dtype=float)


def find_max_product_dilution(
    params: VanDeVusseParameters | None = None,
    bounds: tuple[float, float] | None = None,
) -> VanDeVusseSteadyState:
    """Find the steady-state dilution rate that maximizes ``CB``."""
    p = params if params is not None else VanDeVusseParameters()
    lo, hi = (bounds if bounds is not None else p.dilution_bounds)
    a = float(lo)
    b = float(hi)
    if a <= 0.0 or b <= a:
        raise ValueError(f"bounds must satisfy 0 < low < high, got {(lo, hi)!r}.")

    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc = float(product_concentration(c, params=p))
    fd = float(product_concentration(d, params=p))
    for _ in range(200):
        if abs(b - a) <= 1e-12:
            break
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + inv_phi * (b - a)
            fd = float(product_concentration(d, params=p))
        else:
            b = d
            d = c
            fd = fc
            c = b - inv_phi * (b - a)
            fc = float(product_concentration(c, params=p))
    return solve_steady_state(0.5 * (a + b), params=p)


def step_state(
    x: ArrayLike,
    dilution_rate: float,
    dt_min: float = DECISION_DT_MIN,
    params: VanDeVusseParameters | None = None,
    *,
    n_substeps: int = 1,
) -> np.ndarray:
    """Advance the dynamic plant with constant dilution rate over one interval."""
    if n_substeps < 1:
        raise ValueError(f"n_substeps must be >= 1, got {n_substeps}.")
    p = params if params is not None else VanDeVusseParameters()
    state = np.asarray(x, dtype=float).ravel().copy()
    if state.shape != (len(OUTPUT_NAMES),):
        raise ValueError(
            f"x must have {len(OUTPUT_NAMES)} entries, got shape {state.shape}."
        )
    dt = float(dt_min) / int(n_substeps)
    for _ in range(int(n_substeps)):
        k1 = rhs(state, float(dilution_rate), params=p)
        k2 = rhs(state + 0.5 * dt * k1, float(dilution_rate), params=p)
        k3 = rhs(state + 0.5 * dt * k2, float(dilution_rate), params=p)
        k4 = rhs(state + dt * k3, float(dilution_rate), params=p)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        state = np.clip(state, STATE_BOUNDS[:, 0], STATE_BOUNDS[:, 1])
    return state


def simulate_policy(
    dilution_profile: ArrayLike,
    params: VanDeVusseParameters | None = None,
    x0: ArrayLike | None = None,
    *,
    dt_min: float = DECISION_DT_MIN,
    n_substeps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a piecewise-constant dilution-rate policy on the true plant."""
    profile = np.asarray(dilution_profile, dtype=float).ravel()
    if profile.ndim != 1:
        raise ValueError("dilution_profile must be one-dimensional.")
    p = params if params is not None else VanDeVusseParameters()
    x = (
        np.asarray(x0, dtype=float).ravel()
        if x0 is not None
        else steady_state(float(profile[0]), params=p)
    )
    if x.shape != (len(OUTPUT_NAMES),):
        raise ValueError(
            f"x0 must have {len(OUTPUT_NAMES)} entries, got shape {x.shape}."
        )
    states = [np.clip(x.copy(), STATE_BOUNDS[:, 0], STATE_BOUNDS[:, 1])]
    for d in profile:
        x = step_state(x, float(d), dt_min=dt_min, params=p, n_substeps=n_substeps)
        states.append(x.copy())
    times = np.arange(len(states), dtype=float) * float(dt_min)
    return times, np.asarray(states)


def simulate_step_response(
    initial_dilution_rate: float,
    final_dilution_rate: float,
    t_final: float = 10.0,
    n_eval: int = 201,
    params: VanDeVusseParameters | None = None,
    x0: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a dilution-rate step using the nonlinear dynamic model.

    By default the initial state is the steady state at
    ``initial_dilution_rate``.  The returned time vector is in minutes and
    the state columns are ordered as :data:`OUTPUT_NAMES`.
    """
    p = params if params is not None else VanDeVusseParameters()
    initial = (
        np.asarray(x0, dtype=float).ravel()
        if x0 is not None
        else steady_state(float(initial_dilution_rate), params=p)
    )
    if initial.shape != (len(OUTPUT_NAMES),):
        raise ValueError(
            f"x0 must have {len(OUTPUT_NAMES)} entries, got shape {initial.shape}."
        )
    profile = np.full(int(n_eval) - 1, float(final_dilution_rate), dtype=float)
    dt = float(t_final) / (int(n_eval) - 1)
    return simulate_policy(profile, params=p, x0=initial, dt_min=dt)
