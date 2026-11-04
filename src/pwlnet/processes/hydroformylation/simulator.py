"""Simulation helpers for the hydroformylation semibatch reactor."""

from __future__ import annotations

import numpy as np

from pwlnet.processes.hydroformylation.model import (
    CONTROL_DT_MIN,
    HydroformylationParameters,
    INITIAL_STATE,
    MEASUREMENT_DELAY_MIN,
    MEASUREMENT_INTERVAL_MIN,
    MILP_15S_DECISION_DT_MIN,
    SAMPLES_PER_CONTROL,
    SAMPLE_TIME_MIN,
    STATE_BOUNDS,
    STATE_NAMES,
    T_FINAL,
    rhs,
)

__all__ = [
    "_rk4_step",
    "delayed_measurements",
    "integrate_interval",
    "simulate_policy",
    "simulate_sample_policy",
    "step_state",
]


def integrate_interval(
    x: np.ndarray,
    T: float,
    u: float,
    dt_min: float,
    p: HydroformylationParameters,
    *,
    t_eval: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the stiff kinetic model over one constant-control interval."""
    from scipy.integrate import solve_ivp

    x0 = np.maximum(np.asarray(x, dtype=float), STATE_BOUNDS[:, 0])

    def ode(t, y):
        return rhs(np.maximum(y, STATE_BOUNDS[:, 0]), float(T), float(u), p)

    sol = solve_ivp(
        ode,
        (0.0, float(dt_min)),
        x0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Stiff plant integration failed: {sol.message}")
    times = sol.t if t_eval is not None else np.array([float(dt_min)])
    states = sol.y.T if t_eval is not None else sol.y[:, [-1]].T
    return times, np.maximum(states, STATE_BOUNDS[:, 0])


def step_state(
    x: np.ndarray,
    T: float,
    u: float,
    dt_min: float,
    p: HydroformylationParameters,
) -> np.ndarray:
    """Advance the plant state by one constant-control interval."""
    _, states = integrate_interval(x, T, u, dt_min, p)
    return states[-1]


def _rk4_step(
    x: np.ndarray,
    T: float,
    u: float,
    dt_min: float,
    p: HydroformylationParameters,
) -> np.ndarray:
    """Backward-compatible alias for the notebook's former integration helper."""
    return step_state(x, T, u, dt_min, p)


def simulate_policy(
    T_profile,
    u_profile,
    p: HydroformylationParameters = HydroformylationParameters(),
    x0=INITIAL_STATE,
    *,
    dense: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a coarse 5-minute control policy against the true plant."""
    coarse_states = [np.asarray(x0, dtype=float)]
    coarse_times = [0.0]
    dense_states = [np.asarray(x0, dtype=float)]
    dense_times = [0.0]
    x = np.asarray(x0, dtype=float).copy()
    for k, (T, u) in enumerate(zip(T_profile, u_profile)):
        if dense:
            eval_times = np.arange(1, SAMPLES_PER_CONTROL + 1, dtype=float) * SAMPLE_TIME_MIN
            local_times, local_states = integrate_interval(
                x, float(T), float(u), CONTROL_DT_MIN, p, t_eval=eval_times
            )
            base_time = k * CONTROL_DT_MIN
            dense_times.extend((base_time + local_times).tolist())
            dense_states.extend(local_states)
            x = local_states[-1].copy()
        else:
            x = step_state(x, float(T), float(u), CONTROL_DT_MIN, p)
        coarse_states.append(x.copy())
        coarse_times.append((k + 1) * CONTROL_DT_MIN)
    if dense:
        return np.asarray(dense_times), np.asarray(dense_states)
    return np.asarray(coarse_times), np.asarray(coarse_states)


def simulate_sample_policy(
    T_profile,
    u_profile,
    p: HydroformylationParameters = HydroformylationParameters(),
    x0=INITIAL_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a 15-second control policy against the true plant."""
    states = [np.asarray(x0, dtype=float).copy()]
    x = np.asarray(x0, dtype=float).copy()
    for T, u in zip(T_profile, u_profile):
        x = step_state(x, float(T), float(u), MILP_15S_DECISION_DT_MIN, p)
        states.append(x.copy())
    times = np.arange(len(states)) * MILP_15S_DECISION_DT_MIN
    return times, np.asarray(states)


def delayed_measurements(times: np.ndarray, states: np.ndarray):
    """Return sampled and delayed concentration/pressure measurements."""
    sample_times = np.arange(0.0, T_FINAL + 1e-9, MEASUREMENT_INTERVAL_MIN)
    available_times = sample_times + MEASUREMENT_DELAY_MIN
    keep = available_times <= T_FINAL + 1e-9
    sample_times = sample_times[keep]
    available_times = available_times[keep]
    measured = np.column_stack(
        [np.interp(sample_times, times, states[:, j]) for j in range(len(STATE_NAMES))]
    )
    return sample_times, available_times, measured
