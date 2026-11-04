"""Direct-shooting optimal control for the Van de Vusse plant."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from pwlnet.processes.van_de_vusse.model import (
    CONTROL_BOUNDS,
    DECISION_DT_MIN,
    OPERATING_POINT_1,
    OUTPUT_NAMES,
    T_FINAL_MIN,
    VanDeVusseParameters,
    steady_state,
)
from pwlnet.processes.van_de_vusse.simulator import simulate_policy
from pwlnet.utils.typing import ArrayLike, PathLike

__all__ = [
    "direct_shooting_random_starts",
    "save_optimal_control_solution",
    "solve_direct_shooting_optimal_control",
]

_CB_INDEX = OUTPUT_NAMES.index("CB")


def _objective_value(
    states: np.ndarray,
    controls: np.ndarray,
    *,
    objective: Literal["terminal_cb", "integral_cb", "terminal_plus_integral"],
    dt_min: float,
    move_penalty: float,
    d_prev: float | None,
) -> float:
    terminal = float(states[-1, _CB_INDEX])
    integral = float(np.trapezoid(states[:, _CB_INDEX], dx=float(dt_min)))
    if objective == "terminal_cb":
        value = terminal
    elif objective == "integral_cb":
        value = integral
    elif objective == "terminal_plus_integral":
        horizon = max(float(dt_min) * float(len(states) - 1), 1e-12)
        value = terminal + integral / horizon
    else:
        raise ValueError(
            "objective must be 'terminal_cb', 'integral_cb', or "
            f"'terminal_plus_integral', got {objective!r}."
        )
    if move_penalty > 0.0:
        if d_prev is None:
            diffs = np.diff(controls)
        else:
            diffs = np.diff(np.r_[float(d_prev), controls])
        value -= float(move_penalty) * float(np.sum(diffs * diffs))
    return value


def solve_direct_shooting_optimal_control(
    *,
    dt_min: float = 1.0,
    horizon_min: float = T_FINAL_MIN,
    x0: ArrayLike | None = None,
    params: VanDeVusseParameters | None = None,
    objective: Literal["terminal_cb", "integral_cb", "terminal_plus_integral"] = "terminal_cb",
    move_penalty: float = 0.0,
    d_prev: float | None = None,
    initial_profile: ArrayLike | None = None,
    maxiter: int = 300,
    ftol: float = 1e-10,
    n_substeps_per_interval: int = 8,
) -> dict[str, Any]:
    """Solve one direct-shooting open-loop control problem.

    The decision variables are a piecewise-constant dilution-rate profile
    ``D[0], ..., D[n_steps-1]``. The true nonlinear Van de Vusse ODE is
    integrated for every trial profile; no surrogate model is involved.
    """
    from scipy.optimize import minimize

    if dt_min <= 0.0:
        raise ValueError(f"dt_min must be positive, got {dt_min}.")
    if horizon_min <= 0.0:
        raise ValueError(f"horizon_min must be positive, got {horizon_min}.")
    p = params if params is not None else VanDeVusseParameters()
    n_steps = int(round(float(horizon_min) / float(dt_min)))
    if n_steps < 1:
        raise ValueError(
            f"horizon_min={horizon_min} and dt_min={dt_min} imply no control intervals."
        )
    actual_horizon = n_steps * float(dt_min)
    if x0 is None:
        x_initial = steady_state(OPERATING_POINT_1["D"], params=p)
    else:
        x_initial = np.asarray(x0, dtype=float).ravel()
    if x_initial.shape != (len(OUTPUT_NAMES),):
        raise ValueError(f"x0 must have {len(OUTPUT_NAMES)} entries, got {x_initial.shape}.")

    lo, hi = p.dilution_bounds
    if initial_profile is None:
        profile0 = np.full(n_steps, 0.5 * (lo + hi), dtype=float)
    else:
        profile0 = np.asarray(initial_profile, dtype=float).ravel()
        if profile0.shape != (n_steps,):
            raise ValueError(
                f"initial_profile must have {n_steps} entries, got {profile0.shape}."
            )
        profile0 = np.clip(profile0, lo, hi)

    sim_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def simulate(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = np.asarray(profile, dtype=float).tobytes()
        cached = sim_cache.get(key)
        if cached is None:
            cached = simulate_policy(
                profile,
                params=p,
                x0=x_initial,
                dt_min=float(dt_min),
                n_substeps=max(1, int(n_substeps_per_interval)),
            )
            if len(sim_cache) > 20000:
                sim_cache.clear()
            sim_cache[key] = cached
        return cached

    def merit(profile: np.ndarray) -> float:
        _times, states = simulate(np.clip(profile, lo, hi))
        value = _objective_value(
            states,
            np.clip(profile, lo, hi),
            objective=objective,
            dt_min=float(dt_min),
            move_penalty=move_penalty,
            d_prev=d_prev,
        )
        return -value

    result = minimize(
        merit,
        profile0,
        method="SLSQP",
        bounds=[(lo, hi)] * n_steps,
        options={"maxiter": int(maxiter), "ftol": float(ftol)},
    )

    profile = np.clip(np.asarray(result.x, dtype=float), lo, hi)
    times, states = simulate(profile)
    cb = states[:, _CB_INDEX]
    return {
        "D": profile,
        "states": states,
        "times": times,
        "dt_min": float(dt_min),
        "horizon_min": float(actual_horizon),
        "objective": objective,
        "objective_value": _objective_value(
            states,
            profile,
            objective=objective,
            dt_min=float(dt_min),
            move_penalty=move_penalty,
            d_prev=d_prev,
        ),
        "terminal_CB": float(cb[-1]),
        "integral_CB": float(np.trapezoid(cb, x=times)),
        "max_CB": float(np.max(cb)),
        "success": bool(result.success),
        "message": str(result.message),
        "n_iter": int(getattr(result, "nit", -1)),
        "n_func_eval": int(getattr(result, "nfev", -1)),
    }


def direct_shooting_random_starts(
    *,
    n_starts: int = 12,
    dt_min: float = 1.0,
    horizon_min: float = T_FINAL_MIN,
    x0: ArrayLike | None = None,
    params: VanDeVusseParameters | None = None,
    objective: Literal["terminal_cb", "integral_cb", "terminal_plus_integral"] = "terminal_cb",
    move_penalty: float = 0.0,
    seed: int | None = 2479085,
    maxiter: int = 300,
    ftol: float = 1e-10,
    n_substeps_per_interval: int = 8,
) -> dict[str, Any]:
    """Run random multistart direct shooting and return the best solution."""
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}.")
    p = params if params is not None else VanDeVusseParameters()
    rng = np.random.default_rng(seed)
    n_steps = int(round(float(horizon_min) / float(dt_min)))
    lo, hi = p.dilution_bounds
    starts: list[dict[str, Any]] = []

    deterministic = [
        np.full(n_steps, 0.5 * (lo + hi), dtype=float),
        np.full(n_steps, lo, dtype=float),
        np.full(n_steps, hi, dtype=float),
    ]
    for i in range(int(n_starts)):
        if i < len(deterministic):
            initial = deterministic[i]
        else:
            initial = rng.uniform(lo, hi, size=n_steps)
        sol = solve_direct_shooting_optimal_control(
            dt_min=dt_min,
            horizon_min=horizon_min,
            x0=x0,
            params=p,
            objective=objective,
            move_penalty=move_penalty,
            initial_profile=initial,
            maxiter=maxiter,
            ftol=ftol,
            n_substeps_per_interval=n_substeps_per_interval,
        )
        sol["start_index"] = i
        starts.append(sol)

    best = max(starts, key=lambda item: float(item["objective_value"]))
    start_summary = np.array(
        [
            (
                float(item["objective_value"]),
                float(item["terminal_CB"]),
                float(item["integral_CB"]),
                float(item["success"]),
                float(item["n_iter"]),
            )
            for item in starts
        ],
        dtype=float,
    )
    return {
        **best,
        "n_starts": int(n_starts),
        "seed": -1 if seed is None else int(seed),
        "start_summary": start_summary,
    }


def save_optimal_control_solution(solution: dict[str, Any], path: PathLike) -> Path:
    """Save a Van de Vusse optimal-control solution as a NumPy ``.npz`` file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        D=np.asarray(solution["D"], dtype=float),
        states=np.asarray(solution["states"], dtype=float),
        times=np.asarray(solution["times"], dtype=float),
        dt_min=float(solution["dt_min"]),
        horizon_min=float(solution["horizon_min"]),
        objective=str(solution["objective"]),
        objective_value=float(solution["objective_value"]),
        terminal_CB=float(solution["terminal_CB"]),
        integral_CB=float(solution["integral_CB"]),
        max_CB=float(solution["max_CB"]),
        success=bool(solution["success"]),
        message=str(solution["message"]),
        n_iter=int(solution["n_iter"]),
        n_func_eval=int(solution["n_func_eval"]),
        n_starts=int(solution.get("n_starts", 1)),
        seed=int(solution.get("seed", -1)),
        start_summary=np.asarray(solution.get("start_summary", []), dtype=float),
        output_names=np.asarray(OUTPUT_NAMES),
    )
    return out
