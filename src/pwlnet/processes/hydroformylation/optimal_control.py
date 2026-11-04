"""Model-based open-loop optimal control for the hydroformylation plant.

The optimal control profile ``(T*, u*)`` is computed by direct single-shooting
dynamic optimization of the first-principles stiff kinetic ODE — no data-driven
surrogate is involved. It is used as the nominal trajectory around which training
data for the transition surrogates is perturbed, mirroring the source papers'
"noise around the optimal trajectory" experiments rather than purely random
control profiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pwlnet.processes.hydroformylation.model import (
    CONTROL_BOUNDS,
    CONTROL_DT_MIN,
    DT_RATE_LIMIT,
    IDX,
    INITIAL_STATE,
    N_STEPS,
    HydroformylationParameters,
)
from pwlnet.processes.hydroformylation.simulator import step_state

__all__ = [
    "REFERENCE_15S_PATH",
    "REFERENCE_5MIN_PATH",
    "interpolate_control_profile",
    "optimal_control_profiles",
    "reference_control_profiles",
    "solve_open_loop_optimal_control",
]

#: Verified 15-second open-loop optimum for the nominal plant, produced by
#: ``experiments/Hydroformulation_ms_benchmark_solver/solve_15s_optimal.py``.
#: Reassign this module attribute to point at a different reference.
REFERENCE_15S_PATH = Path(__file__).resolve().parents[4] / "datasets" / "ms_15s_optimal.npz"

#: Verified 5-minute open-loop optimum (free ``u``, hard ``p_total <= 20`` at the
#: interval nodes), produced by ``solve_5min_optimal.py`` in the same directory.
#: A 5-minute feed cannot hold the pressure constraint *within* an interval, so
#: this is a genuinely different trajectory from :data:`REFERENCE_15S_PATH` and
#: must not be obtained by resampling it -- see :func:`reference_control_profiles`.
REFERENCE_5MIN_PATH = Path(__file__).resolve().parents[4] / "datasets" / "ms_5min_optimal.npz"

_REFERENCE_CACHE: dict = {}


def reference_control_profiles(target_times_min, *, path=None):
    """Return the saved 15 s benchmark optimum sampled at ``target_times_min``.

    :func:`optimal_control_profiles` re-solves the open-loop problem from
    scratch, which takes roughly half an hour and is paid again by every fresh
    Python process (the solve cache is in-process only). For the 15 s grid the
    optimum is already known and stored, so load it instead.

    The stored profile is on the 15 s grid; ``u`` there is a spiky
    pressure-holding sequence that starts near 0.49 mol/min and decays by more
    than two orders of magnitude. Resampling it onto a *coarser* grid is
    therefore meaningless -- a 5-minute sample of the opening spike, held for
    five minutes, over-feeds the reactor by a wide margin. Use this only for
    grids at least as fine as the reference.
    """
    ref = Path(path) if path is not None else REFERENCE_15S_PATH
    key = str(ref)
    cached = _REFERENCE_CACHE.get(key)
    if cached is None:
        if not ref.exists():
            raise FileNotFoundError(
                f"reference optimum not found at {ref}. Regenerate it with the "
                "matching script in experiments/Hydroformulation_ms_benchmark_solver "
                "(solve_15s_optimal.py / solve_5min_optimal.py), or point "
                "pwlnet.processes.hydroformylation.optimal_control.REFERENCE_15S_PATH "
                "/ REFERENCE_5MIN_PATH at an existing file."
            )
        with np.load(ref) as data:
            cached = (data["times"][:-1].copy(), data["T"].copy(), data["u"].copy())
        _REFERENCE_CACHE[key] = cached
    ref_t, ref_T, ref_u = cached
    target = np.asarray(target_times_min, dtype=float)
    step = float(np.min(np.diff(target))) if target.size > 1 else float(ref_t[1] - ref_t[0])
    ref_step = float(ref_t[1] - ref_t[0])
    if step < ref_step - 1e-9:
        raise ValueError(
            f"requested grid step {step:g} min is finer than the reference "
            f"{ref_step:g} min; the reference cannot be upsampled faithfully."
        )
    return np.interp(target, ref_t, ref_T), np.interp(target, ref_t, ref_u)


def solve_open_loop_optimal_control(
    *,
    dt_min: float = CONTROL_DT_MIN,
    n_steps: int = N_STEPS,
    x0=INITIAL_STATE,
    plant: HydroformylationParameters = HydroformylationParameters(),
    terminal_cap: float = 0.8,
    w_terminal: float = 500.0,
    max_pressure: float = 20.0,
    T_init=None,
    u_init=None,
    maxiter: int = 400,
    ftol: float = 1e-12,
) -> dict:
    """Direct single-shooting optimal control on the true kinetic ODE.

    Maximizes terminal ``nC13al`` subject to a **hard** maximum total-pressure
    path constraint (``p_total <= max_pressure``), the control bounds, and the
    0.35 K / 15 s temperature-rate limit. No NMPC pressure backoff is applied:
    the open-loop nominal is disturbance-free, so the plain 20 bar limit of the
    original problem is used directly (the backoff only exists to give the
    closed-loop controller margin against disturbances). A soft terminal cap
    (``terminal_cap``) is kept on the product. The stiff plant ODE is integrated
    directly via :func:`step_state`; no surrogate model is involved.

    Returns a dict with the optimal ``T``/``u`` profiles, the resulting true
    state trajectory, the terminal product, and the optimizer status.
    """
    from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

    T_lo, T_hi = CONTROL_BOUNDS[0]
    u_lo, u_hi = CONTROL_BOUNDS[1]
    x0 = np.asarray(x0, dtype=float)

    sim_cache: dict[bytes, np.ndarray] = {}

    def simulate(z: np.ndarray) -> np.ndarray:
        key = z.tobytes()
        traj = sim_cache.get(key)
        if traj is None:
            T_prof, u_prof = z[:n_steps], z[n_steps:]
            x = x0.copy()
            rows = [x.copy()]
            for i in range(n_steps):
                x = step_state(x, float(T_prof[i]), float(u_prof[i]), dt_min, plant)
                rows.append(x.copy())
            if len(sim_cache) > 40000:
                sim_cache.clear()
            traj = np.asarray(rows)
            sim_cache[key] = traj
        return traj

    def merit(z: np.ndarray) -> float:
        traj = simulate(z)
        c_final = traj[-1, IDX["nC13al"]]
        return -c_final + w_terminal * max(c_final - terminal_cap, 0.0) ** 2

    def total_pressure(z: np.ndarray) -> np.ndarray:
        traj = simulate(z)
        return traj[1:, IDX["p_H2"]] + traj[1:, IDX["p_CO"]]

    # Hard maximum total-pressure path constraint p_total <= max_pressure at the
    # end of every control interval. This is the original problem's 20 bar limit;
    # no NMPC backoff is used because the open-loop nominal has no disturbances.
    pressure_con = NonlinearConstraint(total_pressure, -np.inf, max_pressure)

    # temperature rate-limit |T_k - T_{k-1}| <= max_jump as a banded linear constraint
    rate_A = np.zeros((n_steps - 1, 2 * n_steps))
    for i in range(1, n_steps):
        rate_A[i - 1, i] = 1.0
        rate_A[i - 1, i - 1] = -1.0
    max_jump = DT_RATE_LIMIT * dt_min

    if T_init is None:
        T_init = np.full(n_steps, 378.15)
    if u_init is None:
        u_init = np.full(n_steps, 0.01)
    z0 = np.r_[np.asarray(T_init, dtype=float), np.asarray(u_init, dtype=float)]

    res = minimize(
        merit,
        z0,
        method="SLSQP",
        bounds=[(T_lo, T_hi)] * n_steps + [(u_lo, u_hi)] * n_steps,
        constraints=[LinearConstraint(rate_A, -max_jump, max_jump), pressure_con],
        options={"maxiter": maxiter, "ftol": ftol},
    )

    T_opt = np.clip(res.x[:n_steps], T_lo, T_hi)
    u_opt = np.clip(res.x[n_steps:], u_lo, u_hi)
    traj = simulate(np.r_[T_opt, u_opt])
    p_total = traj[:, IDX["p_H2"]] + traj[:, IDX["p_CO"]]
    return {
        "T": T_opt,
        "u": u_opt,
        "states": traj,
        "times": np.arange(n_steps + 1) * dt_min,
        "terminal_nC13al": float(traj[-1, IDX["nC13al"]]),
        "max_total_pressure": float(p_total.max()),
        "success": bool(res.success),
        "message": str(res.message),
        "n_iter": int(res.nit),
    }


def interpolate_control_profile(
    T_coarse: np.ndarray,
    u_coarse: np.ndarray,
    coarse_dt_min: float,
    target_times_min: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a coarse control profile onto an arbitrary time grid.

    Values beyond the coarse grid are held at the last optimal value
    (``numpy.interp`` edge behaviour).
    """
    coarse_t = np.arange(len(T_coarse)) * coarse_dt_min
    target = np.asarray(target_times_min, dtype=float)
    return np.interp(target, coarse_t, T_coarse), np.interp(target, coarse_t, u_coarse)


# The coarse optimum is solved once per (plant, grid) and reused for every
# perturbed training trajectory. HydroformylationParameters is a frozen
# dataclass, so it is hashable and safe to use as a cache key.
_PROFILE_CACHE: dict = {}


def optimal_control_profiles(
    target_times_min,
    *,
    plant: HydroformylationParameters = HydroformylationParameters(),
    coarse_dt_min: float = CONTROL_DT_MIN,
    coarse_n_steps: int = N_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the model-based optimal ``(T*, u*)`` sampled at ``target_times_min``.

    The optimum is computed once by single-shooting on the coarse grid (cached
    per plant/grid) and interpolated to the requested times. This is the nominal
    trajectory around which the dataset generators add control noise.
    """
    key = (plant, round(float(coarse_dt_min), 6), int(coarse_n_steps))
    cached = _PROFILE_CACHE.get(key)
    if cached is None:
        sol = solve_open_loop_optimal_control(
            dt_min=coarse_dt_min, n_steps=coarse_n_steps, plant=plant
        )
        cached = (sol["T"], sol["u"])
        _PROFILE_CACHE[key] = cached
    T_coarse, u_coarse = cached
    return interpolate_control_profile(T_coarse, u_coarse, coarse_dt_min, target_times_min)
