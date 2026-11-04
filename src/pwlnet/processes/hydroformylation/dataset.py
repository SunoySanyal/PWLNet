"""Dataset generation for hydroformylation transition surrogates."""

from __future__ import annotations

import numpy as np

from pwlnet.processes.hydroformylation.model import (
    CONTROL_BOUNDS,
    CONTROL_DT_MIN,
    DT_RATE_LIMIT,
    HydroformylationParameters,
    MILP_15S_DECISION_DT_MIN,
    MILP_15S_TRAIN_TRAJECTORIES,
    N_SAMPLE_STEPS,
    N_STEPS,
    STATE_NAMES,
    T_FINAL,
    sample_parameters,
)
from pwlnet.processes.hydroformylation.optimal_control import (
    REFERENCE_5MIN_PATH,
    optimal_control_profiles,
    reference_control_profiles,
)
from pwlnet.processes.hydroformylation.simulator import (
    simulate_policy,
    simulate_sample_policy,
)

__all__ = [
    "make_sample_transition_dataset",
    "make_transition_dataset",
    "random_control_profiles",
    "random_sample_control_profiles",
]


def random_control_profiles(
    rng: np.random.Generator,
    n_steps: int = N_STEPS,
    *,
    plant: HydroformylationParameters = HydroformylationParameters(),
    sigma_T: float = 0.6,
    offset_T: float = 12.0,
    tilt_T: float = 12.0,
    sigma_log_u: float = 0.25,
    offset_u: float = 0.20,
    tilt_u: float = 0.20,
    nominal=reference_control_profiles,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate coarse 5-minute control profiles as noise around the optimum.

    The nominal trajectory is the model-based open-loop optimal control profile
    (computed once and cached, no surrogate involved); the result is re-clipped
    to the temperature-rate limit and bounds. This concentrates training data
    where the controller actually operates, matching the source papers' "noise
    around the optimal trajectory" experiments.

    Per-trajectory ``offset_T`` and ``tilt_T`` are essential, not cosmetic. The
    nominal ``T`` ramps monotonically from the lower to the upper bound, so
    additive noise alone leaves ``T`` ~93% correlated with the elapsed-time
    feature: at any given time every trajectory sits within ``sigma_T`` of one
    value, and the surrogate cannot separate "T rose" from "time passed". It
    then books the temperature effect onto the time input, which the MILP holds
    fixed -- leaving it to optimise a ``T`` coefficient identified from a few
    tenths of a kelvin of residual noise. Drawing an independent offset and
    tilt per trajectory decorrelates the two without changing the marginal
    coverage of the ``T`` band.

    ``u`` is perturbed **multiplicatively**. It spans three orders of magnitude
    along the optimum (~0.49 mol/min at the opening pressure transient down to
    ~0.001 mol/min at the end), so a single additive ``sigma_u`` cannot work:
    it is invisible against the opening feed and, at the tail, larger than the
    signal, clipping a large share of samples onto ``u = 0``. A log-space
    offset, tilt and jitter give the same *relative* spread everywhere, and
    decorrelate ``u`` from elapsed time the same way ``offset_T`` / ``tilt_T``
    do for temperature.
    """
    grid_t = np.arange(n_steps) * CONTROL_DT_MIN
    if nominal is reference_control_profiles:
        # Load the stored 5-minute optimum rather than re-solving it (a ~30 min
        # solve otherwise repeated in every fresh process). Note this is the
        # 5-minute reference, NOT a resampling of the 15 s one: the 15 s feed
        # peaks near 0.49 mol/min during the opening pressure transient, and
        # holding a sample of that spike for five minutes over-feeds badly.
        T_nom, u_nom = nominal(grid_t, path=REFERENCE_5MIN_PATH)
    else:
        T_nom, u_nom = nominal(grid_t, plant=plant)
    frac = grid_t / T_FINAL
    T = (
        T_nom
        + rng.uniform(-offset_T, offset_T)
        + rng.uniform(-tilt_T, tilt_T) * frac
        + rng.normal(0.0, sigma_T, size=n_steps)
    )
    u = u_nom * np.exp(
        rng.uniform(-offset_u, offset_u)
        + rng.uniform(-tilt_u, tilt_u) * frac
        + rng.normal(0.0, sigma_log_u, size=n_steps)
    )
    max_jump = DT_RATE_LIMIT * CONTROL_DT_MIN
    for k in range(n_steps):
        if k > 0:
            T[k] = np.clip(T[k], T[k - 1] - max_jump, T[k - 1] + max_jump)
        T[k] = np.clip(T[k], *CONTROL_BOUNDS[0])
        u[k] = np.clip(u[k], *CONTROL_BOUNDS[1])
    return T, u


def make_transition_dataset(
    n_trajectories: int = 320,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one-step transition rows for the 5-minute surrogate."""
    rng = np.random.default_rng(2479085) if rng is None else rng
    rows_x, rows_y = [], []
    for _ in range(n_trajectories):
        p = sample_parameters(rng)
        T_profile, u_profile = random_control_profiles(rng)
        t, states = simulate_policy(T_profile, u_profile, p)
        for k in range(N_STEPS):
            rows_x.append(np.r_[t[k] / T_FINAL, states[k], T_profile[k], u_profile[k]])
            rows_y.append(states[k + 1])
    return np.asarray(rows_x), np.asarray(rows_y)


def random_sample_control_profiles(
    rng: np.random.Generator,
    n_steps: int = N_SAMPLE_STEPS,
    *,
    plant: HydroformylationParameters = HydroformylationParameters(),
    sigma_T: float = 0.18,
    offset_T: float = 12.0,
    tilt_T: float = 12.0,
    sigma_log_u: float = 0.25,
    offset_u: float = 0.20,
    tilt_u: float = 0.20,
    nominal=reference_control_profiles,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 15-second control profiles as noise around the optimum.

    Same idea as :func:`random_control_profiles`, including the per-trajectory
    ``offset_T`` / ``tilt_T`` and the multiplicative ``u`` perturbation -- see
    that function's docstring for why additive noise alone is not enough.

    ``nominal`` defaults to :func:`reference_control_profiles`, which loads the
    stored 15 s optimum rather than re-solving the open-loop problem (a ~30 min
    solve that is otherwise repeated in every fresh process). Pass
    ``nominal=optimal_control_profiles`` to solve it instead, e.g. when
    generating data for a non-nominal ``plant``.
    """
    grid_t = np.arange(n_steps) * MILP_15S_DECISION_DT_MIN
    if nominal is reference_control_profiles:
        T_nom, u_nom = nominal(grid_t)
    else:
        T_nom, u_nom = nominal(grid_t, plant=plant)
    frac = grid_t / T_FINAL
    T = (
        T_nom
        + rng.uniform(-offset_T, offset_T)
        + rng.uniform(-tilt_T, tilt_T) * frac
        + rng.normal(0.0, sigma_T, size=n_steps)
    )
    u = u_nom * np.exp(
        rng.uniform(-offset_u, offset_u)
        + rng.uniform(-tilt_u, tilt_u) * frac
        + rng.normal(0.0, sigma_log_u, size=n_steps)
    )
    max_jump = DT_RATE_LIMIT * MILP_15S_DECISION_DT_MIN
    for k in range(n_steps):
        if k > 0:
            T[k] = np.clip(T[k], T[k - 1] - max_jump, T[k - 1] + max_jump)
        T[k] = np.clip(T[k], *CONTROL_BOUNDS[0])
        u[k] = np.clip(u[k], *CONTROL_BOUNDS[1])
    return T, u


def make_sample_transition_dataset(
    n_trajectories: int = MILP_15S_TRAIN_TRAJECTORIES,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one-step transition rows for the 15-second surrogate."""
    rng = np.random.default_rng(2479085) if rng is None else rng
    rows_x, rows_y = [], []
    for _ in range(n_trajectories):
        p = sample_parameters(rng)
        T_profile, u_profile = random_sample_control_profiles(rng)
        times, states = simulate_sample_policy(T_profile, u_profile, p)
        for k in range(len(T_profile)):
            rows_x.append(np.r_[times[k] / T_FINAL, states[k], T_profile[k], u_profile[k]])
            rows_y.append(states[k + 1])
    return np.asarray(rows_x), np.asarray(rows_y)
