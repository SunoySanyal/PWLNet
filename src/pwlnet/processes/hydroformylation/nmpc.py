"""NMPC model builders for hydroformylation transition surrogates."""

from __future__ import annotations

import numpy as np

from pwlnet.core.network import NetworkDefinition
from pwlnet.formulations.pyomo_pwl import build_tanh_pwl_bigm_block
from pwlnet.processes.hydroformylation.model import (
    CONTROL_BOUNDS,
    CONTROL_DT_MIN,
    DT_RATE_LIMIT,
    HydroformylationParameters,
    IDX,
    INITIAL_STATE,
    MEASUREMENT_DELAY_MIN,
    MEASUREMENT_INTERVAL_MIN,
    MILP_15S_CLOSED_LOOP_STEPS,
    MILP_15S_DECISION_DT_MIN,
    MILP_15S_PREDICTION_STEPS,
    N_SAMPLE_STEPS,
    N_STEPS,
    STATE_NAMES,
    T_FINAL,
    pressure_backoff,
)
from pwlnet.processes.hydroformylation.simulator import step_state
from pwlnet.solvers.solver_config import get_milp_solver

__all__ = [
    "estimate_state_from_delayed_measurement",
    "latest_delayed_measurement_time",
    "reachable_input_bounds",
    "reachable_input_bounds_from_data",
    "run_15s_gurobi_milp_nmpc",
    "run_receding_horizon_nmpc",
    "solve_15s_gurobi_milp_move",
    "solve_receding_move",
    "tighten_network_bounds",
]


def reachable_input_bounds(
    X_data: np.ndarray, *, margin_frac: float = 0.10
) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
    """Build tight embedding bounds from transition-training inputs.

    Every feature gets a padded hull of the training data, with two exceptions.

    ``tau`` is pinned to ``[0, 1]`` because it is a normalised time index, not a
    learned quantity.

    ``T`` is opened to the full :data:`CONTROL_BOUNDS` because the training
    profiles genuinely span it: each trajectory draws its own temperature offset
    and tilt, so every point of the 368.15-388.15 K band is covered at every
    time. Widening it here therefore asks the surrogate nothing it has not seen.

    ``u`` is deliberately **not** widened. Its admissible range is 0-7 mol/min,
    but holding the total pressure on the 20 bar constraint only ever calls for
    ~0.001-0.9 mol/min, and sustained larger feeds are not physically reachable
    (they drive the gas phase to hundreds of bar). Opening ``u`` to the full box
    would let the optimizer chase a feed rate roughly eight times beyond the
    data, extrapolating a coefficient the fit never identified -- the same
    failure that made the embedded controller chatter on ``T`` before the
    training profiles were decorrelated. The padded data hull is the honest
    bound; widen it only alongside training data that actually covers the range.
    """
    lo = X_data.min(axis=0).astype(float)
    hi = X_data.max(axis=0).astype(float)
    pad = margin_frac * (hi - lo) + 1e-6
    lo, hi = lo - pad, hi + pad
    lo[0], hi[0] = 0.0, 1.0
    lo[1 + len(STATE_NAMES)], hi[1 + len(STATE_NAMES)] = CONTROL_BOUNDS[0]
    u_idx = 2 + len(STATE_NAMES)
    lo[u_idx] = max(lo[u_idx], CONTROL_BOUNDS[1][0])
    hi[u_idx] = min(hi[u_idx], CONTROL_BOUNDS[1][1])
    return list(zip(lo.tolist(), hi.tolist())), lo[1 : 1 + len(STATE_NAMES)], hi[1 : 1 + len(STATE_NAMES)]


def reachable_input_bounds_from_data(
    X_data: np.ndarray, *, margin_frac: float = 0.10
) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
    """Backward-compatible alias for the notebook's former helper name."""
    return reachable_input_bounds(X_data, margin_frac=margin_frac)


def tighten_network_bounds(network: NetworkDefinition, input_bounds) -> NetworkDefinition:
    """Return a copy of ``network`` with restricted input bounds."""
    return NetworkDefinition(
        layers=network.layers,
        input_scaler=network.input_scaler,
        output_scaler=network.output_scaler,
        input_bounds=input_bounds,
    )


def solve_receding_move(
    network: NetworkDefinition,
    x_meas,
    k: int,
    T_prev: float | None,
    state_lo: np.ndarray,
    state_hi: np.ndarray,
    *,
    pieces: int = 3,
    penalty: float = 25.0,
    time_limit: float = 15.0,
):
    """Solve one coarse one-block PWL-tanh MILP control move."""
    import pyomo.environ as pyo
    from pyomo.contrib.appsi.solvers import Highs

    model = pyo.ConcreteModel(name=f"hydroformylation_nmpc_step_{k}")
    block = build_tanh_pwl_bigm_block(model, network, block_name="step", breakpoints=pieces)
    block.inputs[0].fix(float((k * CONTROL_DT_MIN) / T_FINAL))
    for s in range(len(STATE_NAMES)):
        block.inputs[1 + s].fix(float(np.clip(x_meas[s], state_lo[s], state_hi[s])))

    T_in = block.inputs[1 + len(STATE_NAMES)]
    u_in = block.inputs[2 + len(STATE_NAMES)]
    p_next = block.outputs[IDX["p_H2"]] + block.outputs[IDX["p_CO"]]

    model.s_pressure = pyo.Var(bounds=(0, None))
    model.pressure_backoff = pyo.Constraint(
        expr=p_next <= 20.0 - pressure_backoff((k + 1) * CONTROL_DT_MIN) + model.s_pressure
    )
    penalty_terms = penalty * model.s_pressure
    if k + 1 == N_STEPS:
        model.s_terminal = pyo.Var(bounds=(0, None))
        model.product_terminal = pyo.Constraint(
            expr=block.outputs[IDX["nC13al"]] <= 0.8 + model.s_terminal
        )
        penalty_terms = penalty_terms + penalty * model.s_terminal

    max_jump = DT_RATE_LIMIT * CONTROL_DT_MIN
    if T_prev is not None:
        model.T_rate_up = pyo.Constraint(expr=T_in - T_prev <= max_jump)
        model.T_rate_dn = pyo.Constraint(expr=T_prev - T_in <= max_jump)

    model.objective = pyo.Objective(expr=-block.outputs[IDX["nC13al"]] + penalty_terms)

    opt = Highs()
    opt.config.load_solution = False
    opt.config.time_limit = time_limit
    result = opt.solve(model)
    if result.best_feasible_objective is None:
        return None
    opt.load_vars()
    T = float(np.clip(pyo.value(T_in), *CONTROL_BOUNDS[0]))
    u = float(np.clip(pyo.value(u_in), *CONTROL_BOUNDS[1]))
    predicted = np.array([pyo.value(block.outputs[s]) for s in range(len(STATE_NAMES))])
    return {
        "T": T,
        "u": u,
        "predicted": predicted,
        "pressure_slack": float(pyo.value(model.s_pressure)),
    }


def run_receding_horizon_nmpc(
    network: NetworkDefinition,
    X_bounds_data: np.ndarray,
    *,
    x0=INITIAL_STATE,
    pieces: int = 3,
    penalty: float = 25.0,
    time_limit: float = 15.0,
    plant: HydroformylationParameters = HydroformylationParameters(),
):
    """Run closed-loop coarse NMPC using one PWL-tanh block per move."""
    input_bounds, state_lo, state_hi = reachable_input_bounds(X_bounds_data)
    embed_net = tighten_network_bounds(network, input_bounds)

    x = np.asarray(x0, dtype=float).copy()
    states = [x.copy()]
    controls, predicted, slacks = [], [], []
    T_prev = None
    for k in range(N_STEPS):
        move = solve_receding_move(
            embed_net,
            x,
            k,
            T_prev,
            state_lo,
            state_hi,
            pieces=pieces,
            penalty=penalty,
            time_limit=time_limit,
        )
        if move is None:
            raise RuntimeError(f"NMPC step {k} returned no feasible solution.")
        x = step_state(x, move["T"], move["u"], CONTROL_DT_MIN, plant)
        states.append(x.copy())
        controls.append([move["T"], move["u"]])
        predicted.append(move["predicted"])
        slacks.append(move["pressure_slack"])
        T_prev = move["T"]
    times = np.arange(N_STEPS + 1) * CONTROL_DT_MIN
    return {
        "times": times,
        "states": np.asarray(states),
        "controls": np.asarray(controls),
        "predicted": np.asarray(predicted),
        "pressure_slacks": np.asarray(slacks),
    }


def latest_delayed_measurement_time(t_now: float) -> float | None:
    """Return the newest sample time available after measurement delay."""
    if t_now < MEASUREMENT_DELAY_MIN - 1e-12:
        return 0.0 if abs(t_now) < 1e-12 else None
    m = np.floor((t_now - MEASUREMENT_DELAY_MIN + 1e-12) / MEASUREMENT_INTERVAL_MIN)
    return max(0.0, float(m * MEASUREMENT_INTERVAL_MIN))


def estimate_state_from_delayed_measurement(
    sample_time,
    measured_state,
    t_now,
    control_history,
    *,
    plant: HydroformylationParameters = HydroformylationParameters(),
) -> np.ndarray:
    """Propagate the newest delayed measurement to the current decision time."""
    x_est = np.asarray(measured_state, dtype=float).copy()
    k0 = int(round(sample_time / MILP_15S_DECISION_DT_MIN))
    k1 = int(round(t_now / MILP_15S_DECISION_DT_MIN))
    for k in range(k0, k1):
        if k < len(control_history):
            T, u = control_history[k]
        elif control_history:
            T, u = control_history[-1]
        else:
            T, u = 378.15, 0.008
        x_est = step_state(x_est, float(T), float(u), MILP_15S_DECISION_DT_MIN, plant)
    return x_est


def solve_15s_gurobi_milp_move(
    network: NetworkDefinition,
    x_est,
    k0: int,
    T_prev: float | None,
    state_lo: np.ndarray,
    state_hi: np.ndarray,
    *,
    horizon_steps: int = MILP_15S_PREDICTION_STEPS,
    pieces: int = 3,
    penalty: float = 25.0,
    solver_name: str = "gurobi",
    time_limit: float = 5.0,
    mip_gap: float = 0.02,
):
    """Solve a chained 15-second PWL-tanh MILP and return the first move."""
    import pyomo.environ as pyo

    H = int(min(horizon_steps, N_SAMPLE_STEPS - k0))
    if H <= 0:
        return None
    m = pyo.ConcreteModel(name=f"hydroformylation_15s_milp_{k0}")
    m.x = pyo.Var(
        pyo.RangeSet(0, H),
        pyo.RangeSet(0, len(STATE_NAMES) - 1),
        bounds=lambda mm, j, s: (float(state_lo[int(s)]), float(state_hi[int(s)])),
    )
    m.T = pyo.Var(pyo.RangeSet(0, H - 1), bounds=tuple(CONTROL_BOUNDS[0]))
    m.u = pyo.Var(pyo.RangeSet(0, H - 1), bounds=tuple(CONTROL_BOUNDS[1]))
    m.link = pyo.ConstraintList()
    for s in range(len(STATE_NAMES)):
        m.x[0, s].fix(float(np.clip(x_est[s], state_lo[s], state_hi[s])))

    for j in range(H):
        block = build_tanh_pwl_bigm_block(m, network, block_name=f"step_{j}", breakpoints=pieces)
        block.inputs[0].fix(float(((k0 + j) * MILP_15S_DECISION_DT_MIN) / T_FINAL))
        for s in range(len(STATE_NAMES)):
            m.link.add(block.inputs[1 + s] == m.x[j, s])
            m.link.add(m.x[j + 1, s] == block.outputs[s])
        m.link.add(block.inputs[1 + len(STATE_NAMES)] == m.T[j])
        m.link.add(block.inputs[2 + len(STATE_NAMES)] == m.u[j])

    max_jump = DT_RATE_LIMIT * MILP_15S_DECISION_DT_MIN
    m.rate = pyo.ConstraintList()
    if T_prev is not None:
        m.rate.add(m.T[0] - float(T_prev) <= max_jump)
        m.rate.add(float(T_prev) - m.T[0] <= max_jump)
    for j in range(1, H):
        m.rate.add(m.T[j] - m.T[j - 1] <= max_jump)
        m.rate.add(m.T[j - 1] - m.T[j] <= max_jump)

    m.s_p = pyo.Var(pyo.RangeSet(1, H), bounds=(0, None), initialize=0.0)
    m.backoff = pyo.Constraint(
        pyo.RangeSet(1, H),
        rule=lambda mm, j: mm.x[j, IDX["p_H2"]] + mm.x[j, IDX["p_CO"]]
        <= 20.0 - pressure_backoff((k0 + j) * MILP_15S_DECISION_DT_MIN) + mm.s_p[j],
    )

    penalty_terms = penalty * sum(m.s_p[j] for j in range(1, H + 1))
    if k0 + H >= N_SAMPLE_STEPS:
        m.s_terminal = pyo.Var(bounds=(0, None), initialize=0.0)
        m.terminal = pyo.Constraint(expr=m.x[H, IDX["nC13al"]] <= 0.8 + m.s_terminal)
        penalty_terms = penalty_terms + penalty * m.s_terminal
    m.objective = pyo.Objective(expr=-m.x[H, IDX["nC13al"]] + penalty_terms)

    solver = get_milp_solver(solver_name, options={"TimeLimit": time_limit, "MIPGap": mip_gap})
    result = solver.solve(m, tee=False, load_solutions=False)
    status = str(result.solver.status)
    termination = str(result.solver.termination_condition)
    n_solutions = len(result.solution)
    if n_solutions == 0:
        return {
            "T": None,
            "u": None,
            "predicted": None,
            "status": status,
            "termination": termination,
            "n_solutions": n_solutions,
            "pressure_slack": float("nan"),
        }
    m.solutions.load_from(result)
    T0 = pyo.value(m.T[0], exception=False)
    u0 = pyo.value(m.u[0], exception=False)
    if T0 is None or u0 is None:
        return {
            "T": None,
            "u": None,
            "predicted": None,
            "status": status,
            "termination": termination,
            "n_solutions": n_solutions,
            "pressure_slack": float("nan"),
        }
    predicted = np.array(
        [[pyo.value(m.x[j, s], exception=False) for s in range(len(STATE_NAMES))] for j in range(H + 1)],
        dtype=float,
    )
    return {
        "T": float(np.clip(T0, *CONTROL_BOUNDS[0])),
        "u": float(np.clip(u0, *CONTROL_BOUNDS[1])),
        "predicted": predicted,
        "status": status,
        "termination": termination,
        "n_solutions": n_solutions,
        "pressure_slack": float(
            sum(pyo.value(m.s_p[j], exception=False) or 0.0 for j in range(1, H + 1))
        ),
    }


def run_15s_gurobi_milp_nmpc(
    network: NetworkDefinition,
    X_bounds_data: np.ndarray,
    *,
    x0=INITIAL_STATE,
    pieces: int = 3,
    horizon_steps: int = MILP_15S_PREDICTION_STEPS,
    closed_loop_steps: int = MILP_15S_CLOSED_LOOP_STEPS,
    solver_name: str = "gurobi",
    time_limit: float = 50,
    mip_gap: float = 0.02,
    plant: HydroformylationParameters = HydroformylationParameters(),
):
    """Run 15-second closed-loop PWL-tanh MILP NMPC."""
    import pandas as pd

    input_bounds, state_lo, state_hi = reachable_input_bounds(X_bounds_data)
    embed_net = tighten_network_bounds(network, input_bounds)
    x_true = np.asarray(x0, dtype=float).copy()
    true_states = [x_true.copy()]
    estimate_states = []
    controls = []
    terminations = []
    slacks = []
    measurement_rows = []
    T_prev = None
    last_sample_time = None
    last_measured_state = np.asarray(x0, dtype=float).copy()

    for k in range(closed_loop_steps):
        t_now = k * MILP_15S_DECISION_DT_MIN
        sample_time = latest_delayed_measurement_time(t_now)
        if sample_time is not None and (
            last_sample_time is None or sample_time > last_sample_time + 1e-12
        ):
            sample_idx = int(round(sample_time / MILP_15S_DECISION_DT_MIN))
            last_sample_time = sample_time
            last_measured_state = true_states[min(sample_idx, len(true_states) - 1)].copy()
            measurement_rows.append(
                {"sample_time": sample_time, "available_time": sample_time + MEASUREMENT_DELAY_MIN}
            )
        if last_sample_time is None:
            x_est = true_states[-1].copy()
        else:
            x_est = estimate_state_from_delayed_measurement(
                last_sample_time, last_measured_state, t_now, controls, plant=plant
            )
        estimate_states.append(x_est.copy())

        move = solve_15s_gurobi_milp_move(
            embed_net,
            x_est,
            k,
            T_prev,
            state_lo,
            state_hi,
            horizon_steps=horizon_steps,
            pieces=pieces,
            solver_name=solver_name,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        if move is None or move["T"] is None or move["u"] is None:
            detail = move if isinstance(move, dict) else {"status": "unknown", "termination": "unknown"}
            raise RuntimeError(
                f"15 s Gurobi MILP step {k} returned no incumbent solution "
                f"(status={detail.get('status')}, termination={detail.get('termination')})."
            )
        x_true = step_state(x_true, move["T"], move["u"], MILP_15S_DECISION_DT_MIN, plant)
        true_states.append(x_true.copy())
        controls.append([move["T"], move["u"]])
        terminations.append(move["termination"])
        slacks.append(move["pressure_slack"])
        T_prev = move["T"]

    return {
        "times": np.arange(closed_loop_steps + 1) * MILP_15S_DECISION_DT_MIN,
        "control_times": np.arange(closed_loop_steps) * MILP_15S_DECISION_DT_MIN,
        "states": np.asarray(true_states),
        "estimated_states": np.asarray(estimate_states),
        "controls": np.asarray(controls),
        "termination_conditions": terminations,
        "pressure_slacks": np.asarray(slacks),
        "measurements": pd.DataFrame(measurement_rows),
        "decision_dt_min": MILP_15S_DECISION_DT_MIN,
        "prediction_steps": horizon_steps,
    }
