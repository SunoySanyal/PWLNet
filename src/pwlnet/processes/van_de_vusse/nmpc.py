"""NMPC model builders for Van de Vusse transition surrogates."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np

from pwlnet.core.network import NetworkDefinition
from pwlnet.processes.van_de_vusse.model import (
    CONTROL_BOUNDS,
    DECISION_DT_MIN,
    OUTPUT_NAMES,
    STATE_BOUNDS,
    STATE_NAMES,
    VanDeVusseParameters,
)
from pwlnet.processes.van_de_vusse.simulator import step_state

__all__ = [
    "build_multiple_shooting_model",
    "reachable_input_bounds",
    "run_receding_horizon_nmpc",
    "solve_multiple_shooting_move",
    "solve_receding_move",
    "tighten_network_bounds",
]

_CB_INDEX = OUTPUT_NAMES.index("CB")
_SOSX_FORMULATION = "tanh_pwl_sos2"
_MILP_FORMULATIONS = {"relu_bigm", "tanh_pwl_bigm", _SOSX_FORMULATION}


def reachable_input_bounds(
    X_data: np.ndarray,
    *,
    margin_frac: float = 0.10,
) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
    """Build tight embedding bounds from Van de Vusse transition inputs."""
    data = np.asarray(X_data, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(STATE_NAMES) + 1:
        raise ValueError(
            f"X_data must have shape (n, {len(STATE_NAMES) + 1}), got {data.shape}."
        )
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    pad = margin_frac * (hi - lo) + 1e-6
    lo = lo - pad
    hi = hi + pad
    lo[: len(STATE_NAMES)] = np.maximum(lo[: len(STATE_NAMES)], STATE_BOUNDS[:, 0])
    hi[: len(STATE_NAMES)] = np.minimum(hi[: len(STATE_NAMES)], STATE_BOUNDS[:, 1])
    lo[-1] = CONTROL_BOUNDS[0, 0]
    hi[-1] = CONTROL_BOUNDS[0, 1]
    return list(zip(lo.tolist(), hi.tolist())), lo[: len(STATE_NAMES)], hi[: len(STATE_NAMES)]


def tighten_network_bounds(
    network: NetworkDefinition,
    input_bounds: list[tuple[float, float]],
) -> NetworkDefinition:
    """Return a copy of ``network`` with restricted input bounds."""
    return NetworkDefinition(
        layers=network.layers,
        input_scaler=network.input_scaler,
        output_scaler=network.output_scaler,
        input_bounds=input_bounds,
    )


def _build_block(
    model: Any,
    network: NetworkDefinition,
    formulation: str,
    block_name: str,
    *,
    breakpoints: int | np.ndarray = 5,
) -> Any:
    from pwlnet.embedding.block_builder import build_surrogate_block

    return build_surrogate_block(
        model,
        network=network,
        formulation=formulation,
        block_name=block_name,
        breakpoints=breakpoints,
    )


def build_multiple_shooting_model(
    network: NetworkDefinition,
    x0,
    *,
    horizon_steps: int = 5,
    formulation: str = "tanh_pwl_bigm",
    breakpoints: int | np.ndarray = 5,
    target: Literal["delta", "next"] = "delta",
    d_prev: float | None = None,
    move_penalty: float = 0.0,
    terminal_weight: float = 1.0,
    stage_weight: float = 0.0,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Build a chained transition-surrogate control model.

    The model minimizes the negative product objective, so larger ``CB``
    is better:

    ``-terminal_weight * CB_H - stage_weight * sum(CB_j) + move_penalty * |delta D|``.
    """
    if target not in {"delta", "next"}:
        raise ValueError("target must be 'delta' or 'next'.")
    H = int(horizon_steps)
    if H < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}.")

    import pyomo.environ as pyo

    m = pyo.ConcreteModel(name="van_de_vusse_transition_mpc")
    state_set = pyo.RangeSet(0, len(STATE_NAMES) - 1)
    step_set = pyo.RangeSet(0, H - 1)
    node_set = pyo.RangeSet(0, H)
    m.x = pyo.Var(
        node_set,
        state_set,
        bounds=lambda _m, _j, s: (float(STATE_BOUNDS[int(s), 0]), float(STATE_BOUNDS[int(s), 1])),
    )
    m.D = pyo.Var(step_set, bounds=tuple(CONTROL_BOUNDS[0]))
    m.link = pyo.ConstraintList()

    initial = np.asarray(x0, dtype=float).ravel()
    if initial.shape != (len(STATE_NAMES),):
        raise ValueError(f"x0 must have {len(STATE_NAMES)} entries, got {initial.shape}.")
    for s in range(len(STATE_NAMES)):
        m.x[0, s].fix(float(np.clip(initial[s], STATE_BOUNDS[s, 0], STATE_BOUNDS[s, 1])))

    blocks: list[Any] = []
    for j in range(H):
        block = _build_block(
            m,
            network,
            formulation,
            f"step_{j}",
            breakpoints=breakpoints,
        )
        blocks.append(block)
        for s in range(len(STATE_NAMES)):
            m.link.add(block.inputs[s] == m.x[j, s])
            if target == "delta":
                m.link.add(m.x[j + 1, s] == m.x[j, s] + block.outputs[s])
            else:
                m.link.add(m.x[j + 1, s] == block.outputs[s])
        m.link.add(block.inputs[len(STATE_NAMES)] == m.D[j])

    move_terms = []
    if move_penalty > 0.0:
        m.abs_move = pyo.Var(step_set, bounds=(0.0, None))
        m.move_link = pyo.ConstraintList()
        for j in range(H):
            previous = float(d_prev) if j == 0 and d_prev is not None else (m.D[j - 1] if j > 0 else None)
            if previous is not None:
                m.move_link.add(m.abs_move[j] >= m.D[j] - previous)
                m.move_link.add(m.abs_move[j] >= previous - m.D[j])
                move_terms.append(m.abs_move[j])

    product_terminal = m.x[H, _CB_INDEX]
    product_stage = sum(m.x[j, _CB_INDEX] for j in range(1, H + 1))
    penalty = float(move_penalty) * sum(move_terms) if move_terms else 0.0
    m.objective = pyo.Objective(
        expr=-float(terminal_weight) * product_terminal - float(stage_weight) * product_stage + penalty,
        sense=pyo.minimize,
    )
    track = {"D0": m.D[0], "CB_terminal": product_terminal}
    for s, name in enumerate(STATE_NAMES):
        track[f"{name}_terminal"] = m.x[H, s]
    return m, blocks, track


def _count_binary_variables(model: Any) -> int:
    import pyomo.environ as pyo

    return sum(1 for var in model.component_data_objects(pyo.Var) if var.is_binary())


def solve_multiple_shooting_move(
    network: NetworkDefinition,
    x0,
    *,
    horizon_steps: int = 5,
    formulation: str = "tanh_pwl_bigm",
    breakpoints: int | np.ndarray = 5,
    target: Literal["delta", "next"] = "delta",
    d_prev: float | None = None,
    move_penalty: float = 0.0,
    terminal_weight: float = 1.0,
    stage_weight: float = 0.0,
    solver: str | None = None,
    solver_options: dict[str, Any] | None = None,
    time_limit: float | None = 20.0,
    mip_gap: float | None = 0.01,
    sosx_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve one Van de Vusse surrogate MPC move and return the first control."""
    import pyomo.environ as pyo

    form = str(formulation).strip().lower()
    model, blocks, track = build_multiple_shooting_model(
        network,
        x0,
        horizon_steps=horizon_steps,
        formulation=form,
        breakpoints=breakpoints,
        target=target,
        d_prev=d_prev,
        move_penalty=move_penalty,
        terminal_weight=terminal_weight,
        stage_weight=stage_weight,
    )

    t0 = time.perf_counter()
    sosx_result = None
    if form == _SOSX_FORMULATION:
        from pwlnet.solvers import solve_sosx

        opts = dict(sosx_options or {})
        sosx_result = solve_sosx(
            model,
            blocks=blocks,
            solver=solver or "appsi_highs",
            solver_options=solver_options,
            sense="minimize",
            track=track,
            breakpoints=breakpoints,
            time_limit=time_limit,
            **opts,
        )
        elapsed = time.perf_counter() - t0
        if not sosx_result.found:
            return {
                "D": None,
                "predicted": None,
                "status": sosx_result.status,
                "termination": sosx_result.status,
                "solve_seconds": elapsed,
                "n_binaries": 0,
                "sosx_nodes": sosx_result.n_nodes,
                "sosx_result": sosx_result,
            }
        d0 = float(np.clip(sosx_result.values["D0"], *CONTROL_BOUNDS[0]))
        predicted = np.array(
            [sosx_result.values[f"{name}_terminal"] for name in STATE_NAMES],
            dtype=float,
        )
        return {
            "D": d0,
            "predicted": predicted,
            "status": sosx_result.status,
            "termination": sosx_result.status,
            "solve_seconds": elapsed,
            "n_binaries": 0,
            "sosx_nodes": sosx_result.n_nodes,
            "objective": sosx_result.objective,
            "sosx_result": sosx_result,
        }

    if form in _MILP_FORMULATIONS:
        from pwlnet.solvers.solver_config import get_milp_solver

        options = dict(solver_options or {})
        if time_limit is not None:
            options.setdefault("TimeLimit", float(time_limit))
        if mip_gap is not None:
            options.setdefault("MIPGap", float(mip_gap))
        opt = get_milp_solver(solver or "appsi_highs", options=options)
    else:
        from pwlnet.solvers.solver_config import get_pyomo_solver

        opt = get_pyomo_solver(solver or "ipopt", options=solver_options)

    result = opt.solve(model, tee=False, load_solutions=False)
    elapsed = time.perf_counter() - t0
    status = str(getattr(result.solver, "status", "unknown"))
    termination = str(getattr(result.solver, "termination_condition", "unknown"))
    if len(result.solution) == 0:
        return {
            "D": None,
            "predicted": None,
            "status": status,
            "termination": termination,
            "solve_seconds": elapsed,
            "n_binaries": _count_binary_variables(model),
            "sosx_nodes": None,
            "sosx_result": None,
        }
    model.solutions.load_from(result)
    predicted = np.array([pyo.value(model.x[horizon_steps, s]) for s in range(len(STATE_NAMES))], dtype=float)
    return {
        "D": float(np.clip(pyo.value(model.D[0]), *CONTROL_BOUNDS[0])),
        "predicted": predicted,
        "status": status,
        "termination": termination,
        "solve_seconds": elapsed,
        "n_binaries": _count_binary_variables(model),
        "sosx_nodes": None,
        "objective": float(pyo.value(model.objective)),
        "sosx_result": None,
    }


def solve_receding_move(
    network: NetworkDefinition,
    x0,
    *,
    formulation: str = "tanh_pwl_bigm",
    breakpoints: int | np.ndarray = 5,
    target: Literal["delta", "next"] = "delta",
    d_prev: float | None = None,
    move_penalty: float = 0.0,
    solver: str | None = None,
    solver_options: dict[str, Any] | None = None,
    time_limit: float | None = 20.0,
    mip_gap: float | None = 0.01,
) -> dict[str, Any]:
    """Solve a one-step greedy Van de Vusse MPC move."""
    return solve_multiple_shooting_move(
        network,
        x0,
        horizon_steps=1,
        formulation=formulation,
        breakpoints=breakpoints,
        target=target,
        d_prev=d_prev,
        move_penalty=move_penalty,
        solver=solver,
        solver_options=solver_options,
        time_limit=time_limit,
        mip_gap=mip_gap,
    )


def run_receding_horizon_nmpc(
    network: NetworkDefinition,
    X_bounds_data: np.ndarray,
    *,
    x0,
    params: VanDeVusseParameters | None = None,
    closed_loop_steps: int = 40,
    horizon_steps: int = 1,
    formulation: str = "tanh_pwl_bigm",
    breakpoints: int | np.ndarray = 5,
    target: Literal["delta", "next"] = "delta",
    move_penalty: float = 0.0,
    solver: str | None = None,
    solver_options: dict[str, Any] | None = None,
    time_limit: float | None = 20.0,
    mip_gap: float | None = 0.01,
    dt_min: float = DECISION_DT_MIN,
) -> dict[str, Any]:
    """Run closed-loop Van de Vusse NMPC against the true plant."""
    input_bounds, state_lo, state_hi = reachable_input_bounds(X_bounds_data)
    embed_net = tighten_network_bounds(network, input_bounds)
    p = params if params is not None else VanDeVusseParameters()
    x = np.clip(np.asarray(x0, dtype=float).ravel(), state_lo, state_hi)
    states = [x.copy()]
    controls: list[float] = []
    predictions: list[np.ndarray] = []
    terminations: list[str] = []
    solve_times: list[float] = []
    binaries: list[int] = []
    sosx_nodes: list[float] = []
    d_prev = None
    previous_sosx = None
    for _k in range(int(closed_loop_steps)):
        step_solver_options = dict(solver_options or {})
        if previous_sosx is not None and str(formulation).strip().lower() == _SOSX_FORMULATION:
            step_solver_options["incumbent"] = previous_sosx
        move = solve_multiple_shooting_move(
            embed_net,
            x,
            horizon_steps=horizon_steps,
            formulation=formulation,
            breakpoints=breakpoints,
            target=target,
            d_prev=d_prev,
            move_penalty=move_penalty,
            solver=solver,
            solver_options=step_solver_options,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        if move["D"] is None:
            raise RuntimeError(
                f"Van de Vusse NMPC returned no incumbent at step {_k} "
                f"(status={move['status']}, termination={move['termination']})."
            )
        d = float(move["D"])
        x = step_state(x, d, dt_min=dt_min, params=p, n_substeps=2)
        states.append(x.copy())
        controls.append(d)
        predictions.append(np.asarray(move["predicted"], dtype=float))
        terminations.append(str(move["termination"]))
        solve_times.append(float(move["solve_seconds"]))
        binaries.append(int(move["n_binaries"]))
        sosx_nodes.append(float(move["sosx_nodes"]) if move["sosx_nodes"] is not None else np.nan)
        d_prev = d
        previous_sosx = move.get("sosx_result")
        if previous_sosx is not None and not getattr(previous_sosx, "found", False):
            previous_sosx = None
    return {
        "times": np.arange(len(states), dtype=float) * float(dt_min),
        "control_times": np.arange(len(controls), dtype=float) * float(dt_min),
        "states": np.asarray(states),
        "controls": np.asarray(controls),
        "predicted": np.asarray(predictions),
        "termination_conditions": terminations,
        "solve_times": np.asarray(solve_times),
        "n_binaries": np.asarray(binaries),
        "sosx_nodes": np.asarray(sosx_nodes),
        "decision_dt_min": float(dt_min),
        "prediction_steps": int(horizon_steps),
    }
