"""Pyomo models for the Williams-Otto benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pwlnet.core.network import NetworkDefinition
from pwlnet.processes.williams_otto.model import (
    CELSIUS_TO_KELVIN,
    DEFAULT_INITIAL_GUESS,
    OUTPUT_NAMES,
    WilliamsOttoParameters,
)
from pwlnet.processes.williams_otto.simulator import solve_steady_state
from pwlnet.utils.exceptions import FormulationError

EXACT_OPTIMUM = {
    "FB": 4.3894,
    "TR": 80.495,
    "profit": 75.82,
}

__all__ = [
    "EXACT_OPTIMUM",
    "SurrogateOptimizationResult",
    "build_exact_model",
    "build_surrogate_optimization_model",
    "solve_surrogate_optimization_model",
]


@dataclass(frozen=True)
class SurrogateOptimizationResult:
    """Report from solving an embedded Williams-Otto surrogate model.

    ``surrogate_profit`` is the profit implied by the surrogate's
    predicted mass fractions at the selected operating point (exact
    bilinear economics). ``objective_value`` is the solver's objective,
    which for the McCormick profit formulations is a relaxation and may
    exceed ``surrogate_profit`` by the remaining envelope gap.
    """

    solver_name: str
    status: str
    termination_condition: str
    fb: float
    tr: float
    predicted_outputs: dict[str, float]
    surrogate_profit: float
    exact_outputs: dict[str, float]
    exact_profit: float
    exact_feasible: bool
    exact_optimum: dict[str, float]
    objective_value: float | None = None
    solve_seconds: float | None = None
    operating_constraints: bool = True

    def comparison_rows(self) -> list[dict[str, float | str | bool]]:
        """Rows suitable for a compact notebook/README table."""
        return [
            {
                "case": "exact optimum",
                "FB": self.exact_optimum["FB"],
                "TR": self.exact_optimum["TR"],
                "profit": self.exact_optimum["profit"],
                "XA": 0.12,
                "XG": 0.08,
                "feasible": True,
            },
            {
                "case": "surrogate optimum",
                "FB": self.fb,
                "TR": self.tr,
                "profit": self.surrogate_profit,
                "XA": self.predicted_outputs["XA"],
                "XG": self.predicted_outputs["XG"],
                "feasible": self.predicted_outputs["XA"] <= 0.12
                and self.predicted_outputs["XG"] <= 0.08,
            },
            {
                "case": "exact at surrogate optimum",
                "FB": self.fb,
                "TR": self.tr,
                "profit": self.exact_profit,
                "XA": self.exact_outputs["XA"],
                "XG": self.exact_outputs["XG"],
                "feasible": self.exact_feasible,
            },
        ]


def build_exact_model(
    params: WilliamsOttoParameters | None = None,
    fixed_fb: float | None = None,
    fixed_tr: float | None = None,
    maximize_profit: bool = True,
    include_operating_constraints: bool = True,
) -> Any:
    """Build the exact steady-state Williams-Otto Pyomo model.

    Parameters
    ----------
    params : WilliamsOttoParameters or None, optional
        Parameter set.
    fixed_fb, fixed_tr : float or None, optional
        Fix the operating inputs when provided.
    maximize_profit : bool, optional
        Add the canonical profit objective.
    include_operating_constraints : bool, optional
        Add ``XA <= xa_max`` and ``XG <= xg_max``.

    Returns
    -------
    pyomo.environ.ConcreteModel
        Exact nonlinear model.
    """
    import pyomo.environ as pyo

    p = params if params is not None else WilliamsOttoParameters()
    m = pyo.ConcreteModel(name="williams_otto_exact")
    m.FB = pyo.Var(bounds=p.fb_bounds, initialize=fixed_fb or 4.3894)
    m.TR = pyo.Var(bounds=p.tr_bounds, initialize=fixed_tr or 80.5)
    for name, init in zip(OUTPUT_NAMES, DEFAULT_INITIAL_GUESS):
        setattr(m, name, pyo.Var(bounds=(0.0, 1.0), initialize=float(init)))

    if fixed_fb is not None:
        m.FB.fix(float(fixed_fb)) # type: ignore
    if fixed_tr is not None:
        m.TR.fix(float(fixed_tr)) # type: ignore

    m.F = pyo.Expression(expr=p.fa + m.FB) # type: ignore
    t_kelvin = m.TR + CELSIUS_TO_KELVIN # type: ignore
    m.k1 = pyo.Expression(expr=p.a1 * pyo.exp(-p.b1 / t_kelvin))
    m.k2 = pyo.Expression(expr=p.a2 * pyo.exp(-p.b2 / t_kelvin))
    m.k3 = pyo.Expression(expr=p.a3 * pyo.exp(-p.b3 / t_kelvin))
    m.r1 = pyo.Expression(expr=m.k1 * m.XA * m.XB * p.holdup) # type: ignore
    m.r2 = pyo.Expression(expr=m.k2 * m.XB * m.XC * p.holdup) # type: ignore
    m.r3 = pyo.Expression(expr=m.k3 * m.XC * m.XP * p.holdup) # type: ignore

    m.balance_A = pyo.Constraint(expr=p.fa - m.F * m.XA - m.r1 == 0.0) # type: ignore
    m.balance_B = pyo.Constraint(expr=m.FB - m.F * m.XB - m.r1 - m.r2 == 0.0) # type: ignore
    m.balance_C = pyo.Constraint(
        expr=-m.F * m.XC + 2.0 * m.r1 - 2.0 * m.r2 - m.r3 == 0.0 # type: ignore
    )
    m.balance_E = pyo.Constraint(expr=-m.F * m.XE + 2.0 * m.r2 == 0.0) # type: ignore
    m.balance_G = pyo.Constraint(expr=-m.F * m.XG + 1.5 * m.r3 == 0.0) # type: ignore
    m.balance_P = pyo.Constraint(expr=-m.F * m.XP + m.r2 - 0.5 * m.r3 == 0.0) # type: ignore

    if include_operating_constraints:
        m.xa_limit = pyo.Constraint(expr=m.XA <= p.xa_max) # type: ignore
        m.xg_limit = pyo.Constraint(expr=m.XG <= p.xg_max) # type: ignore

    m.profit = pyo.Expression(
        expr=(p.price_p * m.XP + p.price_e * m.XE) * m.F # type: ignore
        - p.cost_a * p.fa
        - p.cost_b * m.FB # type: ignore
    ) 
    if maximize_profit:
        m.objective = pyo.Objective(expr=m.profit, sense=pyo.maximize)
    return m


def _add_mccormick_profit(
    m: Any,
    network: NetworkDefinition,
    p: WilliamsOttoParameters,
    output: dict[str, Any],
    n_segments: int,
) -> None:
    """Add a (piecewise) McCormick profit formulation to the model.

    The bilinear revenue terms ``FB * XP`` and ``FB * XE`` are replaced
    by auxiliary variables constrained with McCormick envelopes. The
    ``y`` boxes come from interval bound propagation through the
    network (intersected with the ``[0, 1]`` mass-fraction constraints
    already on the model), and the ``FB`` range is partitioned into
    ``n_segments`` intervals selected by shared binaries: the envelope
    gap shrinks proportionally to the segment width, so a single-segment
    plain McCormick relaxation can overstate the achievable profit by
    tens of $/s while ten segments make the relaxation nearly exact.
    """
    import pyomo.environ as pyo

    from pwlnet.core.bounds import propagate_bounds

    fb_lb, fb_ub = (float(b) for b in p.fb_bounds)
    grid = np.linspace(fb_lb, fb_ub, n_segments + 1)

    output_box: dict[str, tuple[float, float]] = {}
    propagated = propagate_bounds(network).output_bounds
    for name, (out_lb, out_ub) in zip(OUTPUT_NAMES, propagated):
        output_box[name] = (max(0.0, float(out_lb)), min(1.0, float(out_ub)))

    m.mccormick_profit = pyo.ConstraintList()
    if n_segments > 1:
        m.fb_segment = pyo.Var(range(n_segments), domain=pyo.Binary)
        m.mccormick_profit.add( 
            sum(m.fb_segment[k] for k in range(n_segments)) == 1 # type: ignore
        )
        m.mccormick_profit.add(
            m.FB >= sum(float(grid[k]) * m.fb_segment[k] for k in range(n_segments)) # type: ignore
        )
        m.mccormick_profit.add(
            m.FB <= sum(float(grid[k + 1]) * m.fb_segment[k] for k in range(n_segments)) # type: ignore
        )

    def plane_range(a: float, b: float, c: float, y_lb: float, y_ub: float) -> tuple[float, float]:
        """Range of ``a*y + b*FB + c`` over the global (FB, y) box."""
        corners = [
            a * y + b * fb + c
            for y in (y_lb, y_ub)
            for fb in (fb_lb, fb_ub)
        ]
        return min(corners), max(corners)

    def add_product(z: Any, y: Any, name: str) -> None:
        y_lb, y_ub = output_box[name]
        z_corners = [fb * yv for fb in (fb_lb, fb_ub) for yv in (y_lb, y_ub)]
        z_lb, z_ub = min(z_corners), max(z_corners)
        for k in range(n_segments):
            f_lo, f_hi = float(grid[k]), float(grid[k + 1])
            # (a, b, c, is_lower) for z >= / z <= a*y + b*FB + c
            planes = [
                (f_lo, y_lb, -f_lo * y_lb, True),
                (f_hi, y_ub, -f_hi * y_ub, True),
                (f_hi, y_lb, -f_hi * y_lb, False),
                (f_lo, y_ub, -f_lo * y_ub, False),
            ]
            for a, b, c, is_lower in planes:
                if is_lower:
                    # Exact big-M: when the segment is inactive the plane
                    # may exceed z by at most plane_max - z_lb.
                    slack = max(0.0, plane_range(a, b, c, y_lb, y_ub)[1] - z_lb)
                    relax = slack * (1 - m.fb_segment[k]) if n_segments > 1 else 0.0
                    m.mccormick_profit.add(z >= a * y + b * m.FB + c - relax)
                else:
                    slack = max(0.0, z_ub - plane_range(a, b, c, y_lb, y_ub)[0])
                    relax = slack * (1 - m.fb_segment[k]) if n_segments > 1 else 0.0
                    m.mccormick_profit.add(z <= a * y + b * m.FB + c + relax)

    xp_box = output_box["XP"]
    xe_box = output_box["XE"]
    m.FB_XP = pyo.Var(
        bounds=(fb_lb * xp_box[0], fb_ub * xp_box[1]),
        initialize=0.5 * fb_ub * (xp_box[0] + xp_box[1]),
    )
    m.FB_XE = pyo.Var(
        bounds=(fb_lb * xe_box[0], fb_ub * xe_box[1]),
        initialize=0.5 * fb_ub * (xe_box[0] + xe_box[1]),
    )
    add_product(m.FB_XP, output["XP"], "XP")
    add_product(m.FB_XE, output["XE"], "XE")
    m.profit = pyo.Expression(
        expr=p.price_p * (p.fa * output["XP"] + m.FB_XP)
        + p.price_e * (p.fa * output["XE"] + m.FB_XE)
        - p.cost_a * p.fa
        - p.cost_b * m.FB
    )

def build_surrogate_optimization_model(
    network: NetworkDefinition,
    formulation: str = "relu_bigm",
    params: WilliamsOttoParameters | None = None,
    block_name: str = "surrogate",
    profit_formulation: str = "piecewise_mccormick",
    fb_segments: int = 10,
    include_operating_constraints: bool = True,
    **formulation_options: Any,
) -> tuple[Any, Any]:
    """Build a Williams-Otto economic model with an embedded NN surrogate.

    The surrogate must map ``(FB, TR)`` to the six mass fractions ordered
    as :data:`pwlnet.processes.williams_otto.model.OUTPUT_NAMES`.
    The default profit formulation partitions ``FB`` into
    ``fb_segments`` intervals and applies McCormick envelopes to
    ``FB * XP`` and ``FB * XE`` per interval (``"piecewise_mccormick"``)
    so the embedded problem stays a MILP that HiGHS can solve while
    keeping the relaxation gap negligible; ``"mccormick"`` is the
    single-segment variant and ``"bilinear"`` keeps the exact products
    for NLP solvers. Extra keyword arguments (for example
    ``breakpoints`` for the PWL-tanh formulations) are passed through
    to :func:`pwlnet.embedding.block_builder.build_surrogate_block`.

    Parameters
    ----------
    include_operating_constraints : bool, optional
        Add the operating constraints ``XA <= xa_max`` and
        ``XG <= xg_max`` on the surrogate's predicted outputs. Default
        True. Set to False for the unconstrained profit-maximization
        baseline (which typically drives the design to a corner of the
        ``(FB, TR)`` box and violates the true process constraints).
    """
    import pyomo.environ as pyo

    from pwlnet.embedding.block_builder import build_surrogate_block

    p = params if params is not None else WilliamsOttoParameters()
    m = pyo.ConcreteModel(name="williams_otto_relu_surrogate")
    m.FB = pyo.Var(bounds=p.fb_bounds, initialize=4.3894)
    m.TR = pyo.Var(bounds=p.tr_bounds, initialize=80.5)

    block = build_surrogate_block(
        m,
        network=network,
        formulation=formulation,
        block_name=block_name,
        **formulation_options,
    )
    input_keys = list(block.inputs.keys())
    output_keys = list(block.outputs.keys())
    if len(input_keys) != 2 or len(output_keys) != len(OUTPUT_NAMES):
        raise ValueError(
            "Williams-Otto surrogate must have 2 inputs and "
            f"{len(OUTPUT_NAMES)} outputs."
        )

    m.link_FB = pyo.Constraint(expr=block.inputs[input_keys[0]] == m.FB)
    m.link_TR = pyo.Constraint(expr=block.inputs[input_keys[1]] == m.TR)
    output = {name: block.outputs[key] for name, key in zip(OUTPUT_NAMES, output_keys)}

    m.mass_fraction_bounds = pyo.ConstraintList()
    for name in OUTPUT_NAMES:
        m.mass_fraction_bounds.add(output[name] >= 0.0) # type: ignore
        m.mass_fraction_bounds.add(output[name] <= 1.0) # type: ignore
    if include_operating_constraints:
        m.xa_limit = pyo.Constraint(expr=output["XA"] <= p.xa_max)
        m.xg_limit = pyo.Constraint(expr=output["XG"] <= p.xg_max)

    profit_name = str(profit_formulation).strip().lower()
    if profit_name == "bilinear":
        m.profit = pyo.Expression(
            expr=(p.price_p * output["XP"] + p.price_e * output["XE"]) * (p.fa + m.FB) # type: ignore
            - p.cost_a * p.fa
            - p.cost_b * m.FB # type: ignore
        )
    elif profit_name in {"mccormick", "piecewise_mccormick"}:
        n_segments = 1 if profit_name == "mccormick" else max(1, int(fb_segments))
        _add_mccormick_profit(m, network, p, output, n_segments)
    else:
        raise ValueError(
            "profit_formulation must be 'mccormick', 'piecewise_mccormick', "
            f"or 'bilinear', got {profit_formulation!r}."
        )
    m.objective = pyo.Objective(expr=m.profit, sense=pyo.maximize)
    return m, block


def solve_surrogate_optimization_model(
    network: NetworkDefinition,
    formulation: str = "relu_bigm",
    solver: str | None = None,
    solver_options: dict[str, Any] | None = None,
    executable: str | None = None,
    params: WilliamsOttoParameters | None = None,
    profit_formulation: str = "piecewise_mccormick",
    include_operating_constraints: bool = True,
    **formulation_options: Any,
) -> SurrogateOptimizationResult:
    """Build and solve the embedded Williams-Otto surrogate optimization model.

    Parameters
    ----------
    network : NetworkDefinition
        Surrogate mapping ``(FB, TR)`` to the six Williams-Otto mass
        fractions in :data:`OUTPUT_NAMES` order.
    formulation : str, optional
        OMLT formulation, usually ``"relu_bigm"`` for the Sprint-1
        MILP baseline.
    solver : str or None, optional
        Pyomo solver factory name. ``None`` uses the default
        open-source HiGHS path (``"appsi_highs"``). Gurobi can be
        selected with names such as ``"gurobi"`` or ``"appsi_gurobi"``.
    solver_options : dict or None, optional
        Options passed to the selected solver.
    executable : str or None, optional
        Optional executable path for shell-based solvers.
    params : WilliamsOttoParameters or None, optional
        Parameter set.
    profit_formulation : {"piecewise_mccormick", "mccormick", "bilinear"}, optional
        Profit formulation passed to
        :func:`build_surrogate_optimization_model`. ``fb_segments`` may
        be supplied as a keyword to control the piecewise partition.
    include_operating_constraints : bool, optional
        Add ``XA <= xa_max`` and ``XG <= xg_max`` on the surrogate
        outputs. Default True; set False for the unconstrained baseline.

    Returns
    -------
    SurrogateOptimizationResult
        Solver status plus surrogate and exact Williams-Otto evaluation
        at the selected point. ``solve_seconds`` times only the solver
        call (model construction excluded), and ``operating_constraints``
        records whether the XA/XG limits were included.
    """
    import time

    import pyomo.environ as pyo

    from pwlnet.solvers.solver_config import get_pyomo_solver

    p = params if params is not None else WilliamsOttoParameters()
    model, block = build_surrogate_optimization_model(
        network=network,
        formulation=formulation,
        params=p,
        profit_formulation=profit_formulation,
        include_operating_constraints=include_operating_constraints,
        **formulation_options,
    )
    solver_name = solver or "appsi_highs"
    opt = get_pyomo_solver(
        name=solver_name,
        options=solver_options,
        executable=executable,
    )
    solve_start = time.perf_counter()
    results = opt.solve(model)
    solve_seconds = time.perf_counter() - solve_start

    # Project onto the declared bounds: solvers report values within
    # their feasibility tolerance, so an optimum at a bound can come
    # back a few ulps outside it.
    fb = float(np.clip(pyo.value(model.FB), *p.fb_bounds)) # type: ignore
    tr = float(np.clip(pyo.value(model.TR), *p.tr_bounds)) # type: ignore
    output_keys = list(block.outputs.keys())
    predicted_outputs = {
        name: float(pyo.value(block.outputs[key])) # type: ignore
        for name, key in zip(OUTPUT_NAMES, output_keys)
    }
    objective_value = float(pyo.value(model.objective)) # type: ignore
    # Exact bilinear economics at the surrogate's predicted outputs; for
    # the McCormick formulations pyo.value(model.profit) would report
    # the relaxed objective instead.
    surrogate_profit = float(
        (p.price_p * predicted_outputs["XP"] + p.price_e * predicted_outputs["XE"])
        * (p.fa + fb)
        - p.cost_a * p.fa
        - p.cost_b * fb
    )

    exact_state = solve_steady_state(fb, tr, params=p)
    exact_outputs = {
        name: float(value) for name, value in zip(OUTPUT_NAMES, exact_state.x)
    }
    exact_feasible = exact_outputs["XA"] <= p.xa_max and exact_outputs["XG"] <= p.xg_max

    solver_info = getattr(results, "solver", results)
    status = str(getattr(solver_info, "status", "unknown"))
    termination = str(getattr(solver_info, "termination_condition", "unknown"))
    if termination == "unknown":
        termination = str(getattr(results, "termination_condition", "unknown"))

    return SurrogateOptimizationResult(
        solver_name=solver_name,
        status=status,
        termination_condition=termination,
        fb=fb,
        tr=tr,
        predicted_outputs=predicted_outputs,
        surrogate_profit=surrogate_profit,
        exact_outputs=exact_outputs,
        exact_profit=float(exact_state.profit),
        exact_feasible=bool(exact_feasible),
        exact_optimum=dict(EXACT_OPTIMUM),
        objective_value=objective_value,
        solve_seconds=float(solve_seconds),
        operating_constraints=bool(include_operating_constraints),
    )
