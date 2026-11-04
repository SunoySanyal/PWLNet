"""Unroll a ReLU RNN cell into a receding-horizon MPC MILP.

The cell's hidden update is embedded with the existing ``relu_bigm``
OMLT formulation (one binary per hidden unit per step); the recurrence
is realised by *chaining* one block per step and equating each block's
hidden output with the next step's hidden input. The linear readout is
added as plain Pyomo constraints (it never needs a big-M).

Decision variables (all in the natural units of the cell):

* ``u[t, i]`` -- exogenous / control inputs, ``t = 0..H-1`` (original
  units, bounded by ``cell.exog_bounds``);
* ``h[t, j]`` -- hidden state, ``t = 0..H`` (latent units). ``h[0]`` is
  fixed to the supplied warm-started ``h0``; ``h[t]`` for ``t >= 1`` is
  bounded by the per-step boxes from :mod:`pwlnet.recurrent.bounds`;
* ``y[t, k]`` -- observable outputs, ``t = 0..H-1`` (original units),
  defined by ``y[t] = readout(h[t+1])``.

Block ``t`` (an OMLT ``relu_bigm`` block built from
:meth:`RecurrentCell.step_network` with step ``t``'s tight hidden box)
is wired as::

    block.inputs[0 .. n_exog-1]            == u[t, .]
    block.inputs[n_exog .. n_exog+n_h-1]   == h[t, .]
    h[t+1, .]                              == block.outputs[.]

An optional setpoint-tracking objective (L1, so the model stays a MILP)
plus an L1 move penalty are provided for convenience; pass
``setpoint=None`` to leave the objective to the caller.

All pyomo / omlt imports are lazy (inside functions).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from pwlnet.recurrent.bounds import RecurrentBounds, propagate_recurrent_bounds
from pwlnet.utils.exceptions import NetworkValidationError
from pwlnet.utils.typing import ArrayLike, BoundsList

if TYPE_CHECKING:
    from pwlnet.recurrent.cell import RecurrentCell

__all__ = ["build_unrolled_mpc_model", "solve_unrolled_mpc"]


def build_unrolled_mpc_model(
    cell: "RecurrentCell",
    h0: ArrayLike,
    horizon: int,
    *,
    recurrent_bounds: RecurrentBounds | None = None,
    exog_bounds: BoundsList | None = None,
    setpoint: ArrayLike | None = None,
    output_weights: ArrayLike | None = None,
    move_penalty: float = 0.0,
    u_prev: ArrayLike | None = None,
    block_prefix: str = "cell",
) -> tuple[Any, dict[str, Any]]:
    """Build the unrolled MPC model and return ``(model, handles)``.

    Parameters
    ----------
    cell : RecurrentCell
        The trained cell to unroll.
    h0 : array_like
        Warm-started initial hidden state, shape ``(n_hidden,)`` (use
        :meth:`RecurrentCell.encode` to obtain it from a control history).
    horizon : int
        Number of steps ``H`` to unroll (``>= 1``).
    recurrent_bounds : RecurrentBounds or None
        Pre-computed per-step bounds. Computed for ``horizon`` steps when
        ``None``.
    exog_bounds : list of (float, float) or None
        Override box for ``u`` (original units). Defaults to
        ``cell.exog_bounds``.
    setpoint : array_like or None
        Target for the observable output ``y`` (original units), shape
        ``(n_outputs,)`` (applied at every step). ``None`` adds no
        objective.
    output_weights : array_like or None
        Per-output weights for the L1 tracking term, shape
        ``(n_outputs,)``. Defaults to all ones.
    move_penalty : float
        Weight on the L1 control-move penalty ``sum_t |u[t] - u[t-1]|``.
    u_prev : array_like or None
        Previously applied control (shape ``(n_exog,)``) used for the
        first move ``|u[0] - u_prev|``. Ignored when ``move_penalty`` is
        zero or ``u_prev`` is ``None``.
    block_prefix : str
        Prefix for the per-step OMLT block names.

    Returns
    -------
    (model, handles)
        ``model`` is a ``pyomo.environ.ConcreteModel``; ``handles`` maps
        ``"u"``, ``"h"``, ``"y"`` to the Pyomo ``Var`` components,
        ``"blocks"`` to the list of per-step OMLT blocks, and
        ``"recurrent_bounds"`` to the :class:`RecurrentBounds` used.

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If ``horizon < 1``, ``h0`` has the wrong shape, or the cell lacks
        the bound boxes needed for embedding.
    """
    import pyomo.environ as pyo

    from pwlnet.formulations.omlt import build_omlt_block

    H = int(horizon)
    if H < 1:
        raise NetworkValidationError(f"horizon must be >= 1, got {horizon}.")

    h0_arr = np.asarray(h0, dtype=float).ravel()
    if h0_arr.shape != (cell.n_hidden,):
        raise NetworkValidationError(
            f"h0 must have {cell.n_hidden} entries, got {h0_arr.shape}."
        )

    u_box = exog_bounds if exog_bounds is not None else cell.exog_bounds
    if u_box is None:
        raise NetworkValidationError(
            "cell.exog_bounds is None; control bounds are required to unroll."
        )

    # Seed bound propagation from the ACTUAL initial hidden state (a point
    # box): h0 is fixed at node 0, so this gives per-step boxes that are
    # valid for THIS solve and -- unlike the generic training box -- are
    # guaranteed to contain h0 (no node-0 bound conflict).
    rb = recurrent_bounds or propagate_recurrent_bounds(
        cell,
        H,
        hidden_init_bounds=[(float(v), float(v)) for v in h0_arr],
        exog_bounds=u_box,
    )

    n_exog, n_hidden, n_out = cell.n_exog, cell.n_hidden, cell.n_outputs

    m = pyo.ConcreteModel(name="relu_rnn_mpc")
    m.step_set = pyo.RangeSet(0, H - 1)
    m.node_set = pyo.RangeSet(0, H)
    m.exog_set = pyo.RangeSet(0, n_exog - 1)
    m.hidden_set = pyo.RangeSet(0, n_hidden - 1)
    m.out_set = pyo.RangeSet(0, n_out - 1)

    # --- Control variables, bounded by the exogenous box. ---
    m.u = pyo.Var(
        m.step_set,
        m.exog_set,
        bounds=lambda _m, _t, i: (float(u_box[i][0]), float(u_box[i][1])),
    )

    # --- Hidden-state variables, bounded by the per-step reachable box. ---
    def _h_bounds(_m: Any, t: int, j: int) -> tuple[float, float]:
        lo, hi = rb.hidden_bounds[t]
        return (float(lo[j]), float(hi[j]))

    m.h = pyo.Var(m.node_set, m.hidden_set, bounds=_h_bounds)
    for j in range(n_hidden):
        m.h[0, j].fix(float(h0_arr[j]))  # warm-started latent initial state

    # --- Observable outputs, bounded by the propagated output box. ---
    def _y_bounds(_m: Any, t: int, k: int) -> tuple[float, float]:
        lo, hi = rb.output_bounds[t]
        return (float(lo[k]), float(hi[k]))

    m.y = pyo.Var(m.step_set, m.out_set, bounds=_y_bounds)

    m.link = pyo.ConstraintList()
    blocks: list[Any] = []

    for t in range(H):
        # One relu_bigm block per step, each with ITS OWN tight hidden box
        # so the big-M constants shrink step by step.
        step_net = cell.step_network(rb.hidden_box(t))
        _model, block = build_omlt_block(
            step_net,
            formulation="relu_bigm",
            model=m,
            block_name=f"{block_prefix}_{t}",
        )
        blocks.append(block)

        # Wire the concatenated input [u_t, h_t] into the block inputs ...
        for i in range(n_exog):
            m.link.add(block.inputs[i] == m.u[t, i])
        for j in range(n_hidden):
            m.link.add(block.inputs[n_exog + j] == m.h[t, j])
        # ... and carry the block's hidden output to the next node.
        for j in range(n_hidden):
            m.link.add(m.h[t + 1, j] == block.outputs[j])

        # Linear readout y_t = readout(h_{t+1}) in original units.
        if cell.readout is None:
            for k in range(n_out):
                m.link.add(m.y[t, k] == m.h[t + 1, k])
        else:
            W = cell.readout.weights  # (n_hidden, n_out)
            b = cell.readout.biases
            for k in range(n_out):
                m.link.add(
                    m.y[t, k]
                    == sum(m.h[t + 1, j] * float(W[j, k]) for j in range(n_hidden))
                    + float(b[k])
                )

    _attach_objective(
        m,
        cell,
        H,
        setpoint=setpoint,
        output_weights=output_weights,
        move_penalty=move_penalty,
        u_prev=u_prev,
    )

    handles = {
        "u": m.u,
        "h": m.h,
        "y": m.y,
        "blocks": blocks,
        "recurrent_bounds": rb,
    }
    return m, handles


def _attach_objective(
    m: Any,
    cell: "RecurrentCell",
    horizon: int,
    *,
    setpoint: ArrayLike | None,
    output_weights: ArrayLike | None,
    move_penalty: float,
    u_prev: ArrayLike | None,
) -> None:
    """Attach an L1 tracking + L1 move-penalty objective (MILP-friendly).

    L1 deviations are linearised with auxiliary non-negative variables and
    pairs of inequalities, so the objective stays linear and the model
    remains a pure MILP (no quadratic terms). No objective is added when
    ``setpoint`` is ``None`` and ``move_penalty`` is zero.
    """
    import pyomo.environ as pyo

    terms: list[Any] = []

    if setpoint is not None:
        target = np.asarray(setpoint, dtype=float).ravel()
        if target.shape != (cell.n_outputs,):
            raise NetworkValidationError(
                f"setpoint must have {cell.n_outputs} entries, got {target.shape}."
            )
        weights = (
            np.ones(cell.n_outputs)
            if output_weights is None
            else np.asarray(output_weights, dtype=float).ravel()
        )
        if weights.shape != (cell.n_outputs,):
            raise NetworkValidationError(
                f"output_weights must have {cell.n_outputs} entries, got "
                f"{weights.shape}."
            )
        # |y[t,k] - target[k]| via dev >= +(.), dev >= -(.).
        m.track_dev = pyo.Var(m.step_set, m.out_set, bounds=(0.0, None))
        m.track_link = pyo.ConstraintList()
        for t in range(horizon):
            for k in range(cell.n_outputs):
                m.track_link.add(m.track_dev[t, k] >= m.y[t, k] - float(target[k]))
                m.track_link.add(m.track_dev[t, k] >= float(target[k]) - m.y[t, k])
                terms.append(float(weights[k]) * m.track_dev[t, k])

    if move_penalty > 0.0:
        prev = None if u_prev is None else np.asarray(u_prev, dtype=float).ravel()
        m.move_dev = pyo.Var(m.step_set, m.exog_set, bounds=(0.0, None))
        m.move_link = pyo.ConstraintList()
        for t in range(horizon):
            for i in range(cell.n_exog):
                if t == 0:
                    if prev is None:
                        continue  # no reference for the first move
                    reference: Any = float(prev[i])
                else:
                    reference = m.u[t - 1, i]
                m.move_link.add(m.move_dev[t, i] >= m.u[t, i] - reference)
                m.move_link.add(m.move_dev[t, i] >= reference - m.u[t, i])
                terms.append(float(move_penalty) * m.move_dev[t, i])

    if terms:
        m.objective = pyo.Objective(expr=sum(terms), sense=pyo.minimize)


def _count_binary_variables(model: Any) -> int:
    """Number of binary variables on the model (relu_bigm indicators)."""
    import pyomo.environ as pyo

    return sum(1 for v in model.component_data_objects(pyo.Var) if v.is_binary())


def solve_unrolled_mpc(
    cell: "RecurrentCell",
    h0: ArrayLike,
    horizon: int,
    *,
    setpoint: ArrayLike | None = None,
    output_weights: ArrayLike | None = None,
    move_penalty: float = 0.0,
    u_prev: ArrayLike | None = None,
    recurrent_bounds: RecurrentBounds | None = None,
    exog_bounds: BoundsList | None = None,
    solver: str | None = None,
    solver_options: dict[str, Any] | None = None,
    time_limit: float | None = 20.0,
    mip_gap: float | None = 0.01,
) -> dict[str, Any]:
    """Build and solve one MPC move; return the first control and forecast.

    Parameters
    ----------
    cell, h0, horizon, setpoint, output_weights, move_penalty, u_prev,
    recurrent_bounds, exog_bounds
        See :func:`build_unrolled_mpc_model`.
    solver : str or None
        MILP solver name (default ``"appsi_highs"``).
    solver_options : dict or None
        Extra options forwarded to the solver.
    time_limit : float or None
        Wall-clock limit in seconds (added as ``TimeLimit`` if set).
    mip_gap : float or None
        Relative MIP gap (added as ``MIPGap`` if set).

    Returns
    -------
    dict
        ``"u0"`` (first control, original units, or ``None`` if no
        incumbent), ``"u"`` (full control plan), ``"y"`` (predicted
        outputs), ``"hidden"`` (predicted hidden trajectory),
        ``"status"``, ``"termination"``, ``"solve_seconds"``,
        ``"n_binaries"``, and ``"objective"``.

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        Propagated from the model builder on shape/bound errors.
    """
    import pyomo.environ as pyo

    from pwlnet.solvers.solver_config import get_milp_solver

    model, handles = build_unrolled_mpc_model(
        cell,
        h0,
        horizon,
        recurrent_bounds=recurrent_bounds,
        exog_bounds=exog_bounds,
        setpoint=setpoint,
        output_weights=output_weights,
        move_penalty=move_penalty,
        u_prev=u_prev,
    )
    if not any(
        True
        for _ in model.component_data_objects(pyo.Objective, active=True)
    ):
        raise NetworkValidationError(
            "No objective on the unrolled model: pass a setpoint (and/or a "
            "move_penalty with u_prev) so solve_unrolled_mpc has something to "
            "minimise, or build the model and set your own objective."
        )

    options = dict(solver_options or {})
    if time_limit is not None:
        options.setdefault("TimeLimit", float(time_limit))
    if mip_gap is not None:
        options.setdefault("MIPGap", float(mip_gap))
    opt = get_milp_solver(solver or "appsi_highs", options=options)

    t0 = time.perf_counter()
    result = opt.solve(model, tee=False, load_solutions=False)
    elapsed = time.perf_counter() - t0
    status = str(getattr(result.solver, "status", "unknown"))
    termination = str(getattr(result.solver, "termination_condition", "unknown"))
    n_binaries = _count_binary_variables(model)

    if len(result.solution) == 0:
        return {
            "u0": None,
            "u": None,
            "y": None,
            "hidden": None,
            "status": status,
            "termination": termination,
            "solve_seconds": elapsed,
            "n_binaries": n_binaries,
            "objective": None,
        }

    model.solutions.load_from(result)
    H = int(horizon)
    u_plan = np.array(
        [[pyo.value(handles["u"][t, i]) for i in range(cell.n_exog)] for t in range(H)],
        dtype=float,
    )
    y_pred = np.array(
        [[pyo.value(handles["y"][t, k]) for k in range(cell.n_outputs)] for t in range(H)],
        dtype=float,
    )
    hidden = np.array(
        [
            [pyo.value(handles["h"][t, j]) for j in range(cell.n_hidden)]
            for t in range(H + 1)
        ],
        dtype=float,
    )
    return {
        "u0": u_plan[0],
        "u": u_plan,
        "y": y_pred,
        "hidden": hidden,
        "status": status,
        "termination": termination,
        "solve_seconds": elapsed,
        "n_binaries": n_binaries,
        "objective": float(pyo.value(model.objective)),
    }
