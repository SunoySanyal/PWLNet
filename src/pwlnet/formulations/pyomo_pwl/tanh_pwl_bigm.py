"""Manual Pyomo big-M formulation for PWL-tanh neural networks.

Unlike the SOS2 variant in
:mod:`pwlnet.formulations.pyomo_pwl.tanh_pwl_sos2`, which needs a solver
with native SOS2 support (Gurobi, CPLEX, CBC, SCIP), the big-M segment
formulation uses ordinary binary variables and is therefore solvable by
HiGHS and MOSEK, the project's default open-source and conic solvers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.core.bounds import propagate_bounds
from pwlnet.core.network import NetworkDefinition
from pwlnet.formulations.pyomo_pwl.piecewise_data import (
    PiecewiseLinearData,
    build_tanh_pwl_data,
    extend_piecewise_linear_data,
)
from pwlnet.utils.exceptions import FormulationError
from pwlnet.utils.typing import ArrayLike

__all__ = [
    "build_tanh_pwl_bigm_block",
    "build_tanh_pwl_bigm_model",
]

_SEGMENT_TOL = 1e-9


def _validate_network(network: NetworkDefinition) -> None:
    if network.input_bounds is None:
        raise FormulationError(
            "Cannot build a PWL-tanh formulation without input bounds."
        )
    hidden = network.activations[:-1]
    if any(act != "tanh" for act in hidden):
        raise FormulationError(
            "tanh_pwl_bigm requires all hidden layers to use tanh activation."
        )
    if network.activations[-1] not in {"linear", "tanh"}:
        raise FormulationError(
            "tanh_pwl_bigm supports only linear or tanh output activations."
        )


def _scaler_arrays(network: NetworkDefinition) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    in_offset = (
        np.asarray(network.input_scaler.offset, dtype=float)
        if network.input_scaler is not None
        else np.zeros(network.n_inputs)
    )
    in_factor = (
        np.asarray(network.input_scaler.factor, dtype=float)
        if network.input_scaler is not None
        else np.ones(network.n_inputs)
    )
    out_offset = (
        np.asarray(network.output_scaler.offset, dtype=float)
        if network.output_scaler is not None
        else np.zeros(network.n_outputs)
    )
    out_factor = (
        np.asarray(network.output_scaler.factor, dtype=float)
        if network.output_scaler is not None
        else np.ones(network.n_outputs)
    )
    return in_offset, in_factor, out_offset, out_factor


def _overlapping_segments(
    pwl: PiecewiseLinearData, lb: float, ub: float
) -> list[int]:
    """Indices of PWL segments intersecting ``[lb, ub]`` with width > tol."""
    segs = [
        s
        for s in range(pwl.n_segments)
        if pwl.breakpoints[s] < ub - _SEGMENT_TOL
        and pwl.breakpoints[s + 1] > lb + _SEGMENT_TOL
    ]
    if not segs:
        # Degenerate (near-)point interval: keep the segment containing it.
        segs = [int(pwl.segment_index(0.5 * (lb + ub)))]
    return segs


def build_tanh_pwl_bigm_block(
    model: Any,
    network: NetworkDefinition,
    *,
    block_name: str = "surrogate",
    breakpoints: int | ArrayLike = 5,
) -> Any:
    """Attach a manual big-M PWL-tanh block to a Pyomo model.

    The block exposes ``block.inputs`` and ``block.outputs`` in original
    units, matching the OMLT and SOS2 block conventions used elsewhere
    in PWLNet. Each tanh neuron selects one PWL segment through binary
    variables; big-M constants are derived from interval bound
    propagation, and neurons whose pre-activation interval fits inside a
    single segment are formulated without binaries.
    """
    _validate_network(network)

    import pyomo.environ as pyo

    if model.component(block_name) is not None:
        raise FormulationError(
            f"Model already has a component named '{block_name}'; choose another name."
        )

    pwl = build_tanh_pwl_data(breakpoints)
    bounds = propagate_bounds(network)
    in_offset, in_factor, out_offset, out_factor = _scaler_arrays(network)

    block = pyo.Block()
    setattr(model, block_name, block)

    input_bounds = list(network.input_bounds or [])
    scaled_input_bounds = network.scaled_input_bounds()
    block.input_index = pyo.RangeSet(0, network.n_inputs - 1)
    block.output_index = pyo.RangeSet(0, network.n_outputs - 1)
    block.inputs = pyo.Var(
        block.input_index,
        bounds=lambda _, i: input_bounds[int(i)],
    )
    block.outputs = pyo.Var(block.output_index)
    block.scaled_inputs = pyo.Var(
        block.input_index,
        bounds=lambda _, i: scaled_input_bounds[int(i)],
    )
    block.scaled_outputs = pyo.Var(block.output_index)

    block.input_scaling = pyo.Constraint(
        block.input_index,
        rule=lambda b, i: b.scaled_inputs[i]
        == (b.inputs[i] - float(in_offset[int(i)])) / float(in_factor[int(i)]),
    )
    block.output_scaling = pyo.Constraint(
        block.output_index,
        rule=lambda b, i: b.outputs[i]
        == b.scaled_outputs[i] * float(out_factor[int(i)]) + float(out_offset[int(i)]),
    )

    prev = block.scaled_inputs
    for layer_idx, layer in enumerate(network.layers):
        j_set = pyo.RangeSet(0, layer.n_outputs - 1)
        setattr(block, f"layer_{layer_idx}_index", j_set)

        pre_bounds = bounds.pre_activation_bounds[layer_idx]
        post_bounds = bounds.post_activation_bounds[layer_idx]
        if layer.activation == "tanh":
            # Flat saturation extension instead of clamping; see the
            # matching comment in tanh_pwl_sos2.
            layer_pwl = extend_piecewise_linear_data(
                pwl,
                float(np.min(pre_bounds[0])),
                float(np.max(pre_bounds[1])),
            )
            pre_lb, pre_ub = pre_bounds
            post_at_lb = layer_pwl.evaluate(pre_bounds[0])
            post_at_ub = layer_pwl.evaluate(pre_bounds[1])
            post_lb = np.minimum(post_at_lb, post_at_ub)
            post_ub = np.maximum(post_at_lb, post_at_ub)
        else:
            layer_pwl = pwl
            pre_lb, pre_ub = pre_bounds
            post_lb, post_ub = post_bounds

        pre = pyo.Var(
            j_set,
            bounds=lambda _, j, lb=pre_lb, ub=pre_ub: (float(lb[int(j)]), float(ub[int(j)])),
        )
        post = pyo.Var(
            j_set,
            bounds=lambda _, j, lb=post_lb, ub=post_ub: (float(lb[int(j)]), float(ub[int(j)])),
        )
        setattr(block, f"layer_{layer_idx}_pre", pre)
        setattr(block, f"layer_{layer_idx}_post", post)

        in_range = range(layer.n_inputs)
        aff = pyo.Constraint(
            j_set,
            rule=lambda _, j, layer=layer, prev=prev, pre=pre: pre[j]
            == sum(float(layer.weights[i, int(j)]) * prev[i] for i in in_range)
            + float(layer.biases[int(j)]),
        )
        setattr(block, f"layer_{layer_idx}_affine", aff)

        if layer.activation == "linear":
            act = pyo.Constraint(j_set, rule=lambda _, j, pre=pre, post=post: post[j] == pre[j])
            setattr(block, f"layer_{layer_idx}_linear", act)
        elif layer.activation == "tanh":
            setattr(block, f"layer_{layer_idx}_pwl", layer_pwl)
            _add_tanh_bigm_activation(
                block,
                layer_idx,
                layer.n_outputs,
                pre,
                post,
                layer_pwl,
                pre_lb,
                pre_ub,
                post_lb,
                post_ub,
            )
        else:
            raise FormulationError(
                f"Unsupported activation '{layer.activation}' in tanh_pwl_bigm."
            )
        prev = post

    last_post = getattr(block, f"layer_{network.n_layers - 1}_post")
    block.output_linking = pyo.Constraint(
        block.output_index,
        rule=lambda b, i: b.scaled_outputs[i] == last_post[i],
    )
    block.pwl_data = pwl
    return block


def _add_tanh_bigm_activation(
    block: Any,
    layer_idx: int,
    n_neurons: int,
    pre: Any,
    post: Any,
    pwl: PiecewiseLinearData,
    pre_lb: np.ndarray,
    pre_ub: np.ndarray,
    post_lb: np.ndarray,
    post_ub: np.ndarray,
) -> None:
    import pyomo.environ as pyo

    constraints = pyo.ConstraintList()
    setattr(block, f"layer_{layer_idx}_bigm", constraints)

    delta_index: list[tuple[int, int]] = []
    neuron_segments: dict[int, list[int]] = {}
    for j in range(n_neurons):
        segs = _overlapping_segments(pwl, float(pre_lb[j]), float(pre_ub[j]))
        neuron_segments[j] = segs
        if len(segs) > 1:
            delta_index.extend((j, s) for s in segs)

    delta = pyo.Var(delta_index, domain=pyo.Binary)
    setattr(block, f"layer_{layer_idx}_delta", delta)

    for j in range(n_neurons):
        lb, ub = float(pre_lb[j]), float(pre_ub[j])
        y_lb, y_ub = float(post_lb[j]), float(post_ub[j])
        segs = neuron_segments[j]

        if len(segs) == 1:
            s = segs[0]
            constraints.add(
                post[j]
                == float(pwl.slopes[s]) * pre[j] + float(pwl.intercepts[s])
            )
            continue

        constraints.add(sum(delta[j, s] for s in segs) == 1)
        for s in segs:
            x_lo = float(pwl.breakpoints[s])
            x_hi = float(pwl.breakpoints[s + 1])
            slope = float(pwl.slopes[s])
            intercept = float(pwl.intercepts[s])

            # Domain selection: delta[j, s] == 1 forces pre[j] into the
            # segment; constraints redundant w.r.t. variable bounds are
            # skipped to keep the LP relaxation small.
            if x_lo > lb + _SEGMENT_TOL:
                constraints.add(
                    pre[j] >= x_lo - (x_lo - lb) * (1 - delta[j, s])
                )
            if x_hi < ub - _SEGMENT_TOL:
                constraints.add(
                    pre[j] <= x_hi + (ub - x_hi) * (1 - delta[j, s])
                )

            # Value selection: when delta[j, s] == 1, the inequalities
            # collapse to post[j] == slope * pre[j] + intercept. The
            # big-M constants bound the gap between the segment's affine
            # function and the neuron's output range over [lb, ub].
            lin_lo = min(slope * lb + intercept, slope * ub + intercept)
            lin_hi = max(slope * lb + intercept, slope * ub + intercept)
            m_upper = max(0.0, y_ub - lin_lo)
            m_lower = max(0.0, lin_hi - y_lb)
            constraints.add(
                post[j]
                <= slope * pre[j] + intercept + m_upper * (1 - delta[j, s])
            )
            constraints.add(
                post[j]
                >= slope * pre[j] + intercept - m_lower * (1 - delta[j, s])
            )


def build_tanh_pwl_bigm_model(
    network: NetworkDefinition,
    *,
    breakpoints: int | ArrayLike = 5,
    block_name: str = "neural_net",
) -> tuple[Any, Any]:
    """Create a fresh Pyomo model containing one PWL-tanh big-M block."""
    import pyomo.environ as pyo

    model = pyo.ConcreteModel(name="pwlnet_tanh_pwl_bigm")
    block = build_tanh_pwl_bigm_block(
        model,
        network,
        block_name=block_name,
        breakpoints=breakpoints,
    )
    return model, block
