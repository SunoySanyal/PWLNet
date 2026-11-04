"""Manual Pyomo SOS2 formulation for tanh neural networks."""

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
    "build_tanh_pwl_sos2_block",
    "build_tanh_pwl_sos2_model",
    "evaluate_tanh_pwl_network",
]


def _validate_network(network: NetworkDefinition) -> None:
    if network.input_bounds is None:
        raise FormulationError(
            "Cannot build a PWL-tanh formulation without input bounds."
        )
    hidden = network.activations[:-1]
    if any(act != "tanh" for act in hidden):
        raise FormulationError(
            "tanh_pwl_sos2 requires all hidden layers to use tanh activation."
        )
    if network.activations[-1] not in {"linear", "tanh"}:
        raise FormulationError(
            "tanh_pwl_sos2 supports only linear or tanh output activations."
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


def build_tanh_pwl_sos2_block(
    model: Any,
    network: NetworkDefinition,
    *,
    block_name: str = "surrogate",
    breakpoints: int | ArrayLike = 5,
) -> Any:
    """Attach a manual SOS2 PWL-tanh block to a Pyomo model.

    The block exposes ``block.inputs`` and ``block.outputs`` in original
    units, matching the OMLT block convention used elsewhere in PWLNet.
    Internally, scaling variables connect those original-unit variables
    to the scaled network represented by affine dense-layer constraints
    and SOS2 convex-combination tanh approximations.
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
            # Extend the PWL data with flat saturation segments instead of
            # clamping the pre-activation variable into the breakpoint
            # domain: clamping would cut off feasible operating points
            # whenever bound propagation shows the pre-activation can
            # leave [-4, 4], while the flat extension reproduces the
            # clipping behavior of ``PiecewiseLinearData.evaluate``.
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
            _add_tanh_sos2_activation(block, layer_idx, j_set, pre, post, layer_pwl)
        else:
            raise FormulationError(
                f"Unsupported activation '{layer.activation}' in tanh_pwl_sos2."
            )
        prev = post

    last_post = getattr(block, f"layer_{network.n_layers - 1}_post")
    block.output_linking = pyo.Constraint(
        block.output_index,
        rule=lambda b, i: b.scaled_outputs[i] == last_post[i],
    )
    block.pwl_data = pwl
    return block


def _add_tanh_sos2_activation(
    block: Any,
    layer_idx: int,
    j_set: Any,
    pre: Any,
    post: Any,
    pwl: PiecewiseLinearData,
) -> None:
    import pyomo.environ as pyo

    k_set = pyo.RangeSet(0, pwl.n_breakpoints - 1)
    setattr(block, f"layer_{layer_idx}_breakpoint_index", k_set)
    lam = pyo.Var(j_set, k_set, bounds=(0.0, 1.0))
    setattr(block, f"layer_{layer_idx}_lambda", lam)

    lambda_sum = pyo.Constraint(
        j_set,
        rule=lambda _, j, lam=lam: sum(lam[j, k] for k in k_set) == 1.0,
    )
    pre_interp = pyo.Constraint(
        j_set,
        rule=lambda _, j, lam=lam, pre=pre: pre[j]
        == sum(float(pwl.breakpoints[int(k)]) * lam[j, k] for k in k_set),
    )
    post_interp = pyo.Constraint(
        j_set,
        rule=lambda _, j, lam=lam, post=post: post[j]
        == sum(float(pwl.values[int(k)]) * lam[j, k] for k in k_set),
    )

    def sos_rule(_, j):
        return [lam[j, k] for k in k_set]

    sos = pyo.SOSConstraint(j_set, rule=sos_rule, sos=2)
    setattr(block, f"layer_{layer_idx}_lambda_sum", lambda_sum)
    setattr(block, f"layer_{layer_idx}_pre_interp", pre_interp)
    setattr(block, f"layer_{layer_idx}_post_interp", post_interp)
    setattr(block, f"layer_{layer_idx}_sos2", sos)


def build_tanh_pwl_sos2_model(
    network: NetworkDefinition,
    *,
    breakpoints: int | ArrayLike = 5,
    block_name: str = "neural_net",
) -> tuple[Any, Any]:
    """Create a fresh Pyomo model containing one PWL-tanh SOS2 block."""
    import pyomo.environ as pyo

    model = pyo.ConcreteModel(name="pwlnet_tanh_pwl_sos2")
    block = build_tanh_pwl_sos2_block(
        model,
        network,
        block_name=block_name,
        breakpoints=breakpoints,
    )
    return model, block


def evaluate_tanh_pwl_network(
    network: NetworkDefinition,
    X: ArrayLike,
    *,
    breakpoints: int | ArrayLike = 5,
) -> np.ndarray:
    """Evaluate the PWL approximation of a tanh network in NumPy."""
    _validate_network(network)
    pwl = build_tanh_pwl_data(breakpoints)
    x_in = np.asarray(X, dtype=float)
    squeeze = x_in.ndim == 1
    arr = np.atleast_2d(x_in).astype(float)
    if arr.shape[1] != network.n_inputs:
        raise FormulationError(
            f"X has {arr.shape[1]} features, expected {network.n_inputs}."
        )
    if network.input_scaler is not None:
        arr = network.input_scaler.transform(arr)

    h = arr
    for layer in network.layers:
        z = h @ layer.weights + layer.biases
        if layer.activation == "linear":
            h = z
        elif layer.activation == "tanh":
            h = pwl.evaluate(z)
        else:
            raise FormulationError(
                f"Unsupported activation '{layer.activation}' in tanh_pwl_sos2."
            )

    if network.output_scaler is not None:
        h = network.output_scaler.inverse_transform(h)
    return h[0] if squeeze else h
