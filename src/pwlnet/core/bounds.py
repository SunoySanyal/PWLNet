"""Interval bound propagation through PWLNet networks.

Standard interval arithmetic on the SCALED network: the weight matrix
is split into positive and negative parts so each pre-activation
interval is exact for a single affine layer, and the (monotone)
activations relu/tanh/linear are applied to the interval endpoints.
Bounds are generally conservative (not tight) for multi-layer networks.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pwlnet.core.activations import get_activation
from pwlnet.core.network import NetworkDefinition
from pwlnet.utils.typing import BoundsList

__all__ = ["BoundsResult", "propagate_bounds"]


@dataclass
class BoundsResult:
    """Per-layer bounds produced by :func:`propagate_bounds`.

    Attributes
    ----------
    scaled_input_bounds : list of (float, float)
        Input box in SCALED space the propagation started from.
    pre_activation_bounds : list of (numpy.ndarray, numpy.ndarray)
        For each layer ``k``, a ``(lb, ub)`` pair of arrays of shape
        ``(n_outputs_k,)`` bounding the pre-activation values
        ``h_k @ W_k + b_k`` in scaled space.
    post_activation_bounds : list of (numpy.ndarray, numpy.ndarray)
        For each layer ``k``, bounds on the post-activation values in
        scaled space.
    output_bounds : list of (float, float)
        Bounds on the network outputs in ORIGINAL (unscaled) space,
        i.e. after inverse output scaling of the last layer's
        post-activation bounds.
    """

    scaled_input_bounds: BoundsList
    pre_activation_bounds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    post_activation_bounds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    output_bounds: BoundsList = field(default_factory=list)


def _affine_interval(
    lb: np.ndarray, ub: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact interval image of ``x @ W + b`` for ``x`` in ``[lb, ub]``.

    Parameters
    ----------
    lb, ub : numpy.ndarray
        Componentwise input bounds, shape ``(n_in,)``.
    weights : numpy.ndarray
        Weight matrix, shape ``(n_in, n_out)``.
    biases : numpy.ndarray
        Bias vector, shape ``(n_out,)``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Lower and upper pre-activation bounds, each shape ``(n_out,)``.
    """
    w_pos = np.maximum(weights, 0.0)
    w_neg = np.minimum(weights, 0.0)
    pre_lb = lb @ w_pos + ub @ w_neg + biases
    pre_ub = ub @ w_pos + lb @ w_neg + biases
    return pre_lb, pre_ub


def propagate_bounds(network: NetworkDefinition) -> BoundsResult:
    """Propagate interval bounds layer by layer through a network.

    Starts from ``network.scaled_input_bounds()`` and pushes an axis-
    aligned box through every dense layer using interval arithmetic
    (positive/negative weight split), applying the monotone activation
    to the interval endpoints. The final post-activation interval is
    mapped back to original units through the output scaler.

    Parameters
    ----------
    network : NetworkDefinition
        Network with ``input_bounds`` set (in original space).

    Returns
    -------
    BoundsResult
        Per-layer pre-/post-activation bounds (scaled space) and output
        bounds in original space.

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If the network has no input bounds.
    """
    scaled_bounds = network.scaled_input_bounds()
    lb = np.array([b[0] for b in scaled_bounds], dtype=float)
    ub = np.array([b[1] for b in scaled_bounds], dtype=float)

    result = BoundsResult(scaled_input_bounds=scaled_bounds)

    for layer in network.layers:
        pre_lb, pre_ub = _affine_interval(lb, ub, layer.weights, layer.biases)
        result.pre_activation_bounds.append((pre_lb, pre_ub))
        act = get_activation(layer.activation)
        # relu, tanh and linear are all monotone non-decreasing, so the
        # interval image is [act(lb), act(ub)].
        lb = act(pre_lb)
        ub = act(pre_ub)
        result.post_activation_bounds.append((lb, ub))

    out_lb, out_ub = lb, ub
    if network.output_scaler is not None:
        a = network.output_scaler.inverse_transform(out_lb)
        b = network.output_scaler.inverse_transform(out_ub)
        out_lb = np.minimum(a, b)
        out_ub = np.maximum(a, b)
    result.output_bounds = [(float(l), float(u)) for l, u in zip(out_lb, out_ub)]
    return result
