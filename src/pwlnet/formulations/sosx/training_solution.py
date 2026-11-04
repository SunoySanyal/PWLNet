"""Extraction and warm-start helpers for SOSX-MAE training.

These turn the brancher's tracked-variable snapshot back into a
:class:`~pwlnet.core.network.NetworkDefinition`, build a warm-start SOSX
incumbent from an existing network (as Y values matching the paper's
formulation), and evaluate the independent PWL MAE used to cross-check
the SOSX objective.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.branching.node import SosKey
from pwlnet.core.layers import DenseLayer
from pwlnet.core.network import NetworkDefinition
from pwlnet.formulations.pyomo_pwl.piecewise_data import (
    PiecewiseLinearData,
    build_tanh_pwl_data,
)
from pwlnet.formulations.sosx.sosx_solution import SosxResult
from pwlnet.formulations.sosx.training_model import SosxTrainingModel
from pwlnet.utils.exceptions import FormulationError

__all__ = [
    "track_training_variables",
    "network_from_tracked_values",
    "training_incumbent_from_network",
    "training_incumbent_lam",
    "training_incumbent_Y",
    "predict_pwl_tanh",
    "pwl_training_mae",
]

#: Keep adjacent convex weights strictly interior so a warm-start point is
#: an unambiguous SOS2 (two-adjacent-support) seed, not a degenerate vertex.
_ALPHA_EPS = 1e-4


def track_training_variables(training_model: SosxTrainingModel) -> dict[str, Any]:
    """Build the brancher ``track`` map for ``W1``/``b1``/``b2``.

    The keys (``"W1:i:j"``, ``"b1:j"``, ``"b2:o"``) are the contract
    consumed by :func:`network_from_tracked_values`.
    """
    m = training_model.model
    track: dict[str, Any] = {}
    for i in m.I:
        for j in m.H:
            track[f"W1:{int(i)}:{int(j)}"] = m.W1[i, j]
    for j in m.H:
        track[f"b1:{int(j)}"] = m.b1[j]
    for o in m.O:
        track[f"b2:{int(o)}"] = m.b2[o]
    return track


def network_from_tracked_values(
    values: dict[str, float],
    W2: np.ndarray,
    *,
    n_inputs: int,
    hidden: int,
    n_outputs: int,
) -> NetworkDefinition:
    """Rebuild the scaled-space network from a tracked-variable snapshot.

    The returned network carries no scalers (it lives in the training
    model's space); the trainer attaches the input/output scalers.
    """
    W2 = np.asarray(W2, dtype=float)
    W1 = np.array(
        [[values[f"W1:{i}:{j}"] for j in range(hidden)] for i in range(n_inputs)],
        dtype=float,
    )
    b1 = np.array([values[f"b1:{j}"] for j in range(hidden)], dtype=float)
    b2 = np.array([values[f"b2:{o}"] for o in range(n_outputs)], dtype=float)
    return NetworkDefinition(
        layers=[
            DenseLayer(W1, b1, "tanh"),
            DenseLayer(W2, b2, "linear"),
        ]
    )


def training_incumbent_from_network(
    training_model: SosxTrainingModel,
    network: NetworkDefinition,
    X_scaled: np.ndarray,
) -> SosxResult:
    """Convert a one-hidden-layer network into a warm-start SOSX incumbent.

    The pre-activation ``W1·x + b1`` for every (sample, neuron) is
    expressed as Y segment-fill values: segments below the active one are
    fully filled, the active segment is partially filled, all higher
    segments are zero. The brancher revalidates these on the current model
    before the root search; infeasible seeds are silently dropped there.
    """
    if network.n_layers != 2:
        raise FormulationError("initial_network must have one hidden layer.")
    return training_incumbent_Y(network, X_scaled, training_model.pwl)


def training_incumbent_lam(
    network: NetworkDefinition,
    X_scaled: np.ndarray,
    pwl: PiecewiseLinearData,
) -> SosxResult:
    """Warm-start SOS-set weights (λ formulation) from a one-hidden-layer network.

    For the SOS2/λ formulation, each (sample, neuron) pre-activation is
    expressed as a convex combination of its two bracketing breakpoints
    (two-adjacent support, strictly interior to avoid degenerate seeds).
    The brancher revalidates these λ values before the root search.
    """
    if network.n_layers != 2:
        raise FormulationError("initial_network must have one hidden layer.")
    X_scaled = np.asarray(X_scaled, dtype=float)
    bp = pwl.breakpoints
    pre = X_scaled @ network.layers[0].weights + network.layers[0].biases
    pre = np.clip(pre, bp[0], bp[-1])

    lam: dict[SosKey, np.ndarray] = {}
    for s in range(pre.shape[0]):
        for j in range(pre.shape[1]):
            value = float(pre[s, j])
            right = int(np.searchsorted(bp, value, side="right"))
            right = min(max(right, 1), len(bp) - 1)
            left = right - 1
            width = float(bp[right] - bp[left])
            alpha = 0.0 if width == 0.0 else (value - float(bp[left])) / width
            alpha = float(np.clip(alpha, _ALPHA_EPS, 1.0 - _ALPHA_EPS))
            weights = np.zeros(len(bp), dtype=float)
            weights[left] = 1.0 - alpha
            weights[right] = alpha
            lam[("training", s, j)] = weights
    return SosxResult(status="initial", objective=None, lam=lam)


def training_incumbent_Y(
    network: NetworkDefinition,
    X_scaled: np.ndarray,
    pwl: PiecewiseLinearData,
) -> SosxResult:
    """Model-free warm-start: compute Y values from a one-hidden-layer network.

    Builds the warm-start segment-fill Y values from a one-hidden-layer
    network and the shared PWL-tanh data alone, without needing a built
    :class:`SosxTrainingModel`. This is what the *parallel* trainer uses,
    since it never builds the Pyomo model on the driver process.

    For each (sample s, neuron j):
      - Compute pre-activation z = X[s]·W1[:,j] + b1[j], clip to [bp[0], bp[P]]
      - Find the active segment k* where bp[k*-1] ≤ z ≤ bp[k*]
      - Set Y_1=u_1, ..., Y_{k*-1}=u_{k*-1}, Y_{k*}=z-bp[k*-1], Y_{k*+1..P}=0
    """
    if network.n_layers != 2:
        raise FormulationError("initial_network must have one hidden layer.")
    X_scaled = np.asarray(X_scaled, dtype=float)
    bp = pwl.breakpoints
    P = len(bp) - 1  # number of segments
    u = np.array([bp[k] - bp[k - 1] for k in range(1, len(bp))], dtype=float)

    pre = X_scaled @ network.layers[0].weights + network.layers[0].biases
    pre = np.clip(pre, bp[0], bp[-1])

    sos_Y: dict[SosKey, np.ndarray] = {}
    for s in range(pre.shape[0]):
        for j in range(pre.shape[1]):
            z = float(pre[s, j])
            # Find active segment k* (1-indexed): bp[k*-1] <= z <= bp[k*]
            k_star = int(np.searchsorted(bp, z, side="right"))
            k_star = min(max(k_star, 1), P)  # clamp to valid segment range
            Y = np.zeros(P, dtype=float)
            for k in range(1, k_star):         # segments below k*: fully filled
                Y[k - 1] = u[k - 1]
            Y[k_star - 1] = z - float(bp[k_star - 1])  # active segment: partial fill
            # segments above k_star remain 0
            sos_Y[("training", s, j)] = Y

    return SosxResult(status="initial", objective=None, lam=sos_Y)


def predict_pwl_tanh(
    network: NetworkDefinition,
    X: np.ndarray,
    *,
    breakpoints: int = 5,
) -> np.ndarray:
    """Forward pass of a one-hidden-layer tanh net using the shared PWL tanh.

    A scaler-free forward pass on ``X`` in the network's own space (no
    ``input_bounds`` required, unlike
    :func:`pwlnet.formulations.pyomo_pwl.evaluate_tanh_pwl_network`), so it
    works on the bare scaled-space network the trainer builds.
    """
    if network.n_layers != 2 or network.layers[0].activation != "tanh":
        raise FormulationError("Expected a one-hidden-layer tanh network.")
    pwl = build_tanh_pwl_data(breakpoints)
    X = np.asarray(X, dtype=float)
    hidden = pwl.evaluate(X @ network.layers[0].weights + network.layers[0].biases)
    return hidden @ network.layers[1].weights + network.layers[1].biases


def pwl_training_mae(
    network: NetworkDefinition,
    X: np.ndarray,
    y: np.ndarray,
    *,
    breakpoints: int = 5,
) -> float:
    """Mean absolute error of the network's PWL-tanh prediction against ``y``.

    Evaluated with the shared PWL-tanh activation so the result is
    directly comparable to the SOSX training objective. ``X``/``y`` are in
    the network's own (scaler-free) space.
    """
    prediction = predict_pwl_tanh(network, X, breakpoints=breakpoints)
    target = np.asarray(y, dtype=float)
    return float(np.mean(np.abs(prediction - target)))
