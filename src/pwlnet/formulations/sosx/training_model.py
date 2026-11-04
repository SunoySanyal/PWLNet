"""The SOSX-MAE single-hidden-layer training model (fixed linear readout).

This builds the Pyomo model behind :class:`pwlnet.training.SosxTrainer`.
The supported architecture is

```text
inputs -> trainable tanh hidden layer -> fixed linear readout
```

with trainable input weights ``W1``, hidden biases ``b1`` and output
biases ``b2``, and a *fixed* output weight matrix ``W2``. The hidden
``tanh`` is approximated by the shared PWL-tanh data using a segment-fill
(Y-variable) formulation: each (sample, hidden-neuron) pair gets its own
fill variable ``Y[s, j, k]`` for each PWL segment k, representing how much
of segment k is used.

Formulation (Koksal et al., 2026)
----------------------------------
Segments k = 1, ..., P (P = K-1 where K = number of breakpoints):

    u_k = bp[k] - bp[k-1]          (segment width)
    g_k = val[k] - val[k-1]        (tanh value increment)

Variables:
    Y[s, j, k] in [0, u_k]         (segment fill, NonNegativeReals)

Pre-activation:
    bp[0] + sum_k Y[s,j,k] == W1·x[s] + b1[j]

Hidden value (linear in Y):
    val[0] + sum_k (g_k/u_k) * Y[s,j,k]

Locally ideal constraints (Keha et al., 2004):
    u_m * Y[s,j,m+1] <= u_{m+1} * Y[s,j,m]   for m = 1..P-1

These replace the SOS2 constraint: at a LP optimum with these constraints
the only way Y_{m+1} > 0 is if Y_m = u_m (segment m is full). The brancher
enforces strict SOS2 by branching on the first (s,j,k) where Y_k < u_k
and Y_{k+1} > 0, fixing either Y_k = u_k or Y_{k+1:P} = 0.

Why every node is an LP
-----------------------
With W2 fixed, the prediction ``W2 * hidden + b2`` is linear in Y. The
pre-activation couples W1/b1 to Y linearly. With the locally ideal
constraints (also linear) replacing SOS2, every node is a pure LP in
(W1, b1, b2, Y, abs_error) -- solvable by HiGHS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pwlnet.branching.node import SosKey
from pwlnet.formulations.pyomo_pwl.piecewise_data import (
    PiecewiseLinearData,
    build_tanh_pwl_data,
)
from pwlnet.formulations.sosx.weight_bounds import WeightBounds
from pwlnet.objectives.data_loss import attach_data_loss
from pwlnet.utils.exceptions import FormulationError

__all__ = [
    "SosxTrainingModel",
    "build_sosx_mae_training_model",
    "Sos2TrainingModel",
    "build_sos2_mae_training_model",
]


@dataclass(frozen=True)
class SosxTrainingModel:
    """A built SOSX-MAE training model and the handles the engine needs.

    Attributes
    ----------
    model : pyomo.environ.ConcreteModel
        The training model with one active minimization objective (the
        mean absolute error) and the locally ideal constraints replacing
        the SOS2 structure.
    Y_vars : dict[SosKey, list]
        Per ``("training", sample, neuron)`` key, the ordered list of
        ``Y`` VarData over segments 1..P. The keys match the
        :class:`~pwlnet.branching.node.SosKey` space the brancher expects.
    u_values : dict[SosKey, numpy.ndarray]
        Per SOS key, the array of segment widths ``u_k = bp[k] - bp[k-1]``
        for k = 1..P. Used by the violation detector and apply_windows.
    pwl : PiecewiseLinearData
        The tanh PWL data (breakpoints, values) used to build the model.
    n_inputs, hidden, n_outputs, n_samples : int
        Problem dimensions.
    """

    model: Any
    Y_vars: dict[SosKey, list[Any]]
    u_values: dict[SosKey, np.ndarray]
    pwl: PiecewiseLinearData
    n_inputs: int
    hidden: int
    n_outputs: int
    n_samples: int

    @property
    def n_segments(self) -> int:
        """Number of PWL segments P = n_breakpoints - 1."""
        return self.pwl.n_breakpoints - 1


def build_sosx_mae_training_model(
    X: np.ndarray,
    y: np.ndarray,
    W2: np.ndarray,
    *,
    breakpoints: int = 5,
    bounds: WeightBounds | None = None,
) -> SosxTrainingModel:
    """Build the fixed-readout SOSX-MAE training model with Y variables.

    Parameters
    ----------
    X : numpy.ndarray
        Training inputs in the model's (scaled) space, shape
        ``(n_samples, n_inputs)``.
    y : numpy.ndarray
        Training targets in the model's (scaled) space, shape
        ``(n_samples, n_outputs)``.
    W2 : numpy.ndarray
        Fixed output weight matrix, shape ``(hidden, n_outputs)``.
    breakpoints : int or array_like, optional
        PWL-tanh breakpoint count (or explicit breakpoints). Default 5.
    bounds : WeightBounds or None, optional
        Finite box bounds for ``W1``/``b1``/``b2``. Defaults to the
        :class:`WeightBounds` defaults.

    Returns
    -------
    SosxTrainingModel
        The Pyomo model plus the handles the relaxation/extraction need.
    """
    import pyomo.environ as pyo

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    if X.ndim != 2 or y.ndim != 2:
        raise FormulationError("X and y must be two-dimensional arrays.")
    if X.shape[0] != y.shape[0]:
        raise FormulationError("X and y must contain the same number of samples.")
    if X.shape[0] == 0:
        raise FormulationError("Cannot build a training model with zero samples.")
    if W2.ndim != 2:
        raise FormulationError("W2 must be a two-dimensional (hidden, outputs) array.")
    if W2.shape[1] != y.shape[1]:
        raise FormulationError(
            f"W2 has {W2.shape[1]} output columns but y has {y.shape[1]}."
        )

    box = bounds if bounds is not None else WeightBounds()
    pwl = build_tanh_pwl_data(breakpoints)
    n_samples, n_inputs = X.shape
    hidden, n_outputs = W2.shape
    K = pwl.n_breakpoints   # number of breakpoints
    P = K - 1               # number of segments
    bp = pwl.breakpoints    # shape (K,)
    val = pwl.values        # shape (K,)

    # Segment widths and tanh increments (1-indexed, stored 0-indexed).
    u = np.array([bp[k] - bp[k - 1] for k in range(1, K)], dtype=float)  # u[0]=u_1..u[P-1]=u_P
    g = np.array([val[k] - val[k - 1] for k in range(1, K)], dtype=float)
    slope = g / u  # (g_k / u_k): slope of each segment

    m = pyo.ConcreteModel(name="sosx_mae_training")
    m.S = pyo.RangeSet(0, n_samples - 1)
    m.I = pyo.RangeSet(0, n_inputs - 1)
    m.H = pyo.RangeSet(0, hidden - 1)
    m.O = pyo.RangeSet(0, n_outputs - 1)
    # Segments are 1-indexed in the paper; we use 1..P here to stay consistent.
    m.SEG = pyo.RangeSet(1, P)

    m.W1 = pyo.Var(m.I, m.H, bounds=(-box.weight, box.weight))
    m.b1 = pyo.Var(m.H, bounds=(-box.bias, box.bias))
    m.b2 = pyo.Var(m.O, bounds=(-box.output_bias, box.output_bias))

    # Y[s, j, k] in [0, u_k]: fill of segment k for sample s, neuron j.
    def _y_bounds(_, s, j, k):
        return (0.0, float(u[k - 1]))

    m.Y = pyo.Var(m.S, m.H, m.SEG, bounds=_y_bounds)

    def _hidden_val(s: int, j: int):
        """Hidden activation as a linear function of Y."""
        return float(val[0]) + sum(
            float(slope[k - 1]) * m.Y[s, j, k] for k in m.SEG
        )

    def _prediction(s: int, o: int):
        return sum(float(W2[j, o]) * _hidden_val(s, j) for j in m.H) + m.b2[o]

    # Pre-activation: bp[0] + Σ_k Y[s,j,k] == W1·x[s] + b1[j]
    m.pre_activation = pyo.Constraint(
        m.S,
        m.H,
        rule=lambda _, s, j: float(bp[0]) + sum(m.Y[s, j, k] for k in m.SEG)
        == sum(m.W1[i, j] * float(X[s, i]) for i in m.I) + m.b1[j],
    )

    # Locally ideal constraints: u_seg * Y[s,j,seg+1] <= u_{seg+1} * Y[s,j,seg]
    # for seg = 1..P-1 (ensures you fill lower segments before higher ones).
    if P >= 2:
        def _locally_ideal(model, s, j, seg):
            return float(u[seg - 1]) * model.Y[s, j, seg + 1] <= float(u[seg]) * model.Y[s, j, seg]

        m.locally_ideal = pyo.Constraint(m.S, m.H, pyo.RangeSet(1, P - 1), rule=_locally_ideal)

    # Mean-absolute-error objective via the shared linear epigraph builder.
    predictions = [[_prediction(s, o) for o in m.O] for s in m.S]
    attach_data_loss(
        m,
        predictions,
        y,
        loss="mae",
        name="mae",
        add_objective=True,
        objective_name="objective",
    )

    # Build the SosKey → Y variable list and u_values for the brancher.
    Y_vars: dict[SosKey, list[Any]] = {}
    u_values: dict[SosKey, np.ndarray] = {}
    for s in m.S:
        for j in m.H:
            key: SosKey = ("training", int(s), int(j))
            Y_vars[key] = [m.Y[s, j, k] for k in m.SEG]
            u_values[key] = u.copy()

    return SosxTrainingModel(
        model=m,
        Y_vars=Y_vars,
        u_values=u_values,
        pwl=pwl,
        n_inputs=n_inputs,
        hidden=hidden,
        n_outputs=n_outputs,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# SOS2 / λ-variable formulation (original, for comparison)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sos2TrainingModel:
    """A built SOS2-MAE training model using convex-combination λ variables.

    This is the original formulation where each (sample, neuron) SOS set
    uses a convex-combination weight ``λ[s, j, k] ∈ [0, 1]`` over the K
    breakpoints with Σ λ = 1.  The brancher enforces SOS2 by fixing
    out-of-window λ values to zero via symmetric window bisection.

    Attributes
    ----------
    model : pyomo.environ.ConcreteModel
        The training LP with the SOS2 structure left entirely to the brancher.
    lambda_vars : dict[SosKey, list]
        Per ``("training", sample, neuron)`` key, the ordered list of
        ``λ`` VarData over breakpoints 0..K-1.
    pwl : PiecewiseLinearData
        The tanh PWL data (breakpoints, values) used to build the model.
    n_inputs, hidden, n_outputs, n_samples : int
        Problem dimensions.
    """

    model: Any
    lambda_vars: dict[SosKey, list[Any]]
    pwl: PiecewiseLinearData
    n_inputs: int
    hidden: int
    n_outputs: int
    n_samples: int

    @property
    def n_breakpoints(self) -> int:
        """Number of PWL breakpoints (the λ width per SOS set)."""
        return self.pwl.n_breakpoints


def build_sos2_mae_training_model(
    X: np.ndarray,
    y: np.ndarray,
    W2: np.ndarray,
    *,
    breakpoints: int = 5,
    bounds: WeightBounds | None = None,
) -> Sos2TrainingModel:
    """Build the fixed-readout SOS2-MAE training model with λ variables.

    Uses the convex-combination (λ) formulation: each (sample, neuron) SOS
    set has K breakpoint weights λ[s,j,k] ∈ [0,1] summing to 1.  The SOS2
    constraint (at most two adjacent λ non-zero) is relaxed at each LP node
    and re-imposed by the brancher via symmetric window bisection.

    Parameters
    ----------
    X, y, W2, breakpoints, bounds
        Same as :func:`build_sosx_mae_training_model`.

    Returns
    -------
    Sos2TrainingModel
    """
    import pyomo.environ as pyo

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    if X.ndim != 2 or y.ndim != 2:
        raise FormulationError("X and y must be two-dimensional arrays.")
    if X.shape[0] != y.shape[0]:
        raise FormulationError("X and y must contain the same number of samples.")
    if X.shape[0] == 0:
        raise FormulationError("Cannot build a training model with zero samples.")
    if W2.ndim != 2:
        raise FormulationError("W2 must be a two-dimensional (hidden, outputs) array.")
    if W2.shape[1] != y.shape[1]:
        raise FormulationError(
            f"W2 has {W2.shape[1]} output columns but y has {y.shape[1]}."
        )

    box = bounds if bounds is not None else WeightBounds()
    pwl = build_tanh_pwl_data(breakpoints)
    n_samples, n_inputs = X.shape
    hidden, n_outputs = W2.shape
    K = pwl.n_breakpoints
    bp = pwl.breakpoints
    val = pwl.values

    m = pyo.ConcreteModel(name="sos2_mae_training")
    m.S = pyo.RangeSet(0, n_samples - 1)
    m.I = pyo.RangeSet(0, n_inputs - 1)
    m.H = pyo.RangeSet(0, hidden - 1)
    m.O = pyo.RangeSet(0, n_outputs - 1)
    m.K = pyo.RangeSet(0, K - 1)

    m.W1 = pyo.Var(m.I, m.H, bounds=(-box.weight, box.weight))
    m.b1 = pyo.Var(m.H, bounds=(-box.bias, box.bias))
    m.b2 = pyo.Var(m.O, bounds=(-box.output_bias, box.output_bias))
    m.lam = pyo.Var(m.S, m.H, m.K, bounds=(0.0, 1.0))

    def _hidden_val(s: int, j: int):
        return sum(float(val[k]) * m.lam[s, j, k] for k in m.K)

    def _prediction(s: int, o: int):
        return sum(float(W2[j, o]) * _hidden_val(s, j) for j in m.H) + m.b2[o]

    m.lambda_sum = pyo.Constraint(
        m.S, m.H, rule=lambda _, s, j: sum(m.lam[s, j, k] for k in m.K) == 1.0
    )
    m.pre_activation = pyo.Constraint(
        m.S,
        m.H,
        rule=lambda _, s, j: sum(float(bp[k]) * m.lam[s, j, k] for k in m.K)
        == sum(m.W1[i, j] * float(X[s, i]) for i in m.I) + m.b1[j],
    )

    predictions = [[_prediction(s, o) for o in m.O] for s in m.S]
    attach_data_loss(
        m,
        predictions,
        y,
        loss="mae",
        name="mae",
        add_objective=True,
        objective_name="objective",
    )

    lambda_vars: dict[SosKey, list[Any]] = {
        ("training", int(s), int(j)): [m.lam[s, j, k] for k in m.K]
        for s in m.S
        for j in m.H
    }
    return Sos2TrainingModel(
        model=m,
        lambda_vars=lambda_vars,
        pwl=pwl,
        n_inputs=n_inputs,
        hidden=hidden,
        n_outputs=n_outputs,
        n_samples=n_samples,
    )
