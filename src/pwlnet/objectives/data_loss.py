"""Reusable Pyomo data-loss builders: MAE, MSE-QP, and MSE-SOCP.

These builders attach a fit-to-data loss over Pyomo expressions (for
example surrogate block outputs, free output-layer weights in
optimization-based training, or SOSX node variables) against numeric
targets. Following the architecture document's formulation matrix:

* ``mae``       -- linear epigraph variables; keeps MILP nodes MILP.
* ``mse_qp``    -- direct quadratic objective; MIQP.
* ``mse_socp``  -- single conic epigraph variable; MISOCP for solvers
  with native conic support such as MOSEK.

The builders never solve anything; they only add components and return
an expression that callers can weigh and combine with physics losses,
regularization, or pruning penalties.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pwlnet.utils.exceptions import FormulationError

__all__ = [
    "DataLossResult",
    "SUPPORTED_DATA_LOSSES",
    "add_mae_loss",
    "add_mse_qp_loss",
    "add_mse_socp_loss",
    "attach_data_loss",
]

SUPPORTED_DATA_LOSSES: tuple[str, ...] = ("mae", "mse_qp", "mse_socp")
"""Loss names accepted by :func:`attach_data_loss` (``"mse"`` aliases
``"mse_qp"``)."""


@dataclass(frozen=True)
class DataLossResult:
    """Handle to a data-loss term added to a Pyomo model.

    Attributes
    ----------
    loss : str
        Canonical loss name (``"mae"``, ``"mse_qp"``, or ``"mse_socp"``).
    expression : object
        Pyomo expression of the mean loss, suitable for an objective.
    block : object or None
        Auxiliary block holding epigraph variables/constraints, when the
        loss needs one (``mae`` and ``mse_socp``).
    n_residuals : int
        Number of residual terms averaged by ``expression``.
    """

    loss: str
    expression: Any
    block: Any | None
    n_residuals: int


def _residual_pairs(
    predictions: Sequence[Any],
    targets: Any,
) -> list[tuple[Any, float]]:
    """Flatten predictions/targets into aligned (expression, value) pairs.

    ``predictions`` may be a flat sequence of Pyomo variables or
    expressions, or a nested sequence (one inner sequence per sample).
    ``targets`` must contain the same number of finite values.
    """
    flat_predictions: list[Any] = []
    for item in predictions:
        if isinstance(item, (list, tuple)):
            flat_predictions.extend(item)
        else:
            flat_predictions.append(item)
    target_array = np.asarray(targets, dtype=float).ravel()
    if target_array.size != len(flat_predictions):
        raise FormulationError(
            f"Got {len(flat_predictions)} prediction expressions but "
            f"{target_array.size} target values."
        )
    if len(flat_predictions) == 0:
        raise FormulationError("Data losses need at least one residual.")
    if not np.all(np.isfinite(target_array)):
        raise FormulationError("Targets must contain only finite values.")
    return [
        (expr, float(value))
        for expr, value in zip(flat_predictions, target_array)
    ]


def _claim_component_name(model: Any, name: str) -> None:
    if model.component(name) is not None:
        raise FormulationError(
            f"Model already has a component named '{name}'; choose another name."
        )


def add_mae_loss(
    model: Any,
    predictions: Sequence[Any],
    targets: Any,
    *,
    name: str = "mae_loss",
) -> DataLossResult:
    """Add a mean-absolute-error loss with linear epigraph variables.

    For each residual ``r_i = prediction_i - target_i`` an auxiliary
    variable ``t_i`` with ``t_i >= r_i`` and ``t_i >= -r_i`` is added;
    the returned expression is ``sum(t_i) / N``. The loss keeps LP/MILP
    problem classes linear.
    """
    import pyomo.environ as pyo

    pairs = _residual_pairs(predictions, targets)
    _claim_component_name(model, name)

    block = pyo.Block()
    setattr(model, name, block)
    block.residual_index = pyo.RangeSet(0, len(pairs) - 1)
    block.abs_residual = pyo.Var(block.residual_index, domain=pyo.NonNegativeReals)
    block.upper = pyo.Constraint(
        block.residual_index,
        rule=lambda b, i: b.abs_residual[i] >= pairs[int(i)][0] - pairs[int(i)][1],
    )
    block.lower = pyo.Constraint(
        block.residual_index,
        rule=lambda b, i: b.abs_residual[i] >= pairs[int(i)][1] - pairs[int(i)][0],
    )
    expression = sum(block.abs_residual[i] for i in block.residual_index) / len(pairs)
    return DataLossResult(
        loss="mae", expression=expression, block=block, n_residuals=len(pairs)
    )


def add_mse_qp_loss(
    model: Any,
    predictions: Sequence[Any],
    targets: Any,
    *,
    name: str = "mse_qp_loss",
) -> DataLossResult:
    """Add a mean-squared-error loss as a direct quadratic expression.

    No auxiliary components are created; the returned expression is
    ``sum(r_i^2) / N``. Combined with binary variables this yields an
    MIQP. The ``model`` and ``name`` arguments are accepted for
    interface symmetry with the other builders.
    """
    pairs = _residual_pairs(predictions, targets)
    del model, name  # quadratic objective needs no model components
    expression = sum(
        (expr - value) * (expr - value) for expr, value in pairs
    ) / len(pairs)
    return DataLossResult(
        loss="mse_qp", expression=expression, block=None, n_residuals=len(pairs)
    )


def add_mse_socp_loss(
    model: Any,
    predictions: Sequence[Any],
    targets: Any,
    *,
    name: str = "mse_socp_loss",
) -> DataLossResult:
    """Add a mean-squared-error loss through a conic epigraph.

    Adds one nonnegative epigraph variable ``t`` with the convex
    quadratic constraint ``sum(r_i^2) <= t`` and returns ``t / N``. The
    quadratic term moves from the objective into a conic-representable
    constraint, giving an MISOCP that solvers like MOSEK handle natively.
    """
    from pwlnet.formulations.pyomo_pwl.conic_epigraphs import (
        add_quadratic_epigraph,
    )

    pairs = _residual_pairs(predictions, targets)
    epigraph_t = add_quadratic_epigraph(
        model,
        [expr - value for expr, value in pairs],
        name=name,
    )
    expression = epigraph_t / len(pairs)
    return DataLossResult(
        loss="mse_socp",
        expression=expression,
        block=epigraph_t.parent_block(),
        n_residuals=len(pairs),
    )


def attach_data_loss(
    model: Any,
    predictions: Sequence[Any],
    targets: Any,
    loss: str = "mae",
    *,
    weight: float = 1.0,
    name: str = "data_loss",
    add_objective: bool = True,
    objective_name: str = "objective",
) -> DataLossResult:
    """Build a data loss and (optionally) a minimization objective.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel or Block
        Model receiving the loss components.
    predictions : sequence
        Pyomo variables/expressions, flat or nested per sample.
    targets : array_like
        Numeric targets, flattened to match ``predictions``.
    loss : {"mae", "mse", "mse_qp", "mse_socp"}, optional
        Loss formulation; ``"mse"`` is an alias for ``"mse_qp"``.
    weight : float, optional
        Multiplier applied in the objective expression.
    name : str, optional
        Component name for auxiliary loss blocks.
    add_objective : bool, optional
        Add ``Objective(expr=weight * loss, sense=minimize)`` to the
        model. Set to ``False`` to combine the returned expression with
        other terms first.
    objective_name : str, optional
        Component name for the added objective.

    Returns
    -------
    DataLossResult
        Handle with the mean-loss expression.
    """
    import pyomo.environ as pyo

    loss_name = str(loss).strip().lower()
    builders = {
        "mae": add_mae_loss,
        "mse": add_mse_qp_loss,
        "mse_qp": add_mse_qp_loss,
        "mse_socp": add_mse_socp_loss,
    }
    if loss_name not in builders:
        raise FormulationError(
            f"Unknown data loss '{loss}'. Supported: "
            f"{list(SUPPORTED_DATA_LOSSES)} (plus alias 'mse')."
        )
    result = builders[loss_name](model, predictions, targets, name=name)
    if add_objective:
        _claim_component_name(model, objective_name)
        setattr(
            model,
            objective_name,
            pyo.Objective(expr=float(weight) * result.expression, sense=pyo.minimize),
        )
    return result
