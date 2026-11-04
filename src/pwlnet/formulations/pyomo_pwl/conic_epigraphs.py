"""Conic epigraph helpers for quadratic objectives.

A quadratic objective ``min sum(e_i^2)`` can be rewritten as the
epigraph problem ``min t  s.t.  sum(e_i^2) <= t``. The constraint is a
convex (conic-representable) quadratic constraint, so a MILP whose only
nonlinearity is the objective becomes an MISOCP that conic solvers such
as MOSEK handle natively. This is the ``mse_socp`` path from the
architecture document's formulation matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pwlnet.utils.exceptions import FormulationError

__all__ = ["add_quadratic_epigraph"]


def add_quadratic_epigraph(
    model: Any,
    expressions: Sequence[Any],
    *,
    name: str = "quadratic_epigraph",
) -> Any:
    """Add ``t >= sum(e^2)`` to a model and return the epigraph variable.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel or Block
        Model that receives a new block named ``name``.
    expressions : sequence
        Pyomo expressions (or variables) whose squared sum is bounded by
        the epigraph variable.
    name : str, optional
        Component name for the added block.

    Returns
    -------
    pyomo.environ.Var
        The scalar epigraph variable ``t`` (nonnegative).

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If ``expressions`` is empty or the component name is taken.
    """
    import pyomo.environ as pyo

    expressions = list(expressions)
    if not expressions:
        raise FormulationError(
            "add_quadratic_epigraph needs at least one expression."
        )
    if model.component(name) is not None:
        raise FormulationError(
            f"Model already has a component named '{name}'; choose another name."
        )

    block = pyo.Block()
    setattr(model, name, block)
    block.t = pyo.Var(domain=pyo.NonNegativeReals)
    block.cone = pyo.Constraint(
        expr=sum(expr * expr for expr in expressions) <= block.t
    )
    return block.t
