"""Conservation / balance physics residuals (affine).

Mass-balance, sum-to-one closure, and general linear conservation laws,
all expressed as :class:`pwlnet.physics.algebraic.AffineResidual`
instances so they reuse the same NumPy / Keras / Pyomo machinery.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pwlnet.physics.algebraic import AffineResidual

__all__ = [
    "sum_to_one_residual",
    "mass_balance_residual",
    "conservation_residual",
]


def sum_to_one_residual(
    n_outputs: int,
    output_indices: Sequence[int] | None = None,
    *,
    total: float = 1.0,
    name: str = "sum_to_one",
) -> AffineResidual:
    """Closure residual ``sum(y[idx]) - total``.

    The canonical mass-fraction constraint: the selected output
    components must sum to ``total`` (default 1). Equality residual.

    Parameters
    ----------
    n_outputs : int
        Surrogate output width.
    output_indices : sequence of int or None, optional
        Components included in the sum. ``None`` uses all outputs.
    total : float, optional
        Target sum. Default ``1.0``.
    name : str, optional
    """
    idx = range(n_outputs) if output_indices is None else output_indices
    w_out = np.zeros((1, n_outputs))
    for j in idx:
        w_out[0, int(j)] = 1.0
    return AffineResidual(None, w_out, -float(total), kind="equality", name=name)


def mass_balance_residual(
    *,
    n_inputs: int,
    n_outputs: int,
    inflow_inputs: dict[int, float] | None = None,
    outflow_outputs: dict[int, float] | None = None,
    const: float = 0.0,
    name: str = "mass_balance",
) -> AffineResidual:
    """Linear mass balance ``sum(a_i x_i) - sum(c_j y_j) + const = 0``.

    Models conditions such as ``Fin - Fout - y_product = 0`` from the
    architecture document, with stoichiometric/flow coefficients given
    per index. Equality residual.

    Parameters
    ----------
    n_inputs, n_outputs : int
        Surrogate widths.
    inflow_inputs : dict of int to float or None
        Input-side coefficients ``a_i`` (e.g. feed flows). Sign as you
        want them to appear in the balance.
    outflow_outputs : dict of int to float or None
        Output-side coefficients ``c_j`` (subtracted: they enter the
        residual with a negative sign).
    const : float, optional
        Constant term.
    name : str, optional
    """
    w_in = None
    if inflow_inputs:
        w_in = np.zeros((1, n_inputs))
        for i, a in inflow_inputs.items():
            w_in[0, int(i)] = float(a)
    w_out = None
    if outflow_outputs:
        w_out = np.zeros((1, n_outputs))
        for j, c in outflow_outputs.items():
            w_out[0, int(j)] = -float(c)
    if w_in is None and w_out is None:
        raise ValueError("mass_balance_residual needs inflow_inputs or outflow_outputs.")
    return AffineResidual(w_in, w_out, float(const), kind="equality", name=name)


def conservation_residual(
    w_in: np.ndarray | None,
    w_out: np.ndarray | None,
    b: np.ndarray | float = 0.0,
    *,
    name: str = "conservation",
) -> AffineResidual:
    """General multi-row linear conservation law ``x W_in^T + y W_out^T + b = 0``.

    A thin alias over :class:`AffineResidual` for several simultaneous
    balance rows (equality).
    """
    return AffineResidual(w_in, w_out, b, kind="equality", name=name)
