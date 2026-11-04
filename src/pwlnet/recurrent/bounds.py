"""Recurrent interval-bound propagation across an unrolled horizon.

The core :func:`pwlnet.core.bounds.propagate_bounds` bounds a *single*
feed-forward network. A ReLU RNN unrolled over a horizon is a chain of
identical cells whose hidden box at step ``t`` depends on the box at
step ``t - 1``, so a single static box reused at every step would be
either loose or -- if the hidden state can grow -- unsound.

This module threads the hidden box step by step:

* start from a box on the initial hidden state ``h_0`` (latent units)
  and a per-step box on the exogenous input ``u`` (original units);
* at each step build the cell's step network specialised to the current
  hidden box (:meth:`RecurrentCell.step_network`) and push the
  ``[u, h_t]`` box through it with the existing interval arithmetic; the
  last layer's post-activation interval is the next hidden box
  ``h_{t+1}``;
* propagate the readout interval to bound the observable outputs.

Because every step reuses the audited interval machinery (including the
exogenous scaler), the per-step hidden boxes are sound: the true
reachable set at each step is a subset of the propagated box. They are
fed back into the per-step ``relu_bigm`` big-M constants in
:mod:`pwlnet.recurrent.unroll`, which is what keeps the unrolled MILP
tight enough to solve.

Note: for a hidden recurrence with spectral radius ``> 1`` the boxes can
grow geometrically with the horizon; they remain valid (and the MILP
correct) but the big-M constants -- hence the LP relaxation -- loosen.
Keep the horizon and hidden width modest, and prune dead units first.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pwlnet.core.bounds import propagate_bounds
from pwlnet.utils.exceptions import NetworkValidationError
from pwlnet.utils.typing import BoundsList

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from pwlnet.recurrent.cell import RecurrentCell

__all__ = ["RecurrentBounds", "propagate_recurrent_bounds"]


@dataclass
class RecurrentBounds:
    """Per-step bounds produced by :func:`propagate_recurrent_bounds`.

    Attributes
    ----------
    horizon : int
        Number of cell applications the bounds cover.
    hidden_bounds : list of (numpy.ndarray, numpy.ndarray)
        Length ``horizon + 1``. Entry ``t`` is the ``(lb, ub)`` box on the
        hidden state at node ``t`` in LATENT units (``hidden_bounds[0]`` is
        the seed box on ``h_0``).
    output_bounds : list of (numpy.ndarray, numpy.ndarray)
        Length ``horizon``. Entry ``t`` is the ``(lb, ub)`` box on the
        observable output ``y_t = readout(h_{t+1})`` in ORIGINAL units.
    exog_bounds : list of (float, float)
        The exogenous box used at every step (original units).
    """

    horizon: int
    hidden_bounds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    output_bounds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    exog_bounds: BoundsList = field(default_factory=list)

    def hidden_box(self, t: int) -> BoundsList:
        """Hidden box at node ``t`` as a list of ``(lb, ub)`` tuples."""
        lo, hi = self.hidden_bounds[t]
        return [(float(l), float(u)) for l, u in zip(lo, hi)]


def _linear_interval(
    lb: np.ndarray, ub: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact interval image of ``x @ W + b`` for ``x`` in ``[lb, ub]``.

    Same positive/negative weight split used by
    :func:`pwlnet.core.bounds.propagate_bounds`; duplicated here (a few
    lines) to bound the affine readout without importing a private
    helper.
    """
    w_pos = np.maximum(weights, 0.0)
    w_neg = np.minimum(weights, 0.0)
    out_lb = lb @ w_pos + ub @ w_neg + biases
    out_ub = ub @ w_pos + lb @ w_neg + biases
    return out_lb, out_ub


def propagate_recurrent_bounds(
    cell: "RecurrentCell",
    horizon: int,
    *,
    hidden_init_bounds: BoundsList | None = None,
    exog_bounds: BoundsList | None = None,
) -> RecurrentBounds:
    """Propagate hidden/output interval bounds over an unrolled horizon.

    Parameters
    ----------
    cell : RecurrentCell
        The trained cell to propagate through.
    horizon : int
        Number of cell applications (``>= 1``).
    hidden_init_bounds : list of (float, float) or None
        Seed box for ``h_0`` (latent units). Defaults to
        ``cell.hidden_init_bounds``.
    exog_bounds : list of (float, float) or None
        Box for the exogenous input at every step (original units).
        Defaults to ``cell.exog_bounds``.

    Returns
    -------
    RecurrentBounds
        Per-step hidden boxes (length ``horizon + 1``) and output boxes
        (length ``horizon``).

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If ``horizon < 1`` or the required bound boxes are unavailable.
    """
    H = int(horizon)
    if H < 1:
        raise NetworkValidationError(f"horizon must be >= 1, got {horizon}.")

    h_box = hidden_init_bounds if hidden_init_bounds is not None else cell.hidden_init_bounds
    if h_box is None:
        raise NetworkValidationError(
            "No hidden_init_bounds available to seed propagation (pass "
            "hidden_init_bounds or set it on the cell)."
        )
    u_box = exog_bounds if exog_bounds is not None else cell.exog_bounds
    if u_box is None:
        raise NetworkValidationError(
            "No exog_bounds available (pass exog_bounds or set it on the cell)."
        )
    if len(h_box) != cell.n_hidden:
        raise NetworkValidationError(
            f"hidden_init_bounds has {len(h_box)} entries but n_hidden="
            f"{cell.n_hidden}."
        )
    if len(u_box) != cell.n_exog:
        raise NetworkValidationError(
            f"exog_bounds has {len(u_box)} entries but n_exog={cell.n_exog}."
        )

    h_lo = np.array([lb for lb, _ in h_box], dtype=float)
    h_hi = np.array([ub for _, ub in h_box], dtype=float)

    result = RecurrentBounds(
        horizon=H,
        hidden_bounds=[(h_lo.copy(), h_hi.copy())],
        exog_bounds=[(float(lb), float(ub)) for lb, ub in u_box],
    )

    for _t in range(H):
        # Specialise the step network to the current hidden box (the exog
        # box is constant and stored on the cell). propagate_bounds then
        # applies the exog scaler and the relu interval map for us.
        current_h_box = [(float(lo), float(hi)) for lo, hi in zip(h_lo, h_hi)]
        step_net = cell.step_network(current_h_box)
        if exog_bounds is not None:
            # Honour a caller-supplied exog box that differs from the cell's.
            step_net.input_bounds = list(u_box) + current_h_box  # type: ignore[assignment]
            step_net.validate()
        step_bounds = propagate_bounds(step_net)
        # The single relu layer's post-activation interval IS h_{t+1}
        # (the step network has no output scaler).
        h_lo, h_hi = step_bounds.post_activation_bounds[-1]
        result.hidden_bounds.append((h_lo.copy(), h_hi.copy()))

        if cell.readout is None:
            result.output_bounds.append((h_lo.copy(), h_hi.copy()))
        else:
            y_lo, y_hi = _linear_interval(
                h_lo, h_hi, cell.readout.weights, cell.readout.biases
            )
            result.output_bounds.append((y_lo, y_hi))

    return result
