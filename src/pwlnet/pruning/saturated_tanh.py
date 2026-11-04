"""Bound-based saturated-tanh pruning for PWLNet (pure numpy).

A hidden ``tanh`` neuron whose entire scaled pre-activation interval
``[lb, ub]`` lies beyond ``+/- abs_threshold`` is effectively constant:
``tanh`` saturates to ``+/- 1`` and varies by at most
``tanh(ub) - tanh(lb)`` across the interval (a very small number once
``|pre-activation| >= 3``, since ``tanh(3) = 0.9951``). Such a neuron can
be replaced by the constant ``c_j = 0.5 * (tanh(lb_j) + tanh(ub_j))`` --
the interval midpoint, which minimises the worst-case post-activation
error -- and folded into the next layer's bias via
:func:`pwlnet.pruning.structure.remove_hidden_neurons`.

This is a TIGHT APPROXIMATION, not an exact transformation
(``exact=False``): tanh is monotone but not truly constant on the
interval, so a small residual remains.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

import numpy as np

from pwlnet.core.bounds import propagate_bounds
from pwlnet.core.network import NetworkDefinition
from pwlnet.pruning import structure
from pwlnet.pruning.base import PruneOutcome
from pwlnet.utils.exceptions import NetworkValidationError
from pwlnet.utils.logging import get_logger

__all__ = ["prune_saturated_tanh"]

_logger = get_logger(__name__)


def _sample_input_box(
    input_bounds, n_samples: int, seed: int
) -> np.ndarray:
    """Draw uniform samples (plus all box corners up to a cap) in the box.

    The box corners are included explicitly because the worst-case output
    deviation of a monotone network is frequently attained at a vertex of
    the input box; interior uniform samples fill in the rest.

    Parameters
    ----------
    input_bounds : list of (float, float)
        Per-input ``(lb, ub)`` bounds in ORIGINAL units.
    n_samples : int
        Number of interior uniform samples to draw.
    seed : int
        Seed for the RNG (reproducible sampling).

    Returns
    -------
    numpy.ndarray
        Sample matrix of shape ``(n_corners + n_samples, n_inputs)`` in
        ORIGINAL units.
    """
    lb = np.array([b[0] for b in input_bounds], dtype=float)
    ub = np.array([b[1] for b in input_bounds], dtype=float)
    n_in = lb.size

    rng = np.random.default_rng(seed)
    interior = lb + (ub - lb) * rng.random((n_samples, n_in))

    # Enumerate box corners only when cheap (<= 4096 corners); the corner
    # set often realises the monotone worst case.
    if n_in <= 12:
        n_corners = 1 << n_in
        bits = ((np.arange(n_corners)[:, None] >> np.arange(n_in)[None, :]) & 1).astype(bool)
        corners = np.where(bits, ub[None, :], lb[None, :])
        return np.vstack([corners, interior])
    return interior


def prune_saturated_tanh(
    network: NetworkDefinition,
    abs_threshold: float = 3.0,
    n_samples: int = 512,
    seed: int = 0,
) -> PruneOutcome:
    """Prune saturated ``tanh`` neurons by folding them to constants.

    For each HIDDEN ``tanh`` layer, interval bound propagation
    (:func:`pwlnet.core.bounds.propagate_bounds`) gives each neuron's
    scaled pre-activation interval ``[lb, ub]``. A neuron is SATURATED
    when that whole interval lies beyond ``+/- abs_threshold``:

    * ``lb >= abs_threshold`` (pinned near ``+1``), or
    * ``ub <= -abs_threshold`` (pinned near ``-1``).

    Each saturated neuron is replaced by the constant
    ``c_j = 0.5 * (tanh(lb_j) + tanh(ub_j))`` (the interval midpoint
    minimises the max post-activation error) and removed via
    :func:`pwlnet.pruning.structure.remove_hidden_neurons`, which folds
    ``c_j`` exactly into the next layer's bias.

    The transformation is an approximation, not exact
    (``exact=False``). See the Notes section for the error analysis.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated). Must have ``input_bounds`` set.
    abs_threshold : float, optional
        Saturation cutoff on the scaled pre-activation magnitude
        (default ``3.0``; ``tanh(3) = 0.9951``). Must be positive.
    n_samples : int, optional
        Number of uniform interior samples used to estimate the actual
        worst-case output change (box corners are added on top for
        ``n_inputs <= 12``). Default ``512``.
    seed : int, optional
        RNG seed for reproducible sampling. Default ``0``.

    Returns
    -------
    PruneOutcome
        Outcome with ``method="saturated_tanh"``, ``exact=False``, the
        removed neurons per layer, and ``max_abs_output_change`` set to
        the SAMPLED maximum ``|predict_before - predict_after|`` over all
        outputs in ORIGINAL units. ``notes`` records both the analytic
        per-neuron error bound (``sum_j e_j``) and the sampled maximum.

    Raises
    ------
    ValueError
        If ``abs_threshold`` is not positive.
    pwlnet.utils.exceptions.NetworkValidationError
        If the network has no ``input_bounds`` (raised by
        :func:`propagate_bounds`).

    Notes
    -----
    **Error bound.** Replacing a saturated neuron's post-activation
    ``tanh(z)`` (for ``z`` in ``[lb_j, ub_j]``) by the midpoint constant
    ``c_j`` incurs a worst-case post-activation error of

    .. math::

        e_j = \\tfrac12 (\\tanh(ub_j) - \\tanh(lb_j)) \\ge 0,

    attained at both endpoints (midpoint is the Chebyshev-optimal
    constant for a monotone function). Because ``tanh`` is monotone and
    these intervals are pinned in a saturated region, ``e_j`` is tiny:
    e.g. an interval entirely in ``[3, +inf)`` has ``e_j <= 1 -
    tanh(3) = 0.0049`` (and shrinks rapidly for larger ``lb``).

    These per-neuron errors propagate to the network output through the
    remaining (now exact) affine/activation layers. A simple, defensible
    *analytic* over-estimate of the output deviation is
    ``sum_j e_j`` scaled by the downstream weight path; rather than track
    a loose product of operator norms we report the conservative
    ``sum_j e_j`` (the total injected post-activation perturbation
    budget) in ``notes`` and use the **sampled** actual deviation --
    ``max |predict_before - predict_after|`` over ``n_samples`` uniform
    points plus box corners, in ORIGINAL output units -- as the headline
    ``max_abs_output_change``. The sampled value is what downstream code
    should trust; ``sum_j e_j`` is a quick upper-bound sanity check.
    """
    if not (abs_threshold > 0):
        raise ValueError(f"abs_threshold must be positive, got {abs_threshold}.")

    # propagate_bounds raises NetworkValidationError if input_bounds is None.
    if network.input_bounds is None:
        raise NetworkValidationError(
            "prune_saturated_tanh requires input_bounds to be set on the "
            "network (needed for bound propagation)."
        )

    bounds = propagate_bounds(network)
    n_layers = network.n_layers

    # Per-layer plan: layer_index -> (neuron_indices, constants, errors).
    removed_neurons: dict[int, list[int]] = {}
    constants_per_layer: dict[int, list[float]] = {}
    total_analytic_error = 0.0
    n_tanh_hidden_layers = 0

    for layer_index in range(n_layers - 1):  # hidden layers only
        layer = network.layers[layer_index]
        if layer.activation != "tanh":
            continue
        n_tanh_hidden_layers += 1

        pre_lb, pre_ub = bounds.pre_activation_bounds[layer_index]
        # Saturated: whole interval beyond +/- abs_threshold.
        sat_pos = pre_lb >= abs_threshold
        sat_neg = pre_ub <= -abs_threshold
        saturated = sat_pos | sat_neg
        sat_idx = np.nonzero(saturated)[0]
        if sat_idx.size == 0:
            continue

        lb_s = pre_lb[sat_idx]
        ub_s = pre_ub[sat_idx]
        consts = 0.5 * (np.tanh(lb_s) + np.tanh(ub_s))
        errs = 0.5 * (np.tanh(ub_s) - np.tanh(lb_s))  # >= 0

        removed_neurons[layer_index] = [int(j) for j in sat_idx]
        constants_per_layer[layer_index] = [float(c) for c in consts]
        total_analytic_error += float(np.sum(errs))

    # Apply removals layer by layer. Original neuron indices stay valid
    # because each layer's columns are addressed independently and the
    # input network is the reference for all indices.
    pruned = structure.copy_network(network)
    for layer_index, neuron_idx in removed_neurons.items():
        pruned = structure.remove_hidden_neurons(
            pruned,
            layer_index,
            neuron_idx,
            constant_outputs=constants_per_layer[layer_index],
        )

    n_removed = sum(len(v) for v in removed_neurons.values())

    # Sampled actual output deviation (ORIGINAL units).
    if n_removed > 0:
        X = _sample_input_box(network.input_bounds, n_samples, seed)
        y_before = network.predict(X)
        y_after = pruned.predict(X)
        sampled_max = float(np.max(np.abs(y_before - y_after)))
    else:
        sampled_max = 0.0

    notes = [
        f"abs_threshold={abs_threshold}",
        f"scanned {n_tanh_hidden_layers} hidden tanh layer(s); "
        f"removed {n_removed} saturated neuron(s)",
        "midpoint constant c_j = 0.5*(tanh(lb_j)+tanh(ub_j)); "
        "per-neuron worst-case post-activation error e_j = "
        "0.5*(tanh(ub_j)-tanh(lb_j))",
        f"analytic sum_j e_j (post-activation perturbation budget) = "
        f"{total_analytic_error:.6e}",
        f"sampled max |predict_before - predict_after| over "
        f"{n_samples} interior + corner points (original units) = "
        f"{sampled_max:.6e}",
    ]

    outcome = PruneOutcome(
        network=pruned,
        method="saturated_tanh",
        removed_neurons=removed_neurons,
        exact=False,
        max_abs_output_change=sampled_max,
        notes=notes,
    )
    _logger.debug(
        "saturated_tanh: removed %d neuron(s); sum_j e_j=%.3e, sampled max=%.3e",
        n_removed,
        total_analytic_error,
        sampled_max,
    )
    return outcome
