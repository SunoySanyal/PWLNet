"""Finite weight/bias bounds for SOSX-MAE training relaxations.

The SOSX training formulation makes the trainable parameters ``W1``,
``b1`` and ``b2`` Pyomo variables. Leaving them unbounded produces weak
or numerically unstable LP relaxations (the architecture's Stage 2
warning), so every training model needs finite box bounds. This module
defines the :class:`WeightBounds` container and helpers that derive
bounds wide enough to contain a warm-start network's parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pwlnet.core.network import NetworkDefinition
from pwlnet.utils.exceptions import FormulationError

__all__ = ["WeightBounds", "bounds_covering_network", "resolve_weight_bounds"]

#: Default symmetric box half-width used when no tighter information is given.
DEFAULT_BOUND = 4.0


@dataclass(frozen=True)
class WeightBounds:
    """Symmetric box bounds for the trainable parameters of a training model.

    Each parameter ``p`` is constrained to ``[-bound, +bound]``.

    Attributes
    ----------
    weight : float
        Half-width for the input-layer weights ``W1``.
    bias : float
        Half-width for the hidden biases ``b1``.
    output_bias : float
        Half-width for the output biases ``b2``.
    """

    weight: float = DEFAULT_BOUND
    bias: float = DEFAULT_BOUND
    output_bias: float = DEFAULT_BOUND

    def __post_init__(self) -> None:
        for name in ("weight", "bias", "output_bias"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise FormulationError(
                    f"WeightBounds.{name} must be a finite positive number, "
                    f"got {getattr(self, name)!r}."
                )
            object.__setattr__(self, name, value)

    def contains_network(self, network: NetworkDefinition) -> bool:
        """Whether a one-hidden-layer network's parameters fit inside the box."""
        if network.n_layers != 2:
            raise FormulationError(
                "WeightBounds.contains_network expects a one-hidden-layer network."
            )
        w1 = np.abs(network.layers[0].weights)
        b1 = np.abs(network.layers[0].biases)
        b2 = np.abs(network.layers[1].biases)
        return bool(
            (w1.size == 0 or w1.max() <= self.weight)
            and (b1.size == 0 or b1.max() <= self.bias)
            and (b2.size == 0 or b2.max() <= self.output_bias)
        )


def bounds_covering_network(
    network: NetworkDefinition,
    *,
    margin: float = 1.5,
    floor: float = DEFAULT_BOUND,
) -> WeightBounds:
    """Derive bounds wide enough to contain ``network``'s ``W1``, ``b1``, ``b2``.

    Each half-width is ``max(floor, margin * max(|parameter|))`` so a
    warm-start network sits strictly inside the box (giving the LP some
    room to move) while never dropping below ``floor``.

    Parameters
    ----------
    network : NetworkDefinition
        A one-hidden-layer ``tanh`` -> ``linear`` network.
    margin : float, optional
        Multiplier applied to the observed maximum magnitude (default 1.5).
    floor : float, optional
        Minimum half-width for every parameter group (default
        :data:`DEFAULT_BOUND`).

    Returns
    -------
    WeightBounds
        Box bounds that contain the network's parameters.
    """
    if network.n_layers != 2:
        raise FormulationError(
            "bounds_covering_network expects a one-hidden-layer network "
            f"(got {network.n_layers} layers)."
        )
    if not np.isfinite(margin) or margin <= 0.0:
        raise FormulationError(f"margin must be a finite positive number, got {margin!r}.")
    if not np.isfinite(floor) or floor <= 0.0:
        raise FormulationError(f"floor must be a finite positive number, got {floor!r}.")

    def _half_width(values: np.ndarray) -> float:
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        return max(float(floor), float(margin) * peak)

    return WeightBounds(
        weight=_half_width(network.layers[0].weights),
        bias=_half_width(network.layers[0].biases),
        output_bias=_half_width(network.layers[1].biases),
    )


def resolve_weight_bounds(
    bounds: WeightBounds | None,
    *,
    weight_bound: float | None = None,
    bias_bound: float | None = None,
    output_bias_bound: float | None = None,
    cover: NetworkDefinition | None = None,
) -> WeightBounds:
    """Resolve the effective :class:`WeightBounds` from the trainer's inputs.

    Precedence, highest first:

    1. An explicit ``bounds`` object.
    2. Per-group scalar overrides (``weight_bound`` / ``bias_bound`` /
       ``output_bias_bound``); any group left ``None`` falls back to the
       cover-derived value, then to :data:`DEFAULT_BOUND`.
    3. Bounds derived from ``cover`` (a warm-start network), when given.
    4. :data:`DEFAULT_BOUND` for every group.
    """
    if bounds is not None:
        if cover is not None and not bounds.contains_network(cover):
            raise FormulationError(
                "The supplied WeightBounds do not contain the warm-start "
                "network's parameters; widen the bounds or drop the warm start."
            )
        return bounds

    base = (
        bounds_covering_network(cover)
        if cover is not None
        else WeightBounds()
    )
    return WeightBounds(
        weight=base.weight if weight_bound is None else float(weight_bound),
        bias=base.bias if bias_bound is None else float(bias_bound),
        output_bias=base.output_bias
        if output_bias_bound is None
        else float(output_bias_bound),
    )
