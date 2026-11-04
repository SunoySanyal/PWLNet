"""Method-name dispatcher for PWLNet pruning (pure numpy at import time).

:func:`prune_network` maps a string ``method`` to the corresponding
pruning entry point under :mod:`pwlnet.pruning` and forwards keyword
arguments to it. Methods that require Keras/TensorFlow (currently only
``"obd"``) import those libraries lazily inside their own modules, so
importing this module never pulls in Keras.

Supported method names (aliases share an implementation):

================  ===========================================================
Method name       Function
================  ===========================================================
``dead_relu``     :func:`pwlnet.pruning.dead_relu.prune_dead_relus`
``always_active`` :func:`pwlnet.pruning.dead_relu.simplify_always_active_relus`
``always_active_relu``  (alias of ``always_active``)
``relus``         :func:`pwlnet.pruning.dead_relu.prune_relus`
``relu``          (alias of ``relus``)
``saturated_tanh``  :func:`pwlnet.pruning.saturated_tanh.prune_saturated_tanh`
``weight_magnitude``  :func:`pwlnet.pruning.magnitude.prune_weights_magnitude`
``magnitude``     (alias of ``weight_magnitude``)
``neuron``        :func:`pwlnet.pruning.neuron_pruning.prune_neurons`
``neuron_magnitude``  (alias of ``neuron``)
``group_lasso``   :func:`pwlnet.pruning.group_lasso.prune_by_group_norm`
``obd``           :func:`pwlnet.pruning.obd.optimal_brain_damage`
================  ===========================================================

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports
at module scope.
"""

from __future__ import annotations

from typing import Any, Callable

from pwlnet.core.network import NetworkDefinition
from pwlnet.pruning.base import PruneOutcome
from pwlnet.pruning.dead_relu import (
    prune_dead_relus,
    prune_relus,
    simplify_always_active_relus,
)
from pwlnet.pruning.group_lasso import prune_by_group_norm
from pwlnet.pruning.magnitude import prune_weights_magnitude
from pwlnet.pruning.neuron_pruning import prune_neurons
from pwlnet.pruning.saturated_tanh import prune_saturated_tanh

__all__ = ["prune_network", "PRUNE_METHODS"]


def _optimal_brain_damage(network: NetworkDefinition, **kwargs: Any) -> PruneOutcome:
    """Lazily import and call OBD (keeps Keras off the import path)."""
    from pwlnet.pruning.obd import optimal_brain_damage

    return optimal_brain_damage(network, **kwargs)


#: Mapping of public method name -> pruning callable. Aliases point at the
#: same function. The OBD entry is wrapped so that ``import
#: pwlnet.pruning.prune`` does not import Keras/TensorFlow.
PRUNE_METHODS: dict[str, Callable[..., PruneOutcome]] = {
    "dead_relu": prune_dead_relus,
    "always_active": simplify_always_active_relus,
    "always_active_relu": simplify_always_active_relus,
    "relus": prune_relus,
    "relu": prune_relus,
    "saturated_tanh": prune_saturated_tanh,
    "weight_magnitude": prune_weights_magnitude,
    "magnitude": prune_weights_magnitude,
    "neuron": prune_neurons,
    "neuron_magnitude": prune_neurons,
    "group_lasso": prune_by_group_norm,
    "obd": _optimal_brain_damage,
}


def prune_network(
    network: NetworkDefinition, method: str = "dead_relu", **kwargs: Any
) -> PruneOutcome:
    """Dispatch to a named pruning method.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated; the method returns a new network).
    method : str, optional
        Name of the pruning method (see :data:`PRUNE_METHODS` for the
        valid names and aliases). Default ``"dead_relu"``.
    **kwargs
        Extra keyword arguments forwarded verbatim to the selected
        pruning function (e.g. ``fraction=`` for magnitude/neuron pruning,
        ``threshold=`` for group lasso, ``X=``/``y=`` for OBD).

    Returns
    -------
    PruneOutcome
        The outcome returned by the selected method.

    Raises
    ------
    ValueError
        If ``method`` is not a recognised name.
    """
    key = str(method).strip().lower()
    func = PRUNE_METHODS.get(key)
    if func is None:
        valid = ", ".join(sorted(PRUNE_METHODS))
        raise ValueError(
            f"Unknown pruning method {method!r}. Valid methods: {valid}."
        )
    return func(network, **kwargs)
