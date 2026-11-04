"""Pruning and optimization-aware compression for PWLNet.

This package exposes the full Phase-3 pruning API:

* :class:`~pwlnet.pruning.base.PruneOutcome` -- the shared result contract
  returned by every pruning method.
* Structural primitives (pure numpy): :func:`copy_network`,
  :func:`remove_hidden_neurons`, :func:`set_layer_activation`,
  :func:`fuse_linear_layers`, :func:`zero_small_weights`.
* Method entry points: bound-based ReLU pruning
  (:func:`prune_dead_relus`, :func:`simplify_always_active_relus`,
  :func:`prune_relus`), saturated-tanh folding
  (:func:`prune_saturated_tanh`), weight-magnitude pruning
  (:func:`prune_weights_magnitude`), importance-based neuron pruning
  (:func:`prune_neurons`, :func:`neuron_importance`), group-lasso
  (:func:`neuron_group_norms`, :func:`make_group_lasso_regularizer`,
  :func:`prune_by_group_norm`), Optimal Brain Damage
  (:func:`obd_saliencies`, :func:`optimal_brain_damage`), and Keras
  fine-tuning (:func:`fine_tune`).
* The :func:`prune_network` method-name dispatcher.
* Reporting / comparison: :class:`PruneReport`, :func:`build_prune_report`,
  :class:`NetworkComparison`, :func:`compare_networks`,
  :func:`count_nonzero_parameters`.

Importing this package never imports Keras or TensorFlow: the three
methods that need them (group lasso's regularizer, OBD, fine-tuning) do
the ``import keras`` lazily inside their functions, so ``import
pwlnet.pruning`` works with or without a deep-learning backend installed.
"""

from pwlnet.pruning.base import PruneOutcome
from pwlnet.pruning.compare import (
    NetworkComparison,
    compare_networks,
    count_nonzero_parameters,
    formulation_size,
)
from pwlnet.pruning.dead_relu import (
    prune_dead_relus,
    prune_relus,
    simplify_always_active_relus,
)
from pwlnet.pruning.fine_tune import fine_tune
from pwlnet.pruning.group_lasso import (
    group_lasso_penalty,
    make_group_lasso_regularizer,
    neuron_group_norms,
    prune_by_group_norm,
)
from pwlnet.pruning.magnitude import neuron_constant_value, prune_weights_magnitude
from pwlnet.pruning.neuron_pruning import neuron_importance, prune_neurons
from pwlnet.pruning.obd import obd_saliencies, optimal_brain_damage
from pwlnet.pruning.prune import PRUNE_METHODS, prune_network
from pwlnet.pruning.reports import PruneReport, build_prune_report
from pwlnet.pruning.saturated_tanh import prune_saturated_tanh
from pwlnet.pruning.structure import (
    copy_network,
    fuse_linear_layers,
    remove_hidden_neurons,
    set_layer_activation,
    zero_small_weights,
)

__all__ = [
    # contract
    "PruneOutcome",
    # structural primitives
    "copy_network",
    "remove_hidden_neurons",
    "set_layer_activation",
    "fuse_linear_layers",
    "zero_small_weights",
    # ReLU bound-based pruning
    "prune_dead_relus",
    "simplify_always_active_relus",
    "prune_relus",
    # tanh
    "prune_saturated_tanh",
    # magnitude
    "prune_weights_magnitude",
    "neuron_constant_value",
    # neuron importance
    "prune_neurons",
    "neuron_importance",
    # group lasso
    "neuron_group_norms",
    "group_lasso_penalty",
    "make_group_lasso_regularizer",
    "prune_by_group_norm",
    # optimal brain damage
    "obd_saliencies",
    "optimal_brain_damage",
    # fine-tuning
    "fine_tune",
    # dispatcher
    "prune_network",
    "PRUNE_METHODS",
    # reports / comparison
    "PruneReport",
    "build_prune_report",
    "NetworkComparison",
    "compare_networks",
    "count_nonzero_parameters",
    "formulation_size",
]
