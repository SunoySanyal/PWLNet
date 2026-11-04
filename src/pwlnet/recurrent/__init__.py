"""Recurrent (ReLU RNN) surrogate engine for PWLNet.

A vanilla Elman cell with a ``relu`` activation is piecewise linear, so
the whole train -> prune -> bound-tighten -> unroll-in-MILP pipeline
reuses the existing core/embedding/pruning stack. This subpackage adds
only the genuinely recurrent pieces:

* :class:`RecurrentCell` -- the one-step cell expressed as a feed-forward
  PWL :class:`~pwlnet.core.network.NetworkDefinition` plus a linear
  readout, with hidden-state carry, warm-start encoding, exact dead-unit
  pruning, and serialization;
* :func:`propagate_recurrent_bounds` / :class:`RecurrentBounds` --
  interval-bound propagation threaded across the unrolled horizon;
* :func:`build_unrolled_mpc_model` / :func:`solve_unrolled_mpc` -- the
  receding-horizon MILP that chains one ``relu_bigm`` block per step.

Training lives in :class:`pwlnet.training.rnn_trainer.RNNTrainer`.
"""

from pwlnet.recurrent.bounds import RecurrentBounds, propagate_recurrent_bounds
from pwlnet.recurrent.cell import RecurrentCell
from pwlnet.recurrent.unroll import build_unrolled_mpc_model, solve_unrolled_mpc

__all__ = [
    "RecurrentCell",
    "RecurrentBounds",
    "propagate_recurrent_bounds",
    "build_unrolled_mpc_model",
    "solve_unrolled_mpc",
]
