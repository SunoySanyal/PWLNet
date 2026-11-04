"""One-step ReLU RNN cell, expressed as a feed-forward PWL network.

A *vanilla* (Elman) recurrent cell with a ``relu`` activation is itself
piecewise linear, so it needs no new MILP machinery: the hidden update

.. math::

    h_{t+1} = \\mathrm{relu}\\big(u_t W_{xh} + h_t W_{hh} + b_h\\big)

is exactly a single :class:`pwlnet.core.layers.DenseLayer` with a
``relu`` activation acting on the *concatenated* input ``[u_t, h_t]`` and
weight matrix ``vstack([W_xh, W_hh])``. The linear readout

.. math::

    y_t = h_t W_{hy} + b_y

is affine, so it never needs a big-M encoding and is kept out of the
embedded block (it becomes plain linear constraints when the cell is
unrolled, see :mod:`pwlnet.recurrent.unroll`).

This module therefore reuses the existing core stack unchanged:

* the step map is a :class:`pwlnet.core.network.NetworkDefinition`
  (so :func:`pwlnet.core.bounds.propagate_bounds`, ``relu_bigm`` OMLT
  embedding, and the ReLU pruners all apply to it verbatim);
* the only genuinely recurrent additions live elsewhere -- bound
  propagation *across* the horizon (:mod:`pwlnet.recurrent.bounds`) and
  the hidden-state carry in the unrolled MILP
  (:mod:`pwlnet.recurrent.unroll`).

Index/space conventions (used everywhere in this subpackage)
------------------------------------------------------------
* The step network's input vector is ordered ``[u (n_exog), h
  (n_hidden)]``; its output vector is ``h_next (n_hidden)``.
* Exogenous inputs ``u`` are in ORIGINAL units; the step network's
  ``input_scaler`` scales only the ``u`` block (identity on ``h``) so the
  affine part reproduces the Keras ``SimpleRNN`` scaled-space update.
* The hidden state ``h`` is a LATENT quantity with no physical units; it
  is never scaled (identity offset/factor).
* The readout maps ``h`` to ``y`` in ORIGINAL units -- the Keras output
  scaler is folded into ``readout`` at construction time, so
  :meth:`RecurrentCell.readout_eval` and the unrolled MILP both produce
  outputs directly in original units.

This module must stay free of keras/tensorflow/torch/pyomo/omlt imports
(those are pulled in lazily by the trainer / unroll modules).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from pwlnet.core.layers import DenseLayer
from pwlnet.core.network import NetworkDefinition
from pwlnet.core.scaling import OffsetScaler
from pwlnet.utils.exceptions import NetworkValidationError
from pwlnet.utils.typing import ArrayLike, BoundsList, PathLike
from pwlnet.utils.validation import as_2d_array

__all__ = ["RecurrentCell"]

_FORMAT_VERSION = 1


@dataclass
class RecurrentCell:
    """A trained one-step ReLU RNN cell with its readout and bound boxes.

    Parameters
    ----------
    step : NetworkDefinition
        The hidden-state update ``[u, h] -> h_next``. Must be a ReLU
        network (every hidden/output activation piecewise linear) whose
        ``n_inputs == n_exog + n_hidden`` and ``n_outputs == n_hidden``.
        Carries an ``input_scaler`` that scales the exogenous block only
        and no output scaler (the hidden state is latent).
    n_exog : int
        Number of exogenous (control / disturbance) inputs ``u``.
    n_hidden : int
        Width of the recurrent hidden state ``h``.
    readout : DenseLayer or None
        Linear map ``h -> y`` in ORIGINAL output units (the Keras output
        scaler is already folded in). ``None`` means the observable
        output *is* the hidden state (``y == h``).
    exog_bounds : list of (float, float)
        Per-input ``(lb, ub)`` box for ``u`` in ORIGINAL units; used as
        the exogenous box for bound propagation and as variable bounds in
        the unrolled MILP.
    hidden_init_bounds : list of (float, float)
        Reachable ``(lb, ub)`` box for the hidden state, in latent units.
        Used to seed recurrent bound propagation and as a fallback box
        for the hidden variables.

    Raises
    ------
    pwlnet.utils.exceptions.NetworkValidationError
        If the shapes of ``step``, ``readout`` and the bound boxes are
        mutually inconsistent.
    """

    step: NetworkDefinition
    n_exog: int
    n_hidden: int
    readout: DenseLayer | None = None
    exog_bounds: BoundsList | None = None
    hidden_init_bounds: BoundsList | None = None

    def __post_init__(self) -> None:
        self.n_exog = int(self.n_exog)
        self.n_hidden = int(self.n_hidden)
        # The step map must consume [u, h] and produce h_next.
        if self.step.n_inputs != self.n_exog + self.n_hidden:
            raise NetworkValidationError(
                f"step.n_inputs={self.step.n_inputs} but n_exog+n_hidden="
                f"{self.n_exog + self.n_hidden}."
            )
        if self.step.n_outputs != self.n_hidden:
            raise NetworkValidationError(
                f"step.n_outputs={self.step.n_outputs} must equal n_hidden="
                f"{self.n_hidden} (the cell is a square hidden recurrence)."
            )
        if any(not info_is_pwl(act) for act in self.step.activations):
            raise NetworkValidationError(
                "RecurrentCell.step must be piecewise linear (relu/linear "
                f"only); got activations {self.step.activations}. tanh/sigmoid "
                "cells are not MILP-exact."
            )
        if self.readout is not None:
            if self.readout.activation != "linear":
                raise NetworkValidationError(
                    "readout must be a linear DenseLayer; got activation "
                    f"'{self.readout.activation}'."
                )
            if self.readout.n_inputs != self.n_hidden:
                raise NetworkValidationError(
                    f"readout.n_inputs={self.readout.n_inputs} must equal "
                    f"n_hidden={self.n_hidden}."
                )
        if self.exog_bounds is not None and len(self.exog_bounds) != self.n_exog:
            raise NetworkValidationError(
                f"exog_bounds has {len(self.exog_bounds)} entries but n_exog="
                f"{self.n_exog}."
            )
        if (
            self.hidden_init_bounds is not None
            and len(self.hidden_init_bounds) != self.n_hidden
        ):
            raise NetworkValidationError(
                f"hidden_init_bounds has {len(self.hidden_init_bounds)} entries "
                f"but n_hidden={self.n_hidden}."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_outputs(self) -> int:
        """Width of the observable output ``y`` (``n_hidden`` if no readout)."""
        return self.n_hidden if self.readout is None else self.readout.n_outputs

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_elman(
        cls,
        W_xh: ArrayLike,
        W_hh: ArrayLike,
        b_h: ArrayLike,
        *,
        readout_weights: ArrayLike | None = None,
        readout_biases: ArrayLike | None = None,
        exog_scaler: OffsetScaler | None = None,
        output_scaler: OffsetScaler | None = None,
        exog_bounds: BoundsList,
        hidden_init_bounds: BoundsList,
    ) -> "RecurrentCell":
        """Assemble a cell from raw Elman-RNN weight matrices.

        This mirrors the parametrisation of a Keras ``SimpleRNN`` layer
        (``activation="relu"``) followed by a linear ``Dense`` readout,
        and folds the input/output scalers into the cell so that the
        embedded step map is exact (see the module docstring for the
        scaled-space algebra).

        Parameters
        ----------
        W_xh : array_like
            Input-to-hidden weights (Keras ``kernel``), shape
            ``(n_exog, n_hidden)`` -- applied to the SCALED exogenous
            input.
        W_hh : array_like
            Hidden-to-hidden weights (Keras ``recurrent_kernel``), shape
            ``(n_hidden, n_hidden)``.
        b_h : array_like
            Hidden bias (Keras ``bias``), shape ``(n_hidden,)``.
        readout_weights, readout_biases : array_like or None
            Linear readout in SCALED output space (Keras ``Dense``
            kernel/bias), shapes ``(n_hidden, n_y)`` and ``(n_y,)``.
            ``None`` means no readout (``y == h``).
        exog_scaler : OffsetScaler or None
            Scaler fitted on the exogenous inputs ``u`` (original ->
            scaled). ``None`` means the inputs are already scaled.
        output_scaler : OffsetScaler or None
            Scaler fitted on the targets ``y``; folded into the readout
            so the stored readout maps ``h`` to ORIGINAL units.
        exog_bounds : list of (float, float)
            Box for ``u`` in original units.
        hidden_init_bounds : list of (float, float)
            Reachable box for the hidden state in latent units.

        Returns
        -------
        RecurrentCell
            The assembled cell.
        """
        W_xh = np.asarray(W_xh, dtype=float)
        W_hh = np.asarray(W_hh, dtype=float)
        b_h = np.asarray(b_h, dtype=float).ravel()
        n_exog, n_hidden = W_xh.shape
        if W_hh.shape != (n_hidden, n_hidden):
            raise NetworkValidationError(
                f"W_hh must be ({n_hidden}, {n_hidden}) to match W_xh's "
                f"n_hidden, got {W_hh.shape}."
            )

        # Stack the two weight blocks so a single dense layer over the
        # concatenated input [u_scaled, h] reproduces
        #   u_scaled @ W_xh + h @ W_hh + b_h.
        stacked = np.vstack([W_xh, W_hh])  # (n_exog + n_hidden, n_hidden)
        step_layer = DenseLayer(weights=stacked, biases=b_h, activation="relu")

        # Build a combined input scaler that scales ONLY the exogenous
        # block and leaves the hidden block untouched (identity).
        input_scaler: OffsetScaler | None = None
        if exog_scaler is not None:
            if exog_scaler.n_features != n_exog:
                raise NetworkValidationError(
                    f"exog_scaler has {exog_scaler.n_features} features but "
                    f"n_exog={n_exog}."
                )
            offset = np.concatenate([exog_scaler.offset, np.zeros(n_hidden)])
            factor = np.concatenate([exog_scaler.factor, np.ones(n_hidden)])
            input_scaler = OffsetScaler(offset=offset, factor=factor, method="manual")

        step = NetworkDefinition(
            layers=[step_layer],
            input_scaler=input_scaler,
            output_scaler=None,  # h_next is latent: no output scaling.
            input_bounds=list(exog_bounds) + list(hidden_init_bounds),
        )

        readout = _fold_readout(readout_weights, readout_biases, output_scaler, n_hidden)

        return cls(
            step=step,
            n_exog=n_exog,
            n_hidden=n_hidden,
            readout=readout,
            exog_bounds=[(float(lb), float(ub)) for lb, ub in exog_bounds],
            hidden_init_bounds=[(float(lb), float(ub)) for lb, ub in hidden_init_bounds],
        )

    # ------------------------------------------------------------------
    # Numpy evaluation (ground truth for the MILP / validation)
    # ------------------------------------------------------------------
    def step_forward(self, u: ArrayLike, h: ArrayLike) -> np.ndarray:
        """One hidden-state update ``h_next = step([u, h])`` (numpy).

        Parameters
        ----------
        u : array_like
            Exogenous input, shape ``(n_exog,)`` (original units).
        h : array_like
            Current hidden state, shape ``(n_hidden,)`` (latent).

        Returns
        -------
        numpy.ndarray
            Next hidden state, shape ``(n_hidden,)``.
        """
        u_arr = np.asarray(u, dtype=float).ravel()
        h_arr = np.asarray(h, dtype=float).ravel()
        if u_arr.shape != (self.n_exog,):
            raise NetworkValidationError(
                f"u must have {self.n_exog} entries, got {u_arr.shape}."
            )
        if h_arr.shape != (self.n_hidden,):
            raise NetworkValidationError(
                f"h must have {self.n_hidden} entries, got {h_arr.shape}."
            )
        x = np.concatenate([u_arr, h_arr])
        # step has an input scaler (scales u, identity on h) and no output
        # scaler, so predict() returns the latent h_next directly.
        return np.asarray(self.step.predict(x), dtype=float)

    def readout_eval(self, h: ArrayLike) -> np.ndarray:
        """Map hidden state(s) to observable output(s) in ORIGINAL units.

        Parameters
        ----------
        h : array_like
            Hidden state, shape ``(n_hidden,)`` or ``(n, n_hidden)``.

        Returns
        -------
        numpy.ndarray
            Output ``y``, shape ``(n_outputs,)`` or ``(n, n_outputs)``.
            With no readout this is ``h`` unchanged.
        """
        arr = np.asarray(h, dtype=float)
        if self.readout is None:
            return arr
        squeeze = arr.ndim == 1
        mat = as_2d_array(arr, name="h")
        y = mat @ self.readout.weights + self.readout.biases
        return y[0] if squeeze else y

    def encode(self, u_history: ArrayLike, h_init: ArrayLike | None = None) -> np.ndarray:
        """Warm-start the hidden state by replaying a control history.

        The latent hidden state is not measurable, so at the start of a
        receding-horizon solve it is reconstructed by running the cell
        forward (in numpy, no optimisation) over the recent exogenous
        history. This is the recurrent counterpart of "measure the
        initial state" in an observable-state NMPC.

        Parameters
        ----------
        u_history : array_like
            Past exogenous inputs, shape ``(window, n_exog)`` in
            chronological order (oldest first), original units.
        h_init : array_like or None
            Hidden state before the window (default zeros, matching the
            Keras default initial state).

        Returns
        -------
        numpy.ndarray
            Hidden state after the last input in ``u_history``, shape
            ``(n_hidden,)``.
        """
        hist = as_2d_array(u_history, name="u_history")
        if hist.shape[1] != self.n_exog:
            raise NetworkValidationError(
                f"u_history has {hist.shape[1]} features, expected n_exog="
                f"{self.n_exog}."
            )
        h = (
            np.zeros(self.n_hidden)
            if h_init is None
            else np.asarray(h_init, dtype=float).ravel()
        )
        for u in hist:
            h = self.step_forward(u, h)
        return h

    def rollout(
        self, h0: ArrayLike, u_seq: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll the cell forward over a control sequence (numpy).

        Convention: ``y[t] = readout(h[t+1])`` -- the observable at step
        ``t`` is read from the hidden state *after* applying ``u[t]``.

        Parameters
        ----------
        h0 : array_like
            Initial hidden state, shape ``(n_hidden,)``.
        u_seq : array_like
            Exogenous sequence, shape ``(horizon, n_exog)``.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            ``hidden`` of shape ``(horizon + 1, n_hidden)`` (``hidden[0]``
            is ``h0``) and ``outputs`` of shape ``(horizon, n_outputs)``.
        """
        useq = as_2d_array(u_seq, name="u_seq")
        if useq.shape[1] != self.n_exog:
            raise NetworkValidationError(
                f"u_seq has {useq.shape[1]} features, expected n_exog="
                f"{self.n_exog}."
            )
        horizon = useq.shape[0]
        hidden = np.empty((horizon + 1, self.n_hidden), dtype=float)
        hidden[0] = np.asarray(h0, dtype=float).ravel()
        outputs = np.empty((horizon, self.n_outputs), dtype=float)
        for t in range(horizon):
            hidden[t + 1] = self.step_forward(useq[t], hidden[t])
            outputs[t] = self.readout_eval(hidden[t + 1])
        return hidden, outputs

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------
    def step_network(self, hidden_bounds: BoundsList | None = None) -> NetworkDefinition:
        """Return the step map specialised to a hidden-state box.

        Each replicated block in the unrolled MILP gets its OWN tight
        hidden box (from :mod:`pwlnet.recurrent.bounds`), which directly
        shrinks the ``relu_bigm`` big-M constants for that step. This
        returns a copy of :attr:`step` with input bounds
        ``exog_bounds + hidden_bounds`` so it can be handed straight to
        the OMLT exporter.

        Parameters
        ----------
        hidden_bounds : list of (float, float) or None
            Hidden-state box (latent units) for this step. Defaults to
            :attr:`hidden_init_bounds`.

        Returns
        -------
        NetworkDefinition
            The step network with input bounds set for this step.
        """
        if self.exog_bounds is None:
            raise NetworkValidationError(
                "Cannot build a step network without exog_bounds."
            )
        h_box = hidden_bounds if hidden_bounds is not None else self.hidden_init_bounds
        if h_box is None:
            raise NetworkValidationError(
                "No hidden bounds available (pass hidden_bounds or set "
                "hidden_init_bounds)."
            )
        return NetworkDefinition(
            layers=self.step.layers,
            input_scaler=self.step.input_scaler,
            output_scaler=self.step.output_scaler,
            input_bounds=list(self.exog_bounds) + list(h_box),
        )

    # ------------------------------------------------------------------
    # Recurrent orchestration (delegated to sibling modules; lazy import
    # to keep this module pyomo/omlt-free at import time).
    # ------------------------------------------------------------------
    def recurrent_bounds(
        self,
        horizon: int,
        *,
        hidden_init_bounds: BoundsList | None = None,
        exog_bounds: BoundsList | None = None,
    ) -> Any:
        """Propagate interval bounds across ``horizon`` steps.

        Thin wrapper around
        :func:`pwlnet.recurrent.bounds.propagate_recurrent_bounds`.
        """
        from pwlnet.recurrent.bounds import propagate_recurrent_bounds

        return propagate_recurrent_bounds(
            self,
            horizon,
            hidden_init_bounds=hidden_init_bounds,
            exog_bounds=exog_bounds,
        )

    def prune_dead_units(
        self,
        *,
        reachable_hidden_bounds: BoundsList | None = None,
        recurrent_bounds: Any | None = None,
    ) -> tuple["RecurrentCell", list[int]]:
        """Exactly remove hidden units that are dead over the operating box.

        A hidden unit whose pre-activation upper bound is ``<= 0`` over the
        reachable hidden/exog region is pinned to ``relu(z) = 0`` and so
        contributes nothing to the recurrence *or* the readout. Such a
        unit is removed from the hidden dimension entirely -- its output
        column, its recurrent input row, its bias entry, its readout row,
        and its init-bound entry are all dropped. The removal is exact
        (function preserving over that region).

        By default the region is the cell's :attr:`hidden_init_bounds` --
        the box of hidden states actually visited on the training data,
        i.e. where the controller will operate -- combined with the
        exogenous box, evaluated through one cell application. This mirrors
        the feed-forward dead-ReLU pruner, which works over the input box.
        Pass ``recurrent_bounds`` instead to prune over the (looser, but
        horizon-valid) union of the propagated per-step boxes; note that a
        recurrence with spectral radius ``> 1`` makes that union grow with
        the horizon, so far fewer units qualify as dead.

        Always-active units (``lb >= 0``) are NOT linearised here: the
        single recurrent layer carries one activation for all units, so a
        per-unit linearisation cannot be expressed. They are left as-is.

        Parameters
        ----------
        reachable_hidden_bounds : list of (float, float) or None
            Explicit operating box for the hidden state (latent units).
            Defaults to :attr:`hidden_init_bounds`.
        recurrent_bounds : RecurrentBounds or None
            If given, the probe box is the union of its per-step hidden
            boxes (overrides ``reachable_hidden_bounds``).

        Returns
        -------
        (RecurrentCell, list of int)
            The pruned cell and the ORIGINAL indices of the removed units.
        """
        from pwlnet.core.bounds import propagate_bounds

        if recurrent_bounds is not None:
            # Horizon-valid (looser) union of every step's box.
            h_lo = np.min([lo for lo, _ in recurrent_bounds.hidden_bounds], axis=0)
            h_hi = np.max([hi for _, hi in recurrent_bounds.hidden_bounds], axis=0)
            h_box = [(float(lo), float(hi)) for lo, hi in zip(h_lo, h_hi)]
        else:
            h_box = reachable_hidden_bounds or self.hidden_init_bounds
            if h_box is None:
                raise NetworkValidationError(
                    "No operating box available: pass reachable_hidden_bounds "
                    "or set hidden_init_bounds on the cell."
                )

        # Pre-activation interval of the (single) recurrent layer over the
        # [u, h] operating box. step_network embeds the exog box + this
        # hidden box, and propagate_bounds applies the exog scaler for us.
        probe = self.step_network(list(h_box))
        pre_ub = propagate_bounds(probe).pre_activation_bounds[0][1]
        dead = sorted(int(j) for j in np.flatnonzero(pre_ub <= 0.0))
        if not dead:
            return self, []
        if len(dead) == self.n_hidden:
            # Removing every unit would leave a 0-width state; refuse and
            # report (correctness over completeness).
            return self, []

        keep = [j for j in range(self.n_hidden) if j not in set(dead)]
        pruned = self._drop_hidden_units(keep)
        return pruned, dead

    def unit_importance(self, U_sequences: ArrayLike) -> np.ndarray:
        """Score each hidden unit by its influence on the outputs.

        The score is ``mean_t |h_j(t)|`` over rollouts of ``U_sequences``
        (how active the unit is in the operating region) times the L1 norm
        of its outgoing readout row (how strongly it drives the outputs).
        A near-zero score means the unit barely moves or barely feeds the
        readout, so dropping it changes the prediction little.

        Parameters
        ----------
        U_sequences : array_like
            Exogenous sequences, shape ``(n_seq, T, n_exog)`` (original
            units), e.g. the training controls.

        Returns
        -------
        numpy.ndarray
            Per-unit importance, shape ``(n_hidden,)``.
        """
        seqs = np.asarray(U_sequences, dtype=float)
        if seqs.ndim == 2:
            seqs = seqs[None, :, :]
        acts = []
        for u in seqs:
            hidden, _ = self.rollout(np.zeros(self.n_hidden), u)
            acts.append(np.abs(hidden[1:]))  # skip the (zero) seed state
        mean_abs = np.concatenate(acts, axis=0).mean(axis=0)  # (n_hidden,)
        outgoing = (
            np.abs(self.readout.weights).sum(axis=1)
            if self.readout is not None
            else np.ones(self.n_hidden)
        )
        return mean_abs * outgoing

    def prune_least_important(
        self,
        U_sequences: ArrayLike,
        *,
        n_keep: int | None = None,
        fraction: float | None = None,
    ) -> tuple["RecurrentCell", list[int]]:
        """Drop the least influential hidden units (lossy, importance-based).

        Unlike :meth:`prune_dead_units` (exact, removes only provably-dead
        units), this trims the smallest-:meth:`unit_importance` units to
        reach a target width. It is approximate -- the prediction changes
        slightly -- but lets you shrink a deliberately over-sized network
        down to a compact cell, which directly cuts the number of big-M
        binaries when the cell is unrolled. Re-check accuracy afterwards
        (and optionally retrain) to confirm the trade-off is acceptable.

        Parameters
        ----------
        U_sequences : array_like
            Sequences used to score importance (see :meth:`unit_importance`).
        n_keep : int or None
            Number of units to retain (the most important). Mutually
            exclusive with ``fraction``.
        fraction : float or None
            Fraction of units to REMOVE, in ``(0, 1)``. Mutually exclusive
            with ``n_keep``.

        Returns
        -------
        (RecurrentCell, list of int)
            The smaller cell and the ORIGINAL indices of the removed units.
        """
        if (n_keep is None) == (fraction is None):
            raise NetworkValidationError(
                "Pass exactly one of n_keep or fraction."
            )
        if n_keep is None:
            if not 0.0 < float(fraction) < 1.0:
                raise NetworkValidationError("fraction must be in (0, 1).")
            n_keep = max(1, int(round((1.0 - float(fraction)) * self.n_hidden)))
        n_keep = int(n_keep)
        if not 1 <= n_keep <= self.n_hidden:
            raise NetworkValidationError(
                f"n_keep must be in [1, {self.n_hidden}], got {n_keep}."
            )
        if n_keep == self.n_hidden:
            return self, []
        importance = self.unit_importance(U_sequences)
        # Keep the n_keep most important units (stable order for the kept set).
        keep = sorted(int(j) for j in np.argsort(importance)[::-1][:n_keep])
        removed = [j for j in range(self.n_hidden) if j not in set(keep)]
        return self._drop_hidden_units(keep), removed

    def _drop_hidden_units(self, keep: list[int]) -> "RecurrentCell":
        """Build a new cell retaining only the hidden units in ``keep``.

        Drops, for each removed unit, its hidden-output column, its
        recurrent-input row, bias entry, readout row, and init-bound
        entry. Exog inputs/scaling are untouched.
        """
        keep_arr = np.asarray(keep, dtype=int)
        layer = self.step.layers[0]
        # Rows: [exog rows (0..n_exog-1)] kept verbatim, then the recurrent
        # rows (n_exog + keep) for the surviving hidden inputs.
        exog_rows = np.arange(self.n_exog)
        recur_rows = self.n_exog + keep_arr
        rows = np.concatenate([exog_rows, recur_rows])
        new_weights = layer.weights[np.ix_(rows, keep_arr)]
        new_biases = layer.biases[keep_arr]
        new_step_layer = DenseLayer(
            weights=new_weights, biases=new_biases, activation="relu"
        )

        # Rebuild the combined input scaler over [u, surviving h].
        input_scaler = None
        if self.step.input_scaler is not None:
            offset = self.step.input_scaler.offset[rows]
            factor = self.step.input_scaler.factor[rows]
            input_scaler = OffsetScaler(offset=offset, factor=factor, method="manual")

        new_hidden_init = (
            [self.hidden_init_bounds[j] for j in keep]
            if self.hidden_init_bounds is not None
            else None
        )
        step = NetworkDefinition(
            layers=[new_step_layer],
            input_scaler=input_scaler,
            output_scaler=None,
            input_bounds=(list(self.exog_bounds or []) + list(new_hidden_init or [])),
        )

        readout = None
        if self.readout is not None:
            readout = DenseLayer(
                weights=self.readout.weights[keep_arr, :],
                biases=self.readout.biases,
                activation="linear",
            )

        return RecurrentCell(
            step=step,
            n_exog=self.n_exog,
            n_hidden=len(keep),
            readout=readout,
            exog_bounds=self.exog_bounds,
            hidden_init_bounds=new_hidden_init,
        )

    def build_mpc(self, h0: ArrayLike, horizon: int, **kwargs: Any) -> Any:
        """Build the unrolled MPC model. See :func:`..unroll.build_unrolled_mpc_model`."""
        from pwlnet.recurrent.unroll import build_unrolled_mpc_model

        return build_unrolled_mpc_model(self, h0, horizon, **kwargs)

    def solve_mpc(self, h0: ArrayLike, horizon: int, **kwargs: Any) -> dict[str, Any]:
        """Build and solve one MPC move. See :func:`..unroll.solve_unrolled_mpc`."""
        from pwlnet.recurrent.unroll import solve_unrolled_mpc

        return solve_unrolled_mpc(self, h0, horizon, **kwargs)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize the cell to a JSON-compatible dictionary."""
        return {
            "format_version": _FORMAT_VERSION,
            "step": self.step.to_dict(),
            "n_exog": self.n_exog,
            "n_hidden": self.n_hidden,
            "readout": (
                None
                if self.readout is None
                else {
                    "weights": self.readout.weights.tolist(),
                    "biases": self.readout.biases.tolist(),
                }
            ),
            "exog_bounds": (
                [[lb, ub] for lb, ub in self.exog_bounds]
                if self.exog_bounds is not None
                else None
            ),
            "hidden_init_bounds": (
                [[lb, ub] for lb, ub in self.hidden_init_bounds]
                if self.hidden_init_bounds is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecurrentCell":
        """Reconstruct a cell from :meth:`to_dict` output."""
        readout = None
        if d.get("readout") is not None:
            readout = DenseLayer(
                weights=np.asarray(d["readout"]["weights"], dtype=float),
                biases=np.asarray(d["readout"]["biases"], dtype=float),
                activation="linear",
            )
        return cls(
            step=NetworkDefinition.from_dict(d["step"]),
            n_exog=d["n_exog"],
            n_hidden=d["n_hidden"],
            readout=readout,
            exog_bounds=(
                [(float(lb), float(ub)) for lb, ub in d["exog_bounds"]]
                if d.get("exog_bounds") is not None
                else None
            ),
            hidden_init_bounds=(
                [(float(lb), float(ub)) for lb, ub in d["hidden_init_bounds"]]
                if d.get("hidden_init_bounds") is not None
                else None
            ),
        )

    def save(self, path: PathLike) -> None:
        """Save the cell to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: PathLike) -> "RecurrentCell":
        """Load a cell previously written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __repr__(self) -> str:
        return (
            f"RecurrentCell(n_exog={self.n_exog}, n_hidden={self.n_hidden}, "
            f"n_outputs={self.n_outputs}, "
            f"readout={'yes' if self.readout is not None else 'no'})"
        )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def info_is_pwl(activation: str) -> bool:
    """True if ``activation`` is piecewise linear (relu / linear)."""
    from pwlnet.core.activations import get_activation

    return get_activation(activation).is_piecewise_linear


def _fold_readout(
    readout_weights: ArrayLike | None,
    readout_biases: ArrayLike | None,
    output_scaler: OffsetScaler | None,
    n_hidden: int,
) -> DenseLayer | None:
    """Fold the output scaler into the readout, returning an original-units map.

    The Keras readout produces ``y_scaled = h @ Wr + br``; the output
    scaler maps that to original units via
    ``y = y_scaled * factor + offset``. Composing,

        y = h @ (Wr * factor) + (br * factor + offset),

    so the folded readout maps the hidden state directly to original
    output units and never needs the scaler again.
    """
    if readout_weights is None:
        return None
    Wr = np.asarray(readout_weights, dtype=float)
    br = (
        np.zeros(Wr.shape[1])
        if readout_biases is None
        else np.asarray(readout_biases, dtype=float).ravel()
    )
    if Wr.shape[0] != n_hidden:
        raise NetworkValidationError(
            f"readout_weights must have {n_hidden} rows (n_hidden), got "
            f"{Wr.shape}."
        )
    if output_scaler is not None:
        if output_scaler.n_features != Wr.shape[1]:
            raise NetworkValidationError(
                f"output_scaler has {output_scaler.n_features} features but the "
                f"readout has {Wr.shape[1]} outputs."
            )
        factor = output_scaler.factor
        offset = output_scaler.offset
        Wr = Wr * factor[None, :]
        br = br * factor + offset
    return DenseLayer(weights=Wr, biases=br, activation="linear")
