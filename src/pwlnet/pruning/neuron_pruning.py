"""Importance-based neuron pruning for PWLNet (pure numpy).

This module ranks the neurons of each HIDDEN layer by an importance
score and removes the least important ones, folding each removed neuron
into the next layer as if its post-activation were the constant ``0``.

That zero-substitution is an APPROXIMATION: a removed neuron rarely
outputs exactly ``0`` over the whole input box, so the pruned network is
*not* function-preserving and every :class:`PruneOutcome` returned here
has ``exact=False``. The magnitude of the resulting error is estimated by
sampling the input box (when ``input_bounds`` are available) and reported
in :attr:`PruneOutcome.max_abs_output_change` in ORIGINAL output units.

The default importance score is ``"outgoing_l2"``: the L2 norm of a
neuron's outgoing weight vector (its row in the next layer's weight
matrix). Optionally an activity weighting is applied -- the mean absolute
SCALED post-activation of the neuron over a supplied dataset -- so that
neurons that fire rarely or weakly are deemed less important.

This module is pure numpy and must stay free of
keras/tensorflow/torch/pyomo/omlt imports.
"""

from __future__ import annotations

import numpy as np

from pwlnet.core.network import NetworkDefinition
from pwlnet.pruning.base import PruneOutcome
from pwlnet.pruning.structure import remove_hidden_neurons
from pwlnet.utils.validation import as_2d_array

__all__ = ["neuron_importance", "prune_neurons"]

_SUPPORTED_METHODS = ("outgoing_l2",)
_N_SAMPLE_DEFAULT = 512


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _hidden_indices(network: NetworkDefinition) -> list[int]:
    """Return the indices of the hidden layers (all but the output layer).

    Parameters
    ----------
    network : NetworkDefinition
        Source network.

    Returns
    -------
    list of int
        ``[0, 1, ..., n_layers - 2]``.
    """
    return list(range(network.n_layers - 1))


def _hidden_post_activations(
    network: NetworkDefinition, X
) -> dict[int, np.ndarray]:
    """Post-activation outputs of each hidden layer over ``X``.

    The input ``X`` is given in ORIGINAL units; it is pushed through the
    input scaler (if any) and then the layers, capturing the SCALED
    post-activation of every hidden layer.

    Parameters
    ----------
    network : NetworkDefinition
        Source network.
    X : array_like
        Inputs in ORIGINAL units, shape ``(n_samples, n_inputs)`` or
        ``(n_inputs,)``.

    Returns
    -------
    dict[int, numpy.ndarray]
        ``layer_index -> array (n_samples, n_neurons)`` of scaled
        post-activation values, for each hidden layer.
    """
    arr = as_2d_array(np.asarray(X, dtype=float), name="X")
    if arr.shape[1] != network.n_inputs:
        raise ValueError(
            f"X has {arr.shape[1]} features, expected {network.n_inputs}."
        )
    if network.input_scaler is not None:
        arr = network.input_scaler.transform(arr)
    acts: dict[int, np.ndarray] = {}
    h = arr
    for k, layer in enumerate(network.layers):
        h = layer.forward(h)
        if k < network.n_layers - 1:  # hidden layer
            acts[k] = h
    return acts


def neuron_importance(
    network: NetworkDefinition, X=None, method: str = "outgoing_l2"
) -> dict[int, np.ndarray]:
    """Per-hidden-layer importance score for every neuron.

    Higher scores mean more important neurons (less safe to remove).

    For ``method="outgoing_l2"`` the base score of neuron ``j`` in hidden
    layer ``k`` is the L2 norm of its outgoing weight vector,
    ``||layers[k + 1].weights[j, :]||_2`` (the neuron's row in the next
    layer's weight matrix -- the weights through which its activation
    influences the downstream computation).

    If ``X`` is provided (in ORIGINAL units), the base score is
    multiplied by the neuron's *activity*: the mean over ``X`` of the
    absolute SCALED post-activation ``a_j(x)``. A neuron with large
    outgoing weights that almost never fires therefore receives a small
    importance, and vice versa.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated).
    X : array_like or None, optional
        Inputs in ORIGINAL units used to weight the score by activity.
        Shape ``(n_samples, n_inputs)`` or ``(n_inputs,)``. If ``None``,
        only the outgoing-weight norm is used (a purely static score).
    method : str, optional
        Importance method. Only ``"outgoing_l2"`` is currently
        supported.

    Returns
    -------
    dict[int, numpy.ndarray]
        ``layer_index -> array (n_neurons,)`` of non-negative importance
        scores, one entry per hidden layer.

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``X`` has the wrong feature count.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown importance method {method!r}; supported: "
            f"{list(_SUPPORTED_METHODS)}."
        )

    activity: dict[int, np.ndarray] | None = None
    if X is not None:
        acts = _hidden_post_activations(network, X)
        activity = {
            k: np.mean(np.abs(a), axis=0) for k, a in acts.items()
        }

    scores: dict[int, np.ndarray] = {}
    for k in _hidden_indices(network):
        # Outgoing weights of layer-k neurons = rows of layers[k+1].weights,
        # shape (n_neurons, n_out_next). L2 norm along axis=1 -> per neuron.
        next_w = network.layers[k + 1].weights
        base = np.linalg.norm(next_w, axis=1)
        if activity is not None:
            base = base * activity[k]
        scores[k] = np.asarray(base, dtype=float)
    return scores


def _resolve_drop_counts(
    network: NetworkDefinition,
    scores: dict[int, np.ndarray],
    drop,
    fraction,
    keep,
) -> dict[int, int]:
    """Translate ``drop``/``fraction``/``keep`` into a per-layer drop count.

    Exactly one of the three selectors must be non-``None``. The returned
    counts are *requested* removals before the "keep >= 1 neuron per
    layer" clamp is applied by the caller.

    Parameters
    ----------
    network : NetworkDefinition
        Source network (for layer widths).
    scores : dict[int, numpy.ndarray]
        Importance scores per hidden layer (used for sizes and, for a
        global ``drop``/``fraction``/``keep``, the global ranking).
    drop : int or dict[int, int] or None
        Number of neurons to remove. An int is a global total spread
        across the least-important neurons network-wide; a dict gives an
        explicit per-layer count.
    fraction : float or None
        Fraction (in ``[0, 1]``) of all hidden neurons to remove,
        selected globally by least importance.
    keep : int or None
        Target number of hidden neurons to RETAIN network-wide; the rest
        (least important) are removed.

    Returns
    -------
    dict[int, int]
        ``layer_index -> requested drop count`` for every hidden layer.

    Raises
    ------
    ValueError
        If not exactly one selector is given, or a selector is malformed
        / out of range.
    """
    n_selectors = sum(s is not None for s in (drop, fraction, keep))
    if n_selectors != 1:
        raise ValueError(
            "Provide exactly one of 'drop', 'fraction' or 'keep' "
            f"(got drop={drop!r}, fraction={fraction!r}, keep={keep!r})."
        )

    hidden = _hidden_indices(network)
    sizes = {k: int(scores[k].size) for k in hidden}
    total_hidden = sum(sizes.values())

    # Per-layer explicit dict.
    if isinstance(drop, dict):
        counts = {k: 0 for k in hidden}
        for k, c in drop.items():
            if k not in sizes:
                raise ValueError(
                    f"drop dict references layer {k}, which is not a hidden "
                    f"layer ({sorted(sizes)})."
                )
            c = int(c)
            if c < 0:
                raise ValueError(f"drop[{k}]={c} must be >= 0.")
            counts[k] = c
        return counts

    # Global integer/fraction/keep -> a single global target removal count,
    # then distribute it onto the globally least-important neurons.
    if drop is not None:
        n_remove = int(drop)
        if n_remove < 0:
            raise ValueError(f"drop={drop} must be >= 0.")
    elif fraction is not None:
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"fraction must be in [0, 1], got {fraction}.")
        n_remove = int(round(fraction * total_hidden))
    else:  # keep is not None
        keep = int(keep)
        if keep < 0:
            raise ValueError(f"keep={keep} must be >= 0.")
        n_remove = max(0, total_hidden - keep)

    n_remove = min(n_remove, total_hidden)

    # Build a global ranking of (score, layer, neuron) ascending by score
    # and select the n_remove least-important entries, counting per layer.
    flat: list[tuple[float, int, int]] = []
    for k in hidden:
        for j in range(sizes[k]):
            flat.append((float(scores[k][j]), k, j))
    # Stable ascending sort by score; ties broken by (layer, neuron) for
    # determinism.
    flat.sort(key=lambda t: (t[0], t[1], t[2]))
    counts = {k: 0 for k in hidden}
    for _, k, _j in flat[:n_remove]:
        counts[k] += 1
    return counts


def _sample_input_box(
    network: NetworkDefinition, n_samples: int, seed: int = 0
) -> np.ndarray | None:
    """Uniformly sample the original-unit input box.

    Parameters
    ----------
    network : NetworkDefinition
        Source network providing ``input_bounds`` in ORIGINAL units.
    n_samples : int
        Number of points to draw.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    numpy.ndarray or None
        Sample of shape ``(n_samples, n_inputs)`` in original units, or
        ``None`` if the network has no ``input_bounds``.
    """
    if network.input_bounds is None:
        return None
    lb = np.array([b[0] for b in network.input_bounds], dtype=float)
    ub = np.array([b[1] for b in network.input_bounds], dtype=float)
    rng = np.random.default_rng(seed)
    u = rng.random((n_samples, network.n_inputs))
    return lb + u * (ub - lb)


def prune_neurons(
    network: NetworkDefinition,
    drop=None,
    fraction=None,
    keep=None,
    X=None,
    method: str = "outgoing_l2",
    per_layer: bool = True,
) -> PruneOutcome:
    """Remove the least-important hidden neurons (approximate pruning).

    Neurons are ranked with :func:`neuron_importance` and the least
    important are removed via
    :func:`pwlnet.pruning.structure.remove_hidden_neurons` with
    ``constant_outputs=0``. Treating a removed neuron's post-activation as
    the constant ``0`` is an APPROXIMATION -- the function is generally
    NOT preserved -- so the returned outcome always has ``exact=False``.

    Specify how many neurons to remove with exactly one of:

    * ``drop`` -- an int (global total, spread over the least-important
      neurons network-wide) or a ``dict[int, int]`` of explicit per-layer
      counts;
    * ``fraction`` -- a float in ``[0, 1]`` giving the fraction of all
      hidden neurons to remove (globally least important);
    * ``keep`` -- the target number of hidden neurons to RETAIN
      network-wide (the rest are removed).

    A hidden layer must always keep at least one neuron, so requested
    counts are clamped to ``n_neurons - 1`` per layer (a note records any
    clamping).

    Parameters
    ----------
    network : NetworkDefinition
        Source network (never mutated).
    drop : int or dict[int, int] or None, optional
        Number of neurons to remove (see above).
    fraction : float or None, optional
        Fraction of hidden neurons to remove (see above).
    keep : int or None, optional
        Target number of hidden neurons to keep (see above).
    X : array_like or None, optional
        Inputs in ORIGINAL units used to activity-weight the importance
        score (passed straight to :func:`neuron_importance`).
    method : str, optional
        Importance method (only ``"outgoing_l2"`` supported).
    per_layer : bool, optional
        Currently only documents intent for the ranking granularity. The
        global selectors (int ``drop``/``fraction``/``keep``) always rank
        neurons globally; an explicit ``drop`` dict is inherently
        per-layer. Retained for API symmetry; has no extra effect.

    Returns
    -------
    PruneOutcome
        Outcome whose ``.network`` is the new pruned network. ``exact``
        is always ``False``; ``max_abs_output_change`` is the sampled
        maximum absolute output change over the input box (~512 points)
        in ORIGINAL output units, or ``None`` if the network has no
        ``input_bounds``.

    Raises
    ------
    ValueError
        If not exactly one of ``drop``/``fraction``/``keep`` is given, a
        selector is malformed, or ``method`` is unknown.
    """
    scores = neuron_importance(network, X=X, method=method)
    requested = _resolve_drop_counts(network, scores, drop, fraction, keep)

    notes: list[str] = []
    removed_neurons: dict[int, list[int]] = {}
    pruned = network
    n_clamped_total = 0

    for k in _hidden_indices(network):
        n_neurons = int(scores[k].size)
        want = int(requested.get(k, 0))
        # Never remove the last neuron of a hidden layer.
        max_removable = max(0, n_neurons - 1)
        take = min(want, max_removable)
        if take < want:
            n_clamped_total += want - take
            notes.append(
                f"layer {k}: requested {want} removals but clamped to "
                f"{take} (must keep >= 1 neuron)."
            )
        if take <= 0:
            continue
        # Remove the `take` least-important neurons of this layer.
        order = np.argsort(scores[k], kind="stable")  # ascending score
        to_remove = sorted(int(j) for j in order[:take])
        # remove_hidden_neurons folds constant 0 (default) into next layer.
        pruned = remove_hidden_neurons(pruned, k, to_remove, constant_outputs=0.0)
        removed_neurons[k] = to_remove

    # Estimate the approximation error by sampling the input box.
    max_abs_output_change: float | None = None
    sample = _sample_input_box(network, _N_SAMPLE_DEFAULT, seed=0)
    if sample is not None:
        y_orig = np.asarray(network.predict(sample), dtype=float)
        y_pruned = np.asarray(pruned.predict(sample), dtype=float)
        max_abs_output_change = float(np.max(np.abs(y_orig - y_pruned)))

    if not removed_neurons:
        notes.append("no neurons removed.")

    notes.append(
        "approximate: removed neurons folded as constant 0 "
        "(function NOT preserved; exact=False)."
    )
    if n_clamped_total:
        notes.append(f"total clamped (un-removed) requests: {n_clamped_total}.")

    return PruneOutcome(
        network=pruned,
        method=f"neuron_importance:{method}",
        removed_neurons=removed_neurons,
        activation_changes={},
        n_zeroed_weights=0,
        exact=False,
        max_abs_output_change=max_abs_output_change,
        notes=notes,
    )
