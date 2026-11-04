"""Trainable piecewise-linear tanh activation for Keras.

The PWL-tanh approximation that the MILP embedding builds
(:func:`pwlnet.formulations.pyomo_pwl.build_tanh_pwl_sos2_block`) is a
continuous piecewise-linear function, so it is a plain sum of ReLUs and
backpropagates like any other activation. Training *through* it removes
the train/embed mismatch instead of shrinking it: a network fitted with
exact ``tanh`` and then embedded as PWL is evaluated by the solver
through a function it was never fitted to, and that gap compounds over a
chained/receding horizon. A network fitted with this activation is
reproduced by its own embedding to float round-off.

Keras is imported lazily inside the factory, so importing this module
never requires TensorFlow/Keras.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pwlnet.formulations.pyomo_pwl.piecewise_data import build_tanh_pwl_data
from pwlnet.training.initialization import import_keras
from pwlnet.utils.typing import ArrayLike

__all__ = ["PWL_TANH", "make_pwl_tanh_activation"]

#: Canonical trainer-side name for the PWL-tanh activation. Layers
#: trained with it are EXPORTED as ``"tanh"`` (see the module docstring
#: of :mod:`pwlnet.training.keras_trainer`).
PWL_TANH = "pwl_tanh"


def make_pwl_tanh_activation(
    breakpoints: int | ArrayLike = 5,
) -> Callable[[Any], Any]:
    """Build a Keras activation that evaluates the PWL tanh exactly.

    The returned callable reproduces
    :meth:`pwlnet.formulations.pyomo_pwl.PiecewiseLinearData.evaluate`
    (with clipping) as a sum of ReLUs::

        f(z) = v[0] + s[0] * (zc - k[0]) + sum_i (s[i] - s[i-1]) * relu(zc - k[i])

    where ``k``/``v`` are the breakpoints and tanh values, ``s`` the
    segment slopes, and ``zc`` is ``z`` clipped to ``[k[0], k[-1]]``.
    The clip reproduces the flat extension the MILP block applies
    outside the tanh domain, and it is what makes the activation agree
    with the embedding for saturated neurons.

    Parameters
    ----------
    breakpoints : int or array_like, optional
        Breakpoint count (``3``, ``5``, or ``7`` select the standard
        SOSX-paper sets) or an explicit strictly increasing breakpoint
        vector spanning ``[-4, 4]``. Must be the SAME value later passed
        to the embedding, otherwise the mismatch this activation exists
        to remove comes straight back. Default 5.

    Returns
    -------
    callable
        A function ``act(z) -> tensor`` usable as the ``activation``
        argument of ``keras.layers.Dense``. It is differentiable
        (piecewise constant gradient) and backend-agnostic: it is built
        from ``keras.ops`` only.

    Raises
    ------
    ValueError
        If ``breakpoints`` is not a valid tanh breakpoint set.
    pwlnet.utils.exceptions.TrainingError
        If Keras cannot be imported.

    Examples
    --------
    >>> act = make_pwl_tanh_activation(5)             # doctest: +SKIP
    >>> keras.layers.Dense(16, activation=act)        # doctest: +SKIP
    """
    keras = import_keras()
    ops = keras.ops

    pwl = build_tanh_pwl_data(breakpoints)
    knots = np.asarray(pwl.breakpoints, dtype=float)
    values = np.asarray(pwl.values, dtype=float)
    slopes = np.asarray(pwl.slopes, dtype=float)

    lo, hi = float(knots[0]), float(knots[-1])
    base_value, base_slope = float(values[0]), float(slopes[0])
    # One ReLU per interior knot, weighted by the slope CHANGE there.
    interior = [
        (float(knot), float(jump))
        for knot, jump in zip(knots[1:-1], np.diff(slopes))
    ]

    def pwl_tanh(z: Any) -> Any:
        zc = ops.clip(z, lo, hi)
        out = base_value + base_slope * (zc - lo)
        for knot, jump in interior:
            out = out + jump * ops.relu(zc - knot)
        return out

    pwl_tanh.__name__ = f"{PWL_TANH}_{pwl.n_segments}_pieces"
    return pwl_tanh
