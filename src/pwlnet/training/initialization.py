"""Seeded weight-initializer helpers for Keras training.

Keras is imported lazily inside functions so that importing
:mod:`pwlnet.training` never requires TensorFlow/Keras at import time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.utils.exceptions import TrainingError

__all__ = ["import_keras", "seed_everything", "layer_seeds", "get_initializer"]

#: Seeds handed to Keras initializers are kept inside the int32 range.
_MAX_SEED = 2**31 - 1


def import_keras() -> Any:
    """Import and return the ``keras`` module (Keras 3, standalone import).

    Returns
    -------
    module
        The imported ``keras`` module.

    Raises
    ------
    pwlnet.utils.exceptions.TrainingError
        If Keras cannot be imported. The message explains how to install
        it; the original ``ImportError`` is chained.
    """
    try:
        import keras  # noqa: PLC0415 -- deliberate lazy import
    except ImportError as exc:
        raise TrainingError(
            "Keras is required by pwlnet.training but could not be imported. "
            "Install the training extras (e.g. `pip install pwlnet[training]` "
            "or `pip install tensorflow`) and retry. "
            f"Original import error: {exc}"
        ) from exc
    return keras


def seed_everything(seed: int) -> None:
    """Seed Keras and make its training bit-for-bit reproducible.

    ``keras.utils.set_random_seed`` fixes the initial weights and the batch
    order, but on the TensorFlow backend multithreaded kernels may still sum
    in a different order on every run. Those float-level differences grow
    over the epochs, so two runs with the same seed return different
    weights. ``tf.config.experimental.enable_op_determinism`` removes them.

    Parameters
    ----------
    seed : int
        Seed for Python, NumPy and the Keras backend.

    Notes
    -----
    Op determinism is process-wide and cannot be switched off again, and it
    can make training slower. Ops without a deterministic kernel then raise
    instead of silently varying.
    """
    keras = import_keras()
    keras.utils.set_random_seed(int(seed))
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf  # noqa: PLC0415 -- deliberate lazy import

        tf.config.experimental.enable_op_determinism()


def layer_seeds(base_seed: int | None, n_layers: int) -> list[int | None]:
    """Derive independent, deterministic per-layer seeds from one base seed.

    Parameters
    ----------
    base_seed : int or None
        Base random seed. ``None`` yields a list of ``None`` (unseeded
        initializers).
    n_layers : int
        Number of layer seeds to generate.

    Returns
    -------
    list of int or None
        ``n_layers`` seeds in ``[0, 2**31 - 1)``, derived through
        ``numpy.random.SeedSequence.spawn`` so layers are statistically
        independent yet fully reproducible.

    Raises
    ------
    ValueError
        If ``n_layers`` is negative.
    """
    if n_layers < 0:
        raise ValueError(f"n_layers must be >= 0, got {n_layers}.")
    if base_seed is None:
        return [None] * n_layers
    seq = np.random.SeedSequence(int(base_seed))
    return [
        int(child.generate_state(1, dtype=np.uint32)[0] % _MAX_SEED)
        for child in seq.spawn(n_layers)
    ]


def get_initializer(name: str = "glorot_uniform", seed: int | None = None) -> Any:
    """Build a Keras initializer, seeded when the initializer supports it.

    Parameters
    ----------
    name : str, optional
        Keras initializer identifier (e.g. ``"glorot_uniform"``,
        ``"he_normal"``, ``"zeros"``). Default ``"glorot_uniform"``.
    seed : int or None, optional
        Seed injected into the initializer's config when it has a
        ``"seed"`` entry (deterministic initializers like ``"zeros"``
        are returned unchanged).

    Returns
    -------
    keras.initializers.Initializer
        The (possibly seeded) initializer instance.

    Raises
    ------
    pwlnet.utils.exceptions.TrainingError
        If Keras is not importable.
    """
    keras = import_keras()
    initializer = keras.initializers.get(name)
    if seed is None:
        return initializer
    config = initializer.get_config()
    if "seed" in config:
        config["seed"] = int(seed)
        initializer = type(initializer).from_config(config)
    return initializer
