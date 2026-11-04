"""Construction helpers for Keras training callbacks.

Keras is imported lazily inside the factory functions, so importing this
module never requires TensorFlow/Keras.
"""

from __future__ import annotations

import logging
from typing import Any

from pwlnet.training.initialization import import_keras
from pwlnet.utils.logging import get_logger

__all__ = ["make_early_stopping", "make_epoch_logger"]


def make_early_stopping(
    patience: int = 10,
    monitor: str = "val_loss",
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
    verbose: int = 0,
) -> Any:
    """Build a ``keras.callbacks.EarlyStopping`` callback.

    Parameters
    ----------
    patience : int, optional
        Number of epochs with no improvement before training stops.
    monitor : str, optional
        Metric to monitor, e.g. ``"val_loss"`` (use ``"loss"`` when
        training without a validation split).
    min_delta : float, optional
        Minimum change in the monitored metric to count as improvement.
    restore_best_weights : bool, optional
        Restore the weights from the best epoch when stopping. Default
        True.
    verbose : int, optional
        Keras verbosity for the callback itself.

    Returns
    -------
    keras.callbacks.EarlyStopping
        The configured callback.

    Raises
    ------
    pwlnet.utils.exceptions.TrainingError
        If Keras is not importable.
    """
    keras = import_keras()
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=int(patience),
        min_delta=float(min_delta),
        restore_best_weights=restore_best_weights,
        verbose=verbose,
    )


def make_epoch_logger(
    log_every: int = 10,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> Any:
    """Build a lightweight callback that logs epoch metrics via ``logging``.

    The callback emits one record per ``log_every`` epochs (plus the
    first epoch) through the ``pwlnet`` logger hierarchy, which is silent
    by default (NullHandler) unless the application configures logging.

    Parameters
    ----------
    log_every : int, optional
        Log every this-many epochs. Values < 1 are clamped to 1.
    logger : logging.Logger or None, optional
        Target logger; defaults to ``pwlnet.training``.
    level : int, optional
        Logging level for the emitted records. Default ``logging.INFO``.

    Returns
    -------
    keras.callbacks.Callback
        The configured epoch-logging callback.

    Raises
    ------
    pwlnet.utils.exceptions.TrainingError
        If Keras is not importable.
    """
    keras = import_keras()
    log = logger if logger is not None else get_logger("training")

    class EpochLogger(keras.callbacks.Callback):
        """Log training metrics every ``every`` epochs."""

        def __init__(self, every: int, log_: logging.Logger, level_: int) -> None:
            super().__init__()
            self.every = max(1, int(every))
            self._log = log_
            self._level = level_

        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            if (epoch + 1) % self.every == 0 or epoch == 0:
                metrics = ", ".join(
                    f"{key}={float(value):.6g}"
                    for key, value in sorted((logs or {}).items())
                )
                self._log.log(self._level, "epoch %d: %s", epoch + 1, metrics)

    return EpochLogger(log_every, log, level)
