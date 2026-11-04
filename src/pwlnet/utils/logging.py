"""Logging helpers for PWLNet library code.

Library modules use :func:`get_logger` instead of ``print``. A
``NullHandler`` is attached to the package root logger so importing
PWLNet never emits "No handlers could be found" warnings; applications
opt in to output via :func:`pwlnet.config.configure_logging`.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "pwlnet"

# Library best practice: silence by default.
logging.getLogger(_ROOT_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger in the ``pwlnet`` namespace.

    Parameters
    ----------
    name : str or None, optional
        Logger name. ``None`` returns the package root logger. Names not
        already under ``"pwlnet"`` are nested beneath it, so
        ``get_logger("core.bounds")`` and ``get_logger(__name__)`` from
        inside the package both yield ``pwlnet.*`` loggers.

    Returns
    -------
    logging.Logger
        The requested logger.
    """
    if name is None or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
