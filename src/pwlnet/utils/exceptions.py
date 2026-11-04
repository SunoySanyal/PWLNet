"""Exception hierarchy for PWLNet.

All package-specific errors derive from :class:`PWLNetError` so callers
can catch a single base class.
"""

from __future__ import annotations


class PWLNetError(Exception):
    """Base class for all PWLNet errors."""


class NetworkValidationError(PWLNetError):
    """Raised when a network definition is structurally invalid.

    Examples include mismatched layer shapes, scaler dimensions that do
    not match the network, unknown activation names, or missing/invalid
    input bounds.
    """


class FormulationError(PWLNetError):
    """Raised when an optimization formulation cannot be built.

    Examples include requesting a ReLU big-M formulation for a tanh
    network, missing bounds required for big-M constants, or an unknown
    formulation name.
    """


class SolverNotAvailableError(PWLNetError):
    """Raised when a requested solver cannot be found or constructed."""


class TrainingError(PWLNetError):
    """Raised when model training fails or produces an unusable model."""
