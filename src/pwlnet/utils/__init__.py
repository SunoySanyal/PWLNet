"""Shared utilities for PWLNet: exceptions, validation, logging, typing."""

from pwlnet.utils.exceptions import (
    PWLNetError,
    NetworkValidationError,
    FormulationError,
    SolverNotAvailableError,
    TrainingError,
)
from pwlnet.utils.logging import get_logger
from pwlnet.utils.validation import as_2d_array, check_shape, check_shapes

__all__ = [
    "PWLNetError",
    "NetworkValidationError",
    "FormulationError",
    "SolverNotAvailableError",
    "TrainingError",
    "get_logger",
    "as_2d_array",
    "check_shape",
    "check_shapes",
]
