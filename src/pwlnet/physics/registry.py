"""Registry of physics-residual builders.

A small name -> factory map so residuals can be referred to by string
(e.g. in config-driven training) and discovered programmatically.
"""

from __future__ import annotations

from typing import Any, Callable

from pwlnet.physics.bounds import (
    box_residual,
    lower_bound_residual,
    nonnegativity_residual,
    upper_bound_residual,
)
from pwlnet.physics.conservation import (
    conservation_residual,
    mass_balance_residual,
    sum_to_one_residual,
)
from pwlnet.physics.algebraic import AffineResidual, linear_residual

__all__ = ["RESIDUAL_BUILDERS", "build_residual", "available_residuals"]

#: Registry of residual builder callables keyed by short name.
RESIDUAL_BUILDERS: dict[str, Callable[..., Any]] = {
    "affine": AffineResidual,
    "linear": linear_residual,
    "sum_to_one": sum_to_one_residual,
    "mass_balance": mass_balance_residual,
    "conservation": conservation_residual,
    "upper_bound": upper_bound_residual,
    "lower_bound": lower_bound_residual,
    "nonnegativity": nonnegativity_residual,
    "box": box_residual,
}


def available_residuals() -> list[str]:
    """Sorted list of registered residual names."""
    return sorted(RESIDUAL_BUILDERS)


def build_residual(name: str, *args: Any, **kwargs: Any) -> Any:
    """Build a residual by registered name.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    key = str(name).strip().lower()
    if key not in RESIDUAL_BUILDERS:
        raise KeyError(
            f"Unknown residual {name!r}. Available: {available_residuals()}."
        )
    return RESIDUAL_BUILDERS[key](*args, **kwargs)
