"""Physics residuals: algebraic, conservation, monotonicity and bounds.

Reusable physical-consistency objects (architecture document section 6).
The first-target residuals are affine in the surrogate inputs/outputs and
expose their coefficient matrices, so one description drives NumPy
evaluation, the differentiable Keras penalty used in physics-informed
gradient training, and (later) Pyomo constraints.
"""

from pwlnet.physics.algebraic import AffineResidual, linear_residual
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
from pwlnet.physics.monotonicity import (
    MonotonicityConstraint,
    finite_difference_violation,
)
from pwlnet.physics.registry import (
    RESIDUAL_BUILDERS,
    available_residuals,
    build_residual,
)
from pwlnet.physics.residuals import RESIDUAL_KINDS, PhysicsResidual

__all__ = [
    "PhysicsResidual",
    "RESIDUAL_KINDS",
    "AffineResidual",
    "linear_residual",
    "sum_to_one_residual",
    "mass_balance_residual",
    "conservation_residual",
    "upper_bound_residual",
    "lower_bound_residual",
    "nonnegativity_residual",
    "box_residual",
    "MonotonicityConstraint",
    "finite_difference_violation",
    "RESIDUAL_BUILDERS",
    "available_residuals",
    "build_residual",
]
