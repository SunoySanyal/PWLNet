"""Custom Pyomo piecewise-linear formulations and data utilities."""

from pwlnet.formulations.pyomo_pwl.breakpoints import (
    STANDARD_TANH_BREAKPOINTS,
    get_tanh_breakpoints,
    validate_breakpoints,
)
from pwlnet.formulations.pyomo_pwl.conic_epigraphs import (
    add_quadratic_epigraph,
)
from pwlnet.formulations.pyomo_pwl.piecewise_data import (
    PiecewiseLinearData,
    build_piecewise_linear_data,
    build_tanh_pwl_data,
    extend_piecewise_linear_data,
)
from pwlnet.formulations.pyomo_pwl.tanh_pwl_bigm import (
    build_tanh_pwl_bigm_block,
    build_tanh_pwl_bigm_model,
)
from pwlnet.formulations.pyomo_pwl.tanh_pwl_sos2 import (
    build_tanh_pwl_sos2_block,
    build_tanh_pwl_sos2_model,
    evaluate_tanh_pwl_network,
)

__all__ = [
    "PiecewiseLinearData",
    "STANDARD_TANH_BREAKPOINTS",
    "add_quadratic_epigraph",
    "build_piecewise_linear_data",
    "build_tanh_pwl_data",
    "build_tanh_pwl_bigm_block",
    "build_tanh_pwl_bigm_model",
    "build_tanh_pwl_sos2_block",
    "build_tanh_pwl_sos2_model",
    "evaluate_tanh_pwl_network",
    "extend_piecewise_linear_data",
    "get_tanh_breakpoints",
    "validate_breakpoints",
]
