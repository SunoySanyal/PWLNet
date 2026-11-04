"""Williams-Otto reactor benchmark: model, simulator, Pyomo model, kinetics."""

from pwlnet.processes.williams_otto.model import (
    CELSIUS_TO_KELVIN,
    DEFAULT_INITIAL_GUESS,
    INPUT_NAMES,
    OUTPUT_NAMES,
    WilliamsOttoParameters,
    profit,
    rate_constants,
    residuals,
)
from pwlnet.processes.williams_otto.physics import WilliamsOttoPhysics
from pwlnet.processes.williams_otto.simulator import (
    WilliamsOttoSteadyState,
    generate_dataset,
    simulate_points,
    solve_steady_state,
)
from pwlnet.processes.williams_otto.pyomo_model import (
    EXACT_OPTIMUM,
    SurrogateOptimizationResult,
    build_exact_model,
    build_surrogate_optimization_model,
    solve_surrogate_optimization_model,
)

__all__ = [
    "CELSIUS_TO_KELVIN",
    "DEFAULT_INITIAL_GUESS",
    "INPUT_NAMES",
    "OUTPUT_NAMES",
    "WilliamsOttoParameters",
    "WilliamsOttoPhysics",
    "WilliamsOttoSteadyState",
    "EXACT_OPTIMUM",
    "SurrogateOptimizationResult",
    "build_exact_model",
    "build_surrogate_optimization_model",
    "generate_dataset",
    "profit",
    "rate_constants",
    "residuals",
    "simulate_points",
    "solve_steady_state",
    "solve_surrogate_optimization_model",
]
