"""SOSX node relaxations, violations and branching rules.

This subpackage provides the *formulation* side of the SOSX engine: the
reusable SOS2-dropped node relaxation (:class:`SosxRelaxation`), the
SOS2 violation detector, and the breakpoint-window branching rules. The
branch-and-bound *driver* lives in :mod:`pwlnet.branching`.
"""

from pwlnet.formulations.sosx.sosx_branching_rules import (
    SELECTION_RULES,
    select_violation,
    split_window,
    split_window_at,
    split_window_Y,
)
from pwlnet.formulations.sosx.sosx_node import SosxRelaxation, discover_sos2_blocks
from pwlnet.formulations.sosx.sosx_solution import NodeSolution, SosxResult
from pwlnet.formulations.sosx.sosx_violation import (
    find_first_violation,
    find_violations,
    find_violations_Y,
    support_spread,
)
from pwlnet.formulations.sosx.training_model import (
    Sos2TrainingModel,
    SosxTrainingModel,
    build_sos2_mae_training_model,
    build_sosx_mae_training_model,
)
from pwlnet.formulations.sosx.training_relaxation import (
    Sos2TrainingRelaxation,
    SosxTrainingRelaxation,
)
from pwlnet.formulations.sosx.training_solution import (
    network_from_tracked_values,
    predict_pwl_tanh,
    pwl_training_mae,
    track_training_variables,
    training_incumbent_from_network,
    training_incumbent_lam,
    training_incumbent_Y,
)
from pwlnet.formulations.sosx.weight_bounds import (
    WeightBounds,
    bounds_covering_network,
    resolve_weight_bounds,
)

__all__ = [
    "NodeSolution",
    "SELECTION_RULES",
    "Sos2TrainingModel",
    "Sos2TrainingRelaxation",
    "SosxRelaxation",
    "SosxResult",
    "SosxTrainingModel",
    "SosxTrainingRelaxation",
    "WeightBounds",
    "bounds_covering_network",
    "build_sos2_mae_training_model",
    "build_sosx_mae_training_model",
    "discover_sos2_blocks",
    "find_first_violation",
    "find_violations",
    "find_violations_Y",
    "network_from_tracked_values",
    "predict_pwl_tanh",
    "pwl_training_mae",
    "resolve_weight_bounds",
    "select_violation",
    "split_window",
    "split_window_at",
    "split_window_Y",
    "support_spread",
    "track_training_variables",
    "training_incumbent_from_network",
    "training_incumbent_lam",
    "training_incumbent_Y",
]
