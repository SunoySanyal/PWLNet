"""PWLNet: optimization-aware neural network surrogates.

PWLNet trains compact ReLU/tanh networks and embeds them into algebraic
optimization models via OMLT, custom Pyomo piecewise-linear formulations,
and a SOSX branching engine.
"""

__version__ = "0.1.1"

from pwlnet.api import PWLNet
from pwlnet.benchmarking import metrics_table, regression_metrics
from pwlnet.branching import SosxBrancher, SosxResult
from pwlnet.core import (
    ACTIVATIONS,
    ActivationInfo,
    BoundsResult,
    DenseLayer,
    ModelSummary,
    NetworkDefinition,
    OffsetScaler,
    estimate_formulation_size,
    get_activation,
    identity,
    propagate_bounds,
    relu,
    summarize,
    tanh,
)
from pwlnet.data import DataSplit, train_test_split, train_val_test_split
from pwlnet.pruning import (
    PruneOutcome,
    PruneReport,
    build_prune_report,
    compare_networks,
)
from pwlnet.recurrent import (
    RecurrentBounds,
    RecurrentCell,
    build_unrolled_mpc_model,
    propagate_recurrent_bounds,
    solve_unrolled_mpc,
)
from pwlnet.solvers import SosxSurrogateResult, solve_sosx
from pwlnet.training.keras_trainer import KerasTrainer
from pwlnet.training.pwl_activation import make_pwl_tanh_activation
from pwlnet.training.rnn_trainer import RNNTrainer
from pwlnet.training.sosx_trainer import SosxTrainer
from pwlnet.training.physics_informed import (
    GenericAffineResidualPhysics,
    PhysicsSpec,
    PhysicsInformedTrainer,
    PhysicsLoss,
    affine_residual_value,
    register_physics,
)
from pwlnet.processes.williams_otto import WilliamsOttoPhysics

__all__ = [
    "__version__",
    "PWLNet",
    "ACTIVATIONS",
    "ActivationInfo",
    "BoundsResult",
    "DataSplit",
    "DenseLayer",
    "GenericAffineResidualPhysics",
    "KerasTrainer",
    "PhysicsSpec",
    "PhysicsInformedTrainer",
    "PhysicsLoss",
    "WilliamsOttoPhysics",
    "ModelSummary",
    "NetworkDefinition",
    "OffsetScaler",
    "PruneOutcome",
    "PruneReport",
    "RNNTrainer",
    "RecurrentBounds",
    "RecurrentCell",
    "SosxBrancher",
    "SosxResult",
    "SosxSurrogateResult",
    "SosxTrainer",
    "build_prune_report",
    "build_unrolled_mpc_model",
    "compare_networks",
    "propagate_recurrent_bounds",
    "solve_unrolled_mpc",
    "estimate_formulation_size",
    "solve_sosx",
    "affine_residual_value",
    "get_activation",
    "make_pwl_tanh_activation",
    "identity",
    "metrics_table",
    "propagate_bounds",
    "regression_metrics",
    "relu",
    "register_physics",
    "summarize",
    "tanh",
    "train_test_split",
    "train_val_test_split",
]
