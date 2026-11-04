"""Keras/torch/optimization-based trainers producing core NetworkDefinitions."""

from pwlnet.training.keras_trainer import KerasTrainer
from pwlnet.training.parallel_sosx import (
    ParallelSosxBrancher,
    SosxTrainingSpec,
    default_worker_count,
)
from pwlnet.training.pwl_activation import PWL_TANH, make_pwl_tanh_activation
from pwlnet.training.physics_informed import (
    GenericAffineResidualPhysics,
    PhysicsSpec,
    PhysicsInformedTrainer,
    PhysicsLoss,
    affine_residual_value,
    register_physics,
)
from pwlnet.training.sosx_trainer import SosxTrainer

__all__ = [
    "GenericAffineResidualPhysics",
    "PWL_TANH",
    "KerasTrainer",
    "ParallelSosxBrancher",
    "PhysicsSpec",
    "PhysicsInformedTrainer",
    "PhysicsLoss",
    "SosxTrainer",
    "SosxTrainingSpec",
    "affine_residual_value",
    "default_worker_count",
    "make_pwl_tanh_activation",
    "register_physics",
]
