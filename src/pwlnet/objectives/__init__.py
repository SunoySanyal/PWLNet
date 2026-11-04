"""Reusable objective terms: data loss, physics loss, regularization, pruning penalties."""

from pwlnet.objectives.data_loss import (
    SUPPORTED_DATA_LOSSES,
    DataLossResult,
    add_mae_loss,
    add_mse_qp_loss,
    add_mse_socp_loss,
    attach_data_loss,
)

__all__ = [
    "DataLossResult",
    "SUPPORTED_DATA_LOSSES",
    "add_mae_loss",
    "add_mse_qp_loss",
    "add_mse_socp_loss",
    "attach_data_loss",
]
