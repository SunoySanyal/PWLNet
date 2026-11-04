"""Prediction-quality metrics for surrogate evaluation.

The architecture document evaluates surrogates on more than test MSE;
these helpers cover the prediction-quality category (MAE, MSE, RMSE,
:math:`R^2`) and a tidy per-split table so train/validation/test
performance can be compared at a glance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.utils.typing import ArrayLike

__all__ = ["regression_metrics", "metrics_table"]


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute aggregate regression metrics over all outputs.

    Parameters
    ----------
    y_true, y_pred : array_like
        Arrays of equal shape ``(n_samples, n_outputs)`` (1D arrays are
        promoted to a single output column). Metrics are averaged over
        every element, so multi-output targets are weighted equally.

    Returns
    -------
    dict
        ``{"MAE", "MSE", "RMSE", "R2", "n_samples"}``. ``R2`` is the
        coefficient of determination against the per-output mean; it is
        ``nan`` when the targets have zero variance.
    """
    true = np.atleast_2d(np.asarray(y_true, dtype=float))
    pred = np.atleast_2d(np.asarray(y_pred, dtype=float))
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true shape {true.shape} != y_pred shape {pred.shape}."
        )
    if true.size == 0:
        raise ValueError("Cannot compute metrics on an empty array.")

    residual = pred - true
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((true - true.mean(axis=0)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2,
        "n_samples": int(true.shape[0]),
    }


def metrics_table(splits: dict[str, tuple[ArrayLike, ArrayLike]]) -> Any:
    """Build a per-split metrics table.

    Parameters
    ----------
    splits : dict
        Maps a split name (e.g. ``"train"``) to a ``(y_true, y_pred)``
        pair. Insertion order is preserved in the output rows.

    Returns
    -------
    pandas.DataFrame
        One row per split with the :func:`regression_metrics` columns
        plus a leading ``split`` column.
    """
    import pandas as pd

    rows = []
    for name, (y_true, y_pred) in splits.items():
        row = {"split": name}
        row.update(regression_metrics(y_true, y_pred))  # type: ignore
        rows.append(row)
    return pd.DataFrame(rows)
