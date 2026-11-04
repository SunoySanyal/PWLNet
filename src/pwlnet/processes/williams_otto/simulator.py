"""Williams-Otto steady-state simulation and dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from pwlnet.processes.williams_otto.model import (
    DEFAULT_INITIAL_GUESS,
    INPUT_NAMES,
    OUTPUT_NAMES,
    WilliamsOttoParameters,
    profit,
    residuals,
)
from pwlnet.utils.typing import ArrayLike
from pwlnet.utils.validation import as_2d_array

__all__ = [
    "WilliamsOttoSteadyState",
    "generate_dataset",
    "simulate_points",
    "solve_steady_state",
]


@dataclass(frozen=True)
class WilliamsOttoSteadyState:
    """Solved Williams-Otto steady state at one operating point."""

    fb: float
    tr: float
    x: np.ndarray
    residual_inf_norm: float
    profit: float

    def to_dict(self) -> dict[str, float]:
        """Return scalar fields keyed by input/output names."""
        row = {"FB": self.fb, "TR": self.tr}
        row.update({name: float(value) for name, value in zip(OUTPUT_NAMES, self.x)})
        row["profit"] = self.profit
        row["residual_inf_norm"] = self.residual_inf_norm
        return row


def solve_steady_state(
    fb: float,
    tr: float,
    initial_guess: ArrayLike | None = None,
    params: WilliamsOttoParameters | None = None,
    residual_tol: float = 1e-10,
    maxfev: int = 200,
) -> WilliamsOttoSteadyState:
    """Solve the six Williams-Otto steady-state balances.

    Parameters
    ----------
    fb, tr : float
        B feed rate [kg/s] and reactor temperature [degC].
    initial_guess : array_like or None, optional
        Initial guess for ``(XA, XB, XC, XE, XG, XP)``.
    params : WilliamsOttoParameters or None, optional
        Parameter set.
    residual_tol : float, optional
        Maximum accepted infinity norm of the balance residuals.
    maxfev : int, optional
        Maximum function evaluations passed to ``scipy.optimize.root``.

    Returns
    -------
    WilliamsOttoSteadyState
        Solved mass fractions, residual norm, and economic profit.

    Raises
    ------
    RuntimeError
        If SciPy fails to converge or the residual/physical checks fail.
    """
    try:
        from scipy.optimize import root
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "scipy is required for Williams-Otto steady-state simulation."
        ) from exc

    p = params if params is not None else WilliamsOttoParameters()
    x0 = (
        np.asarray(initial_guess, dtype=float).ravel()
        if initial_guess is not None
        else DEFAULT_INITIAL_GUESS.copy()
    )
    if x0.shape != (len(OUTPUT_NAMES),):
        raise ValueError(
            f"initial_guess must have {len(OUTPUT_NAMES)} entries, got shape {x0.shape}."
        )

    result = root(
        lambda x: residuals(x, fb=float(fb), tr=float(tr), params=p),
        x0,
        method="hybr",
        options={"maxfev": int(maxfev), "xtol": 1e-11},
    )
    x = np.asarray(result.x, dtype=float)
    res = residuals(x, fb=float(fb), tr=float(tr), params=p)
    residual_norm = float(np.max(np.abs(res)))
    if not bool(result.success) or residual_norm > residual_tol:
        raise RuntimeError(
            "Williams-Otto solve failed at "
            f"FB={fb:.6g}, TR={tr:.6g}: {result.message}; "
            f"max residual={residual_norm:.3e}."
        )
    if np.any(x < -1e-8) or np.any(x > 1.0 + 1e-8):
        raise RuntimeError(
            "Williams-Otto solve produced nonphysical mass fractions at "
            f"FB={fb:.6g}, TR={tr:.6g}: {x}."
        )
    return WilliamsOttoSteadyState(
        fb=float(fb),
        tr=float(tr),
        x=x,
        residual_inf_norm=residual_norm,
        profit=profit(x, float(fb), p),
    )


def simulate_points(
    inputs: ArrayLike,
    params: WilliamsOttoParameters | None = None,
    initial_guess: ArrayLike | None = None,
    warm_start: bool = True,
) -> np.ndarray:
    """Simulate many ``(FB, TR)`` points and return mass fractions."""
    X = as_2d_array(inputs, name="inputs")
    if X.shape[1] != 2:
        raise ValueError(f"inputs must have columns {INPUT_NAMES}, got shape {X.shape}.")
    guess = (
        np.asarray(initial_guess, dtype=float).ravel()
        if initial_guess is not None
        else DEFAULT_INITIAL_GUESS.copy()
    )
    outputs: list[np.ndarray] = []
    for fb, tr in X:
        state = solve_steady_state(
            float(fb), float(tr), initial_guess=guess, params=params
        )
        outputs.append(state.x)
        if warm_start:
            guess = state.x
    return np.vstack(outputs)


def _sample_inputs(
    n_samples: int,
    method: Literal["lhs", "random", "grid"],
    params: WilliamsOttoParameters,
    random_state: int | None,
) -> np.ndarray:
    lb = np.array([params.fb_bounds[0], params.tr_bounds[0]], dtype=float)
    ub = np.array([params.fb_bounds[1], params.tr_bounds[1]], dtype=float)
    method_name = str(method).lower()
    if method_name == "grid":
        n_axis = int(np.ceil(np.sqrt(n_samples)))
        fb = np.linspace(lb[0], ub[0], n_axis)
        tr = np.linspace(lb[1], ub[1], n_axis)
        return np.array([(f, t) for f in fb for t in tr], dtype=float)[:n_samples]
    if method_name == "lhs":
        try:
            from scipy.stats import qmc
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "scipy is required for Latin-hypercube Williams-Otto sampling."
            ) from exc
        sampler = qmc.LatinHypercube(d=2, seed=random_state)
        return qmc.scale(sampler.random(n=int(n_samples)), lb, ub)
    if method_name == "random":
        rng = np.random.default_rng(random_state)
        return rng.uniform(lb, ub, size=(int(n_samples), 2))
    raise ValueError("method must be one of 'lhs', 'random', or 'grid'.")


def generate_dataset(
    n_samples: int = 200,
    method: Literal["lhs", "random", "grid"] = "lhs",
    random_state: int | None = 0,
    params: WilliamsOttoParameters | None = None,
    as_frame: bool = True,
) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
    """Generate a Williams-Otto supervised dataset.

    Returns either a DataFrame with columns ``FB, TR, XA, ..., XP,
    profit, residual_inf_norm`` or a tuple ``(X, y)``.
    """
    p = params if params is not None else WilliamsOttoParameters()
    inputs = _sample_inputs(
        n_samples=int(n_samples), method=method, params=p, random_state=random_state
    )
    rows: list[dict[str, float]] = []
    guess = DEFAULT_INITIAL_GUESS.copy()
    for fb, tr in inputs:
        state = solve_steady_state(float(fb), float(tr), initial_guess=guess, params=p)
        rows.append(state.to_dict())
        guess = state.x
    frame = pd.DataFrame(rows, columns=INPUT_NAMES + OUTPUT_NAMES + ["profit", "residual_inf_norm"])
    if as_frame:
        return frame
    return frame[INPUT_NAMES].to_numpy(dtype=float), frame[OUTPUT_NAMES].to_numpy(dtype=float)
