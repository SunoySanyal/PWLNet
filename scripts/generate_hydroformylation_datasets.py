"""Generate the hydroformylation 15 s transition datasets used by notebook 09.

These files are NOT shipped with the pip package or tracked in git (the final one
is ~50 MB). Run this once after cloning; every stage is skipped when its output
already exists, so re-running is cheap and an interrupted build resumes at the
stage it stopped in.

    python scripts/generate_hydroformylation_datasets.py
    python scripts/generate_hydroformylation_datasets.py --workers 6
    python scripts/generate_hydroformylation_datasets.py --force      # rebuild all

**This takes a while.** It simulates 1230 stiff-ODE trajectories (~2 s each on
one core). Stage 1 is serial (~1 min); stages 2 and 3 are 600 trajectories each,
run in parallel -- measured ~5 min total on 10 workers, ~45 min on one core,
longer on slow machines. ``--max-hours`` is a hard cap (default 3 h); a stage
that hits it is abandoned without writing a file, so a half-built set is never
mistaken for the real one.

Stages (each output is a strict superset of its input: earlier trajectories are
copied through unchanged, new ones come from ``SeedSequence`` children, so the
result is bit-identical regardless of ``--workers``):

1. ``hydroformylation_15s_transitions.npz`` -- 30 trajectories, nominal design
   (noise around the stored 15 s optimum), seed 2479085.
2. ``hydroformylation_15s_transitions_630traj_warp.npz`` -- +600 time-warped
   trajectories (w ~ U[0.25, 4], sigma_log_u 0.6), seed 77410023.
3. ``hydroformylation_15s_transitions_1230traj_warp_tedge.npz`` -- +600
   temperature-edge trajectories (feed cap ~0.5337 mol/min), seed 31887402.
   **This is the file notebook 09 trains on.**

Rows are ``[t/T_FINAL, state(9), T, u] -> next_state(9)``, 280 per trajectory.
Requires only ``ms_15s_optimal.npz`` (the stored optimum, tracked in git).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "pwlnet").exists() and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "datasets"
REFERENCE = DATA_DIR / "ms_15s_optimal.npz"
FINAL_NAME = "hydroformylation_15s_transitions_1230traj_warp_tedge.npz"

#: Measured serial cost of one 280-step trajectory, for the runtime estimate.
SEC_PER_TRAJ = 2.0

#: Feed interlock for the edge design. With u on a warped reference schedule the
#: gas phase stays in the 10-35 bar envelope and this never fires; it only stops
#: one pathological draw from producing a 900 bar trajectory.
P_CEILING = 45.0


@dataclass(frozen=True)
class Stage:
    name: str
    n_new: int
    seed: int
    design: str  # "base" | "warp" | "edge"
    base: str | None = None
    time_warp: tuple[float, float] = (1.0, 1.0)
    sigma_log_u: float = 0.25
    u_cap: float = 0.0


#: Settings recovered from the metadata stored in the original files.
STAGES = (
    Stage("hydroformylation_15s_transitions.npz", 30, 2479085, "base"),
    Stage("hydroformylation_15s_transitions_630traj_warp.npz", 600, 77410023, "warp",
          base="hydroformylation_15s_transitions.npz",
          time_warp=(0.25, 4.0), sigma_log_u=0.6),
    Stage(FINAL_NAME, 600, 31887402, "edge",
          base="hydroformylation_15s_transitions_630traj_warp.npz",
          time_warp=(0.25, 4.0), sigma_log_u=0.6, u_cap=0.5336800871363775),
)


# --------------------------------------------------------------------------
# control designs (worker side)
# --------------------------------------------------------------------------
_REF_CACHE: dict[str, np.ndarray] = {}


def _reference_arrays() -> dict[str, np.ndarray]:
    """The stored 15 s optimum, loaded once per worker process."""
    if not _REF_CACHE:
        with np.load(REFERENCE) as d:
            n = len(np.asarray(d["T"]))
            _REF_CACHE["T"] = np.asarray(d["T"], float)
            _REF_CACHE["u"] = np.asarray(d["u"], float)
            _REF_CACHE["times"] = np.asarray(d["times"], float)[:n]
    return _REF_CACHE


def _warped_profiles(rng, warp_lo, warp_hi, sigma_log_u):
    """Optimum-shaped profiles whose *timing* is randomised.

    Sampling the stored optimum at ``t * w`` for a per-trajectory ``w`` breaks
    the collinearity between elapsed time (an input the optimizer holds fixed)
    and the controls, while keeping every profile physically reachable.
    """
    from pwlnet.processes.hydroformylation import (
        CONTROL_BOUNDS, DT_RATE_LIMIT, MILP_15S_DECISION_DT_MIN, N_SAMPLE_STEPS, T_FINAL,
    )

    grid_t = np.arange(N_SAMPLE_STEPS) * MILP_15S_DECISION_DT_MIN
    w = rng.uniform(warp_lo, warp_hi)
    ref = _reference_arrays()
    warped = np.clip(grid_t * w, 0.0, T_FINAL)
    T_nom = np.interp(warped, ref["times"], ref["T"])
    u_nom = np.interp(warped, ref["times"], ref["u"])

    frac = grid_t / T_FINAL
    T = (T_nom
         + rng.uniform(-12.0, 12.0)
         + rng.uniform(-12.0, 12.0) * frac
         + rng.normal(0.0, 0.18, size=N_SAMPLE_STEPS))
    u = u_nom * np.exp(
        rng.uniform(-0.20, 0.20)
        + rng.uniform(-0.20, 0.20) * frac
        + rng.normal(0.0, sigma_log_u, size=N_SAMPLE_STEPS))

    max_jump = DT_RATE_LIMIT * MILP_15S_DECISION_DT_MIN
    for k in range(N_SAMPLE_STEPS):
        if k > 0:
            T[k] = np.clip(T[k], T[k - 1] - max_jump, T[k - 1] + max_jump)
        T[k] = np.clip(T[k], *CONTROL_BOUNDS[0])
        u[k] = np.clip(u[k], *CONTROL_BOUNDS[1])
    return T, u


def _edge_profiles(rng, warp_lo, warp_hi, sigma_log_u):
    """Piecewise-constant temperature excitation at the control bounds.

    ~80% of segments target a bound, ramped at the rate limit; ``u`` stays on a
    warped reference schedule so pressure remains in the operating envelope.
    """
    from pwlnet.processes.hydroformylation import (
        CONTROL_BOUNDS, DT_RATE_LIMIT, MILP_15S_DECISION_DT_MIN, N_SAMPLE_STEPS,
    )

    _, u = _warped_profiles(rng, warp_lo, warp_hi, sigma_log_u)

    T_lo, T_hi = float(CONTROL_BOUNDS[0][0]), float(CONTROL_BOUNDS[0][1])
    max_jump = DT_RATE_LIMIT * MILP_15S_DECISION_DT_MIN
    n_seg = int(rng.integers(3, 11))
    cuts = np.sort(rng.choice(np.arange(1, N_SAMPLE_STEPS), size=n_seg - 1, replace=False))
    starts = np.r_[0, cuts, N_SAMPLE_STEPS]

    T = np.empty(N_SAMPLE_STEPS)
    T_cur = T_lo if rng.random() < 0.5 else T_hi
    for a, b in zip(starts[:-1], starts[1:]):
        r = rng.random()
        target = T_lo if r < 0.4 else (T_hi if r < 0.8 else rng.uniform(T_lo, T_hi))
        for k in range(a, b):
            T_cur += float(np.clip(target - T_cur, -max_jump, max_jump))
            T[k] = T_cur
    T = np.clip(T + rng.normal(0.0, 0.05, N_SAMPLE_STEPS), T_lo, T_hi)
    return T, u


def _edge_trajectory(T_profile, u_profile, plant):
    """Simulate with a feed interlock above ``P_CEILING``; returns the applied u."""
    from pwlnet.processes.hydroformylation import IDX, INITIAL_STATE, MILP_15S_DECISION_DT_MIN
    from pwlnet.processes.hydroformylation.simulator import step_state

    x = np.asarray(INITIAL_STATE, float).copy()
    states = [x.copy()]
    applied = np.empty(len(T_profile))
    for k, (T_k, u_req) in enumerate(zip(T_profile, u_profile)):
        p_total = x[IDX["p_H2"]] + x[IDX["p_CO"]]
        u_k = 0.0 if p_total >= P_CEILING else float(u_req)
        applied[k] = u_k
        x = step_state(x, float(T_k), u_k, MILP_15S_DECISION_DT_MIN, plant)
        states.append(x.copy())
    times = np.arange(len(states)) * MILP_15S_DECISION_DT_MIN
    return times, np.asarray(states), applied


def one_trajectory(job) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one 15 s trajectory under its own sampled plant."""
    from pwlnet.processes.hydroformylation import STATE_NAMES, T_FINAL, sample_parameters
    from pwlnet.processes.hydroformylation.simulator import simulate_sample_policy

    seed_entropy, design, warp_lo, warp_hi, sigma_log_u = job
    rng = np.random.default_rng(seed_entropy)
    plant = sample_parameters(rng)
    if design == "edge":
        T_profile, u_profile = _edge_profiles(rng, warp_lo, warp_hi, sigma_log_u)
        times, states, u_profile = _edge_trajectory(T_profile, u_profile, plant)
    elif design == "warp":
        T_profile, u_profile = _warped_profiles(rng, warp_lo, warp_hi, sigma_log_u)
        times, states = simulate_sample_policy(T_profile, u_profile, plant)
    else:
        raise ValueError(f"unknown design {design!r}")
    n_state = len(STATE_NAMES)
    rows_x = np.empty((len(T_profile), 1 + n_state + 2), dtype=float)
    rows_y = np.empty((len(T_profile), n_state), dtype=float)
    for k in range(len(T_profile)):
        rows_x[k] = np.r_[times[k] / T_FINAL, states[k], T_profile[k], u_profile[k]]
        rows_y[k] = states[k + 1]
    return rows_x, rows_y


def stage_jobs(stage: Stage) -> list[tuple]:
    children = np.random.SeedSequence(stage.seed).spawn(stage.n_new)
    return [(c.generate_state(4), stage.design, *stage.time_warp, stage.sigma_log_u)
            for c in children]


# --------------------------------------------------------------------------
# stages (driver side)
# --------------------------------------------------------------------------
def _meta(stage: Stage, n_total: int, n_base: int) -> dict:
    from pwlnet.processes.hydroformylation import N_SAMPLE_STEPS, STATE_NAMES, T_FINAL

    meta = dict(
        n_trajectories=n_total, seed=stage.seed,
        steps_per_trajectory=N_SAMPLE_STEPS, t_final_min=T_FINAL,
        feature_names=np.array(["t_over_T_FINAL", *STATE_NAMES, "T", "u"]),
        target_names=np.array(STATE_NAMES),
    )
    if stage.base is not None:
        meta.update(n_base_trajectories=n_base, base_dataset=stage.base,
                    time_warp=np.array(stage.time_warp), sigma_log_u=stage.sigma_log_u)
    if stage.design == "edge":
        meta.update(design="edge", u_cap=stage.u_cap)
    return meta


def _fmt(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def build_base(stage: Stage, deadline: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Stage 1: one sequential RNG stream, so it cannot be parallelised."""
    from pwlnet.processes.hydroformylation import make_sample_transition_dataset

    if time.monotonic() + stage.n_new * SEC_PER_TRAJ > deadline:
        print("  not enough time budget left for this stage")
        return None
    rng = np.random.default_rng(stage.seed)
    return make_sample_transition_dataset(n_trajectories=stage.n_new, rng=rng)


def build_extension(stage: Stage, out_dir: Path, workers: int, deadline: float):
    with np.load(out_dir / stage.base) as d:
        X0, Y0 = np.asarray(d["X"], float), np.asarray(d["Y"], float)
    jobs = stage_jobs(stage)
    new_x, new_y = [], []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (rx, ry) in enumerate(pool.map(one_trajectory, jobs), 1):
            new_x.append(rx)
            new_y.append(ry)
            if i % 25 == 0 or i == len(jobs):
                el = time.monotonic() - t0
                print(f"  {i}/{len(jobs)} trajectories  ({_fmt(el)} elapsed, "
                      f"~{_fmt(el / i * (len(jobs) - i))} left)", flush=True)
            if time.monotonic() > deadline:
                print("  hard time cap reached -- abandoning this stage, nothing written")
                for f in pool._processes.values():  # stop in-flight work promptly
                    f.terminate()
                return None
    return np.vstack([X0, *new_x]), np.vstack([Y0, *new_y]), len(X0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="parallel worker processes for stages 2-3 (default: cores - 2)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild every stage even if its file exists")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR,
                    help="where to write the datasets (default: <repo>/datasets)")
    ap.add_argument("--max-hours", type=float, default=3.0,
                    help="hard wall-clock cap for the whole build (default 3)")
    args = ap.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not REFERENCE.exists():
        print(f"missing {REFERENCE} (the stored 15 s optimum) -- cannot build")
        return 1

    from pwlnet.processes.hydroformylation import N_SAMPLE_STEPS

    todo = [s for s in STAGES if args.force or not (out_dir / s.name).exists()]
    if not todo:
        print(f"all datasets present; {FINAL_NAME} is ready. Use --force to rebuild.")
        return 0
    est = sum(s.n_new * SEC_PER_TRAJ / (1 if s.design == "base" else args.workers)
              for s in todo)
    print(f"building {len(todo)} stage(s) on {args.workers} workers -- estimated "
          f"~{_fmt(est)} (hard cap {args.max_hours:g} h)\n")

    t_start = time.monotonic()
    deadline = t_start + args.max_hours * 3600
    for i, stage in enumerate(STAGES, 1):
        out = out_dir / stage.name
        if stage not in todo:
            print(f"[{i}/3] {stage.name}: exists, skipped")
            continue
        print(f"[{i}/3] {stage.name}: +{stage.n_new} {stage.design} trajectories, "
              f"seed {stage.seed}")
        t0 = time.monotonic()
        if stage.design == "base":
            res = build_base(stage, deadline)
            if res is None:
                return 2
            X, Y = res
            n_base = 0
        else:
            res = build_extension(stage, out_dir, args.workers, deadline)
            if res is None:
                return 2
            X, Y, n_base_rows = res
            n_base = n_base_rows // N_SAMPLE_STEPS
        n_total = len(X) // N_SAMPLE_STEPS
        ok = bool(np.isfinite(X).all() and np.isfinite(Y).all())
        np.savez_compressed(out, X=X, Y=Y, **_meta(stage, n_total, n_base))
        p_next = Y[:, 7] + Y[:, 8]
        print(f"  wrote {out.name}: {n_total} trajectories, X{X.shape}, "
              f"finite={ok}, next p_total [{p_next.min():.1f}, {p_next.max():.1f}] bar, "
              f"{_fmt(time.monotonic() - t0)}\n")

    print(f"done in {_fmt(time.monotonic() - t_start)}. Notebook 09 will now load "
          f"datasets/{FINAL_NAME}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
