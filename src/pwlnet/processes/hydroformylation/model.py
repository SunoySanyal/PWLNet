"""Hydroformylation semibatch reactor model constants and kinetics.

This module contains the Hentschel-aligned kinetics and Aydin-style NMPC
timing assumptions that were developed in notebook 04. The state vector is
ordered as :data:`STATE_NAMES`; liquid concentrations are stored in mol/L and
gas partial pressures in bar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "COMPOUND_NAMES",
    "CONCENTRATION_NAMES",
    "CONTROL_BOUNDS",
    "CONTROL_DT_MIN",
    "CONTROL_NAMES",
    "DT_MIN",
    "DT_RATE_LIMIT",
    "EA_H_CO",
    "EA_H_H2",
    "H0_CO",
    "H0_H2",
    "IDX",
    "INITIAL_C_CO",
    "INITIAL_C_H2",
    "INITIAL_P_CO_BAR",
    "INITIAL_P_H2_BAR",
    "INITIAL_STATE",
    "INITIAL_TEMPERATURE_K",
    "K0_RANGE_FACTORS",
    "MEASUREMENT_DELAY_MIN",
    "MEASUREMENT_DELAY_SEC",
    "MEASUREMENT_INTERVAL_MIN",
    "MEASUREMENT_INTERVAL_SEC",
    "MILP_15S_CLOSED_LOOP_MIN",
    "MILP_15S_CLOSED_LOOP_STEPS",
    "MILP_15S_DECISION_DT_MIN",
    "MILP_15S_HIDDEN_NEURONS",
    "MILP_15S_PREDICTION_STEPS",
    "MILP_15S_TRAIN_TRAJECTORIES",
    "N_SAMPLE_STEPS",
    "N_STEPS",
    "PARAM_RANGES",
    "PLOT_COMPOUND_NAMES",
    "R_KJ",
    "SAMPLES_PER_CONTROL",
    "SAMPLE_TIME_MIN",
    "SAMPLE_TIME_SEC",
    "STATE_BOUNDS",
    "STATE_NAMES",
    "T_FINAL",
    "HydroformylationParameters",
    "arrhenius_table6",
    "dissolved_gas_equilibrium_mol_l",
    "equilibrium_constant",
    "pressure_backoff",
    "rhs",
    "sample_parameters",
]

STATE_NAMES = [
    "c_H2",
    "c_CO",
    "nC12en",
    "iC12en",
    "nC13al",
    "nC12an",
    "iC13al",
    "p_H2",
    "p_CO",
]
COMPOUND_NAMES = ["nC12an", "nC12en", "iC12en", "nC13al", "iC13al"]
PLOT_COMPOUND_NAMES = ["nC12en", "nC13al", "iC12en", "iC13al", "nC12an"]
CONCENTRATION_NAMES = ["c_H2", "c_CO", *STATE_NAMES[2:7]]
CONTROL_NAMES = ["T", "u"]
IDX = {name: i for i, name in enumerate(STATE_NAMES)}

STATE_BOUNDS = np.array(
    [
        [0.00, 0.60],
        [0.00, 0.60],
        [0.00, 1.20],
        [0.00, 0.80],
        [0.00, 0.90],
        [0.00, 0.80],
        [0.00, 0.80],
        [0.10, 22.0],
        [0.10, 22.0],
    ],
    dtype=float,
)
# The partial-pressure upper bounds are a *box* for the surrogate's state space,
# not the process limit -- the 20 bar cap is on the TOTAL pressure and is imposed
# as an explicit path constraint. Training trajectories deliberately over- and
# under-feed around the optimum to expose the constraint region, so a single
# partial pressure reaches ~18 bar; 22 leaves headroom without being vacuous.
CONTROL_BOUNDS = np.array([[368.15, 388.15], [0.0, 7.0]], dtype=float)

T_FINAL = 70.0
SAMPLE_TIME_SEC = 15.0
SAMPLE_TIME_MIN = SAMPLE_TIME_SEC / 60.0
MEASUREMENT_INTERVAL_SEC = 30.0
MEASUREMENT_INTERVAL_MIN = MEASUREMENT_INTERVAL_SEC / 60.0
MEASUREMENT_DELAY_SEC = 5.0
MEASUREMENT_DELAY_MIN = MEASUREMENT_DELAY_SEC / 60.0

CONTROL_DT_MIN = 5.0
DT_MIN = CONTROL_DT_MIN
SAMPLES_PER_CONTROL = int(round(CONTROL_DT_MIN / SAMPLE_TIME_MIN))
N_STEPS = int(T_FINAL // CONTROL_DT_MIN)
N_SAMPLE_STEPS = int(round(T_FINAL / SAMPLE_TIME_MIN))
DT_RATE_LIMIT = 0.35 / SAMPLE_TIME_MIN

MILP_15S_DECISION_DT_MIN = SAMPLE_TIME_MIN
MILP_15S_PREDICTION_STEPS = 8
MILP_15S_CLOSED_LOOP_MIN = 10.0
MILP_15S_CLOSED_LOOP_STEPS = int(round(MILP_15S_CLOSED_LOOP_MIN / MILP_15S_DECISION_DT_MIN))
MILP_15S_TRAIN_TRAJECTORIES = 30
MILP_15S_HIDDEN_NEURONS = 16

INITIAL_TEMPERATURE_K = 378.15
INITIAL_P_H2_BAR = 10.0
INITIAL_P_CO_BAR = 10.0
H0_H2 = 66400.0
H0_CO = 73900.0
EA_H_H2 = -3.06
EA_H_CO = -0.84
R_KJ = 0.008314462618


def dissolved_gas_equilibrium_mol_l(
    p_bar: float, h0: float, ea_h_kj_mol: float, T: float
) -> float:
    """Henry-law dissolved gas concentration in mol/L."""
    H = h0 * np.exp(-ea_h_kj_mol / (R_KJ * T))
    return 1000.0 * p_bar / H


INITIAL_C_H2 = dissolved_gas_equilibrium_mol_l(
    INITIAL_P_H2_BAR, H0_H2, EA_H_H2, INITIAL_TEMPERATURE_K
)
INITIAL_C_CO = dissolved_gas_equilibrium_mol_l(
    INITIAL_P_CO_BAR, H0_CO, EA_H_CO, INITIAL_TEMPERATURE_K
)
INITIAL_STATE = np.array(
    [
        0.0,  # no gas dissolved at t = 0 (Hentschel/Kiedorf batch start condition)
        0.0,  # no gas dissolved at t = 0
        0.85 / 0.9,
        0.0,
        0.0,
        0.0,
        0.0,
        INITIAL_P_H2_BAR,
        INITIAL_P_CO_BAR,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class HydroformylationParameters:
    """Hentschel-aligned hydroformylation kinetic and reactor parameters."""

    activation_energy_kj_mol: tuple[float, ...] = (
        113.08,
        136.89,
        76.11,
        102.26,
        120.84,
        113.08,
    )
    k0: tuple[float, ...] = (
        4.904e16,
        4.878e6,
        5.411e8,
        2.958e4,
        7.619e10,
        3.951e11,
    )
    k1_ads: tuple[float, ...] = (574876.0, 3020413.0, 11732838.0)
    k2_ads: tuple[float, ...] = (38632.0, 226214.0)
    k3_ads: tuple[float, ...] = (2661.2, 7100.0, 1280.0)
    dg2_coeff: tuple[float, ...] = (-11.0034, 0.0, 0.0)
    dg3_coeff: tuple[float, ...] = (-126.275, 0.1266, 6.803e-6)
    h0_h2: float = 66400.0
    h0_co: float = 73900.0
    ea_h_h2: float = -3.06
    ea_h_co: float = -0.84
    kla_h2: float = 9.57
    kla_co: float = 7.08
    kcat_1: float = 3.041e4
    kcat_2: float = 0.0
    kcat_3: float = 0.644
    v_liq_ml: float = 900.0
    v_gas_ml: float = 900.0
    tref: float = 378.15
    gamma: float = 1.0
    initial_olefin_mol: float = 0.85
    catalyst_to_olefin_mol_ratio: float = 1.0 / 10000.0
    active_catalyst_molar_mass_g_mol: float = 258.03


PARAM_RANGES = {
    "kla_h2": (8.0, 10.1),
    "kla_co": (5.5, 7.6),
    "gamma": (0.8, 1.0),
}

K0_RANGE_FACTORS = np.array(
    [
        [2.9 / 4.904, 5.2 / 4.904],
        [3.8 / 4.878, 5.8 / 4.878],
        [1.7 / 2.724, 3.7 / 2.724],
        [1.9 / 2.958, 3.9 / 2.958],
        [2.7 / 3.702, 4.7 / 3.702],
        [2.9 / 3.951, 4.9 / 3.951],
    ],
    dtype=float,
)


def pressure_backoff(t_min: float) -> float:
    """Pressure safety backoff used by the NMPC benchmark."""
    if t_min <= 30.0:
        return 1.3
    if t_min <= 55.0:
        return 0.7
    return 0.0


def sample_parameters(rng: np.random.Generator) -> HydroformylationParameters:
    """Sample uncertain kinetic/transfer parameters used for surrogate data."""
    base = HydroformylationParameters()
    kwargs = {key: rng.uniform(lo, hi) for key, (lo, hi) in PARAM_RANGES.items()}
    factors = rng.uniform(K0_RANGE_FACTORS[:, 0], K0_RANGE_FACTORS[:, 1])
    kwargs["k0"] = tuple(np.asarray(base.k0) * factors)
    return replace(base, **kwargs)


def arrhenius_table6(
    k0: np.ndarray, ea_kj_mol: np.ndarray, T: float, tref: float
) -> np.ndarray:
    """Temperature-adjust the Table-6 kinetic constants."""
    return k0 * np.exp(-ea_kj_mol / R_KJ * (1.0 / T - 1.0 / tref))


def equilibrium_constant(coeff: tuple[float, float, float], T: float) -> float:
    """Return ``Kp = exp(-DeltaG / RT)`` from the Table-6 polynomial."""
    a0, a1, a2 = coeff
    delta_g = a0 + a1 * T + a2 * T * T
    return float(np.exp(-delta_g / (R_KJ * T)))


def rhs(x: np.ndarray, T: float, u: float, p: HydroformylationParameters) -> np.ndarray:
    """Right-hand side of the semibatch hydroformylation plant model."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    c_h2_l, c_co_l, c_n_l, c_i_l, c_al_l, c_alk_l, c_isoal_l, p_h2, p_co = x
    c_h2, c_co, c_n, c_i, c_al, c_alk, c_isoal = x[:7] / 1000.0

    h_h2 = p.h0_h2 * np.exp(-p.ea_h_h2 / (R_KJ * T))
    h_co = p.h0_co * np.exp(-p.ea_h_co / (R_KJ * T))
    j_h2 = p.kla_h2 * (p_h2 / h_h2 - c_h2)
    j_co = p.kla_co * (p_co / h_co - c_co)

    k = arrhenius_table6(
        np.asarray(p.k0, dtype=float),
        np.asarray(p.activation_energy_kj_mol, dtype=float),
        T,
        p.tref,
    )
    kp2 = equilibrium_constant(p.dg2_coeff, T)
    kp3 = equilibrium_constant(p.dg3_coeff, T)

    k1_1, k1_2, k1_3 = p.k1_ads
    k2_1, k2_2 = p.k2_ads
    k3_1, k3_2, k3_3 = p.k3_ads
    cat_denom = 1.0 + p.kcat_1 * max(c_co, 0.0) ** p.kcat_3
    if p.kcat_2 != 0.0:
        cat_denom += p.kcat_2 * max(c_co, 1e-18) ** p.kcat_3 / max(c_h2, 1e-18)
    catalyst_inventory_g_per_ml = (
        p.initial_olefin_mol
        * p.catalyst_to_olefin_mol_ratio
        * p.active_catalyst_molar_mass_g_mol
        / p.v_liq_ml
    )
    active_catalyst_g_per_ml = catalyst_inventory_g_per_ml * p.gamma / cat_denom

    r1 = k[0] * c_n * c_h2 * c_co / (1.0 + k1_1 * c_n + k1_2 * c_al + k1_3 * c_h2)
    r2 = k[1] * (c_n - c_i / kp2) / (1.0 + k2_1 * c_n + k2_2 * c_i)
    r3 = k[2] * (c_n * c_h2 - c_alk / kp3) / (
        1.0 + k3_1 * c_n + k3_2 * c_alk + k3_3 * c_h2
    )
    r4 = k[3] * c_i * c_h2
    r5 = k[4] * c_i * c_h2 * c_co
    r6 = k[5] * c_n * c_h2 * c_co
    q = active_catalyst_g_per_ml * np.array([r1, r2, r3, r4, r5, r6])

    dx = np.zeros_like(x)
    dx[IDX["c_H2"]] = 1000.0 * (j_h2 - q[0] - q[2] - q[3] - q[4] - q[5])
    dx[IDX["c_CO"]] = 1000.0 * (j_co - q[0] - q[4] - q[5])
    dx[IDX["nC12en"]] = 1000.0 * (-q[0] - q[1] - q[2] - q[5])
    dx[IDX["iC12en"]] = 1000.0 * (q[1] - q[3] - q[4])
    dx[IDX["nC13al"]] = 1000.0 * q[0]
    dx[IDX["nC12an"]] = 1000.0 * (q[2] + q[3])
    dx[IDX["iC13al"]] = 1000.0 * (q[4] + q[5])

    gas_gain = R_KJ * 10000.0 * T / p.v_gas_ml
    transfer_h2_mol_min = p.v_liq_ml * j_h2
    transfer_co_mol_min = p.v_liq_ml * j_co
    dx[IDX["p_H2"]] = gas_gain * (0.5 * u - transfer_h2_mol_min)
    dx[IDX["p_CO"]] = gas_gain * (0.5 * u - transfer_co_mol_min)
    return dx
