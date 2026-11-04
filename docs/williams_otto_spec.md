# Williams-Otto Steady-State CSTR Benchmark — Implementation Specification

**Status:** verified, implementation-ready.
**Scope:** the steady-state Williams-Otto *reactor* (CSTR) benchmark on a mass-fraction
basis — the standard real-time-optimization (RTO) formulation descending from
Williams & Otto (1960) via Forbes & Marlin — as used for surrogate training and
downstream economic optimization in PWLNet (`src/pwlnet/processes/williams_otto/`).
The full Williams-Otto *plant* (reactor + decanter + distillation column + recycle),
as simulated dynamically in Pyomo by Schmid et al. [S1], is out of scope here but its
constants are used below as an independent cross-check.

All numbers below were cross-checked against at least two independent sources
(see §9 and the per-section citations). The model was additionally verified
numerically (§8): the six balances close to machine precision and imply
`sum(X) = 1` exactly.

---

## 1. Reaction system and kinetics (mass basis)

Three irreversible, second-order reactions occur in an ideally mixed, isothermal
CSTR ([S1] eqs. (3.1)–(3.3); [S4] eq. (16); [S6]; [S7]):

| # | Reaction | Rate constant |
|---|-------------------|---------------|
| 1 | A + B -> C        | k1 |
| 2 | B + C -> P + E    | k2 |
| 3 | C + P -> G        | k3 |

P is the desired product, E a saleable by-product, C an intermediate, G waste.

### Arrhenius constants (SI, mass-fraction basis — canonical)

With the reactor temperature `T_R` in **degrees Celsius** and the absolute
temperature `T = T_R + 273.15` in **Kelvin**:

```
k1 = 1.6599e6  * exp(-6666.7  / (T_R + 273.15))   [1/s]
k2 = 7.2117e8  * exp(-8333.3  / (T_R + 273.15))   [1/s]
k3 = 2.6745e12 * exp(-11111.0 / (T_R + 273.15))   [1/s]
```

- Units: a_i in s^-1, b_i in K. The k_i have units s^-1 because the rates (§2)
  multiply two dimensionless mass fractions and the mass holdup W (kg) to give kg/s.
- Sources: [S6] (reference implementation, exact lines), [S7] (eqs. (26), same
  numbers, T_R explicitly in degC), [S5] (same numbers; note [S5] prints
  `7.2177e8` for a2 — a typo for `7.2117e8`, see cross-check below).
- Some sources write the activation terms with more digits
  (`6666.67`, `8333.33`, `11111.1`); the difference is far below model noise.

### Cross-check against the original (1960) unit system

The original Williams-Otto constants (masses in klb, time in h, temperature in
degrees Rankine), as restated by Schmid et al. [S1] eqs. (3.12)–(3.14) and
independently by [S10] (Annexe AN6) and [S11] (eq. (S47)):

```
a1 = 5.9755e9  1/h     b1 = 12000 degR     rho = 50 lb/ft^3
a2 = 2.5962e12 1/h     b2 = 15000 degR
a3 = 9.6283e15 1/h     b3 = 20000 degR
```

Exact unit conversion confirms the SI table above:

```
a_i [1/s] = a_i [1/h] / 3600      ->  5.9755e9/3600  = 1.6599e6
                                       2.5962e12/3600 = 7.2117e8   (not 7.2177e8)
                                       9.6283e15/3600 = 2.6745e12
b_i [K]   = b_i [degR] / 1.8      ->  12000/1.8 = 6666.7
                                       15000/1.8 = 8333.3
                                       20000/1.8 = 11111.1
```

This is the strongest available consistency check: two fully independent unit
systems agree to all printed digits.

---

## 2. Steady-state mass balances (mass fractions)

State vector: mass fractions `X = (XA, XB, XC, XE, XG, XP)` in the reactor
(= outlet, ideal mixing). Inputs: feed mass flow rates `FA`, `FB` (kg/s) of pure
A and B, reactor temperature `T_R`. Total outlet flow `F = FA + FB` (kg/s),
reactor mass holdup `W` (kg, constant).

Mass-basis reaction rates (kg/s):

```
r1 = k1 * XA * XB * W      # mass rate of A consumed by reaction 1
r2 = k2 * XB * XC * W      # mass rate of B consumed by reaction 2
r3 = k3 * XC * XP * W      # mass rate of C consumed by reaction 3
```

The six steady-state component balances, with **mass-weighted stoichiometric
coefficients** (from the implied molecular weights MA = MB = MP = 100,
MC = ME = 200, MG = 300 of the original problem):

```
0 = FA - F*XA -     r1                      (A)
0 = FB - F*XB -     r1 -     r2             (B)
0 =    - F*XC + 2.0*r1 - 2.0*r2 -     r3    (C)
0 =    - F*XE          + 2.0*r2             (E)
0 =    - F*XG                   + 1.5*r3    (G)
0 =    - F*XP          +     r2 - 0.5*r3    (P)
```

Coefficient logic (per kg of reference reactant): reaction 1 consumes 1 kg A +
1 kg B and produces 2 kg C; reaction 2 consumes 1 kg B + 2 kg C and produces
1 kg P + 2 kg E; reaction 3 consumes 1 kg C + 0.5 kg P and produces 1.5 kg G.
Every reaction column sums to zero, so adding the six balances gives
`FA + FB - F * sum(X) = 0`, i.e. **`sum(X) = 1` is implied** (do not add it as a
seventh equation; use it as a residual check or physics-informed loss).

Sources: [S6] (exact code lines, steady state), [S7] (eqs. (26a)–(26f), dynamic
form below), [S1] (eqs. (3.6)–(3.11), same weights in the klb/h system),
[S10] (Annexe AN6, same weights in flow form).

### Dynamic form (for completeness / decay variants)

Divide each balance by W ([S7] eqs. (26a)–(26f)):

```
dXA/dt = (FA - F*XA)/W - k1*XA*XB
dXB/dt = (FB - F*XB)/W - k1*XA*XB - k2*XB*XC
dXC/dt = -F*XC/W + 2*k1*XA*XB - 2*k2*XB*XC - k3*XC*XP
dXE/dt = -F*XE/W + 2*k2*XB*XC
dXG/dt = -F*XG/W + 1.5*k3*XC*XP
dXP/dt = -F*XP/W + k2*XB*XC - 0.5*k3*XC*XP
```

Residence time `tau = W / F` (≈ 320 s at nominal flows).

---

## 3. Parameters, nominal operating point, decision ranges

Canonical parameter set (Forbes & Marlin lineage as fixed by Mendoza et al.
[S12] and implemented in [S6]; also used by [S4], [S5], [S8]):

| Quantity | Value | Units | Source |
|---|---|---|---|
| FA (fixed feed of A)    | **1.8275** | kg/s | [S6] (`Fa = 1.8275`), [S12] |
| W (reactor mass holdup) | **2105.2** (often quoted 2105) | kg | [S6] (`Mt = 2105.2`), [S12] |
| FB (decision)           | **[4.0, 7.0]** | kg/s | [S4] eq. (16) |
| T_R (decision)          | **[70, 100]** | degC | [S4] eq. (16) |
| T_R in Kelvin           | [343.15, 373.15] | K | conversion |

Operating constraints (both part of the canonical RTO problem):

```
XA <= 0.12        (residual reactant A)
XG <= 0.08        (waste G)
```

Sources: [S4] eq. (16), [S5], [S6], [S8] eqs. (46b)-(46c).

Nominal operating point — the constrained plant optimum (verified numerically
in §8; both constraints active, matching the optimum location plotted in [S4]
Fig. 6):

```
FB* = 4.3894 kg/s      T_R* = 80.50 degC  (= 353.65 K)
X*  = (XA, XB, XC, XE, XG, XP)
    = (0.12000, 0.39675, 0.02392, 0.27066, 0.08000, 0.10866)
profit* = 75.82 $/s
```

(Ignoring the two operating constraints, the in-box profit maximum is
~88.7 $/s at FB ≈ 4.20 kg/s, T_R ≈ 85.9 degC — useful as an optimizer test.)

### Known parameter variants in the literature (do NOT mix)

| Variant | FA | W | Inputs | Bounds | Used by |
|---|---|---|---|---|---|
| **Canonical (this spec)** | 1.8275 kg/s fixed | 2105.2 kg | (FB, T_R) | FB in [4,7] kg/s, T_R in [70,100] degC | [S4], [S5], [S6], [S8], [S12] |
| Forbes-Marlin 3-input | FA free in [3, 4.5] kg/s | n/a (same model) | (FA, FB, T_R) | FB in [6,11] kg/s, T_R in [80,105] degC | [S9] eq. (4.31) |
| Amrit / de Carvalho dynamic | 3.5 kg/s | 2500 kg | (FB, T_R) | FB ~ N(6,1) kg/s, T_R ~ N(100,15) degC (excitation) | [S7] Table 4 |

---

## 4. Economic (profit) objective

### Canonical coefficient set (label: "Mendoza-2016 prices", $/s)

```
maximize  phi(FB, T_R) = (1043.38*XP + 20.92*XE) * (FA + FB)
                          - 79.23*FA - 118.34*FB        [$/s]
s.t.       XA <= 0.12,   XG <= 0.08,
           FB in [4, 7] kg/s,   T_R in [70, 100] degC
```

Interpretation: product P sells at 1043.38 $/kg-equivalent, by-product E at
20.92, raw material A costs 79.23 and B costs 118.34 (all per kg of stream;
the X*F products are component mass flows in kg/s, so phi is in $/s).

Sources (4 independent statements):
- [S4] eq. (16) — exact form above, with the FB/T_R bounds.
- [S5] — same objective and constraints.
- [S6] — exact code: `obj = -(1043.38*Xp*Fr + 20.92*Xe*Fr - 79.23*Fa - 118.34*Fb)`.
- [S8] eq. (46a) — same, but prints `2092*XE`; this is a typesetting error for
  `20.92*XE` (decimal point lost), as fixed by the three sources above.

### Alternative coefficient set (label: "Forbes-Marlin prices") — for awareness only

```
phi = (1143.38*XP + 25.92*XE) * (FA + FB) - 76.23*FA - 114.34*FB   [$/s]
```

Used by the classic Forbes & Marlin design-cost papers and e.g. [S9]
eq. (4.31) (with 3 inputs and only XG <= 0.08). Both sets appear widely;
**PWLNet adopts the Mendoza-2016 set as canonical** because it is the one used
by the modern surrogate/RTO benchmark implementations [S4], [S5], [S6], [S8]
and ships with an open reference implementation [S6]. Keep the coefficient set
as a config option (`objectives.py`) so variant A can be reproduced.

There is also a return-on-investment objective for the full recycle plant
(coefficients 2207, 50, 168, 252, 2.22, 84, 60; [S10] Annexe AN6, [S11]
eq. (S46)) — not applicable to the reactor-only benchmark.

---

## 5. Surrogate task definition (PWLNet)

**Function to learn:** the steady-state solution map of §2,

```
h : (FB, T_R)  ->  (XA, XB, XC, XE, XG, XP)
```

- **Inputs (2):** `FB` in kg/s, `T_R` in degC. FA = 1.8275 kg/s and
  W = 2105.2 kg are fixed constants of the scenario.
- **Outputs (6):** predict **all six** mass fractions, not only the economic
  subset (XP, XE) and constraint subset (XA, XG). Rationale: (i) the profit and
  both constraints together already require 4 of 6 outputs; (ii) the two extra
  outputs (XB, XC) enable the physics-informed residuals below and the
  `sum(X)=1` consistency loss; (iii) all six are needed to warm-start an exact
  Pyomo model from a surrogate solution. (The black-box RTO literature [S8]
  uses the 4-output subset (XA, XG, XP, XE) when economics-only is wanted.)

**Sampling ranges for dataset generation:**

| Variable | Training/box range | Note |
|---|---|---|
| FB  | [4.0, 7.0] kg/s | canonical decision range [S4] |
| T_R | [70, 100] degC | canonical decision range [S4] |
| FB (extrapolation test, optional) | [3.5, 7.5] kg/s | off-manifold generalization check |
| T_R (extrapolation test, optional) | [65, 105] degC | idem |

Recommended design: Latin hypercube (or full grid for plots), n >= 500 train /
200 test for smooth 2-D maps; add a dense 61x61 grid for contour validation.

**Output ranges over the canonical box** (from the verified solver, §8 — use
for output scaling and bound propagation):

```
XA in [0.046, 0.164]    XB in [0.286, 0.600]    XC in [0.007, 0.038]
XE in [0.175, 0.332]    XG in [0.021, 0.192]    XP in [0.080, 0.111]
```

All outputs in (0, 1); `sum(X) = 1` exactly at every sample.

**Physics-informed residuals available (affine-friendly forms):**
- `sum_i X_i - 1 = 0` (linear in outputs — LP/MILP friendly).
- The six balances of §2 with r_i treated via predicted X (bilinear in outputs;
  use as gradient-training loss, or linearize around samples for SOSX-LP).
- Non-negativity / unit-interval bounds on each X_i (linear).

**Data-generation implementation notes:**
- Solve the six balances with a damped Newton / `scipy.optimize.fsolve` from
  the initial guess `X0 = (0.1, 0.35, 0.02, 0.25, 0.1, 0.1)`, then **continue**
  (warm-start) along the sampling grid: with cold starts ~4% of points on a
  61x61 grid fail to converge; with row-wise warm starts failures vanish.
  Reject any solution with `max|residual| > 1e-10` or any `X_i` outside [0, 1].
- Keep T in Kelvin *inside* the exponentials only; store/report T_R in degC.
- The downstream economic optimization (§4) is the canonical "downstream value"
  benchmark: embed the trained surrogate for h, maximize phi subject to
  XA <= 0.12, XG <= 0.08 and the box bounds, and compare against the verified
  optimum FB* = 4.3894, T_R* = 80.50, phi* = 75.82 $/s.

---

## 6. Reference economics quick-test values

For unit tests (values from the verified implementation, §8):

| FB (kg/s) | T_R (degC) | XA | XG | XP | phi ($/s) | feasible? |
|---|---|---|---|---|---|---|
| 4.3894 | 80.50 | 0.12000 | 0.08000 | 0.10866 | 75.82 | yes (both active) |
| 4.7894 | 89.704 | 0.08743 | 0.10748 | 0.10945 | 84.31 | no (XG > 0.08) |

---

## 7. Mapping to the original Williams-Otto plant (context only)

The original 1960 problem and the Schmid et al. Pyomo paper [S1] simulate the
full plant: CSTR + decanter (removes all G) + distillation column (recovers P
with a 10 wt% E azeotrope retained in the bottoms, `F_P = F_tP - 0.1*F_tE`) +
splitter with recycle ratio (1 - eta). Schmid et al. work in klb / h / degR
with controls `(F_fA, F_fB, T, mu, eta)` and nominal values
`(10, 20, 580, 129.5, 0.2)`; their nominal steady state is
`m* = (3.27, 7.47, 1.12, 9.81, 1.69, 0.22) klb` ([S1] eqs. (3.20), (4.8)).
PWLNet's benchmark is the reactor-only mass-fraction version (§1–§6); the
plant version is a possible later extension and shares the §1 kinetics exactly.

---

## 8. Numerical verification performed for this spec

Script: solve §2 with the §3 parameters using `scipy.optimize.fsolve`
(see `.cache/wo_verify.py`; rerun freely):

- Residuals close to < 1e-10 and `sum(X) = 1.000000000000` at all converged points.
- 61 x 61 grid over the canonical box solved with row-wise warm starts
  (138/3721 cold-start failures -> 0 with continuation).
- SLSQP on the implicit map reproduces the literature optimum (both
  constraints active): `FB* = 4.3894 kg/s, T_R* = 80.495 degC, phi* = 75.82 $/s`,
  consistent with the optimum location shown in [S4] Fig. 6.

---

## 9. Sources

| Tag | Reference | URL |
|---|---|---|
| S1 | J. Schmid, K. Teichert, M. Chioua, T. Schindler, M. Bortz, "Simulation and optimal control of the Williams-Otto process using Pyomo," arXiv:2004.07614 (2020). Reactions (3.1)-(3.3), dynamics (3.6)-(3.11), constants (3.13)-(3.14). | https://arxiv.org/abs/2004.07614 |
| S2 | T. J. Williams, R. E. Otto, "A generalized chemical processing model for the investigation of computer control," AIEE Trans. Part I 79(5), 458-473 (1960). Original problem (constants restated in [S1], [S10], [S11]). | https://doi.org/10.1109/TCE.1960.6367296 |
| S3 | J. F. Forbes, T. E. Marlin, "Design cost: a systematic approach to technology selection for model-based real-time optimization systems," Comput. Chem. Eng. 20(6-7), 717-734 (1996). Mass-fraction CSTR RTO formulation ("Forbes-Marlin prices"). | https://doi.org/10.1016/0098-1354(95)00205-7 |
| S4 | E. A. del Rio Chanona, P. Petsagkourakis, E. Bradford, J. E. A. Graciano, B. Chachuat, "Real-time optimization meets Bayesian optimization and derivative-free optimization: A tale of modifier adaptation," Comput. Chem. Eng. 147, 107249 (2021); arXiv:2009.08819. Problem (16): objective, constraints, FB/T_R bounds. | https://arxiv.org/abs/2009.08819 |
| S5 | D. Krishnamoorthy, "ECCBO: An inherently safe Bayesian optimization with embedded constraint control for real-time optimization," arXiv:2402.18415 (2024). Kinetics + objective + constraints. | https://arxiv.org/abs/2402.18415 |
| S6 | OptiMaL-PSE-Lab, "DDO-4-ChemEng" reference implementation, file `Benchmarking/02_constrained/ML4CE_WO_coursework/ML4CE_WO.py`: `Fa = 1.8275`, `Mt = 2105.2`, exact kinetics, balances, objective, constraints. | https://github.com/OptiMaL-PSE-Lab/DDO-4-ChemEng |
| S7 | M. Woelk et al., "Neural Luenberger state observer for nonautonomous nonlinear systems," arXiv:2602.24252 (2026). Dynamic mass-fraction ODEs (26a)-(26f) with identical kinetics; Amrit-variant parameters (Table 4). | https://arxiv.org/abs/2602.24252 |
| S8 | C. Lu, J. A. Paulson, "No-regret constrained Bayesian optimization of noisy and expensive hybrid models using differentiable quantile function approximations" (CUQB), arXiv:2305.03824. WO RTO problem (46a)-(46c) (note `2092` typo for `20.92`). | https://arxiv.org/abs/2305.03824 |
| S9 | "From real-time optimization techniques to an autopilot for steady-state processes," arXiv:2108.08715. Section 4.6.1: Forbes-Marlin price set (4.31), 3-input bounds, T in K via T_R + 273.15. | https://arxiv.org/abs/2108.08715 |
| S10 | "A trust-region funnel algorithm for grey-box optimisation," arXiv:2511.18998. Annexe AN6: full recycle-plant model in original units (a_i, rho = 50). | https://arxiv.org/abs/2511.18998 |
| S11 | "Machine learning powered feasible path framework with adaptive sampling for black-box optimization," arXiv:2509.21077. Supplementary S6: plant model, original-unit constants (S47), ROI objective (S46). | https://arxiv.org/abs/2509.21077 |
| S12 | D. F. Mendoza, J. E. A. Graciano, F. dos Santos Liporace, G. A. C. Le Roux, "Assessing the reliability of different real-time optimization methodologies," Can. J. Chem. Eng. 94(3), 485-497 (2016). Cited by [S4] as the source of the complete mass-balance/kinetics set; parameter values surfaced through [S6]. | https://doi.org/10.1002/cjce.22402 |
| S13 | A. Marchetti, B. Chachuat, D. Bonvin, "Modifier-adaptation methodology for real-time optimization," Ind. Eng. Chem. Res. 48(13), 6022-6033 (2009). WO CSTR as *the* MA benchmark (2 inputs); see also Processes 4(4):55 (2016) review, Table 1. | https://doi.org/10.1021/ie801352x |
| S14 | R. Amrit, J. B. Rawlings, L. T. Biegler, "Optimizing process economics online using model predictive control," Comput. Chem. Eng. 58, 334-343 (2013). Dynamic WO variant ((FB, T_R) inputs; cited by [S7]). | https://doi.org/10.1016/j.compchemeng.2013.07.015 |
