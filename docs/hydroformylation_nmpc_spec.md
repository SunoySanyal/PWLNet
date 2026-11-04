# Hydroformylation NMPC Surrogate Benchmark

**Status:** notebook benchmark with Hentschel-aligned kinetics and gas balance.
**Notebook:** `notebooks/04_hydroformylation_nmpc_pwl_control.ipynb`.
**Primary model source:** Hentschel et al., "Model-Based Identification and
Experimental Validation of the Optimal Reaction Route for the
Hydroformylation of 1-Dodecene" (`papers/hentschel2015.pdf`).
**Benchmark parameter source:** Aydin, Bonvin, and Sundmacher, "Dynamic optimization of
constrained semi-batch processes using Pontryagin's Minimum Principle -- an
effective quasi-Newton approach" (`papers/aydin_quasi_newton_approach.pdf`),
Problem 3 and Table 6.
**NMPC/backoff context:** Aydin, Bonvin, and Sundmacher, "NMPC using
Pontryagin's Minimum Principle -- Application to a two-phase semi-batch
hydroformylation reactor under uncertainty" (`papers/aydin_2479085_ams.pdf`).

This document describes the third benchmark problem added to PWLNet after the
Williams-Otto notebooks. The notebook now uses Hentschel's source equations for
the reaction rates, active-catalyst expression, gas-liquid transfer, and
gas-phase pressure balances. The Aydin papers provide the 80 minute benchmark
framing, uncertainty ranges, NMPC timing assumptions, and pressure-backoff
context used for rollout stress tests.

---

## 1. Literature Problem

The process is a two-phase semi-batch stirred-tank reactor for hydroformylation
of 1-dodecene using syngas (`H2 + CO`). The desired product is n-tridecanal
(`nC13al`). In the quasi-Newton paper's Problem 3, the fixed final batch time is:

```text
tf = 80 min
```

The manipulated variables are:

```text
T(t)  reactor temperature [K]
u(t)  syngas feedrate [mol/min]
```

The nominal dynamic optimization problem maximizes final desired-product
concentration:

```text
maximize nC13al(tf)
```

subject to liquid-phase species balances, gas-phase partial-pressure balances,
gas-liquid transfer, nonlinear reaction kinetics, pressure limits, and input
bounds.

---

## 2. Operating Constraints

Problem 3 states:

```text
1 bar <= p_total(t) <= 20 bar
0 <= u(t)
368.15 K <= T(t) <= 388.15 K
```

For the exploratory surrogate-control solve, the notebook uses a finite
feedrate box:

```text
0 mol/min <= u(t) <= 0.012 mol/min
```

This keeps the syngas feedrate on the pressure scale implied by the Hentschel
gas-phase balance with `V_gas = 900 mL` after applying the catalyst multiplier
as a liquid-phase mass concentration. The previous scaffolded `10 mol/min`
box belonged to the reduced pressure closure and is not compatible with the
source-model pressure dynamics.

The notebook also keeps the later NMPC paper's operating context for future
closed-loop extensions:

```text
|T(t) - T(t_k)| / (t - t_k) <= 0.35 K / 15 s
b_s(t) = 1.3   if t <= 30 min
b_s(t) = 0.7   if 30 < t <= 55 min
b_s(t) = 0.0   if t > 55 min
```

The simulated plant now uses the NMPC timing values from the later paper:

```text
plant sampling time = 15 s = 0.25 min
concentration measurement period = 30 s = 0.5 min
measurement delay = 5 s = 0.0833 min
```

The backoff curve is shown in the open-loop plot as an NMPC reference, while the
nominal Problem-3 path limit remains `20 bar`.

---

## 3. Table-6 Nominal Parameters

### Reaction Kinetics

The Arrhenius law is:

```text
k_j(T) = k0_j * exp(-(EA_j/R) * (1/T - 1/Tref))
```

The notebook uses `Tref = 378.15 K` for the scaled Arrhenius expression.

| Reaction | `EA` [kJ/mol] | `k0` | Unit | `K1` [mL/mol] | `K2` [mL/mol] | `K3` [mL/mol] |
|---|---:|---:|---|---:|---:|---:|
| `r1` | 113.08 | `4.904e16` | `mL^3/(g min mol^2)` | 574876 | 3020413 | 11732838 |
| `r2` | 136.89 | `4.878e6` | `mL/(g min)` | 38632 | 226214 | |
| `r3` | 76.11 | `5.411e8` | `mL^2/(g min mol)` | 2661.2 | 7100 | 1280 |
| `r4` | 102.26 | `2.958e4` | `mL^2/(g min mol)` | | | |
| `r5` | 120.84 | `7.619e10` | `mL^3/(g min mol^2)` | | | |
| `r6` | 113.08 | `3.951e11` | `mL^3/(g min mol^2)` | | | |

The active-catalyst denominator row is encoded from Hentschel equation 10:

```text
Ccat = Ccat,tot / (1 + Kcat1*c_CO + Kcat2/(c_CO^Kcat3*c_H2))
Kcat1 = 3.041e4, Kcat2 = 0, Kcat3 = 0.644
```

Because `Kcat2 = 0` for this fitted parameter set, the exponent term is inactive
numerically, but the notebook keeps the full structure. Hentschel reports a
catalyst/olefin molar ratio of `1/10000`; the notebook converts this to an
explicit catalyst mass concentration `Ccat,tot*Mcat` in `g/mL` for the liquid
phase balance. Using total catalyst mass instead would over-speed the reaction
rates by approximately the liquid volume factor. The notebook also exposes an
effective `catalyst_activity_factor = 0.05` so the active catalyst fraction can
be calibrated to the observer-style concentration trajectories without changing
the Hentschel rate equations.

### Equilibrium And Solubility

The equilibrium constants use:

```text
Kp,j = exp(-DeltaG_j/(R*T))
DeltaG_j = a0_j + a1_j*T + a2_j*T^2
```

The notebook therefore computes:

```text
Kp,j = exp(-(a0_j + a1_j*T + a2_j*T^2)/(R*T))
```

with:

| Term | `a0` [kJ/mol] | `a1` [kJ/(mol K)] | `a2` [kJ/(mol K^2)] |
|---|---:|---:|---:|
| `DeltaG2` | -11.0034 | 0 | 0 |
| `DeltaG3` | -126.275 | 0.1266 | `6.803e-6` |

Gas solubility and transfer:

| Component | `H0` [bar mL/mol] | `EA,H` [kJ/mol] | `kLa` [min^-1] |
|---|---:|---:|---:|
| `H2` | 66400 | -3.06 | 9.57 |
| `CO` | 73900 | -0.84 | 7.08 |

The solubility relation is:

```text
H_i = H0_i * exp(-EA,H_i/(R*T))
c_i* = p_i / H_i
j_i = (kLa)_i * (c_i* - c_liq,i)
```

The gas-phase pressure balance follows the syngas-fed semibatch model:

```text
dp_i/dt = (R*T/V_gas) * (0.5*u - V_liq*j_i),  i in {H2, CO}
```

where `u` is the total syngas feedrate and each gas receives half of that feed.

### Reactor And Initial Conditions

From the Problem 3 text:

```text
V_liq = 900 mL
V_gas = 900 mL
n_nC12en(0) = 0.85 mol
nC12en(0) = 0.85 mol / 0.9 L
p_H2(0) = 10 bar
p_CO(0) = 10 bar
```

The dissolved gas concentrations are initialized at Henry-law equilibrium with
the initial 20 bar syngas gas phase at `T0 = 378.15 K`:

```text
c_H2(0) = 1000*p_H2(0)/H_H2(T0) = 0.05691 mol/L
c_CO(0) = 1000*p_CO(0)/H_CO(T0) = 0.10359 mol/L
```

All other liquid products and side-products start at zero.

---

## 4. Simulated State And Control Variables

The notebook tracks nine states:

| Name | Meaning |
|---|---|
| `c_H2` | dissolved hydrogen concentration |
| `c_CO` | dissolved carbon monoxide concentration |
| `nC12en` | 1-dodecene concentration |
| `iC12en` | iso-dodecene concentration |
| `nC13al` | desired n-tridecanal concentration |
| `nC12an` | n-dodecane side-product concentration |
| `iC13al` | iso-aldehyde side-product concentration |
| `p_H2` | hydrogen partial pressure |
| `p_CO` | carbon monoxide partial pressure |

The compound naming follows Hentschel's notation:

```text
nC12an, nC12en, iC12en, nC13al, iC13al
```

Internally, the ODE state vector keeps the reaction-equation order
`nC12en, iC12en, nC13al, nC12an, iC13al` after the dissolved gases. The plots use
an explicit display order so labels do not depend on raw array position.

The control vector is:

```text
[T, u]
```

where `T` is the reactor-temperature setpoint and `u` is the syngas feedrate.

---

## 5. Notebook Simulator

The liquid chemistry uses the Table-6 rate laws:

```text
r1 = k1*nC12en*c_H2*c_CO
     / (1 + K1,1*nC12en + K1,2*nC13al + K1,3*c_H2)

r2 = k2*(nC12en - iC12en/Kp,2)
     / (1 + K2,1*nC12en + K2,2*iC12en)

r3 = k3*(nC12en*c_H2 - nC12an/Kp,3)
     / (1 + K3,1*nC12en + K3,2*nC12an + K3,3*c_H2)

r4 = k4*iC12en*c_H2
r5 = k5*iC12en*c_H2*c_CO
r6 = k6*nC12en*c_H2*c_CO
```

Concentrations are stored and plotted in `mol/L`, while the rate-law
calculations convert to `mol/mL` because Table 6 uses `mL` units. The true
plant simulator reports states at the paper's `15 s` sampling time, but
internally uses SciPy's stiff BDF integrator because the Table-6 reversible
kinetics are stiff. The simulator enforces nonnegative states after integration
but does not clip states to the surrogate-training upper bounds; those upper
bounds are used only for embedding and plotting. To keep the current PWLNet
benchmark runnable with the phase-two single-block embeddings, the open-source
baseline control and one-step transition surrogate remain parameterized on a
`5 min` move grid. Concentration measurements are modeled every `30 s` and
plotted at their available time, `5 s` after sampling.

The notebook also includes a Gurobi-only variant that trains a separate
`15 s` one-step transition surrogate and solves a short chained PWL-tanh big-M
MILP horizon at the paper's decision interval. That controller applies only the
first `15 s` move, advances the true plant, updates a delayed-measurement state
estimate, and then re-solves. The default closed-loop window is short so the
notebook remains executable; set `MILP_15S_CLOSED_LOOP_MIN = T_FINAL` in the
notebook to run the full `80 min` batch at `320` decision steps.

The pressure dynamics are no longer a reduced closure; they use the source
gas-phase material balance. The remaining scaffolded part is that the simulator
is still notebook-local rather than a reusable `pwlnet.processes` module.

---

## 6. PWLNet Benchmark Construction

The current PWLNet formulations embed static feed-forward networks. The dynamic
control problem is represented through a learned one-step transition map:

```text
[tau, x_k, T_k, u_k] -> x_{k+1}
```

where:

```text
tau = t_k / tf
x_k = 9-dimensional process state
T_k = reactor-temperature setpoint
u_k = syngas feedrate
```

The notebook generates trajectories under perturbed nominal control profiles
and sampled uncertain parameters, then trains a tanh network for this transition
map. If TensorFlow/Keras is available, it uses `KerasTrainer`. If not, it falls
back to a deterministic single-hidden-layer tanh random-feature model exported
as the same `NetworkDefinition` object.

The trained transition model is embedded with the PWL-tanh big-M block. For the
open-source baseline, a monolithic direct-shooting model that chains one block
per time step and solves once is **intractable for HiGHS**: the per-neuron
segment-selection binaries couple across the state-transition equalities, and
HiGHS finds no feasible point for a 3-step chain in 20 s or for the full 8-step
horizon in 90 s. The notebook therefore keeps a **receding-horizon
(closed-loop NMPC)** baseline that solves one block per `5 min` step:

```text
for k = 0 .. N_STEPS-1:
    fix x_k (the measured state) and tau_k as block inputs   # only T_k, u_k free
    surrogate(tau_k, x_k, T_k, u_k) = x_{k+1}                # single PWL-tanh block
    maximize predicted nC13al(x_{k+1})
        s.t. soft pressure-backoff slack at step k+1
             soft terminal-product cap at the final step
             hard |T_k - T_{k-1}| <= rate limit
    apply (T_k, u_k) to the TRUE simulator -> next measured x_{k+1}
```

Each per-step subproblem is the single-block case, which solves to optimality in
well under a second. The objective is implemented as minimization of
`-nC13al` (plus slack penalties) to match Pyomo's standard minimization form.
The embedding network's input bounds are tightened to the data-reachable range
so the big-M constants and interval propagation stay tight.

With Gurobi, the notebook adds a more faithful paper-timing MILP benchmark:

```text
decision interval = 15 s
transition surrogate = x_k -> x_{k+1} over 15 s
default prediction horizon = 8 steps = 2 min
temperature move limit = 0.35 K per 15 s decision
measurement period = 30 s
measurement delay = 5 s
```

This is still receding-horizon control rather than one giant full-batch MILP.
At `15 s` resolution, an `80 min` simultaneous model would contain `320`
transition blocks, so the notebook uses the standard NMPC pattern: solve a
short Gurobi MILP, apply the first move, and repeat.

---

## 7. What This Tests

This benchmark stresses a different use case than the Williams-Otto steady-state
RTO notebooks:

- repeated surrogate blocks linked over a prediction horizon;
- dynamic path constraints rather than a single operating-point constraint set;
- pressure limits and later-NMPC backoff context under parameter uncertainty;
- formulation error from PWL-tanh approximation versus smooth-network rollout;
- surrogate/model error when the optimized policy is replayed on the simulator.

It is meant to answer whether the current phase-two PWL-tanh big-M machinery can
be useful in a model-predictive-control style workflow before the project has a
full custom dynamic-process module.

---

## 8. Known Limitations

- The simulator is notebook-local rather than a first-class
  `pwlnet.processes.hydroformylation` module.
- The catalyst inventory is inferred from Hentschel's reported
  catalyst/olefin molar ratio and an explicit active-catalyst molar-mass
  assumption.
- The controller uses a one-step prediction horizon (receding-horizon NMPC).
  A genuine multi-step prediction horizon is not yet practical with the
  open-source solver: chaining PWL-tanh blocks makes even feasibility
  intractable beyond ~2 steps, and would need stronger bound tightening, an
  SOS2-capable solver, or a warm-started MIP.
- The open-source baseline control optimization is still parameterized on a
  `5 min` move grid; the plant simulation and concentration-measurement
  visualization use the paper's `15 s` sampling, `30 s` measurement period,
  and `5 s` measurement delay.
- The Gurobi-only MILP path does use `15 s` decisions, but the notebook default
  runs a short closed-loop window. Increase `MILP_15S_CLOSED_LOOP_MIN` to
  `T_FINAL` when a full 320-decision Gurobi run is desired.
- The current benchmark uses a one-step transition surrogate; it does not yet
  implement PMP, indirect adjoining, or the DSM comparison from the paper.
- Grid-point pressure constraints do not guarantee continuous-time feasibility
  between grid points.

---

## 9. Recommended Next Steps

1. Move the Table-6 simulator into
   `src/pwlnet/processes/hydroformylation/`.
2. Move the receding-horizon NMPC builder out of the notebook into a reusable
   process helper.
3. Add regression tests for one-step simulation, pressure limit evaluation,
   transition-network dimensions, and single-step NMPC model construction.
4. Compare ReLU big-M, exact tanh OMLT, and PWL-tanh big-M on the same
   transition-surrogate control task.
5. Extend the receding-horizon loop (now implemented with a one-step prediction
   horizon) to a longer prediction window once a tighter multi-step formulation
   is available.
