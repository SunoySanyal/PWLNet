# Hydroformylation reaction-kinetics fix (notebook 04)

**Symptom.** The commit that added `notebooks/04_hydroformylation_nmpc_pwl_control.ipynb`
flagged that "something is wrong with the reaction kinetics." The simulator ran
and conserved the C12/C13 carbon pool, but the catalyst model carried an
unphysical hand-tuned `catalyst_activity_factor = 0.05`.

**Reference.** B. Hentschel, G. Kiedorf, M. Gerlach, C. Hamel, A.
Seidel-Morgenstern, H. Freund, K. Sundmacher, *Model-Based Identification and
Experimental Validation of the Optimal Reaction Route for the Hydroformylation
of 1-Dodecene*, Ind. Eng. Chem. Res. 2015, 54, 1755-1765
(`papers/hentschel2015.pdf`).

## What was correct

Verified line-by-line against the paper:

- Rate laws `r1..r6` (eqs 1-6, Figure 1 network) — match exactly.
- Reference-temperature Arrhenius `k_j(T) = k_{0,j} exp(-E_{A,j}/R (1/T - 1/T_ref))`,
  `T_ref = 378.15 K` (eq 7). The large `k_{0,j}` (up to 4.9e16) are correct: the
  rate laws use concentrations in **mol/mL** and a **per-gram-catalyst** basis
  (Table 1 units), so the constants are large by construction.
- Equilibrium constants `K_p,j = exp(-ΔG_j/RT)`, `ΔG_j = a0 + a1 T + a2 T²`
  (eqs 8-9, Table 2) — match.
- Gas solubility `C_i* = p_i/H_i`, `H_i = H0_i exp(-E_{A,H,i}/RT)` (eqs 11-12,
  Table 3); gas-liquid transfer `j_i^GL = (kLa)_i (C_i* - C_liq,i)` (eq 15);
  ideal-gas pressure balance (eqs 14, 17) — all dimensionally consistent.

## The bug — catalyst pre-equilibrium (eq 10)

The paper's active-catalyst concentration is

```
C_cat = C_cat,tot / (1 + K_cat,1 · C_CO^{K_cat,3} + K_cat,2 · C_CO^{K_cat,x}/C_H2)
```

with Table 1 values `K_cat,1 = 3.041e4`, `K_cat,2 = 0`, `K_cat,3 = 0.644`. The
paper text states `K_cat,3` was *added as an exponent to account for the unknown
reaction order of CO*. With `K_cat,2 = 0` this reduces to

```
C_cat = C_cat,tot / (1 + 3.041e4 · C_CO^0.644).
```

The notebook instead computed the denominator **linearly in C_CO** —
`1 + K_cat,1·C_CO` — dropping the `K_cat,3 = 0.644` exponent (the exponent was
mistakenly attached only to the disabled `K_cat,2` term). At the nominal
dissolved-CO concentration `C_CO ≈ 1.04e-4 mol/mL` this makes the denominator
`4.15` instead of the correct `83.6` — a ~20× error in catalyst concentration,
hence a ~20× error in **every** reaction rate. That error was then cancelled by
the introduced `catalyst_activity_factor = 0.05` (≈ 4.15/83.6), so the nominal
magnitude looked right while the **CO reaction order was wrong (1.0 vs 0.644)**.

Because the catalyst term is a common factor on all six rates, the nominal
operating point is nearly unchanged (active catalyst `1.172e-7` vs `1.164e-7`
g/mL; final n-tridecanal `0.393` vs `0.391` mol/L), but the model's response to
CO partial pressure and gas dosing — exactly the manipulated variables of the
NMPC benchmark — was governed by the wrong power law.

## The fix

In `notebooks/04_..._control.ipynb` cell 5:

```python
# Hentschel eq 10 (K_cat,2 = 0):
cat_denom = 1.0 + p.kcat_1 * max(c_co, 0.0) ** p.kcat_3   # was: 1 + kcat_1 * c_co
...
active_catalyst_g_per_ml = catalyst_inventory_g_per_ml * p.gamma / cat_denom
# (removed the unphysical catalyst_activity_factor = 0.05)
```

The `catalyst_activity_factor` field was deleted. The model now matches eq 10
exactly. The C12/C13 carbon-skeleton pool remains conserved to ~1e-15.

## Second issue — catalyst mass normalization (M_cat)

After the eq-10 fix the *rate* was still far too slow: under the random control
profiles, 1-dodecene was only ~50% consumed at 20 min and ~93% at 80 min,
whereas Hentschel Figure 2c (semibatch, 105 °C) shows 1-dodecene **fully
consumed within ~20 min**. The rollout plots therefore did not match the paper.

Root cause: the eq-13 liquid balance uses the active-catalyst MASS concentration
`C_cat * M_cat`, and the Table-1 rate constants are normalized **per gram of
catalyst**, not per mole of Rh. The notebook used the Rh **atomic** mass
`M_cat = 103 g/mol`, which made the catalyst too dilute and the reaction
correspondingly too slow.

Fix: `active_catalyst_molar_mass_g_mol = 258.03`, the molar mass on which the
prior articles' rate constants are normalized (established by hand from those
papers; 258.03 g/mol is the mass of the `Rh(acac)(CO)2` catalyst charged). This
is the value in `src/pwlnet/processes/hydroformylation/model.py` and it is
correct — do not "fix" it back to the Rh atomic mass.

### Superseded: the 770 g/mol intermediate

An earlier revision of this document recommended `M_cat = 770.0`, reasoning that
the rate constants were normalized per gram of the *active complex*
`HRh(CO)(biphephos)` (`Rh 103 + biphephos ~639 + CO 28 + H`). That reasoning was
dropped once the normalization basis was traced back to the prior articles.
Recorded here only so the number is recognizable if it turns up in old notebooks
or commits.

Note that `M_cat` is a common multiplier on all six rates, so it does not change
selectivity directly — but it does set where the batch sits on the conversion
curve within the fixed horizon, and therefore how strongly the objective
responds to temperature. Measured on the true ODE with total pressure held at
20 bar over 70 min (terminal `nC13al` vs. constant `T` swept over the admissible
368.15–388.15 K):

| `M_cat` | best T | terminal nC13al | spread over the T band |
|---|---|---|---|
| 103    | 388.15 K (upper bound) | 0.5474 | 37.7 % — rate-limited, monotone increasing |
| 258.03 | 380.65 K (interior)    | 0.5593 | 11.8 % — interior optimum, ~1 % plateau above 376 K |
| 770    | 368.15 K (lower bound) | 0.5700 |  1.8 % — saturated, monotone decreasing |

So with the correct 258.03 the objective has a genuine interior optimum in `T`,
but a shallow one. Any optimizer started on the plateau (e.g. at the 378.15 K
reference temperature) sees `dJ/dT ≈ +8.7e-4 mol/L/K`, i.e. 0.16 %/K, and will
not move off it without multi-start or a continuation in `T`.

Residual caveat (not a kinetics bug): the notebook is a *fixed-feed* semibatch
(the manipulated `u` is bounded), so under low feed the reaction becomes
gas-limited and total pressure sags to ~5-15 bar, whereas the paper's
*pressure-controlled* semibatch doses syngas to hold 20 bar. The 1-dodecene /
product *concentration* dynamics now match; the absolute conversion of the
hydroformylation step remains feed-limited by design.
