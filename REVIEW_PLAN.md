# Code review and fix plan (2026-11-07)

## Status (branch `review-fixes`)

| Item | Status |
|---|---|
| 1.1 Gap junction placement | **Fixed.** `arbor_locations_for_swc_nodes` resolves `(distal (segment k))` in Arbor; test checks XYZ on all 9 spines |
| 1.2 Loop closure | **Done.** `simulation/reference.py` (implicit, loops merged). Validated vs Arbor on trees (<1% peak). `TSSimulator(solver="auto")` uses it for passive runs; `weight or 1.0` bug fixed |
| 1.3 cm / g_pas | **Fixed.** cm 0.9, g_pas = cm/(Rin·C) ≈ 1.5e-5 at 296 K |
| 1.4 Sink | **Fixed.** `sink_lumped=1` gives the sink whole-cell C and Rin; test recovers Rin (5%) and τ (8%) in both solvers |
| 1.5 Synapse normalization | **Fixed.** nmdasyn / gababsyn / effexcsyn peak-normalized; nmdasyn Mg block now Jahr–Stevens with `mg`; catalogue rebuilt |
| 3.3 Parameter scheme | **Done.** `make_icx_parameter_bank(condition, temp_K, background)` |
| Convergence | **Done.** dx 0.25 µm, dt 0.025 ms within 0.6% (`scripts/diagnostics/convergence.py`) |
| Open | temperature-aware active sink; owl synaptic Q10s and gmax; fluctuating (not tonic) background; re-run axon studies |

Measured loop effect (in-vitro bank before cm fix, 5 synapses): TS1 tree 17.2 mV vs loops 9.6 mV; TS4 9.2 vs 7.5; TS3 3.4 vs 3.0; TS2 / TS67 within 3%; Arbor gap junctions at w = 1 were 2–3x low. So the loops matter, contrary to the λ-based guess in 1.2 below.

**Correction to 3.2:** the literature gives lower synaptic Q10s than "2–3": AMPA decay ≈ 1.3–1.5 ([Postlethwaite et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC1156678)), synaptic NMDA ≈ 1.7–2.2 ([Korinek et al. 2010](https://www.pubmed.gov/19883737/)). The bank uses 1.4 and 2.0.

---

Scope: the biophysical model (`toric_spines_sim/model`, `simulation/parameters.py`, NMODL in `mechanisms/my_catalogue`). Geometry, viz and reports were read only where they feed the model. Every claim marked **[measured]** comes from a script in `scripts/diagnostics/` (run with `uv run python scripts/diagnostics/<name>.py ...`).

The test suite passes (276 passed, 1 skipped). None of the bugs below are caught by it.

---

## 1. Critical bugs (results produced so far are affected)

### 1.1 Gap junctions are placed at the wrong points on the cell **[measured]**

`geometry/swc.py::arbor_locations_for_swc_nodes` rebuilds Arbor's branch numbering with its own BFS (`segment_tree_branches`). Arbor numbers branches differently, so the computed `(branch, pos)` points to the wrong place.

- Over all SWC nodes: 55 of 60 TS1 branches are numbered differently, and positional errors reach 25 µm (`check_gj_placement.py`).
- For the cycle-break nodes themselves, the errors are 1–6 µm on a spine only a few µm across. The loops are closed between the wrong points.
- TS1 and TS21: one junction end lands **in the sink** (tag 5). This ties a spine node to the 10 µm sink cylinder.
- TS3: the MULTI_NECK junctions, which should sit on sink node 77, land on spine branches instead.
- The existing test (`test_colocated_cycle_break_maps_to_distinct_locations`) only checks that the two ends are *different* from each other, not that they are *correct*.

**Fix:** drop the hand-rolled mapping. Place each junction with Arbor's own locset, `(distal (segment k))`, where `k` is the sample's index among non-root SWC samples. For a root node, use `(root)`. This gives 0.0 µm error on all nine spines (`check_distal_segment_mapping.py`). Replace the test with one that checks the XYZ of each placed location (`place_pwlin.at(loc)`) against the SWC sample.

**Impact on TS1 [measured, `ts1_gj_sweep.py`]:** 5 synapses fire together on the in-vitro bank.

| Gap junctions | Peak spine depolarization |
|---|---|
| removed | 17.2 mV |
| as currently built (`gj_weight = 1`) | 1.8 mV |

The current junctions suppress the EPSP about 10×, mostly through the spine-to-sink short circuit. Treat every published or saved TS1/TS21/TS3 result as suspect.

### 1.2 Arbor gap junctions cannot faithfully close a low-resistance cable loop **[measured]**

Arbor's `gj` is `i = g·(v − v_peer)`, with `g = 1 µS` scaled by the connection weight. Its conductance goes on the local CV's matrix diagonal only, and `v_peer` is the peer voltage at the *start* of the step. That makes the coupling implicit locally but **explicit in the peer**. When `g·dt/C_cv ≫ 1`, the two sides just swap voltages each step instead of equalizing. Signal transmission then falls as `1/g`.

`gj_cable_closure_test.py` compares a continuous 100 µm cable (r = 0.3 µm, dx = 0.1 µm) with the same cable cut in half and rejoined by a junction. Error is the change in far-end peak.

| weight (µS) | g·dt/C (dt = 0.025) | far-end error | ringing |
|---|---|---|---|
| 0.0001 | 1.3 | −24% (resistive bottleneck, 84 mV jump across the junction) | none |
| 0.001 | 13 | −6% (24 mV jump) | mild |
| 0.015 (Arbor example value) | 200 | −28% | strong |
| **1 (repo default)** | 1.3e4 | **−96%** | strong |
| 100 | 1.3e6 | −99.96% | strong |

No weight works. Small weights act as a series resistor; large weights decouple the two sides numerically. Shrinking `dt` helps only slowly, because the cable's own axial conductance gives `g_ax·dt/C ≈ 4e4`. Arbor's docs and examples ([interconnectivity](https://docs.arbor-sim.org/en/latest/concepts/interconnectivity.html), [`gj.mod`](https://github.com/arbor-sim/arbor/blob/v0.12.2/mechanisms/default/gj.mod), [`gap_junctions.py`](https://github.com/arbor-sim/arbor/blob/v0.12.2/python/example/gap_junctions.py)) only use junctions as weak electrical synapses between separate cells (15 nS). That is the regime the scheme was built for.

The TS1 run shows the same thing: `gj_weight = 0.001` produces a 15 mV artifact that oscillates step to step.

**Plan:**
1. Write a small **reference solver** that supports loops: an implicit-Euler compartmental model on the SWC graph with reconnect edges merged, passive membrane plus conductance synapses, using `scipy.sparse`. A spine has about 100 nodes, so this takes milliseconds per run. Validate it against Arbor on loop-free morphologies (TS48, TS76, cylinder) to within 1%.
2. Use it to measure how much loops actually matter (tree vs. loop) for each spine. A rough estimate: λ is about 1 mm for spine branches, against a spine size of a few µm. Loops will likely change EPSPs by only a few percent, but that must be measured, not assumed.
3. Decide based on the result:
   - Small effect: run Arbor with **no** junctions (a tree) and document the error bound.
   - Large effect: run loop morphologies in the reference solver, or in NEURON with `LinearMechanism`, which closes loops implicitly.
   - Either way, stop using `gj_weight` as a free knob.
4. Fix `recipe.py:291`, `float(gj.weight or 1.0)`, which silently turns a weight of 0 into 1. That is why `gj_weight = 0` produced runs identical to `gj_weight = 1`.

### 1.3 Membrane capacitance is about 9× too high

Every ICx bank uses `cm = 8.559 µF/cm²` ("± 50.475"). An SD six times the mean signals a derived quantity, not a measurement. The specific capacitance of neuronal membrane is about 0.9 µF/cm² and is treated as a biological constant ([Gentet, Stuart & Clements 2000](https://pmc.ncbi.nlm.nih.gov/articles/PMC1300935)). It does not change between in vivo and in vitro, and barely changes with temperature (Q10 0.96, [Trevelyan & Jack 2002](https://pmc.ncbi.nlm.nih.gov/articles/PMC2290153)). Sanculi measured *whole-cell* capacitance in nF. Dividing that by a soma-only area estimate would give a number like 8.6.

The leak `g_pas = 1.44e-4 S/cm²` was then picked to reproduce τ ≈ 60 ms, so it is inflated by the same factor. Because τ = cm/g is unchanged, the error is easy to miss. But it:
- makes every compartment about 9× more capacitive, so a given synaptic charge produces a much smaller and slower local EPSP;
- shortens λ by about 3× (λ ∝ √(1/g)).

From `data/toric_spines_properties_imputed.csv` (n = 20): median τ = 61 ms, Rin = 290 MΩ, C = 0.205 nF. With cm = 0.9 µF/cm², that gives membrane area ≈ 22,800 µm² and **g_pas ≈ 1.5e-5 S/cm²**, which predicts τ = 60 ms, matching the measurement.

There are related comment errors. `ts1_neurosignature.py` says "Arbor's default capacitance = 0.01 uF/cm²", but Arbor's default is 0.01 F/m², which equals 1 µF/cm². `ts2_integration.py` sets `cm = 10` with a "273 K = in vivo" comment.

**Fix:** set `cm = 0.9` in the ICx banks and refit `g_pas` from the CSV (in vitro: about 1.5e-5). Delete the 8.559 provenance text.

### 1.4 The sink does not represent the rest of the cell

The sink is a 10 µm × 20 µm cylinder, about 1,260 µm² of membrane. A real SSN has about 22,800 µm² of membrane (from C/cm), so the sink is roughly 18× too small. With corrected cm/g_pas it would present Rin ≈ 5 GΩ, against a measured 0.29 GΩ. How strongly the spine "leaks into the rest of the cell" is the whole scientific question, so the sink has to be calibrated.

**Fix:** paint the sink region with lumped membrane properties so that its total capacitance and conductance equal the whole-cell `C` and `1/Rin`. Arbor supports per-region `membrane_capacitance` and `pas` density. Alternatively, lengthen the sink to the equivalent area (it stays isopotential, since λ is several mm). Add sink presets for the median cell and for the low/high quartiles, and a test that measures sink Rin and τ with a current step.

### 1.5 Dual-exponential synapses are not peak-normalized

`nmdasyn`, `gababsyn` and the NMDA part of `effexcsyn` compute `gmax·(d − r)/(τd − τr)`. That normalizes the *integral*, not the peak, and has units of 1/ms. With τr = 5 and τd = 50 ms, the peak conductance is only **0.0155·gmax**. So `nmda_gmax_uS = 0.001` gives a 15 pS peak, about 65× weaker than an AMPA synapse with the same nominal "gmax". In `effexcsyn`, `nmda_ratio` therefore mixes a peak-normalized AMPA term with an integral-normalized NMDA term, so 0.5 does not mean half.

**Fix:** use the standard peak factor, `1/(exp(−tp/τd) − exp(−tp/τr))` with `tp = τrτd/(τd−τr)·ln(τd/τr)`, computed in `INITIAL` (as Exp2Syn does). Rebuild the catalogue. Add a test that checks peak g equals gmax for one event.

---

## 2. Other correctness / consistency issues

- **`temp_K` has no effect on the model as built.** None of the custom mechanisms (ampasyn, nmdasyn, hhnotemp, …) use temperature. Reversal potentials are fixed values, not Nernst. `pas` ignores temperature. Switching the in vitro and in vivo banks therefore changes only Vrest, the leak and the synapse values, not kinetics. See section 3.
- `hhnotemp` uses squid kinetics at their native 6.3 °C, with Arbor's default Na/K reversals. It cannot represent owl spikes (measured width 0.85 ms at room temperature). Either label it as a generic "active sink" switch, or replace it with a temperature-scaled model fitted to the CSV spike properties (threshold, width, peak).
- Ion settings use Arbor defaults, not Sanculi's solutions (aCSF K⁺ 1.3 mM, Ca²⁺ 2.5, Mg²⁺ 1.3; pipette K-methanesulfonate 135 mM). That matters only for active channels. For passive runs, document it.
- NMDA Mg block: `nmdasyn` uses `0.33·exp(−0.06v)` (Mg fixed) and `effexcsyn` uses Jahr–Stevens with `mg`. Pick one form and expose `mg_mM` in both, since ACSF has 1.3 mM.
- The default bank has `temp_K = 280` (Arbor's 6.3 °C default) and `gabaa_tau_ms = 10`, but the ICx banks use 2 ms. A GABA_A decay of 2 ms is fast. Check it against owl IC data.
- The three bank factories duplicate about 180 lines each, which is how they drifted. Replace them with one base bank plus a small dict of overrides per condition (see 3.3).
- `discretization_um = 0.1` everywhere. Add a convergence check: compare EPSP peaks at dx = 0.1, 0.25 and 0.5 µm (and dt = 0.02 vs 0.005) to justify the cost.

---

## 3. In vivo vs in vitro: what actually differs, and how to model it

### 3.1 The two data sources

| | Sanculi et al. 2020 (in vitro) | Peña & Konishi 2002 (in vivo) |
|---|---|---|
| Prep | Slices, juvenile owls P35–40 | Anesthetized adult owls (ketamine + diazepam) |
| Recording | Whole-cell patch (K-methanesulfonate) | Sharp electrode (2 M K-acetate) |
| Temperature | **Room temperature** (~22–24 °C, about 296 K) | Body temperature (~40–40.5 °C cloacal, about 313 K) |
| Rest | −65.7 ± 8.7 mV (CSV) | −67.6 ± 9.3 mV (n = 75) |
| Spike threshold | −35.7 mV | −53.5 ± 10.4 mV |
| Rin, τ, C | 290 MΩ, 61 ms, 0.2 nF (medians) | **not reported** |
| Background activity | essentially none | spontaneous firing 3.1 Hz; ongoing synaptic input |

Sources: [Sanculi et al. 2020](https://www.eneuro.org/content/7/1/ENEURO.0197-19.2019), [Peña & Konishi 2002](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758218/).

The current in vivo bank says "Input resistance is lower in vivo than in slice (~3×)" and cites Peña & Konishi. **That paper reports no input resistance**, so the 3× factor has no owl source.

### 3.2 What physically changes between the two conditions

1. **Temperature (about +17 °C).** This is the largest well-defined difference. Typical Q10 values:
   - Leak conductance: Q10 ≈ 2 (Trevelyan & Jack: total conductance Q10 1.98). Leak rises about 1.98^1.7 ≈ **3.2×**. This coincidentally matches the repo's "×3", but for a different reason.
   - Axial resistivity: Q10 ≈ 0.80, so Ri falls about 0.8^1.7 ≈ 0.68×, e.g. 100 → 68 Ω·cm.
   - Cm: unchanged.
   - Synaptic kinetics: Q10 roughly 2–3 for AMPA/NMDA rise and decay, so decays are 3–6× faster at 40 °C. The exact AMPA/NMDA Q10s still need a literature search, preferably avian or auditory-brainstem data.
   - Channel gating, if HH is used: Q10 about 3.
2. **Background synaptic bombardment.** In cortex, in vivo activity lowers Rin and τ substantially, the "high-conductance state" ([Destexhe, Rudolph & Paré 2003](https://link.springer.com/article/10.1038/nrn1198)). This is contested: some in vivo TTX experiments show the reduction is partly intrinsic ([eNeuro 2018](https://www.eneuro.org/content/eneuro/5/2/ENEURO.0447-17.2018.full.pdf)). For owl ICx there is no direct number. The right way to model it is a **conductance**, not a bigger leak: fluctuating excitatory and inhibitory point conductances on the sink, as in Destexhe's point-conductance model. `mechanisms/other/bgampa.mod` and `bggabaa.mod` already exist for this. This changes both Rin and the effective resting potential, and it adds noise, which a scaled leak cannot.
3. **Recording artifacts.** Sharp electrodes add a shunt that depolarizes the cell and lowers Rin. Whole-cell recording dialyzes the cell. Neither should be built into a model of the real cell, but they explain part of the threshold and rest differences.
4. **Anesthesia.** Ketamine is an NMDA receptor antagonist, so in vivo NMDA contributions under ketamine are reduced. That matters when matching Peña & Konishi PSPs.
5. **Age.** Sanculi used juveniles (P35–40); the in vivo data come from adults. Expect some maturation of Rin and kinetics.

### 3.3 Proposed parameter scheme

Replace the three factories with a single `make_icx_parameter_bank(condition="in_vitro" | "in_vivo", background="none" | "low" | "high")`:

- **Intrinsic, from the slice data:** `cm = 0.9`; `g_pas` and `Ri` at the reference temperature `T_ref = 296 K`; sink lumped to the median whole-cell `C` and `Rin`.
- **Temperature layer (explicit Q10s, all in one table):**
  - `g_pas(T) = g_pas_ref·Q10_leak^((T−T_ref)/10)`
  - `Ri(T) = Ri_ref·Q10_Ri^((T−T_ref)/10)`
  - Synaptic τ values divided by `Q10_syn^((T−T_ref)/10)`
  - Implement this either as derived parameters (`DerivedScalarParameter`) or by adding `celsius`-based Q10 code to the NMODL.
- **Background layer (in vivo only):** background conductance that reduces sink Rin by a chosen fraction. Sweep it (e.g. 0%, 50%, 80%), because there is no owl number to pin it to.
- **Rest:** −66 mV for both conditions. The two datasets agree within their SDs.
- **Reporting:** every result states its condition, temperature and background level. Treat in vivo results as a band over the uncertain parameters, not a single run.

### 3.4 Literature still to pin down

- AMPA and NMDA kinetics and their Q10 in avian auditory neurons (e.g. nucleus magnocellularis/laminaris, or owl ICx/ICc slices).
- Any owl IC in vivo Rin or τ measurement, e.g. whole-cell in vivo or other Peña/Konishi/Fischer papers.
- Barn owl brain temperature, as opposed to cloacal temperature.
- ICx unitary EPSC amplitudes, to calibrate `gmax`. Sanculi recorded EPSCs with NBQX/bicuculline, so their voltage-clamp section may give it.

---

## 4. Order of work

1. **Fix gap junction placement** (1.1) and the `weight or 1.0` bug, with an XYZ-accuracy test. Small change; it immediately alters TS1/TS21/TS3 results.
2. **Build the loop-capable reference solver** and validate it against Arbor on loop-free spines (1.2). Measure the loop effect per spine, then decide between junctions off and reference/NEURON solver.
3. **Fix the synapse peak normalization** (1.5) and rebuild the catalogue.
4. **Correct cm/g_pas and calibrate the sink** to Sanculi's whole-cell numbers (1.3, 1.4), with a current-step Rin/τ test.
5. **Restructure the parameter banks** into intrinsic + temperature + background layers (3.3), with provenance in one table in `docs/software.md`.
6. Run a discretization and time-step convergence check; then re-run the axon studies and report how the conclusions changed.
7. Lower priority: a temperature-aware active sink, Mg-block consistency, ion concentrations matched to the solutions.
