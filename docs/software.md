# Software

This project is a small stack of Python packages around [jaxley](https://jaxley.readthedocs.io/), not a single monolith. `toric_spines_sim` is the simulation/analysis repo collaborators clone; the others are installed by `uv sync` from GitHub.

## Environment

Use **[uv](https://docs.astral.sh/uv/)** (not conda) for Python. Each project has a `.venv` at the repo root. You do not activate it: prefix commands with `uv run`.

```bash
uv add numpy                          # PyPI
uv add "git+https://github.com/jmrfox/mascaf.git"
uv run python scripts/append_sink.py TS1
uv run pytest
```

`pyproject.toml` is the source of truth for dependencies. Optional extras:

| Group | Install | What it is for |
|-------|---------|----------------|
| (default) | `uv sync` | simulation, pymcfs, mascaf, tests (`dev` is included) |
| `neurosignature` | `uv sync --group neurosignature` | TS1 descriptor pipeline and example notebook |
| `docs` | `uv sync --group docs` | MkDocs preview (`mkdocs serve`) |

If you truly need conda-forge packages alongside PyPI, [Pixi](https://pixi.prefix.dev/latest/) can pull both; this repo does not require that.

**OS.** Linux or macOS, or **WSL2 Ubuntu** on Windows with the clone on the Linux filesystem (`~/...`, not `/mnt/c/...`). Cursor / VS Code can edit the same tree from Windows. Nothing needs compiling; see [Getting started](getting-started.md).

**Interactivity.** Jupyter notebooks (jupytext `.py` sources) are the lightweight lab notebook. The axon studies also ship [Dash](https://dash.plotly.com/) apps (terminal server + browser UI). [marimo](https://marimo.io/) is an option for new notebooks; nothing here depends on it.

## Packages developed for this work

| Package | Role |
|---------|------|
| **[toric_spines_sim](https://github.com/jmrfox/toric_spines_sim)** | This repo: morphology pipeline CLIs, `TSModel` / `TSSimulator`, axon studies, Dash / PDF |
| **[mascaf](https://github.com/jmrfox/mascaf)** | Mesh And Skeleton CAble model Fitting. Fits an SWC cable to a mesh + skeleton. This project pins the `release` branch. Docs: [mascaf.readthedocs.io](https://mascaf.readthedocs.io/en/latest/). Methods: [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.10.721501v1) |
| **[pymcfs](https://github.com/jmrfox/pymcfs)** | Python mean-curvature flow skeletonization (MCFS). **This is the supported skeletonizer** (`scripts/skeletonize_meshes.py`) |
| **[swctools](https://github.com/jmrfox/swctools)** | SWC I/O and morphology helpers used by mascaf and the simulator |
| **[jscip](https://github.com/jmrfox/jscip)** | Parameter banks (sampling, derived quantities, constraints). Used heavily; you do not need to extend it to run sims |
| **[neurosignature](https://github.com/jmrfox/neurosignature)** | Optional. Descriptors for multi-channel event-in / multi-channel signal-out systems. Design: [Neurosignature](neurosignature.md) |

mascaf’s GitHub `main` branch still contains extra toric-spine research scripts; `release` is the installable package this repo uses.

## Other libraries

| Tool | Role |
|------|------|
| **[jaxley](https://jaxley.readthedocs.io/)** | Differentiable cable simulator (JAX), pinned to 0.14: the loop export edits its private compartment-edge table. Runs HH studies; float64 required |
| **[pynapple](https://pynapple.org/)** | Event and time-series types (`Ts`, `Tsd`, `TsGroup`) for inputs and results |
| **[CGAL](https://www.cgal.org/)** / **[CGALLab](https://www.cgal.org/demo/6.2/CGALlab.zip)** | Computational geometry. Original MCFS implementation used here; still useful for **interactive** mesh repair and one-off skeletonization. Not required for the automated pipeline. Details: [Skeletons](skeletons.md) |

## Biophysics notes

Temperature is Kelvin; capacitance µF/cm²; leak S/cm²; axial resistivity Ω·cm; synaptic `*_gmax_uS` is the **peak** conductance of one event. Tutorials use `make_default_parameter_bank()`. ICx studies use `make_icx_parameter_bank(condition, temp_K=None, background="none")` (`make_icx_parameter_bank_invitro()` / `_invivo()` are shorthands). The bank is built in three layers.

**1. Intrinsic (slice) values** at the Sanculi recording temperature (room temperature, 296 K), from the 20-cell whole-cell table in `data/toric_spines_properties_imputed.csv`:

| Quantity | Value | Source |
|---|---|---|
| Cm | 0.9 µF/cm² | Gentet et al. 2000 (biological constant; same in vivo) |
| Whole-cell C, Rin, τ | 205 pF, 290 MΩ, ≈ 60 ms (medians) | Sanculi et al. 2020 |
| g_pas | cm / (Rin·C) ≈ 1.5e-5 S/cm² | derived |
| Ri | 100 Ω·cm | typical; poorly constrained |
| Rest | −66 mV | Sanculi (−65.7 ± 8.7) |

The sink stands in for the rest of the neuron. With `sink_lumped = 1` (ICx default) its cm / g_pas are scaled so its totals equal `cell_C_pF` and 1/`cell_Rin_MOhm`; the geometric 10 µm sink alone has ~1/18 of the real membrane.

**2. Temperature layer.** Values are scaled from `temp_ref_K` (296 K) to `temp_K` with Q10s stored in the bank: leak 1.98 and Ri 0.80 (Trevelyan & Jack 2002), Cm unchanged, AMPA kinetics 1.4 (Postlethwaite et al. 2007), NMDA kinetics 2.0 (Korinek et al. 2010), GABA 2.0 (unverified). In vivo uses 313 K (barn owl ≈ 40 °C), which alone lowers whole-cell Rin ~3.2× and τ to ≈ 19 ms. The custom NMODL mechanisms do not read temperature themselves; `temp_K` acts only through these factory-computed values.

**3. Background layer (in vivo).** Ongoing synaptic input lowers rest-of-cell Rin; there is no owl measurement (Peña & Konishi 2002 report rest −67.6 ± 9.3 mV and threshold, not Rin), so `background` is swept: `none` 0, `moderate` 0.5, `high` 0.8 fractional Rin drop, applied as a tonic shunt on the lumped sink.

Peña & Konishi recorded under ketamine (an NMDA antagonist) with sharp electrodes; Sanculi used whole-cell in juvenile (P35–40) slices. Keep both in mind when comparing PSPs.

## Solvers and loops

MASCAF stores each mesh loop as a tree plus `# CYCLE_BREAK` (and `# MULTI_NECK`) pairs. `TSModel.build()` merges every pair back into one node of the cable graph (`simulation.reference.LoopCableModel`), so loops are real loops, and both solvers integrate that same graph implicitly:

| Solver | When (`solver="auto"`) | What it supports | Speed (TS3, 1 s simulated, CPU) |
|---|---|---|---|
| `reference` (scipy) | passive models | passive membrane, all synapse types | 2.9 s |
| `jaxley` 0.14 | HH painted (`hh_tags`, `hh_scale`) | + Hodgkin-Huxley; differentiable | ~20 s (+ ~10 s compile, once per model) |

They agree to ~1e-5 mV on passive models (`tests/test_reference.py`, `tests/test_jaxley_backend.py`), and both reproduce the frozen Arbor baselines on loop-free spines, including HH spikes to the time step (`tests/test_golden.py`). Cutting the loops instead (a tree) overestimates TS1 EPSPs by up to ~2x.

Why not Arbor: Arbor needs a tree, and its gap junctions compute the current from the peer voltage at the start of each step. For a junction strong enough to stand in for cable, that acts like an extra capacitance of `g·dt` on each side (50 pF at 1 µS and dt = 0.025 ms, against ~0.2 pF of spine membrane), so the response collapses and depends on dt. Evidence and a frozen copy of the Arbor model: `scripts/arbor_evidence/` (`uv sync --group arbor-evidence`).

jaxley notes: the backend enables float64 (float32 invents a conductance to ground with the tiny branch-point loads), simulates voltages relative to rest, and refuses jaxley's tree-only `jaxley.dhs` solver for looped cells. jax's sparse solver has no `vmap` rule, so `run_batch` on looped cells runs sequentially inside one compiled call.

Convergence (`scripts/diagnostics/convergence.py`): peak EPSPs on TS1 are within 0.6% of the finest run at dx = 0.25 µm, dt = 0.025 ms.
