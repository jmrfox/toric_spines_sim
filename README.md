# toric_spines_sim

Arbor simulations of barn-owl **toric spines** (Sanculi et al., 2020). These are large, hole-containing (genus > 0) spines on space-specific neurons in ICx. The goal is to characterize how an isolated spine integrates synaptic input, including electrical leak into the rest of the cell.

Work in this repo is: turn an EM mesh into a cable (SWC) model, attach a cylindrical **sink**, drive synapses from mapped axons, and analyze sink / compartment voltages.

High-level path: mesh → skeleton → SWC (`# CYCLE_BREAK` / `# MULTI_NECK`) → sink → `TSModel` / `TSRecipe` → event generators → `TSSimulator.run()` → traces, k-matrix, Dash / PDF.

**Documentation:** [https://jmrfox.github.io/toric_spines_sim/](https://jmrfox.github.io/toric_spines_sim/)

## Install

You need **git**, **Python 3.12+**, and **[uv](https://docs.astral.sh/uv/)**. On Windows, use WSL2. Details (catalogue build, macOS toolchain): [Getting started](https://jmrfox.github.io/toric_spines_sim/getting-started/).

```bash
git clone https://github.com/jmrfox/toric_spines_sim.git
cd toric_spines_sim
uv sync
uv run bash scripts/make_custom_catalogue.sh
```

`uv sync` installs Arbor and the GitHub packages `swctools`, `jscip`, `pymcfs`, and `mascaf`. The TS1 neurosignature pipeline is optional: `uv sync --group neurosignature`.

Run from the repository root. Tests: `uv run pytest`. Canonical experiment: `uv run python -m simulations.ts1.axons`.
