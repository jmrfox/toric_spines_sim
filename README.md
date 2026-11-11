# toric_spines_sim

Simulations of barn-owl **toric spines** (Sanculi et al., 2020), with the spines' cable loops closed exactly. These are large, hole-containing (genus > 0) spines on space-specific neurons in ICx. The goal is to characterize how an isolated spine integrates synaptic input, including electrical leak into the rest of the cell.

Work in this repo is: turn an EM mesh into a cable (SWC) model, attach a cylindrical **sink**, drive synapses from mapped axons, and analyze sink / compartment voltages.

High-level path: mesh → skeleton → SWC (`# CYCLE_BREAK` / `# MULTI_NECK`) → sink → `TSModel` (cable graph, loops closed) → event generators → `TSSimulator.run()` → traces, k-matrix, Dash / PDF.

**Documentation** (project overview, reconstruction, skeletons, simulations, API): [https://jmrfox.github.io/toric_spines_sim/](https://jmrfox.github.io/toric_spines_sim/)

## Install

You need **git**, **Python 3.12+**, and **[uv](https://docs.astral.sh/uv/)**. On Windows, use WSL2. Details: [Getting started](https://jmrfox.github.io/toric_spines_sim/getting-started/).

```bash
git clone https://github.com/jmrfox/toric_spines_sim.git
cd toric_spines_sim
uv sync
uv run pytest
```

`uv sync` installs jaxley and the GitHub packages `swctools`, `jscip`, `pymcfs`, and `mascaf`. The TS1 neurosignature pipeline is optional: `uv sync --group neurosignature`.

Run from the repository root. Tests: `uv run pytest`. Full axon PDF study: `uv run python -m simulations.axons TS1`. Loop attenuation for every spine: `uv run python -m simulations.loop_attenuation`.

Simulations use the scipy reference solver for passive models and jaxley 0.14 when Hodgkin-Huxley channels are on; both solve the cable graph with every `# CYCLE_BREAK` loop closed. The project used Arbor until November 2026; Arbor's gap junctions cannot reproduce a cable joint, and the evidence and a frozen Arbor model are on the `arbor-evidence` branch, and the last Arbor version of the package is on `arbor-legacy`.
