# toric_spines_sim

Arbor simulations of barn-owl **toric spines** (Sanculi et al., 2020). These are large, hole-containing (genus > 0) spines on space-specific neurons in ICx. The goal is to characterize how an isolated spine integrates synaptic input, including electrical leak into the rest of the cell.

Work in this repo is: turn an EM mesh into a cable (SWC) model, attach a cylindrical **sink**, drive synapses from mapped axons, and analyze sink / compartment voltages.

High-level path: mesh → skeleton → SWC (`# CYCLE_BREAK` / `# MULTI_NECK`) → sink → `TSModel` / `TSRecipe` → event generators → `TSSimulator.run()` → traces, k-matrix, Dash / PDF.

Start with [Getting started](getting-started.md), then the [morphology pipeline](morphology.md) and [simulations](simulations.md). The [package tour](package.md) and [API reference](api/index.md) describe the Python library.
