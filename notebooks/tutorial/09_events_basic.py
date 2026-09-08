# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 09 — Event generators (basic)
#
# `TSSimulator` consumes a pynapple `TsGroup`: one event stream per synapse,
# timestamps in ms, stream index `i` mapped to synapse `syn_i` in the synpts
# file. Generators produce that `TsGroup`.
#
# Two generator classes:
#
# - `DeterministicEventGenerator` — integrate a rate curve (flat → periodic)
# - `StochasticEventGenerator` — Poisson (or thinned Poisson for a varying rate)
#
# This notebook uses **independent** axons and `FlatRateCurve` only. Shared
# routing, other rate curves, and axon remapping are notebook 10.

# %%
from toric_spines_sim.events import (
    DeterministicEventGenerator,
    FlatRateCurve,
    StochasticEventGenerator,
)
from toric_spines_sim.paths import NOTEBOOKS_DIR
from toric_spines_sim.viz import RasterPlotter

T_MS = 100.0
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

# %% [markdown]
# ## Independent axons
#
# A list of rate curves, one per axon. `n_synapses_per_axon=[1, 1, 1]` means
# three axons with one synapse each — three output channels.

# %%
rates_hz = [50.0, 100.0, 200.0]
labels = ["A", "B", "C"]
curves = [FlatRateCurve(r) for r in rates_hz]

periodic = DeterministicEventGenerator(
    rate_curves=curves,
    n_synapses_per_axon=[1] * len(rates_hz),
    T_ms=T_MS,
    delay_ms=10.0,
    labels=labels,
).generate()

poisson = StochasticEventGenerator(
    rate_curves=curves,
    n_synapses_per_axon=[1] * len(rates_hz),
    T_ms=T_MS,
    delay_ms=10.0,
    seed=0,
    labels=labels,
).generate()

print("periodic counts:", {k: len(periodic[k]) for k in periodic})
print("poisson counts: ", {k: len(poisson[k]) for k in poisson})

plotter = RasterPlotter(title="Independent periodic", xlim=(0.0, T_MS))
plotter.add_streams(periodic)
plotter.show()

plotter = RasterPlotter(title="Independent Poisson", xlim=(0.0, T_MS))
plotter.add_streams(poisson)
plotter.show()

# %% [markdown]
# ## Write / read event files
#
# Generators can dump labeled timestamp lines and reload them. Writes go under
# `_artifacts/` (not tracked by git), not `data/events/`.

# %%
tutorial_output_dir.mkdir(parents=True, exist_ok=True)
periodic_gen = DeterministicEventGenerator(
    rate_curves=curves,
    n_synapses_per_axon=[1] * len(rates_hz),
    T_ms=T_MS,
    labels=labels,
)
periodic_path = tutorial_output_dir / "demo_periodic.txt"
periodic_gen.generate_to_file(periodic_path)
loaded = DeterministicEventGenerator.load_from_file(periodic_path)
print("wrote", periodic_path)
print("reloaded counts:", {k: len(loaded[k]) for k in loaded})

# %% [markdown]
# `TSRecipe` / `TSSimulator` ignore channel labels and map by **TsGroup index**
# to `syn_0`, `syn_1`, …. Keep generator channel order aligned with the synpts
# file, or remap (notebook 10).
