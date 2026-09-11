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
# # 11 — Event generators (basic)
#
# To drive a cell from spike times, pass a pynapple `TsGroup`: one event
# stream per synapse, timestamps in **milliseconds**. `TSSimulator` maps
# stream **index** `i` to synapse `syn_i` in the synpts file. Channel labels
# are ignored.
#
# Python lists or a timestamp file are enough; generators below
# **synthesize** the same `TsGroup`. Axon-order data: `events_advanced`.
# Wiring into a run: `tssimulator`.

# %%
import pynapple as nap

from toric_spines_sim.events import (
    DeterministicEventGenerator,
    EventGenerator,
    FlatRateCurve,
    StochasticEventGenerator,
)
from toric_spines_sim.paths import NOTEBOOKS_DIR, get_data_path
from toric_spines_sim.viz import RasterPlotter

T_MS = 100.0
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

# %% [markdown]
# ## Hand-built `TsGroup`
#
# Three synapses, three Python lists of times (ms). Keys must be `0, 1, 2`
# in synpts order — not names.

# %%
times_ms = {
    0: [10.0, 40.0, 70.0],
    1: [20.0, 50.0],
    2: [15.0, 45.0, 75.0, 95.0],
}
hand_built = nap.TsGroup(
    {i: nap.Ts(t=t, time_units="ms") for i, t in times_ms.items()},
    time_support=nap.IntervalSet(start=[0], end=[T_MS], time_units="ms"),
)
print("type:", type(hand_built).__name__, "n streams:", len(hand_built))
for idx, ts in hand_built.items():
    print(f"  index {idx}: {ts.as_units('ms').index.values.tolist()} ms")

plotter = RasterPlotter(title="Hand-built lists → TsGroup", xlim=(0.0, T_MS))
plotter.add_streams(hand_built)
plotter.show()

# %% [markdown]
# ## Load a timestamp file
#
# Format: one line per stream, `LABEL t1 t2 …` with times in ms. Empty
# streams are a label alone. **Line order is the TsGroup index.** Labels are
# stored as metadata but `TSRecipe` / `TSSimulator` ignore them.
#
# `data/events/demo_periodic.txt` is three streams labeled `A`, `B`, `C`.
# Those names do **not** select synapses; stream 0 still maps to `syn_0`.

# %%
event_path = get_data_path("events", "demo_periodic.txt")
print("file:", event_path)
print("first line:", event_path.read_text().splitlines()[0][:60], "...")

loaded = EventGenerator.load_from_file(str(event_path))
print("n streams:", len(loaded))
print("labels (metadata only):")
print(loaded.get_info("label"))
for idx, ts in loaded.items():
    print(f"  index {idx}: {len(ts)} spikes")

# %% [markdown]
# ## Synthesize streams with generators
#
# `DeterministicEventGenerator` integrates a rate curve (flat → periodic).
# `StochasticEventGenerator` draws Poisson (or thinned Poisson) times.
# `n_synapses_per_axon=[1, 1, 1]` means three independent axons / three
# output channels — still index order, not label order.

# %%
rates_hz = [50.0, 100.0, 200.0]
curves = [FlatRateCurve(r) for r in rates_hz]

periodic = DeterministicEventGenerator(
    rate_curves=curves,
    n_synapses_per_axon=[1] * len(rates_hz),
    T_ms=T_MS,
    delay_ms=10.0,
).generate()

poisson = StochasticEventGenerator(
    rate_curves=curves,
    n_synapses_per_axon=[1] * len(rates_hz),
    T_ms=T_MS,
    delay_ms=10.0,
    seed=0,
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
# ## Write / read
#
# Generators can dump the same labeled timestamp format. Writes go under
# `_artifacts/` (not tracked by git).

# %%
tutorial_output_dir.mkdir(parents=True, exist_ok=True)
periodic_path = tutorial_output_dir / "demo_periodic.txt"
DeterministicEventGenerator(
    rate_curves=curves,
    n_synapses_per_axon=[1] * len(rates_hz),
    T_ms=T_MS,
).generate_to_file(str(periodic_path))
reloaded = EventGenerator.load_from_file(str(periodic_path))
print("wrote", periodic_path)
print("reloaded counts:", {k: len(reloaded[k]) for k in reloaded})

# %% [markdown]
# Any of these `TsGroup`s is valid input to `TSSimulator(events=...)`.
# Channel `i` → `syn_i`. Times grouped by axon instead of synpts row need a
# remap (`events_advanced`).
