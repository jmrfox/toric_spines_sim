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
# # 10 — Event generators (advanced)
#
# Notebook 09 covered independent axons and `FlatRateCurve`. This notebook
# covers other rate curves, shared-source routing, axon-map remapping, and
# `random_axon_events`.
#
# When an axon fires, every synapse on that axon gets the same timestamp.

# %%
import numpy as np

from toric_spines_sim.events import (
    DeterministicEventGenerator,
    FlatRateCurve,
    LinearRateCurve,
    SineRateCurve,
    StepRateCurve,
    StochasticEventGenerator,
)
from toric_spines_sim.paths import get_data_path, get_pointset_path
from toric_spines_sim.simulation import (
    load_axon_events_from_file,
    random_axon_events,
    remap_axon_channel_events_to_synapses,
)
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import RasterPlotter

T_MS = 100.0

# %% [markdown]
# ## Rate curves
#
# `rate_at(t_ms, T_ms)` is Hz. Stochastic generators use `max_rate()` for
# Poisson thinning. `LinearRateCurve` **requires** `T_ms` (the ramp window).

# %%
flat = FlatRateCurve(80.0)
linear = LinearRateCurve(10.0, 80.0)
step = StepRateCurve(rates_hz=[20.0, 80.0, 40.0], step_duration_ms=30.0)
sine = SineRateCurve(peak_rate_hz=200.0, freq_hz=40.0, baseline=0.0)

times = np.linspace(0.0, T_MS, 6)
print("t_ms   flat  linear  step   sine")
for t in times:
    print(
        f"{t:5.1f}  {flat.rate_at(t):5.1f}  "
        f"{linear.rate_at(t, T_ms=T_MS):6.1f}  "
        f"{step.rate_at(t):5.1f}  {sine.rate_at(t):5.1f}"
    )

# %% [markdown]
# ## Shared source
#
# Pass **one** `RateCurve` instead of a list. Routing:
#
# - `"broadcast"` — every master event hits every axon
# - `"roundrobin"` — master events cycle across axons
#
# `n_synapses_per_axon=[2, 1]` → axon 0 has two synapses (identical times),
# axon 1 has one. Output channel order is axon 0's synapses, then axon 1's.

# %%
master = FlatRateCurve(80.0)
n_synapses_per_axon = [2, 1]

broadcast = DeterministicEventGenerator(
    rate_curves=master,
    n_synapses_per_axon=n_synapses_per_axon,
    T_ms=T_MS,
    routing_mode="broadcast",
).generate()

roundrobin = DeterministicEventGenerator(
    rate_curves=master,
    n_synapses_per_axon=n_synapses_per_axon,
    T_ms=T_MS,
    routing_mode="roundrobin",
).generate()

plotter = RasterPlotter(title="Shared periodic, broadcast", xlim=(0.0, T_MS))
plotter.add_streams(broadcast)
plotter.show()

plotter = RasterPlotter(title="Shared periodic, round-robin", xlim=(0.0, T_MS))
plotter.add_streams(roundrobin)
plotter.show()

# %% [markdown]
# ## Time-varying rates
#
# `SineRateCurve` with a stochastic generator uses thinning: events cluster at
# the peaks. Linear / step work the same way.

# %%
sine_events = StochasticEventGenerator(
    rate_curves=sine,
    n_synapses_per_axon=[2, 2],
    T_ms=T_MS,
    seed=1,
).generate()

plotter = RasterPlotter(
    title="Shared Poisson, sine-modulated (40 Hz)",
    xlim=(0.0, T_MS),
)
plotter.add_streams(sine_events)
plotter.show()

ramp_events = StochasticEventGenerator(
    rate_curves=linear,
    n_synapses_per_axon=[1, 1],
    T_ms=T_MS,
    seed=2,
).generate()
plotter = RasterPlotter(title="Independent Poisson, linear ramp", xlim=(0.0, T_MS))
plotter.add_streams(ramp_events)
plotter.show()

# %% [markdown]
# ## TS1 axon map → synapse order
#
# Load `ts1_axons.txt`, generate in **axon order**, then
# `remap_axon_channel_events_to_synapses` so stream `i` matches synpts row
# `i`. `TSSimulator` expects that synapse order.
#
# Notebook 13 uses a simpler independent-per-synapse pattern so the
# construction example stays short. The axon PDF study
# (`python -m simulations.ts1.axons`) uses this remap path.

# %%
synpts_path = get_pointset_path("TS1_synpts.txt", units="microns")
axon_path = get_data_path("ts_axons", "ts1_axons.txt")
n_synapses = len(load_xyz_points(synpts_path))

n_axons = sum(1 for line in axon_path.read_text().splitlines() if line.strip())
axon_rates = [0.0] * n_axons
axon_rates[0] = axon_rates[3] = 40.0

curves, n_per_axon, axon_synapses = load_axon_events_from_file(
    axon_path, axon_rates_hz=axon_rates
)
print("synapses per axon:", n_per_axon)
print("active axons:", [i for i, r in enumerate(axon_rates) if r > 0])

axon_order_events = StochasticEventGenerator(
    rate_curves=curves,
    n_synapses_per_axon=n_per_axon,
    T_ms=T_MS,
    delay_ms=10.0,
    seed=0,
).generate()

synapse_events = remap_axon_channel_events_to_synapses(
    axon_order_events,
    axon_synapses,
    n_synapses=n_synapses,
)
print(f"TsGroup size after remap: {len(synapse_events)} (expect {n_synapses})")

plotter = RasterPlotter(
    title="TS1 axons 0 and 3 → 25 synapse streams",
    xlim=(0.0, T_MS),
    figsize=(12, 6),
)
plotter.add_streams(synapse_events, linelength=0.8)
plotter.show()

# %% [markdown]
# ## Random axon partition
#
# `random_axon_events` draws a synapse→axon assignment and one `SineRateCurve`
# per axon (shared modulation frequency, random phase).

# %%
rand_curves, rand_n = random_axon_events(
    n_synapses=8,
    n_axons=3,
    mod_freq_hz=20.0,
    peak_rate_hz=50.0,
    seed=0,
)
print("synapses per axon:", rand_n)
print("phases (rad):", [c.to_sine_v1_params()["phase_rad"] for c in rand_curves])
rand_events = StochasticEventGenerator(
    rate_curves=rand_curves,
    n_synapses_per_axon=rand_n,
    T_ms=T_MS,
    seed=0,
).generate()
plotter = RasterPlotter(title="random_axon_events (8 syn, 3 axons)", xlim=(0.0, T_MS))
plotter.add_streams(rand_events)
plotter.show()
