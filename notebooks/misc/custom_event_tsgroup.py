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
# # Custom `TsGroup` events
#
# A complete but short TS1 run whose inputs are a hand-built pynapple
# `TsGroup`, not a rate generator. Use this to check that custom spike times
# reach the synapses.
#
# `TSSimulator` maps stream **index** `i` → `syn_i` in synpts row order.
# Channel labels are ignored. `TSRecipe` reads pynapple times in **seconds**
# and converts to milliseconds (`t_s * 1000`). The number of streams must
# equal the number of synapses (silent sites are empty lists).
#
# Needs the NMODL catalogue:
#
# ```bash
# uv run bash scripts/make_custom_catalogue.sh
# ```

# %%
import pynapple as nap

from toric_spines_sim.geometry import sink_endpoint_location_from_swc_file
from toric_spines_sim.model import check_catalogue
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import TSSimulator, make_default_parameter_bank
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter

check_catalogue()

swc_path = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_path = get_pointset_path("TS1_synpts.txt", units="microns")
n_synapses = len(load_xyz_points(synpts_path))
sink_xyz = sink_endpoint_location_from_swc_file(swc_path)
print("SWC:   ", swc_path)
print("synpts:", synpts_path, f"({n_synapses} sites)")
print("sink:  ", sink_xyz)

# %% [markdown]
# ## Parameters and custom times
#
# Short trial, coarse discretization, sink-only recording. Two synapses fire
# at known times (ms); the rest are silent.

# %%
parameter_bank = make_default_parameter_bank()
parameter_bank["T_ms"].value = 80.0
parameter_bank["delay_ms"].value = 0.0
parameter_bank["discretization_um"].value = 1.0
parameter_bank["dt_sim_ms"].value = 0.05
parameter_bank["dt_record_ms"].value = 0.1
parameter_bank["hh_scale"].value = 0.0
parameter_bank["ampa_gmax_uS"].value = 0.1
parameters = parameter_bank.sample()
parameters["hh_tags"] = []

t_ms = float(parameters["T_ms"])
times_ms = {
    0: [10.0, 40.0],
    1: [20.0],
}

events = nap.TsGroup(
    {
        i: nap.Ts(t=times_ms.get(i, []), time_units="ms")
        for i in range(n_synapses)
    },
    time_support=nap.IntervalSet(start=[0], end=[t_ms], time_units="ms"),
)
print(f"TsGroup streams: {len(events)}  (need {n_synapses})")
print("keys:", list(events.keys())[:8], "...")

# %% [markdown]
# ## What the recipe will send to Arbor
#
# Pynapple stores `Ts.index` in seconds. The recipe does `index * 1000` for
# milliseconds. Those two views should agree with the lists above.

# %%
print(f"{'i':>4}  {'label':8}  as_units(ms)          index*1000 (recipe)")
for i in range(n_synapses):
    ts = events[i]
    as_ms = ts.as_units("ms").index.values.tolist()
    recipe_ms = (ts.index.values * 1000.0).tolist()
    if as_ms or recipe_ms:
        print(f"{i:4d}  syn_{i:<5d}  {as_ms}  {recipe_ms}")
        if as_ms != recipe_ms:
            print("    mismatch: as_units(ms) vs recipe conversion")

sim = TSSimulator(
    swc_path,
    synpts_path,
    events,
    parameters,
    record_points={"sink": sink_xyz},
)
synapses = sim.build_synapses()
recipe = sim.build_recipe()
print("n synapses:", len(synapses))
print("n event generators:", len(recipe._evgens))
print("recipe times (ms) for active labels:")
for label, t_list in recipe._events_ms.items():
    if t_list:
        print(f"  {label}: {t_list}")

# %% [markdown]
# ## Run
#
# Sink voltage should show EPSPs near 10, 20, and 40 ms.

# %%
results = sim.run()
traces = results.voltage_traces
print("voltage:", traces.shape, "columns", list(traces.columns))

plotter = TimeSeriesPlotter(
    title="TS1 sink voltage (custom TsGroup)",
    xlim=(0.0, t_ms),
    figsize=(12, 4),
)
plotter.add_time_series(traces["sink"], label="sink")
plotter.show()

raster = RasterPlotter(
    title="Custom input events",
    xlim=(0.0, t_ms),
    figsize=(12, 6),
)
raster.add_streams(results.input_events, linelength=0.8)
raster.show()

# %% [markdown]
# ## Two constructions that fail or fire at the wrong time
#
# 1. Fewer streams than synapses → `ValueError` in `TSRecipe`.
# 2. Times passed without `time_units="ms"` are treated as **seconds**, then
#    multiplied by 1000, so `t=[10.0]` becomes 10 s → 10000 ms (past `T_ms`).

# %%
short_group = nap.TsGroup(
    {
        0: nap.Ts(t=[10.0], time_units="ms"),
        1: nap.Ts(t=[20.0], time_units="ms"),
    },
    time_support=nap.IntervalSet(start=[0], end=[t_ms], time_units="ms"),
)
print("short TsGroup length:", len(short_group), "vs n_synapses", n_synapses)
try:
    TSSimulator(
        swc_path,
        synpts_path,
        short_group,
        parameters,
        record_points={"sink": sink_xyz},
    ).build_recipe()
except ValueError as exc:
    print("expected error (too few streams):", exc)

wrong_units = nap.Ts(t=[10.0, 40.0])  # default: seconds
print(
    "t=[10, 40] with default units → recipe ms:",
    (wrong_units.index.values * 1000.0).tolist(),
    "(wanted 10 and 40 ms)",
)
print(
    "t=[10, 40] with time_units='ms' → recipe ms:",
    (nap.Ts(t=[10.0, 40.0], time_units="ms").index.values * 1000.0).tolist(),
)
