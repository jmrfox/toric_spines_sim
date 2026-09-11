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
# # 15 — `TSSimulator`
#
# To run a cable-cell simulation, pass a micron sink SWC, synpts, a
# synpts-order pynapple `TsGroup`, and a sampled `ParameterSet` into
# `TSSimulator`, then call `run()`. Synapses are always AMPA from the points
# file. Existing spike times can go straight into a `TsGroup`
# (`events_basic`); generators are optional. For receptors other than AMPA,
# use `TSModel` (`synapses_advanced`, `tsmodel_and_tsrecipe`) rather than
# `TSSimulator`.
#
# Needs the NMODL catalogue (`mechanisms`). Result analysis is `simulation_results`.
#
# Full axon PDF / Dash studies:
#
# ```bash
# uv run python -m simulations.ts1.axons
# uv run python -m simulations.ts1.axons_dash
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
n_synapses = len(load_xyz_points(synpts_path))  # 25
sink_xyz = sink_endpoint_location_from_swc_file(swc_path)
print("SWC:   ", swc_path)
print("synpts:", synpts_path, f"({n_synapses} sites)")
print("sink:  ", sink_xyz)

# %% [markdown]
# ## Parameter set and events
#
# Short trial, coarse discretization, sink-only recording. Hand-built
# `TsGroup` in synpts order: two active synapses, the rest silent. Channel
# `i` maps to `syn_i`. Dict events keyed by place tag are a `TSRecipe` path
# (`tsmodel_and_tsrecipe`), not `TSSimulator`. Axon-order times need a remap
# first (`events_advanced`).

# %%
parameter_bank = make_default_parameter_bank()
parameter_bank["T_ms"].value = 150.0
parameter_bank["delay_ms"].value = 20.0
parameter_bank["discretization_um"].value = 1.0
parameter_bank["dt_sim_ms"].value = 0.05
parameter_bank["dt_record_ms"].value = 0.1
parameter_bank["cm_uF_per_cm2"].value = 2.0
parameter_bank["rL_ohm_cm"].value = 150.0
parameter_bank["ampa_gmax_uS"].value = 0.1
parameter_bank["hh_scale"].value = 0.0
parameters = parameter_bank.sample()
parameters["hh_tags"] = []

t_ms = float(parameters["T_ms"])
delay_ms = float(parameters["delay_ms"])
events = nap.TsGroup(
    {
        i: nap.Ts(
            t=([delay_ms, delay_ms + 20.0] if i in (0, 1) else []),
            time_units="ms",
        )
        for i in range(n_synapses)
    },
    time_support=nap.IntervalSet(start=[0], end=[t_ms], time_units="ms"),
)
print(f"event streams: {len(events)} (indices 0 and 1 active)")

sim = TSSimulator(
    swc_path,
    synpts_path,
    events,
    parameters,
    record_points={"sink": sink_xyz},
)
print("TSSimulator:", type(sim).__name__)
print("  swc_filepath:   ", sim.swc_filepath)
print("  synpts_filepath:", sim.synpts_filepath)
print("  n event streams:", len(events))
print("  record_points:  ", sim.record_points_spec)

# %% [markdown]
# ## Build steps (computed on first use, then reused)
#
# Construction order: synapses and GJs, then record points, then cable cell,
# then recipe. Each step is cached on the instance. `"all"` would probe every
# segment center (slow, large `TsdFrame`).

# %%
synapses = sim.build_synapses()
print(f"synapses: {len(synapses)}")
print("first three:", list(synapses)[:3])
print(f"gap junctions: {len(sim.gap_junctions)}")

record_points = sim.build_record_points()
print("record points:", record_points)

cell = sim.build_cell()
print("cell:", type(cell).__name__)
print("n segments:", len(list(sim.segment_tree.segments)))

recipe = sim.build_recipe()
print("recipe:", type(recipe).__name__)

# %% [markdown]
# ## `run()`
#
# Reuses the cached cell and recipe. Event channel `i` → synapse `syn_i`.
# Voltage is a pynapple `TsdFrame` (index in seconds;
# `TimeSeriesPlotter` converts columns to ms).

# %%
results = sim.run()
traces = results.voltage_traces
print("voltage TsdFrame:", traces.shape, "columns", list(traces.columns))

plotter = TimeSeriesPlotter(
    title="TS1 sink voltage",
    xlim=(0.0, float(parameters["T_ms"])),
    figsize=(12, 4),
)
plotter.add_time_series(traces["sink"], label="sink")
plotter.show()

raster = RasterPlotter(
    title="Input events (all streams)",
    xlim=(0.0, float(parameters["T_ms"])),
    figsize=(12, 6),
)
raster.add_streams(results.input_events, linelength=0.8)
raster.show()
