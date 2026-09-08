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
# # 13 — `TSSimulator`
#
# This notebook runs a cable-cell simulation with `TSSimulator`. You pass a
# micron sink SWC and synpts from `data/`, a pynapple event group, and a
# `ParameterSet`; `run()` returns `SimulationResults`. Synapses are always
# AMPA from the points file. For other receptors, build `TSModel` yourself
# (notebook 12).
#
# Needs the NMODL catalogue (notebook 11). Result analysis is notebook 14.
#
# Full axon PDF / Dash studies:
#
# ```bash
# uv run python -m simulations.ts1.axons
# uv run python -m simulations.ts1.axons_dash
# ```

# %%
from toric_spines_sim.events import FlatRateCurve, StochasticEventGenerator
from toric_spines_sim.geometry import sink_endpoint_location_from_swc_file
from toric_spines_sim.paths import PROJECT_ROOT, get_pointset_path, get_swc_path
from toric_spines_sim.simulation import TSSimulator, make_default_parameter_bank
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter

catalogue = PROJECT_ROOT / "toric_spines_sim" / "mechanisms" / "custom-catalogue.so"
if not catalogue.is_file():
    raise FileNotFoundError(
        f"Missing {catalogue}. From the repo root run:\n"
        "  uv run bash scripts/make_custom_catalogue.sh"
    )

swc_path = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_path = get_pointset_path("TS1_synpts.txt", units="microns")
n_synapses = len(load_xyz_points(synpts_path))
sink_xyz = sink_endpoint_location_from_swc_file(swc_path)
print("SWC:   ", swc_path)
print("synpts:", synpts_path, f"({n_synapses} sites)")
print("sink:  ", sink_xyz)

# %% [markdown]
# ## Parameter set and events
#
# Short trial, coarse discretization, sink-only recording. Two synapses at
# 50 Hz; the other streams stay silent. Independent-per-synapse generators
# match synpts order, so no axon remap is required (notebook 10).

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

rates_hz = [0.0] * n_synapses
rates_hz[0] = rates_hz[1] = 50.0
events = StochasticEventGenerator(
    rate_curves=[FlatRateCurve(r) for r in rates_hz],
    n_synapses_per_axon=[1] * n_synapses,
    T_ms=parameters["T_ms"],
    delay_ms=parameters["delay_ms"],
    seed=int(parameters["seed"]),
).generate()
print(f"event streams: {len(events)}")

sim = TSSimulator(
    swc_path,
    synpts_path,
    events,
    parameters,
    record_points={"sink": sink_xyz},
)

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
