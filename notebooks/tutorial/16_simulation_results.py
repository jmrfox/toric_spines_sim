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
# # 16 — `SimulationResults`
#
# `TSSimulator.run()` returns a `SimulationResults` object: voltage traces
# as a pynapple `TsdFrame`, input events, synapse and morphology objects,
# plus helpers to save or load the run (dill) and
# `integrate_voltages_by_tag`. The NMODL catalogue (`mechanisms`) is
# required. The short run uses the same TS1 micron files from `data/` as
# `tssimulator`.
#
# `integrate_voltages_by_tag` only sees probes that were recorded. A sink-only
# run can average tag 5/6; averaging the spine (tag 3) needs `"all"` probes
# or explicit spine XYZ.

# %%
from toric_spines_sim.events import FlatRateCurve, StochasticEventGenerator
from toric_spines_sim.geometry import sink_endpoint_location_from_swc_file
from toric_spines_sim.model import check_catalogue
from toric_spines_sim.paths import (
    NOTEBOOKS_DIR,
    get_pointset_path,
    get_swc_path,
)
from toric_spines_sim.simulation import (
    SimulationResults,
    TSSimulator,
    make_default_parameter_bank,
)
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import TimeSeriesPlotter

check_catalogue()

tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"
swc_path = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_path = get_pointset_path("TS1_synpts.txt", units="microns")
n_synapses = len(load_xyz_points(synpts_path))
sink_xyz = sink_endpoint_location_from_swc_file(swc_path)

parameter_bank = make_default_parameter_bank()
parameter_bank["T_ms"].value = 150.0
parameter_bank["delay_ms"].value = 20.0
parameter_bank["discretization_um"].value = 1.0
parameter_bank["dt_sim_ms"].value = 0.05
parameter_bank["dt_record_ms"].value = 0.1
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

results = TSSimulator(
    swc_path,
    synpts_path,
    events,
    parameters,
    record_points={"sink": sink_xyz},
    # record_points="all",
).run()

print("SimulationResults:", type(results).__name__)
print("  voltage_traces:", type(results.voltage_traces).__name__, results.voltage_traces.shape)
print("  columns:", list(results.voltage_traces.columns))
print("  n input streams:", len(results.input_events))
print("  n synapses:", len(results.synapses))
print("  n gap junctions:", len(results.gap_junctions))
print("  record_points:", results.record_points)
print("  swc_filepath:", results.swc_filepath)
print("  synpts_filepath:", results.synpts_filepath)
print("  tags present on segment tree:", results.get_tags())

# %% [markdown]
# ## Integrate by SWC tag
#
# The `"sink"` probe sits on the sink cylinder (tag 5) or tip (tag 6).

# %%
tags = results.get_tags()
print("tags:", tags)
# print(results.voltage_traces.columns)
try:
    v_sink = results.integrate_voltages_by_tag([5, 6], method="average")
    plotter = TimeSeriesPlotter(
        title="integrate_voltages_by_tag([5, 6])",
        xlim=(0.0, float(parameters["T_ms"])),
        figsize=(12, 4),
    )
    plotter.add_time_series(v_sink, label="sink tags")
    plotter.add_time_series(results.voltage_traces["sink"], label="probe sink")
    plotter.show()
except ValueError as exc:
    print("integrate skipped:", exc)

# %% [markdown]
# ## Save / load
#
# Dill pickle. Arbor objects inside the results are part of the data written
# to the pickle — load on the same Arbor version you used to run.

# %%
tutorial_output_dir.mkdir(parents=True, exist_ok=True)
output_path = tutorial_output_dir / "ts1_sink_results.pkl"
results.save(output_path)
loaded = SimulationResults.load(output_path)
print("wrote", output_path)
print("reloaded columns:", list(loaded.voltage_traces.columns))
print("reloaded shape:", loaded.voltage_traces.shape)
print(
    "(Arbor cell / synapses are not in the pickle; "
    "integrate_voltages_by_tag needs the live instance.)"
)
