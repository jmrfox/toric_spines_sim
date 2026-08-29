# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from pathlib import Path
import arbor as A
import numpy as np
from toric_spines_sim import (
    prepare_gap_junctions,
    prepare_ampa_synapses,
    make_simulator_parameter_bank,
    TSModel,
    TSRecipe,
)
from toric_spines_sim.events import (
    DeterministicEventGenerator,
    StochasticEventGenerator,
    FlatRateCurve,
)
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter
from jscip import ParameterSet
from toric_spines_sim.viz import (
    plot_morphology_3d,
    plot_morphology_frusta_3d,
    VizConfig,
)
from toric_spines_sim.geometry.sink import sink_endpoint_location_from_swc_file
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.paths import get_swc_path, get_pointset_path

import matplotlib.pyplot as plt

from typing import Tuple

# %% [markdown]
# # Integration study of TS1 with sink
#
# First, we set up a function to do the simulation for a given set of input rates and parameters.

# %%
swc_filepath = get_swc_path("TS1_wsink_r20um.swc", units="microns")
synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")
total_synapses = len(load_xyz_points(synpts_filepath))
model_name = "ts1"

logger.info(f"{model_name} has {total_synapses} synapses")

sink_endpoint = sink_endpoint_location_from_swc_file(swc_filepath)
logger.info(f"Sink endpoint: {sink_endpoint}")


# %%
def simulation(
    swc_filepath,
    synpts_filepath,
    input_rates_hz,
    parameters,
    record_point,
    event_type="poisson",
):
    synapses = prepare_ampa_synapses(synpts_filepath, parameters=parameters)
    record_points = {"probe_0": record_point}  # single record point
    gap_junctions = prepare_gap_junctions(swc_filepath, parameters=parameters)
    synapse_labels = list(synapses.keys())
    # Create per-axon rate curves (one per input channel)
    curves = [FlatRateCurve(r) for r in input_rates_hz]
    n_synapses = len(input_rates_hz)

    if event_type == "periodic":
        # ICPeriodicEvents -> DeterministicEventGenerator (independent mode)
        event_generator = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1] * n_synapses,
            T_ms=parameters["T_ms"],
            delay_ms=parameters["delay_ms"],
            labels=synapse_labels,
        )
    elif event_type == "poisson":
        # ICPoissonEvents -> StochasticEventGenerator (independent mode)
        event_generator = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1] * n_synapses,
            T_ms=parameters["T_ms"],
            delay_ms=parameters["delay_ms"],
            seed=parameters["seed"],
            labels=synapse_labels,
        )
    else:
        raise ValueError("event_type must be 'periodic' or 'poisson'")
    events = event_generator.generate()
    tsm = TSModel(
        swc_path=swc_filepath,
        synapses=synapses,
        gap_junctions=gap_junctions,
        record_points=record_points,
        parameters=parameters,
    )
    build_cell_results = tsm.build_cell()
    cell = build_cell_results["cell"]
    recipe = TSRecipe(
        cell,
        synapses=synapses,
        gap_junctions=gap_junctions,
        record_points=record_points,
        events=events,
        parameters=parameters,
    )
    ctx = A.context()
    dec = A.partition_load_balance(recipe, ctx)
    sim = A.simulation(recipe, ctx, dec)
    dt_record_ms = parameters["dt_record_ms"]
    handle = sim.sample(0, "v_probe_0", A.regular_schedule(dt_record_ms * A.units.ms))
    sim.record(A.spike_recording.all)
    T_ms = parameters["T_ms"]
    dt_sim_ms = parameters["dt_sim_ms"]
    sim.run(T_ms * A.units.ms, dt_sim_ms * A.units.ms)
    simulation_results = {
        "probe": sim.samples(handle),
        "events": events,
        "synapses": synapses,
        "gap_junctions": gap_junctions,
        "record_points": record_points,
        "cell": cell,
        "morphology": build_cell_results["morphology"],
        "segment_tree": build_cell_results["segment_tree"],
        "decor": build_cell_results["decor"],
        "labels": build_cell_results["labels"],
        "cvp": build_cell_results["cvp"],
    }
    return simulation_results


# %%
parameter_bank = make_simulator_parameter_bank()
parameter_bank["T_ms"].value = 500.0
parameter_bank["discretization_um"].value = 1.0
parameter_bank["sink_radii_scale"].value = 1.0
parameter_bank["cm_uF_per_cm2"].value = (
    2.0  # default = 0.01 uF/cm^2, analysis from Sanculi gives O(1 uF/cm^2)
)
parameter_bank["rL_ohm_cm"].value = 150
# hh
hh_on = False
hh_scale = 0.05
parameter_bank["hh_leak_e_mV"].value = -54.3
if hh_on:
    parameter_bank["K_gbar_S_per_cm2"].value = 0.036 * hh_scale  # hh value = 0.036
    parameter_bank["Na_gbar_S_per_cm2"].value = 0.12 * hh_scale  # hh value = 0.12
    parameter_bank["hh_leak_g_S_per_cm2"].value = 0.0003
else:
    parameter_bank["K_gbar_S_per_cm2"].value = 0.0
    parameter_bank["Na_gbar_S_per_cm2"].value = 0.0
    parameter_bank["hh_leak_g_S_per_cm2"].value = 0.0
# passive leak
leak_on = True
if leak_on:
    parameter_bank["pas_leak_g_S_per_cm2"].value = (
        0.001  # default = 0.001 ( 1 / 1000 Ohms * cm^2 )
    )
else:
    parameter_bank["pas_leak_g_S_per_cm2"].value = 0.0
# synapse parameters
parameter_bank["ampa_gmax_uS"].value = 0.1
parameter_bank["ampa_tau_ms"].value = 2.0
parameters = parameter_bank.sample()

print("Parameter set:\n", parameters)

# %%
input_rates_hz = [0.0] * total_synapses
input_rates_hz[0] = input_rates_hz[1] = 50.0

results = simulation(
    swc_filepath,
    synpts_filepath,
    input_rates_hz,
    parameters,
    record_point=sink_endpoint,
    event_type="poisson",
)
plotter_ts = TimeSeriesPlotter(
    title="Membrane potential in sink",
    xlim=(0, parameters["T_ms"]),
    figsize=(12, 4),
)
plotter_ts.add_time_series_from_arbor(results["probe"])
plotter_ts.show()
plotter_rp = RasterPlotter(
    title="Input event streams", xlim=(0, parameters["T_ms"]), figsize=(12, 4)
)
plotter_rp.add_streams(results["events"])
plotter_rp.show()

# %%

# %%
