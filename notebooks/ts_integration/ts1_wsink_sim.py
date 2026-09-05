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
# Short TS1 sink-voltage demo (one pair of synapses).
#
# Inputs: ``TS1_wsink_r10um.swc``, ``TS1_synpts.txt`` (microns).

# %%
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from toric_spines_sim.simulation import make_default_parameter_bank
from toric_spines_sim.kmatrix import simulation_probe_dict as simulation
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter
from toric_spines_sim.viz import (
    plot_morphology_frusta_3d,
    VizConfig,
)
from toric_spines_sim.geometry.sink import sink_endpoint_location_from_swc_file
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.paths import get_swc_path, get_pointset_path

# %% [markdown]
# # Integration study of TS1 with sink
#
# First, we set up a function to do the simulation for a given set of input rates and parameters.

# %%
swc_filepath = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")
total_synapses = len(load_xyz_points(synpts_filepath))
model_name = "ts1"

logger.info(f"{model_name} has {total_synapses} synapses")

sink_endpoint = sink_endpoint_location_from_swc_file(swc_filepath)
logger.info(f"Sink endpoint: {sink_endpoint}")


# Simulation helper: ``simulation`` is ``toric_spines_sim.kmatrix.simulation_probe_dict``.


# %%
parameter_bank = make_default_parameter_bank()
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
parameter_bank["hh_scale"].value = hh_scale if hh_on else 0.0
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
parameters["hh_tags"] = [5] if hh_on else []

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
