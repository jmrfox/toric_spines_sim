"""TS2 voltage animation: Poisson drive, all-segment probes, HTML frusta movie.

Inputs
    data/swc/microns/TS2_wsink_r10um.swc
    data/pointsets/microns/TS2_synpts.txt

Run from the repository root::

    uv run python -m simulations.ts2.ts2_integration_viz

Success: ``simulations/ts2/TS2_integration_viz.html``.
"""

from __future__ import annotations

import logging

import numpy as np
from swctools import FrustaSet
from swctools.viz import animate_frusta_timeseries

from toric_spines_sim.events import FlatRateCurve, StochasticEventGenerator
from toric_spines_sim.geometry.sink import sink_endpoint_location_from_swc_file
from toric_spines_sim.paths import get_pointset_path, get_simulation_path, get_swc_path
from toric_spines_sim.simulation import TSSimulator, make_default_parameter_bank
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

swc_filepath = get_swc_path("TS2_wsink_r10um.swc", units="microns")
synpts_filepath = get_pointset_path("TS2_synpts.txt", units="microns")
total_synapses = len(load_xyz_points(synpts_filepath))
model_name = "ts2"

logger.info("%s has %d synapses; swc=%s", model_name, total_synapses, swc_filepath)

sink_endpoint = sink_endpoint_location_from_swc_file(swc_filepath)
logger.info("Sink endpoint: %s", sink_endpoint)

animation_file = get_simulation_path("ts2", "TS2_integration_viz.html")

parameter_bank = make_default_parameter_bank()
parameter_bank["T_ms"].value = 200
parameter_bank["delay_ms"].value = 10
parameter_bank["discretization_um"].value = 1.0
parameter_bank["dt_sim_ms"].value = 0.02
parameter_bank["dt_record_ms"].value = 1.0
parameter_bank["sink_radii_scale"].value = 1.0
parameter_bank["cm_uF_per_cm2"].value = 2.0
parameter_bank["rL_ohm_cm"].value = 150
hh_on = False
parameter_bank["hh_leak_e_mV"].value = -54.3
if hh_on:
    parameter_bank["hh_scale"].value = 0.05
else:
    parameter_bank["hh_scale"].value = 0.0
parameter_bank["pas_leak_g_S_per_cm2"].value = 0.001
parameter_bank["ampa_gmax_uS"].value = 0.1
parameter_bank["ampa_tau_ms"].value = 2.0
parameters = parameter_bank.sample()
parameters["hh_tags"] = [5] if hh_on else []

input_rates_hz = [50.0] * total_synapses
curves = [FlatRateCurve(r) for r in input_rates_hz]
events = StochasticEventGenerator(
    rate_curves=curves,
    n_synapses_per_axon=[1] * total_synapses,
    T_ms=parameters["T_ms"],
    delay_ms=parameters["delay_ms"],
    seed=int(parameters["seed"]),
).generate()

logger.info("Running simulation...")
sim = TSSimulator(
    swc_filepath,
    synpts_filepath,
    events,
    parameters,
    record_points="all",
)
results = sim.run()
logger.info("Simulation complete.")

record_points = results.record_points
segment_indices_to_plot = None

if segment_indices_to_plot:
    plotter_ts = TimeSeriesPlotter(
        title="Membrane potential in sink",
        xlim=(0, parameters["T_ms"]),
        figsize=(12, 4),
        nrows=len(segment_indices_to_plot),
    )
    for i, seg_idx in enumerate(segment_indices_to_plot):
        probe_label = f"probe_seg_{seg_idx}"
        tsd = results.voltage_traces[probe_label]
        times_ms = tsd.index.values * 1000.0
        volts = tsd.values
        plotter_ts.add_time_series(times_ms, volts, row=i)
    plotter_ts.show()

    plotter_rp = RasterPlotter(
        title="Input event streams",
        xlim=(0, parameters["T_ms"]),
        figsize=(12, 4),
    )
    streams_dict = {
        label: ts.index.values * 1000.0 for label, ts in results.input_events.items()
    }
    plotter_rp.add_streams(streams_dict)
    plotter_rp.show()

frusta = FrustaSet.from_swc_file(str(swc_filepath))
record_point_coords = list(record_points.values())
frustum_indices = [frusta.nearest_frustum_index(coord) for coord in record_point_coords]
probe_labels = list(record_points.keys())

time_domain = results.voltage_traces.index.values * 1000.0
n_timepoints = len(time_domain)
n_frusta = frusta.n_frusta
amplitudes = np.full((n_timepoints, n_frusta), np.nan)

for probe_label, frustum_idx in zip(probe_labels, frustum_indices):
    tsd = results.voltage_traces[probe_label]
    amplitudes[:, frustum_idx] = tsd.values

animate_frusta_timeseries(
    frusta,
    time_domain=time_domain,
    amplitudes=amplitudes,
    colorscale="Viridis",
    fps=30,
    stride=1,
    output_path=animation_file,
    auto_open=False,
)
logger.info("Wrote animation to %s", animation_file)
