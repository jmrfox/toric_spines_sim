import logging

logging.basicConfig(level=logging.INFO)
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
from toric_spines_sim.geometry.swc import get_center_coordinates_for_all_segments
from toric_spines_sim.events import (
    DeterministicEventGenerator,
    StochasticEventGenerator,
    FlatRateCurve,
)
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter
import pynapple as nap
from toric_spines_sim.geometry.sink import sink_endpoint_location_from_swc_file
from toric_spines_sim.utils import load_xyz_points
from typing import Tuple
from toric_spines_sim.paths import get_swc_path, get_pointset_path, get_simulation_path

swc_filepath = get_swc_path("TS2_s50_wsink_r20um.swc", units="microns")
synpts_filepath = get_pointset_path("TS2_synpts.txt", units="microns")
total_synapses = len(load_xyz_points(synpts_filepath))
model_name = "ts2"
record_points = get_center_coordinates_for_all_segments(swc_filepath)

logger.info(f"{model_name} has {total_synapses} synapses")

sink_endpoint = sink_endpoint_location_from_swc_file(swc_filepath)
logger.info(f"Sink endpoint: {sink_endpoint}")

animation_file = get_simulation_path("ts2", "TS2_integration_viz.html")

parameter_bank = make_simulator_parameter_bank()
parameter_bank["T_ms"].value = 200
parameter_bank["delay_ms"].value = 10
parameter_bank["discretization_um"].value = 1.0
parameter_bank["dt_sim_ms"].value = 0.02
parameter_bank["dt_record_ms"].value = 1.0
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


def simulation(
    swc_filepath,
    synpts_filepath,
    input_rates_hz,
    parameters,
    record_points,
    event_type="poisson",
):
    synapses = prepare_ampa_synapses(synpts_filepath, parameters=parameters)
    gap_junctions = prepare_gap_junctions(swc_filepath, parameters=parameters)
    synapse_labels = list(synapses.keys())
    curves = [FlatRateCurve(r) for r in input_rates_hz]
    n_synapses_local = len(input_rates_hz)
    if event_type == "periodic":
        event_generator = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1] * n_synapses_local,
            T_ms=parameters["T_ms"],
            delay_ms=parameters["delay_ms"],
            labels=synapse_labels,
        )
    elif event_type == "poisson":
        event_generator = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1] * n_synapses_local,
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
        events_ms=events,
        parameters=parameters,
    )
    ctx = A.context()
    dec = A.partition_load_balance(recipe, ctx)
    sim = A.simulation(recipe, ctx, dec)

    # Sample from all probe points
    probe_handles = {}
    for probe_label in record_points.keys():
        probe_id = f"v_{probe_label}"
        handle = sim.sample(
            0,
            probe_id,
            A.regular_schedule(parameters["dt_record_ms"] * A.units.ms),
        )
        probe_handles[probe_label] = handle

    sim.record(A.spike_recording.all)
    sim.run(
        parameters["T_ms"] * A.units.ms,
        parameters["dt_sim_ms"] * A.units.ms,
    )

    # Collect all probe data and convert to pynapple TsdFrame
    probe_data = {}
    voltage_traces = {}  # Tsd objects for each probe
    for probe_label, handle in probe_handles.items():
        samples = sim.samples(handle)
        probe_data[probe_label] = samples
        # Convert to pynapple Tsd
        if samples and len(samples) > 0:
            data_array, _meta = samples[0]
            times_ms = data_array[:, 0]
            volts = data_array[:, 1]
            # Convert ms to seconds for pynapple
            tsd = nap.Tsd(t=times_ms / 1000.0, d=volts, time_units="s")
            voltage_traces[probe_label] = tsd

    # Build TsdFrame from all voltage traces
    if voltage_traces:
        voltage_tsdframe = nap.TsdFrame(voltage_traces, time_units="s")
    else:
        voltage_tsdframe = None

    simulation_results = {
        "probes": probe_data,
        "voltage_traces": voltage_tsdframe,
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


input_rates_hz = [50.0] * total_synapses

logger.info("Running simulation...")
results = simulation(
    swc_filepath,
    synpts_filepath,
    input_rates_hz,
    parameters,
    record_points=record_points,
    event_type="poisson",
)
logger.info("Simulation complete!")

# plot voltage and input events for selected segments
# if None, skip
segment_indices_to_plot = None  # [0, 10, 20]

if segment_indices_to_plot:
    plotter_ts = TimeSeriesPlotter(
        title="Membrane potential in sink",
        xlim=(0, parameters["T_ms"]),
        figsize=(12, 4),
        nrows=len(segment_indices_to_plot),
    )
    for i, seg_idx in enumerate(segment_indices_to_plot):
        probe_label = f"probe_seg_{seg_idx}"
        tsd = results["voltage_traces"][probe_label]
        times_ms = tsd.index.values * 1000.0  # convert from seconds
        volts = tsd.values
        plotter_ts.add_time_series(times_ms, volts, row=i)
    plotter_ts.show()

    plotter_rp = RasterPlotter(
        title="Input event streams",
        xlim=(0, parameters["T_ms"]),
        figsize=(12, 4),
    )
    # Convert TsGroup to dict for RasterPlotter (times in ms)
    streams_dict = {
        label: ts.index.values * 1000.0 for label, ts in results["events"].items()
    }
    plotter_rp.add_streams(streams_dict)
    plotter_rp.show()

# animation
# test animate_frusta_timeseries
from swctools import FrustaSet
from swctools.viz import animate_frusta_timeseries
import numpy as np
from pathlib import Path

# Create FrustaSet from SWC and animate voltage timeseries
from swctools import FrustaSet
from swctools.viz import animate_frusta_timeseries
import numpy as np

# Create FrustaSet from SWC file
frusta = FrustaSet.from_swc_file(str(swc_filepath))

# Map each record point to its nearest frustum index
record_point_coords = list(record_points.values())  # List of (x, y, z) tuples
frustum_indices = [frusta.nearest_frustum_index(coord) for coord in record_point_coords]

# Get the probe labels in the same order as record_points
probe_labels = list(record_points.keys())  # ["probe_seg_0", "probe_seg_1", ...]

# Extract time domain from voltage_traces TsdFrame
time_domain = results["voltage_traces"].index.values * 1000.0  # convert to ms
n_timepoints = len(time_domain)

# Create amplitude matrix shaped (T, N) where T = timepoints, N = frusta.n_frusta
n_frusta = frusta.n_frusta
amplitudes = np.full((n_timepoints, n_frusta), np.nan)  # Initialize with NaN

# Fill in the voltage data for frusta that have corresponding probes
for probe_label, frustum_idx in zip(probe_labels, frustum_indices):
    tsd = results["voltage_traces"][probe_label]
    amplitudes[:, frustum_idx] = tsd.values  # Voltage values

fig = animate_frusta_timeseries(
    frusta,
    time_domain=time_domain,
    amplitudes=amplitudes,
    colorscale="Viridis",
    fps=30,
    stride=1,
    output_path=animation_file,
    auto_open=False,
)
