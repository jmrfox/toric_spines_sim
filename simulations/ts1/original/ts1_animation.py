import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
import arbor as A
import numpy as np

from toric_spines_sim import make_simulator_parameter_bank
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter, Animation
import pynapple as nap
from toric_spines_sim.paths import get_swc_path, get_pointset_path, get_simulation_path
from toric_spines_sim.simulation import TSSimulator
from toric_spines_sim.events import StochasticEventGenerator, FlatRateCurve
from toric_spines_sim.utils import load_xyz_points

REPORT_NAME = "TS1_animation"
swc_filepath = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")
results_filepath = get_simulation_path("ts1", "original", "results", f"{REPORT_NAME}.pkl")
animation_file = get_simulation_path("ts1", "original", "results", f"{REPORT_NAME}.html")
n_synapses = len(load_xyz_points(synpts_filepath))

parameter_bank = make_simulator_parameter_bank()
parameter_bank["T_ms"].value = 300
parameter_bank["delay_ms"].value = 10
parameter_bank["discretization_um"].value = 0.1
parameter_bank["dt_sim_ms"].value = 0.01
parameter_bank["dt_record_ms"].value = 1.0
parameter_bank["sink_radii_scale"].value = 1.0
parameter_bank["neck_radius_scale"].value = 1.0
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


input_rates_hz = [20.0] * n_synapses
curves = [FlatRateCurve(r) for r in input_rates_hz]
events_generator = StochasticEventGenerator(
    rate_curves=curves,
    n_synapses_per_axon=[1] * n_synapses,
    T_ms=parameters["T_ms"],
    delay_ms=parameters["delay_ms"],
    seed=parameters["seed"],
)
events_tsgroup = events_generator.generate()

# Run simulation using TSSimulator
sim = TSSimulator(
    swc_filepath,
    synpts_filepath,
    events_tsgroup,
    parameters,
    record_points="all",
)
results = sim.run()

# Save results if path provided
if results_filepath:
    results.save(results_filepath)

# plot voltage and input events for selected segments
# if None, skip
segment_indices_to_plot = None  # [0, 50, 100]

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
        label: ts.index.values * 1000.0 for label, ts in results.input_events.items()
    }
    plotter_rp.add_streams(streams_dict)
    plotter_rp.show()

# animation
Animation(results, swc_filepath=swc_filepath).create(
    output_path=animation_file,
    colorscale="Plasma",
    fps=30,
    stride=1,
    auto_open=False,
    clim=None,
    opacity=0.8,
    flatshading=True,
    radius_scale=1.0,
    title=None,
    colorbar_title="Voltage (mV)",
    show_axes=False,
    show_synapses=True,
    synapse_ball_size=0.05,
    synapse_inactive_color="#154f25",
    synapse_active_color="#11f54e",
    synapse_flash_duration_ms=4.0,
)
