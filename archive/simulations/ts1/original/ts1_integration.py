"""
Integration study of TS1 with sink
"""

import logging
from datetime import datetime
from toric_spines_sim.report import PdfReport
from toric_spines_sim.viz import RasterPlotter, TimeSeriesPlotter
from toric_spines_sim.events import StochasticEventGenerator
from toric_spines_sim.paths import get_swc_path, get_pointset_path, get_simulation_path
from toric_spines_sim.simulation import TSSimulator, random_axon_events
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim import make_simulator_parameter_bank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORT_NAME = "TS1_integration"
REPORT_NAME += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_FILEPATH = get_simulation_path("ts1", "original", "results", f"{REPORT_NAME}.pdf")

swc_filepath = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")
results_path = get_simulation_path("ts1", "original", "results")
results_pkl_filepath = results_path / f"{REPORT_NAME}.pkl"
n_synapses = len(load_xyz_points(synpts_filepath))

parameter_bank = make_simulator_parameter_bank()
parameter_bank["T_ms"].value = 1000.0
parameter_bank["delay_ms"].value = 10.0
parameter_bank["discretization_um"].value = 1.0
parameter_bank["dt_sim_ms"].value = 0.02
parameter_bank["dt_record_ms"].value = 0.5
parameter_bank["sink_radii_scale"].value = 1.0
parameter_bank["neck_radius_scale"].value = 1.0
# Arbor's "default" membrane capacitance = 0.01 uF/cm^2
# but analysis from Sanculi gives O(1 uF/cm^2)
parameter_bank["cm_uF_per_cm2"].value = 1.0
parameter_bank["rL_ohm_cm"].value = 150
# hh
hh_on = False
hh_scale = 1.0
parameter_bank["hh_leak_e_mV"].value = -54.3
if hh_on:
    parameter_bank["K_gbar_S_per_cm2"].value = 0.036 * hh_scale  # hh value = 0.036
    parameter_bank["Na_gbar_S_per_cm2"].value = 0.12 * hh_scale  # hh value = 0.12
    parameter_bank["hh_leak_g_S_per_cm2"].value = 0.0003 * hh_scale  # hh value = 0.0003
else:
    parameter_bank["K_gbar_S_per_cm2"].value = 0.0
    parameter_bank["Na_gbar_S_per_cm2"].value = 0.0
    parameter_bank["hh_leak_g_S_per_cm2"].value = 0.0
# passive leak
# Arbor's "default" leak conductance = 0.001 S/cm^2
leak_on = True
if leak_on:
    parameter_bank["pas_leak_g_S_per_cm2"].value = 0.00005
else:
    parameter_bank["pas_leak_g_S_per_cm2"].value = 0.0
# synapse parameters
parameter_bank["ampa_gmax_uS"].value = 0.005
parameter_bank["ampa_tau_ms"].value = 2.0
parameters = parameter_bank.sample()

# Create events configuration using sine-modulated stochastic generator
# Each synapse on its own axon (independent mode with per-axon rate curves)

# curves = [SineRateCurve(peak_rate_hz=500.0, freq_hz=100.0) for _ in range(n_synapses)]
# event_generator = StochasticEventGenerator(
#     rate_curves=curves,
#     n_synapses_per_axon=[1] * n_synapses,
#     T_ms=parameter_bank["T_ms"].value,
#     delay_ms=parameter_bank["delay_ms"].value,
#     seed=parameter_bank["seed"].value,
# )


rate_curves, n_synapses_per_axon = random_axon_events(
    n_synapses=n_synapses,
    n_axons=10,
    mod_freq_hz=100,
    peak_rate_range_hz=[0, 50],
    seed=int(parameters["seed"]),
)
event_generator = StochasticEventGenerator(
    rate_curves=rate_curves,
    n_synapses_per_axon=n_synapses_per_axon,
    T_ms=parameters["T_ms"],
    delay_ms=parameters["delay_ms"],
    seed=parameters["seed"],
)


events_tsgroup = event_generator.generate()

# Run simulation using TSSimulator
sim = TSSimulator(
    swc_filepath,
    synpts_filepath,
    events_tsgroup,
    parameters,
    record_points="all",
)
results = sim.run()

# write cell
write_cell = False
if write_cell:
    sim.write_cell(results_path / "TS1_wsink_r10um.acc")

# Save results if path provided
if results_pkl_filepath:
    results.save(results_pkl_filepath)

# Generate PDF report with integrated voltage traces
logger.info("Generating PDF report...")

# Create report
report = PdfReport(str(REPORT_FILEPATH))
report.add_title("TS1 Integration Passive Simulation Report")

# Add simulation parameters
report.add_heading("Simulation Parameters", level=2)
params_dict = {name: parameters[name] for name in parameters.index}
report.add_dict_table(params_dict, columns=4)

# Add input information
report.add_heading("Input Configuration", level=2)
input_info = {
    "Number of compartments": len(results.voltage_traces.columns),
    "Number of synapses": n_synapses,
    "SWC file": swc_filepath.name,
    "Synapse points file": synpts_filepath.name,
}
report.add_dict_table(input_info, columns=2)
report.add_paragraph("Event generator:\n" + str(event_generator))


# Plot integrated voltage traces (notebook style)
report.add_heading("Integrated Voltage Traces", level=2)
report.add_paragraph(
    "Voltage traces integrated across all segments with the specified tags. "
    "Tag 3 = spine, Tag 5 = sink."
)

# Get integrated voltages for spine (tag=3) and sink (tag=5) as Tsd objects
v_spine = results.integrate_voltages_by_tag(3, method="average")
v_sink = results.integrate_voltages_by_tag(5, method="average")

# Create time series plotter with notebook-style settings
plotter_ts = TimeSeriesPlotter(
    title="Integrated membrane potential (spine vs sink)",
    xlim=(0, parameters["T_ms"]),
    figsize=(12, 6),
)

# Add integrated traces (Tsd objects - TimeSeriesPlotter handles conversion)
plotter_ts.add_time_series(v_spine, label="Spine (tag=3)", linewidth=1.5)
plotter_ts.add_time_series(v_sink, label="Sink (tag=5)", linewidth=1.5)

report.add_figure(plotter_ts.figure)

# Add input raster plot (notebook style)
report.add_heading("Input Event Streams", level=2)
plotter_rp = RasterPlotter(
    title="Input event streams",
    xlim=(0, parameters["T_ms"]),
    figsize=(12, 4),
)
plotter_rp.add_streams(results.input_events)
report.add_figure(plotter_rp.figure)

# Build the report
report.build()
logger.info("Report saved to %s", REPORT_FILEPATH)
