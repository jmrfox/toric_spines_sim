import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from toric_spines_sim.report import PdfReport
from toric_spines_sim.viz import RasterPlotter
import matplotlib.pyplot as plt
from toric_spines_sim.events import StochasticEventGenerator, FlatRateCurve
from toric_spines_sim.paths import get_swc_path, get_pointset_path, get_simulation_path
from toric_spines_sim.simulation import TSSimulator
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim import make_simulator_parameter_bank

# MODEL SPECIFICATION
model_name = "ts2"  # lowercase
report_name = "TS2_integration_hh_frozen"
swc_filepath = get_swc_path("TS2_s50_wsink_r10um.swc", units="microns")
synpts_filepath = get_pointset_path("TS2_synpts.txt", units="microns")
results_filepath = get_simulation_path(model_name, "results/TS2_sim_results.pkl")

report_filepath = get_simulation_path(model_name, f"{report_name}.pdf")
# if report exists with same name, add timestamp
if report_filepath.exists():
    report_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filepath = get_simulation_path(model_name, f"{report_name}.pdf")

n_synapses = len(load_xyz_points(synpts_filepath))

# PARAMETERS
parameter_bank = make_simulator_parameter_bank()
parameter_bank["T_ms"].value = 500
parameter_bank["delay_ms"].value = 20
parameter_bank["discretization_um"].value = 1.0
parameter_bank["dt_sim_ms"].value = 0.02
parameter_bank["dt_record_ms"].value = 1.0
parameter_bank["sink_radii_scale"].value = 1.0
parameter_bank["neck_radius_scale"].value = 1.0
parameter_bank["temp_K"].value = 273  # in vivo ~ 313 K
parameter_bank["cm_uF_per_cm2"].value = 10.0  # generally 1.0, sanculi: 10 - 70 uF/cm^2
parameter_bank["rL_ohm_cm"].value = 100  # typical value 30 - 200 ohm cm
# hh
hh_scale = 1.0  # set 0.0 to turn off HH
parameter_bank["hh_tag"].value = 5
parameter_bank["hh_leak_e_mV"].value = -54.3
parameter_bank["K_gbar_S_per_cm2"].value = 0.036 * hh_scale  # hh value = 0.036
parameter_bank["Na_gbar_S_per_cm2"].value = 0.12 * hh_scale  # hh value = 0.12
parameter_bank["hh_leak_g_S_per_cm2"].value = 0.0003 * hh_scale  # hh value = 0.0003
# passive leak
leak_scale = 1.0
# Arbor's "default" leak conductance = 0.001 S/cm^2
parameter_bank["pas_leak_g_S_per_cm2"].value = 0.001 * leak_scale
# AMPA synapse parameters
parameter_bank["ampa_gmax_uS"].value = 0.4
parameter_bank["ampa_tau_ms"].value = 2.0
parameters = parameter_bank.sample()

# INPUT EVENTS
rate_steps_hz = [10.0, 50.0, 100.0]
curves = [FlatRateCurve(r) for r in rate_steps_hz]
events_generator = StochasticEventGenerator(
    rate_curves=curves * n_synapses,
    n_synapses_per_axon=[1] * (n_synapses * len(rate_steps_hz)),
    T_ms=parameters["T_ms"],
    delay_ms=parameters["delay_ms"],
    seed=parameters["seed"],
)
events_tsgroup = events_generator.generate()

# RUN SIMULATION
logger.info("Running simulation...")
sim = TSSimulator(
    swc_filepath,
    synpts_filepath,
    events_tsgroup,
    parameters,
    record_points="all",
)
results = sim.run()
logger.info("Simulation complete.")

# Save results if path provided
if results_filepath:
    results.save(results_filepath)

# Integrate voltages for regions: spine (tag=3) and sink (tag=5)
time_spine, v_spine = results.integrate_voltages_by_tag(3, method="average")
time_sink, v_sink = results.integrate_voltages_by_tag(5, method="average")

# REPORT: Generate PDF report
logger.info("Generating PDF report...")
report = PdfReport(str(report_filepath))
report.add_title(f"{model_name.upper()} Integration Passive Simulation Report")

# REPORT: simulation parameters
report.add_heading("Simulation Parameters", level=2)
params_dict = {name: parameters[name] for name in parameters.index}
report.add_dict_table(params_dict, columns=4)

# REPORT: input information
report.add_heading("Input Configuration", level=2)
input_info = {
    "Number of synapses": n_synapses,
    "Input rates (Hz)": rate_steps_hz,
    "SWC file": swc_filepath.name,
    "Synapse points file": synpts_filepath.name,
}
report.add_dict_table(input_info, columns=2)

# REPORT: Integrated voltage traces by region tag
report.add_heading("Integrated Voltage Traces", level=2)
report.add_paragraph(
    "Voltage traces integrated across all segments with the specified tags. "
    "Tag 3 = spine, Tag 5 = sink."
)


# PLOT (REPORT): voltage traces in spine and sink
fig_voltage, ax_voltage = plt.subplots(figsize=(10, 4))
ax_voltage.plot(time_spine, v_spine, label="Spine (tag=3)", linewidth=1.5, alpha=0.6)
ax_voltage.plot(time_sink, v_sink, label="Sink (tag=5)", linewidth=1.5, alpha=0.6)
ax_voltage.set_xlabel("Time (ms)")
ax_voltage.set_ylabel("Voltage (mV)")
ax_voltage.set_title("Integrated Voltage Traces")
ax_voltage.set_xlim(0, parameters["T_ms"])
ax_voltage.legend()
ax_voltage.grid(True, alpha=0.3)
# compute y limits
# ylim = (min(v_spine.min(), v_sink.min()), max(v_spine.max(), v_sink.max()))
# ax_voltage.set_ylim(ylim)
report.add_figure(fig_voltage)

# PLOT (REPORT): scatter, spine vs sink voltage
fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
ax_scatter.scatter(v_spine, v_sink, alpha=0.5, s=10)
ax_scatter.set_xlabel("Spine Voltage (mV)")
ax_scatter.set_ylabel("Sink Voltage (mV)")
ax_scatter.set_title("Spine vs Sink Voltage")
ax_scatter.grid(True, alpha=0.3)
# Add diagonal reference line
v_min = min(v_spine.min(), v_sink.min())
v_max = max(v_spine.max(), v_sink.max())
ax_scatter.plot([v_min, v_max], [v_min, v_max], "k--", alpha=0.3, label="y=x")
ax_scatter.legend()
report.add_figure(fig_scatter)

# PLOT (REPORT): input event raster
report.add_heading("Input Event Streams", level=2)
plotter_raster = RasterPlotter(
    title="Synaptic Input Events",
    xlim=(0, parameters["T_ms"]),
    figsize=(10, 4),
)
# Convert TsGroup to dict for RasterPlotter (times in ms)
streams_dict = {
    label: ts.index.values * 1000.0 for label, ts in results.input_events.items()
}
plotter_raster.add_streams(streams_dict)
report.add_figure(plotter_raster._fig)

# REPORT: build
report.build()
logger.info(f"Report saved to {report_filepath}")
