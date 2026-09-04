"""TS2 integration: Poisson synaptic drive onto TS2 with a 10 µm sink.

Inputs
    data/swc/microns/TS2_wsink_r10um.swc
    data/pointsets/microns/TS2_synpts.txt

Run from the repository root::

    uv run python -m simulations.ts2.ts2_integration
    uv run python -m simulations.ts2.ts2_integration --verbose

Success: a PDF report under ``simulations/ts2/`` with spine/sink voltage
traces and an optional pickle of ``SimulationResults``.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import matplotlib.pyplot as plt

from toric_spines_sim.events import FlatRateCurve, StochasticEventGenerator
from toric_spines_sim.paths import get_pointset_path, get_simulation_path, get_swc_path
from toric_spines_sim.report import PdfReport
from toric_spines_sim.simulation import TSSimulator, make_default_parameter_bank
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import RasterPlotter

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG logging (morphology, synapse placement, dt, ...)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    model_name = "ts2"
    report_name = "TS2_integration"
    swc_filepath = get_swc_path("TS2_wsink_r10um.swc", units="microns")
    synpts_filepath = get_pointset_path("TS2_synpts.txt", units="microns")
    results_filepath = get_simulation_path(model_name, "results", "TS2_sim_results.pkl")

    report_filepath = get_simulation_path(model_name, f"{report_name}.pdf")
    if report_filepath.exists():
        report_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filepath = get_simulation_path(model_name, f"{report_name}.pdf")

    n_synapses = len(load_xyz_points(synpts_filepath))

    parameter_bank = make_default_parameter_bank()
    parameter_bank["T_ms"].value = 500
    parameter_bank["delay_ms"].value = 20
    parameter_bank["discretization_um"].value = 1.0
    parameter_bank["dt_sim_ms"].value = 0.02
    parameter_bank["dt_record_ms"].value = 1.0
    parameter_bank["sink_radii_scale"].value = 1.0
    parameter_bank["neck_radius_scale"].value = 1.0
    parameter_bank["temp_K"].value = 273  # in vivo ~ 313 K
    parameter_bank["cm_uF_per_cm2"].value = 10.0
    parameter_bank["rL_ohm_cm"].value = 100
    hh_scale = 1.0
    parameter_bank["hh_scale"].value = hh_scale
    parameter_bank["hh_leak_e_mV"].value = -54.3
    parameter_bank["pas_leak_g_S_per_cm2"].value = 0.001
    parameter_bank["ampa_gmax_uS"].value = 0.4
    parameter_bank["ampa_tau_ms"].value = 2.0
    parameters = parameter_bank.sample()
    parameters["hh_tags"] = [5]

    rate_steps_hz = [10.0, 50.0, 100.0]
    curves = [
        FlatRateCurve(rate_steps_hz[i % len(rate_steps_hz)]) for i in range(n_synapses)
    ]
    events_tsgroup = StochasticEventGenerator(
        rate_curves=curves,
        n_synapses_per_axon=[1] * n_synapses,
        T_ms=parameters["T_ms"],
        delay_ms=parameters["delay_ms"],
        seed=int(parameters["seed"]),
    ).generate()

    logger.info(
        "Running TS2 integration: swc=%s synapses=%d T=%s ms seed=%s",
        swc_filepath,
        n_synapses,
        parameters["T_ms"],
        parameters["seed"],
    )
    sim = TSSimulator(
        swc_filepath,
        synpts_filepath,
        events_tsgroup,
        parameters,
        record_points="all",
    )
    results = sim.run()
    logger.info("Simulation complete.")

    results_filepath.parent.mkdir(parents=True, exist_ok=True)
    results.save(results_filepath)
    logger.info("Wrote results to %s", results_filepath)

    time_spine, v_spine = results.integrate_voltages_by_tag(3, method="average")
    time_sink, v_sink = results.integrate_voltages_by_tag(5, method="average")

    logger.info("Generating PDF report at %s", report_filepath)
    report = PdfReport(str(report_filepath))
    report.add_title(f"{model_name.upper()} Integration Simulation Report")

    report.add_heading("Simulation Parameters", level=2)
    params_dict = {name: parameters[name] for name in parameters.index}
    report.add_dict_table(params_dict, columns=4)

    report.add_heading("Input Configuration", level=2)
    report.add_dict_table(
        {
            "Number of synapses": n_synapses,
            "Input rates (Hz), cycled across synapses": rate_steps_hz,
            "SWC file": swc_filepath.name,
            "Synapse points file": synpts_filepath.name,
        },
        columns=2,
    )

    report.add_heading("Integrated Voltage Traces", level=2)
    report.add_paragraph(
        "Voltage traces averaged across segments. Tag 3 = spine, tag 5 = sink."
    )

    fig_voltage, ax_voltage = plt.subplots(figsize=(10, 4))
    ax_voltage.plot(time_spine, v_spine, label="Spine (tag=3)", linewidth=1.5, alpha=0.6)
    ax_voltage.plot(time_sink, v_sink, label="Sink (tag=5)", linewidth=1.5, alpha=0.6)
    ax_voltage.set_xlabel("Time (ms)")
    ax_voltage.set_ylabel("Voltage (mV)")
    ax_voltage.set_title("Integrated Voltage Traces")
    ax_voltage.set_xlim(0, parameters["T_ms"])
    ax_voltage.legend()
    ax_voltage.grid(True, alpha=0.3)
    report.add_figure(fig_voltage)

    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
    ax_scatter.scatter(v_spine, v_sink, alpha=0.5, s=10)
    ax_scatter.set_xlabel("Spine Voltage (mV)")
    ax_scatter.set_ylabel("Sink Voltage (mV)")
    ax_scatter.set_title("Spine vs Sink Voltage")
    ax_scatter.grid(True, alpha=0.3)
    v_min = min(v_spine.min(), v_sink.min())
    v_max = max(v_spine.max(), v_sink.max())
    ax_scatter.plot([v_min, v_max], [v_min, v_max], "k--", alpha=0.3, label="y=x")
    ax_scatter.legend()
    report.add_figure(fig_scatter)

    report.add_heading("Input Event Streams", level=2)
    plotter_raster = RasterPlotter(
        title="Synaptic Input Events",
        xlim=(0, parameters["T_ms"]),
        figsize=(10, 4),
    )
    streams_dict = {
        label: ts.index.values * 1000.0 for label, ts in results.input_events.items()
    }
    plotter_raster.add_streams(streams_dict)
    report.add_figure(plotter_raster._fig)

    report.build()
    logger.info("Report saved to %s", report_filepath)


if __name__ == "__main__":
    main()
