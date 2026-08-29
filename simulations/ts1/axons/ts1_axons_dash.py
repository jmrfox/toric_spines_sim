"""
TS1 Axon Variation Interactive Dashboard

Run a single axon-configuration simulation and launch a Dash dashboard with
synchronized 3D animation, voltage traces, and event raster panels.
"""

from __future__ import annotations

import logging

from toric_spines_sim.paths import (
    get_pointset_path,
    get_simulation_path,
    get_swc_path,
)
from toric_spines_sim.simulation import TSSimulator
from toric_spines_sim.viz import (
    create_simulation_dash_app,
    prepare_simulation_dashboard_data,
)

from simulations.ts1.ts1_inputs import (
    simultaneous_pulse_input,
    sequential_pulse_input,
    synapse_poisson_input,
    axon_poisson_input,
    axon_periodic_input,
)
from simulations.ts1.ts1_utils import build_synapse_colors_by_axon



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================

# Input scenario: set the function and its kwargs (see ts1_inputs.py).
# SCENARIO = simultaneous_pulse_input()
SCENARIO = sequential_pulse_input()
# SCENARIO = synapse_poisson_input()
# SCENARIO = axon_poisson_input()
# SCENARIO = axon_periodic_input()


# Dashboard server
HOST = "127.0.0.1"
PORT = 8050
DEBUG = False

# Animation playback
FPS = 60
STRIDE = 1
MAX_CLIENTSIDE_FRAMES = 5000

# Disk cache
CACHE_DIR = str(
    get_simulation_path("ts1", "axons", "results")
)
USE_CACHE = False

# =============================================================================


def _run_simulation() -> tuple:
    logger.info("Scenario: %s", SCENARIO.metadata)

    swc_filepath = get_swc_path("TS1_wsink_r10um.swc", units="microns")
    synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")

    sim = TSSimulator(
        swc_filepath,
        synpts_filepath,
        SCENARIO.events_tsgroup,
        SCENARIO.parameters,
        record_points="all",
    )
    results = sim.run()
    return results, swc_filepath


def main() -> None:
    results, swc_filepath = _run_simulation()
    synapse_colors = build_synapse_colors_by_axon(
        SCENARIO.axon_synapses,
        len(results.synapses),
    )

    scenario_label = str(SCENARIO.metadata.get("input_scenario", "scenario"))
    active_axons = SCENARIO.metadata.get("active_axons", [])
    axon_suffix = "_".join(str(a) for a in sorted(active_axons)) or "all"
    title = f"TS1 vary axons ({scenario_label}, axons {axon_suffix})"

    dashboard_data = prepare_simulation_dashboard_data(
        results,
        swc_filepath,
        SCENARIO.parameters,
        axon_synapses=SCENARIO.axon_synapses,
        synapse_colors=synapse_colors,
        animation_kwargs=dict(
            stride=STRIDE,
            opacity=0.8,
            flatshading=True,
            show_axes=False,
            show_synapses=True,
            synapse_ball_size=0.12,
            synapse_stacks=3,
            synapse_slices=6,
            synapse_flash_duration_ms=SCENARIO.synapse_flash_duration_ms,
            colorbar_title="Voltage (mV)",
            frustum_sides=8,
        ),
    )
    dashboard_data.metadata.update(SCENARIO.metadata)

    cache_dir = CACHE_DIR if USE_CACHE else None
    app = create_simulation_dash_app(
        dashboard_data,
        fps=FPS,
        title=title,
        cache_dir=cache_dir,
        clientside_max_frames=MAX_CLIENTSIDE_FRAMES,
    )
    logger.info("Launching dashboard at http://%s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == "__main__":
    main()
