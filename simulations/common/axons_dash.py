"""Shared Dash dashboard runner for axon-mapped simulations."""

from __future__ import annotations

import logging

from toric_spines_sim.simulation import TSSimulator
from toric_spines_sim.viz import (
    create_simulation_dash_app,
    prepare_simulation_dashboard_data,
)

from simulations.common.inputs import InputScenario
from simulations.common.model import ModelConfig
from simulations.common.utils import build_synapse_colors_by_axon

logger = logging.getLogger(__name__)


def run_axons_dash(
    config: ModelConfig,
    scenario: InputScenario,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    fps: int = 60,
    stride: int = 1,
    max_clientside_frames: int = 5000,
    use_cache: bool = False,
) -> None:
    """Run one scenario on ``config`` and launch the Dash voltage dashboard."""
    logger.info("Scenario: %s", scenario.metadata)

    swc_filepath = config.swc_path()
    synpts_filepath = config.synpts_path()

    sim = TSSimulator(
        swc_filepath,
        synpts_filepath,
        scenario.events_tsgroup,
        scenario.parameters,
        record_points="all",
    )
    results = sim.run()

    synapse_colors = build_synapse_colors_by_axon(
        scenario.axon_synapses,
        len(results.synapses),
    )

    scenario_label = str(scenario.metadata.get("input_scenario", "scenario"))
    active_axons = scenario.metadata.get("active_axons", [])
    axon_suffix = "_".join(str(a) for a in sorted(active_axons)) or "all"
    title = (
        f"{config.stem} vary axons ({scenario_label}, axons {axon_suffix})"
    )

    dashboard_data = prepare_simulation_dashboard_data(
        results,
        swc_filepath,
        scenario.parameters,
        axon_synapses=scenario.axon_synapses,
        synapse_colors=synapse_colors,
        animation_kwargs=dict(
            stride=stride,
            opacity=0.8,
            flatshading=True,
            show_axes=False,
            show_synapses=True,
            synapse_ball_size=0.12,
            synapse_stacks=3,
            synapse_slices=6,
            synapse_flash_duration_ms=scenario.synapse_flash_duration_ms,
            colorbar_title="Voltage (mV)",
            frustum_sides=8,
        ),
    )
    dashboard_data.metadata.update(scenario.metadata)

    cache_dir = str(config.results_dir()) if use_cache else None
    app = create_simulation_dash_app(
        dashboard_data,
        fps=fps,
        title=title,
        cache_dir=cache_dir,
        clientside_max_frames=max_clientside_frames,
    )
    logger.info("Launching dashboard at http://%s:%s", host, port)
    app.run(host=host, port=port, debug=debug)
