"""Shared HTML voltage-animation runner for axon-mapped simulations."""

from __future__ import annotations

import logging
from pathlib import Path

from toric_spines_sim.model import check_catalogue
from toric_spines_sim.simulation import TSSimulator
from toric_spines_sim.viz import Animation

from simulations.common.inputs import InputScenario
from simulations.common.model import ModelConfig
from simulations.common.utils import build_synapse_colors_by_axon

logger = logging.getLogger(__name__)


def run_axons_animation(
    config: ModelConfig,
    scenario: InputScenario,
    *,
    temp_k: float | None = None,
    hh_scale: float | None = None,
    hh_tags: list[int] | None = None,
    dt_record_ms: float | None = None,
) -> Path:
    """Run one scenario on ``config`` and write an HTML voltage animation."""
    check_catalogue()
    logger.info("Scenario: %s", scenario.metadata)

    parameters = scenario.parameters
    if temp_k is not None:
        parameters["temp_K"] = temp_k
    if dt_record_ms is not None:
        parameters["dt_record_ms"] = dt_record_ms
    if hh_scale is not None:
        if hh_scale > 0.0:
            parameters["hh_tags"] = list(hh_tags or [6])
            parameters["K_gbar_S_per_cm2"] = 0.036 * hh_scale
            parameters["Na_gbar_S_per_cm2"] = 0.12 * hh_scale
        else:
            parameters["hh_tags"] = []

    swc_filepath = config.swc_path()
    synpts_filepath = config.synpts_path()

    active_axons = scenario.metadata.get("active_axons", [])
    axon_suffix = "_".join(str(a) for a in sorted(active_axons)) or "all"
    mode_suffix = str(scenario.metadata.get("input_scenario", "scenario"))
    hh_on = hh_scale is not None and hh_scale > 0.0
    hh_suffix = "_hh" if hh_on else ""
    results_path = config.results_dir()
    results_path.mkdir(parents=True, exist_ok=True)
    animation_file = (
        results_path
        / (
            f"{config.spine_id}_axons_anim_{mode_suffix}_ax{axon_suffix}"
            f"{hh_suffix}.html"
        )
    )

    sim = TSSimulator(
        swc_filepath,
        synpts_filepath,
        scenario.events_tsgroup,
        parameters,
        record_points="all",
    )
    results = sim.run()

    synapse_colors = build_synapse_colors_by_axon(
        scenario.axon_synapses,
        len(results.synapses),
        muted_kwargs={"value_scale": 0.35, "saturation_scale": 0.7},
        bright_kwargs={
            "saturation_scale": 1.1,
            "value_scale": 1.15,
            "value_offset": 0.25,
        },
    )

    Animation(results, swc_filepath=swc_filepath).create(
        output_path=animation_file,
        colorscale="Plasma",
        fps=10,
        stride=1,
        auto_open=False,
        clim=None,
        opacity=0.8,
        flatshading=True,
        radius_scale=1.0,
        title=None,
        colorbar_title="Voltage (mV)",
        show_axes=True,
        show_synapses=True,
        synapse_ball_size=0.07,
        synapse_flash_duration_ms=scenario.synapse_flash_duration_ms,
        synapse_colors=synapse_colors,
    )
    logger.info("Animation saved to %s", animation_file)
    return animation_file
