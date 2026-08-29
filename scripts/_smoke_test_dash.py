"""Smoke test for the refactored/responsive Dash app path."""

from __future__ import annotations

import logging

from toric_spines_sim.paths import (
    get_pointset_path,
    get_swc_path,
)
from toric_spines_sim.simulation import TSSimulator
from toric_spines_sim.viz import (
    create_simulation_dash_app,
    prepare_simulation_dashboard_data,
)

from ts1_inputs import simultaneous_pulse_input
from ts1_utils import build_synapse_colors_by_axon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    scenario = simultaneous_pulse_input()
    parameters = scenario.parameters
    parameters["temp_K"] = 290.0
    parameters["dt_record_ms"] = 0.25
    parameters["hh_tags"] = [6]
    parameters["K_gbar_S_per_cm2"] = 0.036 * 2.0
    parameters["Na_gbar_S_per_cm2"] = 0.12 * 2.0

    swc_filepath = get_swc_path("TS1_wsink_r10um.swc", units="microns")
    synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")

    sim = TSSimulator(
        swc_filepath,
        synpts_filepath,
        scenario.events_tsgroup,
        parameters,
        record_points="all",
    )
    results = sim.run()
    logger.info("Simulation complete: %s", results.voltage_traces.shape)

    synapse_colors = build_synapse_colors_by_axon(
        scenario.axon_synapses,
        len(results.synapses),
        muted_kwargs={"value_scale": 0.55, "saturation_scale": 0.85},
        bright_kwargs={
            "saturation_scale": 1.25,
            "value_scale": 1.25,
            "value_offset": 0.4,
        },
    )

    dashboard_data = prepare_simulation_dashboard_data(
        results,
        swc_filepath,
        parameters,
        axon_synapses=scenario.axon_synapses,
        synapse_colors=synapse_colors,
        animation_kwargs=dict(
            stride=1,
            opacity=0.8,
            flatshading=True,
            show_axes=False,
            show_synapses=True,
            synapse_ball_size=0.07,
            synapse_flash_duration_ms=1.0,
            colorbar_title="Voltage (mV)",
        ),
    )

    logger.info(
        "Dashboard data ready: %d frames, %d synapses",
        dashboard_data.n_frames,
        dashboard_data.metadata.get("n_synapses", 0),
    )

    app = create_simulation_dash_app(
        dashboard_data,
        fps=10,
        title="Smoke test",
        cache_dir=None,
        clientside_max_frames=600,
    )
    logger.info("App created: %s", type(app).__name__)

    # Verify layout contains expected stores and graphs.
    def _collect_ids(component) -> set[str]:
        ids: set[str] = set()
        if hasattr(component, "id") and component.id:
            ids.add(component.id)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, (list, tuple)):
                children = [children]
            for child in children:
                if hasattr(child, "id") or hasattr(child, "children"):
                    ids.update(_collect_ids(child))
        return ids

    layout_ids = _collect_ids(app.layout)
    required_ids = (
        "graph-3d",
        "graph-voltage",
        "graph-raster",
        "time-slider",
        "frame-idx",
        "frame-bundle",
        "voltage-trace-map",
    )
    for required in required_ids:
        assert required in layout_ids, f"Missing layout id: {required}"
    logger.info("Layout OK")

    # Verify the cache initial figures are valid figure dicts.
    from dash import Patch

    from toric_spines_sim.viz.simulation_dash import load_or_build_playback_cache

    cache = load_or_build_playback_cache(
        dashboard_data, "light", None, 600
    )
    assert isinstance(cache.template_3d, dict), (
        "3D template should be a figure dict"
    )
    assert isinstance(cache.raster_figure, dict), (
        "Raster figure should be a dict"
    )
    assert isinstance(
        cache.voltage_at(0, ["Spine", "Neck", "Sink"]), dict
    ), "Voltage figure should be a dict"
    assert isinstance(cache.raster_at(0), Patch), (
        "Raster patch should be a Patch"
    )
    assert cache.clientside_bundle is not None, "Expected clientside bundle"
    assert cache.clientside_bundle["n_frames"] == dashboard_data.n_frames
    logger.info(
        "Clientside bundle OK: %d frames",
        cache.clientside_bundle["n_frames"],
    )

    # Force server fallback by setting the clientside frame budget to 0.
    logger.info("Testing server fallback path...")
    fallback_app = create_simulation_dash_app(
        dashboard_data,
        fps=10,
        title="Smoke test fallback",
        cache_dir=None,
        clientside_max_frames=0,
    )
    fallback_cache = load_or_build_playback_cache(
        dashboard_data, "light", None, 0
    )
    assert fallback_cache.clientside_bundle is None, "Expected no bundle"
    logger.info("Fallback cache OK (clientside disabled)")
    logger.info("Fallback app created: %s", type(fallback_app).__name__)

    logger.info("Smoke test PASSED")


if __name__ == "__main__":
    main()
