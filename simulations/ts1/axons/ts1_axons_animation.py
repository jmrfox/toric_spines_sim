"""
TS1 Axon Variation Animation

Run a single axon-configuration simulation and produce an interactive HTML
voltage animation.
"""

import logging
from typing import Literal

from toric_spines_sim.paths import (
    get_pointset_path,
    get_simulation_path,
    get_swc_path,
)
from toric_spines_sim.simulation import TSSimulator
from toric_spines_sim.viz import Animation

from simulations.ts1.ts1_inputs import axon_periodic_input, simultaneous_pulse_input
from simulations.ts1.ts1_utils import build_synapse_colors_by_axon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================

# Input scenario owns its own input/temporal config (see ts1_inputs.py).
INPUT_MODE: Literal["pulse", "periodic"] = "pulse"

# Model / physics overrides applied onto the scenario's sampled parameters.
TEMP_K = 290.0
HH_SCALE = 2.0  # multiplier; set 0.0 to turn off HH
HH_TAGS = [6]
HH_ON = HH_SCALE > 0.0
DT_RECORD_MS = 0.25

# =============================================================================

if INPUT_MODE == "pulse":
    scenario = simultaneous_pulse_input()
else:
    scenario = axon_periodic_input()
logger.info("Scenario: %s", scenario.metadata)

parameters = scenario.parameters
parameters["temp_K"] = TEMP_K
parameters["dt_record_ms"] = DT_RECORD_MS
if HH_ON:
    parameters["hh_tags"] = HH_TAGS
    parameters["K_gbar_S_per_cm2"] = 0.036 * HH_SCALE
    parameters["Na_gbar_S_per_cm2"] = 0.12 * HH_SCALE
else:
    parameters["hh_tags"] = []

# File paths and output name
swc_filepath = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_filepath = get_pointset_path("TS1_synpts.txt", units="microns")

active_axons = scenario.metadata.get("active_axons", [])
axon_suffix = "_".join(str(a) for a in sorted(active_axons)) or "all"
mode_suffix = str(scenario.metadata.get("input_scenario", INPUT_MODE))
hh_suffix = "_hh" if HH_ON else ""
results_path = get_simulation_path("ts1", "axons", "results")
animation_file = (
    results_path
    / f"TS1_axons_anim_{mode_suffix}_ax{axon_suffix}{hh_suffix}.html"
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
