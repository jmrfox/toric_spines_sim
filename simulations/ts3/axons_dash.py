"""TS3 axon-input Dash dashboard (3D animation + traces + raster).

Run from the repository root::

    uv run python -m simulations.ts3.axons_dash
"""

from __future__ import annotations

import logging

from simulations.common.axons_dash import run_axons_dash
from simulations.ts3.inputs import (
    MODEL,
    axon_periodic_input,
    axon_poisson_input,
    sequential_pulse_input,
    simultaneous_pulse_input,
    synapse_poisson_input,
)

logging.basicConfig(level=logging.INFO)

# Input scenario: set the function (see simulations/ts3/inputs.py).
# SCENARIO = simultaneous_pulse_input()
SCENARIO = sequential_pulse_input()
# SCENARIO = synapse_poisson_input()
# SCENARIO = axon_poisson_input()
# SCENARIO = axon_periodic_input()

HOST = "127.0.0.1"
PORT = 8050
DEBUG = False
FPS = 60
STRIDE = 1
MAX_CLIENTSIDE_FRAMES = 5000
USE_CACHE = False


def main() -> None:
    run_axons_dash(
        MODEL,
        SCENARIO,
        host=HOST,
        port=PORT,
        debug=DEBUG,
        fps=FPS,
        stride=STRIDE,
        max_clientside_frames=MAX_CLIENTSIDE_FRAMES,
        use_cache=USE_CACHE,
    )


if __name__ == "__main__":
    main()
