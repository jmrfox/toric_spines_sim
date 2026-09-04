"""TS1 axon-input HTML voltage animation.

Run from the repository root::

    uv run python -m simulations.ts1.axons_animation
"""

import logging
from typing import Literal

from simulations.common.axons_animation import run_axons_animation
from simulations.ts1.inputs import (
    MODEL,
    axon_periodic_input,
    simultaneous_pulse_input,
)

logging.basicConfig(level=logging.INFO)

INPUT_MODE: Literal["pulse", "periodic"] = "pulse"

TEMP_K = 290.0
HH_SCALE = 2.0  # multiplier; set 0.0 to turn off HH
HH_TAGS = [6]
DT_RECORD_MS = 0.25

if INPUT_MODE == "pulse":
    scenario = simultaneous_pulse_input()
else:
    scenario = axon_periodic_input()


def main() -> None:
    run_axons_animation(
        MODEL,
        scenario,
        temp_k=TEMP_K,
        hh_scale=HH_SCALE,
        hh_tags=HH_TAGS,
        dt_record_ms=DT_RECORD_MS,
    )


if __name__ == "__main__":
    main()
