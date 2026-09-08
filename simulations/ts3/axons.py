"""TS3 axon PDF study.

Inputs
    data/swc/microns/TS3_wsink_r10um.swc
    data/pointsets/microns/TS3_synpts.txt
    data/ts_axons/ts3_axons.txt

Run from the repository root::

    uv run python -m simulations.ts3.axons

Tweak parameters in ``simulations/ts3/params.py`` and options in
``simulations/ts3/inputs.py``.
"""

import logging

from simulations.common.axons_report import run_axons_study
from simulations.ts3.inputs import MODEL

logging.basicConfig(level=logging.INFO)


def main() -> None:
    run_axons_study(MODEL)


if __name__ == "__main__":
    main()
