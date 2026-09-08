"""TS1 axon PDF study.

Inputs
    data/swc/microns/TS1_wsink_r10um.swc
    data/pointsets/microns/TS1_synpts.txt
    data/ts_axons/ts1_axons.txt

Run from the repository root::

    uv run python -m simulations.ts1.axons

Tweak parameters in ``simulations/ts1/params.py`` and options in
``simulations/ts1/inputs.py``.
"""

import logging

from simulations.common.axons_report import run_axons_study
from simulations.ts1.inputs import MODEL

logging.basicConfig(level=logging.INFO)


def main() -> None:
    run_axons_study(MODEL)


if __name__ == "__main__":
    main()
