"""TS4 axon PDF study.

Inputs
    data/swc/microns/TS4_wsink_r10um.swc
    data/pointsets/microns/TS4_synpts.txt
    data/ts_axons/ts4_axons.txt

Run from the repository root::

    uv run python -m simulations.ts4.axons

Tweak parameters in ``simulations/ts4/params.py`` and knobs in
``simulations/ts4/inputs.py``.
"""

import logging

from simulations.common.axons_report import run_axons_study
from simulations.ts4.inputs import MODEL

logging.basicConfig(level=logging.INFO)


def main() -> None:
    run_axons_study(MODEL)


if __name__ == "__main__":
    main()
