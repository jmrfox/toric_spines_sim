"""TS2 axon PDF study.

Inputs
    data/swc/microns/TS2_wsink_r10um.swc
    data/pointsets/microns/TS2_synpts.txt
    data/ts_axons/ts2_axons.txt

Run from the repository root::

    uv run python -m simulations.ts2.axons

Tweak parameters in ``simulations/ts2/params.py`` and options in
``simulations/ts2/inputs.py``.
"""

import logging

from simulations.common.axons_report import run_axons_study
from simulations.ts2.inputs import MODEL

logging.basicConfig(level=logging.INFO)


def main() -> None:
    run_axons_study(MODEL)


if __name__ == "__main__":
    main()
