"""TS67 axon PDF study.

Inputs
    data/swc/microns/TS67_wsink_r10um.swc
    data/pointsets/microns/TS67_synpts.txt
    data/ts_axons/ts67_axons.txt

Run from the repository root::

    uv run python -m simulations.ts67.axons

Tweak parameters in ``simulations/ts67/params.py`` and options in
``simulations/ts67/inputs.py``.
"""

import logging

from simulations.common.axons_report import run_axons_study
from simulations.ts67.inputs import MODEL

logging.basicConfig(level=logging.INFO)


def main() -> None:
    run_axons_study(MODEL)


if __name__ == "__main__":
    main()
