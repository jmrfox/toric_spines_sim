"""TS76 axon PDF study.

Inputs
    data/swc/microns/TS76_wsink_r10um.swc
    data/pointsets/microns/TS76_synpts.txt
    data/ts_axons/ts76_axons.txt

Run from the repository root::

    uv run python -m simulations.ts76.axons

Tweak parameters in ``simulations/ts76/params.py`` and knobs in
``simulations/ts76/inputs.py``.
"""

import logging

from simulations.common.axons_report import run_axons_study
from simulations.ts76.inputs import MODEL

logging.basicConfig(level=logging.INFO)


def main() -> None:
    run_axons_study(MODEL)


if __name__ == "__main__":
    main()
