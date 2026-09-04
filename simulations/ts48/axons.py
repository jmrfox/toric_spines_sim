"""TS48 axon PDF study.

Inputs
    data/swc/microns/TS48_wsink_r10um.swc
    data/pointsets/microns/TS48_synpts.txt
    data/ts_axons/ts48_axons.txt

Run from the repository root::

    uv run python -m simulations.ts48.axons

Tweak parameters in ``simulations/ts48/params.py`` and knobs in
``simulations/ts48/inputs.py``.
"""

import logging

from simulations.common.axons_report import run_axons_study
from simulations.ts48.inputs import MODEL

logging.basicConfig(level=logging.INFO)


def main() -> None:
    run_axons_study(MODEL)


if __name__ == "__main__":
    main()
