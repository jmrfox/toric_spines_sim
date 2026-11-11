"""Axon PDF study (pairwise, sequential, synchrony) for one or more spines.

Run from the repository root::

    uv run python -m simulations.axons TS1
    uv run python -m simulations.axons TS1 TS3 TS76
    uv run python -m simulations.axons all

Per-spine settings: ``simulations/common/spines.py``. Reports go to
``simulations/ts{id}/axons/results/``.
"""

from __future__ import annotations

import argparse
import logging

from simulations.common.axons_report import run_axons_study
from simulations.common.spines import AXON_SPINES, model_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Axon PDF study for toric spines.")
    ap.add_argument("spines", nargs="+", help=f"spine ids or 'all' ({', '.join(AXON_SPINES)})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    spines = AXON_SPINES if [s.lower() for s in args.spines] == ["all"] else args.spines
    for spine in spines:
        run_axons_study(model_config(spine))


if __name__ == "__main__":
    main()
