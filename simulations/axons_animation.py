"""Axon-input HTML voltage animation for one spine.

Run from the repository root::

    uv run python -m simulations.axons_animation TS1
    uv run python -m simulations.axons_animation TS1 --scenario periodic --hh-scale 0

Defaults match the former per-spine scripts: simultaneous pulse, HH on the
sink tip (tag 6) at 2x, 290 K, recording every 0.25 ms.
"""

from __future__ import annotations

import argparse
import logging

from simulations.common.axons_animation import run_axons_animation
from simulations.common.scenarios import SCENARIOS
from simulations.common.spines import model_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Axon-input HTML voltage animation.")
    ap.add_argument("spine")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="simultaneous")
    ap.add_argument("--temp-k", type=float, default=290.0)
    ap.add_argument("--hh-scale", type=float, default=2.0, help="0 turns HH off")
    ap.add_argument("--hh-tags", type=int, nargs="*", default=[6])
    ap.add_argument("--dt-record-ms", type=float, default=0.25)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = model_config(args.spine)
    run_axons_animation(cfg, SCENARIOS[args.scenario](cfg), temp_k=args.temp_k,
                        hh_scale=args.hh_scale, hh_tags=args.hh_tags, dt_record_ms=args.dt_record_ms)


if __name__ == "__main__":
    main()
