"""Axon-input Dash dashboard (3D animation + traces + raster) for one spine.

Run from the repository root::

    uv run python -m simulations.axons_dash TS1
    uv run python -m simulations.axons_dash TS76 --scenario synapse-poisson --port 8051

Scenarios: ``simulations/common/scenarios.py``.
"""

from __future__ import annotations

import argparse
import logging

from simulations.common.axons_dash import run_axons_dash
from simulations.common.scenarios import SCENARIOS
from simulations.common.spines import model_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Axon-input Dash dashboard.")
    ap.add_argument("spine")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="sequential")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-clientside-frames", type=int, default=5000)
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = model_config(args.spine)
    run_axons_dash(
        cfg, SCENARIOS[args.scenario](cfg), host=args.host, port=args.port, debug=args.debug,
        fps=args.fps, stride=args.stride, max_clientside_frames=args.max_clientside_frames,
        use_cache=args.use_cache,
    )


if __name__ == "__main__":
    main()
