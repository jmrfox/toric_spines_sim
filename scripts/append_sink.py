#!/usr/bin/env python3
"""Append a cylindrical sink to toric-spine SWCs (pixels and microns).

Pixel-space spines live under ``data/swc/pixels/TS{n}.swc``. This script
appends a sink using ``data/pointsets/pixels/<stem>_neckpoint.txt`` when
present (else the SWC root) and writes:

    data/swc/pixels/<stem>_wsink_r<R>um.swc
    data/swc/microns/<stem>_wsink_r<R>um.swc
    data/pointsets/microns/<stem>_neckpoint.txt

Sink dimensions are specified in microns (converted to pixels for attachment).
Sink axis is the optimal direction away from the morphology.

Examples::

    uv run python scripts/append_sink.py --all
    uv run python scripts/append_sink.py TS1 TS2
    uv run python scripts/append_sink.py TS1.swc --radius-um 20
"""

from __future__ import annotations

import argparse
import logging
import sys

from toric_spines_sim.geometry.prepare import (
    DEFAULT_SINK_CONNECTOR_LENGTH_UM,
    DEFAULT_SINK_N_CYLINDERS,
    DEFAULT_SINK_RADIUS_UM,
    DEFAULT_SINK_TAG,
    DEFAULT_SINK_TIP_TAG,
    append_sink_write,
    resolve_swc_targets,
)
from toric_spines_sim.paths import UM_PER_PX

logger = logging.getLogger(__name__)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "swcs",
        nargs="*",
        help="SWC path(s) or bare stems under data/swc/pixels/ (e.g. TS1)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_swcs",
        help="Process every TS{n}.swc under data/swc/pixels/",
    )
    parser.add_argument(
        "--radius-um",
        type=float,
        default=DEFAULT_SINK_RADIUS_UM,
        help=f"Sink cylinder radius in µm (default: {DEFAULT_SINK_RADIUS_UM:g})",
    )
    parser.add_argument(
        "--connector-length-um",
        type=float,
        default=DEFAULT_SINK_CONNECTOR_LENGTH_UM,
        help=(
            "Neck-to-sink connector length in µm "
            f"(default: {DEFAULT_SINK_CONNECTOR_LENGTH_UM:g})"
        ),
    )
    parser.add_argument(
        "--n-cylinders",
        type=int,
        default=DEFAULT_SINK_N_CYLINDERS,
        help=f"Number of sink cylinders (default: {DEFAULT_SINK_N_CYLINDERS})",
    )
    parser.add_argument(
        "--tag",
        type=int,
        default=DEFAULT_SINK_TAG,
        help=f"SWC tag for sink nodes (default: {DEFAULT_SINK_TAG})",
    )
    parser.add_argument(
        "--tip-tag",
        type=int,
        default=DEFAULT_SINK_TIP_TAG,
        help=f"SWC tag for the distal sink tip (default: {DEFAULT_SINK_TIP_TAG})",
    )
    parser.add_argument(
        "--um-per-px",
        type=float,
        default=UM_PER_PX,
        help=f"Microns per pixel (default: {UM_PER_PX:g})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        targets = resolve_swc_targets(args.swcs, all_swcs=args.all_swcs)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, ValueError) else 1

    failures = 0
    for swc_px in targets:
        try:
            written_px, written_um = append_sink_write(
                swc_px,
                radius_um=args.radius_um,
                connector_length_um=args.connector_length_um,
                n_cylinders=args.n_cylinders,
                um_per_px=args.um_per_px,
                tag=args.tag,
                last_segment_tag=args.tip_tag,
            )
        except Exception as exc:
            failures += 1
            logger.exception("Failed to append sink for %s: %s", swc_px, exc)
            print(f"error: failed {swc_px.name}: {exc}", file=sys.stderr)
            continue
        print(f"swc_in: {swc_px}")
        print(f"swc_out_px: {written_px}")
        print(f"swc_out_um: {written_um}")

    if failures:
        print(f"error: {failures} of {len(targets)} SWC(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
