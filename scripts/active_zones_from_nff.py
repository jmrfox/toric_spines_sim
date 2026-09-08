#!/usr/bin/env python3
"""Convert NFF active-zone markers to pointsets, and write micron synpts.

Reads ``data/nff/<spine_id>_AZ.nff`` (pixel coordinates) and writes:

- ``data/pointsets/pixels/<spine_id>_AZ.txt`` — raw AZ XYZ
- ``data/pointsets/microns/<spine_id>_synpts.txt`` — AZ projected onto the matching
  ``data/swc/pixels/<spine_id>.swc``, scaled by the project conversion factor
  (5 nm/pixel)

NFF files with no ``s`` points produce an empty AZ file and skip synpts.

Examples::

    uv run python scripts/active_zones_from_nff.py
    uv run python scripts/active_zones_from_nff.py --no-synpts
"""

from __future__ import annotations

import argparse
import logging
import sys

from toric_spines_sim.geometry.prepare import (
    convert_nff_active_zone,
    nff_spine_id,
    write_synpts_microns,
)
from toric_spines_sim.paths import NFF_DIR, SWC_PIXELS_DIR
from toric_spines_sim.utils import read_nff_s_points

logger = logging.getLogger(__name__)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-synpts",
        action="store_true",
        help="Only write pixel AZ files; do not project/scale synpts",
    )
    parser.add_argument(
        "--um-per-px",
        type=float,
        default=0.005,
        help="Microns per pixel (5 nm/pixel; default: 0.005)",
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

    nff_paths = sorted(NFF_DIR.glob("*.nff"))
    if not nff_paths:
        print(f"error: no .nff files found under {NFF_DIR}", file=sys.stderr)
        return 1

    failures = 0
    for nff_path in nff_paths:
        try:
            az_path = convert_nff_active_zone(nff_path)
        except Exception as exc:
            failures += 1
            logger.exception("Failed to convert %s: %s", nff_path, exc)
            print(f"error: failed {nff_path.name}: {exc}", file=sys.stderr)
            continue

        n_points = len(read_nff_s_points(nff_path))
        print(f"nff: {nff_path}")
        print(f"az: {az_path} ({n_points} points)")

        if args.no_synpts:
            continue
        if n_points == 0:
            logger.warning("Skipping synpts for %s (no AZ points)", nff_path.name)
            continue

        spine_id = nff_spine_id(nff_path)
        swc_path_pixels = SWC_PIXELS_DIR / f"{spine_id}.swc"
        if not swc_path_pixels.is_file():
            logger.warning("No matching SWC %s; skipping synpts", swc_path_pixels)
            continue
        try:
            synpts_path = write_synpts_microns(
                swc_path_pixels, az_path, um_per_px=args.um_per_px
            )
        except Exception as exc:
            failures += 1
            logger.exception("Failed synpts for %s: %s", spine_id, exc)
            print(f"error: failed synpts {spine_id}: {exc}", file=sys.stderr)
            continue
        print(f"synpts: {synpts_path}")

    if failures:
        print(
            f"error: {failures} of {len(nff_paths)} NFF file(s) failed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
