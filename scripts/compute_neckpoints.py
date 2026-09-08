#!/usr/bin/env python3
"""Compute neckpoint(s) for toric-spine meshes using the full cell mesh.

Compares each isolated ``TS*.obj`` to ``cell_wrapped_simplified.obj`` and writes
pixel-space neckpoints to ``data/pointsets/pixels/<spine_id>_neckpoint.txt``.

Requires trimesh (``uv sync``).

Examples::

    uv run python scripts/compute_neckpoints.py TS1.obj
    uv run python scripts/compute_neckpoints.py TS1 TS2 --max-necks 1
    uv run python scripts/compute_neckpoints.py --all --max-necks 1
    uv run python scripts/compute_neckpoints.py TS3 --max-necks 3 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys

from toric_spines_sim.geometry.neckpoint import (
    NeckpointParams,
    compute_neck_points_for_spine,
)
from toric_spines_sim.geometry.mesh_pipeline import resolve_mesh_targets
from toric_spines_sim.paths import get_cell_mesh_path

logger = logging.getLogger(__name__)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meshes",
        nargs="*",
        help="Mesh path(s) or bare spine ids under data/mesh/ (e.g. TS1 or TS1.obj)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_meshes",
        help="Process every TS*.obj under data/mesh/",
    )
    parser.add_argument(
        "--cell-mesh",
        type=str,
        default=None,
        help=(
            "Full cell mesh path or filename under data/mesh/ "
            f"(default: {get_cell_mesh_path().name})"
        ),
    )
    parser.add_argument(
        "--max-necks",
        type=int,
        default=None,
        help="Keep only the N largest necks by cap area (default: keep all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log neckpoints without writing files",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip spines that already have a pixel neckpoint file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        targets = resolve_mesh_targets(args.meshes, all_meshes=args.all_meshes)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2

    if args.max_necks is not None and args.max_necks < 1:
        logger.error("--max-necks must be >= 1")
        return 2

    params = NeckpointParams(max_necks=args.max_necks)
    cell_mesh = args.cell_mesh
    n_ok = 0
    n_empty = 0
    for mesh_path in targets:
        points = compute_neck_points_for_spine(
            mesh_path,
            cell_mesh_path=cell_mesh,
            params=params,
            write=not args.dry_run,
            overwrite=not args.no_overwrite,
        )
        if points:
            n_ok += 1
            logger.info(
                "%s: %d neckpoint(s)%s",
                mesh_path.stem,
                len(points),
                " (dry-run)" if args.dry_run else "",
            )
        else:
            n_empty += 1
            logger.warning("%s: no neckpoints found", mesh_path.stem)

    logger.info("Done: %d with necks, %d empty (of %d)", n_ok, n_empty, len(targets))
    return 0 if n_empty == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
