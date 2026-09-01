#!/usr/bin/env python3
"""Fit SWC cable models from mesh + skeleton with mascaf.

Requires existing polylines (see ``scripts/skeletonize_meshes.py``) and the
optional mesh extra::

    uv sync --extra mesh

Examples::

    uv run python scripts/fit_swc.py TS1.obj
    uv run python scripts/fit_swc.py --all
    uv run python scripts/fit_swc.py TS1.obj --basis-optimize --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from toric_spines_sim.geometry.mesh_pipeline import (
    default_polylines_path,
    default_swc_path,
    fit_swc,
    resolve_mesh_targets,
)

logger = logging.getLogger(__name__)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit SWC from mesh + skeleton with mascaf → data/swc/pixels/."
        ),
    )
    parser.add_argument(
        "meshes",
        nargs="*",
        help="Mesh path(s) or bare filenames under data/mesh/ (e.g. TS1.obj)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_meshes",
        help="Fit every TS*.obj under data/mesh/ that has a default skeleton",
    )
    parser.add_argument(
        "--polylines",
        type=str,
        default=None,
        help=(
            "Override polylines path (only valid with a single mesh argument; "
            "default: data/skeletons/<stem>.polylines.txt)"
        ),
    )
    parser.add_argument(
        "--swc",
        type=str,
        default=None,
        help=(
            "Override SWC output path (only valid with a single mesh argument; "
            "default: data/swc/pixels/<stem>.swc)"
        ),
    )
    parser.add_argument(
        "--max-edge-length-frac",
        type=float,
        default=0.08,
        help=(
            "FitOptions.max_edge_length as a fraction of each mesh's "
            "bounding-box diagonal (default: 0.08)"
        ),
    )
    parser.add_argument(
        "--radius-strategy",
        type=str,
        default="equivalent_area",
        help="mascaf radius strategy (default: equivalent_area)",
    )
    parser.add_argument(
        "--no-scale-radii",
        action="store_true",
        help="Skip scale_radii_to_match_mesh after fitting",
    )
    parser.add_argument(
        "--basis-optimize",
        action="store_true",
        help="Enable mascaf BasisOptimizer during fitting",
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
        targets = resolve_mesh_targets(args.meshes, all_meshes=args.all_meshes)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, ValueError) else 1

    if args.polylines is not None and len(targets) != 1:
        print(
            "error: --polylines is only valid with a single mesh target",
            file=sys.stderr,
        )
        return 2
    if args.swc is not None and len(targets) != 1:
        print(
            "error: --swc is only valid with a single mesh target",
            file=sys.stderr,
        )
        return 2

    failures = 0
    for mesh_path in targets:
        polylines_path = (
            Path(args.polylines)
            if args.polylines is not None
            else default_polylines_path(mesh_path)
        )
        swc_path = (
            Path(args.swc) if args.swc is not None else default_swc_path(mesh_path)
        )
        try:
            written = fit_swc(
                mesh_path,
                polylines_path,
                swc_path,
                max_edge_length_frac=args.max_edge_length_frac,
                radius_strategy=args.radius_strategy,
                scale_radii=not args.no_scale_radii,
                basis_optimize=args.basis_optimize,
            )
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except FileNotFoundError as exc:
            failures += 1
            logger.error("%s", exc)
            print(f"error: {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            failures += 1
            logger.exception("Failed to fit SWC for %s: %s", mesh_path, exc)
            print(f"error: failed {mesh_path.name}: {exc}", file=sys.stderr)
            continue
        print(f"mesh: {mesh_path}")
        print(f"polylines: {polylines_path}")
        print(f"swc: {written}")

    if failures:
        print(f"error: {failures} of {len(targets)} mesh(es) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
