#!/usr/bin/env python3
"""CLI for mesh → skeleton (pymcfs) → SWC (mascaf).

Requires pymcfs and mascaf (``uv sync``).

Examples::

    uv run python scripts/mesh_to_swc.py TS1.obj
    uv run python scripts/mesh_to_swc.py TS1.obj --polylines-only
    uv run python scripts/mesh_to_swc.py TS1.obj --fit-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from toric_spines_sim.geometry.mesh_pipeline import (
    default_polylines_path,
    default_swc_path,
    mesh_to_swc,
    resolve_mesh_path,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Skeletonize a closed mesh with pymcfs and fit an SWC with mascaf."
        ),
    )
    parser.add_argument(
        "mesh",
        type=str,
        help="Mesh path or bare filename under data/mesh/ (e.g. TS1.obj)",
    )
    parser.add_argument(
        "--polylines",
        type=str,
        default=None,
        help="Output/input polylines path (default: data/skeletons/<spine_id>.polylines.txt)",
    )
    parser.add_argument(
        "--swc",
        type=str,
        default=None,
        help="Output SWC path (default: data/swc/pixels/<spine_id>.swc)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="robust",
        help="pymcfs skeletonization profile (default: robust)",
    )
    parser.add_argument(
        "--max-edge-length-frac",
        type=float,
        default=0.08,
        help=(
            "FitOptions.max_edge_length as a fraction of the mesh "
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
        "--polylines-only",
        action="store_true",
        help="Only run pymcfs skeletonization; skip SWC fitting",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Skip skeletonization; fit SWC from an existing polylines file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)

    if args.polylines_only and args.fit_only:
        print(
            "error: --polylines-only and --fit-only are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        mesh_path = resolve_mesh_path(args.mesh)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    polylines_path = (
        Path(args.polylines) if args.polylines else default_polylines_path(mesh_path)
    )
    swc_path = Path(args.swc) if args.swc else default_swc_path(mesh_path)

    try:
        result = mesh_to_swc(
            mesh_path,
            polylines_path=polylines_path,
            swc_path=swc_path,
            skip_skeletonize=args.fit_only,
            polylines_only=args.polylines_only,
            profile=args.profile,
            max_edge_length_frac=args.max_edge_length_frac,
            radius_strategy=args.radius_strategy,
            scale_radii=not args.no_scale_radii,
            basis_optimize=args.basis_optimize,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"mesh: {result.mesh_path}")
    print(f"polylines: {result.polylines_path}")
    if result.swc_path is not None:
        print(f"swc: {result.swc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
