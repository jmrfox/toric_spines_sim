#!/usr/bin/env python3
"""Skeletonize one or more TS meshes with pymcfs.

Uses the same defaults as pymcfs ``toric_spines/scripts/batch_ts_skeletonize.py``:
``profile=\"auto\"``, ``branching=\"sparse\"``, tip extension on, 500 iters,
300s timeout. Requires the optional mesh extra::

    uv sync --extra mesh

Examples::

    uv run python scripts/skeletonize_meshes.py TS1.obj
    uv run python scripts/skeletonize_meshes.py --all
    uv run python scripts/skeletonize_meshes.py TS1.obj TS2.obj --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

from toric_spines_sim.geometry.mesh_pipeline import (
    TORIC_SPINES_SKELETONIZE_DEFAULTS,
    default_polylines_path,
    resolve_mesh_targets,
    skeletonize_mesh,
)

logger = logging.getLogger(__name__)


def make_parser() -> argparse.ArgumentParser:
    defaults = TORIC_SPINES_SKELETONIZE_DEFAULTS
    parser = argparse.ArgumentParser(
        description=(
            "Skeletonize closed TS meshes with pymcfs → data/skeletons/ "
            "(toric_spines batch defaults)."
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
        help="Skeletonize every TS*.obj under data/mesh/",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=str(defaults["profile"]),
        help='pymcfs profile (default: auto = sparse oracle for TS meshes)',
    )
    parser.add_argument(
        "--branching",
        type=str,
        default=str(defaults["branching"]),
        choices=("sparse", "balanced", "dense"),
        help='branching preference for profile="auto" (default: sparse)',
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(defaults["max_iterations"]),
        help="contraction iteration cap (default: 500)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(defaults["timeout_seconds"]),
        help="contraction timeout seconds; <=0 disables (default: 300)",
    )
    parser.add_argument(
        "--max-vertex-growth",
        type=float,
        default=float(defaults["max_vertex_growth"]),
        help="abort remesh if n > this * n0 (default: 4)",
    )
    parser.add_argument(
        "--no-extend-tips",
        action="store_true",
        help="Disable tip extension (batch default is on)",
    )
    parser.add_argument(
        "--tip-extend-scale",
        type=float,
        default=float(defaults["tip_extend_scale"]),
        help="Max tip travel as multiple of bbox diagonal (default: 1.0)",
    )
    parser.add_argument(
        "--no-thick-hubs",
        action="store_true",
        help="Disable thick-hub principal-branch prune",
    )
    parser.add_argument(
        "--keep-hub-branches",
        type=int,
        default=int(defaults["keep_hub_branches"]),
        help="Branches to keep at thick hubs (default: 2)",
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

    timeout = None if args.timeout <= 0 else float(args.timeout)

    failures = 0
    for mesh_path in targets:
        polylines_path = default_polylines_path(mesh_path)
        try:
            written = skeletonize_mesh(
                mesh_path,
                polylines_path,
                profile=args.profile,
                branching=args.branching,
                max_iterations=args.max_iterations,
                timeout_seconds=timeout,
                max_vertex_growth=args.max_vertex_growth,
                extend_tips=not args.no_extend_tips,
                tip_extend_scale=args.tip_extend_scale,
                prune_thick_hubs=not args.no_thick_hubs,
                keep_hub_branches=args.keep_hub_branches,
            )
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            failures += 1
            logger.exception("Failed to skeletonize %s: %s", mesh_path, exc)
            print(f"error: failed {mesh_path.name}: {exc}", file=sys.stderr)
            continue
        print(f"mesh: {mesh_path}")
        print(f"polylines: {written}")

    if failures:
        print(f"error: {failures} of {len(targets)} mesh(es) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
