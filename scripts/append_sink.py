#!/usr/bin/env python3
"""Append a simple cylindrical "sink" geometry to an SWC file.

This script:
- Loads an input SWC (the "spine" model).
- Builds a straight-line chain of one or more cylinders (a "sink") starting from a
  specified trunk/connection point.
- Appends the sink as a separate tree to the SWC text and, by default, writes a
  header annotation `# CYCLE_BREAK reconnect i j` that tools in this repo use to
  place gap junction pairs at runtime.
- Writes a new SWC file.

Notes
-----
- SWC does not encode gap junctions. The `# CYCLE_BREAK reconnect` comment enables
  downstream code (see `parse_cycle_breaks` in `toric_spines_sim/model.py`) to
  infer junction placements and create Arbor gap junction connections.
- This script keeps geometry simple: the sink is colinear along one axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from toric_spines_sim.geometry.sink import SinkGeometry, append_sink_to_swc


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("swc_in", type=str, help="Path to input SWC file")
    p.add_argument(
        "--swc-out",
        type=str,
        default=None,
        help="Path to output SWC (default: <swc_in stem>_with_sink.swc)",
    )

    # Trunk point options
    g = p.add_argument_group("Trunk point (connection site)")
    g.add_argument(
        "--trunk-file",
        type=str,
        default=None,
        help="Path to a text file with three numbers: x y z (whitespace-separated)",
    )
    g.add_argument(
        "--trunk-x", type=float, default=None, help="Trunk X coordinate (µm)"
    )
    g.add_argument(
        "--trunk-y", type=float, default=None, help="Trunk Y coordinate (µm)"
    )
    g.add_argument(
        "--trunk-z", type=float, default=None, help="Trunk Z coordinate (µm)"
    )

    # Sink geometry
    p.add_argument(
        "--radius-um", type=float, default=0.5, help="Sink cylinder radius [µm]"
    )
    p.add_argument(
        "--length-um", type=float, default=100.0, help="Total sink length [µm]"
    )
    p.add_argument(
        "--n-cylinders", type=int, default=1, help="Number of cylinders (segments)"
    )
    p.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="x",
        help="Axis along which to extend the sink",
    )

    # Options
    p.add_argument(
        "--node-type",
        type=int,
        default=3,
        help="SWC node type tag to use for sink nodes (default: 3, dendrite)",
    )
    p.add_argument(
        "--no-gj-annotation",
        action="store_true",
        help="Do not add '# CYCLE_BREAK reconnect' annotation comment to the output",
    )

    return p


def resolve_trunk_arg(args: argparse.Namespace):
    if args.trunk_file:
        return args.trunk_file
    if (
        args.trunk_x is not None
        and args.trunk_y is not None
        and args.trunk_z is not None
    ):
        return (args.trunk_x, args.trunk_y, args.trunk_z)
    raise ValueError(
        "Provide either --trunk-file or all of --trunk-x/--trunk-y/--trunk-z"
    )


def run(args: argparse.Namespace):
    swc_in = Path(args.swc_in).resolve()
    if not swc_in.exists():
        raise FileNotFoundError(f"Input SWC not found: {swc_in}")

    if args.swc_out is None:
        swc_out = swc_in.with_name(f"{swc_in.stem}_with_sink.swc")
    else:
        swc_out = Path(args.swc_out).resolve()

    trunk_point = resolve_trunk_arg(args)

    geom = SinkGeometry(
        radius=args.radius_um,
        length=args.length_um,
        n_cylinders=args.n_cylinders,
        axis=args.axis,
    )

    out = append_sink_to_swc(
        swc_in=swc_in,
        swc_out=swc_out,
        trunk_point=trunk_point,
        geom=geom,
        node_type=args.node_type,
        add_gj_annotation=(not args.no_gj_annotation),
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    run(make_parser().parse_args())
