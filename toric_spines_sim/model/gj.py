"""
Gap junction definitions and preparation utilities.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from jscip import ParameterSet

from toric_spines_sim.geometry.swc import parse_cycle_breaks, read_swc_points
from toric_spines_sim.utils import equal_vectors


@dataclass
class GapJunctionPoint:
    index_pair: Tuple[int, int]
    location: Tuple[float, float, float]
    weight: float


def prepare_gap_junctions(
    swc_file: Path,
    parameters: ParameterSet,
):
    """
    Prepare gap junctions from an SWC file with reconnect annotations.
    """
    weight = parameters["gj_weight"]
    reconnect_pairs = parse_cycle_breaks(
        swc_file
    )  # gives the list of indices of nodes that should be connected by gap junctions
    gap_junctions = {}
    points_by_id = read_swc_points(
        swc_file
    )  # gives the dict of node indices to their 3D locations
    logger.debug(
        "Preparing gap junctions for %d reconnect pairs from %s, weight=%f",
        len(reconnect_pairs),
        swc_file,
        weight,
    )
    for n, (i, j) in enumerate(reconnect_pairs):
        if i not in points_by_id or j not in points_by_id:
            logger.error("Node %s or %s not found in SWC file %s", i, j, swc_file)
            raise ValueError(f"Node {i} or {j} not found in SWC file")
        xi, yi, zi, ri = points_by_id[i]
        xj, yj, zj, rj = points_by_id[j]
        if not equal_vectors((xi, yi, zi), (xj, yj, zj)):
            logger.error(
                "Nodes %s and %s have different locations: (%s,%s,%s) vs (%s,%s,%s)",
                i,
                j,
                xi,
                yi,
                zi,
                xj,
                yj,
                zj,
            )
            raise ValueError(f"Nodes {i} and {j} have different locations")
        gap_junctions[f"gj_{n}"] = GapJunctionPoint((i, j), (xi, yi, zi), weight)
    logger.debug("Prepared %d gap junctions from %s weight=%f", len(gap_junctions), swc_file, weight)
    return gap_junctions
