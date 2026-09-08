"""Gap junction definitions and preparation utilities.

SWC is a directed tree, so mesh cycles and extra necks are restored
electrically: ``prepare_gap_junctions`` reads ``# CYCLE_BREAK reconnect i j``
and ``# MULTI_NECK reconnect i j`` headers and returns one ``GapJunctionPoint``
per pair. ``TSModel`` places the two junction labels at the Arbor locations
of those SWC samples (not a shared XYZ closest-point), then ``TSRecipe``
connects them.

Colocated sample pairs are expected for cycle breaks; extra-neck pairs may
sit at different coordinates. Both cases are valid.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from jscip import ParameterSet

from toric_spines_sim.geometry.swc import parse_reconnect_pairs, read_swc_points


@dataclass
class GapJunctionPoint:
    """One electrical reconnect between two SWC samples.

    Attributes
    ----------
    index_pair
        SWC node IDs ``(i, j)`` to join.
    location
        XYZ of node ``i`` (µm or px, matching the SWC). Display only;
        electrical endpoints are ``index_pair``.
    weight
        Arbor gap-junction weight (dimensionless conductance scale).
    location_b
        XYZ of node ``j`` when it differs from ``i``; otherwise ``None``.
        Display only: ``TSModel`` places the junction from ``index_pair``,
        not from these coordinates.
    """

    index_pair: Tuple[int, int]
    location: Tuple[float, float, float]
    weight: float
    location_b: Optional[Tuple[float, float, float]] = None


def prepare_gap_junctions(
    swc_file: Path,
    parameters: ParameterSet,
):
    """Build gap junctions from CYCLE_BREAK and MULTI_NECK reconnect headers.

    Returns
    -------
    dict[str, GapJunctionPoint]
        Labels ``gj_0``, ``gj_1``, … in header order.

    Examples
    --------
    >>> gjs = prepare_gap_junctions(swc_path, parameters)  # doctest: +SKIP
    >>> gjs["gj_0"].index_pair
    (12, 13)
    """
    weight = parameters["gj_weight"]
    reconnect_pairs = parse_reconnect_pairs(swc_file)
    gap_junctions = {}
    points_by_id = read_swc_points(swc_file)
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
        xi, yi, zi, _ri = points_by_id[i]
        xj, yj, zj, _rj = points_by_id[j]
        loc_a = (xi, yi, zi)
        loc_b = (xj, yj, zj)
        gap_junctions[f"gj_{n}"] = GapJunctionPoint(
            (i, j),
            loc_a,
            weight,
            location_b=None if loc_b == loc_a else loc_b,
        )
    logger.debug(
        "Prepared %d gap junctions from %s weight=%f",
        len(gap_junctions),
        swc_file,
        weight,
    )
    return gap_junctions
