"""
SWC file parsing and geometry utilities.

SWC encodes a directed tree. Toric-spine morphologies restore genus > 0
electrically via header annotations (not extra SWC edges):

- ``# CYCLE_BREAK reconnect i j`` — pair of samples that were split to break
  a mesh cycle. The two samples are often colocated; the cable graph
  (``simulation.reference.LoopCableModel``) merges them back into one node.
- ``# MULTI_NECK reconnect i j`` — extra neck attached to a copy of the sink
  start node (see ``geometry.sink``). These samples may or may not share XYZ.

Precomputed files live under ``data/swc/{pixels,microns}/`` as
``TS{id}_wsink_r{R}um.swc``. Historical names are under ``archive/``.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

from toric_spines_sim.utils import (
    equal_vectors,
)  # noqa: F401  re-exported for backward compatibility

import numpy as np

from swctools import SWCModel
from swctools.io import parse_swc

_RECONNECT_LINE = re.compile(
    r"#\s*(CYCLE_BREAK|MULTI_NECK)\s+reconnect\s+(\d+)\s+(\d+)\s*$",
    re.IGNORECASE,
)
_RECONNECT_LOOSE = re.compile(
    r"#\s*(CYCLE_BREAK|MULTI_NECK)\s+reconnect\b",
    re.IGNORECASE,
)


def parse_cycle_breaks(swc_path: Path):
    """Parse ``# CYCLE_BREAK reconnect i j`` annotations from an SWC file.

    Returns a list of integer pairs ``[(i, j), ...]``.

    Raises ValueError if any CYCLE_BREAK directive is malformed (e.g. non-integer IDs).
    """
    try:
        result = parse_swc(str(swc_path), validate_reconnections=False)
    except Exception:
        logger.error(
            "Failed to read SWC file for cycle breaks: %s", swc_path, exc_info=True
        )
        raise ValueError(f"Invalid SWC file: {swc_path}")

    # Count CYCLE_BREAK directives in the header to detect malformed ones that
    # parse_swc silently skipped (e.g. non-integer node IDs).
    directive_count = sum(
        1
        for line in result.header
        if "cycle_break" in line.lower() and "reconnect" in line.lower()
    )
    if directive_count != len(result.reconnections):
        raise ValueError(
            f"Found {directive_count} CYCLE_BREAK directive(s) in header but only "
            f"{len(result.reconnections)} parsed successfully; check for malformed lines"
        )

    reconnect_pairs = list(result.reconnections)
    logger.debug(
        "Parsed %d cycle break reconnect pairs from %s", len(reconnect_pairs), swc_path
    )
    return reconnect_pairs


def parse_multi_neck_reconnects(swc_path: Path) -> List[Tuple[int, int]]:
    """Parse ``# MULTI_NECK reconnect i j`` annotations from an SWC header.

    Extra necks are electrically tied to the sink start by a gap junction
    between node ``i`` (sink start) and node ``j`` (colocated sink-start copy).
    """
    return _parse_reconnect_kind(swc_path, "MULTI_NECK")


def parse_reconnect_pairs(swc_path: Path) -> List[Tuple[int, int]]:
    """Return CYCLE_BREAK pairs followed by MULTI_NECK pairs (no duplicates)."""
    cycle_pairs = parse_cycle_breaks(swc_path)
    extra = parse_multi_neck_reconnects(swc_path)
    seen = set(cycle_pairs)
    merged = list(cycle_pairs)
    for pair in extra:
        if pair not in seen and (pair[1], pair[0]) not in seen:
            merged.append(pair)
            seen.add(pair)
    logger.debug(
        "Parsed %d reconnect pairs (%d cycle-break, %d extra) from %s",
        len(merged),
        len(cycle_pairs),
        len(merged) - len(cycle_pairs),
        swc_path,
    )
    return merged


def _parse_reconnect_kind(swc_path: Path, kind: str) -> List[Tuple[int, int]]:
    kind_upper = kind.upper()
    pairs: List[Tuple[int, int]] = []
    for line in Path(swc_path).read_text().splitlines():
        stripped = line.strip()
        if not _RECONNECT_LOOSE.search(stripped):
            continue
        match = _RECONNECT_LINE.match(stripped)
        if match is None:
            if stripped.upper().find(kind_upper) >= 0:
                raise ValueError(
                    f"Malformed {kind} reconnect line in {swc_path}: {stripped}"
                )
            continue
        if match.group(1).upper() != kind_upper:
            continue
        pairs.append((int(match.group(2)), int(match.group(3))))
    logger.debug("Parsed %d %s reconnect pairs from %s", len(pairs), kind, swc_path)
    return pairs


def iter_swc_samples(swc_path: Path) -> List[Tuple[int, int, float, float, float, float, int]]:
    """Return SWC samples in file order: ``(id, tag, x, y, z, r, parent)``."""
    samples: List[Tuple[int, int, float, float, float, float, int]] = []
    for line in Path(swc_path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 7:
            raise ValueError(f"Malformed SWC data line in {swc_path}: {stripped}")
        samples.append(
            (
                int(parts[0]),
                int(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
                int(parts[6]),
            )
        )
    return samples


def read_swc_points(swc_path: Path):
    """Return dict id -> (x, y, z, r) from SWC content (ignores non-data lines)."""
    try:
        swc_model = SWCModel.from_swc_file(str(swc_path), validate_reconnections=False)
    except Exception:
        logger.error("Failed to read SWC file: %s", swc_path, exc_info=True)
        raise ValueError(f"Invalid SWC file: {swc_path}")
    points_by_id = {}
    for node_id in swc_model.nodes():
        xyz = swc_model.get_node_xyz(node_id)
        radius = swc_model.get_node_radius(node_id)
        points_by_id[node_id] = (*xyz, radius)
    logger.debug("Read %d SWC points from %s", len(points_by_id), swc_path)
    return points_by_id


def get_center_coordinates_for_all_segments(
    swc_filepath: Union[str, Path],
    use_radius_weighting: bool = False,
) -> Dict[str, Tuple[float, float, float]]:
    """Centre of each SWC segment (parent→child frustum), keyed ``probe_seg_k``.

    ``k`` is the segment index in :class:`~toric_spines_sim.geometry.morphology.SegmentTable`
    order (non-root samples in file order), so ``probe_seg_k`` always refers
    to ``SegmentTable.segments[k]``. With ``use_radius_weighting`` the centre
    is weighted by squared radius toward the thicker end.
    """
    from toric_spines_sim.geometry.morphology import SegmentTable

    table = SegmentTable.from_swc(Path(swc_filepath))
    record_points: Dict[str, Tuple[float, float, float]] = {}
    for k, seg in enumerate(table.segments):
        p, d = seg.prox, seg.dist
        if use_radius_weighting and (p.radius ** 2 + d.radius ** 2) > 0:
            w0, w1 = p.radius ** 2, d.radius ** 2
            c = tuple((w0 * a + w1 * b) / (w0 + w1) for a, b in ((p.x, d.x), (p.y, d.y), (p.z, d.z)))
        else:
            c = seg.center
        record_points[f"probe_seg_{k}"] = tuple(float(v) for v in c)
    logger.info("Prepared %d record points from %s", len(record_points), swc_filepath)
    return record_points
