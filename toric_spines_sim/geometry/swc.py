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
    """Compute the center coordinate of each SWC segment (parent→child edge).

    Returns a dict mapping probe labels like 'probe_seg_0' to (x, y, z) centers.
    """
    swc_filepath = Path(swc_filepath)
    logger.debug(
        "Preparing record points for all segments from %s (radius_weighting=%s)",
        swc_filepath,
        use_radius_weighting,
    )
    swc_model = SWCModel.from_swc_file(str(swc_filepath), validate_reconnections=False)
    node_ids = sorted(swc_model.nodes())
    if not node_ids:
        logger.warning("SWCModel has no nodes for %s", swc_filepath)
        return {}

    logger.debug("Loaded SWCModel with %d nodes from %s", len(node_ids), swc_filepath)

    record_points: Dict[str, Tuple[float, float, float]] = {}
    segment_index = 0
    skipped_root = 0
    for node_id in node_ids:
        parent_id = swc_model.parent_of(node_id)
        if parent_id is None:
            skipped_root += 1
            continue

        x0, y0, z0 = swc_model.get_node_xyz(parent_id)
        x1, y1, z1 = swc_model.get_node_xyz(node_id)

        if use_radius_weighting:
            r0 = swc_model.get_node_radius(parent_id)
            r1 = swc_model.get_node_radius(node_id)
            w0 = r0 * r0
            w1 = r1 * r1
            denom = w0 + w1
            if denom == 0.0:
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                cz = 0.5 * (z0 + z1)
            else:
                cx = (w0 * x0 + w1 * x1) / denom
                cy = (w0 * y0 + w1 * y1) / denom
                cz = (w0 * z0 + w1 * z1) / denom
        else:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            cz = 0.5 * (z0 + z1)

        record_points[f"probe_seg_{segment_index}"] = (cx, cy, cz)
        if segment_index < 5:
            logger.debug(
                "probe_seg_%d (node=%s parent=%s) -> (%.6f, %.6f, %.6f)",
                segment_index,
                node_id,
                parent_id,
                cx,
                cy,
                cz,
            )
        segment_index += 1

    logger.info(
        "Prepared %d record points from %s (skipped roots=%d)",
        len(record_points),
        swc_filepath,
        skipped_root,
    )
    return record_points
