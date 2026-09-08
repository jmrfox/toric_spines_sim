"""Cylindrical sink geometry and SWC append helpers.

Isolated toric-spine SWCs need a surrogate "rest of cell" so current can
diffuse out of the spine. This module builds a straight cylinder (tag 5,
tip tag 6) and appends it at a neck point.

SWC header conventions written by ``append_sink_to_swc*``:

- ``# SINK: start=..., end=..., neck_xyz=...`` — sink sample range and neck
- ``# MULTI_NECK reconnect i j`` — extra neck tied to a copy of the sink start
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np

from toric_spines_sim.geometry.swc import read_swc_points
from toric_spines_sim.utils import load_xyz_points

import logging

logger = logging.getLogger(__name__)


def _parse_sink_header_fields(filepath: Union[str, Path]) -> dict[str, str]:
    """Parse ``# SINK: key=value, ...`` fields from an SWC header.

    Parameters
    ----------
    filepath : path-like
        SWC file written by ``append_sink_to_swc*``.

    Returns
    -------
    dict[str, str]
        Field names mapped to stripped string values.

    Raises
    ------
    ValueError
        If no ``# SINK:`` header is found.
    """
    path = Path(filepath)
    with path.open("r") as f:
        for line in f:
            if line.startswith("# SINK:"):
                header_text = line[len("# SINK:") :].strip()
                fields: dict[str, str] = {}
                for spec in header_text.split(","):
                    spec = spec.strip()
                    if not spec or "=" not in spec:
                        continue
                    key, value = spec.split("=", 1)
                    fields[key.strip()] = value.strip()
                return fields
    raise ValueError(f"No # SINK: header found in {filepath}")


def sink_endpoint_location_from_swc_file(
    filepath: Union[str, Path],
) -> Tuple[float, float, float]:
    """Return XYZ of the distal sink sample referenced by ``end=`` in the header.

    Expects a header line of the form::

        # SINK: start=15, end=22, ...

    Parameters
    ----------
    filepath : path-like
        SWC file with a ``# SINK:`` header.

    Returns
    -------
    tuple of float
        ``(x, y, z)`` of the node whose id is ``end``.

    Raises
    ------
    ValueError
        If the ``# SINK:`` header or ``end=`` field is missing.

    Examples
    --------
    >>> xyz = sink_endpoint_location_from_swc_file("TS1_wsink_r10um.swc")
    >>> len(xyz)
    3
    """
    fields = _parse_sink_header_fields(filepath)
    if "end" not in fields:
        raise ValueError(f"No end= field in SINK header of {filepath}")
    end_idx = int(fields["end"])

    from swctools import SWCModel

    swc_model = SWCModel.from_swc_file(filepath)
    return (
        swc_model.nodes[end_idx]["x"],
        swc_model.nodes[end_idx]["y"],
        swc_model.nodes[end_idx]["z"],
    )


def neck_point_from_swc_file(filepath: Union[str, Path]) -> Tuple[float, float, float]:
    """Read the neck point coordinates from the SINK header in an SWC file.

    Expects a header line of the form::

        # SINK: ..., neck_xyz=<x> <y> <z>

    Parameters
    ----------
    filepath : path-like
        SWC file with a ``# SINK:`` header.

    Returns
    -------
    tuple of float
        ``(x, y, z)`` of the neck attachment.

    Raises
    ------
    ValueError
        If no ``# SINK:`` header or ``neck_xyz`` field is found.

    Examples
    --------
    >>> xyz = neck_point_from_swc_file("TS1_wsink_r10um.swc")
    >>> xyz  # doctest: +SKIP
    (12.3, 4.5, 6.7)
    """
    fields = _parse_sink_header_fields(filepath)
    if "neck_xyz" not in fields:
        raise ValueError(f"No neck_xyz found in SINK header of {filepath}")
    parts = fields["neck_xyz"].split()
    if len(parts) < 3:
        raise ValueError(
            f"neck_xyz field has fewer than 3 values: {fields['neck_xyz']}"
        )
    return (float(parts[0]), float(parts[1]), float(parts[2]))


@dataclass
class SinkGeometry:
    """Straight cylindrical sink (tag 5 body, optional tip tag).

    Attributes
    ----------
    radius : float
        Cylinder radius in the same units as the SWC.
    length : float
        Axial length of the sink (excluding ``connector_length``).
    n_cylinders : int
        Number of frusta along the axis.
    connector_length : float
        Short segment from the snapped neck node to the first sink node.
    axis : str or sequence of float
        ``'x'``/``'y'``/``'z'`` (optional leading ``-``) or a 3-vector.
    """

    radius: float = 0.5
    length: float = 100.0
    n_cylinders: int = 1
    connector_length: float = 1.0
    axis: Union[str, Tuple[float, float, float], List[float]] = "x"


def _as_xyzr(
    value: Union[
        str,
        Path,
        Tuple[float, float, float],
        Tuple[float, float, float, float],
        List[float],
    ],
) -> Tuple[float, float, float, Optional[float]]:
    if isinstance(value, (str, Path)):
        arr = np.loadtxt(str(value))
        arr = np.array(arr).reshape(-1)
        if arr.size < 3:
            raise ValueError(
                "neck/sink point file must contain at least 3 values: x y z [r]"
            )
        x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
        r = float(arr[3]) if arr.size >= 4 else None
        return x, y, z, r
    # sequence input
    seq = list(value)  # type: ignore[arg-type]
    if len(seq) < 3:
        raise ValueError("neck/sink point must have at least 3 values: x y z [r]")
    x, y, z = float(seq[0]), float(seq[1]), float(seq[2])
    r = float(seq[3]) if len(seq) >= 4 else None
    logger.debug("read value from %s: x=%f y=%f z=%f r=%f", value, x, y, z, r)
    return x, y, z, r


def _axis_unit_vector(
    axis: Union[str, Tuple[float, float, float], List[float]],
) -> np.ndarray:
    if isinstance(axis, str):
        axis_str = axis.lower().strip()
        neg = axis_str.startswith("-")
        axis_name = axis_str[1] if neg and len(axis_str) > 1 else axis_str
        sgn = -1.0 if neg else 1.0
        if axis_name == "y":
            v = np.array([0.0, 1.0, 0.0], dtype=float)
        elif axis_name == "z":
            v = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            v = np.array([1.0, 0.0, 0.0], dtype=float)
        return sgn * v

    v = np.asarray(axis, dtype=float).reshape(-1)
    if v.size != 3:
        raise ValueError("axis direction vector must have 3 values: dx dy dz")
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("axis direction vector must be non-zero")
    return v / n


def optimal_sink_direction(
    neck_coords: Union[Tuple[float, float, float], str, Path],
    swc: Union[str, Path, dict],
    average_multiple: bool = True,
) -> Union[Tuple[float, float, float], List[Tuple[float, float, float]]]:
    """
    Compute optimal sink direction(s) pointing away from the morphology.

    Args:
        neck_coords: Either a single (x, y, z) tuple, or a path to a file containing
                  one or more neck points (one per line: x y z).
        swc: SWC file path, SWCModel instance, or dict of node_id -> (x, y, z, r).
        average_multiple: If True and neck_coords is a file with multiple points,
                         return the average direction. If False, return a list
                         of directions (one per neck point).

    Returns:
        Single (dx, dy, dz) direction tuple if average_multiple=True or single neck point.
        List of direction tuples if average_multiple=False and multiple neck points.
    """
    from swctools import SWCModel

    # Load SWC model points
    if isinstance(swc, (str, Path)):
        pts_dict = read_swc_points(Path(swc))
    elif isinstance(swc, SWCModel):
        # SWCModel inherits from networkx.DiGraph
        # Iterate over nodes using the NetworkX API
        pts_dict = {}
        for node_id in swc.nodes():
            node_data = swc.nodes[node_id]
            x = float(node_data["x"])
            y = float(node_data["y"])
            z = float(node_data["z"])
            r = float(node_data.get("r", node_data.get("radius", 0.0)))
            pts_dict[node_id] = (x, y, z, r)
    else:
        # Assume it's already a dict
        pts_dict = swc

    if not pts_dict:
        raise ValueError("SWC model has no points")

    # Load neck point(s)
    if isinstance(neck_coords, (str, Path)):
        # Load from file
        neck_points = load_xyz_points(neck_coords)
    else:
        # Single point provided as tuple
        neck_points = [neck_coords]

    if not neck_points:
        raise ValueError("No neck points provided")

    # Compute direction for each neck point
    directions = []
    for neck_xyz_single in neck_points:
        neck = np.asarray(neck_xyz_single, dtype=float)
        vectors = []
        for _nid, (x, y, z, _r) in pts_dict.items():
            vectors.append(np.array([x, y, z], dtype=float) - neck)
        mean_vec = np.mean(np.stack(vectors, axis=0), axis=0)
        away = -mean_vec
        n = float(np.linalg.norm(away))
        if n == 0.0:
            away = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            away = away / n
        directions.append((float(away[0]), float(away[1]), float(away[2])))

    # Return based on number of points and averaging preference
    if len(directions) == 1:
        return directions[0]

    if average_multiple:
        # Average all directions and normalize
        avg_direction = np.mean(np.array(directions), axis=0)
        n = float(np.linalg.norm(avg_direction))
        if n == 0.0:
            avg_direction = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            avg_direction = avg_direction / n
        return (
            float(avg_direction[0]),
            float(avg_direction[1]),
            float(avg_direction[2]),
        )
    else:
        return directions


def _gen_sink_points(
    neck_coords: Tuple[float, float, float], geom: SinkGeometry
) -> List[Tuple[float, float, float]]:
    """Generate sink node positions.

    Creates n_cylinders+1 nodes that define n_cylinders total segments.
    The first node is at neck + connector_length along the axis.
    Subsequent nodes are spaced evenly over geom.length.
    """
    n_cylinders = max(1, int(geom.n_cylinders))
    total_length = float(geom.length)
    connector_length = float(geom.connector_length)
    step_mag = total_length / n_cylinders
    direction = _axis_unit_vector(geom.axis)
    neck = np.asarray(neck_coords, dtype=float)
    first_arr = neck + connector_length * direction
    first = (float(first_arr[0]), float(first_arr[1]), float(first_arr[2]))
    pts = [first]
    first_base = np.asarray(first, dtype=float)
    for i in range(1, n_cylinders + 1):
        p = first_base + (i * step_mag) * direction
        pts.append((float(p[0]), float(p[1]), float(p[2])))
    logger.debug(
        "generated %d sink nodes (defining %d cylinders)", len(pts), n_cylinders
    )
    return pts


def _snap_to_nearest_node(
    target_xyz: Tuple[float, float, float],
    pts_dict: dict,
) -> Tuple[int, Tuple[float, float, float], float]:
    """Find the SWC node closest to *target_xyz*.

    Returns (node_id, snapped_xyz, radius).
    """
    tx, ty, tz = target_xyz
    nearest_d2 = float("inf")
    best_id = None
    best_xyz = target_xyz
    best_r = 0.0
    for nid, (x, y, z, r) in pts_dict.items():
        d2 = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
        if d2 < nearest_d2:
            nearest_d2 = d2
            best_id = nid
            best_xyz = (float(x), float(y), float(z))
            best_r = float(r)
    return best_id, best_xyz, best_r


def _write_swc_with_header(
    swc_in: Path,
    swc_out: Path,
    header_lines: List[str],
    new_data_lines: List[str],
) -> None:
    """Write *swc_out* by inserting *header_lines* at the end of the original
    header block in *swc_in*, then appending *new_data_lines* after the body."""
    original = swc_in.read_text()
    lines = original.splitlines(keepends=True)
    split_idx = 0
    for i, ln in enumerate(lines):
        if not ln.lstrip().startswith("#"):
            split_idx = i
            break
    else:
        split_idx = len(lines)

    with open(swc_out, "w", encoding="utf-8") as f:
        f.writelines(lines[:split_idx])
        f.writelines(header_lines)
        f.writelines(lines[split_idx:])
        f.writelines(new_data_lines)
    logger.debug("wrote %d new data lines to %s", len(new_data_lines), swc_out)


def append_sink_to_swc(
    swc_in: Union[str, Path],
    swc_out: Union[str, Path],
    neck_coords: Union[str, Path, Tuple[float, float, float], List[float]],
    geom: SinkGeometry,
    tag: int = 5,
    last_segment_tag: Optional[int] = None,
) -> Path:
    """Append a cylindrical sink as a new tree and write ``# SINK:`` metadata.

    The sink starts at the SWC node nearest ``neck_coords`` and extends along
    ``geom.axis``. A connector frustum of length ``geom.connector_length``
    links the neck to the first sink node. When ``last_segment_tag`` is set,
    the distal tip uses that tag (typically 6 for HH) instead of ``tag``.

    Parameters
    ----------
    swc_in, swc_out : path-like
        Input SWC and destination path.
    neck_coords : path-like or sequence of float
        Neck XYZ, or a file with ``x y z [r]``.
    geom : SinkGeometry
        Cylinder radius, length, axis, and segmentation.
    tag : int
        SWC tag for sink body nodes (default 5).
    last_segment_tag : int, optional
        Tag for the distal tip node.

    Returns
    -------
    pathlib.Path
        ``swc_out``.

    Examples
    --------
    >>> geom = SinkGeometry(radius=0.5, length=100.0, axis="x")
    >>> append_sink_to_swc("TS1.swc", "TS1_wsink.swc", (0, 0, 0), geom)  # doctest: +SKIP
    """
    swc_in = Path(swc_in)
    swc_out = Path(swc_out)
    if not swc_in.exists():
        raise FileNotFoundError(f"Input SWC not found: {swc_in}")

    neck_x, neck_y, neck_z, neck_r = _as_xyzr(neck_coords)
    neck_xyz = (neck_x, neck_y, neck_z)
    pts_dict = read_swc_points(swc_in)
    max_id = max(pts_dict.keys()) if pts_dict else 0

    # Snap to nearest existing SWC node so the sink connects exactly.
    snapped_xyz = neck_xyz
    neck_id = None
    if pts_dict:
        neck_id, snapped_xyz, _snapped_r = _snap_to_nearest_node(neck_xyz, pts_dict)
        logger.debug("snapped neck point to node %d at %s", neck_id, snapped_xyz)

    # Build sink nodes (n_cylinders + 1 nodes defining n_cylinders frusta)
    pts = _gen_sink_points(snapped_xyz, geom)
    radius_default = float(geom.radius)

    # Prepare new SWC lines
    logger.debug(
        "building %d sink nodes (defining %d cylinders):", len(pts), geom.n_cylinders
    )
    new_lines = []
    nid = max_id
    sink_start_id = max_id + 1
    logger.debug("sink_start_id: %d", sink_start_id)
    parent = neck_id
    for idx, p in enumerate(pts):
        nid += 1
        x, y, z = p
        node_tag = (
            last_segment_tag
            if last_segment_tag is not None and idx == len(pts) - 1
            else tag
        )
        new_lines.append(
            f"{nid} {node_tag} {x:.6f} {y:.6f} {z:.6f} {radius_default:.6f} {parent}\n"
        )
        parent = nid
        logger.debug("  nid=%d: %s", nid, p)
    sink_end_id = nid
    logger.debug("sink_end_id: %d", sink_end_id)

    header_tag_fields = f"tag={tag}"
    if last_segment_tag is not None:
        header_tag_fields += f", last_segment_tag={last_segment_tag}"
    header = [
        f"# SINK: start={sink_start_id}, end={sink_end_id}, axis={geom.axis}, segments={geom.n_cylinders+1}, nodes={len(pts)}, length={geom.length}, radius={geom.radius}, {header_tag_fields}, neck_xyz={snapped_xyz[0]:.6f} {snapped_xyz[1]:.6f} {snapped_xyz[2]:.6f}\n",
    ]

    _write_swc_with_header(swc_in, swc_out, header, new_lines)
    return swc_out


def append_sink_to_swc_multi_neck_points(
    swc_in: Union[str, Path],
    swc_out: Union[str, Path],
    neck_points: Union[str, Path],
    geom: SinkGeometry,
    tag: int = 5,
    last_segment_tag: Optional[int] = None,
) -> Path:
    """Append one sink, then extra-neck copies of the sink start node.

    ``neck_points`` is a file of ``x y z`` rows. The first row is the primary
    neck; each later row gets a copy of the sink-start sample parented at that
    neck, recorded as ``# MULTI_NECK reconnect i j``.
    """
    swc_in = Path(swc_in)
    swc_out = Path(swc_out)
    if not swc_in.exists():
        raise FileNotFoundError(f"Input SWC not found: {swc_in}")

    neck_xyzs = load_xyz_points(neck_points)
    if len(neck_xyzs) == 0:
        raise ValueError("neck point file contained no points")

    pts_dict = read_swc_points(swc_in)
    max_id = max(pts_dict.keys()) if pts_dict else 0

    # Snap primary neck point
    primary_neck_xyz = neck_xyzs[0]
    snapped_xyz = primary_neck_xyz
    primary_neck_id = None
    if pts_dict:
        primary_neck_id, snapped_xyz, _snapped_r = _snap_to_nearest_node(
            primary_neck_xyz, pts_dict
        )
        logger.debug(
            "snapped primary neck point to node %d at %s", primary_neck_id, snapped_xyz
        )

    pts = _gen_sink_points(snapped_xyz, geom)
    radius_default = float(geom.radius)
    sink_start_xyz = pts[0]

    new_lines: List[str] = []
    nid = max_id
    sink_start_id = max_id + 1
    parent = primary_neck_id if primary_neck_id is not None else -1
    for idx, p in enumerate(pts):
        nid += 1
        x, y, z = p
        node_tag = (
            last_segment_tag
            if last_segment_tag is not None and idx == len(pts) - 1
            else tag
        )
        new_lines.append(
            f"{nid} {node_tag} {x:.6f} {y:.6f} {z:.6f} {radius_default:.6f} {parent}\n"
        )
        parent = nid
    sink_end_id = nid

    # For each additional neck point, connect the (snapped) neck node to a *copy* of the
    # sink start node. We then annotate the SWC header with a directive that downstream
    # code can interpret as a direct connection (e.g. a gap junction) between the true
    # sink start node and its copies.
    reconnect_pairs: List[Tuple[int, int]] = []
    for neck_xyz in neck_xyzs[1:]:
        branch_neck_id = None
        branch_xyz = neck_xyz
        if pts_dict:
            branch_neck_id, branch_xyz, _branch_r = _snap_to_nearest_node(
                neck_xyz, pts_dict
            )

        if (
            abs(branch_xyz[0] - snapped_xyz[0]) < 1e-12
            and abs(branch_xyz[1] - snapped_xyz[1]) < 1e-12
            and abs(branch_xyz[2] - snapped_xyz[2]) < 1e-12
        ):
            continue

        # Copy of the sink start node: colocated with sink start, but parented to this neck.
        nid += 1
        x, y, z = sink_start_xyz
        sink_copy_parent = branch_neck_id if branch_neck_id is not None else -1
        new_lines.append(
            f"{nid} {tag} {x:.6f} {y:.6f} {z:.6f} {radius_default:.6f} {sink_copy_parent}\n"
        )
        reconnect_pairs.append((sink_start_id, nid))

    header_tag_fields = f"tag={tag}"
    if last_segment_tag is not None:
        header_tag_fields += f", last_segment_tag={last_segment_tag}"
    header = [
        f"# SINK: start={sink_start_id}, end={sink_end_id}, axis={geom.axis}, segments={geom.n_cylinders+1}, nodes={len(pts)}, length={geom.length}, radius={geom.radius}, {header_tag_fields}, n_necks={len(neck_xyzs)}, neck_xyz={snapped_xyz[0]:.6f} {snapped_xyz[1]:.6f} {snapped_xyz[2]:.6f}\n",
    ]
    for id_a, id_b in reconnect_pairs:
        header.append(f"# MULTI_NECK reconnect {id_a} {id_b}\n")

    _write_swc_with_header(swc_in, swc_out, header, new_lines)
    return swc_out
