"""
Graph-analysis utilities for SWC morphologies.

Provides geodesic distance computation, compartment classification relative
to a source→target path, and probe-to-node mapping for bridging Arbor
simulation results back to the morphology graph.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Literal, Set, Tuple, Union

import networkx as nx
import numpy as np
from swctools import SWCModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_undirected_graph(swc_filepath: Union[str, Path]) -> nx.Graph:
    """Load an SWC file and return the undirected graph with cycle reconnections.

    The returned graph has node attributes ``x, y, z, r, t`` and edge
    attribute ``length`` (Euclidean distance between endpoints).
    """
    swc_filepath = Path(swc_filepath)
    swc_model = SWCModel.from_swc_file(str(swc_filepath), validate_reconnections=False)
    graph = swc_model.make_cycle_connections()

    # Attach Euclidean edge lengths
    for u, v in graph.edges():
        nu, nv = graph.nodes[u], graph.nodes[v]
        dx = nu["x"] - nv["x"]
        dy = nu["y"] - nv["y"]
        dz = nu["z"] - nv["z"]
        graph.edges[u, v]["length"] = float(np.sqrt(dx**2 + dy**2 + dz**2))

    return graph


def _nearest_node(
    graph: nx.Graph,
    xyz: Tuple[float, float, float],
) -> int:
    """Return the node ID in *graph* closest (Euclidean) to *xyz*."""
    tx, ty, tz = xyz
    best_node = -1
    best_dist_sq = float("inf")
    for node_id, attrs in graph.nodes(data=True):
        dx = attrs["x"] - tx
        dy = attrs["y"] - ty
        dz = attrs["z"] - tz
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_node = node_id
    logger.debug(
        "Nearest node to (%.4f, %.4f, %.4f): node %d (dist=%.6f)",
        tx,
        ty,
        tz,
        best_node,
        np.sqrt(best_dist_sq),
    )
    return best_node


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_geodesic_distances(
    swc_filepath: Union[str, Path],
    source_xyz: Tuple[float, float, float],
) -> Dict[int, float]:
    """Compute geodesic (path-length) distances from a source point to every node.

    Parameters
    ----------
    swc_filepath : path-like
        Path to an SWC file (with optional ``CYCLE_BREAK`` reconnections).
    source_xyz : (x, y, z)
        3-D coordinate of the source point.  The nearest graph node is used.

    Returns
    -------
    dict[int, float]
        Mapping from node ID to shortest-path distance (in the same spatial
        units as the SWC file, typically µm).  Unreachable nodes are omitted.
    """
    graph = _load_undirected_graph(swc_filepath)
    source_node = _nearest_node(graph, source_xyz)
    distances = dict(
        nx.single_source_dijkstra_path_length(graph, source_node, weight="length")
    )
    logger.info(
        "Geodesic distances from node %d: %d reachable nodes, max=%.2f",
        source_node,
        len(distances),
        max(distances.values()) if distances else 0.0,
    )
    return distances


def classify_compartments(
    swc_filepath: Union[str, Path],
    source_xyz: Tuple[float, float, float],
    target_xyz: Tuple[float, float, float],
    mode: Literal["path_based", "distance_based"] = "path_based",
    spine_tag: int = 3,
    sink_tag: int = 5,
) -> Dict[int, str]:
    """Classify every graph node into one of four categories.

    Categories
    ----------
    ``"main_path"``
        Nodes on the shortest path from *source* to *target*.
    ``"branch"``
        Spine nodes (tag = *spine_tag*) that are off the main path but
        structurally related to it (definition depends on *mode*).
    ``"lateral"``
        All other spine nodes.
    ``"sink"``
        Nodes with tag = *sink_tag*.

    Parameters
    ----------
    swc_filepath : path-like
        Path to an SWC file.
    source_xyz, target_xyz : (x, y, z)
        Coordinates of source and target landmarks.  Nearest graph nodes are
        used.
    mode : ``"path_based"`` | ``"distance_based"``
        How to distinguish *branch* from *lateral*:

        * ``"path_based"``: a spine node is *branch* if its shortest path to
          the source passes through at least one interior main-path node.
        * ``"distance_based"``: a spine node is *branch* if its geodesic
          distance to the target is less than the source's geodesic distance
          to the target (i.e., it lies "closer to the target").
    spine_tag : int
        Tag value identifying spine compartments (default 3).
    sink_tag : int
        Tag value identifying sink compartments (default 5).

    Returns
    -------
    dict[int, str]
        Mapping ``{node_id: category_label}`` for every node in the graph.
    """
    graph = _load_undirected_graph(swc_filepath)
    source_node = _nearest_node(graph, source_xyz)
    target_node = _nearest_node(graph, target_xyz)

    # Shortest path (list of node IDs) from source to target
    main_path_nodes: list[int] = nx.shortest_path(
        graph, source_node, target_node, weight="length"
    )
    main_path_set: Set[int] = set(main_path_nodes)
    # Interior main-path nodes (excluding source and target themselves)
    interior_main_path: Set[int] = main_path_set - {source_node, target_node}

    logger.info(
        "Main path %d → %d: %d nodes",
        source_node,
        target_node,
        len(main_path_nodes),
    )

    # Pre-compute distances needed by both modes
    distances_from_source = dict(
        nx.single_source_dijkstra_path_length(graph, source_node, weight="length")
    )

    if mode == "distance_based":
        distances_from_target = dict(
            nx.single_source_dijkstra_path_length(
                graph, target_node, weight="length"
            )
        )
        source_to_target_distance = distances_from_source.get(
            target_node, float("inf")
        )

    if mode == "path_based":
        # For each non-main-path node, check whether its shortest path to
        # the source passes through an interior main-path node.
        paths_from_source = nx.single_source_dijkstra_path(
            graph, source_node, weight="length"
        )

    classification: Dict[int, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        tag = attrs.get("t", 0)

        # Sink
        if tag == sink_tag:
            classification[node_id] = "sink"
            continue

        # Main path
        if node_id in main_path_set:
            classification[node_id] = "main_path"
            continue

        # Only spine nodes get branch / lateral distinction
        if tag != spine_tag:
            # Nodes with other tags (shouldn't normally happen for TS morphologies)
            classification[node_id] = "lateral"
            continue

        if mode == "path_based":
            # Check if shortest path from source to this node crosses a
            # main-path interior node.
            path_to_node = paths_from_source.get(node_id, [])
            if interior_main_path.intersection(path_to_node):
                classification[node_id] = "branch"
            else:
                classification[node_id] = "lateral"

        elif mode == "distance_based":
            node_to_target = distances_from_target.get(node_id, float("inf"))
            if node_to_target < source_to_target_distance:
                classification[node_id] = "branch"
            else:
                classification[node_id] = "lateral"
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'path_based' or 'distance_based'."
            )

    # Log summary
    counts = {}
    for cat in classification.values():
        counts[cat] = counts.get(cat, 0) + 1
    logger.info("Compartment classification (mode=%s): %s", mode, counts)

    return classification


def map_probes_to_nodes(
    swc_filepath: Union[str, Path],
    record_points: Dict[str, Tuple[float, float, float]],
) -> Dict[str, int]:
    """Map simulation probe labels to their nearest graph node IDs.

    Parameters
    ----------
    swc_filepath : path-like
        Path to an SWC file.
    record_points : dict
        Mapping ``{probe_label: (x, y, z)}`` as returned by
        :func:`~toric_spines_sim.swc.get_center_coordinates_for_all_segments`
        or stored in ``SimulationResults.record_points``.

    Returns
    -------
    dict[str, int]
        Mapping ``{probe_label: node_id}``.
    """
    graph = _load_undirected_graph(swc_filepath)

    # Build array of node coordinates for vectorised lookup
    node_ids = list(graph.nodes())
    node_coords = np.array(
        [[graph.nodes[n]["x"], graph.nodes[n]["y"], graph.nodes[n]["z"]] for n in node_ids]
    )

    probe_to_node: Dict[str, int] = {}
    for probe_label, (px, py, pz) in record_points.items():
        diffs = node_coords - np.array([px, py, pz])
        dist_sq = np.sum(diffs**2, axis=1)
        nearest_idx = int(np.argmin(dist_sq))
        probe_to_node[probe_label] = node_ids[nearest_idx]

    logger.info("Mapped %d probes to graph nodes", len(probe_to_node))
    return probe_to_node


def map_xyz_to_nearest_probes(
    record_points: Dict[str, Tuple[float, float, float]],
    xyz_points: Dict[str, Tuple[float, float, float]],
) -> Dict[str, str]:
    """Map named xyz coordinates to nearest recording probe labels.

    Parameters
    ----------
    record_points : dict
        Mapping ``{probe_label: (x, y, z)}`` for available compartments/probes.
    xyz_points : dict
        Mapping ``{name: (x, y, z)}`` for query points.

    Returns
    -------
    dict[str, str]
        Mapping ``{name: nearest_probe_label}``.
    """
    if not record_points:
        raise ValueError("record_points cannot be empty.")

    probe_labels = list(record_points.keys())
    probe_coords = np.array([record_points[label] for label in probe_labels], dtype=float)

    nearest: Dict[str, str] = {}
    for name, (x, y, z) in xyz_points.items():
        diffs = probe_coords - np.array([x, y, z], dtype=float)
        dist_sq = np.sum(diffs**2, axis=1)
        nearest_idx = int(np.argmin(dist_sq))
        nearest[name] = probe_labels[nearest_idx]

    logger.info(
        "Mapped %d xyz points to nearest probes from %d record points",
        len(xyz_points),
        len(record_points),
    )
    return nearest


def geodesic_distances_from_probe(
    swc_filepath: Union[str, Path],
    record_points: Dict[str, Tuple[float, float, float]],
    source_probe: str,
    target_probes: Iterable[str],
) -> Dict[str, float]:
    """Compute geodesic distances from one probe to target probes.

    Parameters
    ----------
    swc_filepath : path-like
        Path to the SWC morphology.
    record_points : dict
        Mapping ``{probe_label: (x, y, z)}`` for all simulation record points.
    source_probe : str
        Probe label used as source for distance calculation.
    target_probes : iterable[str]
        Probe labels to measure from ``source_probe``.

    Returns
    -------
    dict[str, float]
        Mapping ``{probe_label: geodesic_distance}`` in SWC spatial units.
        Targets that cannot be mapped/reached are omitted.
    """
    if source_probe not in record_points:
        raise KeyError(f"Source probe '{source_probe}' not found in record_points.")

    graph = _load_undirected_graph(swc_filepath)
    probe_to_node = map_probes_to_nodes(swc_filepath, record_points)
    if source_probe not in probe_to_node:
        raise KeyError(f"Source probe '{source_probe}' could not be mapped to an SWC node.")

    source_node = probe_to_node[source_probe]
    node_distances = dict(
        nx.single_source_dijkstra_path_length(graph, source_node, weight="length")
    )

    out: Dict[str, float] = {}
    for probe in target_probes:
        node = probe_to_node.get(probe)
        if node is None:
            continue
        dist = node_distances.get(node)
        if dist is None:
            continue
        out[probe] = float(dist)
    return out
