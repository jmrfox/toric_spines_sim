"""Arbor-specific SWC helpers (vendored from toric_spines_sim.geometry.swc, Nov 2026)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import arbor as A
import numpy as np

from toric_spines_sim.geometry.swc import iter_swc_samples

_ARBOR_NO_PARENT = 0xFFFFFFFF


def _segment_length(seg) -> float:
    p, d = seg.prox, seg.dist
    return float(np.sqrt((d.x - p.x) ** 2 + (d.y - p.y) ** 2 + (d.z - p.z) ** 2))


def _is_arbor_root_parent(parent_idx: int) -> bool:
    p = int(parent_idx)
    return p < 0 or p == _ARBOR_NO_PARENT


def segment_tree_branches(segment_tree: A.segment_tree) -> List[List[int]]:
    """Group segment-tree indices into Arbor branches (unbranched chains)."""
    parents = [int(p) for p in segment_tree.parents]
    n = len(parents)
    children: List[List[int]] = [[] for _ in range(n)]
    roots: List[int] = []
    for i, parent in enumerate(parents):
        if _is_arbor_root_parent(parent):
            roots.append(i)
        else:
            children[parent].append(i)

    branches: List[List[int]] = []
    queue = list(roots)
    while queue:
        start = queue.pop(0)
        branch = [start]
        cur = start
        while len(children[cur]) == 1:
            nxt = children[cur][0]
            branch.append(nxt)
            cur = nxt
        branches.append(branch)
        for child in children[branch[-1]]:
            queue.append(child)
    return branches


def arbor_locations_for_swc_nodes(
    swc_path: Path,
    morphology: A.morphology,
    segment_tree: A.segment_tree,
    node_ids: Iterable[int],
) -> Dict[int, A.location]:
    """Map SWC sample IDs to Arbor ``(branch, pos)`` at that sample's endpoint.

    ``A.load_swc_arbor`` makes one segment per non-root SWC sample, in file
    order. The distal end of segment *k* is the *k*-th non-root sample. Using
    this mapping (not ``place_pwlin.closest`` on XYZ) keeps colocated cycle-break
    / multi-neck copies on different branches.

    Arbor itself resolves ``(distal (segment k))``, so branch numbering is
    Arbor's own. An earlier version rebuilt branch IDs with a BFS that did not
    match Arbor's numbering and misplaced nodes by up to ~25 µm.

    Parameters
    ----------
    swc_path : path-like
        SWC used to build ``segment_tree``.
    morphology : arbor.morphology
        Morphology corresponding to ``segment_tree``.
    segment_tree : arbor.segment_tree
        Tree produced by ``arbor.load_swc_arbor``.
    node_ids : iterable of int
        SWC sample IDs to map.

    Returns
    -------
    dict[int, arbor.location]
        Distal ``(branch, pos)`` for each requested sample.

    Raises
    ------
    ValueError
        If sample/segment counts disagree or a node ID is missing.

    Examples
    --------
    >>> locs = arbor_locations_for_swc_nodes(path, morph, tree, [12, 13])  # doctest: +SKIP
    >>> locs[12].branch != locs[13].branch  # colocated CYCLE_BREAK copies
    True
    """
    wanted = set(int(i) for i in node_ids)
    samples = iter_swc_samples(swc_path)
    non_root = [s for s in samples if s[6] != -1]
    if len(non_root) != len(segment_tree.segments):
        raise ValueError(
            f"SWC {swc_path} has {len(non_root)} non-root samples but Arbor "
            f"segment tree has {len(segment_tree.segments)} segments"
        )

    node_to_seg = {sample[0]: i for i, sample in enumerate(non_root)}
    root_ids = {sample[0] for sample in samples if sample[6] == -1}

    exprs: Dict[str, str] = {}
    for node_id in wanted:
        if node_id in node_to_seg:
            exprs[f"n{node_id}"] = f"(distal (segment {node_to_seg[node_id]}))"
        elif node_id in root_ids:
            exprs[f"n{node_id}"] = "(root)"
        else:
            raise ValueError(f"SWC node {node_id} not found in {swc_path}")
    if not exprs:
        return {}

    probe_cell = A.cable_cell(morphology, A.decor(), A.label_dict(exprs))
    locations: Dict[int, A.location] = {}
    for node_id in wanted:
        locs = probe_cell.locations(f'"n{node_id}"')
        if len(locs) != 1:
            raise ValueError(
                f"SWC node {node_id} resolved to {len(locs)} Arbor locations"
            )
        locations[node_id] = locs[0]
    return locations


