"""Arbor segment_tree geometry utilities."""

from __future__ import annotations

import logging
from typing import Tuple

import arbor as A

logger = logging.getLogger(__name__)


def scale_one_radius_in_segment_tree_by_coordinates(
    tree: A.segment_tree,
    scale_factor: float,
    target_xyz: Tuple[float, float, float],
    tolerance: float = 1e-6,
):
    """
    Scale the radius of a specific node in an Arbor segment_tree, identified
    by its (x, y, z) coordinates.

    Every proximal or distal endpoint whose position matches *target_xyz*
    (within *tolerance*) has its radius multiplied by *scale_factor*.
    """
    tx, ty, tz = target_xyz
    tol2 = tolerance * tolerance
    new_tree = A.segment_tree()
    matched = 0
    for parent_idx, seg in zip(tree.parents, tree.segments):
        p, d = seg.prox, seg.dist

        pr = p.radius
        if (p.x - tx) ** 2 + (p.y - ty) ** 2 + (p.z - tz) ** 2 <= tol2:
            pr = p.radius * scale_factor
            matched += 1

        dr = d.radius
        if (d.x - tx) ** 2 + (d.y - ty) ** 2 + (d.z - tz) ** 2 <= tol2:
            dr = d.radius * scale_factor
            matched += 1

        new_prox = A.mpoint(p.x, p.y, p.z, pr)
        new_dist = A.mpoint(d.x, d.y, d.z, dr)
        new_tree.append(parent_idx, new_prox, new_dist, seg.tag)

    if matched == 0:
        logger.warning(
            "scale_radius_at_node: no endpoints matched (%.6f, %.6f, %.6f) "
            "within tolerance %.2e",
            tx,
            ty,
            tz,
            tolerance,
        )
    else:
        logger.debug(
            "scale_radius_at_node: scaled %d endpoint(s) at (%.6f, %.6f, %.6f) by %.4f",
            matched,
            tx,
            ty,
            tz,
            scale_factor,
        )
    return new_tree


def scale_radii_in_segment_tree_by_tag(
    tree: A.segment_tree, scale_factor: float, scale_tag: int
):
    """
    Scale the radii of all segments in an Arbor segment_tree with a given tag.

    The distal radius of a matching segment is always scaled.  The proximal
    radius is only scaled when the parent segment also carries *scale_tag*,
    so that boundary nodes (e.g. a neck node shared with a different-tagged
    region) are not affected.
    """
    segments = tree.segments
    parents = tree.parents
    new_tree = A.segment_tree()
    for i, (parent_idx, seg) in enumerate(zip(parents, segments)):
        p, d = seg.prox, seg.dist

        if seg.tag == scale_tag:
            # Scale proximal only when the parent segment shares the same tag
            parent_has_same_tag = (
                parent_idx >= 0 and segments[parent_idx].tag == scale_tag
            )
            pr = p.radius * scale_factor if parent_has_same_tag else p.radius
            dr = d.radius * scale_factor
        else:
            pr = p.radius
            dr = d.radius

        new_prox = A.mpoint(p.x, p.y, p.z, pr)
        new_dist = A.mpoint(d.x, d.y, d.z, dr)
        new_tree.append(parent_idx, new_prox, new_dist, seg.tag)
    return new_tree
