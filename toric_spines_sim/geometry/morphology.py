"""Plain-Python morphology: SWC segments, branches and point placement.

SWC geometry for plotting and analysis, independent of any simulator.

- :class:`SegmentTable` has one frustum per non-root SWC sample, in file
  order, from its parent sample to itself (the child's tag). Sink and neck
  radii can be rescaled.
- :class:`Morphology` groups segments into unbranched branches.
- :class:`PlacePwlin` maps ``Location(branch, pos)`` to points and finds the
  closest location to a point.

Locations use this module's branch numbering; ``(location b x)`` strings
from pre-2026-11 (Arbor) runs used a different numbering.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .swc import iter_swc_samples

PathLike = Union[str, Path]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPoint:
    """A point with radius (µm)."""

    x: float
    y: float
    z: float
    radius: float


@dataclass(frozen=True)
class MSegment:
    """A frustum from ``prox`` to ``dist``; ``sample_id`` is the SWC id of ``dist``."""

    prox: MPoint
    dist: MPoint
    tag: int
    sample_id: int

    @property
    def length(self) -> float:
        p, d = self.prox, self.dist
        return math.dist((p.x, p.y, p.z), (d.x, d.y, d.z))

    @property
    def lateral_area(self) -> float:
        """Lateral surface area (µm²), no end caps."""
        r1, r2 = self.prox.radius, self.dist.radius
        return math.pi * (r1 + r2) * math.hypot(self.length, r1 - r2)

    @property
    def center(self) -> Tuple[float, float, float]:
        p, d = self.prox, self.dist
        return ((p.x + d.x) / 2, (p.y + d.y) / 2, (p.z + d.z) / 2)


@dataclass(frozen=True)
class Location:
    """A point on a branch: ``pos`` in [0, 1] along its length."""

    branch: int
    pos: float


class SegmentTable:
    """One :class:`MSegment` per non-root SWC sample, in file order.

    Attributes
    ----------
    segments : list[MSegment]
    parents : list[int]
        Index of the parent segment, or -1 when the parent sample is a root.
    sample_ids : list[int]
        SWC id of each segment's distal sample.
    """

    def __init__(self, segments: List[MSegment], parents: List[int]):
        self.segments = segments
        self.parents = parents
        self.sample_ids = [s.sample_id for s in segments]
        self._index_of_sample = {sid: k for k, sid in enumerate(self.sample_ids)}

    def __len__(self) -> int:
        return len(self.segments)

    @classmethod
    def from_swc(
        cls,
        swc_path: PathLike,
        *,
        sink_radii_scale: float = 1.0,
        sink_tag: int = 5,
        neck_radius_scale: float = 1.0,
        neck_xyz: Optional[Sequence[float]] = None,
        tolerance: float = 1e-6,
    ) -> "SegmentTable":
        """Read an SWC; optionally rescale sink radii and the neck radius.

        Sink scaling: segments tagged ``sink_tag`` get their distal radius
        scaled, and their proximal radius only when the parent segment has the
        same tag (the neck end stays). Neck scaling: every segment endpoint at
        ``neck_xyz`` (within ``tolerance``) is scaled.
        """
        samples = iter_swc_samples(Path(swc_path))
        by_id = {s[0]: s for s in samples}
        non_root = [s for s in samples if s[6] != -1]
        index_of = {s[0]: k for k, s in enumerate(non_root)}
        segs: List[MSegment] = []
        parents: List[int] = []
        neck_hits = 0
        for s in non_root:
            sid, tag, x, y, z, r, par = s
            p = by_id[par]
            parent_seg = index_of.get(par, -1)
            r_prox, r_dist = p[5], r
            if sink_radii_scale != 1.0 and tag == sink_tag:
                r_dist *= sink_radii_scale
                if parent_seg >= 0 and non_root[parent_seg][1] == sink_tag:
                    r_prox *= sink_radii_scale
            prox = MPoint(p[2], p[3], p[4], r_prox)
            dist = MPoint(x, y, z, r_dist)
            if neck_radius_scale != 1.0 and neck_xyz is not None:
                tx, ty, tz = neck_xyz
                if math.dist((prox.x, prox.y, prox.z), (tx, ty, tz)) <= tolerance:
                    prox = MPoint(prox.x, prox.y, prox.z, prox.radius * neck_radius_scale)
                    neck_hits += 1
                if math.dist((dist.x, dist.y, dist.z), (tx, ty, tz)) <= tolerance:
                    dist = MPoint(dist.x, dist.y, dist.z, dist.radius * neck_radius_scale)
                    neck_hits += 1
            segs.append(MSegment(prox, dist, tag, sid))
            parents.append(parent_seg)
        if neck_radius_scale != 1.0 and neck_xyz is not None and neck_hits == 0:
            logger.warning("neck radius scaling: no endpoints matched %s within %.1e", tuple(neck_xyz), tolerance)
        return cls(segs, parents)

    def index_of_sample(self, sample_id: int) -> int:
        """Segment whose distal end is SWC sample ``sample_id``."""
        return self._index_of_sample[int(sample_id)]

    def radii_by_sample(self) -> Dict[int, Tuple[float, float]]:
        """``{sample_id: (prox radius, dist radius)}`` for every segment."""
        return {s.sample_id: (s.prox.radius, s.dist.radius) for s in self.segments}

    def lateral_area_by_tags(self, tags: Sequence[int]) -> float:
        wanted = {int(t) for t in tags}
        return float(sum(s.lateral_area for s in self.segments if s.tag in wanted))

    def tags(self) -> List[int]:
        return [s.tag for s in self.segments]


class Morphology:
    """Segments grouped into unbranched branches."""

    def __init__(self, table: SegmentTable):
        self.table = table
        n = len(table)
        children: List[List[int]] = [[] for _ in range(n)]
        roots: List[int] = []
        for i, p in enumerate(table.parents):
            (roots if p < 0 else children[p]).append(i)
        branches: List[List[int]] = []
        branch_parent: List[int] = []
        stack = [(r, -1) for r in reversed(roots)]
        while stack:
            start, parent_branch = stack.pop()
            chain = [start]
            cur = start
            while len(children[cur]) == 1:
                cur = children[cur][0]
                chain.append(cur)
            bid = len(branches)
            branches.append(chain)
            branch_parent.append(parent_branch)
            for c in reversed(children[cur]):
                stack.append((c, bid))
        self._branches = branches
        self._branch_parent = branch_parent
        self._branch_children: List[List[int]] = [[] for _ in branches]
        for b, p in enumerate(branch_parent):
            if p >= 0:
                self._branch_children[p].append(b)

    @classmethod
    def from_swc(cls, swc_path: PathLike, **kwargs) -> "Morphology":
        return cls(SegmentTable.from_swc(swc_path, **kwargs))

    @property
    def num_branches(self) -> int:
        return len(self._branches)

    def branch_segments(self, b: int) -> List[MSegment]:
        return [self.table.segments[i] for i in self._branches[b]]

    def branch_segment_indices(self, b: int) -> List[int]:
        return list(self._branches[b])

    def branch_parent(self, b: int) -> int:
        return self._branch_parent[b]

    def branch_children(self, b: int) -> List[int]:
        return list(self._branch_children[b])

    @property
    def segment_tree(self) -> SegmentTable:
        """Compatibility alias: the underlying :class:`SegmentTable`."""
        return self.table


class PlacePwlin:
    """Map locations to points and points to the closest location."""

    def __init__(self, morph: Morphology):
        self.morph = morph
        self._cum: List[np.ndarray] = []
        for b in range(morph.num_branches):
            lengths = [s.length for s in morph.branch_segments(b)]
            self._cum.append(np.concatenate([[0.0], np.cumsum(lengths)]))

    def at(self, loc: Location) -> MPoint:
        segs = self.morph.branch_segments(loc.branch)
        cum = self._cum[loc.branch]
        total = cum[-1]
        if total <= 0:
            s = segs[0].prox
            return MPoint(s.x, s.y, s.z, s.radius)
        target = float(np.clip(loc.pos, 0.0, 1.0)) * total
        k = int(np.clip(np.searchsorted(cum, target, side="right") - 1, 0, len(segs) - 1))
        seg = segs[k]
        f = 0.0 if seg.length == 0 else (target - cum[k]) / seg.length
        p, d = seg.prox, seg.dist
        return MPoint(p.x + f * (d.x - p.x), p.y + f * (d.y - p.y), p.z + f * (d.z - p.z),
                      p.radius + f * (d.radius - p.radius))

    def closest(self, x: float, y: float, z: float) -> Tuple[Location, float]:
        q = np.array([x, y, z], dtype=float)
        best = (Location(0, 0.0), math.inf)
        for b in range(self.morph.num_branches):
            cum = self._cum[b]
            total = cum[-1]
            for k, seg in enumerate(self.morph.branch_segments(b)):
                a = np.array([seg.prox.x, seg.prox.y, seg.prox.z])
                v = np.array([seg.dist.x, seg.dist.y, seg.dist.z]) - a
                vv = float(v @ v)
                t = 0.0 if vv == 0 else float(np.clip((q - a) @ v / vv, 0.0, 1.0))
                dist = float(np.linalg.norm(a + t * v - q))
                if dist < best[1]:
                    pos = 0.0 if total <= 0 else (cum[k] + t * seg.length) / total
                    best = (Location(b, pos), dist)
        return best


def location(branch: int, pos: float) -> Location:
    return Location(int(branch), float(pos))


def place_pwlin(morph: Morphology, isometry=None) -> PlacePwlin:
    if isometry is not None:
        raise NotImplementedError("isometry transforms are not supported")
    return PlacePwlin(morph)

