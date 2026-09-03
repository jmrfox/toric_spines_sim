"""Compute spine neckpoint(s) by comparing an isolated TS mesh to the full cell mesh.

The isolated TS mesh is closed at each neck; the cell mesh has no closing surface
there (the spine opens into dendrite cytosol). Cap faces disagree with the cell
surface (anti-aligned normals and/or deep negative signed distance). Their
area-weighted centroids are the neckpoints used for sink attachment.

Requires ``trimesh`` (optional ``mesh`` extra)::

    uv sync --extra mesh
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from scipy.spatial import cKDTree

from toric_spines_sim.geometry.dendrite import write_xyz_points
from toric_spines_sim.geometry.mesh_pipeline import (
    resolve_mesh_path,
    resolve_mesh_targets,
)
from toric_spines_sim.paths import (
    get_cell_mesh_path,
    get_mesh_path,
    get_pointset_path,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

_TRIMESH_HINT = (
    "trimesh is required for neckpoint computation "
    "(install with `uv sync --extra mesh`)."
)


def _require_trimesh():
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_TRIMESH_HINT) from exc
    return trimesh


def _load_trimesh(mesh_path: PathLike):
    trimesh = _require_trimesh()
    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry in mesh scene: {mesh_path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(loaded)} for {mesh_path}")
    return loaded


@dataclass
class NeckpointParams:
    """Thresholds for TS-vs-cell neck-cap detection."""

    crop_margin: float = 120.0
    align_thr: float = 0.2
    align_signed_max: float = -3.0
    signed_thr: float = -10.0
    min_faces: int = 6
    planarity_max: float = 0.22
    mean_signed_max: float = -5.0
    # Cap cluster mean normal alignment with nearest cell face (true caps anti-align).
    mean_align_max: float = 0.5
    ray_frac_min: float = 0.6
    # Outward rays must stay in the cell AND leave the TS volume (open into dendrite).
    open_score_min: float = 0.75
    ray_offsets: tuple[float, ...] = (10.0, 25.0, 50.0, 80.0, 120.0)
    dendrite_radius: float = 150.0
    dendrite_far_thr: float = 20.0
    dendrite_frac_floor: float = 0.15
    dendrite_frac_of_max: float = 0.5
    # Boundary-loop / attachment fallback
    fallback_dist_thr: float = 15.0
    fallback_align_thr: float = 0.5
    fallback_min_loop_verts: int = 6
    attachment_dendrite_dist_max: float = 35.0
    attachment_facing_min: float = 0.25
    attachment_min_faces: int = 5
    attachment_planarity_max: float = 0.30
    attachment_ray_offsets: tuple[float, ...] = (10.0, 30.0, 60.0, 100.0)
    # Keep the n largest necks by cap area; None keeps all that pass filters.
    max_necks: Optional[int] = None


@dataclass
class NeckCandidate:
    """One detected neck interface."""

    centroid: np.ndarray
    area: float
    n_faces: int
    planarity: float
    mean_signed: float
    mean_align: float
    ray_frac_inside: float
    dendrite_frac: float
    source: str = "cap"
    # Used by max_necks truncation; defaults to area for cap candidates.
    rank_score: float = 0.0

    def __post_init__(self) -> None:
        if self.rank_score == 0.0 and self.area:
            self.rank_score = float(self.area)


def _crop_local_cell(cell, ts, margin: float):
    trimesh = _require_trimesh()
    bmin, bmax = ts.bounds
    lo = bmin - margin
    hi = bmax + margin
    # Prefer face centers so large triangles (e.g. box faces) whose corners lie
    # outside the crop AABB are still included when they pass near the spine.
    centers = cell.triangles_center
    fmask = np.all((centers >= lo) & (centers <= hi), axis=1)
    if not np.any(fmask):
        vmask = np.all((cell.vertices >= lo) & (cell.vertices <= hi), axis=1)
        fmask = vmask[cell.faces].any(axis=1)
    if not np.any(fmask):
        raise ValueError("No cell faces found near the TS mesh bounding box")
    used = np.unique(cell.faces[fmask].ravel())
    remap = -np.ones(len(cell.vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(
        vertices=cell.vertices[used].copy(),
        faces=remap[cell.faces[fmask]].copy(),
        process=False,
    )


def _face_adjacency_clusters(
    face_indices: Sequence[int],
    mesh,
    *,
    min_size: int,
) -> list[list[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in mesh.face_adjacency:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    selected = {int(i) for i in face_indices}
    visited: set[int] = set()
    clusters: list[list[int]] = []
    for start in selected:
        if start in visited:
            continue
        queue: deque[int] = deque([start])
        visited.add(start)
        component = [start]
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in selected and v not in visited:
                    visited.add(v)
                    queue.append(v)
                    component.append(v)
        if len(component) >= min_size:
            clusters.append(component)
    clusters.sort(key=len, reverse=True)
    return clusters


def _planarity(points: np.ndarray) -> tuple[float, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    rms = float(np.sqrt(np.mean((centered @ normal) ** 2)))
    extent = float(np.linalg.norm(centered, axis=1).max()) + 1e-9
    return rms / extent, normal


def _apply_max_necks(
    candidates: list[NeckCandidate],
    max_necks: Optional[int],
) -> list[NeckCandidate]:
    ordered = sorted(candidates, key=lambda c: c.rank_score, reverse=True)
    if max_necks is None:
        return ordered
    if max_necks < 1:
        raise ValueError(f"max_necks must be >= 1 or None, got {max_necks}")
    return ordered[:max_necks]


def _cap_candidates(
    ts,
    cell,
    local,
    params: NeckpointParams,
) -> list[NeckCandidate]:
    trimesh = _require_trimesh()
    centers = ts.triangles_center
    normals = ts.face_normals
    areas = ts.area_faces

    closest, _dist, tid = trimesh.proximity.closest_point(local, centers)
    cell_n = local.face_normals[tid]
    align = np.einsum("ij,ij->i", normals, cell_n)
    signed = np.einsum("ij,ij->i", centers - closest, cell_n)

    local_centers = local.triangles_center
    _, local_dist_to_ts, _ = trimesh.proximity.closest_point(ts, local_centers)
    local_tree = cKDTree(local_centers)

    mask = ((align < params.align_thr) & (signed < params.align_signed_max)) | (
        signed < params.signed_thr
    )
    clusters = _face_adjacency_clusters(
        np.where(mask)[0], ts, min_size=params.min_faces
    )

    candidates: list[NeckCandidate] = []
    for component in clusters:
        pts = centers[component]
        weights = areas[component]
        plan_rel, _ = _planarity(pts)
        centroid = np.average(pts, axis=0, weights=weights)
        area = float(weights.sum())
        mean_signed = float(signed[component].mean())
        mean_align = float(align[component].mean())
        mean_n = normals[component].mean(axis=0)
        mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-12)

        offsets = np.asarray(params.ray_offsets, dtype=float)
        probes = centroid[None, :] + offsets[:, None] * mean_n[None, :]
        ray_frac = float(cell.contains(probes).mean())
        ray_frac_ts = float(ts.contains(probes).mean())
        open_score = ray_frac - ray_frac_ts

        near = local_tree.query_ball_point(centroid, r=params.dendrite_radius)
        if near:
            dendrite_frac = float(
                (local_dist_to_ts[near] > params.dendrite_far_thr).mean()
            )
        else:
            dendrite_frac = 0.0

        if ray_frac < params.ray_frac_min:
            continue
        if open_score < params.open_score_min:
            continue
        if plan_rel > params.planarity_max:
            continue
        if mean_signed > params.mean_signed_max:
            continue
        if mean_align >= params.mean_align_max:
            continue

        candidates.append(
            NeckCandidate(
                centroid=np.asarray(centroid, dtype=float),
                area=area,
                n_faces=len(component),
                planarity=plan_rel,
                mean_signed=mean_signed,
                mean_align=mean_align,
                ray_frac_inside=ray_frac,
                dendrite_frac=dendrite_frac,
                source="cap",
            )
        )

    if not candidates:
        return []

    max_dend = max(c.dendrite_frac for c in candidates)
    dend_floor = max(params.dendrite_frac_floor, params.dendrite_frac_of_max * max_dend)
    filtered = [c for c in candidates if c.dendrite_frac >= dend_floor]
    if not filtered:
        # Soften dendrite gate rather than returning empty when ray/planarity passed.
        filtered = candidates
        logger.debug(
            "Dendrite-context filter removed all candidates; keeping ray/planarity set"
        )
    return filtered


def _extract_boundary_loops(faces_bool: np.ndarray, mesh) -> list[list[int]]:
    from collections import Counter

    def canon(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    edge_count: Counter = Counter()
    selected = np.where(faces_bool)[0]
    for fi in selected:
        face = mesh.faces[fi]
        for edge in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge_count[canon(*edge)] += 1

    boundary = [e for e, count in edge_count.items() if count == 1]
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary:
        adj[a].add(b)
        adj[b].add(a)

    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in list(adj.keys()):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev: Optional[int] = None
        cur = start
        while True:
            nbrs = list(adj[cur])
            nxt = None
            for n in nbrs:
                if n != prev:
                    nxt = n
                    break
            if nxt is None or nxt == start:
                if nxt == start:
                    loops.append(loop)
                break
            if nxt in visited and nxt != start:
                loops.append(loop)
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
            if len(loop) > 100_000:
                break
    return loops


def _label_spine_cell_faces(ts, local, params: NeckpointParams) -> np.ndarray:
    """Boolean mask of local cell faces that match the TS spine surface."""
    trimesh = _require_trimesh()
    centers = local.triangles_center
    normals = local.face_normals
    _closest, dist, tid = trimesh.proximity.closest_point(ts, centers)
    del _closest
    ts_n = ts.face_normals[tid]
    align = np.einsum("ij,ij->i", normals, ts_n)
    return (dist < params.fallback_dist_thr) & (align > params.fallback_align_thr)


def _fallback_attachment_candidates(
    ts,
    cell,
    local,
    params: NeckpointParams,
) -> list[NeckCandidate]:
    """Find TS face patches that face nearby dendrite (non-spine) cell membrane.

    Used when classic neck-cap detection fails (shallow / poorly closed necks).
    """
    trimesh = _require_trimesh()
    on_spine = _label_spine_cell_faces(ts, local, params)
    dendrite = ~on_spine
    if not np.any(dendrite):
        return []

    centers = ts.triangles_center
    normals = ts.face_normals
    areas = ts.area_faces
    local_centers = local.triangles_center
    dendrite_centers = local_centers[dendrite]
    tree = cKDTree(dendrite_centers)
    d_dend, idx = tree.query(centers)
    to_dend = dendrite_centers[idx] - centers
    to_dend_u = to_dend / (np.linalg.norm(to_dend, axis=1)[:, None] + 1e-12)
    facing = np.einsum("ij,ij->i", normals, to_dend_u)

    mask = (d_dend < params.attachment_dendrite_dist_max) & (
        facing > params.attachment_facing_min
    )
    clusters = _face_adjacency_clusters(
        np.where(mask)[0], ts, min_size=params.attachment_min_faces
    )
    if not clusters:
        return []

    closest, _dist, tid = trimesh.proximity.closest_point(local, centers)
    cell_n = local.face_normals[tid]
    align = np.einsum("ij,ij->i", normals, cell_n)
    signed = np.einsum("ij,ij->i", centers - closest, cell_n)

    scored: list[tuple[float, NeckCandidate]] = []
    for component in clusters:
        pts = centers[component]
        weights = areas[component]
        plan_rel, _ = _planarity(pts)
        if plan_rel > params.attachment_planarity_max:
            continue
        centroid = np.average(pts, axis=0, weights=weights)
        area = float(weights.sum())
        mean_n = normals[component].mean(axis=0)
        mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-12)
        offsets = np.asarray(params.attachment_ray_offsets, dtype=float)
        probes = centroid[None, :] + offsets[:, None] * mean_n[None, :]
        frac_cell = float(cell.contains(probes).mean())
        frac_ts = float(ts.contains(probes).mean())
        # Prefer openings into cell cytosol that leave the spine volume.
        open_score = frac_cell - frac_ts
        if open_score < 0.25:
            continue
        mean_facing = float(facing[component].mean())
        mean_dd = float(d_dend[component].mean())
        score = (
            mean_facing * 2.0
            - mean_dd / 50.0
            + open_score
            + min(area, 1.0e4) / 1.0e4
        )
        cand = NeckCandidate(
            centroid=np.asarray(centroid, dtype=float),
            area=area,
            n_faces=len(component),
            planarity=plan_rel,
            mean_signed=float(signed[component].mean()),
            mean_align=float(align[component].mean()),
            ray_frac_inside=frac_cell,
            dendrite_frac=mean_facing,
            source="attachment",
            rank_score=score,
        )
        scored.append((score, cand))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]


def _fallback_boundary_loop_candidates(
    ts,
    local,
    params: NeckpointParams,
) -> list[NeckCandidate]:
    on_spine = _label_spine_cell_faces(ts, local, params)
    if not np.any(on_spine):
        return []

    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in local.face_adjacency:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    visited: set[int] = set()
    comps: list[list[int]] = []
    for i in np.where(on_spine)[0]:
        i = int(i)
        if i in visited:
            continue
        queue: deque[int] = deque([i])
        visited.add(i)
        comp = [i]
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if on_spine[v] and v not in visited:
                    visited.add(v)
                    queue.append(v)
                    comp.append(v)
        comps.append(comp)
    if not comps:
        return []
    comps.sort(key=len, reverse=True)

    main = np.zeros(len(local.faces), dtype=bool)
    main[comps[0]] = True
    loops = _extract_boundary_loops(main, local)
    loops = [loop for loop in loops if len(loop) >= params.fallback_min_loop_verts]
    if not loops:
        return []

    candidates: list[NeckCandidate] = []
    for loop in loops:
        pts = local.vertices[np.asarray(loop, dtype=int)]
        plan_rel, _ = _planarity(pts)
        centroid = pts.mean(axis=0)
        # Approximate "area" by planar disk from mean radius so max_necks ranking works.
        radius = float(np.linalg.norm(pts - centroid, axis=1).mean())
        area = float(np.pi * radius * radius)
        if plan_rel > params.planarity_max:
            continue
        candidates.append(
            NeckCandidate(
                centroid=np.asarray(centroid, dtype=float),
                area=area,
                n_faces=len(loop),
                planarity=plan_rel,
                mean_signed=float("nan"),
                mean_align=float("nan"),
                ray_frac_inside=float("nan"),
                dendrite_frac=float("nan"),
                source="boundary_loop",
            )
        )
    return candidates


def compute_neck_candidates(
    ts_mesh,
    cell_mesh,
    params: Optional[NeckpointParams] = None,
) -> list[NeckCandidate]:
    """Detect neck interfaces; apply ``params.max_necks`` truncation."""
    params = params or NeckpointParams()
    local = _crop_local_cell(cell_mesh, ts_mesh, params.crop_margin)
    candidates = _cap_candidates(ts_mesh, cell_mesh, local, params)
    if not candidates:
        logger.info("No neck caps passed filters; trying attachment fallback")
        candidates = _fallback_attachment_candidates(
            ts_mesh, cell_mesh, local, params
        )
    if not candidates:
        logger.info("Attachment fallback empty; trying boundary-loop fallback")
        candidates = _fallback_boundary_loop_candidates(ts_mesh, local, params)
    return _apply_max_necks(candidates, params.max_necks)


def compute_neck_points(
    ts_mesh,
    cell_mesh,
    params: Optional[NeckpointParams] = None,
) -> list[np.ndarray]:
    """Return neckpoint XYZ arrays (pixel space), largest-first after ``max_necks``."""
    return [c.centroid.copy() for c in compute_neck_candidates(ts_mesh, cell_mesh, params)]


def default_neckpoint_path(stem: str) -> Path:
    """Pixel-space neckpoint path for a mesh stem (e.g. ``TS1``)."""
    return get_pointset_path(f"{stem}_neckpoint.txt", units="pixels")


def compute_neck_points_for_stem(
    stem: PathLike,
    *,
    cell_mesh_path: Optional[PathLike] = None,
    params: Optional[NeckpointParams] = None,
    write: bool = True,
    overwrite: bool = True,
    output_path: Optional[PathLike] = None,
) -> list[tuple[float, float, float]]:
    """Compute neckpoints for one TS mesh stem and optionally write the pixel file.

    Parameters
    ----------
    stem
        Mesh stem, filename, or path (e.g. ``TS1``, ``TS1.obj``).
    cell_mesh_path
        Full cell mesh; defaults to ``cell_wrapped_simplified.obj``.
    params
        Detection parameters including ``max_necks``.
    write
        If True, write ``data/pointsets/pixels/<stem>_neckpoint.txt``.
    overwrite
        If False and the output exists, skip writing and return existing points
        only when write would be skipped after a successful compute — still
        recomputes unless you check existence first in the CLI.
    output_path
        Override output path.
    """
    params = params or NeckpointParams()
    mesh_path = resolve_mesh_path(stem)
    stem_name = mesh_path.stem
    cell_path = (
        Path(cell_mesh_path)
        if cell_mesh_path is not None
        else get_cell_mesh_path()
    )
    if not cell_path.is_file():
        # Allow bare name under data/mesh/
        alt = get_mesh_path(Path(cell_path).name)
        if alt.is_file():
            cell_path = alt
        else:
            raise FileNotFoundError(f"Cell mesh not found: {cell_path}")

    out = Path(output_path) if output_path is not None else default_neckpoint_path(stem_name)
    if write and out.exists() and not overwrite:
        logger.info("Skipping existing neckpoint file: %s", out)
        from toric_spines_sim.utils import load_xyz_points

        return load_xyz_points(out)

    ts = _load_trimesh(mesh_path)
    cell = _load_trimesh(cell_path)
    candidates = compute_neck_candidates(ts, cell, params)
    points = [
        (float(c.centroid[0]), float(c.centroid[1]), float(c.centroid[2]))
        for c in candidates
    ]
    for i, cand in enumerate(candidates):
        logger.info(
            "%s neck[%d] source=%s area=%.1f n=%d plan=%.3f signed=%.1f "
            "ray=%.2f dend=%.2f xyz=(%.3f, %.3f, %.3f)",
            stem_name,
            i,
            cand.source,
            cand.area,
            cand.n_faces,
            cand.planarity,
            cand.mean_signed,
            cand.ray_frac_inside,
            cand.dendrite_frac,
            cand.centroid[0],
            cand.centroid[1],
            cand.centroid[2],
        )

    if not points:
        logger.warning("No neckpoints found for %s", stem_name)
        return []

    if write:
        write_xyz_points(out, points)
        logger.info("Wrote %d neckpoint(s) to %s", len(points), out)
    return points


def compute_neck_points_all(
    *,
    meshes: Optional[Sequence[str]] = None,
    all_meshes: bool = False,
    cell_mesh_path: Optional[PathLike] = None,
    params: Optional[NeckpointParams] = None,
    write: bool = True,
    overwrite: bool = True,
) -> dict[str, list[tuple[float, float, float]]]:
    """Compute neckpoints for one or more TS meshes."""
    targets = resolve_mesh_targets(meshes, all_meshes=all_meshes)
    results: dict[str, list[tuple[float, float, float]]] = {}
    for mesh_path in targets:
        results[mesh_path.stem] = compute_neck_points_for_stem(
            mesh_path,
            cell_mesh_path=cell_mesh_path,
            params=params,
            write=write,
            overwrite=overwrite,
        )
    return results
