"""Synthetic spiny-dendrite morphology generator.

Builds a straight tapered dendrite trunk with classical neck+head spines and
writes a subsystem triple matching the toric-spine convention:

- ``*.swc`` — morphology (no sink)
- ``*_AZ.txt`` — one XYZ per spine head (synapse sites)
- ``*_neckpoint.txt`` — single proximal trunk endpoint (sink attachment)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from toric_spines_sim.utils import load_xyz_points

logger = logging.getLogger(__name__)

DistributionMode = Literal["even", "random"]
ScaleStrategy = Literal["relative", "absolute_spines"]

# Literature-inspired template defaults (µm) before any global scale.
_DEFAULT_SPINE_LENGTH = 1.0
_DEFAULT_SPINE_NECK_RADIUS = 0.1
_DEFAULT_SPINE_HEAD_RADIUS = 0.25
_DEFAULT_TRUNK_NECK_RADIUS = 0.5
_DEFAULT_TRUNK_TIP_RADIUS = 0.4
_DEFAULT_TRUNK_LENGTH_OVER_SPINE_LENGTH = 20.0

# Default SWC tags treated as sink when measuring TS subsystem SA.
_DEFAULT_SINK_TAGS = frozenset({5, 6})


@dataclass
class SpinyDendriteParams:
    """Explicit parameters for building a spiny dendrite (mode A)."""

    length: float
    trunk_neck_radius: float
    trunk_tip_radius: float
    n_spines: int
    spine_length: float
    spine_neck_radius: float
    spine_head_radius: float
    spine_neck_length_fraction: float = 0.5
    max_spines_per_node: int = 1
    distribution: DistributionMode = "even"
    seed: Optional[int] = None
    azimuth0: float = 0.0
    axis: Union[str, Tuple[float, float, float], List[float]] = "z"
    trunk_tag: int = 3
    spine_neck_tag: int = 3
    spine_head_tag: int = 3

    def validate(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be > 0, got {self.length}")
        if self.trunk_neck_radius <= 0:
            raise ValueError("trunk_neck_radius must be > 0")
        if self.trunk_tip_radius < 0:
            raise ValueError("trunk_tip_radius must be >= 0")
        if self.n_spines < 0:
            raise ValueError(f"n_spines must be >= 0, got {self.n_spines}")
        if self.n_spines > 0:
            if self.spine_length <= 0:
                raise ValueError("spine_length must be > 0 when n_spines > 0")
            if self.spine_neck_radius <= 0 or self.spine_head_radius <= 0:
                raise ValueError("spine radii must be > 0")
            if not (0.0 < self.spine_neck_length_fraction < 1.0):
                raise ValueError(
                    "spine_neck_length_fraction must be in (0, 1), "
                    f"got {self.spine_neck_length_fraction}"
                )
            if self.spine_neck_radius >= self.spine_head_radius:
                raise ValueError(
                    "spine_neck_radius must be < spine_head_radius "
                    f"({self.spine_neck_radius} >= {self.spine_head_radius})"
                )
            if self.spine_head_radius >= self.trunk_neck_radius:
                raise ValueError(
                    "spine_head_radius must be < trunk_neck_radius "
                    f"({self.spine_head_radius} >= {self.trunk_neck_radius})"
                )
        if self.max_spines_per_node < 1:
            raise ValueError("max_spines_per_node must be >= 1")
        if self.distribution not in ("even", "random"):
            raise ValueError(
                f"distribution must be 'even' or 'random', got {self.distribution!r}"
            )


@dataclass
class ToricSpineMatchParams:
    """Template / layout options when matching a toric spine (mode B)."""

    spine_length: float = _DEFAULT_SPINE_LENGTH
    spine_neck_radius: float = _DEFAULT_SPINE_NECK_RADIUS
    spine_head_radius: float = _DEFAULT_SPINE_HEAD_RADIUS
    trunk_neck_radius: float = _DEFAULT_TRUNK_NECK_RADIUS
    trunk_tip_radius: float = _DEFAULT_TRUNK_TIP_RADIUS
    spine_neck_length_fraction: float = 0.5
    trunk_length_over_spine_length: float = _DEFAULT_TRUNK_LENGTH_OVER_SPINE_LENGTH
    max_spines_per_node: int = 1
    distribution: DistributionMode = "even"
    seed: Optional[int] = None
    azimuth0: float = 0.0
    axis: Union[str, Tuple[float, float, float], List[float]] = "z"
    scale_strategy: ScaleStrategy = "relative"
    trunk_tag: int = 3
    spine_neck_tag: int = 3
    spine_head_tag: int = 3
    sink_tags: frozenset[int] = field(default_factory=lambda: _DEFAULT_SINK_TAGS)


@dataclass
class SWCNodeRecord:
    """One SWC data line."""

    node_id: int
    tag: int
    x: float
    y: float
    z: float
    radius: float
    parent: int


@dataclass
class SpinyDendriteMorphology:
    """In-memory spiny dendrite subsystem."""

    nodes: List[SWCNodeRecord]
    az_points: List[Tuple[float, float, float]]
    neck_point: Tuple[float, float, float]
    trunk_node_ids: List[int]
    spine_head_node_ids: List[int]
    spines_per_attach_node: List[int]
    params: SpinyDendriteParams
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def n_spines(self) -> int:
        return len(self.az_points)

    def surface_area(self) -> float:
        return morphology_surface_area(self.nodes)

    def volume(self) -> float:
        return morphology_volume(self.nodes)


def frustum_lateral_area(r1: float, r2: float, length: float) -> float:
    """Lateral surface area of a conical frustum (excluding end caps)."""
    slant = math.sqrt((r1 - r2) ** 2 + length**2)
    return math.pi * (r1 + r2) * slant


def frustum_volume(r1: float, r2: float, length: float) -> float:
    """Volume of a conical frustum."""
    return (math.pi * length / 3.0) * (r1 * r1 + r1 * r2 + r2 * r2)


def morphology_surface_area(nodes: Sequence[SWCNodeRecord]) -> float:
    """Sum lateral frustum areas over parent→child edges."""
    by_id = {n.node_id: n for n in nodes}
    total = 0.0
    for node in nodes:
        if node.parent < 0:
            continue
        parent = by_id[node.parent]
        length = math.dist(
            (parent.x, parent.y, parent.z), (node.x, node.y, node.z)
        )
        total += frustum_lateral_area(parent.radius, node.radius, length)
    return total


def morphology_volume(nodes: Sequence[SWCNodeRecord]) -> float:
    """Sum frustum volumes over parent→child edges."""
    by_id = {n.node_id: n for n in nodes}
    total = 0.0
    for node in nodes:
        if node.parent < 0:
            continue
        parent = by_id[node.parent]
        length = math.dist(
            (parent.x, parent.y, parent.z), (node.x, node.y, node.z)
        )
        total += frustum_volume(parent.radius, node.radius, length)
    return total


def _axis_unit_vector(
    axis: Union[str, Tuple[float, float, float], List[float]],
) -> np.ndarray:
    if isinstance(axis, str):
        axis_str = axis.lower().strip()
        neg = axis_str.startswith("-")
        axis_name = axis_str[1:] if neg else axis_str
        sgn = -1.0 if neg else 1.0
        if axis_name == "y":
            v = np.array([0.0, 1.0, 0.0], dtype=float)
        elif axis_name == "z":
            v = np.array([0.0, 0.0, 1.0], dtype=float)
        elif axis_name == "x":
            v = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            raise ValueError(f"Unknown axis name: {axis!r}")
        return sgn * v
    arr = np.asarray(axis, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"axis vector must have 3 components, got {arr.size}")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("axis vector must be non-zero")
    return arr / norm


def _perpendicular_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane ⊥ *direction*."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


def _n_attach_nodes(n_spines: int, max_spines_per_node: int) -> int:
    if n_spines == 0:
        return 1  # still need a tip node beyond the neck
    return int(math.ceil(n_spines / max_spines_per_node))


def distribute_spine_counts(
    n_spines: int,
    n_attach: int,
    max_spines_per_node: int,
    distribution: DistributionMode = "even",
    seed: Optional[int] = None,
) -> List[int]:
    """Distribute *n_spines* across *n_attach* trunk nodes (each ≤ max)."""
    if n_attach < 1:
        raise ValueError("n_attach must be >= 1")
    if n_spines < 0:
        raise ValueError("n_spines must be >= 0")
    if n_spines > n_attach * max_spines_per_node:
        raise ValueError(
            f"Cannot place {n_spines} spines on {n_attach} nodes with "
            f"max_spines_per_node={max_spines_per_node}"
        )
    if n_spines == 0:
        return [0] * n_attach

    if distribution == "even":
        base = n_spines // n_attach
        rem = n_spines % n_attach
        counts = [base + (1 if i < rem else 0) for i in range(n_attach)]
        if any(c > max_spines_per_node for c in counts):
            raise ValueError(
                "even distribution exceeds max_spines_per_node; "
                "increase n_attach or max_spines_per_node"
            )
        return counts

    if distribution == "random":
        rng = np.random.default_rng(seed)
        counts = [0] * n_attach
        for _ in range(n_spines):
            candidates = [i for i, c in enumerate(counts) if c < max_spines_per_node]
            if not candidates:
                raise ValueError("no capacity left while assigning spines")
            choice = int(rng.choice(candidates))
            counts[choice] += 1
        return counts

    raise ValueError(f"Unknown distribution: {distribution!r}")


def build_spiny_dendrite(params: SpinyDendriteParams) -> SpinyDendriteMorphology:
    """Build an in-memory spiny dendrite from explicit parameters (mode A)."""
    params.validate()
    direction = _axis_unit_vector(params.axis)
    u_hat, v_hat = _perpendicular_basis(direction)

    n_attach = _n_attach_nodes(params.n_spines, params.max_spines_per_node)
    # Trunk: neck (no spines) + n_attach attachment nodes (includes tip).
    n_trunk = 1 + n_attach
    spines_per_attach = distribute_spine_counts(
        params.n_spines,
        n_attach,
        params.max_spines_per_node,
        distribution=params.distribution,
        seed=params.seed,
    )

    nodes: List[SWCNodeRecord] = []
    trunk_node_ids: List[int] = []
    next_id = 1

    # Proximal neck at origin.
    neck_xyz = (0.0, 0.0, 0.0)
    nodes.append(
        SWCNodeRecord(
            node_id=next_id,
            tag=params.trunk_tag,
            x=neck_xyz[0],
            y=neck_xyz[1],
            z=neck_xyz[2],
            radius=params.trunk_neck_radius,
            parent=-1,
        )
    )
    trunk_node_ids.append(next_id)
    next_id += 1

    # Remaining trunk nodes evenly spaced along the axis.
    for i in range(1, n_trunk):
        t = i / (n_trunk - 1)
        pos = t * params.length * direction
        radius = (1.0 - t) * params.trunk_neck_radius + t * params.trunk_tip_radius
        parent_id = trunk_node_ids[-1]
        nodes.append(
            SWCNodeRecord(
                node_id=next_id,
                tag=params.trunk_tag,
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]),
                radius=float(radius),
                parent=parent_id,
            )
        )
        trunk_node_ids.append(next_id)
        next_id += 1

    attach_trunk_ids = trunk_node_ids[1:]  # skip neck
    az_points: List[Tuple[float, float, float]] = []
    spine_head_node_ids: List[int] = []

    neck_seg_len = params.spine_length * params.spine_neck_length_fraction
    for trunk_id, m_spines in zip(attach_trunk_ids, spines_per_attach):
        if m_spines == 0:
            continue
        trunk_node = next(n for n in nodes if n.node_id == trunk_id)
        trunk_pos = np.array([trunk_node.x, trunk_node.y, trunk_node.z], dtype=float)
        for k in range(m_spines):
            theta = params.azimuth0 + (2.0 * math.pi * k) / m_spines
            spine_dir = math.cos(theta) * u_hat + math.sin(theta) * v_hat
            neck_pos = trunk_pos + neck_seg_len * spine_dir
            head_pos = trunk_pos + params.spine_length * spine_dir

            neck_id = next_id
            nodes.append(
                SWCNodeRecord(
                    node_id=neck_id,
                    tag=params.spine_neck_tag,
                    x=float(neck_pos[0]),
                    y=float(neck_pos[1]),
                    z=float(neck_pos[2]),
                    radius=params.spine_neck_radius,
                    parent=trunk_id,
                )
            )
            next_id += 1

            head_id = next_id
            head_xyz = (
                float(head_pos[0]),
                float(head_pos[1]),
                float(head_pos[2]),
            )
            nodes.append(
                SWCNodeRecord(
                    node_id=head_id,
                    tag=params.spine_head_tag,
                    x=head_xyz[0],
                    y=head_xyz[1],
                    z=head_xyz[2],
                    radius=params.spine_head_radius,
                    parent=neck_id,
                )
            )
            next_id += 1
            spine_head_node_ids.append(head_id)
            az_points.append(head_xyz)

    morph = SpinyDendriteMorphology(
        nodes=nodes,
        az_points=az_points,
        neck_point=neck_xyz,
        trunk_node_ids=trunk_node_ids,
        spine_head_node_ids=spine_head_node_ids,
        spines_per_attach_node=spines_per_attach,
        params=params,
    )
    morph.diagnostics = {
        "surface_area": morph.surface_area(),
        "volume": morph.volume(),
        "n_trunk_nodes": float(n_trunk),
        "n_attach_nodes": float(n_attach),
        "n_spines": float(params.n_spines),
    }
    logger.debug(
        "Built spiny dendrite: %d trunk nodes, %d spines, SA=%.6f",
        n_trunk,
        params.n_spines,
        morph.diagnostics["surface_area"],
    )
    return morph


def write_xyz_points(
    path: Union[str, Path],
    points: Sequence[Tuple[float, float, float]],
) -> Path:
    """Write whitespace-delimited XYZ points (``%.6f``), one per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in points]
    path.write_text("".join(lines), encoding="utf-8")
    return path


def write_swc(
    path: Union[str, Path],
    morph: SpinyDendriteMorphology,
    extra_header: Optional[Sequence[str]] = None,
) -> Path:
    """Write morphology nodes as an SWC file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = morph.params
    header = [
        "# generated by toric_spines_sim.geometry.dendrite\n",
        (
            f"# SPINY_DENDRITE: n_spines={p.n_spines}, "
            f"length={p.length}, trunk_neck_radius={p.trunk_neck_radius}, "
            f"trunk_tip_radius={p.trunk_tip_radius}, spine_length={p.spine_length}, "
            f"max_spines_per_node={p.max_spines_per_node}, "
            f"distribution={p.distribution}\n"
        ),
    ]
    if extra_header:
        for line in extra_header:
            header.append(line if line.endswith("\n") else line + "\n")

    body = [
        (
            f"{n.node_id} {n.tag} {n.x:.6f} {n.y:.6f} {n.z:.6f} "
            f"{n.radius:.6f} {n.parent}\n"
        )
        for n in morph.nodes
    ]
    path.write_text("".join(header + body), encoding="utf-8")
    return path


def write_subsystem(
    morph: SpinyDendriteMorphology,
    swc_path: Union[str, Path],
    az_path: Union[str, Path],
    neck_path: Union[str, Path],
) -> Tuple[Path, Path, Path]:
    """Write SWC + AZ + neckpoint files for a subsystem."""
    swc_out = write_swc(swc_path, morph)
    az_out = write_xyz_points(az_path, morph.az_points)
    neck_out = write_xyz_points(neck_path, [morph.neck_point])
    logger.info(
        "Wrote subsystem: swc=%s az=%s neck=%s (%d synapses)",
        swc_out,
        az_out,
        neck_out,
        morph.n_spines,
    )
    return swc_out, az_out, neck_out


def _parse_swc_segments_with_tags(
    swc_path: Path,
) -> List[Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], int]]:
    """Return (prox xyzr, dist xyzr, child_tag) for each parent→child edge."""
    swc_path = Path(swc_path)
    records: Dict[int, Tuple[int, float, float, float, float, int]] = {}
    with open(swc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            nid = int(parts[0])
            tag = int(parts[1])
            x, y, z, r = map(float, parts[2:6])
            parent = int(parts[6])
            records[nid] = (tag, x, y, z, r, parent)

    segments = []
    for nid, (tag, x, y, z, r, parent) in records.items():
        if parent < 0 or parent not in records:
            continue
        _p_tag, px, py, pz, pr, _parent = records[parent]
        segments.append(((px, py, pz, pr), (x, y, z, r), tag))
    return segments


def swc_subsystem_surface_area(
    swc_path: Union[str, Path],
    sink_tags: Optional[Sequence[int]] = None,
) -> float:
    """Lateral SA of an SWC, excluding segments whose child tag is a sink tag."""
    sink = set(sink_tags) if sink_tags is not None else set(_DEFAULT_SINK_TAGS)
    total = 0.0
    for prox, dist, child_tag in _parse_swc_segments_with_tags(Path(swc_path)):
        if child_tag in sink:
            continue
        length = math.dist(prox[:3], dist[:3])
        total += frustum_lateral_area(prox[3], dist[3], length)
    return total


def swc_subsystem_volume(
    swc_path: Union[str, Path],
    sink_tags: Optional[Sequence[int]] = None,
) -> float:
    """Frustum volume of an SWC, excluding sink-tagged child segments."""
    sink = set(sink_tags) if sink_tags is not None else set(_DEFAULT_SINK_TAGS)
    total = 0.0
    for prox, dist, child_tag in _parse_swc_segments_with_tags(Path(swc_path)):
        if child_tag in sink:
            continue
        length = math.dist(prox[:3], dist[:3])
        total += frustum_volume(prox[3], dist[3], length)
    return total


def count_points_file(path: Union[str, Path]) -> int:
    """Count XYZ rows in a pointset file."""
    return len(load_xyz_points(path))


def _params_from_template(
    match: ToricSpineMatchParams,
    n_spines: int,
    scale: float = 1.0,
    trunk_length: Optional[float] = None,
) -> SpinyDendriteParams:
    """Build SpinyDendriteParams from a match template, optionally scaled."""
    spine_length = match.spine_length * scale
    length = (
        trunk_length
        if trunk_length is not None
        else match.trunk_length_over_spine_length * spine_length
    )
    return SpinyDendriteParams(
        length=length,
        trunk_neck_radius=match.trunk_neck_radius * scale,
        trunk_tip_radius=match.trunk_tip_radius * scale,
        n_spines=n_spines,
        spine_length=spine_length,
        spine_neck_radius=match.spine_neck_radius * scale,
        spine_head_radius=match.spine_head_radius * scale,
        spine_neck_length_fraction=match.spine_neck_length_fraction,
        max_spines_per_node=match.max_spines_per_node,
        distribution=match.distribution,
        seed=match.seed,
        azimuth0=match.azimuth0,
        axis=match.axis,
        trunk_tag=match.trunk_tag,
        spine_neck_tag=match.spine_neck_tag,
        spine_head_tag=match.spine_head_tag,
    )


def _solve_trunk_length_for_sa(
    match: ToricSpineMatchParams,
    n_spines: int,
    target_sa: float,
) -> Tuple[SpinyDendriteParams, SpinyDendriteMorphology]:
    """Keep absolute spine sizes; solve trunk length so SA matches *target_sa*."""
    # SA is monotonic increasing in trunk length for fixed radii / topology.
    def sa_at_length(length: float) -> float:
        params = _params_from_template(match, n_spines, scale=1.0, trunk_length=length)
        return build_spiny_dendrite(params).surface_area()

    # Bracket: zero-ish length lower bound vs large upper bound.
    lo = max(match.spine_length * 0.1, 1e-6)
    sa_lo = sa_at_length(lo)
    if sa_lo >= target_sa:
        # Spines alone already meet/exceed target; use minimal trunk.
        params = _params_from_template(match, n_spines, scale=1.0, trunk_length=lo)
        morph = build_spiny_dendrite(params)
        return params, morph

    hi = match.trunk_length_over_spine_length * match.spine_length
    sa_hi = sa_at_length(hi)
    expand = 0
    while sa_hi < target_sa and expand < 40:
        hi *= 2.0
        sa_hi = sa_at_length(hi)
        expand += 1
    if sa_hi < target_sa:
        raise ValueError(
            f"Could not bracket target SA={target_sa}; max tried length={hi}, SA={sa_hi}"
        )

    # Bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        sa_mid = sa_at_length(mid)
        if abs(sa_mid - target_sa) / target_sa < 1e-8:
            lo = hi = mid
            break
        if sa_mid < target_sa:
            lo = mid
        else:
            hi = mid

    length = 0.5 * (lo + hi)
    params = _params_from_template(match, n_spines, scale=1.0, trunk_length=length)
    morph = build_spiny_dendrite(params)
    return params, morph


def build_from_toric_spine(
    swc_path: Union[str, Path],
    az_path: Union[str, Path],
    match: Optional[ToricSpineMatchParams] = None,
    neck_path: Optional[Union[str, Path]] = None,
) -> SpinyDendriteMorphology:
    """Build a comparable spiny dendrite matched to a toric spine subsystem.

    Hard constraints: ``n_spines == n_AZ`` and lateral surface area ≈ TS SA
    (excluding sink-tagged segments).

    Scale strategies (``match.scale_strategy``):

    - ``relative`` (default): scale a typical-proportion template uniformly so SA
      matches (preserves relative geometry).
    - ``absolute_spines``: keep literature absolute spine sizes; solve trunk length
      so SA matches.
    """
    match = match or ToricSpineMatchParams()
    swc_path = Path(swc_path)
    az_path = Path(az_path)

    n_spines = count_points_file(az_path)
    target_sa = swc_subsystem_surface_area(swc_path, sink_tags=match.sink_tags)
    target_volume = swc_subsystem_volume(swc_path, sink_tags=match.sink_tags)

    if target_sa <= 0:
        raise ValueError(f"Target surface area from {swc_path} is non-positive")

    if match.scale_strategy == "relative":
        template_params = _params_from_template(match, n_spines, scale=1.0)
        template_morph = build_spiny_dendrite(template_params)
        template_sa = template_morph.surface_area()
        if template_sa <= 0:
            raise ValueError("Template morphology has non-positive surface area")
        scale = math.sqrt(target_sa / template_sa)
        params = _params_from_template(match, n_spines, scale=scale)
        morph = build_spiny_dendrite(params)
        scale_factor = scale
    elif match.scale_strategy == "absolute_spines":
        params, morph = _solve_trunk_length_for_sa(match, n_spines, target_sa)
        scale_factor = 1.0
    else:
        raise ValueError(f"Unknown scale_strategy: {match.scale_strategy!r}")

    achieved_sa = morph.surface_area()
    achieved_vol = morph.volume()
    morph.diagnostics.update(
        {
            "target_surface_area": target_sa,
            "achieved_surface_area": achieved_sa,
            "surface_area_rel_error": abs(achieved_sa - target_sa) / target_sa,
            "target_volume": target_volume,
            "achieved_volume": achieved_vol,
            "scale_factor": scale_factor,
            "spine_length": morph.params.spine_length,
            "trunk_length": morph.params.length,
            "literature_spine_length": _DEFAULT_SPINE_LENGTH,
            "spine_length_over_literature": morph.params.spine_length
            / _DEFAULT_SPINE_LENGTH,
        }
    )
    if neck_path is not None:
        # Record only; geometry uses a single proximal neck at the origin.
        morph.diagnostics["source_n_neck_points"] = float(count_points_file(neck_path))

    logger.info(
        "Matched toric spine %s: n_spines=%d, SA target=%.6f achieved=%.6f "
        "(rel err=%.3e), strategy=%s",
        swc_path,
        n_spines,
        target_sa,
        achieved_sa,
        match.scale_strategy,
    )
    return morph
