"""Visualization utilities for Arbor: geometry sanity checks and placement overlays.

Primary goal: make it easy to visually verify that model geometry and placements
(e.g., synapses, gap junction sites) are exactly where you expect on an Arbor
`morphology`/`cable_cell`. Optional Plotly support is included for interactive 3D.

Quick start (Matplotlib)
------------------------
- draw_morphology(morph, ax=None, color='k', linewidth=1.0, alpha=0.6)
- terminals_from_morphology(morph) -> list[arbor.location]
- scatter_locations(morph, locs, ax=None, color='r', size=10, label=None, isometry=None)
- plot_morph_and_locations(morph, locs, *, ax=None, line_kwargs=None, scatter_kwargs=None, isometry=None)
- scatter_points(points, ax=None, color='r', size=10, label=None)

Unified 3D plotting (auto backend)
----------------------------------
- plot_morphology_3d(obj, *, overlays=None, overlay_styles=None, backend='auto', isometry=None)
  One entry point for 3D plots. If running in a Jupyter kernel and Plotly is installed,
  defaults to an interactive Plotly figure; otherwise uses Matplotlib. You can force
  a backend with backend='plotly' or backend='mpl'. `obj` may be an `arbor.morphology`,
  a `loaded_morphology`, or a `cable_cell`.
  Overlay item types supported per label: `arbor.location`, explicit '(location b x)' strings,
  3D points as `(x, y, z)` tuples/lists, or `arbor.mpoint` objects.

Arbor-centric helpers
---------------------
- parse_location_expr(expr) -> tuple[int, float] | None
  Parse explicit locset strings like '(location <branch> <pos>)'.
- locations_from_location_exprs(exprs) -> list[arbor.location]
  Convert a list of explicit '(location ...)' strings to Arbor locations.
- plot_cable_cell_with_locations(cell, overlays=None, overlay_styles=None, ax=None)
  Draw a `cable_cell` and overlay explicit placements grouped by label.
  overlays example: {'syn': ['(location 3 0.42)', ...], 'gj': ['(location 1 0.7)', ...]}

Plotly (interactive 3D)
-----------------------
- plotly_morphology_traces(morph, *, color='gray', width=2, opacity=1.0, name='morphology') -> list[go.Scatter3d]
- plotly_locations_trace(morph, locs, *, isometry=None, color='red', size=3, name='locations') -> go.Scatter3d
- plotly_points_trace(points, *, color='red', size=3, name='points') -> go.Scatter3d
- plotly_morph_and_locations(morph, locs, *, isometry=None, line_kwargs=None, scatter_kwargs=None, layout_kwargs=None) -> go.Figure

Trace plotting
--------------
For plotting Arbor simulation traces, see `toric_spines_sim.traces`:
- TracePlotter: easy Matplotlib plotting for `simulation.samples(handle)` results.

Notes
-----
- All Matplotlib functions accept an optional `ax` (Axes3D). If None, a new figure/axes is created.
- Use `locations_to_points` (via Arbor `place_pwlin`) to map `location`s to 3D coordinates.
- For complex locsets (e.g. '(uniform ...)'), record/resolve to explicit '(location ...)' strings
  if you want to overlay them directly.
- Functions that take a `morph` accept either `arbor.morphology` or objects with a `.morphology`
  attribute (e.g., `loaded_morphology` returned by some loaders). Inputs are normalized internally.

Examples
--------
Geometry check:
    >>> import arbor as A
    >>> from toric_spines_sim import viz
    >>> morph = A.load_swc("data/swc/microns/TS2_wsink_r10um.swc")
    >>> terms = viz.terminals_from_morphology(morph)
    >>> ax = viz.plot_morph_and_locations(morph, terms)

Overlay explicit placements on a cable cell:
    >>> overlays = {
    ...     'syn': ['(location 3 0.42)', '(location 5 0.15)'],
    ...     'gj':  ['(location 1 0.70)', '(location 2 0.33)'],
    ... }
    >>> ax = viz.plot_cable_cell_with_locations(cell, overlays=overlays,
    ...     overlay_styles={'syn': {'color': 'r', 'size': 10}, 'gj': {'color': 'c', 'size': 14}})
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass
import re
import sys
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (ensures 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import arbor as A
except Exception as e:  # pragma: no cover
    raise ImportError(
        "toric_spines_sim.viz requires the 'arbor' package. Install with `pip install arbor`."
    ) from e

# Optional Plotly import for interactive 3D figures
try:  # pragma: no cover
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


COLORS = {
    "centroid": "#333333",
    "skeleton": "#333333",
    "synapse_point": "#DC143C",
    "segment": "#A7C7E7",
    "soma": "#4C78A8",
    "axon": "#FF7F0E",
    "terminal": "#FFDF0E",
    "dendrite": "#2E8B57",
    "spine_head": "#9467BD",
    "spine_neck": "#C5B0D5",
    "synapse_region": "#E74C3C",
    "excitatory_synapse": "#E74C3C",
    "inhibitory_synapse": "#3498DB",
    "neuron": "#2E86AB",
    "branch": "#F39C12",
    "branch_point": "#8C564B",
    "section": "#1ABC9C",
    "path": "#7F7F7F",
    "electrode": "#FFD700",
    "recording_site": "#B8860B",
    "stim_site": "#FF1493",
    "mesh": "#95A5A6",
    "background": "#FFFFFF",
    "grid": "#E0E0E0",
    "label_text": "#111111",
    "selection": "#00BCD4",
}

# ---------- helpers ----------


def _ensure_axes3d(ax=None):
    if ax is not None:
        return ax
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    return ax


@dataclass
class VizConfig:
    width: Optional[int] = None
    height: Optional[int] = None
    template: Optional[str] = None
    title: Optional[str] = None
    figsize: Optional[Tuple[float, float]] = None


def _default_plotly_scene(layout_kwargs: dict) -> dict:
    """Build a Plotly scene dict with sensible defaults, merged with user overrides."""
    default_scene = dict(
        xaxis_title="x (µm)",
        yaxis_title="y (µm)",
        zaxis_title="z (µm)",
        aspectmode="data",
    )
    return {**default_scene, **layout_kwargs.pop("scene", {})}


def _apply_viz_config(fig, config: Optional["VizConfig"]) -> None:
    """Apply VizConfig overrides to a Plotly figure (in-place)."""
    if config is None:
        return
    updates = {}
    if config.width is not None:
        updates["width"] = config.width
    if config.height is not None:
        updates["height"] = config.height
    if config.template is not None:
        updates["template"] = config.template
    if config.title is not None:
        updates["title"] = config.title
    if updates:
        fig.update_layout(**updates)


# ---------- Unified 3D plotting ----------


def _split_overlay_items(
    morph, items: Sequence[object]
) -> Tuple[List["A.location"], List[object]]:
    """Split overlay items into (locations, points).

    - Locations: arbor.location instances or '(location b x)' strings.
    - Points: arbor.mpoint-like (has x,y,z) or (x,y,z) tuples/lists.
    """
    locs: List["A.location"] = []
    pts: List[object] = []
    for it in items:
        if isinstance(it, str):
            locs.extend(locations_from_location_exprs([it]))
            continue
        # arbor.location-like
        try:
            _ = it.branch, it.pos
            locs.append(it)
            continue
        except Exception:
            pass
        # arbor.mpoint-like (has x,y,z)
        if hasattr(it, "x") and hasattr(it, "y") and hasattr(it, "z"):
            pts.append(it)
            continue
        # tuple/list (x,y,z)
        try:
            if len(it) == 3:
                x, y, z = it[0], it[1], it[2]
                # Basic numeric check
                float(x)
                float(y)
                float(z)
                pts.append((float(x), float(y), float(z)))
                continue
        except Exception:
            pass
        # Unknown type: ignore silently (could log in future)
        continue
    return locs, pts


def plot_morphology_3d(
    obj,
    *,
    overlays: Optional[dict[str, Sequence[object]]] = None,
    overlay_styles: Optional[dict[str, dict]] = None,
    backend: str = "auto",
    isometry: Optional["A.isometry"] = None,
    line_kwargs: Optional[dict] = None,
    scatter_kwargs: Optional[dict] = None,
    layout_kwargs: Optional[dict] = None,
    config: Optional["VizConfig"] = None,
):
    """Unified 3D plot of an Arbor morphology or cable_cell, with optional overlays.

    Parameters
    ----------
    obj : arbor.morphology | loaded_morphology | cable_cell
        Object whose morphology will be rendered.
    overlays : dict[str, Sequence[location | str]], optional
        Mapping of label -> sequence of overlay items. Each item can be an
        `arbor.location` or an explicit '(location b x)' string.
    overlay_styles : dict[str, dict], optional
        Matplotlib/Plotly styling per overlay label. For Matplotlib, accepts
        {'color': 'r', 'size': 10}. For Plotly, accepts {'color': 'r', 'size': 3}.
    backend : {'auto','plotly','mpl'}
        - 'auto': Plotly in notebooks (if available), otherwise Matplotlib.
        - 'plotly': Force Plotly.
        - 'mpl': Force Matplotlib.
    isometry : arbor.isometry, optional
        Geometry transform for location-to-point mapping.
    line_kwargs, scatter_kwargs, layout_kwargs : dict, optional
        Extra styling forwarded to underlying backend helpers.

    Returns
    -------
    Matplotlib Axes3D (backend='mpl') or plotly.graph_objects.Figure (backend='plotly').
    """
    morph = _as_morphology(obj.morphology if hasattr(obj, "morphology") else obj)
    overlays = overlays or {}
    overlay_styles = overlay_styles or {}
    line_kwargs = line_kwargs or {}
    scatter_kwargs = scatter_kwargs or {}
    layout_kwargs = layout_kwargs or {}

    chosen = _default_backend() if backend == "auto" else backend

    if chosen == "plotly":
        if go is None:
            raise ImportError(
                "Plotly is not installed but backend='plotly' was requested."
            )
        traces = plotly_morphology_traces(morph, **line_kwargs)
        overlay_traces = []
        for label, items in overlays.items():
            locs, pts = _split_overlay_items(morph, items)
            style = overlay_styles.get(label, {})
            color = style.get("color", "red")
            size = style.get("size", 3)
            overlay_traces.append(
                plotly_locations_trace(
                    morph, locs, isometry=isometry, color=color, size=size, name=label
                )
            )
            if pts:
                overlay_traces.append(
                    plotly_points_trace(
                        pts, color=color, size=size, name=f"{label}_pts"
                    )
                )
        fig = (
            __import__("plotly.graph_objects", fromlist=["go"]).Figure(
                data=traces + overlay_traces
            )
            if False
            else None
        )
        # Use our existing helper to build the figure for consistency
        # (recreate layout defaults)
        fig = go.Figure(data=traces + overlay_traces)
        scene = _default_plotly_scene(layout_kwargs)
        fig.update_layout(scene=scene, **layout_kwargs)
        _apply_viz_config(fig, config)
        return fig

    # Matplotlib path
    ax = draw_morphology(morph, **line_kwargs)
    for label, items in overlays.items():
        locs, pts = _split_overlay_items(morph, items)
        style = overlay_styles.get(label, {})
        size = style.get("size", 10)
        color = style.get("color", "r")
        scatter_locations(
            morph,
            locs,
            ax=ax,
            color=color,
            size=size,
            label=label,
            isometry=isometry,
            **scatter_kwargs,
        )
        if pts:
            scatter_points(pts, ax=ax, color=color, size=size, label=f"{label}_pts")
    if config is not None:
        if config.figsize is not None:
            try:
                ax.figure.set_size_inches(*config.figsize)
            except Exception:
                pass
        if config.title is not None:
            try:
                ax.set_title(config.title)
            except Exception:
                pass
    return ax


def _as_morphology(obj) -> "A.morphology":
    """Return an `arbor.morphology` from common Arbor loader outputs.

    Accepts:
    - `arbor.morphology` (returned as-is)
    - Objects with a `.morphology` attribute (e.g., `loaded_morphology`)
    - Segment-tree-like objects acceptable to `A.morphology(segment_tree)`
    """
    try:
        # If it's already an arbor.morphology, return as-is.
        if isinstance(obj, A.morphology):
            return obj
    except Exception:
        # isinstance can fail if obj is proxied; ignore and try attributes/constructor
        pass
    # Try attribute first (loaded_morphology)
    if hasattr(obj, "morphology"):
        return obj.morphology
    # Fall back to attempting to construct a morphology from a segment_tree-like object
    try:
        return A.morphology(obj)
    except Exception as e:
        raise TypeError(
            "Expected arbor.morphology or loaded_morphology or segment_tree; got type %r"
            % type(obj)
        ) from e


def _default_backend() -> str:
    """Return 'plotly' in notebooks when Plotly is available, else 'mpl'."""
    if go is not None and "ipykernel" in sys.modules:
        return "plotly"
    return "mpl"


# ---------- core API ----------


def draw_morphology(
    morph: "A.morphology",
    ax=None,
    *,
    color: str = COLORS["skeleton"],
    linewidth: float = 1.0,
    alpha: float = 0.6,
):
    """Draw the morphology as polylines (prox->dist for each msegment) on a 3D axes.

    Parameters
    ----------
    morph : arbor.morphology
        Morphology to render.
    ax : matplotlib 3D axes, optional
        If None, a new figure and axes are created.
    color : str
        Line color for cables.
    linewidth : float
        Line width for polylines.
    alpha : float
        Line alpha.

    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes used for drawing.
    """
    ax = _ensure_axes3d(ax)
    morph = _as_morphology(morph)

    # Iterate over branches and their segments
    for b in range(morph.num_branches):
        segs = morph.branch_segments(b)
        for s in segs:
            p, q = s.prox, s.dist  # mpoint
            ax.plot(
                [p.x, q.x],
                [p.y, q.y],
                [p.z, q.z],
                color=_color_mpl,
                linewidth=linewidth,
                alpha=alpha,
            )

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    return ax


# ---------- Frusta rendering ----------


def _mpoint_radius(mp) -> float:
    """Return radius from an Arbor mpoint-like object, with fallbacks."""
    try:
        return float(getattr(mp, "radius"))
    except Exception:
        try:
            return float(getattr(mp, "r"))
        except Exception:
            return 0.0


def _orthonormal_frame_from_axis(dx: float, dy: float, dz: float):
    """Given an axis vector, return a tuple of three orthonormal vectors (u, a, b).

    u is the normalized axis (dx,dy,dz)/||. a and b span the plane perpendicular to u.
    """
    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0:
        return (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    ux, uy, uz = dx / L, dy / L, dz / L
    # Pick a helper vector not colinear with u
    if abs(ux) < 0.9:
        hx, hy, hz = 1.0, 0.0, 0.0
    else:
        hx, hy, hz = 0.0, 1.0, 0.0
    # a = normalize(u x h)
    ax = uy * hz - uz * hy
    ay = uz * hx - ux * hz
    az = ux * hy - uy * hx
    aL = math.sqrt(ax * ax + ay * ay + az * az)
    if aL == 0:
        ax, ay, az = 1.0, 0.0, 0.0
        aL = 1.0
    ax, ay, az = ax / aL, ay / aL, az / aL
    # b = u x a
    bx = uy * az - uz * ay
    by = uz * ax - ux * az
    bz = ux * ay - uy * ax
    return (ux, uy, uz), (ax, ay, az), (bx, by, bz)


def draw_morphology_frusta(
    morph: "A.morphology",
    ax=None,
    *,
    n_sides: int = 16,
    color: str = COLORS["segment"],
    alpha: float = 0.8,
    edgecolor: Optional[str] = None,
    linewidth: float = 0.0,
    min_radius: float = 1e-3,
    radius_scale: float = 1.0,
    caps: bool = False,
):
    """Render each segment as a truncated cone (frustum) between endpoints with radii.

    Parameters
    ----------
    morph : arbor.morphology
        Morphology whose segments will be rendered.
    ax : matplotlib 3D axes or None
    n_sides : int
        Number of sides for the circular cross-sections (smoothness).
    color : str
        Face color for the frusta.
    alpha : float
        Face alpha for the frusta.
    edgecolor : str | None
        Edge color for the frusta; None disables edges.
    linewidth : float
        Edge line width.
    min_radius : float
        Minimum radius to use to avoid degenerate rings.
    radius_scale : float
        Uniform multiplier applied to all segment radii for visualization (default 1.0).
    caps : bool
        If True, add end-caps (disks). Defaults to False to avoid overlap artifacts.

    Returns
    -------
    ax : 3D axes used for drawing.
    """
    ax = _ensure_axes3d(ax)
    morph = _as_morphology(morph)

    # Collect all polygon faces across the morphology
    faces: List[List[Tuple[float, float, float]]] = []

    # Precompute circle angles
    twopi = 2.0 * math.pi
    angles = [twopi * k / n_sides for k in range(n_sides)]

    for b in range(morph.num_branches):
        for s in morph.branch_segments(b):
            p, q = s.prox, s.dist  # mpoint endpoints
            r1 = max(min_radius, radius_scale * _mpoint_radius(p))
            r2 = max(min_radius, radius_scale * _mpoint_radius(q))

            dx, dy, dz = (q.x - p.x), (q.y - p.y), (q.z - p.z)
            u, a, bvec = _orthonormal_frame_from_axis(dx, dy, dz)
            ax_, ay_, az_ = a
            bx_, by_, bz_ = bvec

            # Build rings at each end
            ring1: List[Tuple[float, float, float]] = []
            ring2: List[Tuple[float, float, float]] = []
            for th in angles:
                ct = math.cos(th)
                st = math.sin(th)
                # point on circle in the plane perpendicular to axis
                rx1 = r1 * (ct * ax_ + st * bx_)
                ry1 = r1 * (ct * ay_ + st * by_)
                rz1 = r1 * (ct * az_ + st * bz_)
                rx2 = r2 * (ct * ax_ + st * bx_)
                ry2 = r2 * (ct * ay_ + st * by_)
                rz2 = r2 * (ct * az_ + st * bz_)
                ring1.append((p.x + rx1, p.y + ry1, p.z + rz1))
                ring2.append((q.x + rx2, q.y + ry2, q.z + rz2))

            # Connect rings with quads
            for i in range(n_sides):
                j = (i + 1) % n_sides
                v00 = ring1[i]
                v01 = ring2[i]
                v11 = ring2[j]
                v10 = ring1[j]
                faces.append([v00, v01, v11, v10])

            if caps:
                # Simple fan triangulation for caps (optional)
                c1 = (p.x, p.y, p.z)
                c2 = (q.x, q.y, q.z)
                for i in range(1, n_sides - 1):
                    faces.append([ring1[0], ring1[i], ring1[i + 1]])
                    faces.append([ring2[0], ring2[i + 1], ring2[i]])

    if not faces:
        return ax

    poly = Poly3DCollection(faces)
    poly.set_facecolor(COLORS["segment"])
    poly.set_alpha(alpha)
    if edgecolor is None:
        poly.set_edgecolor("none")
    else:
        poly.set_edgecolor(edgecolor)
    poly.set_linewidth(linewidth)
    ax.add_collection3d(poly)

    # Set axes limits based on geometry
    xs = [v[0] for face in faces for v in face]
    ys = [v[1] for face in faces for v in face]
    zs = [v[2] for face in faces for v in face]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    return ax


def plot_morphology_frusta_3d(
    obj,
    *,
    n_sides: int = 16,
    color: str = COLORS["segment"],
    alpha: float = 0.8,
    edgecolor: Optional[str] = None,
    linewidth: float = 0.0,
    overlays: Optional[dict[str, Sequence[object]]] = None,
    overlay_styles: Optional[dict[str, dict]] = None,
    backend: str = "auto",
    layout_kwargs: Optional[dict] = None,
    radius_scale: float = 1.0,
    caps: bool = False,
    isometry: Optional["A.isometry"] = None,
    config: Optional["VizConfig"] = None,
):
    """Convenience wrapper: frusta rendering with optional overlays, MPL or Plotly.

    Parameters
    ----------
    obj : arbor.morphology | loaded_morphology | cable_cell
        Object whose morphology will be rendered as frusta.
    overlays : dict[str, Sequence[location | str | (x,y,z)]], optional
        Overlay points/locations drawn on top of the frusta.
    overlay_styles : dict[str, dict], optional
        Styling per overlay label. For Matplotlib: {'color': 'r', 'size': 10}.
        For Plotly: {'color': 'r', 'size': 3}.
    backend : {'auto','plotly','mpl'}
        Choose rendering backend.
    layout_kwargs : dict, optional
        Additional layout kwargs for Plotly (e.g., scene, title).
    radius_scale : float
        Uniform multiplier applied to all segment radii for visualization (default 1.0).
    caps : bool
        If True, add end-caps to segments.
    isometry : arbor.isometry, optional
        Used only for mapping overlay locations to points; geometry itself is drawn in native coords.

    Returns
    -------
    Matplotlib Axes3D (backend='mpl') or plotly.graph_objects.Figure (backend='plotly').
    """
    morph = _as_morphology(obj.morphology if hasattr(obj, "morphology") else obj)
    overlays = overlays or {}
    overlay_styles = overlay_styles or {}
    chosen = _default_backend() if backend == "auto" else backend

    if chosen == "plotly":
        if go is None:
            raise ImportError(
                "Plotly is not installed but backend='plotly' was requested."
            )
        mesh = plotly_morphology_frusta_trace(
            morph,
            n_sides=n_sides,
            color=color,
            opacity=alpha,
            radius_scale=radius_scale,
            caps=caps,
        )
        overlay_traces = []
        for label, items in overlays.items():
            locs, pts = _split_overlay_items(morph, items)
            style = overlay_styles.get(label, {})
            c = style.get("color", "red")
            size = style.get("size", 3)
            if locs:
                overlay_traces.append(
                    plotly_locations_trace(
                        morph, locs, isometry=isometry, color=c, size=size, name=label
                    )
                )
            if pts:
                overlay_traces.append(
                    plotly_points_trace(pts, color=c, size=size, name=f"{label}_pts")
                )
        fig = go.Figure(data=[mesh] + overlay_traces)
        layout_kwargs = layout_kwargs or {}
        scene = _default_plotly_scene(layout_kwargs)
        fig.update_layout(scene=scene, **layout_kwargs)
        _apply_viz_config(fig, config)
        return fig

    # Matplotlib path
    ax = draw_morphology_frusta(
        morph,
        n_sides=n_sides,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        radius_scale=radius_scale,
        caps=caps,
    )
    for label, items in overlays.items():
        locs, pts = _split_overlay_items(morph, items)
        style = overlay_styles.get(label, {})
        size = style.get("size", 10)
        color_pt = style.get("color", "r")
        if locs:
            scatter_locations(
                morph,
                locs,
                ax=ax,
                color=color_pt,
                size=size,
                label=label,
                isometry=isometry,
            )
        if pts:
            scatter_points(pts, ax=ax, color=color_pt, size=size, label=f"{label}_pts")
    if config is not None:
        if config.figsize is not None:
            try:
                ax.figure.set_size_inches(*config.figsize)
            except Exception:
                pass
        if config.title is not None:
            try:
                ax.set_title(config.title)
            except Exception:
                pass
    return ax


def terminals_from_morphology(morph: "A.morphology") -> List["A.location"]:
    """Return terminal locations (branch-end locations) for the given morphology.

    Each terminal is represented as `location(branch_id, x=1.0)` for branches
    with no children.
    """
    morph = _as_morphology(morph)
    locs: List["A.location"] = []
    for b in range(morph.num_branches):
        if not morph.branch_children(b):
            locs.append(A.location(b, 1.0))
    return locs


def locations_to_points(
    morph: "A.morphology",
    locs: Sequence["A.location"],
    isometry: Optional["A.isometry"] = None,
) -> List["A.mpoint"]:
    """Map a sequence of locations to 3D points with place_pwlin.

    Parameters
    ----------
    morph : arbor.morphology
    locs : sequence of arbor.location
    isometry : arbor.isometry, optional
        If provided, the morphology is transformed before mapping locations.

    Returns
    -------
    list[arbor.mpoint]
    """
    morph = _as_morphology(morph)
    pw = (
        A.place_pwlin(morph, isometry) if isometry is not None else A.place_pwlin(morph)
    )
    pts: List["A.mpoint"] = []
    for l in locs:
        pts.append(pw.at(l))  # any corresponding 3D point
    return pts


def scatter_locations(
    morph: "A.morphology",
    locs: Sequence["A.location"],
    ax=None,
    *,
    color: str = COLORS["synapse_point"],
    size: float = 10,
    label: Optional[str] = None,
    isometry: Optional["A.isometry"] = None,
):
    """Scatter-plot a set of locations on top of a morphology by mapping to 3D points.

    Returns the axes used for plotting.
    """
    ax = _ensure_axes3d(ax)
    morph = _as_morphology(morph)
    pts = locations_to_points(morph, locs, isometry=isometry)
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    ax.scatter(xs, ys, zs, c=color, s=size, label=label)
    if label:
        ax.legend()
    return ax


def scatter_points(
    points: Sequence[object],
    ax=None,
    *,
    color: str = COLORS["synapse_point"],
    size: float = 10,
    label: Optional[str] = None,
):
    """Scatter-plot arbitrary arbor.mpoint coordinates on a 3D axes.

    Useful if you already mapped locations to points, or want to overlay
    custom coordinates.
    """
    ax = _ensure_axes3d(ax)
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for p in points:
        if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
            xs.append(float(p.x))
            ys.append(float(p.y))
            zs.append(float(p.z))
        else:
            try:
                x, y, z = p  # type: ignore[misc]
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
            except Exception:
                continue
    ax.scatter(xs, ys, zs, c=color, s=size, label=label)
    if label:
        ax.legend()
    return ax


def plot_morph_and_locations(
    morph: "A.morphology",
    locs: Sequence["A.location"],
    *,
    ax=None,
    line_kwargs: Optional[dict] = None,
    scatter_kwargs: Optional[dict] = None,
    isometry: Optional["A.isometry"] = None,
):
    """Draw morphology and overlay a set of locations on a single 3D axes.

    Parameters
    ----------
    morph : arbor.morphology
    locs : sequence[arbor.location]
    ax : 3D axes or None
    line_kwargs : dict
        Passed to `draw_morphology` (e.g., color, linewidth, alpha).
    scatter_kwargs : dict
        Passed to `scatter_locations` (e.g., color, size, label).
    isometry : arbor.isometry or None
        Geometry transform used when mapping locations to 3D points.

    Returns
    -------
    The 3D axes used for drawing.
    """
    line_kwargs = line_kwargs or {}
    scatter_kwargs = scatter_kwargs or {}

    ax = draw_morphology(morph, ax=ax, **line_kwargs)
    ax = scatter_locations(morph, locs, ax=ax, isometry=isometry, **scatter_kwargs)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    return ax


# ---------- Plotly API ----------


def plotly_morphology_traces(
    morph: "A.morphology",
    *,
    color: str = COLORS["skeleton"],
    width: float = 2,
    opacity: float = 1.0,
    name: str = "morphology",
):
    """Return Plotly Scatter3d traces representing the morphology as line segments.

    Returns a list of traces, one per segment.
    """
    if go is None:
        raise ImportError(
            "Plotly is not installed. Install with `pip install plotly` to use plotly_* functions."
        )
    morph = _as_morphology(morph)
    traces = []
    for b in range(morph.num_branches):
        segs = morph.branch_segments(b)
        for s in segs:
            p, q = s.prox, s.dist
            traces.append(
                go.Scatter3d(
                    x=[p.x, q.x],
                    y=[p.y, q.y],
                    z=[p.z, q.z],
                    mode="lines",
                    line=dict(color=color, width=width),
                    opacity=opacity,
                    hoverinfo="none",
                    showlegend=False,
                    name=name,
                )
            )
    return traces


def plotly_locations_trace(
    morph: "A.morphology",
    locs: Sequence["A.location"],
    *,
    isometry: Optional["A.isometry"] = None,
    color: str = COLORS["synapse_point"],
    size: float = 3,
    name: str = "locations",
):
    """Return a Plotly Scatter3d trace for a set of locations mapped to 3D points."""
    if go is None:
        raise ImportError(
            "Plotly is not installed. Install with `pip install plotly` to use plotly_* functions."
        )
    pts = locations_to_points(morph, locs, isometry=isometry)
    return go.Scatter3d(
        x=[p.x for p in pts],
        y=[p.y for p in pts],
        z=[p.z for p in pts],
        mode="markers",
        marker=dict(size=size, color=color),
        name=name,
    )


def plotly_points_trace(
    points: Sequence[object],
    *,
    color: str = COLORS["synapse_point"],
    size: float = 3,
    name: str = "points",
):
    """Return a Plotly Scatter3d trace for raw 3D points (mpoint or (x,y,z))."""
    if go is None:
        raise ImportError(
            "Plotly is not installed. Install with `pip install plotly` to use plotly_* functions."
        )
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for p in points:
        if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
            xs.append(float(p.x))
            ys.append(float(p.y))
            zs.append(float(p.z))
        else:
            try:
                x, y, z = p  # type: ignore[misc]
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
            except Exception:
                continue
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=size, color=color),
        name=name,
    )


def plotly_morphology_frusta_trace(
    morph: "A.morphology",
    *,
    n_sides: int = 16,
    color: str = COLORS["segment"],
    opacity: float = 1.0,
    name: str = "morph_frusta",
    radius_scale: float = 1.0,
    caps: bool = False,
):
    """Return a Plotly Mesh3d trace representing the morphology as frusta.

    Each Arbor segment is rendered as a truncated cone between its endpoints using
    their respective radii. `radius_scale` multiplies all radii for visualization.
    Quads are triangulated into two faces for Mesh3d.
    """
    if go is None:
        raise ImportError(
            "Plotly is not installed. Install with `pip install plotly` to use plotly_* functions."
        )
    morph = _as_morphology(morph)

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    ii: List[int] = []
    jj: List[int] = []
    kk: List[int] = []

    # Precompute circle angles
    twopi = 2.0 * math.pi
    angles = [twopi * k / n_sides for k in range(n_sides)]

    def add_vertex(x: float, y: float, z: float) -> int:
        xs.append(x)
        ys.append(y)
        zs.append(z)
        return len(xs) - 1

    for b in range(morph.num_branches):
        for s in morph.branch_segments(b):
            p, q = s.prox, s.dist  # mpoint endpoints
            r1 = max(1e-3, radius_scale * _mpoint_radius(p))
            r2 = max(1e-3, radius_scale * _mpoint_radius(q))

            dx, dy, dz = (q.x - p.x), (q.y - p.y), (q.z - p.z)
            _, a, bvec = _orthonormal_frame_from_axis(dx, dy, dz)
            ax_, ay_, az_ = a
            bx_, by_, bz_ = bvec

            # Build rings at each end; store their vertex indices
            ring1_idx: List[int] = []
            ring2_idx: List[int] = []
            for th in angles:
                ct = math.cos(th)
                st = math.sin(th)
                rx1 = r1 * (ct * ax_ + st * bx_)
                ry1 = r1 * (ct * ay_ + st * by_)
                rz1 = r1 * (ct * az_ + st * bz_)
                rx2 = r2 * (ct * ax_ + st * bx_)
                ry2 = r2 * (ct * ay_ + st * by_)
                rz2 = r2 * (ct * az_ + st * bz_)
                ring1_idx.append(add_vertex(p.x + rx1, p.y + ry1, p.z + rz1))
                ring2_idx.append(add_vertex(q.x + rx2, q.y + ry2, q.z + rz2))

            # Connect rings with triangles for each quad
            for i in range(n_sides):
                j = (i + 1) % n_sides
                v00 = ring1_idx[i]
                v01 = ring2_idx[i]
                v11 = ring2_idx[j]
                v10 = ring1_idx[j]
                # Two triangles: (v00, v01, v11) and (v00, v11, v10)
                ii.append(v00)
                jj.append(v01)
                kk.append(v11)
                ii.append(v00)
                jj.append(v11)
                kk.append(v10)

            if caps:
                # Optional end-caps as triangle fans
                c1 = add_vertex(p.x, p.y, p.z)
                c2 = add_vertex(q.x, q.y, q.z)
                for i in range(1, n_sides - 1):
                    ii.append(c1)
                    jj.append(ring1_idx[i])
                    kk.append(ring1_idx[i + 1])
                    ii.append(c2)
                    jj.append(ring2_idx[i + 1])
                    kk.append(ring2_idx[i])

    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=ii, j=jj, k=kk, color=color, opacity=opacity, name=name
    )


# ---------- Arbor-centric helpers ----------


_LOC_RE = re.compile(
    r"^\(location\s+(?P<branch>\d+)\s+(?P<pos>\d*\.?\d+(?:[eE][+-]?\d+)?)\)$"
)


def parse_location_expr(expr: str) -> Optional[Tuple[int, float]]:
    """Parse a simple locset expression of the form '(location <branch> <pos>)'.

    Returns (branch, pos) or None if the expression does not match.
    This is intentionally simple and only supports explicit single-location expressions,
    which are commonly produced by mapping SWC coordinates via place_pwlin().
    """
    m = _LOC_RE.match(expr.strip())
    if not m:
        return None
    return int(m.group("branch")), float(m.group("pos"))


def locations_from_location_exprs(exprs: Sequence[str]) -> List["A.location"]:
    """Convert a sequence of '(location b x)' expressions into Arbor locations.

    Any expressions that do not match the simple '(location ...)' form are ignored.
    """
    locs: List["A.location"] = []
    for e in exprs:
        parsed = parse_location_expr(e)
        if parsed is None:
            continue
        b, x = parsed
        locs.append(A.location(b, x))
    return locs


def plot_cable_cell_with_locations(
    cell: "A.cable_cell",
    *,
    morph_color: str = "k",
    morph_linewidth: float = 1.0,
    morph_alpha: float = 0.6,
    overlays: Optional[dict[str, Sequence[str]]] = None,
    overlay_styles: Optional[dict[str, dict]] = None,
    ax=None,
):
    """Plot an Arbor cable_cell morphology and overlay explicit '(location ...)' placements.

    Parameters
    ----------
    cell : arbor.cable_cell
        The cell to visualize. Its morphology is drawn.
    overlays : dict[str, Sequence[str]]
        Optional mapping from a label (e.g., 'syn', 'gj') to a list of explicit
        location expressions '(location b x)'. These are converted to Arbor locations
        and scattered with styles from `overlay_styles` if provided.
    overlay_styles : dict[str, dict]
        Matplotlib scatter kwargs per overlay label, e.g., {'syn': {'color': 'r', 'size': 12}}.
    ax : matplotlib 3D axes, optional
        If None, a new 3D axes is created.

    Returns
    -------
    ax : 3D axes used for plotting.

    Notes
    -----
    This function focuses on explicit single-location overlays. Complex locsets
    (e.g., '(uniform ...)') are not evaluated here; pass resolved '(location ...)' strings
    if you want them included. For uniform placements, consider recording the resolved
    locations during model construction for visualization.
    """
    # Draw morphology first
    morph = cell.morphology
    ax = draw_morphology(
        morph, ax=ax, color=morph_color, linewidth=morph_linewidth, alpha=morph_alpha
    )

    if overlays:
        overlay_styles = overlay_styles or {}
        for label, exprs in overlays.items():
            locs = locations_from_location_exprs(exprs)
            style = overlay_styles.get(label, {})
            # Map 'size' -> s for matplotlib scatter
            size = style.pop("size", 10)
            color = style.pop("color", "r")
            ax = scatter_locations(
                morph, locs, ax=ax, color=color, size=size, label=label
            )
    return ax
