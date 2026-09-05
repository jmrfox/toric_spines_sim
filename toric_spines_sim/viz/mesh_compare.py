"""Plotly overlays for comparing meshes with skeletons and fitted SWCs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np

PathLike = Union[str, Path]


def read_polylines_txt(path: PathLike) -> list[np.ndarray]:
    """Parse a pymcfs/mascaf polylines text file.

    Each non-empty line is ``N x1 y1 z1 ... xN yN zN``.
    """
    polylines: list[np.ndarray] = []
    text = Path(path).read_text()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            n_points = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"Invalid polylines line {line_number}: expected leading integer count"
            ) from exc
        coords = parts[1:]
        if len(coords) != 3 * n_points:
            raise ValueError(
                f"Invalid polylines line {line_number}: expected {3 * n_points} "
                f"coordinates for N={n_points}, got {len(coords)}"
            )
        values = np.asarray([float(v) for v in coords], dtype=float).reshape(n_points, 3)
        polylines.append(values)
    return polylines


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotly is required for mesh comparison figures."
        ) from exc
    return go


def _load_trimesh(mesh_path: PathLike):
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "trimesh is required for mesh comparison figures "
            "(install with `uv sync`)."
        ) from exc

    loaded = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry in mesh scene: {mesh_path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(loaded)} for {mesh_path}")
    return loaded


def _mesh_surface_trace(mesh, *, opacity: float = 0.35, name: str = "mesh"):
    go = _require_plotly()
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=opacity,
        color="#8aa0b4",
        name=name,
        flatshading=True,
        showlegend=True,
    )


def _polyline_traces(
    polylines: Sequence[np.ndarray],
    *,
    color: str = "#c0392b",
    width: float = 4.0,
    name: str = "skeleton",
):
    go = _require_plotly()
    traces = []
    for index, pts in enumerate(polylines):
        if len(pts) == 0:
            continue
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="lines",
                line=dict(color=color, width=width),
                name=name if index == 0 else f"{name}_{index}",
                showlegend=index == 0,
            )
        )
    return traces


def _default_layout(title: str):
    return dict(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )


def figure_mesh_and_skeleton(
    mesh_path: PathLike,
    polylines_path: PathLike,
    *,
    mesh_opacity: float = 0.35,
):
    """Build a Plotly figure overlaying a mesh with its skeleton polylines."""
    go = _require_plotly()
    mesh_path = Path(mesh_path)
    polylines_path = Path(polylines_path)
    mesh = _load_trimesh(mesh_path)
    polylines = read_polylines_txt(polylines_path)

    fig = go.Figure()
    fig.add_trace(_mesh_surface_trace(mesh, opacity=mesh_opacity, name=mesh_path.name))
    for trace in _polyline_traces(polylines):
        fig.add_trace(trace)
    fig.update_layout(
        **_default_layout(f"{mesh_path.stem}: mesh + skeleton")
    )
    return fig


def figure_mesh_and_swc(
    mesh_path: PathLike,
    swc_path: PathLike,
    *,
    mesh_opacity: float = 0.35,
    show_centroid: bool = False,
    plot_endcaps: bool = False,
    cable_opacity: float = 0.8,
    cable_color: str = "lightblue",
    sides: int = 16,
    neck_points: Sequence[Sequence[float]] | None = None,
    neck_point_size: float = 6.0,
    neck_point_color: str = "#e74c3c",
):
    """Build a Plotly figure overlaying a mesh with a fitted cable model.

    Uses ``swctools.plot_model`` (same path as mascaf demos) so the SWC is
    rendered as frusta with optional terminal endcaps, not just the centroid
    graph.

    Parameters
    ----------
    neck_points
        Optional XYZ points (same units as the mesh/SWC) drawn as markers,
        typically pixel-space neckpoints from ``*_neckpoint.txt``.
    """
    from swctools import SWCModel, plot_model

    mesh_path = Path(mesh_path)
    swc_path = Path(swc_path)
    mesh = _load_trimesh(mesh_path)
    model = SWCModel.from_swc_file(str(swc_path))

    fig = plot_model(
        swc_model=model,
        slider=False,
        title=f"{mesh_path.stem}: mesh + cable",
        show_axes=False,
        show_frusta=True,
        show_centroid=show_centroid,
        plot_endcaps=plot_endcaps,
        opacity=cable_opacity,
        color=cable_color,
        sides=sides,
        width=1200,
        height=900,
    )
    mesh_trace = _mesh_surface_trace(mesh, opacity=mesh_opacity, name=mesh_path.name)
    go = _require_plotly()
    traces = [mesh_trace, *fig.data]
    if neck_points:
        pts = np.asarray(neck_points, dtype=float).reshape(-1, 3)
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=neck_point_size,
                    color=neck_point_color,
                    symbol="diamond",
                    line=dict(width=1, color="#922b21"),
                ),
                name="neckpoint",
                showlegend=True,
            )
        )
    combined = go.Figure(data=traces, layout=fig.layout)
    title = f"{mesh_path.stem}: mesh + cable"
    if neck_points:
        title = f"{mesh_path.stem}: mesh + cable + neckpoint(s)"
    combined.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
    )
    return combined
