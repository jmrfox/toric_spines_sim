"""3D voltage animation on morphology frusta with optional synapse overlay."""

from __future__ import annotations

import logging
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from swctools import FrustaSet, PointSet

if TYPE_CHECKING:
    from toric_spines_sim.simulation.results import SimulationResults

logger = logging.getLogger(__name__)

ExtraFrameTraces = Callable[[int], Sequence[Any]]
DEFAULT_SYNAPSE_FLASH_DURATION_MS = 1.0


def synapse_flash_active_mask(
    frame_times_ms: np.ndarray,
    event_times_ms: np.ndarray,
    flash_duration_ms: float,
) -> np.ndarray:
    """Return per-frame booleans for event-centered synapse flash windows.

    A frame at time ``t`` is active when ``|t - event| <= duration / 2`` for
    any input event at ``event``. For example, an event at 10 ms with
    duration 10 ms is active on ``[5, 15]`` ms.
    """
    t = np.asarray(frame_times_ms, dtype=float)
    if event_times_ms.size == 0:
        return np.zeros(len(t), dtype=bool)
    events = np.asarray(event_times_ms, dtype=float)
    half = flash_duration_ms / 2.0
    offset = t[:, None] - events[None, :]
    return np.any((offset >= -half) & (offset <= half), axis=1)


def _default_scene(show_axes: bool) -> dict:
    scene = dict(
        aspectmode="data",
        xaxis=dict(title="x (µm)"),
        yaxis=dict(title="y (µm)"),
        zaxis=dict(title="z (µm)"),
    )
    if not show_axes:
        scene["xaxis"]["visible"] = False
        scene["yaxis"]["visible"] = False
        scene["zaxis"]["visible"] = False
    return scene


def _animation_layout_kwargs(
    *,
    title: Optional[str],
    show_axes: bool,
    frame_duration: int,
    n_frames: int,
    time_domain_arr: np.ndarray,
) -> dict:
    """Layout for animated 3D figures with room for colorbar and time slider."""
    slider_steps = [
        {
            "label": f"{time_domain_arr[t]:.1f}",
            "method": "animate",
            "args": [
                [f"frame_{t}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": frame_duration},
                    "transition": {"duration": 0},
                },
            ],
        }
        for t in range(n_frames)
    ]
    return dict(
        template="plotly_white",
        title=title,
        width=1200,
        height=900,
        margin=dict(l=10, r=120, t=60, b=120),
        showlegend=False,
        uirevision="animation-camera",
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "t = ", "suffix": " ms", "visible": True},
                "steps": slider_steps,
                "x": 0.1,
                "y": 0,
                "transition": {"duration": 0},
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "&#9654;",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": frame_duration},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "&#9724;",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        scene=_default_scene(show_axes),
    )


def _frusta_colorbar(colorbar_title: Optional[str]) -> dict:
    return dict(
        title=dict(text=colorbar_title or "Voltage (mV)", side="right"),
        len=0.6,
        thickness=18,
        x=1.02,
        xanchor="left",
    )


def _frusta_mesh_trace(
    x,
    y,
    z,
    i,
    j,
    k,
    *,
    intensity: Sequence[float],
    colorscale: str,
    cmin: float,
    cmax: float,
    opacity: float,
    flatshading: bool,
    colorbar_title: Optional[str] = None,
    showscale: bool = True,
) -> go.Mesh3d:
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        opacity=opacity,
        flatshading=flatshading,
        intensity=list(intensity),
        intensitymode="cell",
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        showscale=showscale,
        name="frusta",
        colorbar=_frusta_colorbar(colorbar_title),
        hovertemplate="Voltage: %{intensity:.2f} mV<extra></extra>",
    )


def _build_frusta_timeseries_figure(
    frusta: FrustaSet,
    time_domain: Sequence[float],
    amplitudes: Sequence[Sequence[float]],
    *,
    colorscale: str = "Viridis",
    clim: Optional[Tuple[float, float]] = None,
    opacity: float = 0.8,
    flatshading: bool = True,
    radius_scale: float = 1.0,
    fps: int = 30,
    stride: int = 1,
    title: str = "Animation",
    colorbar_title: Optional[str] = None,
    show_axes: bool = True,
    extra_traces: Optional[Sequence[Any]] = None,
    extra_frame_traces: Optional[ExtraFrameTraces] = None,
) -> go.Figure:
    """Build a frusta timeseries animation figure without writing HTML."""
    time_domain_arr = np.asarray(time_domain, dtype=float)
    amplitudes_arr = np.asarray(amplitudes, dtype=float)

    if stride > 1:
        time_domain_arr = time_domain_arr[::stride]
        amplitudes_arr = amplitudes_arr[::stride]
        logger.info(
            "Applied stride=%d: reduced to %d frames",
            stride,
            len(time_domain_arr),
        )

    fr = frusta if radius_scale == 1.0 else frusta.scaled(radius_scale)
    x, y, z, i, j, k = fr.to_mesh3d_arrays()
    slices = list(fr.frustum_face_slices_map().values())

    if len(amplitudes_arr) == 0:
        raise ValueError("amplitudes must have at least one time step")
    if len(time_domain_arr) != len(amplitudes_arr):
        raise ValueError(
            f"time_domain length ({len(time_domain_arr)}) must match "
            f"amplitudes time axis length ({len(amplitudes_arr)})"
        )

    n_frames = len(amplitudes_arr)
    n_frusta = fr.n_frusta
    if any(len(vt) != n_frusta for vt in amplitudes_arr):
        raise ValueError(
            f"each time step must have {n_frusta} values (frusta.n_frusta)"
        )

    def faces_intensity(vt: Sequence[float]) -> list[float]:
        arr: list[float] = []
        for (_start, count), val in zip(slices, vt):
            arr.extend([float(val)] * count)
        return arr

    if clim is None:
        cmin = float(np.nanmin(amplitudes_arr))
        cmax = float(np.nanmax(amplitudes_arr))
    else:
        cmin, cmax = clim

    logger.info("Color limits: [%.3f, %.3f]", cmin, cmax)

    static_traces = list(extra_traces or [])
    initial_extra = (
        list(extra_frame_traces(0)) if extra_frame_traces is not None else []
    )

    mesh = _frusta_mesh_trace(
        x,
        y,
        z,
        i,
        j,
        k,
        intensity=faces_intensity(amplitudes_arr[0]),
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        opacity=opacity,
        flatshading=flatshading,
        colorbar_title=colorbar_title,
        showscale=True,
    )

    frames: list[go.Frame] = []
    for frame_idx, vt in enumerate(amplitudes_arr):
        frame_data: list[Any] = [
            _frusta_mesh_trace(
                x,
                y,
                z,
                i,
                j,
                k,
                intensity=faces_intensity(vt),
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                opacity=opacity,
                flatshading=flatshading,
                colorbar_title=colorbar_title,
                showscale=True,
            )
        ]
        frame_data.extend(static_traces)
        if extra_frame_traces is not None:
            frame_data.extend(extra_frame_traces(frame_idx))
        frames.append(go.Frame(name=f"frame_{frame_idx}", data=frame_data))

    frame_duration = int(1000 / max(1, fps))
    fig = go.Figure(data=[mesh, *static_traces, *initial_extra], frames=frames)
    fig.update_layout(
        **_animation_layout_kwargs(
            title=title,
            show_axes=show_axes,
            frame_duration=frame_duration,
            n_frames=n_frames,
            time_domain_arr=time_domain_arr,
        )
    )

    logger.info(
        "Animation figure created with %d frames at %d fps", n_frames, fps
    )
    return fig


def _synapse_mesh_trace(
    syn_x,
    syn_y,
    syn_z,
    syn_i,
    syn_j,
    syn_k,
    facecolor: Sequence[str],
) -> go.Mesh3d:
    return go.Mesh3d(
        x=syn_x,
        y=syn_y,
        z=syn_z,
        i=syn_i,
        j=syn_j,
        k=syn_k,
        facecolor=list(facecolor),
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=0.8, diffuse=0.2),
        hoverinfo="skip",
        showscale=False,
        name="synapses",
    )


@dataclass(frozen=True)
class AnimationFrameCache:
    """Precomputed frame data for 3D voltage animation."""

    time_ms: np.ndarray
    amplitudes: np.ndarray
    mesh_x: np.ndarray
    mesh_y: np.ndarray
    mesh_z: np.ndarray
    mesh_i: np.ndarray
    mesh_j: np.ndarray
    mesh_k: np.ndarray
    face_slices: tuple[tuple[int, int], ...]
    cmin: float
    cmax: float
    colorscale: str
    opacity: float
    flatshading: bool
    colorbar_title: Optional[str]
    show_axes: bool
    synapse_coords: Optional[tuple[tuple[float, float, float], ...]] = None
    synapse_labels: Optional[tuple[str, ...]] = None
    synapse_mesh_arrays: Optional[tuple[Any, ...]] = None
    synapse_facecolors_per_frame: Optional[tuple[list[str], ...]] = None
    synapse_facecolor_indices: Optional[tuple[np.ndarray, ...]] = None
    synapse_color_palette: Optional[tuple[str, ...]] = None

    @property
    def n_frames(self) -> int:
        return len(self.time_ms)

    def faces_intensity(self, frame_idx: int) -> list[float]:
        vt = self.amplitudes[frame_idx]
        arr: list[float] = []
        for (_start, count), val in zip(self.face_slices, vt):
            arr.extend([float(val)] * count)
        return arr

    def faces_intensity_array(self, frame_idx: int) -> np.ndarray:
        """Return the 3D frusta intensity as a flat Float32 array."""
        vt = self.amplitudes[frame_idx]
        counts = np.array([count for _, count in self.face_slices], dtype=np.int32)
        return np.repeat(vt, counts).astype(np.float32)

    def synapse_facecolor_hex(self, frame_idx: int) -> list[str]:
        """Expand palette-index facecolors to full hex strings."""
        if self.synapse_facecolor_indices is None:
            return list(self.synapse_facecolors_per_frame[frame_idx])
        indices = self.synapse_facecolor_indices[frame_idx]
        palette = self.synapse_color_palette
        return [palette[idx] for idx in indices]

    def build_3d_clientside_bundle(self) -> dict[str, Any]:
        """Return JSON-serializable per-frame data for clientside 3D updates."""
        bundle: dict[str, Any] = {
            "time_ms": self.time_ms.tolist(),
            "n_frames": self.n_frames,
            "mesh_intensity": [
                self.faces_intensity_array(i).tolist()
                for i in range(self.n_frames)
            ],
        }
        if self.synapse_facecolor_indices is not None:
            bundle["synapse_facecolor_indices"] = [
                arr.tolist() for arr in self.synapse_facecolor_indices
            ]
            bundle["synapse_color_palette"] = list(self.synapse_color_palette)
        elif self.synapse_facecolors_per_frame is not None:
            bundle["synapse_facecolor"] = list(self.synapse_facecolors_per_frame)
        return bundle


def build_single_frame_figure(
    cache: AnimationFrameCache,
    frame_idx: int,
    *,
    title: Optional[str] = None,
    template: str = "plotly_white",
    autosize: bool = False,
) -> go.Figure:
    """Build a single-frame 3D voltage figure from cached animation data."""
    if frame_idx < 0 or frame_idx >= cache.n_frames:
        raise IndexError(f"frame_idx {frame_idx} out of range [0, {cache.n_frames})")

    traces: list[Any] = [
        _frusta_mesh_trace(
            cache.mesh_x,
            cache.mesh_y,
            cache.mesh_z,
            cache.mesh_i,
            cache.mesh_j,
            cache.mesh_k,
            intensity=cache.faces_intensity(frame_idx),
            colorscale=cache.colorscale,
            cmin=cache.cmin,
            cmax=cache.cmax,
            opacity=cache.opacity,
            flatshading=cache.flatshading,
            colorbar_title=cache.colorbar_title,
            showscale=True,
        )
    ]

    if (
        cache.synapse_coords is not None
        and cache.synapse_labels is not None
        and cache.synapse_mesh_arrays is not None
        and (cache.synapse_facecolors_per_frame is not None
             or cache.synapse_facecolor_indices is not None)
    ):
        traces.append(
            _synapse_hover_trace(cache.synapse_coords, cache.synapse_labels)
        )
        syn_x, syn_y, syn_z, syn_i, syn_j, syn_k = cache.synapse_mesh_arrays
        traces.append(
            _synapse_mesh_trace(
                syn_x,
                syn_y,
                syn_z,
                syn_i,
                syn_j,
                syn_k,
                cache.synapse_facecolor_hex(frame_idx),
            )
        )

    t_ms = float(cache.time_ms[frame_idx])
    fig = go.Figure(data=traces)
    layout_kwargs: dict[str, Any] = dict(
        template=template,
        title=title if title is not None else f"t = {t_ms:.1f} ms",
        margin=dict(l=10, r=120, t=50, b=10),
        showlegend=False,
        scene=_default_scene(cache.show_axes),
        uirevision="dashboard-camera",
        datarevision=frame_idx,
    )
    if autosize:
        layout_kwargs["autosize"] = True
    else:
        layout_kwargs["width"] = 1200
        layout_kwargs["height"] = 700
    fig.update_layout(**layout_kwargs)
    return fig


def build_3d_template_figure(
    cache: AnimationFrameCache,
    *,
    template: str = "plotly_white",
    autosize: bool = True,
) -> go.Figure:
    """Build the base 3D figure (frame 0); update intensity via Patch during playback."""
    return build_single_frame_figure(
        cache,
        0,
        template=template,
        autosize=autosize,
    )


def synapse_mesh_trace_index(cache: AnimationFrameCache) -> int | None:
    """Index of the synapse Mesh3d trace in the 3D figure, if present."""
    if (
        cache.synapse_facecolors_per_frame is not None
        or cache.synapse_facecolor_indices is not None
    ):
        return 2
    return None


def patch_3d_frame(
    cache: AnimationFrameCache,
    frame_idx: int,
    *,
    mesh_trace_idx: int = 0,
    synapse_trace_idx: int | None = None,
) -> Any:
    """Return a Dash Patch updating only 3D voltage intensity and synapse colors."""
    from dash import Patch

    if synapse_trace_idx is None:
        synapse_trace_idx = synapse_mesh_trace_index(cache)

    patch = Patch()
    patch["data"][mesh_trace_idx]["intensity"] = cache.faces_intensity(frame_idx)
    if synapse_trace_idx is not None and (
        cache.synapse_facecolors_per_frame is not None
        or cache.synapse_facecolor_indices is not None
    ):
        patch["data"][synapse_trace_idx]["facecolor"] = cache.synapse_facecolor_hex(
            frame_idx
        )
    t_ms = float(cache.time_ms[frame_idx])
    patch["layout"]["title"]["text"] = f"t = {t_ms:.1f} ms"
    patch["layout"]["datarevision"] = frame_idx
    return patch


def _synapse_hover_trace(
    synapse_coords: Sequence[Tuple[float, float, float]],
    synapse_labels: Sequence[str],
) -> go.Scatter3d:
    xs, ys, zs = zip(*synapse_coords)
    return go.Scatter3d(
        x=list(xs),
        y=list(ys),
        z=list(zs),
        mode="markers",
        marker=dict(size=1, opacity=0),
        text=list(synapse_labels),
        name="synapse_coords",
        hovertemplate=(
            "%{text}<br>x=%{x:.2f} µm<br>y=%{y:.2f} µm<br>z=%{z:.2f} µm<extra></extra>"
        ),
    )


class Animation:
    """Build and save voltage animations from simulation results."""

    def __init__(
        self,
        results: SimulationResults,
        swc_filepath: Union[str, Path],
    ) -> None:
        self.results = results
        self.swc_filepath = Path(swc_filepath)
        self._fig: Optional[go.Figure] = None

    def prepare_frame_cache(
        self,
        *,
        colorscale: str = "Plasma",
        stride: int = 1,
        clim: Optional[Tuple[float, float]] = None,
        opacity: float = 0.8,
        flatshading: bool = True,
        radius_scale: float = 1.0,
        colorbar_title: Optional[str] = None,
        show_axes: bool = True,
        show_synapses: bool = True,
        synapse_ball_size: float = 0.5,
        synapse_stacks: int = 6,
        synapse_slices: int = 12,
        synapse_inactive_color: str = "#8b0000",
        synapse_active_color: str = "#ff4444",
        synapse_flash_duration_ms: float = DEFAULT_SYNAPSE_FLASH_DURATION_MS,
        synapse_colors: Optional[Mapping[str, Tuple[str, str]]] = None,
        frustum_sides: int = 16,
    ) -> AnimationFrameCache:
        """Precompute strided frame data for animation or Dash dashboards."""
        results = self.results
        logger.info(
            "Preparing animation frames from %d voltage traces",
            len(results.voltage_traces.columns),
        )

        frusta = FrustaSet.from_swc_file(
            str(self.swc_filepath), sides=frustum_sides
        )
        logger.debug(
            "Loaded FrustaSet with %d frusta (sides=%d)",
            frusta.n_frusta,
            frustum_sides,
        )

        probe_coords = list(results.record_points.values())
        frustum_indices = [
            frusta.nearest_frustum_index(coord) for coord in probe_coords
        ]
        probe_labels = list(results.record_points.keys())

        time_domain_ms = results.voltage_traces.as_units("ms").index.values
        n_timepoints = len(time_domain_ms)

        n_frusta = frusta.n_frusta
        amplitudes = np.full((n_timepoints, n_frusta), np.nan)
        for probe_label, frustum_idx in zip(probe_labels, frustum_indices):
            if probe_label in results.voltage_traces.columns:
                amplitudes[:, frustum_idx] = results.voltage_traces[probe_label].values

        logger.info(
            "Mapped %d probes to frusta, preparing %d time steps...",
            len(probe_labels),
            n_timepoints,
        )

        if stride > 1:
            time_domain_ms = time_domain_ms[::stride]
            amplitudes = amplitudes[::stride]
            logger.info(
                "Applied stride=%d: reduced to %d frames",
                stride,
                len(time_domain_ms),
            )

        fr = frusta if radius_scale == 1.0 else frusta.scaled(radius_scale)
        mesh_x, mesh_y, mesh_z, mesh_i, mesh_j, mesh_k = fr.to_mesh3d_arrays()
        face_slices = tuple(fr.frustum_face_slices_map().values())

        if clim is None:
            cmin = float(np.nanmin(amplitudes))
            cmax = float(np.nanmax(amplitudes))
        else:
            cmin, cmax = clim

        logger.info("Color limits: [%.3f, %.3f]", cmin, cmax)

        synapse_coords: Optional[tuple[tuple[float, float, float], ...]] = None
        synapse_labels: Optional[tuple[str, ...]] = None
        synapse_mesh_arrays: Optional[tuple[Any, ...]] = None
        synapse_facecolors: Optional[tuple[list[str], ...]] = None

        if show_synapses:
            overlay = self._prepare_synapse_overlay(
                stride=stride,
                time_domain_ms=results.voltage_traces.as_units("ms").index.values,
                synapse_ball_size=synapse_ball_size,
                synapse_stacks=synapse_stacks,
                synapse_slices=synapse_slices,
                synapse_inactive_color=synapse_inactive_color,
                synapse_active_color=synapse_active_color,
                synapse_flash_duration_ms=synapse_flash_duration_ms,
                synapse_colors=synapse_colors,
            )
            if overlay is not None:
                synapse_coords = tuple(overlay["coords"])
                synapse_labels = tuple(overlay["labels"])
                synapse_mesh_arrays = overlay["mesh_arrays"]
                synapse_facecolors = tuple(overlay["facecolors_per_frame"])
                synapse_facecolor_indices = tuple(
                    overlay["facecolor_indices_per_frame"]
                )
                synapse_color_palette = tuple(overlay["facecolor_palette"])

        return AnimationFrameCache(
            time_ms=np.asarray(time_domain_ms, dtype=float),
            amplitudes=np.asarray(amplitudes, dtype=float),
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            mesh_z=mesh_z,
            mesh_i=mesh_i,
            mesh_j=mesh_j,
            mesh_k=mesh_k,
            face_slices=face_slices,
            cmin=cmin,
            cmax=cmax,
            colorscale=colorscale,
            opacity=opacity,
            flatshading=flatshading,
            colorbar_title=colorbar_title,
            show_axes=show_axes,
            synapse_coords=synapse_coords,
            synapse_labels=synapse_labels,
            synapse_mesh_arrays=synapse_mesh_arrays,
            synapse_facecolors_per_frame=synapse_facecolors,
            synapse_facecolor_indices=synapse_facecolor_indices,
            synapse_color_palette=synapse_color_palette,
        )

    def build(
        self,
        *,
        colorscale: str = "Plasma",
        fps: int = 30,
        stride: int = 1,
        clim: Optional[Tuple[float, float]] = None,
        opacity: float = 0.8,
        flatshading: bool = True,
        radius_scale: float = 1.0,
        title: Optional[str] = None,
        colorbar_title: Optional[str] = None,
        show_axes: bool = True,
        show_synapses: bool = True,
        synapse_ball_size: float = 0.5,
        synapse_stacks: int = 6,
        synapse_slices: int = 12,
        synapse_inactive_color: str = "#8b0000",
        synapse_active_color: str = "#ff4444",
        synapse_flash_duration_ms: float = DEFAULT_SYNAPSE_FLASH_DURATION_MS,
        synapse_colors: Optional[Mapping[str, Tuple[str, str]]] = None,
        frustum_sides: int = 16,
    ) -> go.Figure:
        """Build the animation figure (frusta voltage + optional synapse overlay)."""
        cache = self.prepare_frame_cache(
            colorscale=colorscale,
            stride=stride,
            clim=clim,
            opacity=opacity,
            flatshading=flatshading,
            radius_scale=radius_scale,
            colorbar_title=colorbar_title,
            show_axes=show_axes,
            show_synapses=show_synapses,
            synapse_ball_size=synapse_ball_size,
            synapse_stacks=synapse_stacks,
            synapse_slices=synapse_slices,
            synapse_inactive_color=synapse_inactive_color,
            synapse_active_color=synapse_active_color,
            synapse_flash_duration_ms=synapse_flash_duration_ms,
            synapse_colors=synapse_colors,
            frustum_sides=frustum_sides,
        )

        frusta = FrustaSet.from_swc_file(
            str(self.swc_filepath), sides=frustum_sides
        )

        extra_traces: list[Any] = []
        extra_frame_traces: Optional[ExtraFrameTraces] = None

        if cache.synapse_coords is not None and cache.synapse_mesh_arrays is not None:
            extra_traces.append(
                _synapse_hover_trace(cache.synapse_coords, cache.synapse_labels or ())
            )
            mesh_arrays = cache.synapse_mesh_arrays
            facecolors_per_frame = cache.synapse_facecolors_per_frame or ()

            def _synapse_frame_traces(
                frame_idx: int,
                _arrays=mesh_arrays,
                _facecolors=facecolors_per_frame,
            ) -> list[go.Mesh3d]:
                syn_x, syn_y, syn_z, syn_i, syn_j, syn_k = _arrays
                return [
                    _synapse_mesh_trace(
                        syn_x,
                        syn_y,
                        syn_z,
                        syn_i,
                        syn_j,
                        syn_k,
                        _facecolors[frame_idx],
                    )
                ]

            extra_frame_traces = _synapse_frame_traces

        build_kwargs: dict = dict(
            colorscale=cache.colorscale,
            fps=fps,
            stride=1,
            opacity=cache.opacity,
            flatshading=cache.flatshading,
            radius_scale=radius_scale,
            show_axes=cache.show_axes,
            extra_traces=extra_traces or None,
            extra_frame_traces=extra_frame_traces,
            clim=(cache.cmin, cache.cmax),
        )
        if title is not None:
            build_kwargs["title"] = title
        if cache.colorbar_title is not None:
            build_kwargs["colorbar_title"] = cache.colorbar_title

        fig = _build_frusta_timeseries_figure(
            frusta,
            cache.time_ms,
            cache.amplitudes,
            **build_kwargs,
        )
        self._fig = fig
        return fig

    def save(
        self,
        output_path: Union[str, Path],
        fig: Optional[go.Figure] = None,
        *,
        auto_open: bool = False,
    ) -> go.Figure:
        """Write the animation figure to HTML."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = fig if fig is not None else self._fig
        if fig is None:
            raise RuntimeError("No figure to save; call build() first.")

        fig.write_html(str(output_path), auto_play=False)
        logger.info("Animation saved to %s", output_path)

        if auto_open:
            webbrowser.open(f"file://{output_path.absolute()}")

        return fig

    def create(
        self,
        output_path: Union[str, Path],
        *,
        auto_open: bool = False,
        **build_kwargs,
    ) -> go.Figure:
        """Build and save the animation in one call."""
        fig = self.build(**build_kwargs)
        return self.save(output_path, fig, auto_open=auto_open)

    def _prepare_synapse_overlay(
        self,
        *,
        stride: int,
        time_domain_ms: np.ndarray,
        synapse_ball_size: float,
        synapse_stacks: int,
        synapse_slices: int,
        synapse_inactive_color: str,
        synapse_active_color: str,
        synapse_flash_duration_ms: float,
        synapse_colors: Optional[Mapping[str, Tuple[str, str]]],
    ) -> Optional[dict]:
        results = self.results
        synapse_coords = [syn.location for syn in results.synapses.values()]
        if not synapse_coords:
            logger.warning(
                "show_synapses=True but no synapse location data is available; "
                "skipping synapse overlay."
            )
            return None

        logger.info(
            "Adding synapse overlay for %d synapses (ball_size=%s, flash=%s ms)",
            len(synapse_coords),
            synapse_ball_size,
            synapse_flash_duration_ms,
        )

        point_set = PointSet.from_points(
            synapse_coords,
            base_radius=synapse_ball_size,
            stacks=synapse_stacks,
            slices=synapse_slices,
        )
        mesh_arrays = point_set.to_mesh3d_arrays()
        logger.info(
            "Synapse spheres: %d synapses, %d faces (stacks=%d, slices=%d)",
            len(synapse_coords),
            len(point_set.faces),
            synapse_stacks,
            synapse_slices,
        )

        n_synapses = len(synapse_coords)
        total_faces = len(point_set.faces)
        faces_per_sphere = total_faces // n_synapses
        synapse_labels = list(results.synapses.keys())

        strided_time_domain_ms = (
            time_domain_ms[::stride] if stride > 1 else time_domain_ms
        )
        n_frames = len(strided_time_domain_ms)
        t_grid = np.asarray(strided_time_domain_ms, dtype=float)

        # Build palette: each synapse contributes inactive + active color.
        palette: list[str] = []
        synapse_palette_base: list[int] = []
        for syn_label in synapse_labels:
            if synapse_colors and syn_label in synapse_colors:
                inactive, active = synapse_colors[syn_label]
            else:
                inactive, active = synapse_inactive_color, synapse_active_color
            palette.extend([inactive, active])
            synapse_palette_base.append(len(palette) - 2)

        # Vectorized active mask: n_frames x n_synapses boolean matrix.
        active_mask = np.zeros((n_frames, n_synapses), dtype=bool)
        for syn_idx, syn_label in enumerate(synapse_labels):
            try:
                syn_key = int(syn_label.split("_", 1)[1])
                ts = results.input_events[syn_key]
            except (ValueError, KeyError, IndexError):
                continue
            event_times = np.asarray(
                ts.as_units("ms").index.values, dtype=float
            )
            if event_times.size == 0:
                continue
            active_mask[:, syn_idx] = synapse_flash_active_mask(
                t_grid,
                event_times,
                synapse_flash_duration_ms,
            )

        # Per-frame facecolor index arrays (uint8) for the clientside bundle.
        facecolor_indices_per_frame: list[np.ndarray] = []
        facecolors_per_frame: list[list[str]] = []
        for frame_active in active_mask:
            indices = np.zeros(total_faces, dtype=np.uint8)
            colors: list[str] = []
            for syn_idx, is_active in enumerate(frame_active):
                base = synapse_palette_base[syn_idx]
                idx = base + (1 if is_active else 0)
                indices[syn_idx * faces_per_sphere : (syn_idx + 1) * faces_per_sphere] = idx
                color = palette[idx]
                colors.extend([color] * faces_per_sphere)
            remainder = total_faces - len(colors)
            if remainder > 0:
                last_idx = base + (1 if frame_active[-1] else 0)
                indices[-remainder:] = last_idx
                colors.extend([palette[last_idx]] * remainder)
            facecolor_indices_per_frame.append(indices)
            facecolors_per_frame.append(colors)

        return {
            "coords": synapse_coords,
            "labels": synapse_labels,
            "mesh_arrays": mesh_arrays,
            "facecolors_per_frame": facecolors_per_frame,
            "facecolor_indices_per_frame": facecolor_indices_per_frame,
            "facecolor_palette": palette,
        }
