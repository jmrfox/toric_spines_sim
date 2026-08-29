"""Dash application for synchronized simulation animation dashboards."""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, clientside_callback, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from toric_spines_sim.geometry import map_xyz_to_nearest_probes, neck_point_from_swc_file
from toric_spines_sim.simulation.results import SimulationResults

from .animation import (
    Animation,
    AnimationFrameCache,
    build_3d_template_figure,
    patch_3d_frame,
    synapse_mesh_trace_index,
)
from .simulation_dash_assets import (
    CLIENTSIDE_FRAME_UPDATE,
    CLIENTSIDE_PAUSE_OR_RESET,
    CLIENTSIDE_PLAYBACK_TRANSPORT,
    CLIENTSIDE_PROBE_VISIBILITY,
    CLIENTSIDE_SERVER_DISABLE_TICK,
    CLIENTSIDE_SERVER_SLIDER_SYNC,
    DASHBOARD_LAYOUT_CSS,
)
from .dash_theme import base_layout_styles, dropdown_theme_css, get_theme_tokens, plotly_template_for

logger = logging.getLogger(__name__)

REGION_LABELS = ("Spine", "Neck", "Sink")
REGION_TRACE_STYLE: dict[str, dict[str, str]] = {
    "Spine": {"color": "#1f77b4", "dash": "solid"},
    "Neck": {"color": "#ff7f0e", "dash": "dash"},
    "Sink": {"color": "#2ca2a2", "dash": "dot"},
}
DEFAULT_PROBE_COLUMNS = list(REGION_LABELS)
_CURSOR_COLOR = "#888888"
_CURSOR_WIDTH = 1
_DIM_COLOR = "rgba(150, 150, 150, 0.35)"


@dataclass(frozen=True)
class TraceSeries:
    """A named voltage trace for dashboard plotting."""

    label: str
    t_ms: np.ndarray
    v_mV: np.ndarray
    color: str
    dash: str = "solid"


@dataclass(frozen=True)
class RasterStream:
    """One synapse row in the event raster."""

    syn_idx: int
    label: str
    times_ms: tuple[float, ...]
    color: str


@dataclass
class SimulationDashboardData:
    """Normalized data for the simulation animation dashboard."""

    frame_cache: AnimationFrameCache
    region_traces: dict[str, TraceSeries]
    probe_traces: dict[str, TraceSeries]
    raster_streams: list[RasterStream]
    t_max_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.frame_cache.n_frames

    @property
    def default_probe_columns(self) -> list[str]:
        return list(DEFAULT_PROBE_COLUMNS)

    def trace_options(self) -> list[dict[str, str]]:
        options = [{"label": label, "value": label} for label in REGION_LABELS]
        for label in sorted(self.probe_traces.keys()):
            options.append({"label": label, "value": label})
        return options

    def resolve_traces(self, selected_columns: Sequence[str] | None) -> list[TraceSeries]:
        cols = list(selected_columns) if selected_columns else self.default_probe_columns
        traces: list[TraceSeries] = []
        for col in cols:
            if col in self.region_traces:
                traces.append(self.region_traces[col])
            elif col in self.probe_traces:
                traces.append(self.probe_traces[col])
        return traces


def _axon_color_hex(axon_idx: int) -> str:
    tab10 = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    from matplotlib import colors as mcolors

    return mcolors.to_hex(tab10[axon_idx % len(tab10)])


def _synapse_to_axon_map(axon_synapses: list[list[int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for axon_idx, synapse_indices in enumerate(axon_synapses):
        for syn_idx in synapse_indices:
            mapping[syn_idx] = axon_idx
    return mapping


def _resolve_neck_probe(results: SimulationResults, swc_filepath: Path) -> str:
    neck_xyz = neck_point_from_swc_file(swc_filepath)
    nearest = map_xyz_to_nearest_probes(results.record_points, {"neck_point": neck_xyz})
    return nearest["neck_point"]


def prepare_simulation_dashboard_data(
    results: SimulationResults,
    swc_filepath: str | Path,
    parameters: dict[str, Any],
    *,
    axon_synapses: list[list[int]] | None = None,
    synapse_colors: Mapping[str, Tuple[str, str]] | None = None,
    animation_kwargs: dict[str, Any] | None = None,
) -> SimulationDashboardData:
    """Build dashboard data from simulation results."""
    swc_path = Path(swc_filepath)
    anim_kwargs = dict(animation_kwargs or {})
    anim_kwargs.setdefault("colorscale", "Plasma")
    anim_kwargs.setdefault("show_synapses", True)
    anim_kwargs.setdefault("show_axes", True)
    anim_kwargs.setdefault("colorbar_title", "Voltage (mV)")

    animation = Animation(results, swc_filepath=swc_path)
    frame_cache = animation.prepare_frame_cache(
        synapse_colors=synapse_colors,
        **anim_kwargs,
    )

    v_spine = results.integrate_voltages_by_tag(
        parameters["spine_tag"], method="average"
    )
    v_sink = results.integrate_voltages_by_tag(
        parameters["sink_tag"], method="average"
    )
    neck_probe = _resolve_neck_probe(results, swc_path)
    v_neck = results.voltage_traces[neck_probe]

    region_traces = {
        "Spine": TraceSeries(
            "Spine",
            v_spine.as_units("ms").index.values,
            v_spine.values,
            **REGION_TRACE_STYLE["Spine"],
        ),
        "Neck": TraceSeries(
            "Neck",
            v_neck.as_units("ms").index.values,
            v_neck.values,
            **REGION_TRACE_STYLE["Neck"],
        ),
        "Sink": TraceSeries(
            "Sink",
            v_sink.as_units("ms").index.values,
            v_sink.values,
            **REGION_TRACE_STYLE["Sink"],
        ),
    }

    probe_traces: dict[str, TraceSeries] = {}
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, col in enumerate(results.voltage_traces.columns):
        tsd = results.voltage_traces[col]
        from matplotlib import colors as mcolors

        probe_traces[str(col)] = TraceSeries(
            str(col),
            tsd.as_units("ms").index.values,
            tsd.values,
            color=mcolors.to_hex(palette[idx % len(palette)]),
        )

    syn_to_axon = _synapse_to_axon_map(axon_synapses) if axon_synapses else {}
    raster_streams: list[RasterStream] = []
    for stream_idx in sorted(results.input_events.keys()):
        syn_idx = int(stream_idx)
        ts = results.input_events[syn_idx]
        times = tuple(float(t) for t in ts.as_units("ms").index.values)
        if syn_to_axon:
            color = _axon_color_hex(syn_to_axon.get(syn_idx, syn_idx % 10))
        else:
            color = "#374151"
        raster_streams.append(
            RasterStream(
                syn_idx=syn_idx,
                label=f"syn_{syn_idx}",
                times_ms=times,
                color=color,
            )
        )

    t_max_ms = float(frame_cache.time_ms[-1]) if len(frame_cache.time_ms) else 0.0

    return SimulationDashboardData(
        frame_cache=frame_cache,
        region_traces=region_traces,
        probe_traces=probe_traces,
        raster_streams=raster_streams,
        t_max_ms=t_max_ms,
        metadata={
            "T_ms": parameters.get("T_ms"),
            "n_frames": frame_cache.n_frames,
            "n_synapses": len(raster_streams),
            "swc": str(swc_path),
        },
    )


def build_3d_frame_figure(
    data: SimulationDashboardData,
    frame_idx: int,
    *,
    theme_mode: str = "light",
) -> go.Figure:
    """Build the 3D voltage panel for one animation frame."""
    return build_3d_template_figure(
        data.frame_cache,
        template=plotly_template_for(theme_mode),
        autosize=True,
    )


def _probe_cache_key(probe_columns: Sequence[str] | None) -> tuple[str, ...]:
    cols = list(probe_columns) if probe_columns else DEFAULT_PROBE_COLUMNS
    return tuple(sorted(cols))


def _precompute_voltage_figure(
    data: SimulationDashboardData,
    probe_columns: Sequence[str],
    theme_mode: str,
) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
    """Pre-build the static voltage figure and trace index map."""
    t_ms = float(data.frame_cache.time_ms[0])
    fig, trace_index_map = build_voltage_figure(
        data, t_ms, probe_columns, theme_mode=theme_mode
    )
    return fig.to_dict(), trace_index_map


@dataclass
class DashboardPlaybackCache:
    """Precomputed figures and 3D patch data for smooth dashboard playback."""

    template_3d: dict[str, Any]
    synapse_trace_idx: int | None
    fig_raster: dict[str, Any]
    voltage_by_probes: dict[tuple[str, ...], dict[str, Any]]
    voltage_trace_index_map: dict[tuple[str, ...], dict[str, tuple[int, int]]]
    data: SimulationDashboardData
    theme_mode: str
    clientside_bundle: dict[str, Any] | None = None

    def patch_3d(self, frame_idx: int) -> Patch:
        return patch_3d_frame(
            self.data.frame_cache,
            frame_idx,
            synapse_trace_idx=self.synapse_trace_idx,
        )

    @property
    def raster_figure(self) -> dict[str, Any]:
        return self.fig_raster

    def raster_at(self, frame_idx: int) -> Patch:
        return build_raster_cursor_patch(
            float(self.data.frame_cache.time_ms[frame_idx])
        )

    def voltage_at(
        self,
        frame_idx: int,
        probe_columns: Sequence[str] | None,
    ) -> dict[str, Any]:
        key = _probe_cache_key(probe_columns)
        if key not in self.voltage_by_probes:
            logger.info(
                "Precomputing voltage traces for probes: %s",
                ", ".join(key),
            )
            t0 = time.perf_counter()
            fig_dict, trace_index_map = _precompute_voltage_figure(
                self.data, list(key), self.theme_mode
            )
            self.voltage_by_probes[key] = fig_dict
            self.voltage_trace_index_map[key] = trace_index_map
            logger.info(
                "Voltage precompute done in %.1fs",
                time.perf_counter() - t0,
            )
        return self.voltage_by_probes[key]

    def voltage_marker_patch_at(
        self,
        frame_idx: int,
        probe_columns: Sequence[str] | None,
    ) -> Patch:
        key = _probe_cache_key(probe_columns)
        trace_index_map = self.voltage_trace_index_map[key]
        return build_voltage_marker_patch(
            self.data, frame_idx, trace_index_map
        )

    def voltage_visibility_patch_at(
        self,
        probe_columns: Sequence[str],
    ) -> Patch:
        key = _probe_cache_key(probe_columns)
        trace_index_map = self.voltage_trace_index_map[key]
        return build_voltage_visibility_patch(probe_columns, trace_index_map)


_DASHBOARD_CACHE_VERSION = "v3"


def dashboard_cache_key(
    data: SimulationDashboardData,
    theme_mode: str,
) -> str:
    """Hash key for disk-cached playback bundle."""
    fc = data.frame_cache
    parts = [
        _DASHBOARD_CACHE_VERSION,
        str(data.metadata.get("swc", "")),
        str(data.metadata.get("T_ms", "")),
        str(data.n_frames),
        str(fc.cmin),
        str(fc.cmax),
        str(fc.colorscale) if isinstance(fc.colorscale, str) else "custom",
        theme_mode,
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:16]


def load_playback_cache(path: Path) -> DashboardPlaybackCache | None:
    """Load a cached playback bundle if present."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            cache = pickle.load(fh)
        if isinstance(cache, DashboardPlaybackCache):
            logger.info("Loaded dashboard playback cache from %s", path)
            return cache
    except (pickle.UnpicklingError, OSError, TypeError) as exc:
        logger.warning("Failed to load dashboard cache %s: %s", path, exc)
    return None


def save_playback_cache(cache: DashboardPlaybackCache, path: Path) -> None:
    """Persist a playback bundle to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved dashboard playback cache to %s", path)


def build_dashboard_playback_cache(
    data: SimulationDashboardData,
    theme_mode: str,
    clientside_max_frames: int = 600,
) -> DashboardPlaybackCache:
    """Precompute static figures and optional clientside bundle."""
    n = data.n_frames
    theme = plotly_template_for(theme_mode)
    t0 = time.perf_counter()

    logger.info("Building 3D template figure...")
    template_3d = build_3d_template_figure(
        data.frame_cache,
        template=theme,
        autosize=True,
    ).to_dict()
    syn_idx = synapse_mesh_trace_index(data.frame_cache)

    logger.info("Precomputing static raster figure...")
    t_ms = float(data.frame_cache.time_ms[0])
    fig_raster = build_raster_figure(data, t_ms, theme_mode=theme_mode).to_dict()

    default_key = _probe_cache_key(DEFAULT_PROBE_COLUMNS)
    logger.info("Precomputing default voltage traces...")
    voltage_by_probes: dict[tuple[str, ...], dict[str, Any]] = {}
    voltage_trace_index_map: dict[tuple[str, ...], dict[str, tuple[int, int]]] = {}
    fig_dict, trace_index_map = _precompute_voltage_figure(
        data, list(default_key), theme_mode
    )
    voltage_by_probes[default_key] = fig_dict
    voltage_trace_index_map[default_key] = trace_index_map

    clientside_bundle: dict[str, Any] | None = None
    use_clientside = n <= clientside_max_frames
    if use_clientside:
        logger.info("Building clientside frame bundle (%d frames)...", n)
        clientside_bundle = build_clientside_frame_bundle(data)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Dashboard playback cache ready in %.1fs (%d frames, clientside=%s)",
        elapsed,
        n,
        use_clientside,
    )

    return DashboardPlaybackCache(
        template_3d=template_3d,
        synapse_trace_idx=syn_idx,
        fig_raster=fig_raster,
        voltage_by_probes=voltage_by_probes,
        voltage_trace_index_map=voltage_trace_index_map,
        data=data,
        theme_mode=theme_mode,
        clientside_bundle=clientside_bundle,
    )


def build_clientside_frame_bundle(data: SimulationDashboardData) -> dict[str, Any]:
    """Build a compact JSON-serializable bundle for clientside playback."""
    t0 = time.perf_counter()
    n = data.n_frames
    time_ms = data.frame_cache.time_ms

    # Voltage marker values per frame for all traces.
    all_series = list(data.region_traces.values()) + list(data.probe_traces.values())
    marker_values = np.zeros((n, len(all_series)), dtype=np.float32)
    for s_idx, series in enumerate(all_series):
        values = np.interp(
            time_ms,
            np.asarray(series.t_ms, dtype=float),
            np.asarray(series.v_mV, dtype=float),
            left=float(series.v_mV[0]) if len(series.v_mV) else np.nan,
            right=float(series.v_mV[-1]) if len(series.v_mV) else np.nan,
        )
        marker_values[:, s_idx] = values.astype(np.float32)

    labels = [series.label for series in all_series]
    bundle = {
        "n_frames": n,
        "time_ms": time_ms.tolist(),
        "voltage_labels": labels,
        "voltage_marker_values": marker_values.tolist(),
        "raster_cursor_x": time_ms.tolist(),
        **_3d_clientside_bundle(data.frame_cache),
    }

    elapsed = time.perf_counter() - t0
    logger.info("Clientside frame bundle ready in %.1fs", elapsed)
    return bundle


def _3d_clientside_bundle(frame_cache: AnimationFrameCache) -> dict[str, Any]:
    """3D portion of the clientside bundle."""
    return {
        "mesh_trace_idx": 0,
        "synapse_trace_idx": synapse_mesh_trace_index(frame_cache),
        **frame_cache.build_3d_clientside_bundle(),
    }


def load_or_build_playback_cache(
    data: SimulationDashboardData,
    theme_mode: str,
    cache_dir: str | Path | None = None,
    clientside_max_frames: int = 600,
) -> DashboardPlaybackCache:
    """Load playback cache from disk or build and optionally persist it."""
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"dashboard_{dashboard_cache_key(data, theme_mode)}.pkl"
        loaded = load_playback_cache(cache_path)
        if loaded is not None:
            return loaded
    else:
        cache_path = None

    cache = build_dashboard_playback_cache(
        data, theme_mode, clientside_max_frames=clientside_max_frames
    )
    if cache_path is not None:
        save_playback_cache(cache, cache_path)
    return cache


def _mask_trace_past_current(
    t_ms: np.ndarray,
    v_mV: np.ndarray,
    current_t_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t_ms, dtype=float)
    v = np.asarray(v_mV, dtype=float)
    past_mask = t <= current_t_ms
    future_mask = t >= current_t_ms
    return t[past_mask], v[past_mask], t[future_mask], v[future_mask]


def _trace_value_at(t_ms: np.ndarray, v_mV: np.ndarray, target_t_ms: float) -> float:
    """Linearly interpolate the trace value at target_t_ms."""
    t = np.asarray(t_ms, dtype=float)
    v = np.asarray(v_mV, dtype=float)
    if len(t) == 0:
        return np.nan
    if target_t_ms <= t[0]:
        return float(v[0])
    if target_t_ms >= t[-1]:
        return float(v[-1])
    return float(np.interp(target_t_ms, t, v))


def build_voltage_figure(
    data: SimulationDashboardData,
    current_t_ms: float,
    selected_columns: Sequence[str] | None,
    *,
    theme_mode: str = "light",
    include_all_traces: bool = True,
) -> tuple[go.Figure, dict[str, tuple[int, int]]]:
    """Build static full-trace voltage figure with moving markers.

    Returns the figure and a mapping from trace label to (line_trace_idx,
    marker_trace_idx) for clientside updates.
    """
    fig = go.Figure()
    selected_labels = {s.label for s in data.resolve_traces(selected_columns)}
    trace_order = list(data.region_traces.values()) + sorted(
        data.probe_traces.values(), key=lambda s: s.label
    )
    trace_index_map: dict[str, tuple[int, int]] = {}

    for series in trace_order:
        visible = series.label in selected_labels
        line_dash = series.dash if series.dash != "solid" else None

        # Static full line.
        fig.add_trace(
            go.Scatter(
                x=series.t_ms,
                y=series.v_mV,
                mode="lines",
                name=series.label,
                line=dict(color=series.color, dash=line_dash),
                visible=visible,
                legendgroup=series.label,
                showlegend=True,
            )
        )
        line_idx = len(fig.data) - 1

        # Moving marker at current time.
        v_at_t = _trace_value_at(series.t_ms, series.v_mV, current_t_ms)
        fig.add_trace(
            go.Scatter(
                x=[current_t_ms],
                y=[v_at_t],
                mode="markers",
                name=f"{series.label} marker",
                marker=dict(color=series.color, size=8, symbol="circle"),
                visible=visible,
                legendgroup=series.label,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        marker_idx = len(fig.data) - 1
        trace_index_map[series.label] = (line_idx, marker_idx)

    fig.add_vline(
        x=current_t_ms,
        line=dict(color=_CURSOR_COLOR, width=_CURSOR_WIDTH),
    )

    fig.update_layout(
        template=plotly_template_for(theme_mode),
        xaxis=dict(title="Time (ms)", range=[0, data.t_max_ms]),
        yaxis_title="Voltage (mV)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=48, r=16, t=32, b=40),
        autosize=True,
        uirevision="dashboard-voltage",
        datarevision=current_t_ms,
    )
    return fig, trace_index_map


def build_voltage_marker_patch(
    data: SimulationDashboardData,
    frame_idx: int,
    trace_index_map: Mapping[str, tuple[int, int]],
) -> Patch:
    """Return a Patch that updates marker positions for the current frame."""
    t_ms = float(data.frame_cache.time_ms[frame_idx])
    patch = Patch()
    for series in list(data.region_traces.values()) + list(
        data.probe_traces.values()
    ):
        line_idx, marker_idx = trace_index_map[series.label]
        v_at_t = _trace_value_at(series.t_ms, series.v_mV, t_ms)
        patch["data"][marker_idx]["x"] = [t_ms]
        patch["data"][marker_idx]["y"] = [v_at_t]
    patch["layout"]["shapes"][0]["x0"] = t_ms
    patch["layout"]["shapes"][0]["x1"] = t_ms
    patch["layout"]["datarevision"] = t_ms
    return patch


def build_voltage_visibility_patch(
    selected_columns: Sequence[str],
    trace_index_map: Mapping[str, tuple[int, int]],
) -> Patch:
    """Return a Patch that toggles trace visibility from the dropdown."""
    selected = set(selected_columns)
    patch = Patch()
    for label, (line_idx, marker_idx) in trace_index_map.items():
        visible = label in selected
        patch["data"][line_idx]["visible"] = visible
        patch["data"][marker_idx]["visible"] = visible
    return patch


def build_raster_figure(
    data: SimulationDashboardData,
    current_t_ms: float,
    *,
    theme_mode: str = "light",
) -> go.Figure:
    """Build static event raster with all events and a moving cursor."""
    fig = go.Figure()

    for stream in data.raster_streams:
        x_lines: list[float | None] = []
        y_lines: list[float | None] = []
        y_center = float(stream.syn_idx)
        for event_t in stream.times_ms:
            x_lines.extend([event_t, event_t, None])
            y_lines.extend([y_center - 0.4, y_center + 0.4, None])
        if not x_lines:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color=stream.color, width=1.2),
                name=stream.label,
                hovertemplate=f"{stream.label}<br>t=%{{x:.3f}} ms<extra></extra>",
                showlegend=False,
            )
        )

    fig.add_vline(
        x=current_t_ms,
        line=dict(color=_CURSOR_COLOR, width=_CURSOR_WIDTH),
    )

    n_streams = len(data.raster_streams)
    yaxis: dict[str, Any] = {"dtick": 1, "title": "Synapse"}
    if n_streams > 0:
        yaxis["range"] = [-0.5, n_streams - 0.5]

    fig.update_layout(
        title="Input events",
        template=plotly_template_for(theme_mode),
        xaxis=dict(title="Time (ms)", range=[0, data.t_max_ms]),
        yaxis=yaxis,
        margin=dict(l=48, r=16, t=48, b=40),
        autosize=True,
        showlegend=False,
        uirevision="dashboard-raster",
        datarevision=current_t_ms,
    )
    return fig


def build_raster_cursor_patch(current_t_ms: float) -> Patch:
    """Return a Patch that moves the raster cursor line."""
    patch = Patch()
    patch["layout"]["shapes"][0]["x0"] = current_t_ms
    patch["layout"]["shapes"][0]["x1"] = current_t_ms
    patch["layout"]["datarevision"] = current_t_ms
    return patch


def _slider_marks(time_ms: np.ndarray, max_marks: int = 8) -> dict[int, str]:
    n = len(time_ms)
    if n == 0:
        return {0: "0"}
    if n <= max_marks:
        return {i: f"{time_ms[i]:.1f}" for i in range(n)}
    step = max(1, n // (max_marks - 1))
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return {i: f"{time_ms[i]:.1f}" for i in indices}


def create_simulation_dash_app(
    data: SimulationDashboardData,
    *,
    fps: int = 10,
    title: str = "Simulation Dashboard",
    theme: str = "light",
    cache_dir: str | Path | None = None,
    clientside_max_frames: int = 600,
) -> Dash:
    """Create a Dash app with synchronized 3D animation, traces, and raster."""
    theme_mode = theme.lower()
    style_map = base_layout_styles(get_theme_tokens(theme_mode))
    card_style = style_map["card_style"]
    card_title_style = style_map["card_title_style"]
    button_style = style_map["button_style"]

    n_frames = data.n_frames
    slider_max = max(0, n_frames - 1)
    marks = _slider_marks(data.frame_cache.time_ms)

    playback = load_or_build_playback_cache(
        data,
        theme_mode,
        cache_dir=cache_dir,
        clientside_max_frames=clientside_max_frames,
    )
    use_clientside = playback.clientside_bundle is not None
    playback_fps = fps if use_clientside else min(fps, 10)
    if not use_clientside and fps > playback_fps:
        logger.warning(
            "Using server playback fallback for %d frames; capping FPS "
            "from %d to %d to avoid callback backlog",
            n_frames,
            fps,
            playback_fps,
        )
    interval_ms = max(16, int(1000 / max(1, playback_fps)))

    app = Dash(__name__)
    theme_css = (
        f"{dropdown_theme_css('light')}\n"
        f"{dropdown_theme_css('dark')}\n"
        f"{DASHBOARD_LAYOUT_CSS}"
    )
    app.index_string = app.index_string.replace(
        "</head>", f"<style>{theme_css}</style></head>"
    )

    app.layout = html.Div(
        [
            html.H2(title, className="dashboard-header"),
            html.Div(
                [
                    html.Span(f"T = {data.metadata.get('T_ms', data.t_max_ms):.1f} ms"),
                    html.Span(" · "),
                    html.Span(f"{n_frames} frames"),
                    html.Span(" · "),
                    html.Span(f"{data.metadata.get('n_synapses', 0)} synapses"),
                ],
                style=style_map["helper_text_style"],
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="graph-3d",
                        className="dashboard-graph-3d",
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "responsive": True,
                        },
                    ),
                ],
                className="dashboard-panel dashboard-panel-3d",
                style=card_style,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Voltage traces", style=card_title_style),
                            html.Label("Traces", className="themed-section-label"),
                            dcc.Dropdown(
                                id="probe-columns",
                                options=data.trace_options(),
                                value=data.default_probe_columns,
                                multi=True,
                                className="themed-dropdown",
                            ),
                            dcc.Graph(
                                id="graph-voltage",
                                className="dashboard-graph-voltage",
                                config={
                                    "displayModeBar": True,
                                    "responsive": True,
                                },
                            ),
                        ],
                        className="dashboard-panel-lower dashboard-panel-voltage",
                        style=card_style,
                    ),
                    html.Div(
                        [
                            html.H4("Input events", style=card_title_style),
                            dcc.Graph(
                                id="graph-raster",
                                className="dashboard-graph-raster",
                                config={
                                    "displayModeBar": True,
                                    "responsive": True,
                                },
                            ),
                        ],
                        className="dashboard-panel-lower dashboard-panel-raster",
                        style=card_style,
                    ),
                ],
                className="dashboard-plots-grid",
            ),
            html.Div(
                [
                    html.H4("Timeline", style=card_title_style),
                    html.Div(
                        [
                            html.Button("▶ Play", id="play-btn", n_clicks=0, style=button_style),
                            html.Button("⏸ Pause", id="pause-btn", n_clicks=0, style=button_style),
                            html.Button("⏮ Reset", id="reset-btn", n_clicks=0, style=button_style),
                            html.Div(id="time-readout", style={"marginLeft": "8px"}),
                        ],
                        className="dashboard-controls",
                    ),
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=slider_max,
                        step=1,
                        value=0,
                        marks=marks,
                        tooltip={"placement": "top", "always_visible": False},
                    ),
                ],
                className="dashboard-panel dashboard-timeline",
                style=card_style,
            ),
            dcc.Store(id="frame-idx", data=0),
            dcc.Store(id="playing", data=False),
            dcc.Store(id="figures-ready", data=False),
            dcc.Store(id="theme-mode", data=theme_mode),
            dcc.Store(id="frame-bundle", data=playback.clientside_bundle),
            dcc.Store(
                id="voltage-trace-map",
                data=playback.voltage_trace_index_map.get(
                    _probe_cache_key(DEFAULT_PROBE_COLUMNS), {}
                ),
            ),
            dcc.Interval(id="tick", interval=interval_ms, n_intervals=0, disabled=True),
        ],
        id="app-root",
        style=style_map["app_style"],
        className=f"theme-{theme_mode} dashboard-stack",
    )

    # Initial load: send the full static figures once, then flag ready.
    @app.callback(
        Output("graph-3d", "figure"),
        Output("graph-voltage", "figure"),
        Output("graph-raster", "figure"),
        Output("time-readout", "children"),
        Output("figures-ready", "data"),
        Input("frame-idx", "data"),
        State("figures-ready", "data"),
    )
    def initial_load_figures(frame_idx, figures_ready):
        if figures_ready:
            raise PreventUpdate
        idx = max(0, min(int(frame_idx or 0), slider_max))
        current_t = float(data.frame_cache.time_ms[idx])
        readout = html.B(
            f"t = {current_t:.2f} ms  (frame {idx + 1}/{n_frames})"
        )
        return (
            playback.template_3d,
            playback.voltage_at(idx, DEFAULT_PROBE_COLUMNS),
            playback.raster_figure,
            readout,
            True,
        )

    if use_clientside:
        # Immediate pause/reset: disable the interval and stop playing
        # synchronously, before any queued tick callbacks can run.
        clientside_callback(
            CLIENTSIDE_PAUSE_OR_RESET,
            Output("playing", "data", allow_duplicate=True),
            Output("tick", "disabled", allow_duplicate=True),
            Input("pause-btn", "n_clicks"),
            Input("reset-btn", "n_clicks"),
            prevent_initial_call=True,
        )

        # Clientside transport: single source of truth for playback state.
        clientside_callback(
            CLIENTSIDE_PLAYBACK_TRANSPORT,
            Output("frame-idx", "data"),
            Output("playing", "data"),
            Output("tick", "disabled", allow_duplicate=True),
            Input("play-btn", "n_clicks"),
            Input("pause-btn", "n_clicks"),
            Input("reset-btn", "n_clicks"),
            Input("time-slider", "drag_value"),
            Input("tick", "n_intervals"),
            State("playing", "data"),
            State("frame-idx", "data"),
            State("frame-bundle", "data"),
            prevent_initial_call=True,
        )

        # Clientside frame update: restyle/relayout the existing figures.
        clientside_callback(
            CLIENTSIDE_FRAME_UPDATE,
            Output("time-readout", "children"),
            Output("time-slider", "value"),
            Input("frame-idx", "data"),
            State("frame-bundle", "data"),
            State("voltage-trace-map", "data"),
            prevent_initial_call=True,
        )

        # Clientside probe selection: toggle visibility of existing
        # traces and markers.
        clientside_callback(
            CLIENTSIDE_PROBE_VISIBILITY,
            Output("graph-voltage", "figure"),
            Input("probe-columns", "value"),
            State("voltage-trace-map", "data"),
            prevent_initial_call=True,
        )
    else:
        # Server fallback transport: slider sync, interval disable, and the
        # frame-advance state machine.
        clientside_callback(
            CLIENTSIDE_SERVER_SLIDER_SYNC,
            Output("time-slider", "value"),
            Input("frame-idx", "data"),
        )

        clientside_callback(
            CLIENTSIDE_SERVER_DISABLE_TICK,
            Output("tick", "disabled", allow_duplicate=True),
            Input("pause-btn", "n_clicks"),
            Input("reset-btn", "n_clicks"),
            Input("time-slider", "drag_value"),
            prevent_initial_call=True,
        )

        @app.callback(
            Output("frame-idx", "data"),
            Output("playing", "data"),
            Output("tick", "disabled", allow_duplicate=True),
            Input("play-btn", "n_clicks"),
            Input("pause-btn", "n_clicks"),
            Input("reset-btn", "n_clicks"),
            Input("time-slider", "drag_value"),
            Input("time-slider", "value"),
            Input("tick", "n_intervals"),
            State("playing", "data"),
            State("frame-idx", "data"),
            prevent_initial_call=True,
        )
        def transport_control(
            _play_clicks,
            _pause_clicks,
            _reset_clicks,
            slider_drag,
            slider_value,
            _n_intervals,
            playing,
            frame_idx,
        ):
            triggered = ctx.triggered_id
            triggered_prop = (
                ctx.triggered[0]["prop_id"] if ctx.triggered else ""
            )
            frame_idx = int(frame_idx or 0)
            playing = bool(playing)

            if triggered == "play-btn":
                if frame_idx >= slider_max:
                    frame_idx = 0
                return frame_idx, True, False

            if triggered == "pause-btn":
                return frame_idx, False, True

            if triggered == "reset-btn":
                return 0, False, True

            if triggered == "time-slider":
                if triggered_prop == "time-slider.value":
                    # Programmatic slider sync updates value, not drag_value.
                    if playing:
                        raise PreventUpdate
                    target = int(slider_value or 0)
                else:
                    target = int(
                        slider_drag
                        if slider_drag is not None
                        else slider_value or 0
                    )
                if target == frame_idx:
                    raise PreventUpdate
                return target, False, True

            if triggered == "tick":
                if not playing:
                    raise PreventUpdate
                if frame_idx >= slider_max:
                    return slider_max, False, True
                # Advance only; avoid re-asserting playing/disabled.
                return frame_idx + 1, no_update, no_update

            raise PreventUpdate

        # Server fallback: Patch 3D, voltage markers, and raster cursor.
        @app.callback(
            Output("graph-3d", "figure", allow_duplicate=True),
            Output("graph-voltage", "figure", allow_duplicate=True),
            Output("graph-raster", "figure", allow_duplicate=True),
            Output("time-readout", "children", allow_duplicate=True),
            Output("voltage-trace-map", "data"),
            Input("frame-idx", "data"),
            Input("probe-columns", "value"),
            State("figures-ready", "data"),
            prevent_initial_call=True,
        )
        def update_figures_server(frame_idx, probe_columns, figures_ready):
            if not figures_ready:
                raise PreventUpdate
            t0 = time.perf_counter()
            triggered = ctx.triggered_id
            idx = max(0, min(int(frame_idx or 0), slider_max))
            current_t = float(data.frame_cache.time_ms[idx])
            readout = html.B(
                f"t = {current_t:.2f} ms  (frame {idx + 1}/{n_frames})"
            )

            if triggered == "probe-columns":
                fig_v = playback.voltage_at(idx, probe_columns)
                trace_map = playback.voltage_trace_index_map[
                    _probe_cache_key(probe_columns)
                ]
                return (
                    no_update,
                    fig_v,
                    no_update,
                    readout,
                    trace_map,
                )

            fig_3d = playback.patch_3d(idx)
            fig_v = playback.voltage_marker_patch_at(idx, probe_columns)
            fig_r = playback.raster_at(idx)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > 50:
                logger.debug(
                    "Server figure callback took %.1f ms (frame %d)",
                    elapsed_ms,
                    idx,
                )
            return fig_3d, fig_v, fig_r, readout, no_update

    return app
