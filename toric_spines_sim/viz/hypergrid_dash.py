"""Dash application for viewing hypergrid simulation outputs."""

from __future__ import annotations

import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
import plotly.graph_objects as go

from .dash_theme import (
    base_layout_styles,
    dropdown_theme_css,
    get_theme_tokens,
    plotly_template_for,
)

_AXON_SYNAPSE_LABEL = re.compile(r"^A(\d+)S(\d+)$")


@dataclass(frozen=True)
class HypergridRun:
    """Normalized hypergrid run artifact."""

    run_id: str
    grid_index: int
    parameters: dict[str, Any]
    voltage_t_ms: list[float]
    voltage_d: list[list[float]]
    voltage_columns: list[str]
    voltage_traces_by_name: dict[str, dict[str, Any]]
    events: list[dict[str, Any]]
    rate_curves: dict[str, Any] | None
    raw: dict[str, Any]


def _run_from_payload(payload: dict[str, Any], source: str) -> HypergridRun:
    results = payload["results"]
    voltage = results["voltage_traces"]
    voltage_traces_by_name: dict[str, dict[str, Any]]

    # Backward-compatible voltage parsing:
    # 1) Legacy matrix format: {"t": [...], "d": [[...]], "columns": [...]}
    # 2) Named-trace format: {"format":"named_traces","traces": {...}}
    if isinstance(voltage, dict) and "t" in voltage and "d" in voltage and "columns" in voltage:
        voltage_t_ms = voltage["t"]
        voltage_d = voltage["d"]
        voltage_columns = voltage["columns"]
        voltage_traces_by_name = {}
        for idx, col in enumerate(voltage_columns):
            voltage_traces_by_name[col] = {
                "label": col,
                "t": voltage_t_ms,
                "d": [row[idx] for row in voltage_d],
                "time_units": voltage.get("time_units", "ms"),
            }
    elif (
        isinstance(voltage, dict)
        and voltage.get("format") == "named_traces"
        and isinstance(voltage.get("traces"), dict)
    ):
        traces = voltage["traces"]
        ordered_names = sorted(traces.keys())
        voltage_columns = ordered_names
        voltage_traces_by_name = traces

        # Build matrix-compatible view for existing UI controls.
        first_name = ordered_names[0] if ordered_names else None
        if first_name is None:
            voltage_t_ms = []
            voltage_d = []
        else:
            voltage_t_ms = traces[first_name].get("t", [])
            series = [traces[name].get("d", []) for name in ordered_names]
            # transpose series list-of-lists into row-major matrix
            voltage_d = [list(row) for row in zip(*series)] if series and voltage_t_ms else []
    else:
        raise ValueError(f"Unsupported voltage_traces format in {source}")

    input_events = results["input_events"]
    events = []
    for stream_id, stream in input_events.items():
        events.append(
            {
                "stream_id": str(stream_id),
                "label": stream.get("label", str(stream_id)),
                "t": stream.get("t", []),
                "time_units": stream.get("time_units", "ms"),
            }
        )
    events.sort(key=lambda e: int(e["stream_id"]) if e["stream_id"].isdigit() else e["stream_id"])

    rate_curves = results.get("rate_curves")
    if not isinstance(rate_curves, dict):
        rate_curves = None

    return HypergridRun(
        run_id=str(payload["run_id"]),
        grid_index=int(payload.get("grid_index", -1)),
        parameters=payload["parameters"],
        voltage_t_ms=voltage_t_ms,
        voltage_d=voltage_d,
        voltage_columns=voltage_columns,
        voltage_traces_by_name=voltage_traces_by_name,
        events=events,
        rate_curves=rate_curves,
        raw=payload,
    )


def _read_run(path: Path) -> HypergridRun:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _run_from_payload(payload, str(path))


def _load_runs_from_zip(archive_path: Path) -> list[HypergridRun]:
    runs: list[HypergridRun] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = sorted(
            name for name in zf.namelist() if name.lower().endswith(".json") and not name.endswith("/")
        )
        for member in members:
            with zf.open(member, "r") as fh:
                payload = json.loads(fh.read().decode("utf-8"))
            runs.append(_run_from_payload(payload, f"{archive_path}::{member}"))
    return runs


def _load_runs_from_tar(archive_path: Path) -> list[HypergridRun]:
    runs: list[HypergridRun] = []
    with tarfile.open(archive_path, "r:*") as tf:
        members = sorted(
            (m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".json")),
            key=lambda m: m.name,
        )
        for member in members:
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            payload = json.loads(extracted.read().decode("utf-8"))
            runs.append(_run_from_payload(payload, f"{archive_path}::{member.name}"))
    return runs


def load_hypergrid_runs(data_dir: str | Path) -> list[HypergridRun]:
    """Load and normalize all hypergrid run artifacts."""
    base = Path(data_dir)
    if base.is_file():
        suffixes = [s.lower() for s in base.suffixes]
        if ".zip" in suffixes:
            runs = _load_runs_from_zip(base)
        elif ".tar" in suffixes or ".tgz" in suffixes or (".gz" in suffixes and ".tar" in suffixes):
            runs = _load_runs_from_tar(base)
        else:
            raise ValueError(f"Unsupported archive format: {base}")
        if not runs:
            raise FileNotFoundError(f"No hypergrid JSON files found in archive: {base}")
        return runs

    run_files = sorted(base.glob("*.json"))
    if not run_files:
        raise FileNotFoundError(f"No hypergrid JSON files found in: {base}")
    return [_read_run(path) for path in run_files]


def infer_varied_parameters(runs: list[HypergridRun]) -> dict[str, list[Any]]:
    """Return parameters that vary across run set."""
    if not runs:
        return {}

    keys = sorted(runs[0].parameters.keys())
    varied: dict[str, list[Any]] = {}
    for key in keys:
        seen = []
        for run in runs:
            value = run.parameters.get(key)
            if value not in seen:
                seen.append(value)
        if len(seen) > 1:
            varied[key] = _sorted_parameter_values(seen)
    return varied


def _sorted_parameter_values(values: list[Any]) -> list[Any]:
    """Sort varied parameter values from low to high where possible."""

    def sort_key(value: Any):
        if isinstance(value, (int, float)):
            return (0, float(value))
        if isinstance(value, list):
            if all(isinstance(x, (int, float)) for x in value):
                return (1, tuple(float(x) for x in value))
            return (3, json.dumps(value, sort_keys=True))
        if isinstance(value, str):
            try:
                return (0, float(value))
            except ValueError:
                return (2, value)
        return (4, str(value))

    return sorted(values, key=sort_key)


def _active_parameter_summary(run: HypergridRun, varied_parameters: dict[str, list[Any]]) -> str:
    parts = [f"{k}={run.parameters[k]}" for k in varied_parameters]
    return ", ".join(parts) if parts else "default parameters"


def run_label(run: HypergridRun, varied_parameters: dict[str, list[Any]]) -> str:
    summary = _active_parameter_summary(run, varied_parameters)
    return f"{summary} ({run.run_id})"


def _run_options(runs: list[HypergridRun], varied_parameters: dict[str, list[Any]]) -> list[dict[str, str]]:
    return [{"label": run_label(run, varied_parameters), "value": run.run_id} for run in runs]


def _filter_runs(runs: list[HypergridRun], filter_values: dict[str, list[Any]]) -> list[HypergridRun]:
    def keep(run: HypergridRun) -> bool:
        for key, allowed in filter_values.items():
            if allowed and run.parameters.get(key) not in allowed:
                return False
        return True

    return [run for run in runs if keep(run)]


def _default_voltage_columns(run: HypergridRun, max_traces: int = 20) -> list[str]:
    """Default to integrated spine/sink traces + neck trace when available."""
    preferred = [
        col
        for col in run.voltage_columns
        if col in {"integrated_spine", "integrated_sink"} or col.startswith("neck_compartment::")
    ]
    if preferred:
        return preferred[:max_traces]
    return run.voltage_columns[:max_traces]


def _voltage_time_range(
    run: HypergridRun,
    selected_columns: list[str] | None = None,
    *,
    max_traces: int = 20,
) -> tuple[float, float] | None:
    """Time axis limits from displayed voltage traces."""
    cols = selected_columns or run.voltage_columns[:max_traces]
    t_min: float | None = None
    t_max: float | None = None
    for col in cols:
        trace = run.voltage_traces_by_name.get(col)
        if trace is None:
            continue
        t_vals = trace.get("t", run.voltage_t_ms)
        if not t_vals:
            continue
        lo = float(min(t_vals))
        hi = float(max(t_vals))
        t_min = lo if t_min is None else min(t_min, lo)
        t_max = hi if t_max is None else max(t_max, hi)
    if t_min is not None and t_max is not None:
        return t_min, t_max
    if run.voltage_t_ms:
        return float(min(run.voltage_t_ms)), float(max(run.voltage_t_ms))
    return None


def _voltage_figure(
    run: HypergridRun,
    selected_columns: list[str] | None,
    varied_parameters: dict[str, list[Any]],
    *,
    theme_mode: str,
    max_traces: int = 20,
    x_range: tuple[float, float] | None = None,
) -> go.Figure:
    cols = selected_columns or run.voltage_columns[:max_traces]
    if x_range is None:
        x_range = _voltage_time_range(run, cols, max_traces=max_traces)

    fig = go.Figure()
    for col in cols:
        trace = run.voltage_traces_by_name.get(col)
        if trace is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=trace.get("t", run.voltage_t_ms),
                y=trace.get("d", []),
                mode="lines",
                name=trace.get("label", col),
            )
        )

    active_summary = _active_parameter_summary(run, varied_parameters)
    xaxis: dict[str, Any] = {"title": "Time (ms)"}
    if x_range is not None:
        xaxis["range"] = [x_range[0], x_range[1]]

    fig.update_layout(
        title=dict(
            text=f"{active_summary}<br><sup>{run.run_id}</sup>",
            x=0.01,
            xanchor="left",
        ),
        template=plotly_template_for(theme_mode),
        xaxis=xaxis,
        yaxis_title="Voltage (mV)",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=30, r=20, t=80, b=80),
    )
    return fig


def _events_figure(
    run: HypergridRun,
    selected_stream_ids: list[str] | None,
    *,
    theme_mode: str,
    x_range: tuple[float, float] | None = None,
) -> go.Figure:
    fig = go.Figure()
    events = run.events
    if selected_stream_ids:
        selected = set(selected_stream_ids)
        events = [e for e in events if e["stream_id"] in selected]

    for event in events:
        t = event["t"]
        if not t:
            continue
        try:
            y_center = int(event["stream_id"])
        except (TypeError, ValueError):
            continue
        x_lines = []
        y_lines = []
        for event_t in t:
            x_lines.extend([event_t, event_t, None])
            y_lines.extend([y_center - 0.4, y_center + 0.4, None])
        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(width=1.2),
                name=event["label"],
                hovertemplate="stream=%{text}<br>t=%{x:.3f} ms<extra></extra>",
                text=[event["label"] if x is not None else "" for x in x_lines],
            )
        )

    n_streams = len(run.events)
    yaxis: dict[str, Any] = {"dtick": 1, "title": "Synapse index"}
    if n_streams > 0:
        yaxis["range"] = [-0.5, n_streams - 0.5]

    if x_range is None:
        x_range = _voltage_time_range(run)
    xaxis: dict[str, Any] = {"title": "Time (ms)"}
    if x_range is not None:
        xaxis["range"] = [x_range[0], x_range[1]]

    fig.update_layout(
        title=f"Input events<br><sup>{run.run_id}</sup>",
        template=plotly_template_for(theme_mode),
        xaxis=xaxis,
        yaxis=yaxis,
        margin=dict(l=30, r=20, t=60, b=40),
        showlegend=False,
    )
    return fig


def _format_synapse_list(synapse_indices: list[int]) -> str:
    if not synapse_indices:
        return ""
    if len(synapse_indices) == 1:
        return f"Syn {synapse_indices[0]}"
    return "Syn " + ", ".join(str(s) for s in synapse_indices)


def _synapses_per_axon_display_map(run: HypergridRun, n_axons: int) -> dict[int, list[int]]:
    """Map 0-based axon index to sorted 0-based synapse indices (event-plot rows)."""
    rc = run.rate_curves
    if rc and isinstance(rc.get("synapses_per_axon"), list):
        return {
            axon_idx: list(syns)
            for axon_idx, syns in enumerate(rc["synapses_per_axon"])
            if axon_idx < n_axons
        }

    by_axon: dict[int, list[int]] = {}
    for event in run.events:
        label = str(event.get("label", ""))
        match = _AXON_SYNAPSE_LABEL.match(label)
        if not match:
            continue
        axon_idx = int(match.group(1))
        if axon_idx < 0 or axon_idx >= n_axons:
            continue
        try:
            syn_index = int(event["stream_id"])
        except (TypeError, ValueError):
            continue
        by_axon.setdefault(axon_idx, []).append(syn_index)
    return {axon_idx: sorted(set(syns)) for axon_idx, syns in by_axon.items()}


def _axon_display_name(axon_idx: int, synapse_map: dict[int, list[int]]) -> str:
    syns = synapse_map.get(axon_idx, [])
    if not syns:
        return f"Ax {axon_idx}"
    return f"Ax {axon_idx} ({_format_synapse_list(syns)})"


def _rate_curve_axon_options(run: HypergridRun) -> list[dict[str, str]]:
    rc = run.rate_curves
    if not rc or rc.get("format") != "sine_v1":
        return []
    curves = rc.get("curves", [])
    n_axons = len(curves)
    synapse_map = _synapses_per_axon_display_map(run, n_axons)
    return [
        {"label": _axon_display_name(idx, synapse_map), "value": str(idx)}
        for idx in range(n_axons)
    ]


def _eval_sine_v1_rates(params: dict[str, Any], t_ms: np.ndarray) -> np.ndarray:
    peak = float(params["peak_rate_hz"])
    freq = float(params["freq_hz"])
    phase = float(params.get("phase_rad", 0.0))
    baseline = float(params.get("baseline", 0.0))
    if peak <= 0:
        return np.zeros_like(t_ms, dtype=float)
    omega = 2.0 * np.pi * freq / 1000.0
    val = np.sin(omega * t_ms + phase) + baseline
    return peak * np.maximum(0.0, val)


def _selected_axon_indices(selected_axon_ids: list[str] | None, n_axons: int) -> list[int]:
    if not selected_axon_ids:
        return list(range(n_axons))
    indices: list[int] = []
    for axon_id in selected_axon_ids:
        try:
            idx = int(axon_id)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_axons:
            indices.append(idx)
    return indices


def _rate_curves_figure(
    run: HypergridRun,
    selected_axon_ids: list[str] | None,
    *,
    theme_mode: str,
    n_time_points: int = 400,
) -> go.Figure:
    fig = go.Figure()
    rc = run.rate_curves
    if not rc or rc.get("format") != "sine_v1":
        fig.update_layout(
            title=f"Axon rate curves ({run.run_id})",
            template=plotly_template_for(theme_mode),
            annotations=[
                dict(
                    text="No rate curve data in this run (re-run hypergrid to include).",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14),
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=30, r=20, t=60, b=40),
        )
        return fig

    curves = rc.get("curves", [])
    if not curves:
        return fig

    t_start = float(rc.get("t_start_ms", 0.0))
    t_end = float(rc.get("t_end_ms", 0.0))
    if t_end <= t_start:
        t_end = t_start + 1.0
    t_ms = np.linspace(t_start, t_end, n_time_points)

    axon_indices = _selected_axon_indices(selected_axon_ids, len(curves))
    synapse_map = _synapses_per_axon_display_map(run, len(curves))
    for axon_idx in axon_indices:
        params = curves[axon_idx]
        rates = _eval_sine_v1_rates(params, t_ms)
        y_center = float(axon_idx)
        peak = float(np.max(rates)) if rates.size else 0.0
        if peak <= 0:
            y_scaled = np.full_like(rates, y_center, dtype=float)
        else:
            y_scaled = y_center - 0.4 + 0.8 * (rates / peak)
        axon_name = _axon_display_name(axon_idx, synapse_map)
        fig.add_trace(
            go.Scatter(
                x=t_ms,
                y=y_scaled,
                mode="lines",
                name=axon_name,
                hovertemplate=(
                    f"{axon_name}<br>"
                    "t=%{x:.2f} ms<br>"
                    "rate=%{customdata:.2f} Hz<extra></extra>"
                ),
                customdata=rates,
            )
        )

    n_axons = len(curves)
    yaxis: dict[str, Any] = {"dtick": 1}
    if n_axons > 0:
        yaxis["range"] = [-0.5, n_axons - 0.5]

    fig.update_layout(
        title=f"Axon rate curves ({run.run_id})",
        template=plotly_template_for(theme_mode),
        xaxis_title="Time (ms)",
        yaxis_title="Axon index",
        yaxis=yaxis,
        margin=dict(l=30, r=20, t=60, b=40),
        showlegend=len(axon_indices) <= 8,
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
    )
    return fig


def _parameter_value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _differing_parameter_keys(run_a: HypergridRun, run_b: HypergridRun) -> set[str]:
    all_keys = set(run_a.parameters) | set(run_b.parameters)
    return {
        key
        for key in all_keys
        if _parameter_value_key(run_a.parameters.get(key))
        != _parameter_value_key(run_b.parameters.get(key))
    }


def _metadata_block(
    run: HypergridRun,
    style_map: dict[str, dict[str, str]],
    *,
    highlight_keys: set[str] | None = None,
) -> html.Div:
    diff_keys = highlight_keys or set()
    diff_style = style_map.get("metadata_diff_cell_style", {})
    rows = []
    for key, value in sorted(run.parameters.items()):
        highlight = key in diff_keys
        key_style = {**style_map["metadata_cell_key_style"], **(diff_style if highlight else {})}
        value_style = {
            **style_map["metadata_cell_value_style"],
            **(diff_style if highlight else {}),
        }
        rows.append(
            html.Tr(
                [
                    html.Td(key, style=key_style),
                    html.Td(str(value), style=value_style),
                ]
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span("Run ID: ", style=style_map["run_id_label_style"]),
                    html.Code(run.run_id, style=style_map["code_style"]),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Parameter", style=style_map["metadata_header_cell_style"]),
                                html.Th("Value", style=style_map["metadata_header_cell_style"]),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                style=style_map["metadata_table_style"],
            ),
        ]
    )


def create_hypergrid_dash_app(data_dir: str | Path) -> Dash:
    """Create a Dash app for hypergrid result exploration."""
    runs = load_hypergrid_runs(data_dir)
    by_id = {run.run_id: run for run in runs}
    varied = infer_varied_parameters(runs)
    run_options = _run_options(runs, varied)
    initial_run = run_options[0]["value"]
    initial_theme = "light"
    style_map = base_layout_styles(get_theme_tokens(initial_theme))
    app_style = style_map["app_style"]
    card_style = style_map["card_style"]
    card_title_style = style_map["card_title_style"]

    app = Dash(__name__)
    theme_css = f"{dropdown_theme_css('light')}\n{dropdown_theme_css('dark')}"
    app.index_string = app.index_string.replace(
        "</head>", f"<style>{theme_css}</style></head>"
    )
    app.layout = html.Div(
        [
            html.H2("Hypergrid Results Explorer"),
            html.Div([html.B("Data directory: "), html.Code(str(Path(data_dir)))]),
            html.Div(
                [
                    html.H4("View mode", style=card_title_style),
                    dcc.RadioItems(
                        id="view-mode",
                        options=[
                            {"label": " Single run", "value": "single"},
                            {"label": " Compare two runs", "value": "compare"},
                        ],
                        value="single",
                        inline=True,
                        className="themed-radio",
                    ),
                ],
                id="view-mode-card",
                style=card_style,
            ),
            html.Div(
                [
                    html.H4("Parameter filters and stepping", style=card_title_style),
                    html.Div(
                        "Use dropdowns or Prev/Next steppers to move through active parameter values.",
                        style=style_map["helper_text_style"],
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(key, className="themed-section-label"),
                                    dcc.Dropdown(
                                        id={"type": "param-filter", "param": key},
                                        options=[{"label": str(v), "value": v} for v in values],
                                        value=[],
                                        multi=True,
                                        placeholder=f"All {key}",
                                        className="themed-dropdown",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Prev",
                                                id={"type": "param-step-prev", "param": key},
                                                n_clicks=0,
                                                style=style_map["button_style"],
                                                className="themed-step-button",
                                            ),
                                            html.Div(
                                                id={"type": "param-step-label", "param": key},
                                                style=style_map["stepper_label_style"],
                                            ),
                                            html.Button(
                                                "Next",
                                                id={"type": "param-step-next", "param": key},
                                                n_clicks=0,
                                                style=style_map["button_style"],
                                                className="themed-step-button",
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "gap": "8px",
                                            "alignItems": "center",
                                            "marginTop": "8px",
                                        },
                                    ),
                                ],
                                style={"minWidth": "280px", "flex": "1"},
                            )
                            for key, values in varied.items()
                        ],
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                    ),
                ],
                id="filters-card",
                style=card_style,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Single-run selection", style=card_title_style),
                            dcc.Dropdown(id="run-single", options=run_options, value=initial_run, className="themed-dropdown"),
                            html.Label("Voltage traces", className="themed-section-label"),
                            dcc.Dropdown(id="single-columns", multi=True, placeholder="Default subset", className="themed-dropdown"),
                            html.Label("Event streams", className="themed-section-label"),
                            dcc.Dropdown(id="single-streams", multi=True, placeholder="All streams", className="themed-dropdown"),
                        ],
                        id="single-controls",
                        style=card_style,
                    ),
                    html.Div(
                        [
                            html.H4("Compare selection", style=card_title_style),
                            html.Label("Run A", className="themed-section-label"),
                            dcc.Dropdown(id="run-a", options=run_options, value=initial_run, className="themed-dropdown"),
                            html.Label("Run B", className="themed-section-label"),
                            dcc.Dropdown(
                                id="run-b",
                                options=run_options,
                                value=run_options[min(1, len(run_options) - 1)]["value"],
                                className="themed-dropdown",
                            ),
                        ],
                        id="compare-controls",
                        style=card_style,
                    ),
                ],
                id="selection-wrapper",
            ),
            html.Div(id="empty-state", style=style_map["empty_state_style"]),
            html.Div(
                [
                    dcc.Graph(id="single-voltage"),
                    dcc.Graph(id="single-events"),
                    html.Label("Rate curves", className="themed-section-label"),
                    dcc.Dropdown(
                        id="single-rate-axons",
                        multi=True,
                        placeholder="All axons",
                        className="themed-dropdown",
                    ),
                    dcc.Graph(id="single-rate"),
                ],
                id="single-graphs-card",
                style=card_style,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id="compare-voltage-a"),
                            dcc.Graph(id="compare-events-a"),
                            html.Label("Rate curves", className="themed-section-label"),
                            dcc.Dropdown(
                                id="compare-rate-axons-a",
                                multi=True,
                                placeholder="All axons",
                                className="themed-dropdown",
                            ),
                            dcc.Graph(id="compare-rate-a"),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="compare-voltage-b"),
                            dcc.Graph(id="compare-events-b"),
                            html.Label("Rate curves", className="themed-section-label"),
                            dcc.Dropdown(
                                id="compare-rate-axons-b",
                                multi=True,
                                placeholder="All axons",
                                className="themed-dropdown",
                            ),
                            dcc.Graph(id="compare-rate-b"),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                    ),
                ],
                id="compare-graphs",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "12px",
                    **card_style,
                },
            ),
            html.Div(
                [
                    html.H4("Metadata", style=card_title_style),
                    html.Div(
                        [
                            html.Div(
                                [html.H5("Run (single / A)"), html.Div(id="meta-a")],
                                style={"flex": 1},
                            ),
                            html.Div([html.H5("Run B"), html.Div(id="meta-b")], style={"flex": 1}),
                        ],
                        style={"display": "flex", "gap": "12px"},
                    ),
                ],
                id="metadata-card",
                style=card_style,
            ),
            dcc.Store(id="filtered-run-ids"),
            dcc.Store(id="theme-mode", data=initial_theme),
        ],
        id="app-root",
        style=app_style,
        className=f"theme-{initial_theme}",
    )

    @app.callback(
        Output("app-root", "style"),
        Output("app-root", "className"),
        Output("view-mode-card", "style"),
        Output("filters-card", "style"),
        Output("single-controls", "style"),
        Output("compare-controls", "style"),
        Output("single-graphs-card", "style"),
        Output("compare-graphs", "style"),
        Output("metadata-card", "style"),
        Output("empty-state", "style"),
        Output({"type": "param-step-label", "param": ALL}, "style"),
        Output({"type": "param-step-prev", "param": ALL}, "style"),
        Output({"type": "param-step-next", "param": ALL}, "style"),
        Input("theme-mode", "data"),
        Input("view-mode", "value"),
    )
    def apply_theme(theme_mode, view_mode):
        themed_style_map = base_layout_styles(get_theme_tokens(theme_mode))
        themed_card = themed_style_map["card_style"]
        show_single = view_mode == "single"
        single_style = {**themed_card, "display": "block" if show_single else "none"}
        compare_style = {**themed_card, "display": "block" if not show_single else "none"}
        compare_graph_style = (
            {**themed_card, "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}
            if not show_single
            else {**themed_card, "display": "none"}
        )
        stepper_styles = [dict(themed_style_map["stepper_label_style"]) for _ in varied.keys()]
        button_styles = [dict(themed_style_map["button_style"]) for _ in varied.keys()]
        return (
            themed_style_map["app_style"],
            f"theme-{(theme_mode or 'light').lower()}",
            themed_card,
            themed_card,
            single_style,
            compare_style,
            single_style,
            compare_graph_style,
            themed_card,
            themed_style_map["empty_state_style"],
            stepper_styles,
            button_styles,
            button_styles,
        )

    @app.callback(
        Output({"type": "param-filter", "param": ALL}, "value"),
        Input({"type": "param-step-prev", "param": ALL}, "n_clicks"),
        Input({"type": "param-step-next", "param": ALL}, "n_clicks"),
        State({"type": "param-filter", "param": ALL}, "value"),
        State({"type": "param-filter", "param": ALL}, "id"),
    )
    def step_parameter_values(_prev_clicks, _next_clicks, current_values, filter_ids):
        if not filter_ids:
            return []
        if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
            return current_values

        triggered = ctx.triggered_id
        key = triggered.get("param")
        direction = "prev" if triggered.get("type") == "param-step-prev" else "next"

        output_values = list(current_values)
        key_to_index = {item["param"]: idx for idx, item in enumerate(filter_ids)}
        idx = key_to_index.get(key)
        if idx is None:
            return output_values

        allowed_values = varied.get(key, [])
        if not allowed_values:
            return output_values

        selected = output_values[idx] or []
        if len(selected) == 1 and selected[0] in allowed_values:
            current_index = allowed_values.index(selected[0])
        else:
            current_index = 0

        if direction == "prev":
            next_index = max(0, current_index - 1)
        else:
            next_index = min(len(allowed_values) - 1, current_index + 1)

        output_values[idx] = [allowed_values[next_index]]
        return output_values

    @app.callback(
        Output({"type": "param-step-label", "param": ALL}, "children"),
        Output({"type": "param-step-prev", "param": ALL}, "disabled"),
        Output({"type": "param-step-next", "param": ALL}, "disabled"),
        Input({"type": "param-filter", "param": ALL}, "value"),
        State({"type": "param-filter", "param": ALL}, "id"),
    )
    def update_stepper_labels(values, ids):
        labels: list[str] = []
        prev_disabled: list[bool] = []
        next_disabled: list[bool] = []
        if not ids:
            return labels, prev_disabled, next_disabled

        for item, selected in zip(ids, values):
            key = item["param"]
            allowed_values = varied.get(key, [])
            selected = selected or []
            if len(selected) == 1 and selected[0] in allowed_values:
                current_index = allowed_values.index(selected[0])
                label = f"{key}: {selected[0]}"
            elif len(selected) == 0:
                current_index = 0
                label = f"{key}: All (step at {allowed_values[0]})" if allowed_values else f"{key}: All"
            else:
                current_index = 0
                label = f"{key}: Multiple selected"

            labels.append(label)
            prev_disabled.append(current_index <= 0)
            next_disabled.append(current_index >= len(allowed_values) - 1 if allowed_values else True)

        return labels, prev_disabled, next_disabled

    @app.callback(
        Output("filtered-run-ids", "data"),
        Output("empty-state", "children"),
        Input({"type": "param-filter", "param": ALL}, "value"),
        State({"type": "param-filter", "param": ALL}, "id"),
    )
    def update_filtered_ids(values: list[list[Any]], ids: list[dict[str, str]]):
        filter_values = {item["param"]: (val or []) for item, val in zip(ids, values)}
        filtered = _filter_runs(runs, filter_values)
        if not filtered:
            return [], "No runs match current filters."
        return [run.run_id for run in filtered], ""

    @app.callback(
        Output("run-single", "options"),
        Output("run-a", "options"),
        Output("run-b", "options"),
        Output("run-single", "value"),
        Output("run-a", "value"),
        Output("run-b", "value"),
        Input("filtered-run-ids", "data"),
        State("run-single", "value"),
        State("run-a", "value"),
        State("run-b", "value"),
    )
    def update_run_selectors(filtered_ids, current_single, current_a, current_b):
        valid_runs = [by_id[rid] for rid in filtered_ids] if filtered_ids else []
        options = _run_options(valid_runs, varied)
        if not options:
            return [], [], [], None, None, None

        valid_ids = {opt["value"] for opt in options}

        def keep_or_first(value):
            return value if value in valid_ids else options[0]["value"]

        single = keep_or_first(current_single)
        run_a = keep_or_first(current_a)
        run_b_default = options[min(1, len(options) - 1)]["value"]
        run_b = current_b if current_b in valid_ids else run_b_default
        return options, options, options, single, run_a, run_b

    @app.callback(
        Output("single-columns", "options"),
        Output("single-columns", "value"),
        Output("single-streams", "options"),
        Output("single-streams", "value"),
        Output("single-rate-axons", "options"),
        Output("single-rate-axons", "value"),
        Input("run-single", "value"),
        State("single-columns", "value"),
        State("single-streams", "value"),
        State("single-rate-axons", "value"),
    )
    def update_single_controls(run_id, selected_columns, selected_streams, selected_axons):
        if not run_id:
            return [], [], [], [], [], []
        run = by_id[run_id]
        column_options = [{"label": c, "value": c} for c in run.voltage_columns]
        stream_options = [{"label": e["label"], "value": e["stream_id"]} for e in run.events]
        axon_options = _rate_curve_axon_options(run)
        default_columns = selected_columns or _default_voltage_columns(run, max_traces=20)
        default_streams = selected_streams or []
        default_axons = selected_axons or []
        return (
            column_options,
            default_columns,
            stream_options,
            default_streams,
            axon_options,
            default_axons,
        )

    @app.callback(
        Output("compare-rate-axons-a", "options"),
        Output("compare-rate-axons-a", "value"),
        Output("compare-rate-axons-b", "options"),
        Output("compare-rate-axons-b", "value"),
        Input("run-a", "value"),
        Input("run-b", "value"),
        State("compare-rate-axons-a", "value"),
        State("compare-rate-axons-b", "value"),
    )
    def update_compare_rate_controls(run_a, run_b, selected_a, selected_b):
        options_a = _rate_curve_axon_options(by_id[run_a]) if run_a else []
        options_b = _rate_curve_axon_options(by_id[run_b]) if run_b else []
        return options_a, selected_a or [], options_b, selected_b or []

    @app.callback(
        Output("single-voltage", "figure"),
        Output("single-events", "figure"),
        Output("single-rate", "figure"),
        Output("compare-voltage-a", "figure"),
        Output("compare-events-a", "figure"),
        Output("compare-rate-a", "figure"),
        Output("compare-voltage-b", "figure"),
        Output("compare-events-b", "figure"),
        Output("compare-rate-b", "figure"),
        Output("meta-a", "children"),
        Output("meta-b", "children"),
        Input("view-mode", "value"),
        Input("run-single", "value"),
        Input("single-columns", "value"),
        Input("single-streams", "value"),
        Input("single-rate-axons", "value"),
        Input("run-a", "value"),
        Input("run-b", "value"),
        Input("compare-rate-axons-a", "value"),
        Input("compare-rate-axons-b", "value"),
        Input("theme-mode", "data"),
    )
    def update_figures(
        view_mode,
        run_single,
        columns,
        streams,
        rate_axons,
        run_a,
        run_b,
        rate_axons_a,
        rate_axons_b,
        theme_mode,
    ):
        empty = go.Figure()

        if view_mode == "single":
            if not run_single:
                return empty, empty, empty, empty, empty, empty, empty, empty, empty, "", ""
            run = by_id[run_single]
            voltage_cols = columns or _default_voltage_columns(run, max_traces=20)
            x_range = _voltage_time_range(run, voltage_cols)
            single_v = _voltage_figure(
                run, columns, varied, theme_mode=theme_mode, x_range=x_range
            )
            single_e = _events_figure(run, streams, theme_mode=theme_mode, x_range=x_range)
            single_r = _rate_curves_figure(run, rate_axons, theme_mode=theme_mode)
            current_style_map = base_layout_styles(get_theme_tokens(theme_mode))
            return (
                single_v,
                single_e,
                single_r,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                _metadata_block(run, current_style_map),
                "",
            )

        if not run_a or not run_b:
            return empty, empty, empty, empty, empty, empty, empty, empty, empty, "", ""
        left = by_id[run_a]
        right = by_id[run_b]
        current_style_map = base_layout_styles(get_theme_tokens(theme_mode))
        diff_keys = _differing_parameter_keys(left, right)
        cols_a = _default_voltage_columns(left, max_traces=20)
        cols_b = _default_voltage_columns(right, max_traces=20)
        x_range_a = _voltage_time_range(left, cols_a)
        x_range_b = _voltage_time_range(right, cols_b)
        return (
            empty,
            empty,
            empty,
            _voltage_figure(left, cols_a, varied, theme_mode=theme_mode, x_range=x_range_a),
            _events_figure(left, None, theme_mode=theme_mode, x_range=x_range_a),
            _rate_curves_figure(left, rate_axons_a, theme_mode=theme_mode),
            _voltage_figure(right, cols_b, varied, theme_mode=theme_mode, x_range=x_range_b),
            _events_figure(right, None, theme_mode=theme_mode, x_range=x_range_b),
            _rate_curves_figure(right, rate_axons_b, theme_mode=theme_mode),
            _metadata_block(left, current_style_map, highlight_keys=diff_keys),
            _metadata_block(right, current_style_map, highlight_keys=diff_keys),
        )

    return app

