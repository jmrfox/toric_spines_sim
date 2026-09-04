"""Shared pairwise/sequential/synchrony axon PDF report.

Per-spine scripts call ``run_axons_study(config)`` with a ``ModelConfig``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pynapple as nap

from toric_spines_sim.report import PdfReport
from toric_spines_sim.viz import RasterPlotter
from toric_spines_sim.simulation import (
    TSSimulator,
    remap_axon_channel_events_to_synapses,
)
from toric_spines_sim.geometry import (
    neck_point_from_swc_file, map_xyz_to_nearest_probes
)

from simulations.common.model import ModelConfig
from simulations.common.utils import (
    axon_synapses_to_synapse_axon_map,
    build_pulse_axon_times,
    build_pulse_events_tsgroup,
    load_axon_topology,
    synchronous_spike_times,
)

logger = logging.getLogger(__name__)

calculations = {
    "pairwise_heatmaps": True,
    "sequential_axons": True,
    "synchronicity_sweep": True,
    "random_jitter_sweep": True,
}

SHORT_T_MS = 300.0
PULSE_TIME_MS = 50.0
N_AXONS = 10
SEQUENTIAL_AXON_ORDER: List[int] = []
SYNC_X_MAX_MS = 80.0
SYNC_X_STEP_MS = 20.0
SYNC_N_TRIALS = 10
SYNC_RNG_SEED = 0
SYNC_EXTRA_AXON_SETS: List[List[int]] = [[0, 1, 2, 3, 4]]
# False: bar widths share one global count scale; True: each x column fills bar_max_width
SYNC_HIST_NORMALIZE_PER_COLUMN = False

PAIR_DELAY_MIN_MS = -100.0
PAIR_DELAY_MAX_MS = 100.0
PAIR_DELAY_STEP_MS = 10.0
REPORT_STEM = "TS"
REPORT_FILEPATH: Path | None = None
swc_filepath: Path | None = None
synpts_filepath: Path | None = None
axon_assignment_file: Path | None = None

REGION_TRACE_STYLE = (
    ("Spine", "C0", "-"),
    ("Neck", "C1", "--"),
    ("Sink", "C2", "-."),
)

# Nominal figure aspect ratios (inches); scaled to page in _report_figsize()
_FIG_TRACE_RASTER = (5.0, 5.0)
_FIG_STITCHED = (5.0, 5.0)
_FIG_HEATMAP = (5.0, 6.0)
_FIG_SYNC = (5.0, 6.0)
HEATMAP_CELL_FS = 4

_FIGSIZES: Dict[str, Tuple[float, float]] = {}


def _configure_report_style(report: PdfReport) -> None:
    """Set matplotlib defaults for conference-style PDF figures."""
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "figure.dpi": 150,
    })
    _FIGSIZES["trace_raster"] = report.figure_size(*_FIG_TRACE_RASTER)
    _FIGSIZES["stitched"] = report.figure_size(*_FIG_STITCHED)
    _FIGSIZES["heatmap"] = report.figure_size(*_FIG_HEATMAP)
    _FIGSIZES["sync"] = report.figure_size(*_FIG_SYNC)


def _figsize(kind: Literal["trace_raster", "stitched", "heatmap", "sync"]) -> Tuple[float, float]:
    if kind not in _FIGSIZES:
        raise RuntimeError("Call _configure_report_style(report) before building figures.")
    return _FIGSIZES[kind]


def _get_neck_probe(results, swc_path: Path) -> str:
    """Find the probe nearest to the neck point."""
    neck_xyz = neck_point_from_swc_file(swc_path)
    lookup_points = {"neck_point": neck_xyz}
    nearest_probes = map_xyz_to_nearest_probes(
        results.record_points, lookup_points
    )
    return nearest_probes["neck_point"]


def _axon_color(axon_idx: int) -> str:
    """Matplotlib color for an axon index."""
    return f"C{axon_idx % 10}"


def _add_raster_axon_legend(ax, axon_indices: List[int]) -> None:
    """Legend on raster axis: one entry per axon."""
    present = sorted(set(axon_indices))
    handles = [
        Line2D([0], [0], color=_axon_color(a), lw=2, label=f"Axon {a}")
        for a in present
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6, ncol=2)


def _add_colored_raster_streams(
    plotter: RasterPlotter,
    streams: nap.TsGroup,
    axon_synapses: List[List[int]],
    *,
    ax=None,
) -> None:
    """Add raster streams with event ticks colored by parent axon."""
    syn_to_axon = axon_synapses_to_synapse_axon_map(axon_synapses)
    present_axons: List[int] = []
    for idx in sorted(streams.keys()):
        syn_idx = int(idx)
        axon_idx = syn_to_axon[syn_idx]
        present_axons.append(axon_idx)
        plotter.add_stream(
            streams[idx],
            label=f"syn_{syn_idx}",
            color=_axon_color(axon_idx),
        )
    if ax is not None:
        _add_raster_axon_legend(ax, present_axons)


def _add_colored_merged_raster_streams(
    plotter: RasterPlotter,
    merged_times: Dict[int, List[float]],
    labels: Dict[int, str],
    axon_synapses: List[List[int]],
    *,
    ax=None,
) -> None:
    """Add merged raster streams with event ticks colored by parent axon."""
    syn_to_axon = axon_synapses_to_synapse_axon_map(axon_synapses)
    present_axons: List[int] = []
    for stream_key in sorted(merged_times.keys()):
        syn_idx = int(stream_key)
        axon_idx = syn_to_axon[syn_idx]
        present_axons.append(axon_idx)
        plotter.add_stream(
            merged_times[stream_key],
            label=labels[stream_key],
            color=_axon_color(axon_idx),
        )
    if ax is not None:
        _add_raster_axon_legend(ax, present_axons)


def _build_synapse_sanity_table(
    synapses: Dict,
    axon_synapses: List[List[int]],
) -> List[List[str]]:
    """Build report rows from simulator synapses and event axon mapping."""
    syn_to_axon = axon_synapses_to_synapse_axon_map(axon_synapses)
    rows = [["Synapse", "x (µm)", "y (µm)", "z (µm)", "Axon"]]
    for syn_idx in range(len(synapses)):
        label = f"syn_{syn_idx}"
        x, y, z = synapses[label].location
        rows.append([
            str(syn_idx),
            f"{x:.3f}",
            f"{y:.3f}",
            f"{z:.3f}",
            str(syn_to_axon[syn_idx]),
        ])
    return rows


def _run_pulse_simulation(
    axon_spike_times: Dict[int, float],
    parameters,
    n_synapses_per_axon: List[int],
    axon_synapses: List[List[int]],
):
    """Run a short simulation with explicit per-axon spike times."""
    pulse_parameters = parameters.copy()
    pulse_parameters["T_ms"] = SHORT_T_MS

    axon_times = build_pulse_axon_times(axon_spike_times, N_AXONS)
    axon_channel_events = build_pulse_events_tsgroup(
        axon_times, n_synapses_per_axon, SHORT_T_MS
    )
    events_tsgroup = remap_axon_channel_events_to_synapses(
        axon_channel_events,
        axon_synapses,
    )

    sim = TSSimulator(
        swc_filepath,
        synpts_filepath,
        events_tsgroup,
        pulse_parameters,
        record_points="all",
    )
    return sim.run()


def _random_jitter_spike_times(
    axon_set: List[int],
    x_ms: float,
    rng: np.random.Generator,
) -> Dict[int, float]:
    """Each axon fires at PULSE_TIME_MS + U(0, x_ms)."""
    return {
        axon_idx: PULSE_TIME_MS + float(rng.uniform(0.0, x_ms))
        for axon_idx in axon_set
    }


def _pair_delay_values() -> np.ndarray:
    """Inter-pulse delay values from PAIR_DELAY_MIN_MS to PAIR_DELAY_MAX_MS."""
    n_steps = int(round(
        (PAIR_DELAY_MAX_MS - PAIR_DELAY_MIN_MS) / PAIR_DELAY_STEP_MS
    ))
    return PAIR_DELAY_MIN_MS + np.arange(n_steps + 1) * PAIR_DELAY_STEP_MS


def _pair_delay_spike_times(
    axon_i: int, axon_j: int, delta_t_ms: float
) -> Dict[int, float]:
    """Spike times for axon pair with Δt = t_j − t_i (ms)."""
    t_i = PULSE_TIME_MS + PAIR_DELAY_MAX_MS
    return {axon_i: t_i, axon_j: t_i + delta_t_ms}


def _extract_peaks(results, parameters) -> Dict[str, float]:
    """Peak spine, neck, and sink voltages from simulation results."""
    v_spine = results.integrate_voltages_by_tag(
        parameters["spine_tag"], method="average"
    )
    v_sink = results.integrate_voltages_by_tag(
        parameters["sink_tag"], method="average"
    )
    neck_probe = _get_neck_probe(results, swc_filepath)
    v_neck = results.voltage_traces[neck_probe]
    return {
        "spine": float(np.max(v_spine.values)),
        "neck": float(np.max(v_neck.values)),
        "sink": float(np.max(v_sink.values)),
    }


def _get_region_traces(results, parameters):
    """Return spine, neck, and sink voltage traces in ms."""
    v_spine = results.integrate_voltages_by_tag(
        parameters["spine_tag"], method="average"
    )
    v_sink = results.integrate_voltages_by_tag(
        parameters["sink_tag"], method="average"
    )
    neck_probe = _get_neck_probe(results, swc_filepath)
    v_neck = results.voltage_traces[neck_probe]
    return (
        v_spine.as_units("ms").index.values, v_spine.values,
        v_neck.as_units("ms").index.values, v_neck.values,
        v_sink.as_units("ms").index.values, v_sink.values,
    )


def _plot_region_voltage_traces(ax, t_sp, v_sp, t_nk, v_nk, t_sk, v_sk) -> None:
    """Plot spine, neck, and sink traces with distinct colors and line styles."""
    for (t, v), (label, color, linestyle) in zip(
        ((t_sp, v_sp), (t_nk, v_nk), (t_sk, v_sk)),
        REGION_TRACE_STYLE,
    ):
        ax.plot(t, v, label=label, color=color, linestyle=linestyle)


def _stitch_voltage_traces(
    run_results: List,
    parameters,
    segment_T_ms: float,
) -> Tuple[List[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Concatenate voltage traces with time offsets between segments."""
    boundaries: List[float] = []
    stitched: Dict[str, Tuple[List[float], List[float]]] = {
        "spine": ([], []),
        "neck": ([], []),
        "sink": ([], []),
    }

    for seg_idx, results in enumerate(run_results):
        offset = seg_idx * segment_T_ms
        if seg_idx > 0:
            boundaries.append(offset)

        t_sp, v_sp, t_nk, v_nk, t_sk, v_sk = _get_region_traces(
            results, parameters
        )
        for key, t_vals, v_vals in (
            ("spine", t_sp, v_sp),
            ("neck", t_nk, v_nk),
            ("sink", t_sk, v_sk),
        ):
            stitched[key][0].extend(t_vals + offset)
            stitched[key][1].extend(v_vals)

    arrays = {
        key: (np.array(t), np.array(v))
        for key, (t, v) in stitched.items()
    }
    return boundaries, arrays


def _merge_stitched_raster_times(
    run_results: List,
    segment_T_ms: float,
) -> Tuple[Dict[int, List[float]], Dict[int, str], float]:
    """Merge per-segment input events into one raster dict."""
    total_T_ms = len(run_results) * segment_T_ms
    reference = run_results[0].input_events
    stream_keys = sorted(reference.keys())
    labels = {
        key: _get_input_stream_label(reference, key) for key in stream_keys
    }
    merged_times: Dict[int, List[float]] = {key: [] for key in stream_keys}

    for seg_idx, results in enumerate(run_results):
        offset = seg_idx * segment_T_ms
        for stream_key in stream_keys:
            times = results.input_events[stream_key].as_units("ms").index.values
            if len(times) > 0:
                merged_times[stream_key].extend((times + offset).tolist())

    return merged_times, labels, total_T_ms


def _make_trace_raster_figure(
    parameters,
    results,
    axon_synapses: List[List[int]],
    *,
    title: str,
    xlim: float,
    figsize: Tuple[float, float],
    boundaries: Optional[List[float]] = None,
) -> plt.Figure:
    """Build a two-row figure: voltage traces over input raster."""
    t_sp, v_sp, t_nk, v_nk, t_sk, v_sk = _get_region_traces(results, parameters)

    fig, (ax_trace, ax_raster) = plt.subplots(
        2, 1, figsize=figsize, height_ratios=[3, 3], sharex=True
    )
    _plot_region_voltage_traces(
        ax_trace, t_sp, v_sp, t_nk, v_nk, t_sk, v_sk
    )
    ax_trace.set_ylabel("Voltage (mV)")
    ax_trace.legend(loc="upper right")
    ax_trace.set_xlim(0, xlim)

    if boundaries:
        for boundary in boundaries:
            ax_trace.axvline(boundary, color="gray", linestyle="--")
            ax_raster.axvline(boundary, color="gray", linestyle="--")

    plotter = RasterPlotter(ax=ax_raster, xlim=(0, xlim), ylabel="Synapse")
    _add_colored_raster_streams(
        plotter, results.input_events, axon_synapses, ax=ax_raster
    )

    ax_raster.set_xlabel("Time (ms)")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _make_stitched_trace_raster_figure(
    run_results: List,
    parameters,
    segment_T_ms: float,
    axon_synapses: List[List[int]],
    title: str,
    figsize: Tuple[float, float],
) -> plt.Figure:
    """Build stitched voltage + raster on one figure."""
    boundaries, traces = _stitch_voltage_traces(
        run_results, parameters, segment_T_ms
    )
    total_T_ms = len(run_results) * segment_T_ms
    merged_times, labels, _ = _merge_stitched_raster_times(
        run_results, segment_T_ms
    )

    fig, (ax_trace, ax_raster) = plt.subplots(
        2, 1, figsize=figsize, height_ratios=[3, 3], sharex=True
    )
    _plot_region_voltage_traces(
        ax_trace,
        traces["spine"][0], traces["spine"][1],
        traces["neck"][0], traces["neck"][1],
        traces["sink"][0], traces["sink"][1],
    )
    ax_trace.set_ylabel("Voltage (mV)")
    ax_trace.legend(loc="upper right")
    ax_trace.set_xlim(0, total_T_ms)

    for boundary in boundaries:
        ax_trace.axvline(boundary, color="gray", linestyle="--")
        ax_raster.axvline(boundary, color="gray", linestyle="--")

    plotter = RasterPlotter(ax=ax_raster, xlim=(0, total_T_ms), ylabel="Synapse")
    _add_colored_merged_raster_streams(
        plotter, merged_times, labels, axon_synapses, ax=ax_raster
    )

    ax_raster.set_xlabel("Time (ms)")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _add_trace_raster_page(
    report: PdfReport,
    results,
    parameters,
    axon_synapses: List[List[int]],
    *,
    title: str,
) -> None:
    """Add one page with voltage traces and input raster."""
    fig = _make_trace_raster_figure(
        parameters,
        results,
        axon_synapses,
        title=title,
        xlim=SHORT_T_MS,
        figsize=_figsize("trace_raster"),
    )
    report.add_figure(fig)
    plt.close(fig)


def _add_stitched_trace_raster_page(
    report: PdfReport,
    run_results: List,
    parameters,
    segment_T_ms: float,
    axon_synapses: List[List[int]],
    *,
    title: str,
    description: str,
) -> None:
    """Add heading, description, and one combined stitched trace+raster page."""
    report.add_heading(title, level=3)
    report.add_paragraph(description)
    fig = _make_stitched_trace_raster_figure(
        run_results,
        parameters,
        segment_T_ms,
        axon_synapses,
        title=title,
        figsize=_figsize("stitched"),
    )
    report.add_figure(fig)
    plt.close(fig)
    report.add_page_break()


def _get_input_stream_label(input_events, stream_key) -> str:
    """Label for a stream, matching RasterPlotter.add_streams convention."""
    stream = input_events[stream_key]
    if hasattr(stream, "label") and stream.label:
        return str(stream.label)
    if hasattr(input_events, "metadata") and "label" in input_events.metadata:
        labels = input_events.get_info("label")
        if stream_key in labels.index:
            return str(labels[stream_key])
    if isinstance(stream_key, int):
        return f"syn_{stream_key}"
    return str(stream_key)


def _validate_axon_order(axon_order: List[int], n_axons: int) -> None:
    """Validate axon order indices are unique and in range."""
    if len(axon_order) != len(set(axon_order)):
        raise ValueError(f"axon_order contains duplicates: {axon_order}")
    for axon_idx in axon_order:
        if axon_idx < 0 or axon_idx >= n_axons:
            raise ValueError(
                f"axon_order index {axon_idx} out of range for n_axons={n_axons}"
            )


def _run_sequential_short_runs(
    axon_order: List[int],
    mode: Literal["individual", "cumulative"],
    parameters,
    n_synapses_per_axon: List[int],
    axon_synapses: List[List[int]],
) -> List:
    """Run one short pulse simulation per sequential step."""
    _validate_axon_order(axon_order, N_AXONS)
    run_results = []

    for step in range(len(axon_order)):
        if mode == "individual":
            active = [axon_order[step]]
        else:
            active = axon_order[: step + 1]

        spike_times = synchronous_spike_times(active, PULSE_TIME_MS)
        logger.info(
            f"Sequential {mode} step {step}: axons {active} @ {PULSE_TIME_MS} ms"
        )
        results = _run_pulse_simulation(
            spike_times, parameters, n_synapses_per_axon, axon_synapses
        )
        run_results.append(results)

    return run_results


def _sync_x_values() -> np.ndarray:
    """Jitter values from 0 to SYNC_X_MAX_MS inclusive."""
    n_steps = int(round(SYNC_X_MAX_MS / SYNC_X_STEP_MS))
    return np.arange(n_steps + 1) * SYNC_X_STEP_MS


def _collect_sync_axon_sets(
    highlighted_pairs: List[Tuple[int, int, str]],
) -> List[List[int]]:
    """Highlighted pairs plus extra configurable axon sets."""
    sets: List[List[int]] = []
    seen: set = set()

    for axon_i, axon_j, _ in highlighted_pairs:
        key = frozenset((axon_i, axon_j))
        if key not in seen:
            sets.append([axon_i, axon_j])
            seen.add(key)

    for axon_set in SYNC_EXTRA_AXON_SETS:
        key = frozenset(axon_set)
        if key not in seen:
            sets.append(list(axon_set))
            seen.add(key)

    return sets


def _axon_set_label(axon_set: List[int]) -> str:
    """Human-readable label for an axon set."""
    if len(axon_set) == 2:
        return f"Axons {axon_set[0]} & {axon_set[1]}"
    return "Axons " + ", ".join(str(a) for a in axon_set)


def _run_synchronicity_sweep(
    axon_set: List[int],
    parameters,
    n_synapses_per_axon: List[int],
    axon_synapses: List[List[int]],
) -> Tuple[np.ndarray, List[Dict[str, List[float]]], Dict[float, object], Dict[float, object]]:
    """Sweep coherence x with random delays; return peaks per trial."""
    x_values = _sync_x_values()
    trial_peaks: List[Dict[str, List[float]]] = [
        {"spine": [], "neck": [], "sink": []}
        for _ in range(SYNC_N_TRIALS)
    ]
    example_sync: Dict[float, object] = {}
    example_async: Dict[float, object] = {}

    for trial in range(SYNC_N_TRIALS):
        rng = np.random.default_rng(SYNC_RNG_SEED + trial)

        for x_ms in x_values:
            if x_ms == 0.0:
                spike_times = synchronous_spike_times(axon_set, PULSE_TIME_MS)
            else:
                spike_times = _random_jitter_spike_times(axon_set, x_ms, rng)

            results = _run_pulse_simulation(
                spike_times, parameters, n_synapses_per_axon, axon_synapses
            )
            peaks = _extract_peaks(results, parameters)
            for region in ("spine", "neck", "sink"):
                trial_peaks[trial][region].append(peaks[region])

            if trial == 0 and x_ms == 0.0:
                example_sync[x_ms] = results
            if trial == 0 and x_ms == SYNC_X_MAX_MS:
                example_async[x_ms] = results

    return x_values, trial_peaks, example_sync, example_async


def _plot_coherence_histograms(
    ax,
    x_values: np.ndarray,
    trial_peaks: List[Dict[str, List[float]]],
    region: str,
    bar_max_width: float,
    *,
    color: str = "C0",
    normalize_per_column: bool = SYNC_HIST_NORMALIZE_PER_COLUMN,
) -> None:
    """Sideways bar histogram of peak voltage at each coherence x.

    Bin edges are shared across all x. By default bar widths use a shared count
    scale (global max); set normalize_per_column=True to fill each column independently.
    """
    per_x_values: List[Tuple[float, np.ndarray]] = []
    all_values: List[float] = []

    for j, x_c in enumerate(x_values):
        values = np.array([
            trial_peaks[trial][region][j]
            for trial in range(len(trial_peaks))
        ])
        if len(values) == 0:
            continue
        all_values.extend(values.tolist())
        per_x_values.append((x_c, values))

    if not all_values:
        return

    all_values_arr = np.asarray(all_values, dtype=float)
    n_bins = max(int(np.sqrt(len(trial_peaks))), 7)
    _, shared_bin_edges = np.histogram(all_values_arr, bins=n_bins)

    histograms: List[Tuple[float, np.ndarray]] = []
    global_max_count = 0
    for x_c, values in per_x_values:
        counts, _ = np.histogram(values, bins=shared_bin_edges)
        if counts.max() == 0:
            continue
        global_max_count = max(global_max_count, int(counts.max()))
        histograms.append((x_c, counts))

    if not histograms:
        return

    y_pad = max((all_values_arr.max() - all_values_arr.min()) * 0.05, 0.02)
    ax.set_ylim(all_values_arr.min() - y_pad, all_values_arr.max() + y_pad)

    heights = shared_bin_edges[1:] - shared_bin_edges[:-1]
    for x_c, counts in histograms:
        scale_max = int(counts.max()) if normalize_per_column else global_max_count
        widths = counts.astype(float) / scale_max * bar_max_width
        ax.barh(
            shared_bin_edges[:-1],
            widths,
            height=heights,
            left=x_c,
            align="edge",
            color=color,
            alpha=0.55,
            edgecolor=color,
        )


def _run_pair_delay_sweep(
    axon_i: int,
    axon_j: int,
    parameters,
    n_synapses_per_axon: List[int],
    axon_synapses: List[List[int]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[object]]:
    """Sweep Δt = t_j − t_i; return delay values, peak arrays, and Δt=0 results."""
    delay_values = _pair_delay_values()
    peaks: Dict[str, List[float]] = {
        "spine": [], "neck": [], "sink": [],
    }
    example_sync = None

    for delta_t in delay_values:
        spike_times = _pair_delay_spike_times(axon_i, axon_j, float(delta_t))
        results = _run_pulse_simulation(
            spike_times, parameters, n_synapses_per_axon, axon_synapses
        )
        trial_peaks = _extract_peaks(results, parameters)
        for region in peaks:
            peaks[region].append(trial_peaks[region])
        if delta_t == 0.0:
            example_sync = results

    peak_arrays = {region: np.array(vals) for region, vals in peaks.items()}
    return delay_values, peak_arrays, example_sync


def _add_pair_delay_section(
    report: PdfReport,
    axon_i: int,
    axon_j: int,
    reason: str,
    delay_values: np.ndarray,
    peaks: Dict[str, np.ndarray],
    example_sync: Optional[object],
    parameters,
    axon_synapses: List[List[int]],
) -> None:
    """Add peak voltage vs inter-pulse delay for one axon pair."""
    title = f"Axons {axon_i} & {axon_j} — {reason}"
    report.add_heading(title, level=3)
    report.add_paragraph(
        f"Deterministic two-pulse input: axon {axon_i} at "
        f"{PULSE_TIME_MS + PAIR_DELAY_MAX_MS:.0f} ms, axon {axon_j} at "
        f"t_i + Δt. Sweep Δt = t_j − t_i from {PAIR_DELAY_MIN_MS:.0f} to "
        f"{PAIR_DELAY_MAX_MS:.0f} ms in {PAIR_DELAY_STEP_MS:.0f} ms steps "
        f"(negative Δt: axon {axon_j} leads)."
    )

    if example_sync is not None:
        _add_trace_raster_page(
            report,
            example_sync,
            parameters,
            axon_synapses,
            title=f"{title} — Δt=0 ms (synchronous)",
        )

    fig, axes = plt.subplots(3, 1, figsize=_figsize("sync"), sharex=True)
    for ax, (region_label, color, linestyle), region in zip(
        axes, REGION_TRACE_STYLE, ("spine", "neck", "sink")
    ):
        ax.plot(
            delay_values, peaks[region],
            color=color, linestyle=linestyle, label=region_label,
        )
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Peak voltage (mV)")
        ax.set_title(region_label)
        ax.set_xlim(PAIR_DELAY_MIN_MS, PAIR_DELAY_MAX_MS)

    axes[-1].set_xlabel("Δt = t_j − t_i (ms)")
    fig.suptitle(f"Peak Voltage vs Pulse Delay — Axons {axon_i} & {axon_j}", y=0.995)
    fig.subplots_adjust(hspace=0.2, top=0.93)
    report.add_figure(fig)
    plt.close(fig)
    report.add_page_break()


def _add_synchronicity_section(
    report: PdfReport,
    axon_set: List[int],
    x_values: np.ndarray,
    trial_peaks: List[Dict[str, List[float]]],
    example_sync: Dict[float, object],
    example_async: Dict[float, object],
    parameters,
    axon_synapses: List[List[int]],
    reason: Optional[str] = None,
) -> None:
    """Add synchronicity analysis plots for one axon set."""
    label = _axon_set_label(axon_set)
    report.add_heading(label, level=3)

    reason_text = f" ({reason})" if reason else ""
    report.add_paragraph(
        f"Each axon fires once at {PULSE_TIME_MS} ms + U(0, x) ms jitter. "
        f"x=0 is fully synchronous; larger x relaxes timing coherence{reason_text}."
    )

    if 0.0 in example_sync:
        _add_trace_raster_page(
            report,
            example_sync[0.0],
            parameters,
            axon_synapses,
            title=f"{label} — x=0 ms (synchronous)",
        )

    if SYNC_X_MAX_MS in example_async:
        _add_trace_raster_page(
            report,
            example_async[SYNC_X_MAX_MS],
            parameters,
            axon_synapses,
            title=f"{label} — x={SYNC_X_MAX_MS:.0f} ms (example)",
        )

    fig, axes = plt.subplots(3, 1, figsize=_figsize("sync"), sharex=True)
    region_titles = ("Spine", "Neck", "Sink")
    bar_max_width = SYNC_X_STEP_MS * 0.85

    for ax, region, region_title in zip(
        axes, ("spine", "neck", "sink"), region_titles
    ):
        _plot_coherence_histograms(
            ax, x_values, trial_peaks, region, bar_max_width
        )
        ax.set_ylabel("Peak voltage (mV)")
        ax.set_title(region_title)
        ax.set_xlim(
            0.0,
            float(x_values[-1]) + bar_max_width + SYNC_X_STEP_MS * 0.15,
        )

    axes[-1].set_xlabel("Maximum jitter x (ms)")
    fig.suptitle(f"Peak Voltage vs Jitter — {label}", y=0.995)
    fig.subplots_adjust(hspace=0.2, top=0.93)
    report.add_figure(fig)
    plt.close(fig)
    report.add_page_break()


def _compute_pair_contribution(
    peak_matrix: np.ndarray,
    i: int,
    j: int,
    diagonal_values: np.ndarray,
) -> float:
    """Compute the pair contribution (interaction) for axons i and j."""
    if i == j:
        return 0.0
    pair_peak = peak_matrix[i, j]
    single_i = diagonal_values[i]
    single_j = diagonal_values[j]
    return pair_peak - single_i - single_j


def _select_highlighted_pairs(
    peak_matrix: np.ndarray
) -> List[Tuple[int, int, str]]:
    """Select pairs to highlight based on impact criteria."""
    n = peak_matrix.shape[0]
    diagonal = np.diag(peak_matrix)

    upper_indices = np.triu_indices(n, k=1)
    upper_values = peak_matrix[upper_indices]

    max_idx = np.argmax(upper_values)
    max_i, max_j = upper_indices[0][max_idx], upper_indices[1][max_idx]

    min_idx = np.argmin(upper_values)
    min_i, min_j = upper_indices[0][min_idx], upper_indices[1][min_idx]

    median_val = np.median(upper_values)
    median_diff = np.abs(upper_values - median_val)
    median_idx = np.argmin(median_diff)
    med_i, med_j = upper_indices[0][median_idx], upper_indices[1][median_idx]

    max_interaction = -np.inf
    max_int_pair = (0, 1)
    for idx in range(len(upper_indices[0])):
        i, j = upper_indices[0][idx], upper_indices[1][idx]
        interaction = _compute_pair_contribution(peak_matrix, i, j, diagonal)
        if interaction > max_interaction:
            max_interaction = interaction
            max_int_pair = (i, j)

    min_interaction = np.inf
    min_int_pair = (0, 1)
    for idx in range(len(upper_indices[0])):
        i, j = upper_indices[0][idx], upper_indices[1][idx]
        interaction = _compute_pair_contribution(peak_matrix, i, j, diagonal)
        if interaction < min_interaction:
            min_interaction = interaction
            min_int_pair = (i, j)

    return [
        (max_i, max_j,
         f"highest peak ({peak_matrix[max_i, max_j]:.2f} mV)"),
        (min_i, min_j,
         f"lowest peak ({peak_matrix[min_i, min_j]:.2f} mV)"),
        (med_i, med_j,
         f"median peak ({peak_matrix[med_i, med_j]:.2f} mV)"),
        (max_int_pair[0], max_int_pair[1],
         f"max interaction ({max_interaction:.2f} mV)"),
        (min_int_pair[0], min_int_pair[1],
         f"min interaction ({min_interaction:.2f} mV)"),
    ]


def _create_heatmap(matrix: np.ndarray, title: str, ax) -> None:
    """Create a symmetric matrix heatmap on the given axis."""
    im = ax.imshow(
        matrix,
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Axon j")
    ax.set_ylabel("Axon i")

    n = matrix.shape[0]
    ax.set_xticks(np.arange(0, n))
    ax.set_yticks(np.arange(0, n))
    ax.set_xticklabels(np.arange(0, n))
    ax.set_yticklabels(np.arange(0, n))

    plt.colorbar(im, ax=ax, label="Voltage (mV)")

    for i in range(n):
        ax.text(
            i, i, f"{matrix[i, i]:.1f}",
            ha="center", va="center",
            color="white" if matrix[i, i] < np.mean(matrix)
            else "black",
            fontsize=HEATMAP_CELL_FS,
        )

        for j in range(i):
            ax.text(
                j, i, f"{matrix[i, j]:.1f}",
                ha="center", va="center",
                color="white" if matrix[i, j] < np.mean(matrix)
                else "black",
                fontsize=HEATMAP_CELL_FS,
            )
            ax.text(
                i, j, f"{matrix[j, i]:.1f}",
                ha="center", va="center",
                color="white" if matrix[j, i] < np.mean(matrix)
                else "black",
                fontsize=HEATMAP_CELL_FS,
            )


def _run_pairwise_heatmap_calculations(
    parameters,
    n_synapses_per_axon: List[int],
    axon_synapses: List[List[int]],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[Tuple[int, int], object],
    List[Tuple[int, int, str]],
    Dict,
    List[List[int]],
]:
    """Run single- and pair-axon pulse simulations; build peak matrices."""
    peak_spine = np.zeros((N_AXONS, N_AXONS))
    peak_neck = np.zeros((N_AXONS, N_AXONS))
    peak_sink = np.zeros((N_AXONS, N_AXONS))
    highlighted_results: Dict[Tuple[int, int], object] = {}
    sanity_synapses: Optional[Dict] = None
    sanity_axon_synapses: Optional[List[List[int]]] = None

    logger.info("Running single-axon pulse simulations (diagonal)...")
    for i in range(N_AXONS):
        logger.info("Running single axon %d...", i)
        results = _run_pulse_simulation(
            synchronous_spike_times([i], PULSE_TIME_MS),
            parameters,
            n_synapses_per_axon,
            axon_synapses,
        )
        if sanity_synapses is None:
            sanity_synapses = results.synapses
            sanity_axon_synapses = axon_synapses

        peaks = _extract_peaks(results, parameters)
        peak_spine[i, i] = peaks["spine"]
        peak_neck[i, i] = peaks["neck"]
        peak_sink[i, i] = peaks["sink"]

    logger.info("Running pair pulse simulations...")
    for i in range(N_AXONS):
        for j in range(i + 1, N_AXONS):
            logger.info("Running pair (%d, %d)...", i, j)
            results = _run_pulse_simulation(
                synchronous_spike_times([i, j], PULSE_TIME_MS),
                parameters,
                n_synapses_per_axon,
                axon_synapses,
            )
            peaks = _extract_peaks(results, parameters)
            peak_spine[i, j] = peaks["spine"]
            peak_spine[j, i] = peaks["spine"]
            peak_neck[i, j] = peaks["neck"]
            peak_neck[j, i] = peaks["neck"]
            peak_sink[i, j] = peaks["sink"]
            peak_sink[j, i] = peaks["sink"]
            highlighted_results[(i, j)] = results

    highlighted_pairs = _select_highlighted_pairs(peak_spine)
    logger.info("Selected %d pairs for detailed plots", len(highlighted_pairs))
    return (
        peak_spine,
        peak_neck,
        peak_sink,
        highlighted_results,
        highlighted_pairs,
        sanity_synapses,
        sanity_axon_synapses,
    )


def _ensure_sanity_synapses(
    sanity_synapses: Optional[Dict],
    sanity_axon_synapses: Optional[List[List[int]]],
    parameters,
    n_synapses_per_axon: List[int],
    axon_synapses: List[List[int]],
) -> Tuple[Dict, List[List[int]]]:
    """Run one reference simulation if synapse metadata is not yet available."""
    if sanity_synapses is not None and sanity_axon_synapses is not None:
        return sanity_synapses, sanity_axon_synapses

    logger.info("Running reference simulation for synapse appendix...")
    results = _run_pulse_simulation(
        synchronous_spike_times([0], PULSE_TIME_MS),
        parameters,
        n_synapses_per_axon,
        axon_synapses,
    )
    return results.synapses, axon_synapses


def _add_pairwise_heatmap_pages(
    report: PdfReport,
    peak_spine: np.ndarray,
    peak_neck: np.ndarray,
    peak_sink: np.ndarray,
    parameters,
) -> None:
    report.add_heading("Peak Voltage Heatmaps", level=2)

    vrest = parameters["Vrest_mV"]
    fig, axes = plt.subplots(3, 2, figsize=_figsize("heatmap"))
    _create_heatmap(peak_spine, "Spine peak (mV)", axes[0, 0])
    _create_heatmap(peak_spine - vrest, "Spine excitation (mV)", axes[0, 1])
    _create_heatmap(peak_neck, "Neck peak (mV)", axes[1, 0])
    _create_heatmap(peak_neck - vrest, "Neck excitation (mV)", axes[1, 1])
    _create_heatmap(peak_sink, "Sink peak (mV)", axes[2, 0])
    _create_heatmap(peak_sink - vrest, "Sink excitation (mV)", axes[2, 1])
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    report.add_figure(fig)
    plt.close(fig)
    report.add_page_break()


def _add_detailed_pair_trace_pages(
    report: PdfReport,
    highlighted_pairs: List[Tuple[int, int, str]],
    highlighted_results: Dict[Tuple[int, int], object],
    parameters,
    axon_synapses: List[List[int]],
) -> None:
    report.add_heading("Detailed Traces for Selected Pairs", level=2)
    report.add_paragraph(
        "Voltage traces and event rasters for highlighted pairs "
        f"(synchronous pulse at {PULSE_TIME_MS} ms, {SHORT_T_MS:.0f} ms run)."
    )

    for axon_i, axon_j, reason in highlighted_pairs:
        min_ij = min(axon_i, axon_j)
        results = highlighted_results.get((min_ij, max(axon_i, axon_j)))
        if results is None:
            continue
        title = f"Axons {axon_i} & {axon_j} — {reason}"
        _add_trace_raster_page(
            report, results, parameters, axon_synapses, title=title
        )
        report.add_page_break()


def _add_sequential_axon_pages(
    report: PdfReport,
    parameters,
    n_synapses_per_axon: List[int],
    axon_synapses: List[List[int]],
) -> None:
    report.add_heading("Sequential Axon Activation", level=2)
    report.add_paragraph(
        f"{len(SEQUENTIAL_AXON_ORDER)} stitched {SHORT_T_MS:.0f} ms runs "
        f"in order {SEQUENTIAL_AXON_ORDER}. "
        f"Each segment has one pulse at {PULSE_TIME_MS} ms. "
        f"Dashed vertical lines mark segment boundaries."
    )

    order_str = ", ".join(str(a) for a in SEQUENTIAL_AXON_ORDER)
    logger.info("Running sequential individual short runs...")
    individual_runs = _run_sequential_short_runs(
        SEQUENTIAL_AXON_ORDER,
        "individual",
        parameters,
        n_synapses_per_axon,
        axon_synapses,
    )
    _add_stitched_trace_raster_page(
        report,
        individual_runs,
        parameters,
        SHORT_T_MS,
        axon_synapses,
        title="Individual Activation",
        description=(
            f"Segment k activates only axon_order[k] (order: {order_str}), "
            f"one axon per segment, pulse at {PULSE_TIME_MS} ms."
        ),
    )

    logger.info("Running sequential cumulative short runs...")
    cumulative_runs = _run_sequential_short_runs(
        SEQUENTIAL_AXON_ORDER,
        "cumulative",
        parameters,
        n_synapses_per_axon,
        axon_synapses,
    )
    _add_stitched_trace_raster_page(
        report,
        cumulative_runs,
        parameters,
        SHORT_T_MS,
        axon_synapses,
        title="Cumulative Activation",
        description=(
            f"Segment k activates axons order[0..k] ({order_str}), "
            f"all synchronously at {PULSE_TIME_MS} ms within each segment."
        ),
    )


def _add_pair_delay_sweep_pages(
    report: PdfReport,
    highlighted_pairs: List[Tuple[int, int, str]],
    parameters,
    n_synapses_per_axon: List[int],
    axon_synapses: List[List[int]],
) -> None:
    report.add_heading("Pairwise Pulse Delay", level=3)
    report.add_paragraph(
        f"For each highlighted pair, axon i fires at "
        f"{PULSE_TIME_MS + PAIR_DELAY_MAX_MS:.0f} ms and axon j at t_i + Δt. "
        f"Δt sweeps from {PAIR_DELAY_MIN_MS:.0f} to {PAIR_DELAY_MAX_MS:.0f} ms "
        f"in {PAIR_DELAY_STEP_MS:.0f} ms steps."
    )

    for axon_i, axon_j, reason in highlighted_pairs:
        logger.info(
            "Running pair delay sweep for axons (%d, %d)...", axon_i, axon_j
        )
        delay_values, delay_peaks, example_sync = _run_pair_delay_sweep(
            axon_i,
            axon_j,
            parameters,
            n_synapses_per_axon,
            axon_synapses,
        )
        _add_pair_delay_section(
            report,
            axon_i,
            axon_j,
            reason,
            delay_values,
            delay_peaks,
            example_sync,
            parameters,
            axon_synapses,
        )


def _add_random_jitter_sweep_pages(
    report: PdfReport,
    sync_axon_sets: List[List[int]],
    highlighted_pairs: List[Tuple[int, int, str]],
    parameters,
    n_synapses_per_axon: List[int],
    axon_synapses: List[List[int]],
) -> None:
    report.add_heading("Random Jitter", level=3)
    report.add_paragraph(
        f"Random delay jitter up to x ms added to {PULSE_TIME_MS} ms baseline. "
        f"Sweep x from 0 to {SYNC_X_MAX_MS:.0f} ms in "
        f"{SYNC_X_STEP_MS:.0f} ms steps, {SYNC_N_TRIALS} trials each. "
        f"Peak voltage vs x shown as sideways bar histograms at each x "
        f"(spine, neck, sink stacked subplots; bins = trial count)."
    )

    highlighted_reasons = {
        frozenset((a, b)): reason for a, b, reason in highlighted_pairs
    }

    for axon_set in sync_axon_sets:
        key = frozenset(axon_set)
        reason = highlighted_reasons.get(key)
        logger.info("Running synchronicity sweep for axons %s...", axon_set)
        x_values, trial_peaks, example_sync, example_async = (
            _run_synchronicity_sweep(
                axon_set,
                parameters,
                n_synapses_per_axon,
                axon_synapses,
            )
        )
        _add_synchronicity_section(
            report,
            axon_set,
            x_values,
            trial_peaks,
            example_sync,
            example_async,
            parameters,
            axon_synapses,
            reason=reason,
        )


def _apply_config(config: ModelConfig) -> None:
    """Bind module-level study knobs from ``config``."""
    global calculations, SHORT_T_MS, PULSE_TIME_MS, N_AXONS
    global SEQUENTIAL_AXON_ORDER, SYNC_X_MAX_MS, SYNC_X_STEP_MS
    global SYNC_N_TRIALS, SYNC_RNG_SEED, SYNC_EXTRA_AXON_SETS
    global SYNC_HIST_NORMALIZE_PER_COLUMN
    global PAIR_DELAY_MIN_MS, PAIR_DELAY_MAX_MS, PAIR_DELAY_STEP_MS
    global REPORT_STEM, REPORT_FILEPATH
    global swc_filepath, synpts_filepath, axon_assignment_file

    calculations = dict(config.calculations)
    SHORT_T_MS = config.short_t_ms
    PULSE_TIME_MS = config.pulse_time_ms
    N_AXONS = config.n_axons()
    SEQUENTIAL_AXON_ORDER = config.resolved_sequential_axon_order()
    SYNC_X_MAX_MS = config.sync_x_max_ms
    SYNC_X_STEP_MS = config.sync_x_step_ms
    SYNC_N_TRIALS = config.sync_n_trials
    SYNC_RNG_SEED = config.sync_rng_seed
    SYNC_EXTRA_AXON_SETS = config.resolved_sync_extra_axon_sets()
    SYNC_HIST_NORMALIZE_PER_COLUMN = config.sync_hist_normalize_per_column
    PAIR_DELAY_MIN_MS = config.pair_delay_min_ms
    PAIR_DELAY_MAX_MS = config.pair_delay_max_ms
    PAIR_DELAY_STEP_MS = config.pair_delay_step_ms
    REPORT_STEM = config.stem
    swc_filepath = config.swc_path()
    synpts_filepath = config.synpts_path()
    axon_assignment_file = config.axon_assignment_file
    report_name = (
        f"{config.stem}_axons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    REPORT_FILEPATH = config.results_dir() / report_name
    REPORT_FILEPATH.parent.mkdir(parents=True, exist_ok=True)


def run_axons_study(config: ModelConfig) -> Path:
    """Run the axon PDF study for ``config`` and return the report path."""
    _apply_config(config)
    assert swc_filepath is not None
    assert synpts_filepath is not None
    assert axon_assignment_file is not None
    assert REPORT_FILEPATH is not None

    parameter_bank = config.make_parameter_bank()
    parameters = parameter_bank.sample()
    parameters["T_ms"] = SHORT_T_MS

    n_synapses_per_axon, axon_synapses = load_axon_topology(
        axon_assignment_file,
        T_ms=SHORT_T_MS,
        seed=parameters["seed"],
        n_axons=N_AXONS,
    )

    peak_spine: Optional[np.ndarray] = None
    peak_neck: Optional[np.ndarray] = None
    peak_sink: Optional[np.ndarray] = None
    highlighted_results: Dict[Tuple[int, int], object] = {}
    highlighted_pairs: List[Tuple[int, int, str]] = []
    sanity_synapses: Optional[Dict] = None
    sanity_axon_synapses: Optional[List[List[int]]] = None

    if calculations["pairwise_heatmaps"]:
        (
            peak_spine,
            peak_neck,
            peak_sink,
            highlighted_results,
            highlighted_pairs,
            sanity_synapses,
            sanity_axon_synapses,
        ) = _run_pairwise_heatmap_calculations(
            parameters, n_synapses_per_axon, axon_synapses
        )

    sync_axon_sets: List[List[int]] = []
    if calculations["random_jitter_sweep"]:
        sync_axon_sets = _collect_sync_axon_sets(highlighted_pairs)

    logger.info("Generating PDF report...")
    report = PdfReport(str(REPORT_FILEPATH))
    _configure_report_style(report)
    report.add_title(f"{REPORT_STEM} Axons Study (Short Simulations)")

    report.add_heading("Simulation Parameters", level=2)
    params_dict = {name: np.round(parameters[name], 6) for name in parameters.index}
    report.add_dict_table(params_dict, columns=4)
    report.add_page_break()

    report.add_heading("Input Configuration", level=2)
    input_info = {
        "Enabled calculations": str(calculations),
        "Number of axons": N_AXONS,
        "Short simulation T_ms": SHORT_T_MS,
        "Pulse time (ms)": PULSE_TIME_MS,
        "Sequential axon order": str(SEQUENTIAL_AXON_ORDER),
        "Synchronicity x max (ms)": SYNC_X_MAX_MS,
        "Synchronicity x step (ms)": SYNC_X_STEP_MS,
        "Synchronicity trials": SYNC_N_TRIALS,
        "Extra synchronicity sets": str(SYNC_EXTRA_AXON_SETS),
        "Pair delay min (ms)": PAIR_DELAY_MIN_MS,
        "Pair delay max (ms)": PAIR_DELAY_MAX_MS,
        "Pair delay step (ms)": PAIR_DELAY_STEP_MS,
        "SWC file": swc_filepath.name,
        "Synapse points file": synpts_filepath.name,
        "Axon assignment file": axon_assignment_file.name,
    }
    report.add_dict_table(input_info, columns=2)
    report.add_page_break()

    if calculations["pairwise_heatmaps"]:
        assert peak_spine is not None and peak_neck is not None and peak_sink is not None
        _add_pairwise_heatmap_pages(
            report, peak_spine, peak_neck, peak_sink, parameters
        )
        _add_detailed_pair_trace_pages(
            report,
            highlighted_pairs,
            highlighted_results,
            parameters,
            axon_synapses,
        )

    if calculations["sequential_axons"]:
        _add_sequential_axon_pages(
            report, parameters, n_synapses_per_axon, axon_synapses
        )

    if calculations["synchronicity_sweep"] or calculations["random_jitter_sweep"]:
        report.add_heading("Axonal Input Synchronicity", level=2)
        report.add_paragraph(
            "Timing analyses: (1) deterministic inter-pulse delay Δt = t_j − t_i "
            "for highlighted axon pairs; (2) random jitter added to a common "
            "baseline for selected axon sets."
        )

    if calculations["synchronicity_sweep"]:
        if highlighted_pairs:
            _add_pair_delay_sweep_pages(
                report,
                highlighted_pairs,
                parameters,
                n_synapses_per_axon,
                axon_synapses,
            )
        else:
            logger.warning(
                "Skipping pair delay sweep: no highlighted pairs "
                "(enable pairwise_heatmaps)"
            )

    if calculations["random_jitter_sweep"]:
        if sync_axon_sets:
            _add_random_jitter_sweep_pages(
                report,
                sync_axon_sets,
                highlighted_pairs,
                parameters,
                n_synapses_per_axon,
                axon_synapses,
            )
        else:
            logger.warning(
                "Skipping random jitter sweep: no axon sets configured"
            )

    sanity_synapses, sanity_axon_synapses = _ensure_sanity_synapses(
        sanity_synapses,
        sanity_axon_synapses,
        parameters,
        n_synapses_per_axon,
        axon_synapses,
    )

    report.add_page_break()
    report.add_heading("Appendix: Synapse Coordinates", level=2)
    report.add_paragraph(
        "Synapse coordinates from the simulator morphology and axon assignments "
        "from the event configuration used in the simulations."
    )
    synapse_rows = _build_synapse_sanity_table(
        sanity_synapses,
        sanity_axon_synapses,
    )
    report.add_table(synapse_rows, header_row=True)

    report.build()
    logger.info("Report saved to %s", REPORT_FILEPATH)
    return REPORT_FILEPATH
