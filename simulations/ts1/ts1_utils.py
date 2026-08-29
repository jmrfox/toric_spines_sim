"""Shared utilities for TS1 axon-variation simulations."""

from __future__ import annotations

import colorsys
from typing import Dict, List, Literal, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pynapple as nap

from toric_spines_sim.events import FlatRateCurve, RateCurve, SineRateCurve
from toric_spines_sim.simulation import load_axon_events_from_file

_N_AXONS = 10
_TAB10 = plt.rcParams["axes.prop_cycle"].by_key()["color"]
RateCurveKind = Literal["flat", "sine"]


def axon_base_hex(axon_idx: int) -> str:
    """Matplotlib tab10 hex for an axon index."""
    return mcolors.to_hex(_TAB10[axon_idx % len(_TAB10)])


def muted_hex(
    hex_color: str,
    *,
    saturation_scale: float = 0.85,
    value_scale: float = 0.55,
    value_offset: float = 0.0,
) -> str:
    """Return a muted version of ``hex_color``."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return mcolors.to_hex(
        colorsys.hsv_to_rgb(
            h,
            min(1.0, max(0.0, s * saturation_scale)),
            min(1.0, max(0.0, v * value_scale + value_offset)),
        )
    )


def bright_hex(
    hex_color: str,
    *,
    saturation_scale: float | None = None,
    value_scale: float | None = None,
    value_offset: float = 0.0,
) -> str:
    """Return a brighter version of ``hex_color``."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    saturation = 1.0 if saturation_scale is None else s * saturation_scale
    value = 1.0 if value_scale is None else v * value_scale + value_offset
    return mcolors.to_hex(
        colorsys.hsv_to_rgb(
            h,
            min(1.0, max(0.0, saturation)),
            min(1.0, max(0.0, value)),
        )
    )


def build_synapse_colors_by_axon(
    axon_synapses: list[list[int]],
    n_synapses: int,
    *,
    muted_kwargs: dict[str, float] | None = None,
    bright_kwargs: dict[str, float] | None = None,
) -> dict[str, tuple[str, str]]:
    """Per-synapse (muted inactive, bright active) colors keyed by axon."""
    syn_to_axon = {
        syn: axon for axon, syns in enumerate(axon_synapses) for syn in syns
    }
    muted_options = dict(muted_kwargs or {})
    bright_options = dict(bright_kwargs or {})
    return {
        f"syn_{syn_idx}": (
            muted_hex(axon_base_hex(syn_to_axon[syn_idx]), **muted_options),
            bright_hex(axon_base_hex(syn_to_axon[syn_idx]), **bright_options),
        )
        for syn_idx in range(n_synapses)
    }


def axon_synapses_to_synapse_axon_map(
    axon_synapses: list[list[int]],
) -> dict[int, int]:
    """Map 0-based synapse index to 0-based parent axon index."""
    return {
        syn_idx: axon_idx
        for axon_idx, synapse_indices in enumerate(axon_synapses)
        for syn_idx in synapse_indices
    }


def load_axon_topology(
    axon_assignment_file,
    *,
    T_ms: float,
    seed: int,
    n_axons: int = _N_AXONS,
) -> tuple[list[int], list[list[int]]]:
    """Load axon-to-synapse assignment without generating events."""
    zero_rates = [0.0] * n_axons
    _, n_synapses_per_axon, axon_synapses = load_axon_events_from_file(
        axon_assignment_file=axon_assignment_file,
        axon_rates_hz=zero_rates,
    )
    return n_synapses_per_axon, axon_synapses


def synchronous_spike_times(
    active_axons: List[int],
    pulse_time_ms: float,
) -> Dict[int, float]:
    """All active axons fire once at ``pulse_time_ms``."""
    return {axon_idx: pulse_time_ms for axon_idx in active_axons}


def sequential_spike_times(
    axon_order: Sequence[int],
    first_pulse_time_ms: float,
    delta_t_ms: float,
) -> Dict[int, float]:
    """Pulse each axon once, separated by ``delta_t_ms``."""
    return {
        int(axon_idx): float(first_pulse_time_ms) + idx * float(delta_t_ms)
        for idx, axon_idx in enumerate(axon_order)
    }


def build_pulse_axon_times(
    axon_spike_times: Dict[int, float],
    n_axons: int,
) -> List[List[float]]:
    """Per-axon spike times; inactive axons get empty lists."""
    axon_times: List[List[float]] = [[] for _ in range(n_axons)]
    for axon_idx, spike_time in axon_spike_times.items():
        axon_times[axon_idx] = [float(spike_time)]
    return axon_times


def build_pulse_events_tsgroup(
    axon_times: List[List[float]],
    n_synapses_per_axon: List[int],
    T_ms: float,
) -> nap.TsGroup:
    """Fan out per-axon spike times to axon-ordered event channels."""
    ts_dict: dict = {}
    channel_idx = 0
    for axon_idx, n_syn in enumerate(n_synapses_per_axon):
        times = axon_times[axon_idx]
        for _ in range(n_syn):
            ts_dict[channel_idx] = nap.Ts(t=times, time_units="ms")
            channel_idx += 1

    time_support = nap.IntervalSet(
        start=[0], end=[max(T_ms, 1.0)], time_units="ms"
    )
    return nap.TsGroup(ts_dict, time_support=time_support)


def rate_curve(
    kind: RateCurveKind,
    *,
    rate_hz: float,
    mod_freq_hz: float,
    baseline: float,
    phase_rad: float = 0.0,
) -> RateCurve:
    if kind == "flat":
        return FlatRateCurve(rate_hz=rate_hz)
    if kind == "sine":
        return SineRateCurve(
            peak_rate_hz=rate_hz,
            freq_hz=mod_freq_hz,
            baseline=baseline,
            phase_rad=phase_rad,
        )
    raise ValueError(f"Unsupported rate curve kind: {kind}")


def build_active_axon_rate_curves(
    active_axons: Sequence[int],
    n_axons: int,
    *,
    rate_curve_kind: RateCurveKind,
    event_rate_hz: float,
    mod_freq_hz: float = 1.0,
    sine_baseline: float = 0.0,
) -> list[RateCurve]:
    active = set(int(axon_idx) for axon_idx in active_axons)
    phases = np.linspace(0.0, 2.0 * np.pi, n_axons, endpoint=False)
    return [
        rate_curve(
            rate_curve_kind,
            rate_hz=event_rate_hz if axon_idx in active else 0.0,
            mod_freq_hz=mod_freq_hz,
            baseline=sine_baseline,
            phase_rad=float(phases[axon_idx]),
        )
        for axon_idx in range(n_axons)
    ]
