"""Input scenario builders for axon-variation simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pynapple as nap
from jscip import ParameterBank, ParameterSet

from toric_spines_sim.events import (
    DeterministicEventGenerator,
    RateCurve,
    StochasticEventGenerator,
)
from toric_spines_sim.simulation import remap_axon_channel_events_to_synapses

from simulations.common.utils import (
    RateCurveKind,
    build_active_axon_rate_curves,
    build_pulse_axon_times,
    build_pulse_events_tsgroup,
    count_assignment_axons,
    load_axon_topology,
    rate_curve,
    sequential_spike_times,
    synchronous_spike_times,
)


@dataclass(frozen=True)
class InputScenario:
    """Input events and temporal specs for one axon-mapped run."""

    parameters: ParameterSet
    events_tsgroup: nap.TsGroup
    axon_synapses: list[list[int]]
    n_synapses_per_axon: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)
    rate_curves: list[RateCurve] | None = None
    synapse_flash_duration_ms: float = 1.0


def _sample_parameters(parameter_bank: ParameterBank) -> ParameterSet:
    return parameter_bank.sample()


def _load_topology(axon_assignment_file: Path, parameters: ParameterSet):
    return load_axon_topology(
        axon_assignment_file,
        T_ms=parameters["T_ms"],
        seed=parameters["seed"],
    )


def build_simultaneous_pulse_input(
    axon_assignment_file: Path,
    parameter_bank: ParameterBank,
    *,
    active_axons: Sequence[int],
    pulse_time_ms: float,
    t_ms: float,
) -> InputScenario:
    """Active axons fire once synchronously at ``pulse_time_ms``."""
    parameters = _sample_parameters(parameter_bank)
    parameters["T_ms"] = t_ms
    n_synapses_per_axon, axon_synapses = _load_topology(
        axon_assignment_file, parameters
    )
    n_axons = count_assignment_axons(axon_assignment_file)
    spike_times = synchronous_spike_times(list(active_axons), pulse_time_ms)
    axon_channel_events = build_pulse_events_tsgroup(
        build_pulse_axon_times(spike_times, n_axons),
        n_synapses_per_axon,
        parameters["T_ms"],
    )
    events_tsgroup = remap_axon_channel_events_to_synapses(
        axon_channel_events,
        axon_synapses,
    )
    return InputScenario(
        parameters=parameters,
        events_tsgroup=events_tsgroup,
        axon_synapses=axon_synapses,
        n_synapses_per_axon=n_synapses_per_axon,
        metadata={
            "input_scenario": "simultaneous_pulse",
            "active_axons": list(active_axons),
            "pulse_time_ms": float(pulse_time_ms),
        },
    )


def build_sequential_pulse_input(
    axon_assignment_file: Path,
    parameter_bank: ParameterBank,
    *,
    axon_order: Sequence[int] | None,
    step_ms: float,
    start_ms: float | None = None,
    tail_ms: float | None = None,
    synapse_flash_duration_ms: float = 10.0,
    dt_record_ms: float = 1.0,
) -> InputScenario:
    """Pulse each axon in sequence within one run."""
    n_axons = count_assignment_axons(axon_assignment_file)
    resolved_order = (
        list(range(n_axons)) if axon_order is None else list(axon_order)
    )
    start = step_ms if start_ms is None else start_ms
    tail = step_ms if tail_ms is None else tail_ms

    parameters = _sample_parameters(parameter_bank)
    parameters["dt_record_ms"] = dt_record_ms
    parameters["T_ms"] = (
        start + max(0, len(resolved_order) - 1) * step_ms + tail
    )
    n_synapses_per_axon, axon_synapses = _load_topology(
        axon_assignment_file, parameters
    )
    spike_times = sequential_spike_times(resolved_order, start, step_ms)
    axon_channel_events = build_pulse_events_tsgroup(
        build_pulse_axon_times(spike_times, n_axons),
        n_synapses_per_axon,
        parameters["T_ms"],
    )
    events_tsgroup = remap_axon_channel_events_to_synapses(
        axon_channel_events,
        axon_synapses,
    )
    return InputScenario(
        parameters=parameters,
        events_tsgroup=events_tsgroup,
        axon_synapses=axon_synapses,
        n_synapses_per_axon=n_synapses_per_axon,
        synapse_flash_duration_ms=synapse_flash_duration_ms,
        metadata={
            "input_scenario": "sequential_pulse",
            "axon_order": list(resolved_order),
            "pulse_time_ms": float(start),
            "sequential_delta_t_ms": float(step_ms),
            "synapse_flash_duration_ms": float(synapse_flash_duration_ms),
        },
    )


def build_axon_periodic_input(
    axon_assignment_file: Path,
    parameter_bank: ParameterBank,
    *,
    active_axons: Sequence[int],
    t_ms: float,
    delay_ms: float,
    event_rate_hz: float,
    rate_curve_kind: RateCurveKind = "flat",
    mod_freq_hz: float = 1.0,
    sine_baseline: float = 0.0,
) -> InputScenario:
    """Deterministic periodic events on selected axons."""
    parameters = _sample_parameters(parameter_bank)
    parameters["T_ms"] = t_ms
    parameters["delay_ms"] = delay_ms
    n_synapses_per_axon, axon_synapses = _load_topology(
        axon_assignment_file, parameters
    )
    n_axons = count_assignment_axons(axon_assignment_file)
    rate_curves = build_active_axon_rate_curves(
        active_axons,
        n_axons,
        rate_curve_kind=rate_curve_kind,
        event_rate_hz=event_rate_hz,
        mod_freq_hz=mod_freq_hz,
        sine_baseline=sine_baseline,
    )
    event_generator = DeterministicEventGenerator(
        rate_curves=rate_curves,
        n_synapses_per_axon=n_synapses_per_axon,
        T_ms=parameters["T_ms"],
        delay_ms=parameters["delay_ms"],
    )
    events_tsgroup = remap_axon_channel_events_to_synapses(
        event_generator.generate(),
        axon_synapses,
    )
    return InputScenario(
        parameters=parameters,
        events_tsgroup=events_tsgroup,
        axon_synapses=axon_synapses,
        n_synapses_per_axon=n_synapses_per_axon,
        rate_curves=rate_curves,
        metadata={
            "input_scenario": "axon_periodic",
            "active_axons": list(active_axons),
            "generator": "deterministic",
            "rate_curve_kind": rate_curve_kind,
            "event_rate_hz": float(event_rate_hz),
        },
    )


def build_axon_poisson_input(
    axon_assignment_file: Path,
    parameter_bank: ParameterBank,
    *,
    active_axons: Sequence[int],
    t_ms: float,
    delay_ms: float,
    event_rate_hz: float,
    rate_curve_kind: RateCurveKind = "flat",
    mod_freq_hz: float = 1.0,
    sine_baseline: float = 0.0,
    poisson_arp_ms: float = 0.0,
) -> InputScenario:
    """Stochastic Poisson events on selected axons."""
    parameters = _sample_parameters(parameter_bank)
    parameters["T_ms"] = t_ms
    parameters["delay_ms"] = delay_ms
    n_synapses_per_axon, axon_synapses = _load_topology(
        axon_assignment_file, parameters
    )
    n_axons = count_assignment_axons(axon_assignment_file)
    rate_curves = build_active_axon_rate_curves(
        active_axons,
        n_axons,
        rate_curve_kind=rate_curve_kind,
        event_rate_hz=event_rate_hz,
        mod_freq_hz=mod_freq_hz,
        sine_baseline=sine_baseline,
    )
    event_generator = StochasticEventGenerator(
        rate_curves=rate_curves,
        n_synapses_per_axon=n_synapses_per_axon,
        T_ms=parameters["T_ms"],
        delay_ms=parameters["delay_ms"],
        seed=parameters["seed"],
        arp_ms=poisson_arp_ms,
    )
    events_tsgroup = remap_axon_channel_events_to_synapses(
        event_generator.generate(),
        axon_synapses,
    )
    return InputScenario(
        parameters=parameters,
        events_tsgroup=events_tsgroup,
        axon_synapses=axon_synapses,
        n_synapses_per_axon=n_synapses_per_axon,
        rate_curves=rate_curves,
        metadata={
            "input_scenario": "axon_poisson",
            "active_axons": list(active_axons),
            "generator": "stochastic",
            "rate_curve_kind": rate_curve_kind,
            "event_rate_hz": float(event_rate_hz),
        },
    )


def build_synapse_poisson_input(
    axon_assignment_file: Path,
    parameter_bank: ParameterBank,
    *,
    t_ms: float,
    delay_ms: float,
    event_rate_hz: float,
    rate_curve_kind: RateCurveKind = "flat",
    mod_freq_hz: float = 1.0,
    sine_baseline: float = 0.0,
    poisson_arp_ms: float = 0.0,
    seed: int = 42,
    synapse_flash_duration_ms: float = 10.0,
    dt_record_ms: float = 1.0,
    pas_leak_g_S_per_cm2: float | None = 0.001,
) -> InputScenario:
    """Independent stochastic stream for every synapse."""
    parameters = _sample_parameters(parameter_bank)
    parameters["T_ms"] = t_ms
    parameters["delay_ms"] = delay_ms
    parameters["seed"] = seed
    parameters["dt_record_ms"] = dt_record_ms
    if pas_leak_g_S_per_cm2 is not None:
        parameters["pas_leak_g_S_per_cm2"] = pas_leak_g_S_per_cm2

    n_synapses_per_axon, axon_synapses = _load_topology(
        axon_assignment_file, parameters
    )
    n_synapses = sum(n_synapses_per_axon)
    phases = np.linspace(0.0, 2.0 * np.pi, n_synapses, endpoint=False)
    rate_curves = [
        rate_curve(
            rate_curve_kind,
            rate_hz=event_rate_hz,
            mod_freq_hz=mod_freq_hz,
            baseline=sine_baseline,
            phase_rad=float(phase),
        )
        for phase in phases
    ]
    event_generator = StochasticEventGenerator(
        rate_curves=rate_curves,
        n_synapses_per_axon=[1] * n_synapses,
        T_ms=parameters["T_ms"],
        delay_ms=parameters["delay_ms"],
        seed=parameters["seed"],
        labels=[f"syn_{syn_idx}" for syn_idx in range(n_synapses)],
        arp_ms=poisson_arp_ms,
    )
    return InputScenario(
        parameters=parameters,
        events_tsgroup=event_generator.generate(),
        synapse_flash_duration_ms=synapse_flash_duration_ms,
        axon_synapses=axon_synapses,
        n_synapses_per_axon=n_synapses_per_axon,
        rate_curves=rate_curves,
        metadata={
            "input_scenario": "synapse_poisson",
            "generator": "stochastic",
            "rate_curve_kind": rate_curve_kind,
            "event_rate_hz": float(event_rate_hz),
        },
    )
