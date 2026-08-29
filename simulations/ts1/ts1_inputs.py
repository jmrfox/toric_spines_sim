"""Input scenario builders for TS1 axon-variation simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pynapple as nap
from jscip import ParameterSet


from toric_spines_sim.events import (
    DeterministicEventGenerator,
    RateCurve,
    StochasticEventGenerator,
)
from toric_spines_sim.simulation import remap_axon_channel_events_to_synapses

from toric_spines_sim.paths import get_data_path

from simulations.ts1.ts1_params import make_ts1_parameter_bank
from simulations.ts1.ts1_utils import (
    RateCurveKind,
    build_active_axon_rate_curves,
    build_pulse_axon_times,
    build_pulse_events_tsgroup,
    load_axon_topology,
    rate_curve,
    sequential_spike_times,
    synchronous_spike_times,
)

N_AXONS = 10
AXON_ASSIGNMENT_FILE = get_data_path("ts_axons", "ts1_axons.txt")
PARAMETER_BANK = make_ts1_parameter_bank()

ScenarioName = Literal[
    "simultaneous_pulse",
    "axon_periodic",
    "axon_poisson",
    "synapse_poisson",
    "sequential_pulse",
]


@dataclass(frozen=True)
class InputScenario:
    """Input events and temporal specs for one TS1 run."""
    parameters: ParameterSet
    events_tsgroup: nap.TsGroup
    axon_synapses: list[list[int]]
    n_synapses_per_axon: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)
    rate_curves: list[RateCurve] | None = None
    synapse_flash_duration_ms: float = 1.0


def simultaneous_pulse_input() -> InputScenario:
    """Active axons fire once synchronously at ``pulse_time_ms``."""

    # configure
    ACTIVE_AXONS = [0, 3]
    PULSE_TIME_MS = 10.0
    T_MS = 35.0

    # construct
    parameters = PARAMETER_BANK.sample()
    parameters["T_ms"] = T_MS
    n_synapses_per_axon, axon_synapses = load_axon_topology(AXON_ASSIGNMENT_FILE, T_ms=parameters["T_ms"], seed=parameters["seed"])
    spike_times = synchronous_spike_times(list(ACTIVE_AXONS), PULSE_TIME_MS)
    axon_channel_events = build_pulse_events_tsgroup(
        build_pulse_axon_times(spike_times, N_AXONS),
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
            "active_axons": list(ACTIVE_AXONS),
            "pulse_time_ms": float(PULSE_TIME_MS),
        },
    )


def sequential_pulse_input() -> InputScenario:
    """Pulse each axon in sequence within one run."""

    # configure
    AXON_ORDER = [5, 6, 9, 8 , 2 , 4, 7, 1, 3 ,0]
    STEP_MS = 50.0
    START_MS = STEP_MS
    TAIL_MS = STEP_MS
    SYNAPSE_FLASH_DURATION_MS = 10.0

    # construct
    parameters = PARAMETER_BANK.sample()
    # parameters["pas_leak_g_S_per_cm2"] = 0.001
    parameters["dt_record_ms"] = 1.0
    parameters["T_ms"] = (
        START_MS
        + max(0, len(AXON_ORDER) - 1) * STEP_MS
        + TAIL_MS
    )
    n_synapses_per_axon, axon_synapses = load_axon_topology(
        AXON_ASSIGNMENT_FILE, T_ms=parameters["T_ms"], seed=parameters["seed"]
    )
    spike_times = sequential_spike_times(
        AXON_ORDER, START_MS, STEP_MS
    )
    axon_channel_events = build_pulse_events_tsgroup(
        build_pulse_axon_times(spike_times, N_AXONS),
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
        synapse_flash_duration_ms=SYNAPSE_FLASH_DURATION_MS,
        metadata={
            "input_scenario": "sequential_pulse",
            "axon_order": list(AXON_ORDER),
            "pulse_time_ms": float(START_MS),
            "sequential_delta_t_ms": float(STEP_MS),
            "synapse_flash_duration_ms": float(SYNAPSE_FLASH_DURATION_MS),
        },
    )


def axon_periodic_input() -> InputScenario:
    """Deterministic periodic events on selected axons."""

    # configure
    ACTIVE_AXONS = [0, 3]
    T_MS = 1000.0
    DELAY_MS = 20.0
    EVENT_RATE_HZ = 1.0
    RATE_CURVE_KIND: RateCurveKind = "flat"
    MOD_FREQ_HZ = 1.0
    SINE_BASELINE = 0.0

    # construct
    parameters = PARAMETER_BANK.sample()
    parameters["T_ms"] = T_MS
    parameters["delay_ms"] = DELAY_MS
    n_synapses_per_axon, axon_synapses = load_axon_topology(
        AXON_ASSIGNMENT_FILE, T_ms=parameters["T_ms"], seed=parameters["seed"]
    )
    rate_curves = build_active_axon_rate_curves(
        ACTIVE_AXONS,
        N_AXONS,
        rate_curve_kind=RATE_CURVE_KIND,
        event_rate_hz=EVENT_RATE_HZ,
        mod_freq_hz=MOD_FREQ_HZ,
        sine_baseline=SINE_BASELINE,
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
            "active_axons": list(ACTIVE_AXONS),
            "generator": "deterministic",
            "rate_curve_kind": RATE_CURVE_KIND,
            "event_rate_hz": float(EVENT_RATE_HZ),
        },
    )


def axon_poisson_input() -> InputScenario:
    """Stochastic Poisson events on selected axons."""

    # configure
    ACTIVE_AXONS = [0, 1, 2, 3, 4]
    T_MS = 1000.0
    DELAY_MS = 20.0
    EVENT_RATE_HZ = 5.0
    RATE_CURVE_KIND: RateCurveKind = "flat"
    MOD_FREQ_HZ = 1.0
    SINE_BASELINE = 0.0
    POISSON_ARP_MS = 0.0

    # construct
    parameters = PARAMETER_BANK.sample()
    parameters["T_ms"] = T_MS
    parameters["delay_ms"] = DELAY_MS
    n_synapses_per_axon, axon_synapses = load_axon_topology(
        AXON_ASSIGNMENT_FILE, T_ms=parameters["T_ms"], seed=parameters["seed"]
    )
    rate_curves = build_active_axon_rate_curves(
        ACTIVE_AXONS,
        N_AXONS,
        rate_curve_kind=RATE_CURVE_KIND,
        event_rate_hz=EVENT_RATE_HZ,
        mod_freq_hz=MOD_FREQ_HZ,
        sine_baseline=SINE_BASELINE,
    )
    event_generator = StochasticEventGenerator(
        rate_curves=rate_curves,
        n_synapses_per_axon=n_synapses_per_axon,
        T_ms=parameters["T_ms"],
        delay_ms=parameters["delay_ms"],
        seed=parameters["seed"],
        arp_ms=POISSON_ARP_MS,
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
            "active_axons": list(ACTIVE_AXONS),
            "generator": "stochastic",
            "rate_curve_kind": RATE_CURVE_KIND,
            "event_rate_hz": float(EVENT_RATE_HZ),
        },
    )


def synapse_poisson_input() -> InputScenario:
    """Independent stochastic stream for every synapse."""

    # configure
    T_MS = 500.0
    DELAY_MS = 10.0
    EVENT_RATE_HZ = 2.0
    RATE_CURVE_KIND: RateCurveKind = "flat"
    MOD_FREQ_HZ = 1.0
    SINE_BASELINE = 0.0
    POISSON_ARP_MS = 0.0
    SEED = 42
    SYNAPSE_FLASH_DURATION_MS = 10.0
    
    # construct
    parameters = PARAMETER_BANK.sample()
    parameters["T_ms"] = T_MS
    parameters["delay_ms"] = DELAY_MS
    parameters["seed"] = SEED
    parameters["dt_record_ms"] = 1.0
    parameters["pas_leak_g_S_per_cm2"] = 0.001
    
    n_synapses_per_axon, axon_synapses = load_axon_topology(
        AXON_ASSIGNMENT_FILE, T_ms=parameters["T_ms"], seed=parameters["seed"]
    )
    n_synapses = sum(n_synapses_per_axon)
    phases = np.linspace(0.0, 2.0 * np.pi, n_synapses, endpoint=False)
    rate_curves = [
        rate_curve(
            RATE_CURVE_KIND,
            rate_hz=EVENT_RATE_HZ,
            mod_freq_hz=MOD_FREQ_HZ,
            baseline=SINE_BASELINE,
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
        arp_ms=POISSON_ARP_MS,
    )
    return InputScenario(
        parameters=parameters,
        events_tsgroup=event_generator.generate(),
        synapse_flash_duration_ms=SYNAPSE_FLASH_DURATION_MS,
        axon_synapses=axon_synapses,
        n_synapses_per_axon=n_synapses_per_axon,
        rate_curves=rate_curves,
        metadata={
            "input_scenario": "synapse_poisson",
            "generator": "stochastic",
            "rate_curve_kind": RATE_CURVE_KIND,
            "event_rate_hz": float(EVENT_RATE_HZ),
        },
    )
