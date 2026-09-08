"""User-editable input options for TS4 axon studies."""

from __future__ import annotations

from toric_spines_sim.paths import get_data_path

from simulations.common.inputs import (
    build_axon_periodic_input,
    build_axon_poisson_input,
    build_sequential_pulse_input,
    build_simultaneous_pulse_input,
    build_synapse_poisson_input,
)
from simulations.common.model import ModelConfig
from simulations.common.utils import RateCurveKind
from simulations.ts4.params import make_parameter_bank

MODEL = ModelConfig(
    spine_id="TS4",
    sim_key="ts4",
    swc_name="TS4_wsink_r10um.swc",
    synpts_name="TS4_synpts.txt",
    axon_assignment_file=get_data_path("ts_axons", "ts4_axons.txt"),
    make_parameter_bank=make_parameter_bank,
    sequential_axon_order=None,
)


def simultaneous_pulse_input():
    """Active axons fire once synchronously at ``pulse_time_ms``."""
    ACTIVE_AXONS = [0, 3]
    PULSE_TIME_MS = 10.0
    T_MS = 35.0
    return build_simultaneous_pulse_input(
        MODEL.axon_assignment_file,
        MODEL.make_parameter_bank(),
        active_axons=ACTIVE_AXONS,
        pulse_time_ms=PULSE_TIME_MS,
        t_ms=T_MS,
    )


def sequential_pulse_input():
    """Pulse each axon in sequence within one run."""
    AXON_ORDER = MODEL.sequential_axon_order
    STEP_MS = 50.0
    START_MS = STEP_MS
    TAIL_MS = STEP_MS
    SYNAPSE_FLASH_DURATION_MS = 10.0
    return build_sequential_pulse_input(
        MODEL.axon_assignment_file,
        MODEL.make_parameter_bank(),
        axon_order=AXON_ORDER,
        step_ms=STEP_MS,
        start_ms=START_MS,
        tail_ms=TAIL_MS,
        synapse_flash_duration_ms=SYNAPSE_FLASH_DURATION_MS,
    )


def axon_periodic_input():
    """Deterministic periodic events on selected axons."""
    ACTIVE_AXONS = [0, 3]
    T_MS = 1000.0
    DELAY_MS = 20.0
    EVENT_RATE_HZ = 1.0
    RATE_CURVE_KIND: RateCurveKind = "flat"
    MOD_FREQ_HZ = 1.0
    SINE_BASELINE = 0.0
    return build_axon_periodic_input(
        MODEL.axon_assignment_file,
        MODEL.make_parameter_bank(),
        active_axons=ACTIVE_AXONS,
        t_ms=T_MS,
        delay_ms=DELAY_MS,
        event_rate_hz=EVENT_RATE_HZ,
        rate_curve_kind=RATE_CURVE_KIND,
        mod_freq_hz=MOD_FREQ_HZ,
        sine_baseline=SINE_BASELINE,
    )


def axon_poisson_input():
    """Stochastic Poisson events on selected axons."""
    ACTIVE_AXONS = [0, 1, 2, 3, 4]
    T_MS = 1000.0
    DELAY_MS = 20.0
    EVENT_RATE_HZ = 5.0
    RATE_CURVE_KIND: RateCurveKind = "flat"
    MOD_FREQ_HZ = 1.0
    SINE_BASELINE = 0.0
    POISSON_ARP_MS = 0.0
    return build_axon_poisson_input(
        MODEL.axon_assignment_file,
        MODEL.make_parameter_bank(),
        active_axons=ACTIVE_AXONS,
        t_ms=T_MS,
        delay_ms=DELAY_MS,
        event_rate_hz=EVENT_RATE_HZ,
        rate_curve_kind=RATE_CURVE_KIND,
        mod_freq_hz=MOD_FREQ_HZ,
        sine_baseline=SINE_BASELINE,
        poisson_arp_ms=POISSON_ARP_MS,
    )


def synapse_poisson_input():
    """Independent stochastic stream for every synapse."""
    T_MS = 500.0
    DELAY_MS = 10.0
    EVENT_RATE_HZ = 2.0
    RATE_CURVE_KIND: RateCurveKind = "flat"
    MOD_FREQ_HZ = 1.0
    SINE_BASELINE = 0.0
    POISSON_ARP_MS = 0.0
    SEED = 42
    SYNAPSE_FLASH_DURATION_MS = 10.0
    return build_synapse_poisson_input(
        MODEL.axon_assignment_file,
        MODEL.make_parameter_bank(),
        t_ms=T_MS,
        delay_ms=DELAY_MS,
        event_rate_hz=EVENT_RATE_HZ,
        rate_curve_kind=RATE_CURVE_KIND,
        mod_freq_hz=MOD_FREQ_HZ,
        sine_baseline=SINE_BASELINE,
        poisson_arp_ms=POISSON_ARP_MS,
        seed=SEED,
        synapse_flash_duration_ms=SYNAPSE_FLASH_DURATION_MS,
    )
