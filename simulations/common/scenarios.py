"""Named input scenarios for the axon studies, shared by every spine.

Each builder takes a :class:`~simulations.common.model.ModelConfig` (from
``simulations.common.spines.model_config``) plus keyword overrides, and
returns an ``InputScenario``. Defaults are the values the per-spine
``inputs.py`` files used.
"""

from __future__ import annotations

from typing import Callable, Dict

from simulations.common.inputs import (
    InputScenario,
    build_axon_periodic_input,
    build_axon_poisson_input,
    build_sequential_pulse_input,
    build_simultaneous_pulse_input,
    build_synapse_poisson_input,
)
from simulations.common.model import ModelConfig


def simultaneous_pulse(cfg: ModelConfig, *, active_axons=(0, 3), pulse_time_ms=10.0, t_ms=35.0) -> InputScenario:
    """Active axons fire once together at ``pulse_time_ms``."""
    return build_simultaneous_pulse_input(
        cfg.axon_assignment_file, cfg.make_parameter_bank(),
        active_axons=list(active_axons), pulse_time_ms=pulse_time_ms, t_ms=t_ms,
    )


def sequential_pulse(cfg: ModelConfig, *, step_ms=50.0, synapse_flash_duration_ms=10.0) -> InputScenario:
    """Each axon fires once, in ``cfg``'s sequential order, one step apart."""
    return build_sequential_pulse_input(
        cfg.axon_assignment_file, cfg.make_parameter_bank(),
        axon_order=cfg.resolved_sequential_axon_order(), step_ms=step_ms, start_ms=step_ms,
        tail_ms=step_ms, synapse_flash_duration_ms=synapse_flash_duration_ms,
    )


def axon_periodic(cfg: ModelConfig, *, active_axons=(0, 3), t_ms=1000.0, delay_ms=20.0,
                  event_rate_hz=1.0, rate_curve_kind="flat", mod_freq_hz=1.0,
                  sine_baseline=0.0) -> InputScenario:
    """Deterministic periodic events on selected axons."""
    return build_axon_periodic_input(
        cfg.axon_assignment_file, cfg.make_parameter_bank(),
        active_axons=list(active_axons), t_ms=t_ms, delay_ms=delay_ms, event_rate_hz=event_rate_hz,
        rate_curve_kind=rate_curve_kind, mod_freq_hz=mod_freq_hz, sine_baseline=sine_baseline,
    )


def axon_poisson(cfg: ModelConfig, *, active_axons=(0, 1, 2, 3, 4), t_ms=1000.0, delay_ms=20.0,
                 event_rate_hz=5.0, rate_curve_kind="flat", mod_freq_hz=1.0, sine_baseline=0.0,
                 poisson_arp_ms=0.0) -> InputScenario:
    """Poisson events on selected axons."""
    n = cfg.n_axons()
    return build_axon_poisson_input(
        cfg.axon_assignment_file, cfg.make_parameter_bank(),
        active_axons=[a for a in active_axons if a < n], t_ms=t_ms, delay_ms=delay_ms,
        event_rate_hz=event_rate_hz, rate_curve_kind=rate_curve_kind, mod_freq_hz=mod_freq_hz,
        sine_baseline=sine_baseline, poisson_arp_ms=poisson_arp_ms,
    )


def synapse_poisson(cfg: ModelConfig, *, t_ms=500.0, delay_ms=10.0, event_rate_hz=2.0,
                    rate_curve_kind="flat", mod_freq_hz=1.0, sine_baseline=0.0, poisson_arp_ms=0.0,
                    seed=42, synapse_flash_duration_ms=10.0) -> InputScenario:
    """An independent Poisson stream on every synapse."""
    return build_synapse_poisson_input(
        cfg.axon_assignment_file, cfg.make_parameter_bank(),
        t_ms=t_ms, delay_ms=delay_ms, event_rate_hz=event_rate_hz, rate_curve_kind=rate_curve_kind,
        mod_freq_hz=mod_freq_hz, sine_baseline=sine_baseline, poisson_arp_ms=poisson_arp_ms,
        seed=seed, synapse_flash_duration_ms=synapse_flash_duration_ms,
    )


SCENARIOS: Dict[str, Callable[..., InputScenario]] = {
    "simultaneous": simultaneous_pulse,
    "sequential": sequential_pulse,
    "periodic": axon_periodic,
    "axon-poisson": axon_poisson,
    "synapse-poisson": synapse_poisson,
}
