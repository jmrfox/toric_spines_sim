"""Rate curve classes for defining time-varying event rates.

Rate curves define the instantaneous rate of event generation over time.
They are used by event generators to produce either deterministic or
stochastic event streams.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class RateCurve(ABC):
    """Abstract base class for rate curves.

    Subclasses must implement ``rate_at`` for getting instantaneous rate,
    ``max_rate`` for the upper bound (used in thinning), and ``get_isis``
    for deterministic event generation.
    """

    @abstractmethod
    def rate_at(self, t_ms: float, T_ms: float | None = None) -> float:
        """Return the instantaneous rate at time ``t_ms`` (Hz).

        Parameters
        ----------
        t_ms : float
            Time in milliseconds.
        T_ms : float, optional
            Total duration in milliseconds. Required by curves whose shape
            is defined over a finite window (currently ``LinearRateCurve``).
        """
        pass

    @abstractmethod
    def max_rate(self) -> float:
        """Return the maximum rate over all time (for thinning algorithm)."""
        pass

    @abstractmethod
    def get_isis(self, T_ms: float, delay_ms: float) -> List[float]:
        """Compute inter-spike intervals for deterministic generation.

        Parameters
        ----------
        T_ms : float
            Total simulation time in milliseconds.
        delay_ms : float
            Initial delay before events start.

        Returns
        -------
        list of float
            ISIs in ms. Cumulative sum plus delay gives event times.
        """
        pass


class FlatRateCurve(RateCurve):
    """Constant (flat) rate curve.

    Args:
        rate_hz: Constant rate in Hz.
    """

    def __init__(self, rate_hz: float):
        self._rate_hz = float(rate_hz)

    def rate_at(self, t_ms: float, T_ms: float | None = None) -> float:
        return max(0.0, self._rate_hz)

    def max_rate(self) -> float:
        return max(0.0, self._rate_hz)

    def get_isis(self, T_ms: float, delay_ms: float) -> List[float]:
        if self._rate_hz <= 0:
            return []
        period_ms = 1000.0 / self._rate_hz
        isis = []
        t = delay_ms
        end = T_ms - delay_ms
        while t < end:
            isis.append(period_ms)
            t += period_ms
        return isis


class LinearRateCurve(RateCurve):
    """Linear ramp from ``rate_start_hz`` at t=0 to ``rate_end_hz`` at t=T_ms.

    Stochastic generators pass ``T_ms`` into ``rate_at`` so Poisson thinning
    follows the same ramp as deterministic ``get_isis``.

    Parameters
    ----------
    rate_start_hz : float
        Rate at t=0 (Hz).
    rate_end_hz : float
        Rate at t=T_ms (Hz).

    Examples
    --------
    >>> curve = LinearRateCurve(10.0, 20.0)
    >>> curve.rate_at(500.0, T_ms=1000.0)
    15.0
    """

    def __init__(self, rate_start_hz: float, rate_end_hz: float):
        self._rate_start_hz = float(rate_start_hz)
        self._rate_end_hz = float(rate_end_hz)

    def rate_at(self, t_ms: float, T_ms: float | None = None) -> float:
        if T_ms is None:
            raise ValueError(
                "LinearRateCurve.rate_at requires T_ms (the duration over which "
                "the rate ramps from rate_start_hz to rate_end_hz)"
            )
        if T_ms <= 0:
            raise ValueError("T_ms must be positive")
        frac = float(np.clip(t_ms / T_ms, 0.0, 1.0))
        rate = self._rate_start_hz + (self._rate_end_hz - self._rate_start_hz) * frac
        return max(0.0, rate)

    def max_rate(self) -> float:
        return max(0.0, self._rate_start_hz, self._rate_end_hz)

    def get_isis(self, T_ms: float, delay_ms: float) -> List[float]:
        # For linear rate: r(t) = r0 + (r1-r0)*t/T
        # The ISI at time t is approximately 1000/r(t) for small intervals
        # For exact deterministic firing, we integrate: ∫r(t)dt = n
        if T_ms <= 2 * delay_ms:
            return []

        r0 = self._rate_start_hz
        r1 = self._rate_end_hz
        slope = (r1 - r0) / T_ms if T_ms > 0 else 0

        isis = []
        t = delay_ms
        n = 0
        end = T_ms - delay_ms

        while t < end:
            # Rate at current time
            rate = max(0.0, r0 + slope * t)
            if rate <= 0:
                # Move forward until rate becomes positive or we hit end
                if slope > 0 and r0 < 0:
                    t = -r0 / slope
                    if t >= end:
                        break
                    continue
                else:
                    break

            # For deterministic firing at rate r(t), ISI = 1000/r(t)
            isi = 1000.0 / rate
            isis.append(isi)
            t += isi
            n += 1

            # Safety limit
            if n > 1000000:
                break

        return isis


class StepRateCurve(RateCurve):
    """Piecewise-constant (step) rate curve.

    Args:
        rates_hz: List of rates for each step phase (in Hz).
        step_duration_ms: Duration of each step phase in ms.
    """

    def __init__(self, rates_hz: List[float], step_duration_ms: float):
        self._rates_hz = [float(r) for r in rates_hz]
        self._step_duration_ms = float(step_duration_ms)
        self._n_steps = len(self._rates_hz)

    def rate_at(self, t_ms: float, T_ms: float | None = None) -> float:
        if self._step_duration_ms <= 0:
            return 0.0
        step_idx = int(t_ms // self._step_duration_ms)
        if 0 <= step_idx < self._n_steps:
            return max(0.0, self._rates_hz[step_idx])
        return 0.0

    def max_rate(self) -> float:
        if not self._rates_hz:
            return 0.0
        return max(0.0, max(self._rates_hz))

    def get_isis(self, T_ms: float, delay_ms: float) -> List[float]:
        if T_ms <= 2 * delay_ms or self._step_duration_ms <= 0:
            return []

        isis = []
        t = delay_ms
        end = T_ms - delay_ms

        while t < end:
            # Determine current step
            step_idx = int(t // self._step_duration_ms)
            if step_idx >= self._n_steps:
                break

            rate = max(0.0, self._rates_hz[step_idx]) if step_idx < len(self._rates_hz) else 0.0
            if rate <= 0:
                # Skip to next step
                next_step_start = (step_idx + 1) * self._step_duration_ms
                t = max(t, next_step_start)
                continue

            isi = 1000.0 / rate

            # Check if ISI crosses step boundary
            step_end = (step_idx + 1) * self._step_duration_ms
            if t + isi > step_end:
                # Partial ISI to end of step
                isi = step_end - t

            isis.append(isi)
            t += isi

        return isis


class SineRateCurve(RateCurve):
    """Sinusoidally modulated rate curve.

    Rate follows: r(t) = peak_rate_hz * max(0, sin(2π * freq_hz * t_ms/1000 + phase) + baseline)

    Args:
        peak_rate_hz: Peak amplitude of the sine wave in Hz.
        freq_hz: Frequency of modulation in Hz.
        baseline: Offset added to sine wave (default: 0.0).
        phase_rad: Phase offset in radians (mutually exclusive with phase_deg).
        phase_deg: Phase offset in degrees (mutually exclusive with phase_rad).

    Raises:
        ValueError: If both phase_rad and phase_deg are provided.
    """

    def __init__(
        self,
        peak_rate_hz: float,
        freq_hz: float,
        baseline: float = 0.0,
        phase_rad: float | None = None,
        phase_deg: float | None = None,
    ):
        if phase_rad is not None and phase_deg is not None:
            raise ValueError("Cannot specify both phase_rad and phase_deg")

        self._peak_rate_hz = float(peak_rate_hz)
        self._freq_hz = float(freq_hz)
        self._baseline = float(baseline)

        if phase_rad is not None:
            self._phase = float(phase_rad)
        elif phase_deg is not None:
            self._phase = float(phase_deg) * np.pi / 180.0
        else:
            self._phase = 0.0

        # Precompute angular frequency (rad per ms)
        self._omega = 2.0 * np.pi * self._freq_hz / 1000.0

    def to_sine_v1_params(self) -> dict[str, float]:
        """Return compact parameters for JSON serialization and plot reconstruction."""
        return {
            "peak_rate_hz": self._peak_rate_hz,
            "freq_hz": self._freq_hz,
            "phase_rad": self._phase,
            "baseline": self._baseline,
        }

    def rate_at(self, t_ms: float, T_ms: float | None = None) -> float:
        if self._peak_rate_hz <= 0:
            return 0.0
        val = np.sin(self._omega * t_ms + self._phase) + self._baseline
        return self._peak_rate_hz * max(0.0, val)

    def max_rate(self) -> float:
        if self._peak_rate_hz <= 0:
            return 0.0
        # Maximum of sin + baseline is 1 + baseline
        return self._peak_rate_hz * max(0.0, 1.0 + self._baseline)

    def get_isis(self, T_ms: float, delay_ms: float) -> List[float]:
        # For sine-modulated rate, we numerically integrate to find event times
        # This is approximate - we use adaptive stepping based on local rate
        if T_ms <= 2 * delay_ms or self._peak_rate_hz <= 0:
            return []

        isis = []
        t = delay_ms
        end = T_ms - delay_ms
        dt_max = 1.0  # Maximum step size in ms

        while t < end:
            rate = self.rate_at(t)
            if rate <= 0:
                # Find next time where rate becomes positive
                # Search forward in small increments
                t_search = t
                found = False
                while t_search < end:
                    if self.rate_at(t_search) > 0:
                        t = t_search
                        found = True
                        break
                    t_search += dt_max
                if not found:
                    break
                continue

            # ISI based on current rate
            isi = min(1000.0 / rate, dt_max)
            isis.append(isi)
            t += isi

        return isis
