"""Unified event generator classes for simulation event streams.

This module provides deterministic and stochastic event generators that can
operate in either shared-source mode (one master process routed to multiple
axons) or independent mode (each axon has its own process).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pynapple as nap

from .base import EventGenerator
from .rate_curves import RateCurve


class DeterministicEventGenerator(EventGenerator):
    """Deterministic event generator using rate curves.

    Generates events at times determined by integrating the rate curve.
    For a flat rate curve, this produces periodic (regularly spaced) events.
    For varying rate curves, event density follows the rate profile.

    Can operate in shared mode (single rate curve routed to multiple axons)
    or independent mode (per-axon rate curves).

    Args:
        rate_curves: Single RateCurve for shared mode, or sequence of
            RateCurves (one per axon) for independent mode.
        n_synapses_per_axon: Number of synapses attached to each axon.
            ``len(n_synapses_per_axon)`` is the number of axons (N_A).
            The total number of output channels is ``sum(n_synapses_per_axon)``.
        T_ms: Total simulation time in milliseconds.
        delay_ms: Time before events start (ms). Default: 0.0.
        labels: Optional labels for each output channel. Length must equal
            ``sum(n_synapses_per_axon)``. Default labels are ``A{a}S{s}``.
            ``TSRecipe`` / ``TSSimulator`` ignore those labels and map by
            TsGroup **index** to ``syn_0``, ``syn_1``, … — use
            ``remap_axon_channel_events_to_synapses`` when axon assignments
            are not point-file order.
        routing_mode: How to route master events to axons in shared mode.
            "broadcast": every master event goes to all axons.
            "roundrobin": events distributed cyclically among axons.
            Only used in shared mode. Defaults to "roundrobin".
    """

    def __init__(
        self,
        rate_curves: Union[RateCurve, Sequence[RateCurve]],
        n_synapses_per_axon: Sequence[int],
        T_ms: float,
        delay_ms: float = 0.0,
        labels: Optional[Sequence[str]] = None,
        routing_mode: str = "roundrobin",
    ):
        self._n_synapses_per_axon: List[int] = list(n_synapses_per_axon)
        self._n_axons: int = len(self._n_synapses_per_axon)
        self._n_channels: int = sum(self._n_synapses_per_axon)
        self._T_ms = float(T_ms)
        self._delay_ms = float(delay_ms)
        self._labels: Optional[List[str]] = list(labels) if labels is not None else None

        # Determine shared vs independent mode
        if isinstance(rate_curves, RateCurve):
            # Shared mode: one curve for all axons
            self._shared_mode = True
            self._shared_curve: Optional[RateCurve] = rate_curves
            self._axon_curves: List[RateCurve] = []

            # Validate routing_mode
            if routing_mode not in ("broadcast", "roundrobin"):
                raise ValueError(
                    f"routing_mode must be 'broadcast' or 'roundrobin', got {routing_mode}"
                )
            self._routing_mode: str = routing_mode
        else:
            # Independent mode: per-axon curves
            self._shared_mode = False
            self._shared_curve = None
            self._axon_curves: List[RateCurve] = list(rate_curves)

            if len(self._axon_curves) != self._n_axons:
                raise ValueError(
                    f"Number of rate_curves ({len(self._axon_curves)}) must match "
                    f"number of axons ({self._n_axons})"
                )
            self._routing_mode: str = routing_mode

    def _resolve_labels(self, override: Optional[Sequence[str]]) -> List[str]:
        """Resolve stream labels from override, stored labels, or auto-generate."""
        if override is not None:
            labs = list(override)
        elif self._labels is not None:
            labs = list(self._labels)
        else:
            # Generate labels: A0S0, A0S1, ..., A1S0, A1S1, etc.
            labs = []
            for axon_idx, n_syn in enumerate(self._n_synapses_per_axon):
                for syn_idx in range(n_syn):
                    labs.append(f"A{axon_idx}S{syn_idx}")

        if len(labs) != self._n_channels:
            raise ValueError(
                f"labels length {len(labs)} must match number of channels {self._n_channels}"
            )
        if len(set(labs)) != self._n_channels:
            raise ValueError("labels must be unique")

        return labs

    def generate(self, labels: Optional[Sequence[str]] = None) -> nap.TsGroup:
        """Generate deterministic event times for all output channels."""
        labs = self._resolve_labels(labels)

        if self._shared_mode:
            # Shared mode: generate master times from single curve
            assert self._shared_curve is not None
            master_isis = self._shared_curve.get_isis(self._T_ms, self._delay_ms)
            master_times = self._cumulative_to_times(master_isis)

            # Route events to axons (deterministic routing)
            axon_times = self._route_shared_times(master_times)
        else:
            # Independent mode: each axon has its own times
            axon_times = []
            for curve in self._axon_curves:
                isis = curve.get_isis(self._T_ms, self._delay_ms)
                times = self._cumulative_to_times(isis)
                axon_times.append(times)

        # Fan-out: assign axon event times to each synapse channel
        ts_dict: dict = {}
        channel_idx = 0
        for axon_idx, n_syn in enumerate(self._n_synapses_per_axon):
            times = axon_times[axon_idx]
            for _ in range(n_syn):
                ts_dict[channel_idx] = nap.Ts(t=times, time_units="ms")
                channel_idx += 1

        time_support = nap.IntervalSet(
            start=[0], end=[max(self._T_ms, 1.0)], time_units="ms"
        )
        return nap.TsGroup(ts_dict, label=labs, time_support=time_support)

    def _cumulative_to_times(self, isis: List[float]) -> List[float]:
        """Convert list of ISIs to cumulative event times starting from delay."""
        times = []
        t = self._delay_ms
        end = self._T_ms - self._delay_ms
        for isi in isis:
            if t >= end:
                break
            times.append(float(t))
            t += isi
        return times

    def _route_shared_times(self, master_times: List[float]) -> List[List[float]]:
        """Route master event times to axons based on routing_mode."""
        axon_times: List[List[float]] = [[] for _ in range(self._n_axons)]

        if not master_times:
            return axon_times

        if self._routing_mode == "broadcast":
            # Broadcast: every master event goes to all axons
            for t in master_times:
                for axon_idx in range(self._n_axons):
                    axon_times[axon_idx].append(t)
        else:  # roundrobin
            # Distribute events cyclically among axons
            for i, t in enumerate(master_times):
                axon_idx = i % self._n_axons
                axon_times[axon_idx].append(t)

        return axon_times

    def __repr__(self) -> str:
        """Return informative string representation."""
        mode = "shared" if self._shared_mode else "independent"
        routing = f", routing={self._routing_mode}" if self._shared_mode else ""
        return (
            f"DeterministicEventGenerator("
            f"mode={mode}, axons={self._n_axons}, "
            f"channels={self._n_channels}, T={self._T_ms}ms"
            f"{routing}, delay={self._delay_ms}ms)"
        )


class StochasticEventGenerator(EventGenerator):
    """Stochastic event generator using inhomogeneous Poisson process.

    Generates events via thinning algorithm applied to rate curves.
    Supports absolute refractory period (ARP) to enforce minimum ISI.

    Can operate in shared mode (single rate curve routed to multiple axons)
    or independent mode (per-axon rate curves with independent processes).

    Args:
        rate_curves: Single RateCurve for shared mode, or sequence of
            RateCurves (one per axon) for independent mode.
        n_synapses_per_axon: Number of synapses attached to each axon.
            ``len(n_synapses_per_axon)`` is the number of axons (N_A).
            The total number of output channels is ``sum(n_synapses_per_axon)``.
        T_ms: Total simulation time in milliseconds.
        delay_ms: Time before events start (ms). Default: 0.0.
        seed: Random seed for reproducibility. Default: None.
        labels: Optional labels for each output channel. Length must equal
            ``sum(n_synapses_per_axon)``. Default labels are ``A{a}S{s}``.
            ``TSRecipe`` / ``TSSimulator`` map by TsGroup **index** to
            ``syn_i``; use ``remap_axon_channel_events_to_synapses`` when
            axon assignments are not point-file order.
        routing_weights: Probability of routing each master event to each axon.
            Only used in shared mode. Must sum to 1. Defaults to uniform.
        arp_ms: Absolute refractory period in ms. Can be scalar (same for all
            axons) or per-axon list. Default: 0.0.
    """

    def __init__(
        self,
        rate_curves: Union[RateCurve, Sequence[RateCurve]],
        n_synapses_per_axon: Sequence[int],
        T_ms: float,
        delay_ms: float = 0.0,
        seed: Optional[int] = None,
        labels: Optional[Sequence[str]] = None,
        routing_weights: Optional[Sequence[float]] = None,
        arp_ms: Union[float, Sequence[float]] = 0.0,
    ):
        self._n_synapses_per_axon: List[int] = list(n_synapses_per_axon)
        self._n_axons: int = len(self._n_synapses_per_axon)
        self._n_channels: int = sum(self._n_synapses_per_axon)
        self._T_ms = float(T_ms)
        self._delay_ms = float(delay_ms)
        self._seed = seed
        self._labels: Optional[List[str]] = list(labels) if labels is not None else None

        # Resolve ARP
        if isinstance(arp_ms, (int, float)):
            self._arp_ms: List[float] = [float(arp_ms)] * self._n_axons
        else:
            self._arp_ms = [float(a) for a in arp_ms]
            if len(self._arp_ms) != self._n_axons:
                raise ValueError(
                    f"arp_ms length {len(self._arp_ms)} must match number of axons {self._n_axons}"
                )

        # Determine shared vs independent mode
        if isinstance(rate_curves, RateCurve):
            # Shared mode: one curve for all axons
            self._shared_mode = True
            self._shared_curve: Optional[RateCurve] = rate_curves
            self._axon_curves: List[RateCurve] = []

            # Validate routing_weights
            if routing_weights is not None:
                if len(routing_weights) != self._n_axons:
                    raise ValueError(
                        f"routing_weights length {len(routing_weights)} must match "
                        f"number of axons {self._n_axons}"
                    )
                weight_sum = sum(routing_weights)
                if not np.isclose(weight_sum, 1.0):
                    raise ValueError(
                        f"routing_weights must sum to 1.0, got {weight_sum}"
                    )
                self._routing_weights: Optional[List[float]] = list(routing_weights)
            else:
                self._routing_weights = None
        else:
            # Independent mode: per-axon curves
            self._shared_mode = False
            self._shared_curve = None
            self._axon_curves: List[RateCurve] = list(rate_curves)

            if len(self._axon_curves) != self._n_axons:
                raise ValueError(
                    f"Number of rate_curves ({len(self._axon_curves)}) must match "
                    f"number of axons ({self._n_axons})"
                )
            self._routing_weights = None

    def _resolve_labels(self, override: Optional[Sequence[str]]) -> List[str]:
        """Resolve stream labels from override, stored labels, or auto-generate."""
        if override is not None:
            labs = list(override)
        elif self._labels is not None:
            labs = list(self._labels)
        else:
            # Generate labels: A0S0, A0S1, ..., A1S0, A1S1, etc.
            labs = []
            for axon_idx, n_syn in enumerate(self._n_synapses_per_axon):
                for syn_idx in range(n_syn):
                    labs.append(f"A{axon_idx}S{syn_idx}")

        if len(labs) != self._n_channels:
            raise ValueError(
                f"labels length {len(labs)} must match number of channels {self._n_channels}"
            )
        if len(set(labs)) != self._n_channels:
            raise ValueError("labels must be unique")

        return labs

    @staticmethod
    def _apply_arp_filter(times: List[float], arp_ms: float) -> List[float]:
        """Greedy refractory filter: keep first event, drop any within arp_ms of it."""
        if arp_ms <= 0.0:
            return times
        accepted: List[float] = []
        last = -np.inf
        for t in times:  # times must be sorted
            if t - last >= arp_ms:
                accepted.append(t)
                last = t
        return accepted

    def generate(self, labels: Optional[Sequence[str]] = None) -> nap.TsGroup:
        """Generate stochastic event times for all output channels."""
        labs = self._resolve_labels(labels)

        # Set up RNGs
        if self._seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self._seed)

        if self._shared_mode:
            # Shared mode: generate master times, then route
            assert self._shared_curve is not None
            master_times = self._generate_thinning(self._shared_curve, rng)

            # Route events to axons
            axon_times = self._route_shared_times_stochastic(master_times, rng)
        else:
            # Independent mode: each axon generates its own times
            axon_times = []
            for curve in self._axon_curves:
                times = self._generate_thinning(curve, rng)
                axon_times.append(times)

        # Apply ARP filter per axon
        axon_times = [
            self._apply_arp_filter(times, self._arp_ms[i])
            for i, times in enumerate(axon_times)
        ]

        # Fan-out: assign axon event times to each synapse channel
        ts_dict: dict = {}
        channel_idx = 0
        for axon_idx, n_syn in enumerate(self._n_synapses_per_axon):
            times = axon_times[axon_idx]
            for _ in range(n_syn):
                ts_dict[channel_idx] = nap.Ts(t=times, time_units="ms")
                channel_idx += 1

        time_support = nap.IntervalSet(
            start=[0], end=[max(self._T_ms, 1.0)], time_units="ms"
        )
        metadata: dict = {"label": labs}
        if self._seed is not None:
            metadata["seed"] = [self._seed] * self._n_channels
        return nap.TsGroup(ts_dict, time_support=time_support, **metadata)

    def _generate_thinning(
        self, curve: RateCurve, rng: np.random.Generator
    ) -> List[float]:
        """Generate events via Lewis-Shedler thinning algorithm."""
        if self._T_ms <= 2 * self._delay_ms:
            return []

        rate_max = curve.max_rate()
        if rate_max <= 0:
            return []

        scale_ms = 1000.0 / rate_max
        t = self._delay_ms + rng.exponential(scale=scale_ms)
        end = self._T_ms - self._delay_ms
        times: List[float] = []

        while t < end:
            acceptance_prob = curve.rate_at(t, T_ms=self._T_ms) / rate_max
            if rng.random() < acceptance_prob:
                times.append(float(t))
            t += rng.exponential(scale=scale_ms)

        return times

    def _route_shared_times_stochastic(
        self, master_times: List[float], rng: np.random.Generator
    ) -> List[List[float]]:
        """Route master event times to axons using random sampling."""
        axon_times: List[List[float]] = [[] for _ in range(self._n_axons)]

        if not master_times:
            return axon_times

        weights = self._routing_weights  # None -> uniform
        master_array = np.asarray(master_times)

        assigned_axons = rng.choice(self._n_axons, size=len(master_array), p=weights)

        for axon_idx in range(self._n_axons):
            axon_times[axon_idx] = master_array[assigned_axons == axon_idx].tolist()

        return axon_times

    def __repr__(self) -> str:
        """Return informative string representation."""
        mode = "shared" if self._shared_mode else "independent"
        routing_info = ""
        if self._shared_mode and self._routing_weights:
            routing_info = f", weights={self._routing_weights}"
        elif self._shared_mode:
            routing_info = ", uniform routing"

        seed_info = f", seed={self._seed}" if self._seed is not None else ""
        arp_info = (
            f", arp={self._arp_ms[0]}ms"
            if len(set(self._arp_ms)) == 1
            else f", arp={self._arp_ms}ms"
        )

        return (
            f"StochasticEventGenerator("
            f"mode={mode}, axons={self._n_axons}, "
            f"channels={self._n_channels}, T={self._T_ms}ms"
            f"{routing_info}{seed_info}{arp_info}, delay={self._delay_ms}ms)"
        )
