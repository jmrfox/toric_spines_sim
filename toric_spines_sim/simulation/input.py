"""Axon assignment files, event remapping, and random axon rate curves."""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pynapple as nap

from toric_spines_sim.events import SineRateCurve, FlatRateCurve

logger = logging.getLogger(__name__)


def load_axon_events_from_file(
    axon_assignment_file: Path,
    axon_rates_hz: List[float],
) -> Tuple[List[FlatRateCurve], List[int], List[List[int]]]:
    """Build per-axon flat rate curves from a synapse assignment file.

    Each line is one axon: comma-separated **1-based** synapse indices.

    Parameters
    ----------
    axon_assignment_file : path-like
        Assignment file (one axon per line).
    axon_rates_hz : list of float
        Rate in Hz for each axon. Length must match the number of lines.
        Use ``0.0`` to silence an axon.

    Returns
    -------
    rate_curves : list of FlatRateCurve
    n_synapses_per_axon : list of int
    axon_synapses : list of list of int
        0-based synapse indices per axon (for ``remap_axon_channel_events_to_synapses``).
    """
    # Read synapse assignments from file
    axon_synapses = []
    with open(axon_assignment_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse comma-separated indices (1-based), convert to 0-based
            synapse_indices = [int(x.strip()) - 1 for x in line.split(',')]
            axon_synapses.append(synapse_indices)

    n_axons = len(axon_synapses)
    if len(axon_rates_hz) != n_axons:
        raise ValueError(
            f"axon_rates_hz length ({len(axon_rates_hz)}) must match "
            f"number of axons in file ({n_axons})"
        )

    # Create flat rate curves for each axon
    rate_curves = []
    for rate_hz in axon_rates_hz:
        rate_curves.append(FlatRateCurve(rate_hz=rate_hz))

    # Count synapses per axon
    n_synapses_per_axon = [len(synapses) for synapses in axon_synapses]

    total_synapses = sum(n_synapses_per_axon)
    logger.info(
        f"Loaded {n_axons} axons for {total_synapses} synapses from "
        f"{axon_assignment_file} (synapses_per_axon: {n_synapses_per_axon})"
    )

    return rate_curves, n_synapses_per_axon, axon_synapses


def remap_axon_channel_events_to_synapses(
    events_tsgroup: nap.TsGroup,
    axon_synapses: List[List[int]],
    n_synapses: Optional[int] = None,
) -> nap.TsGroup:
    """Map axon-ordered event channels onto synapse point-file indices.

    Event generators fan out channels in axon order (axon 0 synapses, then
    axon 1, …), but ``TSRecipe`` maps TsGroup index ``i`` to synapse
    ``syn_i``. This scatters axon-channel events onto the correct indices.

    Parameters
    ----------
    events_tsgroup : pynapple.TsGroup
        Channels ordered by axon assignment (generator output).
    axon_synapses : list of list of int
        Per-axon 0-based synapse indices (from ``load_axon_events_from_file``).
    n_synapses : int, optional
        Total synapses. Defaults to ``max(index) + 1`` over assignments.

    Returns
    -------
    pynapple.TsGroup
        Indexed ``0 .. n_synapses-1`` with labels ``syn_0``, ``syn_1``, …

    Examples
    --------
    >>> # Axon 0 hits synapses 2 then 0; axon 1 hits synapse 1
    >>> axon_synapses = [[2, 0], [1]]
    >>> remapped = remap_axon_channel_events_to_synapses(events, axon_synapses)  # doctest: +SKIP
    >>> remapped.get_info("label")[0]
    'syn_0'
    """
    expected_channels = sum(len(synapses) for synapses in axon_synapses)
    if len(events_tsgroup) != expected_channels:
        raise ValueError(
            f"events_tsgroup has {len(events_tsgroup)} channels, expected "
            f"{expected_channels} from axon_synapses"
        )

    if n_synapses is None:
        if not any(axon_synapses):
            raise ValueError("axon_synapses is empty; cannot infer n_synapses")
        n_synapses = max(syn_idx for synapses in axon_synapses for syn_idx in synapses) + 1

    remapped_by_index: dict[int, nap.Ts] = {}
    channel_idx = 0
    for synapse_indices in axon_synapses:
        for syn_idx in synapse_indices:
            if syn_idx in remapped_by_index:
                raise ValueError(
                    f"Synapse index {syn_idx} appears in multiple axon assignments"
                )
            if syn_idx < 0 or syn_idx >= n_synapses:
                raise ValueError(
                    f"Synapse index {syn_idx} out of range for n_synapses={n_synapses}"
                )
            remapped_by_index[syn_idx] = events_tsgroup[channel_idx]
            channel_idx += 1

    # Build 0..n-1 in order so TsGroup label metadata aligns with synapse indices.
    remapped = {
        syn_idx: remapped_by_index.get(
            syn_idx, nap.Ts(t=[], time_units="ms")
        )
        for syn_idx in range(n_synapses)
    }

    labels = [f"syn_{i}" for i in range(n_synapses)]
    return nap.TsGroup(
        remapped,
        label=labels,
        time_support=events_tsgroup.time_support,
    )


def random_axon_events(
    n_synapses: int,
    n_axons: int,
    mod_freq_hz: float,
    peak_rate_hz: float = None,
    peak_rate_range_hz: Tuple[float, float] = None,
    phase_range_rad: Tuple[float, float] = (0.0, 2.0 * np.pi),
    seed: int = 42,
) -> Tuple[List[SineRateCurve], List[int]]:
    """Create random sine rate curves and a random synapse-to-axon partition.

    Provide exactly one of ``peak_rate_hz`` or ``peak_rate_range_hz``.

    Parameters
    ----------
    n_synapses : int
        Total number of synapses.
    n_axons : int
        Number of axons (must be ``<= n_synapses``).
    mod_freq_hz : float
        Common sine frequency (Hz).
    peak_rate_hz : float, optional
        Peak rate for every axon (Hz).
    peak_rate_range_hz : tuple of float, optional
        ``(min, max)`` from which each axon's peak is drawn.
    phase_range_rad : tuple of float
        Phase draw range in radians. Default ``(0, 2π)``.
    seed : int
        RNG seed.

    Returns
    -------
    rate_curves : list of SineRateCurve
    n_synapses_per_axon : list of int
    """
    if n_axons > n_synapses:
        raise ValueError(f"n_axons ({n_axons}) must be <= n_synapses ({n_synapses})")

    rng = np.random.default_rng(seed)

    # Randomly assign synapses to axons
    # Start with at least 1 synapse per axon, then distribute remaining
    base_assignment = np.arange(n_axons) % n_synapses
    remaining = n_synapses - n_axons
    extra_assignments = rng.integers(0, n_axons, size=remaining)
    assignments = np.concatenate([base_assignment, extra_assignments])
    rng.shuffle(assignments)

    # Count synapses per axon
    n_synapses_per_axon = [int((assignments == i).sum()) for i in range(n_axons)]

    # Create random rate curves for each axon
    rate_curves = []
    for i in range(n_axons):
        if peak_rate_hz is not None:
            peak_rate = peak_rate_hz
        elif peak_rate_range_hz is not None:
            peak_rate = rng.uniform(peak_rate_range_hz[0], peak_rate_range_hz[1])
        else:
            raise ValueError("Either peak_rate_hz or peak_rate_range_hz must be provided")
        phase = rng.uniform(phase_range_rad[0], phase_range_rad[1])
        rate_curves.append(
            SineRateCurve(peak_rate_hz=peak_rate, freq_hz=mod_freq_hz, phase_rad=phase)
        )

    logger.info(
        f"Created {n_axons} axons for {n_synapses} synapses "
        f"(synapses_per_axon: {n_synapses_per_axon})"
    )
    return rate_curves, n_synapses_per_axon
