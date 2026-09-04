import numpy as np
import pynapple as nap
from typing import Tuple, List, Optional
from pathlib import Path
from toric_spines_sim.events import SineRateCurve, FlatRateCurve
import logging

logger = logging.getLogger(__name__)


def load_axon_events_from_file(
    axon_assignment_file: Path,
    axon_rates_hz: List[float],
) -> Tuple[List[FlatRateCurve], List[int], List[List[int]]]:
    """Create axon rate curves from a synapse assignment file with periodic.

    Reads a text file where each line corresponds to an axon and contains
    comma-separated synapse indices (1-based) for that axon.

    Args:
        axon_assignment_file: Path to text file with synapse assignments.
            Format: one line per axon, each line is "syn1, syn2, syn3, ..."
            Synapse indices are 1-based.
        axon_rates_hz: List of rates for each axon (Hz). Length must match
            number of lines in the file. Use 0.0 to turn an axon off.

    Returns:
        Tuple of ``(rate_curves, n_synapses_per_axon, axon_synapses)``:
        - rate_curves: List of FlatRateCurve objects (one per axon)
        - n_synapses_per_axon: List of synapse counts per axon
        - axon_synapses: 0-based synapse index lists per axon
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
    """Map axon-ordered event channels to synapse point-file indices.

    Event generators fan out channels in axon order (axon 0 synapses, then axon 1,
    etc.), but the simulator maps TsGroup index ``i`` to synapse ``syn_i`` in the
    order of the synapse points file. This function scatters axon-channel events
    onto the correct synapse indices from an axon assignment file.

    Args:
        events_tsgroup: TsGroup with channels ordered by axon assignment.
        axon_synapses: Per-axon lists of 0-based synapse indices (as returned
            by ``load_axon_events_from_file``).
        n_synapses: Total number of synapses. Defaults to
            ``max(index) + 1`` over all assigned synapses.

    Returns:
        TsGroup indexed by synapse point-file order (0 .. n_synapses - 1).
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
    phase_range_rad: Tuple[float, float] = None,
    seed: int = 42,
) -> Tuple[List[SineRateCurve], List[int]]:
    """Create random axon rate curves and assign synapses to axons.

    Args:
        n_synapses: Total number of synapses
        n_axons: Number of axons (must be <= n_synapses)
        mod_freq_hz: Common frequency for all sine rate curves (Hz)
        peak_rate_hz: Peak rate for all sine rate curves (Hz)
        peak_rate_range_hz: (min, max) range for random peak rates
        phase_range_rad: (min, max) range for random phases (default 0 to 2π)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (rate_curves, n_synapses_per_axon) where:
        - rate_curves: List of SineRateCurve objects (one per axon)
        - n_synapses_per_axon: List of synapse counts per axon
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
