"""Tests for animation helpers."""

from __future__ import annotations

import numpy as np

from toric_spines_sim.viz.animation import synapse_flash_active_mask


def test_synapse_flash_active_mask_is_centered_on_event():
    frame_times = np.arange(0.0, 20.0, 1.0)
    event_times = np.array([10.0])

    active = synapse_flash_active_mask(frame_times, event_times, 10.0)

    assert active[4] is np.False_
    assert active[5:16].all()
    assert active[16] is np.False_
    assert active.sum() == 11


def test_synapse_flash_active_mask_merges_overlapping_events():
    frame_times = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    event_times = np.array([0.0, 10.0])

    active = synapse_flash_active_mask(frame_times, event_times, 6.0)

    np.testing.assert_array_equal(active, [True, False, True, False, False])
