"""Tests for toric_spines_sim.simulation.input module."""

import numpy as np
import pynapple as nap
import pytest

from toric_spines_sim.simulation.input import (
    load_axon_events_from_file,
    remap_axon_channel_events_to_synapses,
)


class TestRemapAxonChannelEventsToSynapses:
    @staticmethod
    def _make_tsgroup(ts_dict: dict[int, nap.Ts]) -> nap.TsGroup:
        time_support = nap.IntervalSet(start=[0], end=[1000.0], time_units="ms")
        return nap.TsGroup(ts_dict, time_support=time_support)

    def test_remaps_channels_to_synapse_file_order(self):
        # Axon 0: synapses 2 and 0; axon 1: synapse 1
        axon_synapses = [[2, 0], [1]]
        events = self._make_tsgroup(
            {
                0: nap.Ts(t=[10.0], time_units="ms"),
                1: nap.Ts(t=[20.0], time_units="ms"),
                2: nap.Ts(t=[30.0], time_units="ms"),
            }
        )

        remapped = remap_axon_channel_events_to_synapses(events, axon_synapses)

        assert len(remapped) == 3
        np.testing.assert_array_equal(remapped[0].as_units("ms").index.values, [20.0])
        np.testing.assert_array_equal(remapped[1].as_units("ms").index.values, [30.0])
        np.testing.assert_array_equal(remapped[2].as_units("ms").index.values, [10.0])

    def test_remap_assigns_labels_matching_synapse_indices(self):
        axon_synapses = [[2, 0], [1]]
        events = self._make_tsgroup(
            {
                0: nap.Ts(t=[10.0], time_units="ms"),
                1: nap.Ts(t=[20.0], time_units="ms"),
                2: nap.Ts(t=[30.0], time_units="ms"),
            }
        )

        remapped = remap_axon_channel_events_to_synapses(events, axon_synapses)
        labels = remapped.get_info("label")

        for syn_idx in range(3):
            assert labels[syn_idx] == f"syn_{syn_idx}"

    def test_load_and_remap_ts1_axon_assignments(self, tmp_path):
        assignment_file = tmp_path / "axons.txt"
        assignment_file.write_text(
            "16, 17, 18, 19, 20, 22, 23, 24\n"
            "2, 3, 6, 7\n"
            "15\n"
            "4, 8, 10, 12\n"
            "1, 5\n"
            "21\n"
            "13\n"
            "9, 11\n"
            "14\n"
            "25\n"
        )

        _, _, axon_synapses = load_axon_events_from_file(
            axon_assignment_file=assignment_file,
            axon_rates_hz=[5.0] * 10,
            T_ms=100.0,
        )

        channel_events = self._make_tsgroup(
            {
                i: nap.Ts(t=[float(i + 1)], time_units="ms")
                for i in range(sum(len(s) for s in axon_synapses))
            }
        )
        remapped = remap_axon_channel_events_to_synapses(
            channel_events,
            axon_synapses,
            n_synapses=25,
        )

        assert len(remapped) == 25
        # Synapse 1 in the file is index 0; assigned to axon 4 as its first synapse.
        axon4_start = sum(len(s) for s in axon_synapses[:4])
        np.testing.assert_array_equal(
            remapped[0].as_units("ms").index.values,
            channel_events[axon4_start].as_units("ms").index.values,
        )

    def test_rejects_duplicate_synapse_indices(self):
        events = self._make_tsgroup(
            {
                0: nap.Ts(t=[1.0], time_units="ms"),
                1: nap.Ts(t=[2.0], time_units="ms"),
            }
        )
        with pytest.raises(ValueError, match="multiple axon assignments"):
            remap_axon_channel_events_to_synapses(events, [[0], [0]])
