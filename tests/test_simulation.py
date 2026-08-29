"""Tests for toric_spines_sim.simulation module."""

import pytest
import numpy as np
import pickle
from pathlib import Path
import pynapple as nap
from toric_spines_sim.simulation import SimulationResults


class TestSimulationResults:
    """Test suite for SimulationResults class."""

    @pytest.fixture
    def mock_results_dict(self):
        """Create a mock results dictionary for testing."""
        # Create minimal mock data structures with pynapple types
        # voltage_traces: TsdFrame with time index and probe columns
        times = np.array([0.0, 0.1, 0.2])
        voltage_data = np.array([[-65.0, -64.8], [-64.5, -64.3], [-64.0, -63.8]])
        mock_voltage_traces = nap.TsdFrame(
            t=times,
            d=voltage_data,
            time_units="ms",
            columns=["probe_seg_0", "probe_seg_1"],
        )

        # input_events: TsGroup with synapse timestamps
        mock_input_events = nap.TsGroup(
            {
                0: nap.Ts(t=[10.0, 20.0, 30.0], time_units="ms"),
                1: nap.Ts(t=[15.0, 25.0, 35.0], time_units="ms"),
            },
            label=["syn_0", "syn_1"],
        )

        mock_record_points = {
            "probe_seg_0": (1.0, 0.0, 0.0),
            "probe_seg_1": (2.0, 0.0, 0.0),
        }

        # Create mock segment tree with minimal structure
        class MockSegment:
            def __init__(self, tag):
                self.tag = tag
                self.prox = type(
                    "obj", (object,), {"x": 0.0, "y": 0.0, "z": 0.0, "radius": 1.0}
                )()
                self.dist = type(
                    "obj", (object,), {"x": 1.0, "y": 0.0, "z": 0.0, "radius": 0.8}
                )()

        class MockSegmentTree:
            def __init__(self):
                self.segments = [MockSegment(1), MockSegment(1)]

        return {
            "voltage_traces": mock_voltage_traces,
            "input_events": mock_input_events,
            "synapses": {},
            "gap_junctions": [],
            "record_points": mock_record_points,
            "cell": None,
            "morphology": None,
            "segment_tree": MockSegmentTree(),
            "decor": None,
            "labels": None,
            "cvp": None,
        }

    def test_init(self, mock_results_dict):
        """Test SimulationResults initialization."""
        results = SimulationResults(mock_results_dict)
        assert isinstance(results.voltage_traces, nap.TsdFrame)
        assert isinstance(results.input_events, nap.TsGroup)
        assert results.record_points == mock_results_dict["record_points"]
        assert results.voltage_traces.shape == (3, 2)  # 3 time points, 2 probes
        assert len(results.input_events) == 2  # 2 synapses

    def test_get_tags(self, mock_results_dict):
        """Test getting unique tags."""
        results = SimulationResults(mock_results_dict)
        tags = results.get_tags()
        assert isinstance(tags, set)
        assert 1 in tags

    def test_get_segments_by_tag(self, mock_results_dict):
        """Test getting segments by tag."""
        results = SimulationResults(mock_results_dict)
        segments = results.get_segments_by_tag(1)
        assert isinstance(segments, list)
        assert len(segments) == 2

    def test_integrate_voltages_by_tag_average(self, mock_results_dict):
        """Test voltage integration with average method."""
        results = SimulationResults(mock_results_dict)
        tsd = results.integrate_voltages_by_tag(1, method="average")
        assert isinstance(tsd, nap.Tsd)
        time = tsd.index.values
        voltage = tsd.values
        assert isinstance(time, np.ndarray)
        assert isinstance(voltage, np.ndarray)
        assert len(time) == len(voltage)

    def test_integrate_voltages_by_tag_surface_weighted(self, mock_results_dict):
        """Test voltage integration with surface-weighted method."""
        results = SimulationResults(mock_results_dict)
        tsd = results.integrate_voltages_by_tag(1, method="surface_weighted")
        assert isinstance(tsd, nap.Tsd)
        assert isinstance(tsd.index.values, np.ndarray)
        assert isinstance(tsd.values, np.ndarray)

    def test_integrate_voltages_multiple_tags(self, mock_results_dict):
        """Test voltage integration with multiple tags."""
        results = SimulationResults(mock_results_dict)
        tsd = results.integrate_voltages_by_tag([1], method="average")
        assert isinstance(tsd, nap.Tsd)
        assert isinstance(tsd.index.values, np.ndarray)

    def test_integrate_voltages_invalid_method(self, mock_results_dict):
        """Test error with invalid integration method."""
        results = SimulationResults(mock_results_dict)
        with pytest.raises(ValueError, match="Unknown method"):
            results.integrate_voltages_by_tag(1, method="invalid")

    def test_integrate_voltages_no_matching_segments(self, mock_results_dict):
        """Test error when no segments match the tag."""
        results = SimulationResults(mock_results_dict)
        with pytest.raises(ValueError, match="No segments found"):
            results.integrate_voltages_by_tag(999)

    def test_to_dict(self, mock_results_dict):
        """Test converting back to dictionary."""
        results = SimulationResults(mock_results_dict)
        result_dict = results.to_dict()
        assert "voltage_traces" in result_dict
        assert "input_events" in result_dict
        assert "record_points" in result_dict
        assert isinstance(result_dict["voltage_traces"], nap.TsdFrame)
        assert isinstance(result_dict["input_events"], nap.TsGroup)

    def test_save_and_load(self, mock_results_dict, temp_dir):
        """Test saving and loading results."""
        results = SimulationResults(mock_results_dict)
        save_path = temp_dir / "results.pkl"

        # Save
        results.save(save_path)
        assert save_path.exists()

        # Load
        loaded_results = SimulationResults.load(save_path)
        # Compare TsdFrame values
        np.testing.assert_array_equal(
            loaded_results.voltage_traces.values, results.voltage_traces.values
        )
        # Compare TsGroup values
        assert len(loaded_results.input_events) == len(results.input_events)
        for idx in results.input_events.keys():
            np.testing.assert_array_equal(
                loaded_results.input_events[idx].index.values,
                results.input_events[idx].index.values,
            )

    def test_load_nonexistent_file(self, temp_dir):
        """Test error when loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            SimulationResults.load(temp_dir / "nonexistent.pkl")

    def test_save_creates_parent_directory(self, mock_results_dict, temp_dir):
        """Test that save creates parent directories."""
        results = SimulationResults(mock_results_dict)
        save_path = temp_dir / "subdir" / "results.pkl"
        results.save(save_path)
        assert save_path.exists()

    def test_caching(self, mock_results_dict):
        """Test that internal caches work correctly."""
        results = SimulationResults(mock_results_dict)

        # First call should populate cache
        tags1 = results._get_segment_tags()
        # Second call should use cache
        tags2 = results._get_segment_tags()
        assert tags1 == tags2

        # Same for surface areas
        areas1 = results._get_segment_surface_areas()
        areas2 = results._get_segment_surface_areas()
        assert areas1 == areas2

    def test_probe_to_segment_mapping(self, mock_results_dict):
        """Test probe to segment mapping."""
        results = SimulationResults(mock_results_dict)
        mapping = results._map_probes_to_segments()
        assert isinstance(mapping, dict)
        assert "probe_seg_0" in mapping
        assert "probe_seg_1" in mapping


class TestTSSimulator:
    """Test suite for TSSimulator class."""

    @pytest.fixture
    def sample_events(self):
        """Minimal TsGroup for simulator construction."""
        time_support = nap.IntervalSet(start=[0], end=[100], time_units="ms")
        return nap.TsGroup(
            {
                0: nap.Ts(t=[10.0], time_units="ms"),
                1: nap.Ts(t=[20.0], time_units="ms"),
                2: nap.Ts(t=[30.0], time_units="ms"),
            },
            time_support=time_support,
        )

    @pytest.fixture
    def sample_parameters(self):
        """Short-run parameter set for integration tests."""
        from toric_spines_sim.simulation import make_default_parameter_bank

        bank = make_default_parameter_bank()
        bank["T_ms"].value = 10.0
        bank["dt_sim_ms"].value = 0.025
        bank["dt_record_ms"].value = 0.5
        bank["seed"].value = 42
        return bank.sample()

    def _make_simulator(
        self,
        swc_filepath,
        synpts_filepath,
        events,
        parameters,
        record_points="all",
    ):
        from toric_spines_sim.simulation import TSSimulator

        return TSSimulator(
            swc_filepath,
            synpts_filepath,
            events,
            parameters,
            record_points=record_points,
        )

    def test_init_stores_configuration(self, sample_events, sample_parameters):
        """Test that constructor stores all configuration."""
        sim = self._make_simulator(
            "/path/to/test.swc",
            "/path/to/synpts.txt",
            sample_events,
            sample_parameters,
        )
        assert sim.swc_filepath == Path("/path/to/test.swc")
        assert sim.synpts_filepath == Path("/path/to/synpts.txt")
        assert sim.events is sample_events
        assert sim.parameters["T_ms"] == 10.0
        assert sim.record_points_spec == "all"

    def test_init_explicit_record_points(self, sample_events, sample_parameters):
        """Test constructor with explicit record point dict."""
        points = {"probe_0": (0.0, 0.0, 0.0), "probe_1": (1.0, 0.0, 0.0)}
        sim = self._make_simulator(
            "/path/to/test.swc",
            "/path/to/synpts.txt",
            sample_events,
            sample_parameters,
            record_points=points,
        )
        assert sim.record_points_spec == points
        assert sim.record_points == points

    def test_properties_build_lazily(
        self,
        sample_swc_file,
        sample_synpts_file,
        sample_events,
        sample_parameters,
    ):
        """Test that properties trigger building when accessed."""
        sim = self._make_simulator(
            sample_swc_file,
            sample_synpts_file,
            sample_events,
            sample_parameters,
        )
        assert sim._synapses is None
        assert sim._build_cell_results is None

        synapses = sim.synapses
        assert synapses is not None
        assert len(synapses) == 3
        assert sim._synapses is synapses

        cell = sim.cell
        assert cell is not None
        assert sim._build_cell_results is not None

    def test_run_is_deterministic_on_repeated_calls(
        self,
        sample_swc_file,
        sample_synpts_file,
        sample_events,
        sample_parameters,
    ):
        """Two consecutive run() calls with the same inputs yield equal traces."""
        sim = self._make_simulator(
            sample_swc_file,
            sample_synpts_file,
            sample_events,
            sample_parameters,
        )
        results_a = sim.run()
        results_b = sim.run()

        np.testing.assert_array_equal(
            results_a.voltage_traces.values,
            results_b.voltage_traces.values,
        )
        np.testing.assert_array_equal(
            results_a.voltage_traces.index.values,
            results_b.voltage_traces.index.values,
        )

class TestSimulationResultsIntegration:
    """Integration tests for SimulationResults with more realistic data."""

    @pytest.fixture
    def realistic_results_dict(self):
        """Create more realistic mock data with pynapple types."""
        # Generate voltage traces as TsdFrame
        time_points = np.linspace(0, 100, 1000)
        voltage_data = np.column_stack(
            [-65.0 + np.sin(time_points * 0.1) * 5 for _ in range(5)]
        )
        mock_voltage_traces = nap.TsdFrame(
            t=time_points,
            d=voltage_data,
            time_units="ms",
            columns=[f"probe_seg_{i}" for i in range(5)],
        )

        # Events as TsGroup
        mock_input_events = nap.TsGroup(
            {
                i: nap.Ts(t=np.random.uniform(10, 90, 10), time_units="ms")
                for i in range(3)
            },
            label=[f"syn_{i}" for i in range(3)],
        )

        mock_record_points = {f"probe_seg_{i}": (float(i), 0.0, 0.0) for i in range(5)}

        class MockSegment:
            def __init__(self, tag, x):
                self.tag = tag
                self.prox = type(
                    "obj", (object,), {"x": x, "y": 0.0, "z": 0.0, "radius": 1.0}
                )()
                self.dist = type(
                    "obj", (object,), {"x": x + 1.0, "y": 0.0, "z": 0.0, "radius": 0.8}
                )()

        class MockSegmentTree:
            def __init__(self):
                self.segments = [
                    MockSegment(1, 0.0),
                    MockSegment(1, 1.0),
                    MockSegment(3, 2.0),
                    MockSegment(3, 3.0),
                    MockSegment(3, 4.0),
                ]

        return {
            "voltage_traces": mock_voltage_traces,
            "input_events": mock_input_events,
            "synapses": {},
            "gap_junctions": [],
            "record_points": mock_record_points,
            "cell": None,
            "morphology": None,
            "segment_tree": MockSegmentTree(),
            "decor": None,
            "labels": None,
            "cvp": None,
        }

    def test_integrate_multiple_tags_realistic(self, realistic_results_dict):
        """Test integration with multiple different tags."""
        results = SimulationResults(realistic_results_dict)
        tags = results.get_tags()
        assert len(tags) == 2  # tags 1, 3

        # Integrate each tag
        for tag in tags:
            tsd = results.integrate_voltages_by_tag(tag, method="average")
            assert isinstance(tsd, nap.Tsd)
            assert len(tsd) > 0

    def test_surface_area_calculation(self, realistic_results_dict):
        """Test surface area calculation."""
        results = SimulationResults(realistic_results_dict)
        areas = results._get_segment_surface_areas()
        assert len(areas) == 5
        assert all(area > 0 for area in areas.values())

    def test_round_trip_save_load(self, realistic_results_dict, temp_dir):
        """Test complete save/load cycle with realistic data."""
        results = SimulationResults(realistic_results_dict)
        save_path = temp_dir / "realistic_results.pkl"

        # Save
        results.save(save_path)

        # Load
        loaded = SimulationResults.load(save_path)

        # Compare
        assert loaded.voltage_traces.shape == results.voltage_traces.shape
        assert len(loaded.input_events) == len(results.input_events)
        # Note: segment_tree is None after loading, so we can't call get_tags()
        # Instead, compare the voltage probe labels
        assert set(loaded.voltage_traces.columns) == set(results.voltage_traces.columns)
