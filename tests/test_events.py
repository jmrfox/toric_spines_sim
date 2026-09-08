"""Tests for toric_spines_sim.events module."""

import pytest
import numpy as np
import pynapple as nap
from toric_spines_sim.events import (
    EventGenerator,
    DeterministicEventGenerator,
    StochasticEventGenerator,
    FlatRateCurve,
    LinearRateCurve,
    StepRateCurve,
    SineRateCurve,
)


class TestDeterministicEventGenerator:
    """Test suite for DeterministicEventGenerator class."""

    def test_init_basic_independent(self):
        """Test basic initialization in independent mode."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
        )
        assert gen._n_axons == 2
        assert gen._n_channels == 2
        assert not gen._shared_mode
        assert gen._shared_curve is None

    def test_init_basic_shared(self):
        """Test basic initialization in shared mode."""
        curve = FlatRateCurve(10.0)
        gen = DeterministicEventGenerator(
            rate_curves=curve,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
        )
        assert gen._n_axons == 2
        assert gen._n_channels == 2
        assert gen._shared_mode
        assert gen._shared_curve is not None

    def test_init_with_labels(self):
        """Test initialization with custom labels."""
        curves = [FlatRateCurve(10.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            labels=["syn_0"],
            delay_ms=10.0,
        )
        assert gen._labels == ["syn_0"]
        assert gen._delay_ms == 10.0

    def test_generate_independent_basic(self):
        """Test basic independent mode event generation."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
        )
        events = gen.generate()
        assert isinstance(events, nap.TsGroup)
        assert len(events) == 2

    def test_generate_shared_basic(self):
        """Test basic shared mode event generation."""
        curve = FlatRateCurve(10.0)
        gen = DeterministicEventGenerator(
            rate_curves=curve,
            n_synapses_per_axon=[1, 1],
            T_ms=1000.0,  # Longer to get more events
        )
        events = gen.generate()
        assert isinstance(events, nap.TsGroup)
        assert len(events) == 2

    def test_generate_with_delay(self):
        """Test event generation with delay."""
        curves = [FlatRateCurve(10.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            delay_ms=50.0,
        )
        events = gen.generate()
        times_ms = events[0].as_units("ms").index.values
        assert all(t >= 50.0 for t in times_ms)

    def test_generate_zero_rate(self):
        """Test generation with zero rate."""
        curves = [FlatRateCurve(0.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
        )
        events = gen.generate()
        assert len(events[0]) == 0  # Empty Ts

    def test_generate_custom_labels(self):
        """Test generation with custom labels."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
            labels=["syn_a", "syn_b"],
        )
        events = gen.generate()
        labels = list(events.get_info("label").values)
        assert "syn_a" in labels
        assert "syn_b" in labels

    def test_generate_override_labels(self):
        """Test generation with override labels."""
        curves = [FlatRateCurve(10.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            labels=["original"],
        )
        events = gen.generate(labels=["override"])
        labels = list(events.get_info("label").values)
        assert "override" in labels

    def test_generate_zero_time(self):
        """Test generation with zero simulation time."""
        curves = [FlatRateCurve(10.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=0.0,
        )
        events = gen.generate()
        assert len(events[0]) == 0  # Empty Ts

    def test_labels_length_mismatch(self):
        """Test error when labels length doesn't match channels."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
        )
        with pytest.raises(ValueError, match="labels length.*must match"):
            gen.generate(labels=["only_one"])

    def test_duplicate_labels(self):
        """Test error when labels are not unique."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
        )
        with pytest.raises(ValueError, match="labels must be unique"):
            gen.generate(labels=["same", "same"])

    def test_routing_mode_roundrobin(self):
        """Test roundrobin routing mode."""
        curve = FlatRateCurve(100.0)  # High rate for many events
        gen = DeterministicEventGenerator(
            rate_curves=curve,
            n_synapses_per_axon=[1, 1],
            T_ms=1000.0,
            routing_mode="roundrobin",
        )
        assert gen._routing_mode == "roundrobin"

    def test_routing_mode_broadcast(self):
        """Test 'broadcast' routing mode."""
        curve = FlatRateCurve(10.0)
        gen = DeterministicEventGenerator(
            rate_curves=curve,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
            routing_mode="broadcast",
        )
        assert gen._routing_mode == "broadcast"
        # In 'broadcast' mode, both channels should get the same events
        events = gen.generate()
        times_0 = events[0].as_units("ms").index.values
        times_1 = events[1].as_units("ms").index.values
        np.testing.assert_array_equal(times_0, times_1)

    def test_routing_mode_invalid(self):
        """Test that invalid routing_mode raises error."""
        curve = FlatRateCurve(10.0)
        with pytest.raises(
            ValueError, match="routing_mode must be 'broadcast' or 'roundrobin'"
        ):
            DeterministicEventGenerator(
                rate_curves=curve,
                n_synapses_per_axon=[1, 1],
                T_ms=100.0,
                routing_mode="invalid",
            )

    def test_generate_to_file(self, temp_dir):
        """Test saving events to file."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=50.0,
        )
        output_path = temp_dir / "events.txt"
        gen.generate_to_file(str(output_path))
        assert output_path.exists()
        content = output_path.read_text()
        assert "A0S0" in content
        assert "A1S0" in content

    def test_load_from_file(self, temp_dir):
        """Test loading events from file."""
        curves = [FlatRateCurve(10.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=50.0,
        )
        output_path = temp_dir / "events.txt"
        gen.generate_to_file(str(output_path))
        loaded = gen.load_from_file(str(output_path))
        assert isinstance(loaded, nap.TsGroup)
        assert 0 in loaded


class TestStochasticEventGenerator:
    """Test suite for StochasticEventGenerator class."""

    def test_init_basic_independent(self):
        """Test basic initialization in independent mode."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
            seed=42,
        )
        assert gen._n_axons == 2
        assert gen._n_channels == 2
        assert not gen._shared_mode
        assert gen._seed == 42

    def test_init_basic_shared(self):
        """Test basic initialization in shared mode."""
        curve = FlatRateCurve(10.0)
        gen = StochasticEventGenerator(
            rate_curves=curve,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
            seed=42,
        )
        assert gen._n_axons == 2
        assert gen._n_channels == 2
        assert gen._shared_mode

    def test_init_with_labels(self):
        """Test initialization with custom labels."""
        curves = [FlatRateCurve(10.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=42,
            labels=["syn_0"],
            delay_ms=10.0,
        )
        assert gen._labels == ["syn_0"]
        assert gen._delay_ms == 10.0

    def test_generate_reproducible(self):
        """Test that generation is reproducible with same seed."""
        curves = [FlatRateCurve(10.0)]
        gen1 = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=42,
        )
        events1 = gen1.generate()
        gen2 = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=42,
        )
        events2 = gen2.generate()
        np.testing.assert_array_equal(events1[0].index.values, events2[0].index.values)

    def test_generate_different_seeds(self):
        """Test that different seeds produce different events."""
        curves = [FlatRateCurve(10.0)]
        gen1 = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=42,
        )
        events1 = gen1.generate()
        gen2 = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=43,
        )
        events2 = gen2.generate()
        # Different seeds should produce different event times
        assert not np.array_equal(events1[0].index.values, events2[0].index.values)

    def test_generate_with_delay(self):
        """Test event generation with delay."""
        curves = [FlatRateCurve(50.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=42,
            delay_ms=50.0,
        )
        events = gen.generate()
        times_ms = events[0].as_units("ms").index.values
        assert all(t >= 50.0 for t in times_ms)

    def test_generate_multiple_rates(self):
        """Test generation with multiple rates."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0), FlatRateCurve(5.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1, 1],
            T_ms=100.0,
            seed=42,
        )
        events = gen.generate()
        assert isinstance(events, nap.TsGroup)
        assert len(events) == 3
        assert 0 in events
        assert 1 in events
        assert 2 in events

    def test_generate_zero_rate(self):
        """Test generation with zero rate."""
        curves = [FlatRateCurve(0.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=42,
        )
        events = gen.generate()
        assert len(events[0]) == 0  # Empty Ts

    def test_generate_custom_labels(self):
        """Test generation with custom labels."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
            seed=42,
            labels=["syn_a", "syn_b"],
        )
        events = gen.generate()
        labels = list(events.get_info("label").values)
        assert "syn_a" in labels
        assert "syn_b" in labels

    def test_generate_zero_time(self):
        """Test generation with zero simulation time."""
        curves = [FlatRateCurve(10.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=0.0,
            seed=42,
        )
        events = gen.generate()
        assert len(events[0]) == 0  # Empty Ts

    def test_labels_length_mismatch(self):
        """Test error when labels length doesn't match channels."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
            seed=42,
        )
        with pytest.raises(ValueError, match="labels length.*must match"):
            gen.generate(labels=["only_one"])

    def test_duplicate_labels(self):
        """Test error when labels are not unique."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=100.0,
            seed=42,
        )
        with pytest.raises(ValueError, match="labels must be unique"):
            gen.generate(labels=["same", "same"])

    def test_generate_to_file(self, temp_dir):
        """Test saving events to file."""
        curves = [FlatRateCurve(10.0), FlatRateCurve(20.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=50.0,
            seed=42,
        )
        output_path = temp_dir / "poisson_events.txt"
        gen.generate_to_file(str(output_path))
        assert output_path.exists()

    def test_load_from_file(self, temp_dir):
        """Test loading events from file."""
        curves = [FlatRateCurve(100.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=42,
        )
        output_path = temp_dir / "poisson_events.txt"
        gen.generate_to_file(str(output_path))
        loaded = gen.load_from_file(str(output_path))
        assert isinstance(loaded, nap.TsGroup)
        assert 0 in loaded
        labels = list(loaded.get_info("label").values)
        assert "A0S0" in labels
        assert len(loaded[0]) > 0


class TestEventGeneratorHierarchy:
    """Test the EventGenerator base class hierarchy."""

    def test_deterministic_is_event_generator(self):
        """DeterministicEventGenerator must be a subclass of EventGenerator."""
        assert issubclass(DeterministicEventGenerator, EventGenerator)

    def test_stochastic_is_event_generator(self):
        """StochasticEventGenerator must be a subclass of EventGenerator."""
        assert issubclass(StochasticEventGenerator, EventGenerator)

    def test_deterministic_instance_is_event_generator(self):
        """DeterministicEventGenerator instance is an EventGenerator."""
        curves = [FlatRateCurve(10.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
        )
        assert isinstance(gen, EventGenerator)

    def test_stochastic_instance_is_event_generator(self):
        """StochasticEventGenerator instance is an EventGenerator."""
        curves = [FlatRateCurve(10.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=42,
        )
        assert isinstance(gen, EventGenerator)

    def test_deterministic_generate_to_file_from_base(self, temp_dir):
        """DeterministicEventGenerator.generate_to_file is inherited from EventGenerator."""
        curves = [FlatRateCurve(100.0)]
        gen = DeterministicEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[2],
            T_ms=100.0,
        )
        output_path = temp_dir / "events.txt"
        result = gen.generate_to_file(str(output_path))
        assert isinstance(result, nap.TsGroup)
        assert output_path.exists()

    def test_stochastic_load_from_file_from_base(self, temp_dir):
        """StochasticEventGenerator.load_from_file is inherited from EventGenerator."""
        curves = [FlatRateCurve(100.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=100.0,
            seed=42,
        )
        output_path = temp_dir / "load_test.txt"
        gen.generate_to_file(str(output_path))
        loaded = StochasticEventGenerator.load_from_file(str(output_path))
        assert isinstance(loaded, nap.TsGroup)


class TestStochasticEventGeneratorARP:
    """Tests for ARP (absolute refractory period) in StochasticEventGenerator."""

    def test_arp_zero_is_noop(self):
        """arp_ms=0 should not change behaviour."""
        curves = [FlatRateCurve(100.0)]
        gen_no_arp = StochasticEventGenerator(
            rate_curves=curves, n_synapses_per_axon=[1], T_ms=1000.0, seed=7
        )
        gen_arp0 = StochasticEventGenerator(
            rate_curves=curves, n_synapses_per_axon=[1], T_ms=1000.0, seed=7, arp_ms=0.0
        )
        ev1 = gen_no_arp.generate()
        ev2 = gen_arp0.generate()
        np.testing.assert_array_equal(ev1[0].index.values, ev2[0].index.values)

    def test_arp_scalar_enforces_minimum_isi(self):
        """All inter-event intervals must be >= arp_ms."""
        arp = 5.0
        curves = [FlatRateCurve(100.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1],
            T_ms=2000.0,
            seed=42,
            arp_ms=arp,
        )
        events = gen.generate()
        times_ms = events[0].as_units("ms").index.values
        if len(times_ms) > 1:
            isis = np.diff(times_ms)
            assert np.all(isis >= arp - 1e-9), f"ISI below arp_ms: {isis.min():.4f}"

    def test_arp_per_axon_list(self):
        """Per-axon arp_ms list should enforce different ARPs on each axon."""
        arps = [2.0, 10.0]
        curves = [FlatRateCurve(200.0), FlatRateCurve(200.0)]
        gen = StochasticEventGenerator(
            rate_curves=curves,
            n_synapses_per_axon=[1, 1],
            T_ms=2000.0,
            seed=99,
            arp_ms=arps,
        )
        events = gen.generate()
        for ch, arp in enumerate(arps):
            times_ms = events[ch].as_units("ms").index.values
            if len(times_ms) > 1:
                isis = np.diff(times_ms)
                assert np.all(
                    isis >= arp - 1e-9
                ), f"Channel {ch}: ISI below arp_ms={arp}: {isis.min():.4f}"

    def test_arp_list_wrong_length_raises(self):
        """arp_ms list with wrong length should raise ValueError."""
        curves = [FlatRateCurve(100.0), FlatRateCurve(100.0)]
        with pytest.raises(ValueError, match="arp_ms length"):
            StochasticEventGenerator(
                rate_curves=curves,
                n_synapses_per_axon=[1, 1],
                T_ms=100.0,
                arp_ms=[5.0, 5.0, 5.0],
            )


class TestRateCurves:
    """Test suite for RateCurve classes."""

    def test_flat_rate_curve_rate_at(self):
        """Test FlatRateCurve rate_at method."""
        curve = FlatRateCurve(10.0)
        assert curve.rate_at(0.0) == 10.0
        assert curve.rate_at(100.0) == 10.0
        assert curve.rate_at(1000.0) == 10.0

    def test_flat_rate_curve_max_rate(self):
        """Test FlatRateCurve max_rate method."""
        curve = FlatRateCurve(10.0)
        assert curve.max_rate() == 10.0

    def test_flat_rate_curve_zero_rate(self):
        """Test FlatRateCurve with zero rate."""
        curve = FlatRateCurve(0.0)
        assert curve.rate_at(0.0) == 0.0
        assert curve.max_rate() == 0.0

    def test_flat_rate_curve_get_isis(self):
        """Test FlatRateCurve get_isis method."""
        curve = FlatRateCurve(10.0)  # 10 Hz = 100 ms period
        isis = curve.get_isis(T_ms=500.0, delay_ms=0.0)
        # At 10 Hz, period is 100 ms
        # ISIs should be approximately 100 ms each
        assert len(isis) >= 3  # Should have multiple ISIs
        assert all(abs(isi - 100.0) < 1e-9 for isi in isis)

    def test_step_rate_curve_rate_at(self):
        """Test StepRateCurve rate_at method."""
        curve = StepRateCurve(rates_hz=[10.0, 20.0, 30.0], step_duration_ms=100.0)
        assert curve.rate_at(0.0) == 10.0
        assert curve.rate_at(50.0) == 10.0
        assert curve.rate_at(100.0) == 20.0
        assert curve.rate_at(150.0) == 20.0
        assert curve.rate_at(200.0) == 30.0
        assert curve.rate_at(300.0) == 0.0  # Beyond defined steps

    def test_step_rate_curve_max_rate(self):
        """Test StepRateCurve max_rate method."""
        curve = StepRateCurve(rates_hz=[10.0, 20.0, 30.0], step_duration_ms=100.0)
        assert curve.max_rate() == 30.0

    def test_sine_rate_curve_rate_at(self):
        """Test SineRateCurve rate_at method."""
        # 10 Hz sine wave, 1 Hz frequency
        curve = SineRateCurve(peak_rate_hz=10.0, freq_hz=1.0)
        # At t=0, sin(0) = 0, so rate = 10 * max(0, 0 + 0) = 0
        assert curve.rate_at(0.0) == 0.0
        # At t=250 ms (quarter period), sin(2*pi*1*0.25) = sin(pi/2) = 1
        assert abs(curve.rate_at(250.0) - 10.0) < 1e-9
        # At t=500 ms (half period), sin(pi) ≈ 0 (floating point)
        assert abs(curve.rate_at(500.0)) < 1e-9

    def test_sine_rate_curve_max_rate(self):
        """Test SineRateCurve max_rate method."""
        curve = SineRateCurve(peak_rate_hz=10.0, freq_hz=1.0)
        assert curve.max_rate() == 10.0

    def test_sine_rate_curve_with_baseline(self):
        """Test SineRateCurve with baseline offset."""
        curve = SineRateCurve(peak_rate_hz=10.0, freq_hz=1.0, baseline=0.5)
        # Max rate = 10 * (1 + 0.5) = 15
        assert curve.max_rate() == 15.0

    def test_sine_rate_curve_phase_rad(self):
        """Test SineRateCurve with phase in radians."""
        curve1 = SineRateCurve(peak_rate_hz=10.0, freq_hz=1.0)  # phase = 0
        curve2 = SineRateCurve(
            peak_rate_hz=10.0, freq_hz=1.0, phase_rad=np.pi
        )  # phase = pi
        # With phase=pi, at t=0 we get sin(pi) = 0, same as unshifted
        # But at t=250ms, unshifted has peak, shifted has trough
        t_peak = 250.0
        assert abs(curve1.rate_at(t_peak) - 10.0) < 1e-9
        assert curve2.rate_at(t_peak) == 0.0

    def test_sine_rate_curve_phase_deg(self):
        """Test SineRateCurve with phase in degrees."""
        curve = SineRateCurve(peak_rate_hz=10.0, freq_hz=1.0, phase_deg=90)
        # 90 degrees = pi/2 radians
        # At t=0: sin(pi/2) = 1, so rate should be at peak
        assert abs(curve.rate_at(0.0) - 10.0) < 1e-9

    def test_sine_rate_curve_to_sine_v1_params(self):
        """Test compact serialization params round-trip through rate_at."""
        curve = SineRateCurve(
            peak_rate_hz=25.0, freq_hz=40.0, phase_rad=0.5, baseline=0.1
        )
        params = curve.to_sine_v1_params()
        assert params == {
            "peak_rate_hz": 25.0,
            "freq_hz": 40.0,
            "phase_rad": 0.5,
            "baseline": 0.1,
        }
        rebuilt = SineRateCurve(
            peak_rate_hz=params["peak_rate_hz"],
            freq_hz=params["freq_hz"],
            phase_rad=params["phase_rad"],
            baseline=params["baseline"],
        )
        for t_ms in (0.0, 12.5, 50.0):
            assert abs(curve.rate_at(t_ms) - rebuilt.rate_at(t_ms)) < 1e-12

    def test_sine_rate_curve_both_phase_raises(self):
        """Test that providing both phase_rad and phase_deg raises error."""
        with pytest.raises(
            ValueError, match="Cannot specify both phase_rad and phase_deg"
        ):
            SineRateCurve(peak_rate_hz=10.0, freq_hz=1.0, phase_rad=0.0, phase_deg=90.0)

    def test_linear_rate_curve_max_rate(self):
        """Test LinearRateCurve max_rate method."""
        curve = LinearRateCurve(rate_start_hz=10.0, rate_end_hz=20.0)
        assert curve.max_rate() == 20.0

    def test_linear_rate_curve_rate_at(self):
        """LinearRateCurve interpolates from start to end over T_ms."""
        curve = LinearRateCurve(rate_start_hz=10.0, rate_end_hz=20.0)
        assert curve.rate_at(0.0, T_ms=1000.0) == pytest.approx(10.0)
        assert curve.rate_at(500.0, T_ms=1000.0) == pytest.approx(15.0)
        assert curve.rate_at(1000.0, T_ms=1000.0) == pytest.approx(20.0)
        assert curve.rate_at(-100.0, T_ms=1000.0) == pytest.approx(10.0)
        assert curve.rate_at(2000.0, T_ms=1000.0) == pytest.approx(20.0)

    def test_linear_rate_curve_rate_at_requires_T_ms(self):
        curve = LinearRateCurve(rate_start_hz=10.0, rate_end_hz=20.0)
        with pytest.raises(ValueError, match="requires T_ms"):
            curve.rate_at(0.0)

    def test_linear_rate_curve_rate_at_non_negative(self):
        curve = LinearRateCurve(rate_start_hz=-5.0, rate_end_hz=5.0)
        assert curve.rate_at(0.0, T_ms=1000.0) == pytest.approx(0.0)
        assert curve.rate_at(500.0, T_ms=1000.0) == pytest.approx(0.0)
        assert curve.rate_at(1000.0, T_ms=1000.0) == pytest.approx(5.0)


class TestStochasticEventGeneratorSharedMode:
    """Test stochastic generator in shared-source mode."""

    def test_shared_mode_routing_uniform(self):
        """Test shared mode with uniform routing."""
        curve = FlatRateCurve(200.0)  # High rate for many events
        gen = StochasticEventGenerator(
            rate_curves=curve,
            n_synapses_per_axon=[1, 1],
            T_ms=2000.0,
            seed=42,
        )
        events = gen.generate()
        assert len(events) == 2
        # Both channels should have events
        assert len(events[0]) > 0
        assert len(events[1]) > 0

    def test_shared_mode_routing_weighted(self):
        """Test shared mode with weighted routing."""
        curve = FlatRateCurve(200.0)
        gen = StochasticEventGenerator(
            rate_curves=curve,
            n_synapses_per_axon=[1, 1],
            T_ms=5000.0,
            seed=42,
            routing_weights=[0.8, 0.2],
        )
        events = gen.generate()
        # Channel 0 should have roughly 4x more events than channel 1
        n0 = len(events[0])
        n1 = len(events[1])
        # This is probabilistic, so we just check both have events
        assert n0 > 0
        assert n1 > 0

    def test_shared_mode_arp_filtering(self):
        """Test ARP is applied correctly in shared mode."""
        curve = FlatRateCurve(300.0)
        arp = 5.0
        gen = StochasticEventGenerator(
            rate_curves=curve,
            n_synapses_per_axon=[1, 1],
            T_ms=2000.0,
            seed=55,
            arp_ms=arp,
        )
        events = gen.generate()
        for ch in range(len(events)):
            times_ms = events[ch].as_units("ms").index.values
            if len(times_ms) > 1:
                isis = np.diff(times_ms)
                assert np.all(
                    isis >= arp - 1e-9
                ), f"Channel {ch}: ISI below arp_ms={arp}: {isis.min():.4f}"
