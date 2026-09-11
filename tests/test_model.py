"""Tests for toric_spines_sim.model module."""

import pytest
import numpy as np
from pathlib import Path
from jscip import IndependentScalarParameter, ParameterBank
from toric_spines_sim.geometry.swc import (
    parse_cycle_breaks,
    read_swc_points,
    get_center_coordinates_for_all_segments,
)
from toric_spines_sim.utils import equal_vectors
from toric_spines_sim.simulation.parameters import make_default_parameter_bank
from toric_spines_sim import (
    SynapsePoint,
    SynapsePopulation,
    GapJunctionPoint,
    prepare_gap_junctions,
)


class TestParseCycleBreaks:
    """Test suite for parse_cycle_breaks function."""

    def test_parse_basic(self, sample_swc_file):
        """Test parsing basic cycle break annotation."""
        pairs = parse_cycle_breaks(sample_swc_file)
        assert len(pairs) == 1
        assert pairs[0] == (3, 4)

    def test_parse_no_cycle_breaks(self, temp_dir):
        """Test parsing SWC with no cycle breaks."""
        swc_content = """# No cycle breaks here
1 1 0.0 0.0 0.0 1.0 -1
2 1 1.0 0.0 0.0 0.8 1
"""
        swc_path = temp_dir / "no_breaks.swc"
        swc_path.write_text(swc_content)
        pairs = parse_cycle_breaks(swc_path)
        assert pairs == []

    def test_parse_multiple_breaks(self, temp_dir):
        """Test parsing multiple cycle breaks."""
        swc_content = """# CYCLE_BREAK reconnect 3 4
# CYCLE_BREAK reconnect 5 6
1 1 0.0 0.0 0.0 1.0 -1
"""
        swc_path = temp_dir / "multi_breaks.swc"
        swc_path.write_text(swc_content)
        pairs = parse_cycle_breaks(swc_path)
        assert len(pairs) == 2
        assert (3, 4) in pairs
        assert (5, 6) in pairs

    def test_parse_case_insensitive(self, temp_dir):
        """Test that parsing is case insensitive."""
        swc_content = """# cycle_break reconnect 3 4
# CYCLE_BREAK RECONNECT 5 6
1 1 0.0 0.0 0.0 1.0 -1
"""
        swc_path = temp_dir / "case_test.swc"
        swc_path.write_text(swc_content)
        pairs = parse_cycle_breaks(swc_path)
        assert len(pairs) == 2

    def test_parse_invalid_format(self, temp_dir):
        """Test error on invalid cycle break format."""
        swc_content = """# CYCLE_BREAK reconnect invalid data
1 1 0.0 0.0 0.0 1.0 -1
"""
        swc_path = temp_dir / "invalid.swc"
        swc_path.write_text(swc_content)
        with pytest.raises(ValueError):
            parse_cycle_breaks(swc_path)


class TestReadSwcPoints:
    """Test suite for read_swc_points function."""

    def test_read_basic(self, sample_swc_file):
        """Test reading basic SWC file."""
        points = read_swc_points(sample_swc_file)
        assert len(points) == 5
        assert 1 in points
        assert points[1] == (0.0, 0.0, 0.0, 1.0)

    def test_read_coordinates(self, sample_swc_file):
        """Test that coordinates are read correctly."""
        points = read_swc_points(sample_swc_file)
        assert points[2] == (1.0, 0.0, 0.0, 0.8)
        assert points[3] == (2.0, 0.0, 0.0, 0.6)

    def test_read_ignores_comments(self, temp_dir):
        """Test that comments are ignored."""
        swc_content = """# This is a comment
# Another comment
1 1 0.0 0.0 0.0 1.0 -1
# More comments
2 1 1.0 0.0 0.0 0.8 1
"""
        swc_path = temp_dir / "comments.swc"
        swc_path.write_text(swc_content)
        points = read_swc_points(swc_path)
        assert len(points) == 2

    def test_read_invalid_line(self, temp_dir):
        """Test error on invalid SWC line."""
        swc_content = """1 1 0.0 0.0 0.0 1.0 -1
invalid line
"""
        swc_path = temp_dir / "invalid.swc"
        swc_path.write_text(swc_content)
        with pytest.raises(ValueError):
            read_swc_points(swc_path)


class TestCenterCoordinatesForAllSegments:
    """Test suite for center_coordinates_for_all_segments function."""

    def test_basic_functionality(self, sample_swc_file):
        """Test basic coordinate generation."""
        coords = get_center_coordinates_for_all_segments(sample_swc_file)
        assert isinstance(coords, dict)
        assert len(coords) > 0
        assert all(isinstance(k, str) for k in coords.keys())
        assert all(len(v) == 3 for v in coords.values())

    def test_probe_naming(self, sample_swc_file):
        """Test that probes are named correctly."""
        coords = get_center_coordinates_for_all_segments(sample_swc_file)
        assert "probe_seg_0" in coords

    def test_with_radius_weighting(self, sample_swc_file):
        """Test with radius weighting enabled."""
        coords = get_center_coordinates_for_all_segments(
            sample_swc_file, use_radius_weighting=True
        )
        assert len(coords) > 0


class TestEqualVectors:
    """Test suite for equal_vectors function."""

    def test_equal_vectors(self):
        """Test that equal vectors are detected."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]
        assert equal_vectors(v1, v2)

    def test_unequal_vectors(self):
        """Test that unequal vectors are detected."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 4.0]
        assert not equal_vectors(v1, v2)

    def test_within_tolerance(self):
        """Test vectors within tolerance."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0000001, 2.0, 3.0]
        assert equal_vectors(v1, v2, tol=1e-5)

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert equal_vectors(v1, v2)


class TestMakeDefaultParameterBank:
    """Test suite for make_default_parameter_bank function."""

    def test_creates_parameter_bank(self):
        """Test that parameter bank is created."""
        parameter_bank = make_default_parameter_bank()
        assert isinstance(parameter_bank, ParameterBank)

    def test_contains_required_parameters(self):
        """Test that required parameters are present."""
        parameter_bank = make_default_parameter_bank()
        required_params = [
            "seed",
            "T_ms",
            "delay_ms",
            "dt_sim_ms",
            "dt_record_ms",
            "cm_uF_per_cm2",
            "rL_ohm_cm",
            "ampa_gmax_uS",
            "ampa_tau_ms",
            "gabaa_gmax_uS",
            "effexc_gmax_uS",
        ]
        for param in required_params:
            assert param in parameter_bank.parameters

    def test_default_values(self):
        """Test that default values are reasonable after sampling."""
        parameter_bank = make_default_parameter_bank()
        pset = parameter_bank.sample()
        assert pset["T_ms"] == 1000.0
        assert pset["dt_sim_ms"] == 0.02
        assert pset["ampa_gmax_uS"] == 0.002

    def test_tau_m_ms_derived_after_sample(self):
        """Test that derived tau_m_ms is computed by sample()."""
        from toric_spines_sim.simulation.parameters import get_tau_m

        parameter_bank = make_default_parameter_bank()
        pset = parameter_bank.sample()
        assert pset["tau_m_ms"] == get_tau_m(pset)


class TestIcxParameterBanks:
    """Sampled values for the in-vitro and in-vivo ICx banks."""

    def test_invitro_biophysics(self):
        from toric_spines_sim.simulation.parameters import (
            make_icx_parameter_bank_invitro,
        )

        pset = make_icx_parameter_bank_invitro().sample()
        assert pset["temp_K"] == 297.0
        assert pset["pas_leak_g_S_per_cm2"] == 0.000144
        assert pset["Vrest_mV"] == -67.6

    def test_invivo_biophysics(self):
        from toric_spines_sim.simulation.parameters import (
            make_icx_parameter_bank_invivo,
        )

        pset = make_icx_parameter_bank_invivo().sample()
        assert pset["temp_K"] == 313.0
        assert pset["pas_leak_g_S_per_cm2"] == 0.00042
        assert pset["Vrest_mV"] == -67.6


class TestSynapsePoint:
    """Test suite for SynapsePoint dataclass."""

    def test_dataclass_fields(self):
        syn = SynapsePoint(
            location=(1.0, 2.0, 3.0),
            model="ampa",
            mechanism="ampasyn",
            synapse_params={"gmax_uS": 0.002, "tau_ms": 2.0, "e_mV": 0.0},
            mechanism_params={"gmax": 0.002, "tau": 2.0, "e": 0.0},
        )
        assert syn.location == (1.0, 2.0, 3.0)
        assert syn.model == "ampa"
        assert syn.mechanism == "ampasyn"
        assert syn.synapse_params["gmax_uS"] == 0.002
        assert syn.mechanism_params["tau"] == 2.0


class TestGapJunctionPoint:
    """Test suite for GapJunctionPoint dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        gj = GapJunctionPoint(
            index_pair=(1, 2), location=(3.0, 4.0, 5.0), weight=1.0
        )
        assert gj.index_pair == (1, 2)
        assert gj.location == (3.0, 4.0, 5.0)
        assert gj.weight == 1.0

    def test_init_with_weight(self):
        """Test initialization with custom weight."""
        gj = GapJunctionPoint(index_pair=(1, 2), location=(3.0, 4.0, 5.0), weight=2.5)
        assert gj.weight == 2.5


class TestSynapsePopulation:
    """Test suite for SynapsePopulation."""

    def test_from_file_ampa(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        population = SynapsePopulation.from_file(
            sample_synpts_file, model="ampa", global_parameters=pset
        )
        assert len(population.synapses) == 3
        assert population.synapses["syn_0"].model == "ampa"
        assert population.synapses["syn_0"].mechanism == "ampasyn"
        assert population.synapses["syn_0"].mechanism_params == {
            "gmax": 0.002,
            "tau": 2.0,
            "e": 0.0,
        }

    def test_from_file_ampa_dict(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        synapses = SynapsePopulation.from_file(
            sample_synpts_file, model="ampa", global_parameters=pset
        ).synapses
        assert len(synapses) == 3
        assert synapses["syn_0"].mechanism == "ampasyn"

    def test_shared_params_without_sampling(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        population = SynapsePopulation.from_file(
            sample_synpts_file, model="ampa", global_parameters=pset
        )
        gmax_values = {
            syn.synapse_params["gmax_uS"] for syn in population.synapses.values()
        }
        assert len(gmax_values) == 1

    def test_per_synapse_sampling_with_override(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        override = ParameterBank(
            {
                "ampa_gmax_uS": IndependentScalarParameter(
                    0.01, is_sampled=True, range=(0.005, 0.02)
                ),
            }
        )
        population = SynapsePopulation.from_file(
            sample_synpts_file,
            model="ampa",
            global_parameters=pset,
            parameter_override=override,
        )
        gmax_values = [
            syn.synapse_params["gmax_uS"] for syn in population.synapses.values()
        ]
        assert len(set(gmax_values)) > 1

    def test_nmda_mechanism_mapping(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        population = SynapsePopulation.from_file(
            sample_synpts_file, model="nmda", global_parameters=pset
        )
        syn = population.synapses["syn_0"]
        assert syn.mechanism == "nmdasyn"
        assert syn.mechanism_params["tau_r"] == 5.0
        assert syn.mechanism_params["tau_d"] == 50.0

    def test_gabaa_mechanism_mapping(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        population = SynapsePopulation.from_file(
            sample_synpts_file, model="gabaa", global_parameters=pset
        )
        syn = population.synapses["syn_0"]
        assert syn.mechanism == "gabaasyn"
        assert syn.mechanism_params["tau"] == 10.0
        assert syn.mechanism_params["e"] == -75.0

    def test_effexc_mechanism_mapping(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        population = SynapsePopulation.from_file(
            sample_synpts_file, model="effexc", global_parameters=pset
        )
        syn = population.synapses["syn_0"]
        assert syn.mechanism == "effexcsyn"
        assert syn.mechanism_params["nmda_ratio"] == 0.5
        assert syn.mechanism_params["mg"] == 1.0

    def test_invalid_model_raises(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        with pytest.raises(ValueError, match="Invalid model type"):
            SynapsePopulation.from_file(
                sample_synpts_file, model="invalid", global_parameters=pset
            )

    def test_merge_populations(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        ampa = SynapsePopulation.from_file(
            sample_synpts_file,
            model="ampa",
            global_parameters=pset,
            label_prefix="ampa",
        )
        nmda = SynapsePopulation.from_file(
            sample_synpts_file,
            model="nmda",
            global_parameters=pset,
            label_prefix="nmda",
        )
        merged = SynapsePopulation.merge(ampa, nmda)
        assert len(merged) == 6
        assert "ampa_0" in merged
        assert "nmda_0" in merged

    def test_prepare_locations(self, sample_synpts_file):
        pset = make_default_parameter_bank().sample()
        synapses = SynapsePopulation.from_file(
            sample_synpts_file, model="ampa", global_parameters=pset
        ).synapses
        assert synapses["syn_0"].location == (1.5, 0.0, 0.0)


class TestPrepareGapJunctions:
    """Test suite for prepare_gap_junctions function."""

    def test_prepare_basic(self, sample_swc_file):
        """Test basic gap junction preparation."""
        pset = make_default_parameter_bank().sample()
        gap_junctions = prepare_gap_junctions(sample_swc_file, parameters=pset)
        assert len(gap_junctions) == 1
        assert "gj_0" in gap_junctions

    def test_prepare_with_custom_weight(self, sample_swc_file):
        """Test preparation with custom weight."""
        parameter_bank = make_default_parameter_bank()
        parameter_bank["gj_weight"].value = 2.0
        pset = parameter_bank.sample()
        gap_junctions = prepare_gap_junctions(sample_swc_file, parameters=pset)
        assert gap_junctions["gj_0"].weight == 2.0

    def test_prepare_with_parameters(self, sample_swc_file):
        """Test preparation with sampled parameters."""
        parameter_bank = make_default_parameter_bank()
        parameter_bank["gj_weight"].value = 1.5
        pset = parameter_bank.sample()
        gap_junctions = prepare_gap_junctions(sample_swc_file, parameters=pset)
        assert gap_junctions["gj_0"].weight == 1.5

    def test_prepare_index_pairs(self, sample_swc_file):
        """Test that index pairs are correct."""
        pset = make_default_parameter_bank().sample()
        gap_junctions = prepare_gap_junctions(sample_swc_file, parameters=pset)
        assert gap_junctions["gj_0"].index_pair == (3, 4)

    def test_prepare_no_cycle_breaks(self, temp_dir):
        """Test with SWC file with no cycle breaks."""
        swc_content = """# No cycle breaks
1 1 0.0 0.0 0.0 1.0 -1
2 1 1.0 0.0 0.0 0.8 1
"""
        swc_path = temp_dir / "no_breaks.swc"
        swc_path.write_text(swc_content)
        pset = make_default_parameter_bank().sample()
        gap_junctions = prepare_gap_junctions(swc_path, parameters=pset)
        assert len(gap_junctions) == 0


class TestReconnectAndGapJunctionMapping:
    """MULTI_NECK parsing and distinct Arbor locations for colocated samples."""

    def test_parse_multi_neck_and_cycle_together(self, temp_dir):
        from toric_spines_sim.geometry.swc import (
            parse_multi_neck_reconnects,
            parse_reconnect_pairs,
        )

        swc_content = """# CYCLE_BREAK reconnect 3 4
# MULTI_NECK reconnect 1 4
1 1 0.0 0.0 0.0 1.0 -1
2 1 1.0 0.0 0.0 0.8 1
3 1 2.0 0.0 0.0 0.6 2
4 1 2.0 0.0 0.0 0.6 2
"""
        swc_path = temp_dir / "multi.swc"
        swc_path.write_text(swc_content)
        assert parse_multi_neck_reconnects(swc_path) == [(1, 4)]
        pairs = parse_reconnect_pairs(swc_path)
        assert (3, 4) in pairs
        assert (1, 4) in pairs

    def test_prepare_gap_junctions_includes_multi_neck(self, temp_dir):
        swc_content = """# CYCLE_BREAK reconnect 3 4
# MULTI_NECK reconnect 1 4
1 1 0.0 0.0 0.0 1.0 -1
2 1 1.0 0.0 0.0 0.8 1
3 1 2.0 0.0 0.0 0.6 2
4 1 2.0 0.0 0.0 0.6 2
"""
        swc_path = temp_dir / "multi.swc"
        swc_path.write_text(swc_content)
        pset = make_default_parameter_bank().sample()
        gap_junctions = prepare_gap_junctions(swc_path, parameters=pset)
        assert len(gap_junctions) == 2
        index_pairs = {gj.index_pair for gj in gap_junctions.values()}
        assert (3, 4) in index_pairs
        assert (1, 4) in index_pairs

    def test_colocated_cycle_break_maps_to_distinct_locations(self, sample_swc_file):
        import arbor as A
        from toric_spines_sim.geometry.swc import arbor_locations_for_swc_nodes

        loaded = A.load_swc_arbor(str(sample_swc_file))
        segment_tree = loaded.segment_tree
        morphology = A.morphology(segment_tree)
        locs = arbor_locations_for_swc_nodes(
            sample_swc_file, morphology, segment_tree, [3, 4]
        )
        a, b = locs[3], locs[4]
        assert (a.branch, round(a.pos, 6)) != (b.branch, round(b.pos, 6))

    def test_kmatrix_event_generator_api(self):
        from toric_spines_sim.kmatrix import _events_for_rates

        pset = make_default_parameter_bank().sample()
        events = _events_for_rates(
            [0.0, 10.0],
            pset,
            "poisson",
            ["syn_0", "syn_1"],
        )
        assert len(events) == 2

    def test_missing_catalogue_error_mentions_rebuild(self, monkeypatch, sample_swc_file, tmp_path):
        from pathlib import Path
        from toric_spines_sim.model import TSModel
        from toric_spines_sim.model import model as model_mod

        monkeypatch.setattr(
            model_mod, "CUSTOM_CATALOGUE_PATH", tmp_path / "missing-catalogue.so"
        )
        pset = make_default_parameter_bank().sample()
        tsm = TSModel(
            swc_path=sample_swc_file,
            synapses={},
            gap_junctions={},
            record_points={},
            parameters=pset,
        )
        with pytest.raises(FileNotFoundError, match="make_custom_catalogue"):
            tsm.build_cell()


class TestCheckCatalogue:
    """Tests for check_catalogue."""

    def test_required_mechanisms_match_mod_stems(self):
        from toric_spines_sim.model.model import required_catalogue_mechanisms

        names = required_catalogue_mechanisms()
        assert names == [
            "ampasyn",
            "effexcsyn",
            "gabaasyn",
            "gababsyn",
            "hhnotemp",
            "nmdasyn",
        ]

    def test_missing_file(self, tmp_path):
        from toric_spines_sim.model import check_catalogue

        missing = tmp_path / "missing-catalogue.so"
        with pytest.raises(FileNotFoundError, match="make_custom_catalogue"):
            check_catalogue(path=missing)

    def test_load_failure(self, monkeypatch, tmp_path):
        from toric_spines_sim.model import check_catalogue
        from toric_spines_sim.model import model as model_mod

        fake_so = tmp_path / "broken-catalogue.so"
        fake_so.write_bytes(b"not a shared library")

        def boom(_path):
            raise OSError("incompatible catalogue")

        monkeypatch.setattr(model_mod.A, "load_catalogue", boom)
        with pytest.raises(RuntimeError, match="Could not load NMODL catalogue"):
            check_catalogue(path=fake_so)

    def test_missing_mechanism(self, monkeypatch, tmp_path):
        from toric_spines_sim.model import check_catalogue
        from toric_spines_sim.model import model as model_mod

        fake_so = tmp_path / "incomplete-catalogue.so"
        fake_so.write_bytes(b"placeholder")

        class EmptyCatalogue:
            def __contains__(self, name):
                return False

            def __getitem__(self, name):
                raise KeyError(name)

        monkeypatch.setattr(
            model_mod.A, "load_catalogue", lambda _path: EmptyCatalogue()
        )
        with pytest.raises(RuntimeError, match="missing mechanisms"):
            check_catalogue(path=fake_so)

