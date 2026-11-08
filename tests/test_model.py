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
from toric_spines_sim import SynapsePoint, SynapsePopulation


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
        assert pset["temp_K"] == 296.0
        assert pset["cm_uF_per_cm2"] == 0.9
        # g_pas = cm / (Rin * C) with Sanculi medians (290 MOhm, 205 pF)
        assert pset["pas_leak_g_S_per_cm2"] == pytest.approx(1.514e-5, rel=1e-3)
        assert pset["tau_m_ms"] == pytest.approx(290 * 205 * 1e-3, rel=1e-3)
        assert pset["sink_lumped"] == 1
        assert pset["Vrest_mV"] == -66.0

    def test_invivo_biophysics(self):
        from toric_spines_sim.simulation.parameters import (
            make_icx_parameter_bank_invivo,
        )

        from toric_spines_sim.simulation.parameters import (
            make_icx_parameter_bank_invitro,
        )

        pset = make_icx_parameter_bank_invivo().sample()
        vitro = make_icx_parameter_bank_invitro().sample()
        assert pset["temp_K"] == 313.0
        assert pset["Vrest_mV"] == -67.6
        assert pset["cm_uF_per_cm2"] == vitro["cm_uF_per_cm2"]
        f = 1.7  # (313 - 296) / 10
        assert pset["pas_leak_g_S_per_cm2"] == pytest.approx(vitro["pas_leak_g_S_per_cm2"] * 1.98**f)
        assert pset["rL_ohm_cm"] == pytest.approx(100 * 0.8**f)
        assert pset["ampa_tau_ms"] == pytest.approx(2.0 / 1.4**f)
        assert pset["nmda_tau_d_ms"] == pytest.approx(50.0 / 2.0**f)

    def test_invivo_background_lowers_cell_rin(self):
        from toric_spines_sim.simulation.parameters import make_icx_parameter_bank

        none = make_icx_parameter_bank("in_vivo").sample()
        high = make_icx_parameter_bank("in_vivo", background="high").sample()
        assert high["cell_Rin_MOhm"] == pytest.approx(0.2 * none["cell_Rin_MOhm"])
        assert high["pas_leak_g_S_per_cm2"] == none["pas_leak_g_S_per_cm2"]
        with pytest.raises(ValueError):
            make_icx_parameter_bank("in_vitro", background="high")


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


class TestReconnectAndGapJunctionMapping:
    """MULTI_NECK / CYCLE_BREAK reconnect parsing."""

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

    def test_kmatrix_event_generator_api(self):
        from toric_spines_sim.kmatrix import _events_for_rates

        pset = make_default_parameter_bank().sample()
        events = _events_for_rates([0.0, 10.0], pset, "poisson", ["syn_0", "syn_1"])
        assert len(events) == 2


class TestTSModelBuild:
    """TSModel.build(): graph with loops closed, synapses, record nodes, HH spec."""

    def _build(self, name="TS1", **over):
        import logging

        from toric_spines_sim.model import TSModel
        from toric_spines_sim.paths import get_pointset_path, get_swc_path
        from toric_spines_sim.simulation import make_icx_parameter_bank

        bank = make_icx_parameter_bank("in_vitro")
        bank["discretization_um"].value = 0.5
        for k, v in over.items():
            if k != "hh_tags":
                bank[k].value = v
        p = bank.sample()
        p["hh_tags"] = list(over.get("hh_tags", []))
        swc = get_swc_path(f"{name}_wsink_r10um.swc", units="microns")
        syn = SynapsePopulation.from_file(
            get_pointset_path(f"{name}_synpts.txt", units="microns"), model="nmda", global_parameters=p
        ).synapses
        logging.disable(logging.WARNING)
        built = TSModel(swc, syn, {"a": tuple(next(iter(syn.values())).location)}, p).build()
        logging.disable(logging.NOTSET)
        return built, syn, p

    def test_loops_closed_and_synapses_in_label_order(self):
        built, syn, _ = self._build("TS1")
        assert built.graph.n_loops == 7  # MASCAF's TS1 cycle count
        assert len(built.reconnects) == 7
        assert built.synapse_labels == list(syn)
        assert len(built.graph.synapses) == len(syn)
        (_node, comps), = built.graph.synapses[:1]
        assert comps[0].kind == "exp2" and comps[0].mg_mM is not None

    def test_lumped_sink_totals(self):
        built, _, p = self._build("TS76")
        sink_um2 = sum(built.graph.area_by_tag.get(t, 0).sum() for t in (5, 6))
        cm_sink = built.membrane_overrides[5]["cm"]
        c_sink_pF = sink_um2 * 1e-8 * cm_sink * 1e6
        assert c_sink_pF == pytest.approx(p["cell_C_pF"], rel=1e-9)
        g_sink_nS = sink_um2 * 1e-8 * built.membrane_overrides[5]["g_pas"] * 1e9
        assert 1e3 / g_sink_nS == pytest.approx(p["cell_Rin_MOhm"], rel=1e-9)

    def test_hh_spec(self):
        built, _, p = self._build("TS76", hh_tags=[5], hh_scale=0.5, sink_lumped=0)
        assert built.hh.tags == (5,)
        assert built.hh.gna == pytest.approx(0.5 * p["Na_gbar_S_per_cm2"])
        built, _, _ = self._build("TS76")
        assert built.hh is None
