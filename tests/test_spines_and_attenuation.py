"""Spine registry, scenarios, CLI entry points, and loop attenuation."""

import logging

import numpy as np
import pytest

from simulations.common.scenarios import SCENARIOS
from simulations.common.spines import ALL_SPINES, AXON_SPINES, model_config, normalize


def test_registry_normalizes_and_rejects():
    assert normalize("ts1") == normalize("1") == "TS1"
    with pytest.raises(ValueError):
        normalize("TS999")
    with pytest.raises(ValueError, match="axon map"):
        model_config("TS21")  # no synapse file
    assert set(AXON_SPINES) < set(ALL_SPINES)


@pytest.mark.parametrize("spine", AXON_SPINES)
def test_every_scenario_builds_for_every_spine(spine):
    cfg = model_config(spine)
    for name, build in SCENARIOS.items():
        sc = build(cfg)
        assert len(sc.axon_synapses) == cfg.n_axons(), name


def test_custom_parameter_bank_factory():
    from toric_spines_sim.simulation import make_icx_parameter_bank

    cfg = model_config("TS2", make_parameter_bank=lambda: make_icx_parameter_bank("in_vitro"))
    assert cfg.make_parameter_bank().sample()["temp_K"] == 296.0


@pytest.mark.parametrize("module", ["simulations.axons", "simulations.axons_dash",
                                    "simulations.axons_animation", "simulations.loop_attenuation"])
def test_entry_points_parse_help(module, capsys):
    import importlib

    mod = importlib.import_module(module)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out


@pytest.fixture(scope="module")
def ts1_attenuation():
    from toric_spines_sim.attenuation import loop_attenuation

    logging.disable(logging.WARNING)
    res = loop_attenuation("TS1", dx_um=0.5)
    logging.disable(logging.NOTSET)
    return res


def test_ts1_loops_lower_the_synaptic_peak(ts1_attenuation):
    s = ts1_attenuation.summary
    assert s["loops"] == 7 and s["stimulus"] == "synapse 0"
    # cutting TS1's loops overestimates the synaptic peak substantially (~80% at dx 0.25)
    assert s["cut_overestimate_at_synapse"] > 0.5
    t = ts1_attenuation.table
    assert len(t) == 93 and (t["geodesic_um"] >= 0).all()
    # peak falls off with distance from the synapse in the closed arm
    near = t[t.geodesic_um < 1.0].peak_closed_mV.mean()
    far = t[t.geodesic_um > 8.0].peak_closed_mV.mean()
    assert near > 2 * far


def test_loop_free_spine_arms_match_and_no_synpts_spine_runs():
    from toric_spines_sim.attenuation import loop_attenuation

    logging.disable(logging.WARNING)
    tree = loop_attenuation("TS76", dx_um=0.5, T_ms=20.0)  # no loops: closed == cut
    nosyn = loop_attenuation("TS21", dx_um=0.5, T_ms=20.0)  # no synapse file
    logging.disable(logging.NOTSET)
    assert tree.summary["loops"] == 0
    np.testing.assert_allclose(tree.table.peak_closed_mV, tree.table.peak_cut_mV)
    assert nosyn.summary["stimulus"] == "farthest spine point"
    assert nosyn.summary["peak_synapse_closed_mV"] > 0


def test_plot_and_save(ts1_attenuation, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from toric_spines_sim.attenuation import plot_loop_attenuation

    for style in ("paper", "poster"):
        ax = plot_loop_attenuation(ts1_attenuation, style=style)
        assert len(ax.collections) == 2  # cut and closed scatters
        assert any("sink" in t.get_text() for t in ax.get_legend().get_texts())
        spine = ts1_attenuation.table.query("tag == 3").geodesic_um
        assert ax.get_xlim()[1] < 1.1 * spine.max()  # x range limited to the spine
    table, summary = ts1_attenuation.save(tmp_path)
    assert table.exists() and summary.exists()
