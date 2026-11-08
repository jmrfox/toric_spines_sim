"""jaxley backend: loops closed exactly, batching, compile-once cache."""

import logging
import math
import time

import numpy as np
import pytest

pytest.importorskip("jaxley")

from toric_spines_sim.model import SynapsePopulation, TSModel  # noqa: E402
from toric_spines_sim.paths import get_pointset_path, get_swc_path  # noqa: E402
from toric_spines_sim.simulation import jaxley_backend as jb  # noqa: E402
from toric_spines_sim.simulation import make_icx_parameter_bank  # noqa: E402


def ring_spine(tmp_path, n=48):
    """Toric-spine-like ring: cut at the top, thin neck at the bottom, sink."""
    R = 2.0

    def xy(deg):
        a = math.radians(deg)
        return R * math.cos(a), R * math.sin(a)

    lines = [f"# CYCLE_BREAK reconnect {n + 1} 1"]
    for i in range(1, n + 1):
        x, y = xy(90.0 - 360.0 * (i - 1) / n)
        lines.append(f"{i} 3 {x:.6f} {y:.6f} 0 0.25 {-1 if i == 1 else i - 1}")
    x, y = xy(90.0)
    lines.append(f"{n + 1} 3 {x:.6f} {y:.6f} 0 0.25 {n}")
    bx, by = xy(-90.0)
    k = n + 2
    lines += [f"{k} 3 {bx:.6f} {by - 0.75:.6f} 0 0.12 {n // 2 + 1}",
              f"{k + 1} 3 {bx:.6f} {by - 1.5:.6f} 0 0.12 {k}",
              f"{k + 2} 5 {bx:.6f} {by - 2.5:.6f} 0 3.0 {k + 1}",
              f"{k + 3} 5 {bx:.6f} {by - 6.5:.6f} 0 3.0 {k + 2}",
              f"{k + 4} 6 {bx:.6f} {by - 10.5:.6f} 0 3.0 {k + 3}"]
    swc = tmp_path / "ring.swc"
    swc.write_text("\n".join(lines) + "\n")
    syn = tmp_path / "ring_syn.txt"
    sx, sy = xy(70.0)
    syn.write_text(f"{sx} {sy} 0\n")
    return swc, syn


def _params(**over):
    bank = make_icx_parameter_bank("in_vitro")
    bank["discretization_um"].value = 0.25
    for k, v in over.items():
        bank[k].value = v
    p = bank.sample()
    p["hh_tags"] = []
    return p


@pytest.mark.parametrize("model,tol", [("ampa", 1e-4), ("nmda", 5e-3)])
def test_ring_loop_matches_reference(tmp_path, model, tol):
    swc, synf = ring_spine(tmp_path)
    p = _params(ampa_gmax_uS=0.004, nmda_gmax_uS=0.016)
    syn = SynapsePopulation.from_file(synf, model=model, global_parameters=p).synapses
    logging.disable(logging.WARNING)
    built = TSModel(swc, syn, {}, p).build()
    logging.disable(logging.NOTSET)
    assert built.graph.n_loops == 1
    nodes = list(range(0, built.graph.n_nodes, 5))
    ev = {0: [1.0]}
    T = 30.0
    ref = built.graph.run(ev, T_ms=T, dt_ms=0.025, v0_mV=p["Vrest_mV"]).V[:, nodes]
    jv = jb.run(built.graph, ev, T_ms=T, dt_ms=0.025, v0_mV=p["Vrest_mV"], record_nodes=nodes)
    peak = (ref - p["Vrest_mV"]).max()
    assert peak > 1.0
    assert np.abs(ref - jv).max() < tol * peak


def test_batch_equals_single_and_compiles_once():
    logging.disable(logging.WARNING)
    p = _params(discretization_um=0.5)
    swc = get_swc_path("TS2_wsink_r10um.swc", units="microns")
    syn = SynapsePopulation.from_file(get_pointset_path("TS2_synpts.txt", units="microns"),
                                      model="ampa", global_parameters=p).synapses
    built = TSModel(swc, syn, {}, p).build()
    logging.disable(logging.NOTSET)
    sim = jb.get_simulation(built.graph, T_ms=15.0, dt_ms=0.025, v_rest_mV=p["Vrest_mV"],
                            record_nodes=[0, 1, 2])
    assert sim.n_loops == built.graph.n_loops == 1
    evs = [{0: [2.0]}, {1: [3.0], 2: [4.0]}, {}]
    singles = [sim.run(e) for e in evs]
    t0 = time.time()
    again = sim.run(evs[0])
    assert time.time() - t0 < 5.0  # cached and compiled: no rebuild
    batch = sim.run_batch(evs)
    for a, b in zip(singles, batch):
        np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_array_equal(singles[0], again)
    assert jb.get_simulation(built.graph, T_ms=15.0, dt_ms=0.025, v_rest_mV=p["Vrest_mV"],
                             record_nodes=[0, 1, 2]) is sim


def test_version_guard(monkeypatch):
    import jaxley

    monkeypatch.setattr(jaxley, "__version__", "0.15.0")
    with pytest.raises(RuntimeError, match="not supported"):
        jb._check_jaxley_version()
