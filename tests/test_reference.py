"""Tests for the loop-capable reference solver."""

import logging
import math

import numpy as np
import pytest

from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import TSSimulator, make_icx_parameter_bank_invitro
from toric_spines_sim.simulation.reference import LoopCableModel, exp2_peak_factor
from toric_spines_sim.utils import load_xyz_points


def _params(**over):
    bank = make_icx_parameter_bank_invitro()
    bank["T_ms"].value = 30.0
    bank["dt_sim_ms"].value = 0.025
    bank["dt_record_ms"].value = 0.025
    bank["discretization_um"].value = 0.25
    for k, v in over.items():
        bank[k].value = v
    p = bank.sample()
    p["hh_tags"] = []
    return p


def _ring_swc(path, n=24, radius_um=3.0, r=0.3):
    """Ring of n samples; the closing edge goes to a duplicate of sample 1."""
    lines = ["# CYCLE_BREAK reconnect %d 1" % (n + 1)]
    for i in range(n):
        a = 2 * math.pi * i / n
        parent = -1 if i == 0 else i
        lines.append(f"{i + 1} 3 {radius_um * math.cos(a):.6f} {radius_um * math.sin(a):.6f} 0 {r} {parent}")
    lines.append(f"{n + 1} 3 {radius_um:.6f} 0.000000 0 {r} {n}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_exp2_peak_factor_makes_unit_peak():
    tr, td = 5.0, 50.0
    t = np.linspace(0, 300, 300001)
    g = exp2_peak_factor(tr, td) * (np.exp(-t / td) - np.exp(-t / tr))
    assert g.max() == pytest.approx(1.0, rel=1e-6)


def test_ring_is_symmetric_only_when_loop_restored(tmp_path):
    swc = _ring_swc(tmp_path / "ring.swc")
    kw = dict(dx_um=0.2, cm_uF_per_cm2=0.9, g_pas_S_per_cm2=1.5e-5, e_pas_mV=-65.0, rL_ohm_cm=100.0)
    n = 24
    pts = [(3 * math.cos(2 * math.pi * i / n), 3 * math.sin(2 * math.pi * i / n), 0.0) for i in range(n)]
    out = {}
    for restore in (True, False):
        m = LoopCableModel.from_swc(swc, restore_loops=restore, **kw)
        s = m.add_synapse(pts[6], "ampa", gmax_uS=0.002, tau_ms=2.0)
        res = m.run({s: [1.0]}, T_ms=10.0, dt_ms=0.01, v0_mV=-65.0)
        left, right = res.at(pts[3]), res.at(pts[9])  # equidistant from node 6 on the ring
        out[restore] = np.abs(left - right).max() / (res.at(pts[6]).max() + 65.0)
    assert out[True] < 1e-3
    assert out[False] > 10 * out[True]


@pytest.mark.parametrize("name", ["TS76", "TS1"])
def test_tssimulator_solvers_agree(name):
    """jaxley and the reference solver agree through the public TSSimulator API."""
    pytest.importorskip("jaxley")
    logging.disable(logging.WARNING)
    p = _params(discretization_um=0.5, T_ms=20.0)
    swc = get_swc_path(f"{name}_wsink_r10um.swc", units="microns")
    synpts = get_pointset_path(f"{name}_synpts.txt", units="microns")
    n = len(load_xyz_points(synpts))
    events = {f"syn_{i}": ([5.0] if i < 3 else []) for i in range(n)}
    ref = TSSimulator(swc, synpts, events, p, solver="reference").run()
    jx = TSSimulator(swc, synpts, events, p, solver="jaxley").run()
    logging.disable(logging.NOTSET)
    va, vb = ref.voltage_traces.values, jx.voltage_traces.values
    assert va.shape == vb.shape
    peak = (va - p["Vrest_mV"]).max()
    assert peak > 0.5
    assert np.abs(va - vb).max() < 1e-3 * peak


def test_invalid_solver():
    p = _params()
    swc = get_swc_path("TS76_wsink_r10um.swc", units="microns")
    synpts = get_pointset_path("TS76_synpts.txt", units="microns")
    with pytest.raises(ValueError):
        TSSimulator(swc, synpts, {}, p, solver="arbor")


def test_auto_solver_selection():
    p = _params()
    swc = get_swc_path("TS76_wsink_r10um.swc", units="microns")
    synpts = get_pointset_path("TS76_synpts.txt", units="microns")
    assert TSSimulator(swc, synpts, {}, p).solver == "reference"
    p["hh_tags"] = [5]
    assert TSSimulator(swc, synpts, {}, p).solver == "jaxley"
    assert TSSimulator(swc, synpts, {}, p, solver="reference").solver == "reference"


def test_reference_refuses_hh():
    p = _params()
    p["hh_tags"] = [5]
    swc = get_swc_path("TS76_wsink_r10um.swc", units="microns")
    synpts = get_pointset_path("TS76_synpts.txt", units="microns")
    with pytest.raises(ValueError, match="passive-only"):
        TSSimulator(swc, synpts, {}, p, solver="reference").run()


@pytest.mark.parametrize("condition,bg", [("in_vitro", "none"), ("in_vivo", "moderate")])
def test_lumped_sink_reproduces_cell_rin_and_tau(condition, bg):
    """A current step into the lumped sink recovers the bank's Rin and C*Rin."""
    from toric_spines_sim.geometry.swc import iter_swc_samples
    from toric_spines_sim.model import TSModel
    from toric_spines_sim.simulation import make_icx_parameter_bank

    logging.disable(logging.WARNING)
    bank = make_icx_parameter_bank(condition, background=bg)
    bank["discretization_um"].value = 1.0
    p = bank.sample()
    p["hh_tags"] = []
    swc = get_swc_path("TS76_wsink_r10um.swc", units="microns")
    sink_xyz = [s[2:5] for s in iter_swc_samples(swc) if s[1] == 6][-1]
    m = TSModel(swc, {}, {}, p).build().graph
    amp_nA, t0, T, dt = -0.01, 5.0, 600.0, 0.1
    v = m.run({}, T_ms=T, dt_ms=dt, v0_mV=p["Vrest_mV"], iclamps=[(sink_xyz, t0, T, amp_nA)]).at(sink_xyz)
    logging.disable(logging.NOTSET)
    e0 = p["Vrest_mV"]
    t = np.arange(len(v)) * dt
    dv_inf = v[-1] - e0
    assert dv_inf / amp_nA == pytest.approx(p["cell_Rin_MOhm"], rel=0.05)
    k = np.argmax((v - e0) / dv_inf >= 1 - np.exp(-1))
    assert t[k] - t0 == pytest.approx(p["cell_Rin_MOhm"] * p["cell_C_pF"] * 1e-3, rel=0.08)
