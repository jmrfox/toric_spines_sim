"""Test: an explicit Arbor gap junction acts like extra capacitance g*dt per end.

Arbor's update for the two junction compartments A, B (peer voltage from the
start of the step, own voltage implicit):
    (C/dt + g) A' = (C/dt) A + g B     (+ axial terms)
    (C/dt + g) B' = (C/dt) B + g A
Net junction current into the pair = g(A - B') + g(B - A') = -g * d(A+B)
per step, i.e. it opposes any change of the pair's voltage like a capacitor of
size g*dt on each compartment. Prediction: Arbor(gj = g) on the ring matches the
correctly closed loop with 2*g*dt extra capacitance at the cut node, and the
error shrinks in proportion to dt.

Run: uv run --group arbor-evidence python scripts/arbor_evidence/gj_fake_capacitance.py
"""

import logging
import sys
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent / "ring_animation"))
from ring_geometry import CUT_DEG, SYN_DEG, T_SYN, write_geometry, xy  # noqa: E402

from arbor_model import ArborSimulator

from toric_spines_sim.model import TSModel
from toric_spines_sim.simulation import make_icx_parameter_bank

tmp = Path(__file__).parent / "ring_animation" / "_geometry"
tmp.mkdir(exist_ok=True)
swc, syn, _ = write_geometry(tmp)
K = 48
probes = {f"r{k}": (*xy(CUT_DEG - 360.0 * (k + 0.5) / K), 0.0) for k in range(K)}


def params(w, dt):
    b = make_icx_parameter_bank("in_vitro")
    for k, v in dict(T_ms=8.0, dt_sim_ms=dt, dt_record_ms=dt, discretization_um=0.1,
                     ampa_gmax_uS=0.004).items():
        b[k].value = v
    p = b.sample()
    p["hh_tags"] = []
    return p


def arbor_peak(w, dt):
    p = params(w, dt)
    r = ArborSimulator(swc, syn, {"syn_0": [T_SYN]}, p, record_points=probes, gj_weight=w).run()
    return (r.voltage_traces.values - p["Vrest_mV"]).max()


def reference_peak(extra_pF, dt):
    p = params(0.0, dt)
    m = TSModel(swc, {}, {}, p).build().graph  # loops closed, lumped sink
    m.C_uF[m.closest_node((*xy(CUT_DEG), 0.0))] += extra_pF * 1e-6  # pF -> uF
    s = m.add_synapse((*xy(SYN_DEG), 0.0), "ampa", gmax_uS=p["ampa_gmax_uS"],
                      tau_ms=p["ampa_tau_ms"], e_mV=p["ampa_e_mV"])
    res = m.run({s: [T_SYN]}, T_ms=8.0, dt_ms=dt, v0_mV=p["Vrest_mV"])
    return max((res.at(xyz) - p["Vrest_mV"]).max() for xyz in probes.values())


print("ring peak depolarization (mV)")
print(f"{'g (uS)':>7} {'dt (ms)':>8} {'fake C = 2*g*dt':>16} {'Arbor gap junction':>19} {'closed loop + fake C':>21}")
print(f"{'-':>7} {0.025:8.4f} {'0 pF':>16} {'':>19} {reference_peak(0.0, 0.025):21.2f}   (correct)")
for dt in (0.025, 0.0125):
    for w in (0.1, 1.0, 10.0):
        fake_pF = 2 * w * 1e-6 * dt * 1e-3 * 1e12
        print(f"{w:7g} {dt:8.4f} {fake_pF:13.1f} pF {arbor_peak(w, dt):19.2f} {reference_peak(fake_pF, dt):21.2f}")
