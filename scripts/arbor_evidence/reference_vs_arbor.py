"""Validate the loop-capable reference solver against Arbor, then measure loops.

1. Tree check: reference with loops cut vs Arbor with no gap junctions
   (identical morphology). These must agree.
2. Loop effect: reference with loops restored vs cut (truth vs tree).
3. Arbor GJ: Arbor with gap junctions at gj_weight vs the reference loops.

Run: uv run --group arbor-evidence python scripts/arbor_evidence/reference_vs_arbor.py [TS1 TS3 ...]
"""

import logging
import sys

import numpy as np

logging.disable(logging.WARNING)

from toric_spines_sim.paths import get_pointset_path, get_swc_path
from arbor_model import ArborSimulator

from toric_spines_sim.model import TSModel
from toric_spines_sim.simulation import make_icx_parameter_bank_invitro
from toric_spines_sim.utils import load_xyz_points

SPINES = sys.argv[1:] or ["TS1", "TS2", "TS3", "TS4", "TS48", "TS67", "TS76"]
T_MS, DT_MS, N_ACTIVE = 60.0, 0.025, 5


def params(gj_weight=0.0):
    b = make_icx_parameter_bank_invitro()
    b["T_ms"].value = T_MS
    b["dt_sim_ms"].value = DT_MS
    b["dt_record_ms"].value = DT_MS
    p = b.sample()
    p["hh_tags"] = []
    return p


def arbor(name, synpts, gj_weight):
    p = params(gj_weight)
    events = {f"syn_{i}": ([10.0] if i < N_ACTIVE else []) for i in range(len(synpts))}
    sim = ArborSimulator(
        get_swc_path(f"{name}_wsink_r10um.swc", units="microns"),
        get_pointset_path(f"{name}_synpts.txt", units="microns"),
        events, p, gj_weight=gj_weight,
    )
    r = sim.run()
    return r.voltage_traces.values, r.voltage_traces.columns, r.record_points, p


def reference(name, synpts, p, restore):
    m = TSModel(get_swc_path(f"{name}_wsink_r10um.swc", units="microns"), {}, {}, p,
                restore_loops=restore).build().graph
    for xyz in synpts:
        m.add_synapse(xyz, "ampa", gmax_uS=p["ampa_gmax_uS"], tau_ms=p["ampa_tau_ms"], e_mV=p["ampa_e_mV"])
    return m.run({i: [10.0] for i in range(N_ACTIVE)}, T_ms=T_MS, dt_ms=DT_MS, v0_mV=p["Vrest_mV"])


def at_probes(ref, cols, record_points, n_t):
    return np.column_stack([ref.at(record_points[c])[:n_t] for c in cols])


print(f"{'spine':6} {'tree: max|ref-arbor|':>22} {'peak arb/ref':>14} {'loops: peak cut/restored':>26} {'max|cut-loop|':>14} {'arborGJ(w=1) peak':>18}")
for name in SPINES:
    synpts = load_xyz_points(get_pointset_path(f"{name}_synpts.txt", units="microns"))
    v_arb, cols, rp, p = arbor(name, synpts, 0.0)
    n_t = v_arb.shape[0]
    cut = at_probes(reference(name, synpts, p, False), cols, rp, n_t)
    loop = at_probes(reference(name, synpts, p, True), cols, rp, n_t)
    v_gj = arbor(name, synpts, 1.0)[0]
    rest = p["Vrest_mV"]
    pk = lambda v: (v - rest).max()
    print(f"{name:6} {np.abs(cut - v_arb).max():20.3f}mV {pk(v_arb)/pk(cut):14.4f} "
          f"{pk(cut):12.2f} / {pk(loop):6.2f} mV {np.abs(cut - loop).max():12.3f}mV {pk(v_gj):16.2f}mV")
