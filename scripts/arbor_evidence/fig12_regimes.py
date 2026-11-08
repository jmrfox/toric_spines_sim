"""Re-run the MASCAF paper's Fig 12 setup under three loop-handling regimes.

TS1, one strong AMPA synapse at the synapse farthest from the neck, passive
membrane (default bank, as in the paper era), peak V vs geodesic distance.

  tree      : junction ends at the same point (original placement) -> inert,
              equivalent to Arbor with loops cut (gj_weight = 0)
  misplaced : Sep-2026 BFS branch mapping, gj_weight = 1 uS (pre-fix code)
  correct   : loop-capable reference solver (loops merged exactly)

Run: uv run --group arbor-evidence python scripts/arbor_evidence/fig12_regimes.py
"""

import logging
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

logging.disable(logging.WARNING)

import arbor_model.model as arbor_model_mod
from arbor_model import ArborSimulator
from arbor_model.swc_buggy_original import arbor_locations_for_swc_nodes as buggy_locations
from toric_spines_sim.geometry.sink import neck_point_from_swc_file
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import TSSimulator, make_default_parameter_bank
from toric_spines_sim.simulation.reference import LoopCableModel
from toric_spines_sim.utils import load_xyz_points

swc = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_path = get_pointset_path("TS1_synpts.txt", units="microns")
pts = np.array(load_xyz_points(synpts_path))
neck = np.array(neck_point_from_swc_file(swc))
syn_i = int(np.argmax(np.linalg.norm(pts - neck, axis=1)))
events = {f"syn_{i}": ([5.0] if i == syn_i else []) for i in range(len(pts))}


def params(gj_weight):
    b = make_default_parameter_bank()
    b["T_ms"].value = 40.0
    b["dt_sim_ms"].value = 0.025
    b["dt_record_ms"].value = 0.025
    b["discretization_um"].value = 0.25
    b["ampa_gmax_uS"].value = 0.05
    p = b.sample()
    p["hh_tags"] = []
    return p


def run(solver, gj_weight):
    if solver == "arbor":
        r = ArborSimulator(swc, synpts_path, events, params(gj_weight), gj_weight=gj_weight).run()
    else:
        r = TSSimulator(swc, synpts_path, events, params(gj_weight), solver=solver).run()
    return r.voltage_traces, r.record_points


traces = {}
traces["tree"] = run("arbor", 0.0)
traces["correct"] = run("reference", 0.0)
arbor_model_mod.arbor_locations_for_swc_nodes = buggy_locations  # pre-fix placement
traces["misplaced"] = run("arbor", 1.0)

# geodesic distance on the true (looped) graph from the synapse
p0 = params(0.0)
m = LoopCableModel.from_swc(swc, dx_um=0.25, cm_uF_per_cm2=1, g_pas_S_per_cm2=1e-3,
                            e_pas_mV=-65, rL_ohm_cm=35, restore_loops=True)
coo = m.G_axial.tocoo()
mask = coo.row != coo.col
lens = np.linalg.norm(m.node_xyz[coo.row[mask]] - m.node_xyz[coo.col[mask]], axis=1)
graph = csr_matrix((lens, (coo.row[mask], coo.col[mask])), shape=m.G_axial.shape)
dist = dijkstra(graph, indices=m.closest_node(tuple(pts[syn_i])))

vt, rp = traces["correct"]
cols = list(vt.columns)
d = np.array([dist[m.closest_node(rp[c])] for c in cols])
tags = np.array([m.node_tag[m.closest_node(rp[c])] for c in cols])
spine = tags == 3
rest = p0["Vrest_mV"]
peaks = {k: (v[0][cols].values - rest).max(axis=0) for k, v in traces.items()}

print(f"TS1 synapse {syn_i}, {spine.sum()} spine probes; peak depolarization (mV)")
print(f"{'regime':10} {'at synapse':>10} {'neck(<1um of max d)':>20} {'spine mean':>11} "
      f"{'vs correct: mean |err|':>22} {'max |err|':>9}")
neck_sel = spine & (d > np.nanmax(d[spine]) - 1.0)
for k, pk in peaks.items():
    err = pk[spine] - peaks["correct"][spine]
    print(f"{k:10} {pk[spine].max():10.2f} {pk[neck_sel].mean():20.2f} {pk[spine].mean():11.2f} "
          f"{np.abs(err).mean():22.2f} {np.abs(err).max():9.2f}")

print("\nbinned peak depolarization (mV) vs geodesic distance, spine probes")
bins = np.arange(0, np.nanmax(d[spine]) + 2, 2)
idx = np.digitize(d[spine], bins)
hdr = "dist(um) " + " ".join(f"{k:>10}" for k in peaks)
print(hdr)
for b in np.unique(idx):
    sel = idx == b
    print(f"{bins[b-1]:4.0f}-{bins[b-1]+2:<4.0f}" + " ".join(f"{peaks[k][spine][sel].mean():10.2f}" for k in peaks))
