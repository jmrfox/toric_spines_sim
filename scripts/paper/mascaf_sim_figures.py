"""MASCAF paper, simulation figures (Figs 11-12) with cable loops closed.

Replaces the Arbor versions, in which loops were not actually closed (see
``scripts/arbor_evidence``). TS1, ICx in-vitro bank (cm 0.9 µF/cm², lumped
sink with the Sanculi whole-cell C and Rin), dt = 0.025 ms, dx = 0.25 µm.

Fig 12: one AMPA synapse at the TS1 site farthest from the neck; peak
depolarization at every segment vs geodesic distance along the *looped*
cable graph. Loops closed (jaxley, checked against the reference solver)
and, for comparison, the cut tree stored in the SWC.

Fig 11: 3D snapshot of TS1 coloured by membrane potential at the time of
the spine's peak response, after two synapses fire together; the 25
synapse sites are dots, the two active ones are marked.

Run: uv run python scripts/paper/mascaf_sim_figures.py
Outputs: simulations/ts1/mascaf_paper/fig11_snapshot.png, fig12_attenuation.png,
fig12_attenuation.csv, and a summary printed to the terminal.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402

from toric_spines_sim.geometry.sink import neck_point_from_swc_file  # noqa: E402
from toric_spines_sim.paths import get_pointset_path, get_simulation_path, get_swc_path  # noqa: E402
from toric_spines_sim.simulation import TSSimulator, make_icx_parameter_bank  # noqa: E402
from toric_spines_sim.utils import load_xyz_points  # noqa: E402

logging.basicConfig(level=logging.WARNING)
OUT = get_simulation_path("ts1", "mascaf_paper")
SWC = get_swc_path("TS1_wsink_r10um.swc", units="microns")
SYNPTS = get_pointset_path("TS1_synpts.txt", units="microns")
GMAX_US = 0.005  # 5 nS, so the attenuation profile is well above numerical noise
T_MS, DT_MS, DX_UM = 40.0, 0.025, 0.25


def parameters():
    bank = make_icx_parameter_bank("in_vitro")
    bank["T_ms"].value = T_MS
    bank["dt_sim_ms"].value = DT_MS
    bank["dt_record_ms"].value = DT_MS
    bank["discretization_um"].value = DX_UM
    bank["ampa_gmax_uS"].value = GMAX_US
    p = bank.sample()
    p["hh_tags"] = []
    return p


def geodesic_from(graph, node):
    coo = graph.G_axial.tocoo()
    keep = coo.row != coo.col
    lens = np.linalg.norm(graph.node_xyz[coo.row[keep]] - graph.node_xyz[coo.col[keep]], axis=1)
    g = csr_matrix((lens, (coo.row[keep], coo.col[keep])), shape=graph.G_axial.shape)
    return dijkstra(g, indices=node)


def fig12():
    p = parameters()
    pts = np.array(load_xyz_points(SYNPTS))
    neck = np.array(neck_point_from_swc_file(SWC))
    syn = int(np.argmax(np.linalg.norm(pts - neck, axis=1)))
    events = {f"syn_{i}": ([5.0] if i == syn else []) for i in range(len(pts))}
    runs = {
        "jaxley": TSSimulator(SWC, SYNPTS, events, p, solver="jaxley").run(),
        "reference": TSSimulator(SWC, SYNPTS, events, p, solver="reference").run(),
        "tree": TSSimulator(SWC, SYNPTS, events, p, solver="reference", restore_loops=False).run(),
    }
    rest = p["Vrest_mV"]
    cols = list(runs["jaxley"].voltage_traces.columns)
    loop_sim = TSSimulator(SWC, SYNPTS, events, p, solver="reference")
    graph = loop_sim.graph
    dist = geodesic_from(graph, graph.closest_node(tuple(pts[syn])))
    rp = runs["jaxley"].record_points
    d = np.array([dist[graph.closest_node(rp[c])] for c in cols])
    seg_tags = [s.tag for s in loop_sim.segment_tree.segments]
    tags = np.array([seg_tags[int(c.split("_")[-1])] for c in cols])
    peaks = {k: r.voltage_traces[cols].values.max(axis=0) - rest for k, r in runs.items()}
    cross = float(np.abs(runs["jaxley"].voltage_traces.values - runs["reference"].voltage_traces.values).max())

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "fig12_attenuation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe", "tag", "geodesic_um", "peak_closed_mV", "peak_reference_mV", "peak_tree_mV"])
        for i, c in enumerate(cols):
            w.writerow([c, int(tags[i]), f"{d[i]:.3f}", f"{peaks['jaxley'][i]:.4f}",
                        f"{peaks['reference'][i]:.4f}", f"{peaks['tree'][i]:.4f}"])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    spine = tags == 3
    ax.scatter(d[spine], peaks["tree"][spine] + rest, s=14, facecolors="none", edgecolors="0.6",
               label="loops cut (tree)")
    ax.scatter(d[spine], peaks["jaxley"][spine] + rest, s=16, color="#1f5fbf", label="loops closed")
    ax.scatter(d[~spine], peaks["jaxley"][~spine] + rest, s=16, color="#c4501a", marker="s",
               label="sink (loops closed)")
    ax.set_xlabel("Geodesic distance from synapse (µm)")
    ax.set_ylabel("Peak membrane potential (mV)")
    ax.set_title("Voltage attenuation along TS1")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig12_attenuation.png", dpi=200)
    plt.close(fig)
    return {
        "synapse": syn,
        "peak_at_synapse_closed_mV": float(peaks["jaxley"][spine].max()),
        "peak_at_synapse_tree_mV": float(peaks["tree"][spine].max()),
        "neck_region_closed_mV": float(peaks["jaxley"][spine][d[spine] > d[spine].max() - 1].mean()),
        "jaxley_vs_reference_max_diff_mV": cross,
        "loops": graph.n_loops,
    }


def _frustum_polys(seg, sides=10):
    p = np.array([seg.prox.x, seg.prox.y, seg.prox.z])
    q = np.array([seg.dist.x, seg.dist.y, seg.dist.z])
    axis = q - p
    n = np.linalg.norm(axis)
    if n == 0:
        return []
    u = axis / n
    h = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    a = np.cross(u, h)
    a /= np.linalg.norm(a)
    b = np.cross(u, a)
    ang = np.linspace(0, 2 * math.pi, sides + 1)
    ring = lambda c, r: [c + r * (math.cos(t) * a + math.sin(t) * b) for t in ang]  # noqa: E731
    r0, r1 = ring(p, seg.prox.radius), ring(q, seg.dist.radius)
    return [[r0[i], r0[i + 1], r1[i + 1], r1[i]] for i in range(sides)]


def fig11():
    p = parameters()
    pts = np.array(load_xyz_points(SYNPTS))
    active = [0, 12]
    events = {f"syn_{i}": ([5.0] if i in active else []) for i in range(len(pts))}
    sim = TSSimulator(SWC, SYNPTS, events, p, solver="jaxley")
    res = sim.run()
    vt = res.voltage_traces
    segs = sim.segment_tree.segments
    spine_cols = [c for c in vt.columns if segs[int(c.split("_")[-1])].tag == 3]
    k = int(np.argmax(vt[spine_cols].values.max(axis=1)))
    v = vt.values[k]
    v_of = {c: v[i] for i, c in enumerate(vt.columns)}
    vmin, vmax = float(p["Vrest_mV"]), float(v.max())
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection="3d")
    for i, seg in enumerate(segs):
        if seg.tag not in (3,):
            continue  # draw the spine; the 10 µm sink would dwarf it
        val = v_of.get(f"probe_seg_{i}", vmin)
        polys = _frustum_polys(seg)
        if polys:
            ax.add_collection3d(Poly3DCollection(polys, facecolor=cmap((val - vmin) / max(vmax - vmin, 1e-9)),
                                                 edgecolor="none", alpha=0.95))
    ax.scatter(*pts.T, s=8, color="k", depthshade=False)
    ax.scatter(*pts[active].T, s=220, marker="o", facecolors="none", edgecolors="#d62728",
               depthshade=False, linewidths=2.5, label="active synapses")
    ax.legend(loc="upper left", frameon=False)
    spine_xyz = np.array([[s.dist.x, s.dist.y, s.dist.z] for s in segs if s.tag == 3])
    lo, hi = spine_xyz.min(0), spine_xyz.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Membrane potential (mV)")
    ax.set_title(f"TS1, loops closed, t = {vt.index.values[k] * 1000:.2f} ms")
    fig.tight_layout()
    fig.savefig(OUT / "fig11_snapshot.png", dpi=200)
    plt.close(fig)
    return {"snapshot_ms": float(vt.index.values[k] * 1000), "spine_peak_mV": vmax - vmin}


if __name__ == "__main__":
    s12 = fig12()
    s11 = fig11()
    print("Fig 12:", {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in s12.items()})
    print("Fig 11:", {k: round(v, 3) for k, v in s11.items()})
    print("Wrote", OUT)
