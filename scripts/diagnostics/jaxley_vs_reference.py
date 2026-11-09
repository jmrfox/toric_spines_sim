"""Prototype check: jaxley (loops closed via extra compartment edges) vs reference.

Run: uv run python scripts/diagnostics/jaxley_vs_reference.py [ring|TS1|...] [--nmda]
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)

from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.model import TSModel
from toric_spines_sim.simulation import make_icx_parameter_bank
from toric_spines_sim.simulation import jaxley_backend as jb
from toric_spines_sim.utils import load_xyz_points


def setup(name, dx, kind="ampa"):
    """Built graph with synapses for a spine (or the toric ring from the evidence scripts)."""
    if name == "ring":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ring_geometry import write_geometry  # no Arbor needed

        tmp = Path(__file__).parent / "_ring_geometry"
        tmp.mkdir(exist_ok=True)
        swc, synf, _ = write_geometry(tmp)
        gmax, n_active = 0.004, 1
    else:
        swc = get_swc_path(f"{name}_wsink_r10um.swc", units="microns")
        synf = get_pointset_path(f"{name}_synpts.txt", units="microns")
        gmax, n_active = 0.002, 5
    b = make_icx_parameter_bank("in_vitro")
    b["discretization_um"].value = dx
    p = b.sample()
    p["hh_tags"] = []
    m = TSModel(swc, {}, {}, p).build().graph
    pts = load_xyz_points(synf)
    for xyz in pts:
        if kind == "ampa":
            m.add_synapse(xyz, "ampa", gmax_uS=gmax, tau_ms=2.0)
        else:
            m.add_synapse(xyz, "nmda", gmax_uS=gmax * 4, tau_r_ms=5.0, tau_d_ms=50.0, mg_mM=1.3)
    events = {i: [1.0 + 0.5 * i] for i in range(min(n_active, len(pts)))}
    return m, events, p


def compare(name, dx=0.25, dts=(0.025, 0.0125), kind="ampa", T=20.0):
    m, events, p = setup(name, dx, kind)
    rec = sorted({node for node, _ in m.synapses} | {0, m.n_nodes // 2, m.n_nodes - 1})
    print(f"\n{name}: {m.n_nodes} compartments, kind={kind}")
    for dt in dts:
        ref = m.run(events, T_ms=T, dt_ms=dt, v0_mV=p["Vrest_mV"]).V[:, rec]
        t0 = time.time()
        jx_v = jb.run(m, events, T_ms=T, dt_ms=dt, v0_mV=p["Vrest_mV"], record_nodes=rec)
        el = time.time() - t0
        n = min(len(ref), len(jx_v))
        ref, jx_v = ref[:n], jx_v[:n]
        peak = (ref - p["Vrest_mV"]).max()
        err = np.abs(ref - jx_v).max()
        print(f"  dt={dt}: peak ref {peak:.4f} mV, jaxley {(jx_v - p['Vrest_mV']).max():.4f} mV, "
              f"max |diff| {err:.5f} mV ({err / peak * 100:.3f}% of peak), wall {el:.1f}s")


if __name__ == "__main__":
    args = sys.argv[1:] or ["ring", "TS1"]
    kind = "nmda" if "--nmda" in args else "ampa"
    for name in [a for a in args if not a.startswith("--")]:
        compare(name, kind=kind, T=60.0 if kind == "nmda" else 20.0)
