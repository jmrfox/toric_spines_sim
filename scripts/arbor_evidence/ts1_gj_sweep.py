"""TS1: how Arbor gap junctions change the response vs no junctions vs closed loops.

Arbor arms use the frozen Arbor model (placement fixed); "closed" uses the
package's reference solver. 5 synapses fire at 10 ms, in-vitro bank.
Run: uv run --group arbor-evidence python scripts/arbor_evidence/ts1_gj_sweep.py
"""

import logging

import numpy as np

logging.disable(logging.WARNING)

from arbor_model import ArborSimulator

from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import TSSimulator, make_icx_parameter_bank_invitro
from toric_spines_sim.utils import load_xyz_points

swc = get_swc_path("TS1_wsink_r10um.swc", units="microns")
syn = get_pointset_path("TS1_synpts.txt", units="microns")
n = len(load_xyz_points(syn))
events = {f"syn_{i}": ([10.0] if i < 5 else []) for i in range(n)}
b = make_icx_parameter_bank_invitro()
b["T_ms"].value = 60
p = b.sample()
p["hh_tags"] = []
rest = p["Vrest_mV"]

closed = TSSimulator(swc, syn, events, p, solver="reference").run().voltage_traces.values
print(f"loops closed (reference): peak {(closed - rest).max():.3f} mV")
out = {}
for w in (0.0, 1e-3, 1.0):
    v = ArborSimulator(swc, syn, events, p, gj_weight=w).run().voltage_traces.values
    out[w] = v
    label = "no junctions (tree)" if w == 0 else f"Arbor gap junctions, w={w:g} uS"
    print(f"{label:34}: peak {(v - rest).max():7.3f} mV, max |V - closed| {np.abs(v - closed).max():6.3f} mV, "
          f"max step-to-step 2nd diff {np.abs(np.diff(v, 2, axis=0)).max():.3f} mV")
