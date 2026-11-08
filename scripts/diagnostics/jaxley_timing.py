"""Wall-clock per simulated run: jaxley (cold / warm / batched) vs reference.

TS3 (10 loops) and TS1 at dx = 0.25 µm, dt = 0.025 ms, all synapses Poisson
at 20 Hz for T ms. Run: uv run python scripts/diagnostics/jaxley_timing.py [T_ms]
"""

import logging
import sys
import time

import numpy as np

logging.disable(logging.WARNING)

from toric_spines_sim.model import SynapsePopulation, TSModel
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import jaxley_backend as jb
from toric_spines_sim.simulation import make_icx_parameter_bank

T = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
DT = 0.025
rng = np.random.default_rng(0)
for name in ("TS3", "TS1"):
    bank = make_icx_parameter_bank("in_vitro")
    bank["discretization_um"].value = 0.25
    p = bank.sample()
    p["hh_tags"] = []
    swc = get_swc_path(f"{name}_wsink_r10um.swc", units="microns")
    syn = SynapsePopulation.from_file(get_pointset_path(f"{name}_synpts.txt", units="microns"),
                                      model="ampa", global_parameters=p).synapses
    built = TSModel(swc, syn, {}, p).build()
    g = built.graph
    evs = [{i: sorted(rng.uniform(0, T, rng.poisson(20 * T / 1000)).tolist()) for i in range(len(syn))}
           for _ in range(8)]
    rec = sorted({nd for nd, _ in g.synapses})
    t0 = time.time(); g.run(evs[0], T_ms=T, dt_ms=DT, v0_mV=p["Vrest_mV"]); t_ref = time.time() - t0
    t0 = time.time()
    sim = jb.get_simulation(g, T_ms=T, dt_ms=DT, v_rest_mV=p["Vrest_mV"], record_nodes=rec)
    sim.run(evs[0]); t_cold = time.time() - t0
    t0 = time.time(); sim.run(evs[1]); t_warm = time.time() - t0
    sim.run_batch(evs[:2])  # compile the batch path
    t0 = time.time(); sim.run_batch(evs); t_batch = (time.time() - t0) / len(evs)
    print(f"{name}: {g.n_nodes} compartments, {g.n_loops} loops, T={T:g} ms | reference {t_ref:.1f}s | "
          f"jaxley cold {t_cold:.1f}s, warm {t_warm:.1f}s, batched {t_batch:.1f}s per run")
