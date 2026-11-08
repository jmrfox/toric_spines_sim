# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 14 — `TSModel` and the two solvers
#
# `TSModel` describes what to simulate (SWC, synapses, record points,
# parameters). `TSModel.build()` returns a `BuiltModel`:
#
# - `segments` / `morphology`: the SWC frusta (sink / neck radius scaling
#   applied) and their branches, for plotting and tag-based analysis;
# - `graph`: the cable graph with every `# CYCLE_BREAK` / `# MULTI_NECK` pair
#   merged back, so toric-spine loops are real loops; synapses are placed in
#   label order;
# - `record_nodes`, `reconnects`, `membrane_overrides` (lumped sink) and `hh`.
#
# The same graph runs in either solver: the scipy **reference** solver
# (passive, fast on CPU) or **jaxley** (adds HH, differentiable). `run()` with
# packaging into `SimulationResults` is `tssimulator`; this notebook drives the
# graph directly. For mixed receptor types (`synapses_advanced`), `TSModel` is
# the direct path.

# %%
import matplotlib.pyplot as plt
import numpy as np

from toric_spines_sim.geometry import sink_endpoint_location_from_swc_file
from toric_spines_sim.model import SynapsePopulation, TSModel
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import make_icx_parameter_bank

swc_path = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_path = get_pointset_path("TS1_synpts.txt", units="microns")
sink_xyz = sink_endpoint_location_from_swc_file(swc_path)

bank = make_icx_parameter_bank("in_vitro")
bank["T_ms"].value = 60.0
bank["discretization_um"].value = 0.25
parameters = bank.sample()
parameters["hh_tags"] = []

# %% [markdown]
# ## Mixed synapses: AMPA and NMDA at every site

# %%
ampa = SynapsePopulation.from_file(synpts_path, "ampa", parameters, label_prefix="ampa")
nmda = SynapsePopulation.from_file(synpts_path, "nmda", parameters, label_prefix="nmda")
synapses = SynapsePopulation.merge(ampa, nmda)
print(f"{len(synapses)} synapses, e.g. {list(synapses)[:2]} ... {list(synapses)[-1]}")

# %% [markdown]
# ## Build

# %%
first_site = next(iter(synapses.values())).location
model = TSModel(swc_path, synapses, {"syn_site": first_site, "sink": sink_xyz}, parameters)
built = model.build()
g = built.graph
print(f"compartments: {g.n_nodes}, loops closed: {g.n_loops} (reconnect pairs: {len(built.reconnects)})")
print("record nodes:", built.record_nodes)
print("lumped sink cm, g_pas:", built.membrane_overrides)
print("HH:", built.hh)

# %% [markdown]
# ## Run the same graph in both solvers
#
# Events map synapse **index** (label order) to times in ms. Here the first
# AMPA and NMDA synapses fire together at 10 ms.

# %%
from toric_spines_sim.simulation import jaxley_backend

labels = built.synapse_labels
events = {labels.index("ampa_0"): [10.0], labels.index("nmda_0"): [10.0]}
nodes = [built.record_nodes["syn_site"], built.record_nodes["sink"]]
T, dt, rest = float(parameters["T_ms"]), 0.025, float(parameters["Vrest_mV"])

ref = g.run(events, T_ms=T, dt_ms=dt, v0_mV=rest)
v_ref = ref.V[:, nodes]
v_jx = jaxley_backend.run(g, events, T_ms=T, dt_ms=dt, v0_mV=rest, record_nodes=nodes)
print(f"max |reference - jaxley| = {np.abs(v_ref - v_jx).max():.2e} mV")

plt.plot(ref.t_ms, v_ref[:, 0], label="synapse site (reference)")
plt.plot(ref.t_ms, v_jx[:, 0], "--", label="synapse site (jaxley)")
plt.plot(ref.t_ms, v_ref[:, 1], label="sink")
plt.xlabel("time (ms)")
plt.ylabel("V (mV)")
plt.legend()
plt.show()

# %% [markdown]
# ## Loops matter
#
# `restore_loops=False` simulates the tree stored in the SWC (what you would
# get by ignoring the reconnect headers). On TS1 the cut tree overestimates
# the local EPSP substantially.

# %%
tree = TSModel(swc_path, synapses, {"syn_site": first_site}, parameters, restore_loops=False).build()
v_tree = tree.graph.run(events, T_ms=T, dt_ms=dt, v0_mV=rest).V[:, tree.record_nodes["syn_site"]]
print(f"peak at the synapse: loops closed {v_ref[:, 0].max() - rest:.2f} mV, "
      f"loops cut {v_tree.max() - rest:.2f} mV")
