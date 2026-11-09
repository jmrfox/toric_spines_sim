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
# # 13 — Mechanisms (synaptic kinetics and HH)
#
# There is no compiled catalogue any more. Synapses are described in Python
# (`toric_spines_sim.model.synaptic`) and shared by both solvers:
#
# - every registry model (`ampa`, `nmda`, `gabaa`, `gabab`, `effexc`) is one or
#   two **components**: a conductance time course that does not depend on
#   voltage (single exponential, or a peak-normalized difference of
#   exponentials), a reversal potential, and an optional Mg block evaluated
#   from the membrane voltage during the solve;
# - `*_gmax_uS` is the **peak** conductance of one event;
# - Hodgkin-Huxley channels (same rates as the old `hhnotemp`) run in jaxley on
#   the frusta tagged `hh_tags`, scaled by `hh_scale`.
#
# The old NMODL sources are kept on the `arbor-evidence` and `arbor-legacy` branches.

# %%
import matplotlib.pyplot as plt
import numpy as np

from toric_spines_sim.model import hh_spec_from_parameters, synaptic
from toric_spines_sim.model.synapse import MODEL_REGISTRY
from toric_spines_sim.simulation import make_icx_parameter_bank

parameters = make_icx_parameter_bank("in_vitro").sample()
print(f"{'model':8s}  global ParameterSet keys")
for key, spec in MODEL_REGISTRY.items():
    print(f"{key:8s}  {spec['global_keys']}")

# %% [markdown]
# ## Components of each model
#
# `synaptic.components(model, synapse_params)` turns a synapse's parameters
# into components. `effexc` has two: an AMPA part and an NMDA part with Mg
# block, weighted by `nmda_ratio` (a ratio of peak conductances).

# %%
params = {
    "ampa": dict(gmax_uS=0.001, tau_ms=2.0, e_mV=0.0),
    "nmda": dict(gmax_uS=0.001, tau_r_ms=5.0, tau_d_ms=50.0, e_mV=0.0, mg_mM=1.3),
    "gabaa": dict(gmax_uS=0.001, tau_ms=2.0, e_mV=-70.0),
    "gabab": dict(gmax_uS=0.0001, tau_r_ms=30.0, tau_d_ms=200.0, e_mV=-95.0),
    "effexc": dict(gmax_uS=0.0001, nmda_ratio=0.5, tau_ampa_ms=2.0, tau_nmda_rise_ms=2.0,
                   tau_nmda_decay_ms=50.0, e_mV=0.0, mg_mM=1.3),
}
for model, p in params.items():
    for c in synaptic.components(model, p):
        print(f"{model:7s} {c.kind:5s} peak {c.scale_uS * 1e3:6.3f} nS  e={c.e_mV:6.1f} mV  "
              f"Mg block: {'yes' if c.mg_mM is not None else 'no'}")

# %% [markdown]
# ## Conductance after one event
#
# Each component peaks at its `scale_uS`. Before the 2026 fix, the
# dual-exponential mechanisms peaked at only ~1.5% of `gmax`.

# %%
dt, T = 0.025, 300.0
n = int(T / dt)
t = np.arange(n) * dt
fig, ax = plt.subplots(figsize=(8, 3.5))
for model, p in params.items():
    g = sum(synaptic.series(c, [10.0], n, dt) for c in synaptic.components(model, p))
    ax.plot(t, g / g.max(), label=f"{model} (peak {g.max() * 1e3:.3f} nS)")
ax.set_xlim(0, 150)
ax.set_xlabel("time (ms)")
ax.set_ylabel("g / peak")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## Mg block (Jahr & Stevens 1990)
#
# Evaluated from the membrane voltage at every step. ACSF Mg is 1.3 mM (in vitro);
# the in-vivo bank uses 1.0 mM.

# %%
v = np.linspace(-90, 20, 200)
for mg in (1.0, 1.3):
    plt.plot(v, synaptic.mg_block(v, mg), label=f"{mg} mM")
plt.xlabel("V (mV)")
plt.ylabel("unblocked fraction")
plt.legend()
plt.show()

# %% [markdown]
# ## Hodgkin-Huxley
#
# `hh_spec_from_parameters` reads `hh_tags`, `hh_scale` and the Na/K/leak
# densities and reversals. `None` means the cell is passive; then
# `TSSimulator(solver="auto")` uses the fast reference solver. With HH it uses
# jaxley.

# %%
print("passive:", hh_spec_from_parameters(parameters))
parameters["hh_tags"] = [5]
print("HH on the sink:", hh_spec_from_parameters(parameters))
parameters["hh_tags"] = []
