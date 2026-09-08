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
# # 08 — Parameters
#
# ## `jscip`
#
# Parameter management is [jscip](https://github.com/jmrfox/jscip). The package
# prioritizes random sampling. Even without sampling, a `ParameterBank` is the
# usual interface: set attributes, then call `bank.sample()` to get a
# `ParameterSet` for `TSModel` / `TSSimulator`.
#
# This project ships `make_default_parameter_bank()`. Per-spine experiments
# override a few `.value` lines in `simulations/ts{id}/params.py` and leave
# the rest. You can also build your own bank from scratch.
#
# Units:
#
# - time: ms
# - temperature: Kelvin (in vitro ~280 K; barn owl in vivo ~313 K)
# - membrane capacitance `cm_uF_per_cm2`: µF/cm²
# - leak `pas_leak_g_S_per_cm2`: S/cm²
# - axial resistivity `rL_ohm_cm`: Ω·cm
# - synapse conductances: µS
#
# SWC tags: spine=3, sink=5, sink tip=6.

# %%
from jscip import (
    DerivedScalarParameter,
    IndependentScalarParameter,
    ParameterBank,
)
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from toric_spines_sim.paths import get_data_path
from toric_spines_sim.simulation import (
    make_default_parameter_bank,
    make_icx_parameter_bank,
)

# %% [markdown]
# ## Bank vs set
#
# A `ParameterBank` holds independent scalars/vectors plus **derived**
# quantities. `bank.sample()` draws (or copies) values into a `ParameterSet`.
# Derived entries are recomputed from independents at sample time.

# %%
demo = ParameterBank(
    {
        "cm_uF_per_cm2": IndependentScalarParameter(1.0),
        "pas_leak_g_S_per_cm2": IndependentScalarParameter(0.001),
        "tau_m_ms": DerivedScalarParameter(
            lambda params: params["cm_uF_per_cm2"]
            / params["pas_leak_g_S_per_cm2"]
            / 1000.0
        ),
        "g_sampled": IndependentScalarParameter(
            0.001, is_sampled=True, range=(1e-4, 1e-2)
        ),
    }
)
print("one sample:\n", demo.sample())
print("another sample (g_sampled moves):\n", demo.sample())

# %% [markdown]
# ## Default bank
#
# Almost every entry has `is_sampled=False`, so `sample()` returns the `.value`
# you set. Membrane time constant `tau_m_ms` is derived:
# $C_m / g_{\mathrm{pas}} / 1000$.

# %%
parameter_bank = make_default_parameter_bank()
sampled_parameters = parameter_bank.sample()
print("bank entries:")
for name in sampled_parameters.index:
    print(f"  {name}")

# %%
parameter_bank["T_ms"].value = 200.0
parameter_bank["delay_ms"].value = 20.0
parameter_bank["discretization_um"].value = 1.0
parameter_bank["cm_uF_per_cm2"].value = 2.0
parameter_bank["rL_ohm_cm"].value = 150.0
parameter_bank["pas_leak_g_S_per_cm2"].value = 0.001
parameter_bank["ampa_gmax_uS"].value = 0.1
parameter_bank["ampa_tau_ms"].value = 2.0
parameter_bank["hh_scale"].value = 0.0

parameters = parameter_bank.sample()
# hh_tags is a vector parameter: assign on the sampled set, not the bank.
parameters["hh_tags"] = []

print("T_ms =", parameters["T_ms"])
print("discretization_um =", parameters["discretization_um"])
print("cm_uF_per_cm2 =", parameters["cm_uF_per_cm2"])
print("pas_leak_g_S_per_cm2 =", parameters["pas_leak_g_S_per_cm2"])
print("tau_m_ms (derived) =", parameters["tau_m_ms"])
print("ampa_gmax_uS =", parameters["ampa_gmax_uS"])
print("hh_scale =", parameters["hh_scale"], "hh_tags =", list(parameters["hh_tags"]))
print(
    "spine/sink tags =",
    parameters["spine_tag"],
    parameters["sink_tag"],
    parameters["sink_tip_tag"],
)
print(
    "radius scales: sink =",
    parameters["sink_radii_scale"],
    "neck =",
    parameters["neck_radius_scale"],
)

# %% [markdown]
# ### Hodgkin–Huxley
#
# Passive cell: empty `hh_tags` and `hh_scale = 0`. To apply HH on the sink:
#
# ```python
# parameters["hh_tags"] = [5]
# parameters["hh_scale"] = 1.0
# ```
#
# The catalogue mechanism is `hhnotemp` (temperature is a separate `temp_K`
# parameter). See notebook 11.

# %% [markdown]
# ## ICx-like leak vs package default
#
# `make_icx_parameter_bank()` uses a smaller leak (Sanculi-like). In-vivo leak
# is often taken as ~3× the in-vitro Sanculi value.

# %%
icx = make_icx_parameter_bank().sample()
default = make_default_parameter_bank().sample()
print("default  g_pas =", default["pas_leak_g_S_per_cm2"], " tau_m_ms =", default["tau_m_ms"])
print("icx      g_pas =", icx["pas_leak_g_S_per_cm2"], " tau_m_ms =", icx["tau_m_ms"])

# %% [markdown]
# ## Where experiments override
#
# Do not fork the factory for one spine. In `simulations/ts1/params.py`:
#
# ```python
# def make_parameter_bank():
#     parameter_bank = make_default_parameter_bank()
#     # parameter_bank["pas_leak_g_S_per_cm2"].value = 0.0005
#     # parameter_bank["ampa_gmax_uS"].value = 0.008
#     return parameter_bank
# ```
#
# Notebook 13 samples a short-run bank the same way, then passes the
# `ParameterSet` into `TSSimulator`.

# %% [markdown]
# ## Sanculi $C_m$
#
# Spine properties in `data/toric_spines_properties_imputed.csv`. Capacitance
# is nF; convert to µF/cm² with a representative surface area
# (spine + extra ~500 µm²).

# %%
props = pd.read_csv(get_data_path("toric_spines_properties_imputed.csv"))
print("RMP mean (mV):", props["RMP (mV)"].mean())
avg_area_um2 = 1895 + 500
cm_nF_per_um2 = props["Capacitance (nF)"] / avg_area_um2
# 1 nF/µm² = 10^5 µF/cm²
cm_uF_per_cm2 = cm_nF_per_um2 * 1e5
print(
    f"Cm (µF/cm²) = {cm_uF_per_cm2.median():0.3f} +/- {cm_uF_per_cm2.std():0.3f}"
)

# %% [markdown]
# ## AMPA $\tau$ from an EPSC
#
# Digitized decay from Sanculi 2019 Fig. 16F. An exponential fit gives the
# `ampa_tau_ms` scale used in the default bank (~2 ms).

# %%
raw_epsc = """
0.9558534806753762, 82.43226566444157
2.0463342121501036, 76.63992819879955
3.3477309698771442, 71.45565714926796
4.5453525551130305, 65.86582150670333
6.417793234980646, 59.561900488023795
8.064172322881017, 53.56369551803445
9.934930162113648, 47.36130588433275
12.43507039883323, 43.18617826891793
15.394065182027251, 37.99349301621136
19.113142985359286, 33.609132215179216
21.946485667807256, 29.33079037415157
25.21848880910977, 25.25326751556628
29.811521848880904, 21.473607449374548
33.63774050597408, 17.29174847141976
39.54002355976888, 14.520670892466484
45.43052672911875, 12.460313008358114
50.77018006394793, 10.30122847366355
56.76950692769394, 8.34184102765468
64.6272507993493, 7.591294104448309
71.4769731306445, 7.65804678296967
77.80613675884892, 5.7985078813036495
82.99826106467717, 5.873674762999954
89.31227912716666, 4.927918326134531
96.39647725360408, 4.181298031076466
102.05755314971671, 2.629718965613975
107.91776518763672, 2.396926011106764
114.1145453525551, 1.8578560610310362
120.64284512256688, 1.3171032703203167
127.94300779716158, 0.873955236439123
134.6738093902507, 1.4489257867280116
140.86049251135915, 1.5190441465193487
147.38542660010094, 1.181354125764308
"""
epsc = np.array(
    [
        [float(x) for x in line.split(",")]
        for line in raw_epsc.strip().splitlines()
    ]
)


def epsc_decay(t, scale, tau, offset):
    return scale * np.exp(-t / tau) + offset


popt, _ = curve_fit(epsc_decay, epsc[:, 0], epsc[:, 1])
plt.plot(epsc[:, 0], epsc[:, 1], "o", label="Sanculi Fig. 16F")
plt.plot(epsc[:, 0], epsc_decay(epsc[:, 0], *popt), label="fit")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (pA)")
plt.legend()
plt.show()
print(f"scale: {popt[0]:.2f} pA")
print(f"tau:   {popt[1]:.2f} ms   (bank default ampa_tau_ms = 2.0)")
print(f"offset: {popt[2]:.2f} pA")
