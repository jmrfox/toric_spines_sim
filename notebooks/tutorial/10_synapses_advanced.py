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
# # 10 — Synapses (advanced)
#
# To mix receptor types, use more than one `SynapsePopulation` (different
# `MODEL_REGISTRY` keys) and `merge` them. Axon maps live in the same
# notebook. `synapses_basic` built one AMPA population from synpts file
# order. Catalogue and cell construction are `mechanisms` and
# `tsmodel_and_tsrecipe` — this notebook does not load the NMODL catalogue or
# call `run()`.

# %%
from jscip import IndependentScalarParameter, ParameterBank

from toric_spines_sim.model.synapse import MODEL_REGISTRY, SynapsePopulation
from toric_spines_sim.paths import get_data_path, get_pointset_path
from toric_spines_sim.simulation import (
    load_axon_events_from_file,
    make_default_parameter_bank,
)
from toric_spines_sim.utils import load_xyz_points

spine_id = "TS1"
synpts_path = get_pointset_path(f"{spine_id}_synpts.txt", units="microns")
axon_path = get_data_path("ts_axons", "ts1_axons.txt")
parameters = make_default_parameter_bank().sample()
n_synapses = len(load_xyz_points(synpts_path))
print(f"synpts: {n_synapses} rows ({synpts_path.name})")

# %% [markdown]
# ## How do I choose a synapse mechanism?
#
# `MODEL_REGISTRY` keys are the `model=` argument to
# `SynapsePopulation.from_file`. Each key maps to an NMODL mechanism name
# and the `ParameterSet` fields that fill it.
#
# `TSSimulator` always builds AMPA. For any other type (or a mix), build
# population(s) yourself and pass the merged dict into `TSModel`
# (`tsmodel_and_tsrecipe`). NMODL sources and the compiled catalogue are
# `mechanisms`.

# %%
print(f"{'key':8s}  {'NMODL':12s}  ParameterSet keys")
for key, spec in MODEL_REGISTRY.items():
    print(f"{key:8s}  {spec['mechanism']:12s}  {spec['global_keys']}")

nmda_population = SynapsePopulation.from_file(synpts_path, "nmda", parameters)
print("\nNMDA syn_0 mechanism:", nmda_population.synapses["syn_0"].mechanism)
print("NMDA syn_0 mechanism_params:", nmda_population.synapses["syn_0"].mechanism_params)

# %% [markdown]
# ## Mixed populations
#
# `SynapsePopulation.merge` concatenates label → `SynapsePoint` dicts.
# Labels must be unique, so use a different `label_prefix` per population
# (or build the dict by hand). Insertion order is the order `TSRecipe` uses
# when mapping a `TsGroup` by index.
#
# The example below places AMPA and NMDA at **every** synpts site (two
# mechanisms per XYZ). The cell after it uses two locations as disjoint
# subsets.

# %%
ampa = SynapsePopulation.from_file(
    synpts_path, "ampa", parameters, label_prefix="ampa"
)
nmda = SynapsePopulation.from_file(
    synpts_path, "nmda", parameters, label_prefix="nmda"
)
merged = SynapsePopulation.merge(ampa, nmda)
print(f"merged {len(merged)} points; first/last keys:", list(merged)[0], list(merged)[-1])
print("ampa_0 mechanism:", merged["ampa_0"].mechanism)
print("nmda_0 mechanism:", merged["nmda_0"].mechanism)

# %% [markdown]
# ## Disjoint subsets
#
# Two locations, two models. Pass filtered location lists into
# `SynapsePopulation` (not the full file). `merge` still requires unique
# labels.

# %%
locations = load_xyz_points(synpts_path)
ampa_only = SynapsePopulation(
    "ampa", locations[:1], parameters, label_prefix="ampa"
)
nmda_only = SynapsePopulation(
    "nmda", locations[1:2], parameters, label_prefix="nmda"
)
disjoint = SynapsePopulation.merge(ampa_only, nmda_only)
print("disjoint keys:", list(disjoint))
for label, syn in disjoint.items():
    print(f"  {label}: {syn.mechanism} at {syn.location}")

# %% [markdown]
# ## Can synapses have names?
#
# Synpts files have no name column. `label_prefix` only changes the prefix
# of positional IDs (`syn_0` → `foo_0`). Meaningful names exist only on a
# hand-built `Dict[str, SynapsePoint]` passed to `TSModel`.
#
# Event mapping depends on how times are passed:
#
# - **pynapple `TsGroup`:** channel **index** `i` → `list(synapses.keys())[i]`.
#   Channel labels (`A0S0`, `"A"`, even `syn_0`) are ignored.
# - **`dict[str, list[float]]`:** keys must match Arbor place tags.
#
# Generator channel order should match synpts row order, or be remapped
# (below). Rate-curve details are `events_basic` and `events_advanced`.

# %%
print("default labels (file order):", list(ampa.synapses)[:4], "...")
print("custom prefix does not add semantics: still index 0, 1, 2, …")

# %% [markdown]
# ## Axon maps (a second index)
#
# `data/ts_axons/ts1_axons.txt` is one axon per line, comma-separated
# **1-based** synpts indices covering `1..N` uniquely.
# `load_axon_events_from_file` converts those to **0-based** lists.
#
# Event generators fan out channels in **axon order** (all synapses of axon
# 0, then axon 1, …). `TSRecipe` / `TSSimulator` still require **synpts
# order**. `remap_axon_channel_events_to_synapses` scatters axon-order
# channels onto indices `0 .. N-1`. `events_advanced` runs that remap.

# %%
n_axons = sum(1 for line in axon_path.read_text().splitlines() if line.strip())
_, n_per_axon, axon_synapses = load_axon_events_from_file(
    axon_path, axon_rates_hz=[0.0] * n_axons
)
print(f"axons: {n_axons}  ({axon_path.name})")
for axon_idx, syns in enumerate(axon_synapses):
    one_based = [i + 1 for i in syns]
    print(f"  axon {axon_idx}: synapses {one_based} ({n_per_axon[axon_idx]} sites)")

# %% [markdown]
# ## Per-site parameter sampling
#
# By default every synapse in a population gets the same sampled
# `ParameterSet` values (`is_sampled=False` on those keys). Pass a
# `parameter_override` `ParameterBank` with `is_sampled=True` to draw per
# site.

# %%
override = ParameterBank(
    {
        "ampa_gmax_uS": IndependentScalarParameter(
            0.01, is_sampled=True, range=(0.005, 0.02)
        ),
    }
)
sampled_population = SynapsePopulation.from_file(
    synpts_path,
    "ampa",
    parameters,
    parameter_override=override,
)
gmax_values = [
    syn.synapse_params["gmax_uS"] for syn in sampled_population.synapses.values()
]
print(f"unique gmax values among {len(gmax_values)} synapses: {len(set(gmax_values))}")
print("first five gmax_uS:", [round(v, 6) for v in gmax_values[:5]])
