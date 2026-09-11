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
# # 14 — `TSModel` and `TSRecipe`
#
# To build a cable cell from the TS1 micron sink SWC and synpts, use
# `TSModel`. To attach events and probes, wrap that cell in a `TSRecipe`.
# `run()` is `tssimulator`. The NMODL catalogue (`mechanisms`) is required:
#
# ```bash
# uv run bash scripts/make_custom_catalogue.sh
# ```
#
# For mixed synapse types (`synapses_advanced`) or a custom probe set,
# `TSModel` is the direct path. `TSSimulator` uses this same construction
# internally.

# %%
import pynapple as nap

from toric_spines_sim.geometry import sink_endpoint_location_from_swc_file
from toric_spines_sim.model import TSModel, TSRecipe, check_catalogue
from toric_spines_sim.model.gj import prepare_gap_junctions
from toric_spines_sim.model.synapse import SynapsePopulation
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from toric_spines_sim.simulation import make_default_parameter_bank
from toric_spines_sim.utils import load_xyz_points

check_catalogue()

swc_path = get_swc_path("TS1_wsink_r10um.swc", units="microns")
synpts_path = get_pointset_path("TS1_synpts.txt", units="microns")
n_synapses = len(load_xyz_points(synpts_path))
sink_xyz = sink_endpoint_location_from_swc_file(swc_path)

parameter_bank = make_default_parameter_bank()
parameter_bank["T_ms"].value = 150.0
parameter_bank["delay_ms"].value = 20.0
parameter_bank["discretization_um"].value = 1.0
parameter_bank["hh_scale"].value = 0.0
parameters = parameter_bank.sample()
parameters["hh_tags"] = []

print("SWC:", swc_path)
print("sink XYZ:", sink_xyz)
print("n synapses:", n_synapses)

# %% [markdown]
# ## Synapses and gap junctions
#
# AMPA population from synpts. `prepare_gap_junctions` reads `# CYCLE_BREAK`
# / `# MULTI_NECK` and returns `gj_0`, `gj_1`, … (`GapJunctionPoint` holds
# SWC node IDs; `TSModel` maps those IDs to Arbor locations).

# %%
synapses = SynapsePopulation.from_file(synpts_path, "ampa", parameters).synapses
gap_junctions = prepare_gap_junctions(swc_path, parameters)
print(f"synapses: {len(synapses)}  (keys e.g. {list(synapses)[:3]})")
print(f"gap junctions: {len(gap_junctions)}")
for label, gj in list(gap_junctions.items())[:4]:
    print(f"  {label}: nodes {gj.index_pair}  weight={gj.weight}")

# %% [markdown]
# ## `TSModel.build_cell()`
#
# Loads the catalogue, optional sink/neck radius scales, applies leak (and HH
# if `hh_tags` + `hh_scale`), places synapses and GJs, attaches voltage
# probes at `record_points`.

# %%
record_points = {"sink": sink_xyz}
model = TSModel(
    swc_path=swc_path,
    synapses=synapses,
    gap_junctions=gap_junctions,
    record_points=record_points,
    parameters=parameters,
)
print("TSModel:", type(model).__name__)
print("  swc_path:", model.swc_path)
print("  n synapses:", len(model.synapses))
build_result = model.build_cell()
print("build_cell keys:", sorted(build_result))
print("cell:", type(build_result["cell"]).__name__)
print("n segments:", len(list(build_result["segment_tree"].segments)))
print("labels:", type(build_result["labels"]).__name__)

# %% [markdown]
# ## `TSRecipe`
#
# Subclass of `arbor.recipe`. Two ways to pass times (ms):
#
# - **`TsGroup`:** index `i` → `list(synapses.keys())[i]` (`syn_0`, …). Labels
#   ignored. Build this from lists or a file (`events_basic`).
# - **`dict[str, list[float]]`:** keys must match place tags.
#
# Stop here — `run()` is `tssimulator`.

# %%
t_ms = float(parameters["T_ms"])
delay_ms = float(parameters["delay_ms"])
tsgroup_events = nap.TsGroup(
    {
        i: nap.Ts(
            t=([delay_ms, delay_ms + 20.0] if i in (0, 1) else []),
            time_units="ms",
        )
        for i in range(n_synapses)
    },
    time_support=nap.IntervalSet(start=[0], end=[t_ms], time_units="ms"),
)
print("TsGroup streams:", len(tsgroup_events), "(indices 0 and 1 active)")

dict_events = {
    label: ([delay_ms, delay_ms + 20.0] if label in ("syn_0", "syn_1") else [])
    for label in synapses
}
print("dict keys sample:", list(dict_events)[:4], "...")

recipe_from_tsgroup = TSRecipe(
    build_result["cell"],
    synapses=synapses,
    gap_junctions=gap_junctions,
    record_points=record_points,
    events=tsgroup_events,
    parameters=parameters,
    custom_catalogue=build_result["custom_catalogue"],
)
recipe_from_dict = TSRecipe(
    build_result["cell"],
    synapses=synapses,
    gap_junctions=gap_junctions,
    record_points=record_points,
    events=dict_events,
    parameters=parameters,
    custom_catalogue=build_result["custom_catalogue"],
)
print("recipe from TsGroup:", type(recipe_from_tsgroup).__name__)
print("recipe from dict:   ", type(recipe_from_dict).__name__)
print("num_cells:", recipe_from_tsgroup.num_cells())
