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
# # 12 — `TSModel` and `TSRecipe`
#
# This notebook builds an Arbor cable cell from the TS1 micron sink SWC and
# synpts in `data/`, without running a simulation. It needs the NMODL
# catalogue (notebook 11):
#
# ```bash
# uv run bash scripts/make_custom_catalogue.sh
# ```
#
# `TSSimulator` (notebook 13) uses this same construction path. Use `TSModel`
# directly when you need mixed synapse types or a custom probe set.

# %%
from toric_spines_sim.events import FlatRateCurve, StochasticEventGenerator
from toric_spines_sim.geometry import sink_endpoint_location_from_swc_file
from toric_spines_sim.model import TSModel, TSRecipe
from toric_spines_sim.model.gj import prepare_gap_junctions
from toric_spines_sim.model.synapse import SynapsePopulation
from toric_spines_sim.paths import PROJECT_ROOT, get_pointset_path, get_swc_path
from toric_spines_sim.simulation import make_default_parameter_bank
from toric_spines_sim.utils import load_xyz_points

catalogue = PROJECT_ROOT / "toric_spines_sim" / "mechanisms" / "custom-catalogue.so"
if not catalogue.is_file():
    raise FileNotFoundError(
        f"Missing {catalogue}. From the repo root run:\n"
        "  uv run bash scripts/make_custom_catalogue.sh"
    )

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
build_result = model.build_cell()
print("build_cell keys:", sorted(build_result))
print("cell:", type(build_result["cell"]).__name__)
print("n segments:", len(list(build_result["segment_tree"].segments)))
print("labels:", type(build_result["labels"]).__name__)

# %% [markdown]
# ## `TSRecipe`
#
# Subclass of `arbor.recipe`. A pynapple `TsGroup` is mapped by **index** to
# `list(synapses.keys())` order (`syn_0`, `syn_1`, …), not by channel label.
# A dict maps synapse labels to time lists directly.
#
# Stop here — `run()` is notebook 13.

# %%
rates_hz = [0.0] * n_synapses
rates_hz[0] = rates_hz[1] = 50.0
events = StochasticEventGenerator(
    rate_curves=[FlatRateCurve(r) for r in rates_hz],
    n_synapses_per_axon=[1] * n_synapses,
    T_ms=parameters["T_ms"],
    delay_ms=parameters["delay_ms"],
    seed=int(parameters["seed"]),
).generate()

recipe = TSRecipe(
    build_result["cell"],
    synapses=synapses,
    gap_junctions=gap_junctions,
    record_points=record_points,
    events=events,
    parameters=parameters,
    custom_catalogue=build_result["custom_catalogue"],
)
print("recipe:", type(recipe).__name__)
print("num_cells:", recipe.num_cells())
