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
# # 09 — Synapses (basic)
#    
# Information flow:
#
# 1. XYZ synapse locations file (`TS{id}_synpts.txt`), one row per synapse.
# 2. `SynapsePopulation` builds a `SynapsePoint` per row (`syn_0`, `syn_1`, …).
# 3. `TSModel` places each point on the morphology under that label.
# 4. `TSRecipe` / `TSSimulator` attach event stream `i` to `syn_i`.
#

# %%
from swctools import PointSet, SWCModel, plot_model

from toric_spines_sim.geometry.prepare import (
    convert_nff_active_zone,
    write_synpts_microns,
)
from toric_spines_sim.model.synapse import SynapsePoint, SynapsePopulation
from toric_spines_sim.paths import (
    NOTEBOOKS_DIR,
    get_data_path,
    get_pointset_path,
    get_swc_path,
)
from toric_spines_sim.simulation import make_default_parameter_bank
from toric_spines_sim.utils import load_xyz_points, read_nff_s_points

spine_id = "TS1"
RECOMPUTE = False  # convert NFF into tutorial_output_dir
UM_PER_PX = 0.005
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

swc_path_microns = get_swc_path(f"{spine_id}_wsink_r10um.swc", units="microns")
synpts_path = get_pointset_path(f"{spine_id}_synpts.txt", units="microns")
nff_path = get_data_path("nff", f"{spine_id}_AZ.nff")

# %% [markdown]
# ## One `SynapsePoint`
#
# Synapses are managed using the `SynapsePoint` dataclass, which specifies 
# location (in the same coordinate space as the swc model), synapse model/mechanism,
# and mechanism parameters.
#
# When creating a `SynapsePoint`, you don't need to specify the NMODL mechanism specifically: there is a
# structured dict called `MODEL_REGISTRY` that handles wiring synapse
# model names to the NMODL mechanism, as well as routing from your simulation 
# parameters to the synapse.
#
# Each site is a `SynapsePoint`: `location`, `model` (registry key),
# `mechanism` (NMODL name), `synapse_params` (human-readable), and
# `mechanism_params` (Arbor names). Construct a one-site population from a
# location list — the same class `from_file` uses for every row.

# %%
print("SynapsePoint fields:", SynapsePoint.__dataclass_fields__.keys())

parameters = make_default_parameter_bank().sample()
one_site = SynapsePopulation(
    model="ampa",
    locations=[(0.0, 0.0, 0.0)],
    global_parameters=parameters,
)
syn = one_site.synapses["syn_0"]
print("labels:", list(one_site.synapses))
print("syn_0 location:", syn.location)
print("syn_0 model / mechanism:", syn.model, "/", syn.mechanism)
print("syn_0 synapse_params:", syn.synapse_params)
print("syn_0 mechanism_params:", syn.mechanism_params)

# %% [markdown]
# ## NFF → AZ → synpts
#
# IMOD `s` markers in NFF are pixel XYZ. `convert_nff_active_zone` writes
# `*_AZ.txt`; `write_synpts_microns` projects onto the pixel SWC and scales.
# Precomputed micron synpts already exist for TS1 — only recompute into
# `_artifacts/` if you set the flag (or if the precomputed file is missing).
#
# A synpts file is whitespace-delimited `x y z` (µm). No header, no ID
# column, no names. **Row `i` (0-based) is synapse `i`.**

# %%
if nff_path.is_file():
    nff_points = read_nff_s_points(nff_path)
    print(f"NFF {nff_path.name}: {len(nff_points)} s-markers")
else:
    print(f"No NFF at {nff_path}")

need_convert = RECOMPUTE or not synpts_path.is_file()
if need_convert and nff_path.is_file():
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    az_output_path = tutorial_output_dir / f"{spine_id}_AZ.txt"
    convert_nff_active_zone(nff_path, az_output_path)
    print("wrote AZ", az_output_path)
    swc_path_pixels = get_swc_path(f"{spine_id}_wsink_r10um.swc", units="pixels")
    if not swc_path_pixels.is_file():
        swc_path_pixels = get_swc_path(f"{spine_id}.swc", units="pixels")
    synpts_output_path = tutorial_output_dir / f"{spine_id}_synpts.txt"
    write_synpts_microns(
        swc_path_pixels, az_output_path, synpts_output_path, um_per_px=UM_PER_PX
    )
    synpts_path = synpts_output_path
    print("wrote synpts", synpts_path)

if not synpts_path.is_file():
    raise FileNotFoundError(
        f"Missing {synpts_path}. Convert NFF with scripts/active_zones_from_nff.py"
    )

synpts = load_xyz_points(synpts_path)
print(f"synapses: {len(synpts)}  ({synpts_path})")
print("first three XYZ:")
for index, location in enumerate(synpts[:3]):
    print(f"  row {index}  syn_{index}  {location}")

# %%
model = SWCModel.from_swc_file(str(swc_path_microns), validate_reconnections=False)
syn_pointset = PointSet.from_txt_file(synpts_path)
fig = plot_model(
    swc_model=model,
    point_set=syn_pointset,
    point_size=0.15,
    point_color="crimson",
    slider=False,
    title=f"{spine_id} sink SWC + synpts (µm)",
    show_axes=False,
    show_frusta=True,
    show_centroid=False,
    width=1200,
    height=900,
)
fig.show()

# %% [markdown]
# ## `SynapsePopulation.from_file`
#
# To place a group of homogeneous synapses from a synpts file (list of locations), you can use `SynapsePopulation`, which
# builds one `SynapsePoint` per row of the file. Firing those sites needs one event
# stream per site in **file row order** (`events_basic`, `tssimulator`).
# Mixed receptors: `synapses_advanced`. This notebook does not build a cell
# or call `run()`.
#
# `from_file` reads the XYZ rows in order, samples a small parameter bank per
# site (defaults are not sampled, so every AMPA synapse gets the same `gmax`
# / `tau`), and stores `syn_{index}` → `SynapsePoint`. **Row `i` is synapse
# `i`.**

# %%
synapse_population = SynapsePopulation.from_file(synpts_path, "ampa", parameters)
syn0 = synapse_population.synapses["syn_0"]
print(f"{len(synapse_population.synapses)} AMPA synapses")
print("labels:", list(synapse_population.synapses)[:5], "...")
print("syn_0 location:", syn0.location)
print("syn_0 model / mechanism:", syn0.model, "/", syn0.mechanism)

# %% [markdown]
# ## How simulations use this
#
# `TSSimulator` always builds AMPA this way from the synpts file. It maps
# pynapple `TsGroup` **index** `i` to `syn_i`. Channel labels are ignored.
#
# Firing these sites needs **N event streams in that order** — Python lists,
# a timestamp file, or a generator (`events_basic`) — passed as a `TsGroup`
# to `TSSimulator` (`tssimulator`). Axon-order data needs a remap
# (`events_advanced`). For receptors other than AMPA, pass a synapse dict
# into `TSModel` (`synapses_advanced`, `tsmodel_and_tsrecipe`).
