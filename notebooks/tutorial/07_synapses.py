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
# # 07 — Synapses (sites and populations)
#
# This notebook is about placement: XYZ files, axon maps, and
# `SynapsePopulation`. NMODL mechanism names (`ampasyn`, and so on) are
# notebook 11.
#
# ```bash
# uv run python scripts/active_zones_from_nff.py TS1
# ```
#
# `TS1_synpts.txt` is one XYZ per active zone (microns). `ts1_axons.txt` maps
# axons to **1-based** synapse indices, one axon per line, covering `1..N`
# uniquely.

# %%
from swctools import PointSet, SWCModel, plot_model

from toric_spines_sim.geometry.prepare import (
    convert_nff_active_zone,
    write_synpts_microns,
)
from toric_spines_sim.model.synapse import MODEL_REGISTRY, SynapsePopulation
from toric_spines_sim.paths import (
    NOTEBOOKS_DIR,
    get_data_path,
    get_pointset_path,
    get_swc_path,
)
from toric_spines_sim.simulation import (
    load_axon_events_from_file,
    make_default_parameter_bank,
)
from toric_spines_sim.utils import load_xyz_points, read_nff_s_points

spine_id = "TS1"
RECOMPUTE = False  # convert NFF into tutorial_output_dir
UM_PER_PX = 0.005
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

swc_path_microns = get_swc_path(f"{spine_id}_wsink_r10um.swc", units="microns")
synpts_path = get_pointset_path(f"{spine_id}_synpts.txt", units="microns")
axon_path = get_data_path("ts_axons", "ts1_axons.txt")
nff_path = get_data_path("nff", f"{spine_id}_AZ.nff")

# %% [markdown]
# ## NFF → AZ → synpts
#
# IMOD `s` markers in NFF are pixel XYZ. `convert_nff_active_zone` writes
# `*_AZ.txt`; `write_synpts_microns` projects onto the pixel SWC and scales.
# Precomputed micron synpts already exist for TS1 — only recompute into
# `_artifacts/` if you set the flag (or if the precomputed file is missing).

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

# %% [markdown]
# ## Axon map
#
# One axon per line, 1-based synapse indices. `load_axon_events_from_file`
# returns per-axon synapse lists in **0-based** index. Event generation
# from that map is notebook 10.

# %%
n_axons = sum(1 for line in axon_path.read_text().splitlines() if line.strip())
_, n_per_axon, axon_synapses = load_axon_events_from_file(
    axon_path, axon_rates_hz=[0.0] * n_axons
)
print(f"axons: {n_axons}  ({axon_path.name})")
for axon_idx, syns in enumerate(axon_synapses):
    one_based = [i + 1 for i in syns]
    print(f"  axon {axon_idx}: synapses {one_based} ({n_per_axon[axon_idx]} sites)")

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
# ## `SynapsePopulation` / `MODEL_REGISTRY`
#
# Registry keys (`ampa`, `nmda`, …) map to NMODL names and parameter keys.
# `SynapsePopulation.from_file` samples a small bank per site (defaults are
# not sampled, so every AMPA synapse gets the same `gmax` / `tau`).
# `TSSimulator` always builds AMPA this way; mixed receptors → build
# populations yourself and pass them into `TSModel` (notebook 12).

# %%
print("MODEL_REGISTRY:")
for key, spec in MODEL_REGISTRY.items():
    print(f"  {key:8s} → {spec['mechanism']}")

parameters = make_default_parameter_bank().sample()
synapse_population = SynapsePopulation.from_file(synpts_path, "ampa", parameters)
syn0 = synapse_population.synapses["syn_0"]
print(f"\n{len(synapse_population.synapses)} AMPA synapses")
print("syn_0 location:", syn0.location)
print("syn_0 mechanism:", syn0.mechanism)
print("syn_0 synapse_params:", syn0.synapse_params)
print("syn_0 mechanism_params:", syn0.mechanism_params)
