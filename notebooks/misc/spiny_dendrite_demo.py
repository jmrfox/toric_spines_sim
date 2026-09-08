# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Spiny dendrite morphology demo
#
# Build a synthetic dendrite trunk with classical neck+head spines from explicit
# parameters, write the subsystem files (`SWC` + `AZ` + `neckpoint`), visualize,
# and optionally attach a sink with the existing geometry API.

# %%
from swctools import SWCModel, PointSet, plot_model

from toric_spines_sim.geometry.dendrite import (
    SpinyDendriteParams,
    build_spiny_dendrite,
    write_subsystem,
)
from toric_spines_sim.geometry.sink import SinkGeometry, append_sink_to_swc
from toric_spines_sim.paths import get_pointset_path, get_swc_path

# %% [markdown]
# ## 1. Build from explicit parameters
#
# The trunk is a straight tapered cable. Spines attach to trunk nodes (never the
# proximal neck), each with a thin neck node and a larger head node. Active zones
# sit at the head node XYZ — one synapse per spine.

# %%
params = SpinyDendriteParams(
    length=20.0,
    trunk_neck_radius=0.5,
    trunk_tip_radius=0.35,
    n_spines=12,
    spine_length=1.0,
    spine_neck_radius=0.1,
    spine_head_radius=0.25,
    spine_neck_length_fraction=0.5,
    max_spines_per_node=3,
    distribution="even",
    azimuth0=0.0,
    axis="z",
)

morph = build_spiny_dendrite(params)

print(f"trunk nodes: {len(morph.trunk_node_ids)}")
print(f"spines / AZ: {morph.n_spines}")
print(f"spines per attach node: {morph.spines_per_attach_node}")
print(f"surface area: {morph.surface_area():.3f} µm²")
print(f"volume:       {morph.volume():.3f} µm³")
print(f"neck point:   {morph.neck_point}")

# %% [markdown]
# ## 2. Write subsystem files
#
# Same convention as toric-spine prep: morphology SWC, AZ pointset, and a single
# neck point for later sink attachment.

# %%
swc_path = get_swc_path("spiny_dendrite.swc", units="microns")
az_path = get_pointset_path("spiny_dendrite_AZ.txt", units="microns")
neck_path = get_pointset_path("spiny_dendrite_neckpoint.txt", units="microns")

write_subsystem(morph, swc_path, az_path, neck_path)
print(f"wrote {swc_path}")
print(f"wrote {az_path}")
print(f"wrote {neck_path}")

# %% [markdown]
# ## 3. Visualize morphology + AZ + neck

# %%
swc_model = SWCModel.from_swc_file(str(swc_path))
az_points = PointSet.from_txt_file(str(az_path))
neck_points = PointSet.from_txt_file(str(neck_path))

fig = plot_model(
    swc_model=swc_model,
    point_set=az_points,
    point_size=0.15,
    point_color="crimson",
)
fig.show()

# %%
fig_neck = plot_model(
    swc_model=swc_model,
    point_set=neck_points,
    point_size=0.3,
    point_color="steelblue",
)
fig_neck.show()

# %% [markdown]
# ## 4. Optional: append a sink at the neck
#
# Downstream of subsystem generation, reuse `append_sink_to_swc` exactly as for
# toric spines / the cylinder example.

# %%
sink_radius_um = 5.0
swc_with_sink_path = get_swc_path(
    f"spiny_dendrite_wsink_r{sink_radius_um:g}um.swc", units="microns"
)

geom = SinkGeometry(
    radius=sink_radius_um,
    length=2 * sink_radius_um,
    n_cylinders=5,
    axis="-z",
    connector_length=1.0,
)
append_sink_to_swc(
    swc_in=swc_path,
    swc_out=swc_with_sink_path,
    neck_coords=neck_path,
    geom=geom,
    tag=5,
)
print(f"wrote {swc_with_sink_path}")

# %%
swc_with_sink = SWCModel.from_swc_file(str(swc_with_sink_path))
fig_sink = plot_model(
    swc_model=swc_with_sink,
    point_set=az_points,
    point_size=0.15,
    point_color="crimson",
)
fig_sink.show()

# %% [markdown]
# ## 5. Layout variant: one spine per trunk node
#
# With `max_spines_per_node=1`, the trunk is subdivided so each attach node gets
# a single spine (still skipping the proximal neck).

# %%
params_single = SpinyDendriteParams(
    length=20.0,
    trunk_neck_radius=0.5,
    trunk_tip_radius=0.35,
    n_spines=8,
    spine_length=1.2,
    spine_neck_radius=0.08,
    spine_head_radius=0.22,
    max_spines_per_node=1,
    distribution="even",
    azimuth0=0.4,
    axis="z",
)
morph_single = build_spiny_dendrite(params_single)
print(f"trunk nodes: {len(morph_single.trunk_node_ids)}")
print(f"spines per attach node: {morph_single.spines_per_attach_node}")

swc_single = get_swc_path("spiny_dendrite_one_per_node.swc", units="microns")
az_single = get_pointset_path("spiny_dendrite_one_per_node_AZ.txt", units="microns")
neck_single = get_pointset_path(
    "spiny_dendrite_one_per_node_neckpoint.txt", units="microns"
)
write_subsystem(morph_single, swc_single, az_single, neck_single)

fig_single = plot_model(
    swc_model=SWCModel.from_swc_file(str(swc_single)),
    point_set=PointSet.from_txt_file(str(az_single)),
    point_size=0.15,
    point_color="darkorange",
)
fig_single.show()
