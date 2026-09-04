# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
from toric_spines_sim.geometry.sink import (
    SinkGeometry,
    append_sink_to_swc,
    optimal_sink_direction,
)
from toric_spines_sim.paths import (
    get_swc_path,
    get_pointset_path,
    get_simulation_path,
)
from swctools import SWCModel, plot_model, PointSet, FrustaSet

# %%
# set parameters
um_per_px = 5 / 1000  # microns per pixel
sink_radius_um = 10
sink_connector_length_um = 5

# Use centralized path management - no more relative paths!
swc_spine_px_filepath = get_swc_path("TS1.swc", units="pixels")
neckpt_px_filepath = get_pointset_path("TS1_neckpoint.txt", units="pixels")
swc_with_sink_px_filepath = get_swc_path(
    f"TS1_wsink_r{sink_radius_um}um.swc", units="pixels"
)
swc_with_sink_um_filepath = get_swc_path(
    f"TS1_wsink_r{sink_radius_um}um.swc", units="microns"
)
az_px_filepath = get_pointset_path("TS1_AZ.txt", units="pixels")
synpts_um_filepath = get_pointset_path("TS1_synpts.txt", units="microns")

viz_output_path = get_simulation_path("ts1", f"TS1_wsink_r{sink_radius_um}um.html")

# %%
# convert um to px
sink_radius_px = sink_radius_um / um_per_px
sink_connector_length_px = sink_connector_length_um / um_per_px

# create geometries
swc_spine_px = SWCModel.from_swc_file(swc_spine_px_filepath)
spine_frusta_px = FrustaSet.from_swc_model(swc_spine_px, sides=20, end_caps=False)

# project AZ points
az_pointset_px = PointSet.from_txt_file(az_px_filepath)
az_pointset_px_proj = az_pointset_px.project_onto_frusta(spine_frusta_px)
synpts_pointset_um = az_pointset_px_proj.scale(um_per_px)
synpts_pointset_um.to_txt_file(synpts_um_filepath)

# neck point(s): load in pixels for visualization
neckpt_pointset_px = PointSet.from_txt_file(neckpt_px_filepath)

# convert to microns for visualization
swc_spine_um = swc_spine_px.scale(um_per_px)
neckpt_pointset_um = neckpt_pointset_px.scale(um_per_px)

# view spine with neck point
fig1 = plot_model(
    swc_model=swc_spine_um,
    point_set=neckpt_pointset_um,
    point_size=10,
    slider=True,
    title="Spine with neck point",
)
fig1.show()

# view spine with synapse points
fig2 = plot_model(
    swc_model=swc_spine_um,
    point_set=synpts_pointset_um,
    point_size=10,
    slider=True,
    title="Spine with synapse points",
)
fig2.show()

# %%
# compute optimal sink direction (in pixel space)
sink_direction = optimal_sink_direction(neckpt_px_filepath, swc_spine_px_filepath)
logger.info("Optimal sink direction: %s", sink_direction)

# add sink in px then convert to um and write swc out
geom = SinkGeometry(
    radius=sink_radius_px,
    length=2 * sink_radius_px,
    n_cylinders=5,
    axis=sink_direction,
    connector_length=sink_connector_length_px,
)
out_path = append_sink_to_swc(
    swc_in=swc_spine_px_filepath,
    swc_out=swc_with_sink_px_filepath,
    neck_coords=neckpt_px_filepath,  # or (x,y,z)
    geom=geom,
    tag=5,
)

swc_model_with_sink_px = SWCModel.from_swc_file(swc_with_sink_px_filepath)
swc_model_with_sink_px.to_swc_file(
    swc_with_sink_px_filepath
)  # write out combined swc in px
swc_model_with_sink_um = swc_model_with_sink_px.scale(um_per_px)
swc_model_with_sink_um.to_swc_file(
    swc_with_sink_um_filepath
)  # write out combined swc in um

# %%
# final visualization
fig = plot_model(
    swc_model=swc_model_with_sink_um,
    point_set=synpts_pointset_um,
    point_size=20,
    point_color="crimson",
    slider=True,
    title="TS1 with sink and synapses",
)
fig.show()

# %%

# %%
