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

logging.basicConfig(level=logging.DEBUG)

from pathlib import Path
from toric_spines_sim.geometry.sink import SinkGeometry, append_sink_to_swc
from swctools import SWCModel, plot_model, PointSet, FrustaSet

# %%
um_per_px = 5 / 1000  # microns per pixel
sink_radius_um = 20
sink_connector_length_um = 5

sink_radius_px = sink_radius_um / um_per_px
sink_connector_length_px = sink_connector_length_um / um_per_px

swc_spine_px_filepath = Path("../data/swc/pixels/TS3_s200_equivalent_area.swc")
neckpt_px_filepath = Path("../data/pointsets/pixels/TS3_neckpoint.txt")
swc_with_sink_px_filepath = Path(
    f"../data/swc/pixels/TS3_s200_wsink_r{sink_radius_um}um.swc"
)
swc_with_sink_um_filepath = Path(
    f"../data/swc/microns/TS3_s200_wsink_r{sink_radius_um}um.swc"
)
az_px_filepath = Path("../data/pointsets/pixels/TS3_AZ.txt")
synpts_um_filepath = Path("../data/pointsets/microns/TS3_synpts.txt")

swc_spine_px = SWCModel.from_swc_file(swc_spine_px_filepath)
spine_frusta_px = FrustaSet.from_swc_model(swc_spine_px, sides=20, end_caps=False)

az_pointset_px = PointSet.from_txt_file(az_px_filepath)
az_pointset_px_proj = az_pointset_px.project_onto_frusta(spine_frusta_px)
synpts_pointset_um = az_pointset_px_proj.scale(um_per_px)
synpts_pointset_um.to_txt_file(synpts_um_filepath)

# %%
swc_spine_model_px = SWCModel.from_swc_file(swc_spine_px_filepath)
neckpt_pointset_px = PointSet.from_txt_file(neckpt_px_filepath)
plot_model(swc_model=swc_spine_model_px, point_set=neckpt_pointset_px, point_size=10)

# %%
from toric_spines_sim.geometry.sink import append_sink_to_swc_multi_neck_points

# add sink in px and write swc out
geom = SinkGeometry(
    radius=sink_radius_px,
    length=2 * sink_radius_px,
    n_segments=5,
    axis="-z",
    connector_length=sink_connector_length_px,
)  # radius 1000 px = 5 um
out_path = append_sink_to_swc_multi_neck_points(
    swc_in=swc_spine_px_filepath,
    swc_out=swc_with_sink_px_filepath,
    neck_points=neckpt_px_filepath,  # or (x,y,z)
    geom=geom,
    tag=5,
)

# convert to um and write swc out
swc_model_with_sink_px = SWCModel.from_swc_file(swc_with_sink_px_filepath)
swc_model_with_sink_um = swc_model_with_sink_px.scale(um_per_px)
swc_model_with_sink_um.to_swc_file(swc_with_sink_um_filepath)

# %%
fig = plot_model(
    swc_model=swc_model_with_sink_um,
    point_set=synpts_pointset_um,
    point_size=10,
    point_color="crimson",
)

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
#     )
# )

fig.show()

# %%
