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

# %%
import logging

logging.basicConfig(level=logging.DEBUG)

from pathlib import Path

from toric_spines_sim.geometry.sink import SinkGeometry, append_sink_to_swc
from toric_spines_sim.paths import get_pointset_path, get_swc_path
from swctools import SWCModel, plot_model, PointSet, FrustaSet

# %% [markdown]
# Canonical example: append a cylindrical sink to ``cylinder.swc``.
#
# For toric-spine SWCs prefer ``uv run python scripts/append_sink.py TS1``.

# %%
sink_radius_um = 20

swc_spine_um_filepath = get_swc_path("cylinder.swc", units="microns")
neckpt_um_filepath = get_pointset_path("cylinder_neckpoint.txt", units="microns")
swc_with_sink_um_filepath = get_swc_path(
    f"cylinder_wsink_r{sink_radius_um}um.swc", units="microns"
)
synpts_um_filepath = get_pointset_path("cylinder_synpts.txt", units="microns")

swc_spine_um = SWCModel.from_swc_file(swc_spine_um_filepath)
spine_frusta_um = FrustaSet.from_swc_model(swc_spine_um, sides=20, end_caps=False)
synpts_pointset_um = PointSet.from_txt_file(synpts_um_filepath)


# %%
swc_spine_model_um = SWCModel.from_swc_file(swc_spine_um_filepath)
neckpt_pointset_um = PointSet.from_txt_file(neckpt_um_filepath)
plot_model(swc_model=swc_spine_model_um, point_set=neckpt_pointset_um, point_size=0.1)

# %%
geom = SinkGeometry(
    radius=sink_radius_um, length=2 * sink_radius_um, n_cylinders=5, axis="-z"
)
out_path = append_sink_to_swc(
    swc_in=swc_spine_um_filepath,
    swc_out=swc_with_sink_um_filepath,
    neck_point=neckpt_um_filepath,
    geom=geom,
    tag=5,
)


# %%
swc_model_with_sink_um = SWCModel.from_swc_file(swc_with_sink_um_filepath)
fig = plot_model(
    swc_model=swc_model_with_sink_um,
    point_set=synpts_pointset_um,
    point_size=0.2,
    point_color="crimson",
)

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
#     )
# )

fig.show()
