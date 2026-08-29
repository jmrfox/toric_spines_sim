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
import sys

sys.path.append("../")

from swctools import SWCModel, PointSet, FrustaSet, plot_model

import arbor as A
from toric_spines_sim.viz import *

# %% [markdown]
# This notebook demos the functionality in `arborviz`.
#
# Before looking at `arborviz`, however, we can load out model in using my `swctools` package. This consists of a morphology and a set of locations.

# %%
swc_filepath = "../data/swc/pixels/TS2_s50.swc"
ps_filepath = "../data/pointsets/pixels/TS2_AZ.txt"

swc = SWCModel.from_swc_file(swc_filepath)
frusta = FrustaSet.from_swc_model(swc, sides=20, end_caps=False)
ps = PointSet.from_txt(ps_filepath)

fig = plot_model(
    swc_model=swc,
    frusta=frusta,
    show_frusta=True,
    show_centroid=True,
    slider=True,
    point_set=ps,  # use the PointSet
    point_size=5,  # multiplies base_radius (0.05 * 1.5 = 0.075)
)
fig.show()


# %% [markdown]
# We can see that our points in the PointSet (red) do not sit on the surface of the SWC model. We can use the `PointSet.project_onto_frusta` method to project the points onto the surface of the SWC model.

# %%
ps_projected = ps.project_onto_frusta(frusta)

fig = plot_model(
    swc_model=swc,
    frusta=frusta,
    show_frusta=True,
    show_centroid=True,
    slider=True,
    point_set=ps_projected,  # use the PointSet
    point_size=5,  # multiplies base_radius (0.05 * 1.5 = 0.075)
)
fig.show()

# %% [markdown]
# Now, we want to bring this model into Arbor and view it directly from Arbor objects.

# %%
morph = A.load_swc_arbor(swc_filepath)

synapse_locations = ps_projected.points
print(f"Coordinates of points from projected PointSet: {synapse_locations}")

config = VizConfig(width=900, height=700)
# fig = plot_morphology_frusta_3d(
#     morph,
#     overlays={'syn': synapse_locations},
#     config=config,
# )
# fig

fig = plot_morphology_frusta_3d(
    morph,
    backend="plotly",  # force Plotly
    overlays={"syn": synapse_locations},
    n_sides=10,
    alpha=0.8,
    config=config,
)
fig.show()

# %%
