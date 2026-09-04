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
from swctools import *

# %%
swc_filepath = "../data/swc/pixels/TS2_s50.swc"
ps_filepath = "../data/pointsets/pixels/TS2_AZ.txt"
swc_model = SWCModel.from_swc_file(swc_filepath)
ps = PointSet.from_txt_file(ps_filepath)

swc_model.print_attributes()
print(ps)

# %%
# compute frusta and point sets
frusta = FrustaSet.from_swc_model(swc_model, sides=10, end_caps=False)

# project pointset onto frustaset
ps_projected = ps.project_onto_frusta(frusta)

fig = plot_model(
    swc_model=swc_model,
    frusta=frusta,
    show_frusta=True,
    show_centroid=True,
    slider=True,
    point_set=ps_projected,  # use the PointSet
    point_size=5,  # multiplies base_radius (0.05 * 1.5 = 0.075)
    point_color="crimson",
)

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    )
)

fig.show()
