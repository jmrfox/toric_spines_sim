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
from swctools import SWCModel, PointSet
from toric_spines_sim.paths import get_swc_path, get_pointset_path

# %%
from toric_spines_sim.paths import get_swc_path, get_pointset_path

swc_spine_filepath = get_swc_path("TS1_s200_equivalent_area.swc", units="microns")
swc_wsink_filepath = get_swc_path("TS1_s200_wsink_r20um.swc", units="microns")
synpts_ps_filepath = get_pointset_path("TS1_synpts.txt", units="microns")
neckpoint_ps_filepath = get_pointset_path("TS1_neckpoint.txt", units="microns")
swc_spine = SWCModel.from_swc_file(swc_spine_filepath)
swc_wsink = SWCModel.from_swc_file(swc_wsink_filepath)
ps = PointSet.from_txt_file(synpts_ps_filepath)
neckpoint_ps = PointSet.from_txt_file(neckpoint_ps_filepath)

swc_wsink.print_attributes()

# %%
# compute frusta and point sets
frusta = FrustaSet.from_swc_model(swc_, sides=10, end_caps=False)

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
