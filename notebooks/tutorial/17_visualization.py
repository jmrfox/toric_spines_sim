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
# # 17 — Visualization
#
# This notebook uses the package Plotly / matplotlib / Dash helpers; it is not
# a general plotting course. Morphology quality checks do not need Arbor.
# Animation and the simulation dashboard need a `SimulationResults` from
# notebook 13/14 (leave `RUN_PLAYBACK = False` unless you want to build
# frames).
#
# Production Dash: `uv run python -m simulations.ts1.axons_dash`.
# Parameter-sweep browser: `create_hypergrid_dash_app(data_dir)` on a
# hypergrid archive (see `docs/simulations.md`).

# %%
import arbor as A
from swctools import PointSet, SWCModel, plot_model

from toric_spines_sim.paths import (
    get_mesh_path,
    get_pointset_path,
    get_skeleton_path,
    get_swc_path,
)
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import (
    VizConfig,
    figure_mesh_and_skeleton,
    figure_mesh_and_swc,
    plot_morphology_frusta_3d,
)

spine_id = "TS1"
RUN_PLAYBACK = False

mesh_path = get_mesh_path(f"{spine_id}.obj")
polylines_path = get_skeleton_path(f"{spine_id}.polylines.txt")
swc_path_pixels = get_swc_path(f"{spine_id}.swc", units="pixels")
swc_path_microns = get_swc_path(f"{spine_id}_wsink_r10um.swc", units="microns")
synpts_path = get_pointset_path(f"{spine_id}_synpts.txt", units="microns")
neckpoint_path_pixels = get_pointset_path(f"{spine_id}_neckpoint.txt", units="pixels")

# %% [markdown]
# ## Mesh vs skeleton / SWC
#
# Same helpers as notebooks 02 and 04.

# %%
if polylines_path.is_file():
    figure_mesh_and_skeleton(mesh_path, polylines_path).show()
else:
    print("skip skeleton overlay: missing", polylines_path)

neck_points = load_xyz_points(neckpoint_path_pixels) if neckpoint_path_pixels.is_file() else None
if swc_path_pixels.is_file():
    figure_mesh_and_swc(mesh_path, swc_path_pixels, neck_points=neck_points).show()
else:
    print("skip SWC overlay: missing", swc_path_pixels)

# %% [markdown]
# ## SWC frusta (swctools vs Arbor)
#
# `plot_model` is swctools. `plot_morphology_frusta_3d` draws the Arbor
# morphology loaded from the same file.

# %%
model = SWCModel.from_swc_file(str(swc_path_microns), validate_reconnections=False)
syn_pointset = PointSet.from_txt_file(synpts_path)
plot_model(
    swc_model=model,
    point_set=syn_pointset,
    point_size=0.15,
    point_color="crimson",
    slider=False,
    title=f"{spine_id} swctools",
    show_axes=False,
    show_frusta=True,
    show_centroid=False,
    width=900,
    height=700,
).show()

morph = A.load_swc_arbor(str(swc_path_microns))
synpts = load_xyz_points(synpts_path)
plot_morphology_frusta_3d(
    morph,
    backend="plotly",
    overlays={"syn": synpts},
    n_sides=10,
    alpha=0.8,
    config=VizConfig(width=900, height=700),
).show()

# %% [markdown]
# ## Animation and Dash (optional)
#
# Voltage-colored frusta need probes on many segments (`record_points="all"`
# in `TSSimulator` — large). A sink-only result still builds, but the mesh
# is nearly one color.
#
# ```python
# from toric_spines_sim.paths import NOTEBOOKS_DIR
# from toric_spines_sim.viz import Animation, create_simulation_dash_app
# from toric_spines_sim.viz import prepare_simulation_dashboard_data
#
# tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"
# anim = Animation(results, swc_filepath=swc_path_microns)
# anim.create(tutorial_output_dir / "playback.html", stride=5, fps=10)
#
# data = prepare_simulation_dashboard_data(results, swc_path_microns, dict(parameters))
# app = create_simulation_dash_app(data, title="TS1")
# app.run(debug=False)  # then open the printed URL
# ```
#
# Traces / rasters: `TimeSeriesPlotter`, `RasterPlotter`, `HistogramGridPlotter`.

# %%
if RUN_PLAYBACK:
    print("Set RUN_PLAYBACK after you have SimulationResults from notebook 13.")
else:
    print("RUN_PLAYBACK is False; morphology figures above are the live demo.")
