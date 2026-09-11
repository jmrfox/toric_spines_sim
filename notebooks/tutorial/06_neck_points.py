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
# # 06 — Neck points
#
# Isolated TS meshes are closed at each neck; the parent cell mesh is open
# there. Cap faces that disagree with the cell surface mark the neck. Their
# area-weighted centroids are the XYZ used for sink attachment
# (`sink_attachment`). `NeckpointParams` controls the search;
# `compute_neck_points_for_spine` runs it.
#
# ```bash
# uv run python scripts/compute_neckpoints.py TS1 --max-necks 1
# ```
#
# By default this notebook loads the precomputed neckpoint files in `data/`.
# `RECOMPUTE = True` writes a copy under `notebooks/tutorial/_artifacts/`
# (it does not overwrite `data/`).

# %%
from toric_spines_sim.geometry.neckpoint import (
    NeckCandidate,
    NeckpointParams,
    compute_neck_points_for_spine,
    default_neckpoint_path,
)
from toric_spines_sim.geometry.mesh_pipeline import list_ts_meshes
from toric_spines_sim.paths import (
    NOTEBOOKS_DIR,
    get_cell_mesh_path,
    get_mesh_path,
    get_pointset_path,
    get_swc_path,
)
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import figure_mesh_and_swc

spine_id = "TS1"
RECOMPUTE = False  # re-detect neckpoints into tutorial_output_dir
SHOW_ALL = False  # print neckpoints for every spine
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

params = NeckpointParams(max_necks=1)
print("NeckpointParams:", params)
print("  max_necks:", params.max_necks)
print("  crop_margin:", params.crop_margin)
print("NeckCandidate fields:", NeckCandidate.__dataclass_fields__.keys())

# %% [markdown]
# ## Precomputed neckpoint files
#
# Pixel-space XYZ near the cut where the spine left the dendrite.
# Micron copies are written later by the sink pipeline (`sink_attachment`).

# %%
neckpoint_path_pixels = get_pointset_path(f"{spine_id}_neckpoint.txt", units="pixels")
neckpoint_path_microns = get_pointset_path(f"{spine_id}_neckpoint.txt", units="microns")
print(
    "pixel neckpoint:",
    neckpoint_path_pixels,
    "exists" if neckpoint_path_pixels.is_file() else "MISSING",
)
if neckpoint_path_pixels.is_file():
    neck_points = load_xyz_points(neckpoint_path_pixels)
    print(f"  {len(neck_points)} point(s): {neck_points}")
print(
    "micron neckpoint:",
    neckpoint_path_microns,
    "exists" if neckpoint_path_microns.is_file() else "MISSING",
)
print("default_neckpoint_path:", default_neckpoint_path(spine_id))
print("cell mesh:", get_cell_mesh_path())

# %% [markdown]
# ## Optional recompute
#
# `compute_neck_points_for_spine` takes a `NeckpointParams` instance.
# `max_necks` keeps the n largest caps.

# %%
if RECOMPUTE:
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = tutorial_output_dir / f"{spine_id}_neckpoint.txt"
    points = compute_neck_points_for_spine(
        spine_id,
        params=params,
        write=True,
        overwrite=True,
        output_path=output_path,
    )
    print(f"wrote {len(points)} point(s) to {output_path}")
    print(points)
else:
    print(
        "RECOMPUTE is False; not re-detecting. "
        "Load the precomputed file above."
    )

# %% [markdown]
# Overlay on the pixel SWC (red square = neck). Mesh-vs-cell detection is
# independent of the SWC; the SWC is only for the figure.

# %%
mesh_path = get_mesh_path(f"{spine_id}.obj")
swc_path_pixels = get_swc_path(f"{spine_id}.swc", units="pixels")
neck_for_plot = (
    load_xyz_points(neckpoint_path_pixels) if neckpoint_path_pixels.is_file() else None
)
if swc_path_pixels.is_file() and neck_for_plot:
    fig = figure_mesh_and_swc(mesh_path, swc_path_pixels, neck_points=neck_for_plot)
    fig.show()
else:
    print("Skip overlay: need pixel SWC and neckpoint files.")

# %% [markdown]
# ## All spines (optional)
#
# ```bash
# uv run python scripts/compute_neckpoints.py --all --max-necks 1
# ```

# %%
if SHOW_ALL:
    for mesh in list_ts_meshes():
        path = default_neckpoint_path(mesh.stem)
        if not path.is_file():
            print(f"  {mesh.stem}: MISSING {path.name}")
            continue
        neck_points = load_xyz_points(path)
        print(
            f"  {mesh.stem}: {len(neck_points)} point(s) "
            f"{neck_points[0] if neck_points else ''}"
        )
else:
    print("SHOW_ALL is False; skip gallery.")
