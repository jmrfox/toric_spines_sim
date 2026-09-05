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
# # View mesh + skeleton overlays
#
# Run skeletonization first:
#
# ```bash
# uv sync --extra mesh
# uv run python scripts/skeletonize_meshes.py --all
# ```
#
# Then open this notebook to compare each `data/mesh/TS*.obj` with its
# `data/skeletons/<stem>.polylines.txt`.

# %%
from pathlib import Path

from toric_spines_sim.geometry.mesh_pipeline import (
    default_polylines_path,
    list_ts_meshes,
)
from toric_spines_sim.viz import figure_mesh_and_skeleton

# Optional: restrict to a subset of stems, e.g. ["TS1", "TS2"]. Empty = all paired.
STEMS: list[str] = []

# %%
pairs: list[tuple[Path, Path]] = []
for mesh_path in list_ts_meshes():
    if STEMS and mesh_path.stem not in STEMS:
        continue
    polylines_path = default_polylines_path(mesh_path)
    if polylines_path.is_file():
        pairs.append((mesh_path, polylines_path))

print(f"Found {len(pairs)} mesh/skeleton pair(s)")
for mesh_path, polylines_path in pairs:
    print(f"  {mesh_path.name}  <->  {polylines_path.name}")

if not pairs:
    raise FileNotFoundError(
        "No mesh/skeleton pairs found. Run scripts/skeletonize_meshes.py first."
    )

# %%
for mesh_path, polylines_path in pairs:
    fig = figure_mesh_and_skeleton(mesh_path, polylines_path)
    fig.show()
