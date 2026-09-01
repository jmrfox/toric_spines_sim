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
# # View mesh + fitted cable overlays
#
# After skeletons exist, fit SWCs:
#
# ```bash
# uv sync --extra mesh
# uv run python scripts/fit_swc.py --all
# ```
#
# Then open this notebook to compare each `data/mesh/TS*.obj` with its
# fitted cable model from `data/swc/pixels/<stem>.swc` (swctools frusta via
# `plot_model`, matching mascaf demos — not just the SWC centroid graph).

# %%
from pathlib import Path

from toric_spines_sim.geometry.mesh_pipeline import (
    default_swc_path,
    list_ts_meshes,
)
from toric_spines_sim.viz import figure_mesh_and_swc

# Optional: restrict to a subset of stems, e.g. ["TS1", "TS2"]. Empty = all paired.
STEMS: list[str] = []

# %%
pairs: list[tuple[Path, Path]] = []
for mesh_path in list_ts_meshes():
    if STEMS and mesh_path.stem not in STEMS:
        continue
    swc_path = default_swc_path(mesh_path)
    if swc_path.is_file():
        pairs.append((mesh_path, swc_path))

print(f"Found {len(pairs)} mesh/SWC pair(s)")
for mesh_path, swc_path in pairs:
    print(f"  {mesh_path.name}  <->  {swc_path.name}")

if not pairs:
    raise FileNotFoundError(
        "No mesh/SWC pairs found. Run scripts/fit_swc.py after skeletonization."
    )

# %%
for mesh_path, swc_path in pairs:
    fig = figure_mesh_and_swc(mesh_path, swc_path)
    fig.show()
