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
# Optional neckpoints (pixel space):
#
# ```bash
# uv run python scripts/compute_neckpoints.py TS1 TS2 TS4 TS24 TS48 TS67 TS76 --max-necks 1
# uv run python scripts/compute_neckpoints.py TS21 --max-necks 2
# uv run python scripts/compute_neckpoints.py TS3 --max-necks 3
# ```
#
# Then open this notebook to compare each `data/mesh/TS*.obj` with its
# fitted cable model from `data/swc/pixels/<stem>.swc` (swctools frusta via
# `plot_model`, matching mascaf demos — not just the SWC centroid graph).
# Neckpoints from `data/pointsets/pixels/<stem>_neckpoint.txt` are overlaid
# when present.

# %%
from pathlib import Path

from toric_spines_sim.geometry.mesh_pipeline import (
    default_swc_path,
    list_ts_meshes,
)
from toric_spines_sim.geometry.neckpoint import default_neckpoint_path
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import figure_mesh_and_swc

# Optional: restrict to a subset of stems, e.g. ["TS1", "TS2"]. Empty = all paired.
STEMS: list[str] = []

# %%
pairs: list[tuple[Path, Path, Path | None]] = []
for mesh_path in list_ts_meshes():
    if STEMS and mesh_path.stem not in STEMS:
        continue
    swc_path = default_swc_path(mesh_path)
    if not swc_path.is_file():
        continue
    neck_path = default_neckpoint_path(mesh_path.stem)
    pairs.append((mesh_path, swc_path, neck_path if neck_path.is_file() else None))

print(f"Found {len(pairs)} mesh/SWC pair(s)")
for mesh_path, swc_path, neck_path in pairs:
    neck_note = f"  +  {neck_path.name}" if neck_path else "  (no neckpoint file)"
    print(f"  {mesh_path.name}  <->  {swc_path.name}{neck_note}")

if not pairs:
    raise FileNotFoundError(
        "No mesh/SWC pairs found. Run scripts/fit_swc.py after skeletonization."
    )

# %%
for mesh_path, swc_path, neck_path in pairs:
    neck_points = load_xyz_points(neck_path) if neck_path is not None else None
    fig = figure_mesh_and_swc(mesh_path, swc_path, neck_points=neck_points)
    fig.show()
