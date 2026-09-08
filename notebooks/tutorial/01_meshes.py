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
# # 01 — Meshes (paths, quality checks, repair)
#
# Closed triangle meshes live under `data/mesh/`: one file per spine
# (`TS{id}.obj`) plus the parent cell `cell_wrapped_simplified.obj`. Path
# helpers are in `toric_spines_sim.paths`. Mesh algorithms such as hole filling
# belong to [trimesh](https://trimesh.org/) and
# [pymcfs](https://github.com/jmrfox/pymcfs). This notebook covers this
# project's files, basic quality checks, and the repair step the skeletonizer
# already calls.
#
# IMOD surface reconstruction is documented separately
# (`docs/reconstruction.md`). Skeletonization is notebook 02.

# %%
import trimesh

from toric_spines_sim.geometry.mesh_pipeline import (
    list_ts_meshes,
    resolve_mesh_path,
    resolve_mesh_targets,
)
from toric_spines_sim.paths import (
    DATA_DIR,
    MESH_DIR,
    get_cell_mesh_path,
    get_mesh_path,
)

spine_id = "TS1"
print("DATA_DIR:", DATA_DIR)
print("MESH_DIR:", MESH_DIR)

# %% [markdown]
# ## Resolve paths
#
# `get_mesh_path("TS1.obj")` looks up a file under `data/mesh/`.
# `resolve_mesh_path` also accepts a spine id (`TS1`) or an existing file path.
# `resolve_mesh_targets` is what the command-line scripts use (`TS1` vs `--all`).

# %%
mesh_path = get_mesh_path(f"{spine_id}.obj")
print("get_mesh_path:", mesh_path)
print("resolve_mesh_path('TS1'):", resolve_mesh_path("TS1"))
print("list_ts_meshes:", [p.name for p in list_ts_meshes()])
print("targets TS1:", [p.name for p in resolve_mesh_targets(["TS1"])])

cell_path = get_cell_mesh_path()
print("cell mesh:", cell_path, "exists" if cell_path.is_file() else "MISSING")

# %% [markdown]
# ## Load and quality checks
#
# pymcfs needs a watertight, single-component surface. Genus > 0 (topological
# holes) is allowed — that is the point of toric spines.
# On a closed surface, $\chi = 2 - 2g$.

# %%
mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
print(f"file:       {mesh_path.name}")
print(f"vertices:   {len(mesh.vertices)}")
print(f"faces:      {len(mesh.faces)}")
print(f"watertight: {mesh.is_watertight}")
print(f"euler χ:    {mesh.euler_number}")
print(f"bounds:     {mesh.bounds}")
print(f"extents:    {mesh.extents}")
bodies = mesh.split(only_watertight=False)
print(f"components: {len(bodies)}")

# %% [markdown]
# ## Repair (as the pipeline uses it)
#
# `skeletonize_mesh` calls `pymcfs.load_and_repair` before mean-curvature
# flow. That function checks whether the mesh is watertight and, if not, runs
# `pymcfs.MeshManager.repair_mesh()`. The repair is limited: it looks for a
# few common defects. If a mesh still fails to load, repair or remesh it first
# (see `docs/skeletons.md` for CGALLab).

# %%
from pymcfs import load_and_repair

repaired = load_and_repair(str(mesh_path))
print(f"repaired vertices: {len(repaired.vertices)}")
print(f"repaired faces:    {len(repaired.faces)}")
print(f"repaired watertight: {repaired.is_watertight}")

# %% [markdown]
# ## All spines (optional)
#
# Quality-check every `data/mesh/TS*.obj`.

# %%
for path in list_ts_meshes():
    other_mesh = trimesh.load(str(path), force="mesh", process=False)
    print(
        f"{path.name:12s}  V={len(other_mesh.vertices):6d}  "
        f"F={len(other_mesh.faces):6d}  "
        f"watertight={other_mesh.is_watertight}  χ={other_mesh.euler_number}"
    )
