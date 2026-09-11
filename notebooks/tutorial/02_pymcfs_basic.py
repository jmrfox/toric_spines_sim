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
# # 02 — pymcfs (basic)
#
# To skeletonize a closed triangle mesh, use `skeletonize_mesh`, the package
# wrapper around [pymcfs](https://github.com/jmrfox/pymcfs). Mean-curvature
# flow contracts the mesh to a 1D skeleton; this project writes polylines.
# Package defaults are enough for TS meshes; kwargs are in `pymcfs_advanced`.
# Cable fitting is `mascaf_basic`.
#
# The matching command-line script is:
#
# ```bash
# uv run python scripts/skeletonize_meshes.py TS1.obj
# uv run python scripts/skeletonize_meshes.py --all
# ```
#
# Package defaults match the toric-spine batch: `profile="auto"`,
# `branching="sparse"`, tip extension on. `profile="auto"` automatically
# chooses contraction settings.
#
# Set `RECOMPUTE = True` to run pymcfs and write under
# `notebooks/tutorial/_artifacts/`. Leave it `False` to load the precomputed
# skeleton in `data/skeletons/`.
#
# IMOD reconstruction and CGALLab: `docs/reconstruction.md`,
# `docs/skeletons.md`. Algorithm internals live in the pymcfs docs.

# %%
from toric_spines_sim.geometry import (
    TORIC_SPINES_SKELETONIZE_DEFAULTS,
    skeletonize_mesh,
)
from toric_spines_sim.geometry.mesh_pipeline import (
    default_polylines_path,
    list_ts_meshes,
)
from toric_spines_sim.paths import NOTEBOOKS_DIR, get_mesh_path, get_skeleton_path
from toric_spines_sim.viz import figure_mesh_and_skeleton, read_polylines_txt

spine_id = "TS1"
RECOMPUTE = False  # re-run pymcfs into tutorial_output_dir
SHOW_ALL = False  # plot every mesh/skeleton pair (slow)
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

mesh_path = get_mesh_path(f"{spine_id}.obj")
print("mesh:", mesh_path)

# %% [markdown]
# ## Package defaults
#
# `skeletonize_mesh` copies `TORIC_SPINES_SKELETONIZE_DEFAULTS`, then applies
# `profile` / `branching` and any extra keyword arguments. This notebook uses
# those defaults as-is. `pymcfs_advanced` lists every key and how overrides merge.

# %%
print("pymcfs defaults:")
for key, value in TORIC_SPINES_SKELETONIZE_DEFAULTS.items():
    print(f"  {key}: {value}")

if RECOMPUTE:
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    polylines_path = tutorial_output_dir / f"{spine_id}.polylines.txt"
    skeletonize_mesh(mesh_path, polylines_path)
else:
    polylines_path = get_skeleton_path(f"{spine_id}.polylines.txt")

if not polylines_path.is_file():
    raise FileNotFoundError(
        f"No skeleton at {polylines_path}. Run the CLI or set RECOMPUTE = True."
    )

polylines = read_polylines_txt(polylines_path)
n_points = sum(len(line) for line in polylines)
print(f"skeleton: {polylines_path}")
print(f"polylines: {len(polylines)}, points: {n_points}")

# %%
fig = figure_mesh_and_skeleton(mesh_path, polylines_path)
fig.show()

# %% [markdown]
# ## All spines (optional)
#
# Set `SHOW_ALL = True` to overlay every `data/mesh/TS*.obj` with its
# skeleton. Generate missing polylines first:
#
# ```bash
# uv run python scripts/skeletonize_meshes.py --all
# ```

# %%
if SHOW_ALL:
    pairs = []
    for other_mesh in list_ts_meshes():
        other_polylines = default_polylines_path(other_mesh)
        if other_polylines.is_file():
            pairs.append((other_mesh, other_polylines))
    print(f"Found {len(pairs)} mesh/skeleton pair(s)")
    if not pairs:
        raise FileNotFoundError(
            "No mesh/skeleton pairs found. Run scripts/skeletonize_meshes.py first."
        )
    for other_mesh, other_polylines in pairs:
        print(f"  {other_mesh.name}  <->  {other_polylines.name}")
        figure_mesh_and_skeleton(other_mesh, other_polylines).show()
else:
    print("SHOW_ALL is False; skip gallery. Set True to plot every spine.")

# %%
