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
# # 03 — pymcfs (advanced)
#
# To change contraction or branching on one mesh (`TS1`), pass explicit
# kwargs through `skeletonize_mesh`. `pymcfs_basic` used package defaults
# (and optionally a gallery). This notebook walks the kwargs that wrapper
# forwards.
#
# For the MCFS algorithm — see the
# [pymcfs](https://github.com/jmrfox/pymcfs) docs. Cable fitting is
# `mascaf_basic` and `mascaf_advanced`. Leave `RECOMPUTE = False` unless you
# want a long pymcfs run into `notebooks/tutorial/_artifacts/`.

# %%
from toric_spines_sim.geometry import (
    TORIC_SPINES_SKELETONIZE_DEFAULTS,
    skeletonize_mesh,
)
from toric_spines_sim.paths import NOTEBOOKS_DIR, get_mesh_path, get_skeleton_path
from toric_spines_sim.viz import figure_mesh_and_skeleton, read_polylines_txt

spine_id = "TS1"
RECOMPUTE = False  # re-run pymcfs into tutorial_output_dir
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

mesh_path = get_mesh_path(f"{spine_id}.obj")
precomputed_skeleton_path = get_skeleton_path(f"{spine_id}.polylines.txt")
print("mesh:", mesh_path)
print("precomputed skeleton:", precomputed_skeleton_path)

# %% [markdown]
# ## How overrides merge
#
# `skeletonize_mesh(mesh, polylines_path, profile=..., branching=..., **kwargs)`:
#
# 1. Copy `TORIC_SPINES_SKELETONIZE_DEFAULTS`.
# 2. Overwrite `profile` and `branching` from the explicit arguments
#    (defaults `"auto"` / `"sparse"`).
# 3. Apply extra keyword arguments last (prune flags, tip extension,
#    timeouts, …).
#
# Those kwargs are forwarded to `pymcfs.skeletonize`. Matching CLI:
# `scripts/skeletonize_meshes.py`.

# %%
print("keys you can override:")
for key, value in TORIC_SPINES_SKELETONIZE_DEFAULTS.items():
    print(f"  {key:20s} {value!r}")

# %% [markdown]
# ## Option reference
#
# | Key | Default | Role |
# |-----|---------|------|
# | `profile` | `"auto"` | Named contraction preset. `"auto"` chooses settings from the mesh (not CGALLab QST/MCST numbers). |
# | `branching` | `"sparse"` | With `profile="auto"`: `"sparse"`, `"balanced"`, or `"dense"`. Sparse is the toric-spine batch default. |
# | `max_iterations` | `500` | Contraction iteration cap. |
# | `timeout_seconds` | `300.0` | Abort contraction after this many seconds. The CLI `--timeout <= 0` disables the limit (`None`). |
# | `max_vertex_growth` | `4.0` | Abort remesh if vertex count exceeds this multiple of the start count. |
# | `resample` | `False` | Optional resampling before contraction. |
# | `prune_exterior` | `True` | Drop skeleton pieces that leave the mesh. |
# | `prune_short_leaves` | `True` | Drop short dangling branches. |
# | `prune_thick_hubs` | `True` | At thick hubs, keep a principal subset of branches. |
# | `keep_hub_branches` | `2` | How many hub branches to keep when `prune_thick_hubs` is on. |
# | `extend_tips` | `True` | Grow skeleton tips toward the mesh boundary. |
# | `tip_extend_scale` | `1.0` | Max tip travel as a multiple of the mesh bounding-box diagonal. |
# | `validate` | `False` | Extra mesh validation inside `skeletonize`. The wrapper already ran `load_and_repair`, so this stays off. |
#
# If a skeleton is bad, change `--branching` or inspect watertightness
# (`meshes`) before hunting scalar weights. `profile="auto"` settings are
# not interchangeable with CGALLab sliders.

# %%
precomputed_polylines = read_polylines_txt(precomputed_skeleton_path)
print(
    f"precomputed: {len(precomputed_polylines)} polylines, "
    f"{sum(len(line) for line in precomputed_polylines)} points"
)

# %% [markdown]
# ## Optional recompute with an override
#
# Example: keep sparse branching but turn tip extension off. Writes only
# under `_artifacts/`. Compare polyline counts to the precomputed skeleton
# in `data/skeletons/`.

# %%
if RECOMPUTE:
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    no_tip_extend_path = tutorial_output_dir / f"{spine_id}_no_tip_extend.polylines.txt"
    skeletonize_mesh(
        mesh_path,
        no_tip_extend_path,
        profile="auto",
        branching="sparse",
        extend_tips=False,
    )
    no_tip_extend_polylines = read_polylines_txt(no_tip_extend_path)
    print(f"precomputed polylines: {len(precomputed_polylines)}")
    print(f"no-extend polylines:   {len(no_tip_extend_polylines)}")
    figure_mesh_and_skeleton(mesh_path, no_tip_extend_path).show()
else:
    print("RECOMPUTE is False; skip override run. Precomputed skeleton:")
    print(" ", precomputed_skeleton_path)
    figure_mesh_and_skeleton(mesh_path, precomputed_skeleton_path).show()

# %% [markdown]
# ## Troubleshooting
#
# - Not watertight → `meshes` (`load_and_repair`); IMOD / CGALLab for
#   reconstruction issues (`docs/skeletons.md`).
# - Timeouts / vertex growth: raise `timeout_seconds` or `max_vertex_growth`.
# - CHOLMOD (SuiteSparse) is optional and only speeds large meshes.
# - Combined mesh → polylines → SWC is `mesh_to_swc` in `mascaf_basic`.
