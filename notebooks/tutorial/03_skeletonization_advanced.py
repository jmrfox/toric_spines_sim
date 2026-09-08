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
# # 03 — Skeletonization (pymcfs, advanced)
#
# Notebook 02 ran `skeletonize_mesh` with package defaults. This notebook
# documents the options that wrapper forwards, plus `mesh_to_swc` (skeleton
# and mascaf fit in one call). It does not teach the pymcfs algorithm — see
# that project's docs. Leave `RECOMPUTE = False` unless you want a long
# pymcfs run into `notebooks/tutorial/_artifacts/`.

# %%
from toric_spines_sim.geometry import (
    TORIC_SPINES_SKELETONIZE_DEFAULTS,
    mesh_to_swc,
    skeletonize_mesh,
)
from toric_spines_sim.geometry.mesh_pipeline import (
    MeshToSwcResult,
    default_polylines_path,
    default_swc_path,
)
from toric_spines_sim.paths import NOTEBOOKS_DIR, get_mesh_path, get_skeleton_path
from toric_spines_sim.viz import figure_mesh_and_skeleton, read_polylines_txt

spine_id = "TS1"
RECOMPUTE = False  # re-run pymcfs into tutorial_output_dir
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

mesh_path = get_mesh_path(f"{spine_id}.obj")
precomputed_skeleton_path = get_skeleton_path(f"{spine_id}.polylines.txt")

# %% [markdown]
# ## Defaults vs overrides
#
# `skeletonize_mesh(mesh, polylines_path, profile=..., branching=..., **kwargs)` copies
# `TORIC_SPINES_SKELETONIZE_DEFAULTS`, overwrites `profile` / `branching`,
# then applies extra keyword arguments (prune flags, tip extension, timeouts,
# and so on).

# %%
print("keys you can override:")
for key, value in TORIC_SPINES_SKELETONIZE_DEFAULTS.items():
    print(f"  {key:20s} {value!r}")

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
    precomputed_polylines = read_polylines_txt(precomputed_skeleton_path)
    print(f"precomputed polylines: {len(precomputed_polylines)}")
    print(f"no-extend polylines: {len(no_tip_extend_polylines)}")
    figure_mesh_and_skeleton(mesh_path, no_tip_extend_path).show()
else:
    print("RECOMPUTE is False; skip override run. Precomputed skeleton:")
    print(" ", precomputed_skeleton_path)
    print(" polylines:", len(read_polylines_txt(precomputed_skeleton_path)))

# %% [markdown]
# ## `mesh_to_swc`
#
# One call for mesh → polylines → SWC. Default outputs are
# `data/skeletons/<spine_id>.polylines.txt` and
# `data/swc/pixels/<spine_id>.swc`. Do not pass those defaults from a
# notebook — write under `_artifacts/` instead. Flags:
#
# - `polylines_only=True` — stop after skeletonization
# - `skip_skeletonize=True` — reuse existing polylines, fit SWC only
#
# Cable-fit details (maximum edge length, `# CYCLE_BREAK`) are notebook 04.

# %%
print("default polylines:", default_polylines_path(mesh_path))
print("default SWC:      ", default_swc_path(mesh_path))
print("MeshToSwcResult fields:", MeshToSwcResult.__dataclass_fields__.keys())

if RECOMPUTE:
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    result = mesh_to_swc(
        spine_id,
        polylines_path=tutorial_output_dir / f"{spine_id}.polylines.txt",
        swc_path=tutorial_output_dir / f"{spine_id}.swc",
        skip_skeletonize=precomputed_skeleton_path.is_file(),
        max_edge_length_frac=0.08,
        basis_optimize=False,
    )
    print(result)
else:
    print(
        "RECOMPUTE is False; not writing a combined mesh_to_swc result.\n"
        "Equivalent CLI: uv run python scripts/mesh_to_swc.py TS1.obj"
    )

# %% [markdown]
# ## Troubleshooting
#
# - Not watertight → notebook 01 (`load_and_repair`); IMOD / CGALLab for
#   reconstruction issues.
# - Missing polylines with `skip_skeletonize=True` → `FileNotFoundError`.
# - CHOLMOD is optional (SuiteSparse) and only speeds large meshes.
# - Timeouts / vertex growth: raise `timeout_seconds` or `max_vertex_growth`.
