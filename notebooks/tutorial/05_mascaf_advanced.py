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
# # 05 — mascaf (advanced)
#
# `fit_swc` arguments (`FitOptions.max_edge_length`, radius scaling,
# `BasisOptimizer`) are easier to see on a single mesh. `mascaf_basic`
# fitted TS1 with package defaults; this notebook stays on the same mesh
# and skeleton.
#
# Object path:
#
# 1. `MeshManager` loads the triangle mesh and reports its bounding-box
#    diagonal.
# 2. `SkeletonGraph.from_txt` loads the pymcfs polylines.
# 3. `FitOptions` + `CableFitter.fit` build a cable graph.
# 4. Optional `scale_radii_to_match_mesh` rescales radii to a mesh metric.
# 5. Export SWC (`# CYCLE_BREAK` / `# MULTI_NECK` comments included).
#
# `fit_swc` is that sequence. Algorithm internals:
# [mascaf.readthedocs.io](https://mascaf.readthedocs.io/en/latest/).
# Leave `RECOMPUTE = False` unless you want extra fits under `_artifacts/`.

# %%
from toric_spines_sim.geometry import fit_swc
from toric_spines_sim.paths import (
    NOTEBOOKS_DIR,
    get_mesh_path,
    get_skeleton_path,
    get_swc_path,
)
from toric_spines_sim.viz import figure_mesh_and_swc

spine_id = "TS1"
RECOMPUTE = False  # extra fits into tutorial_output_dir
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

mesh_path = get_mesh_path(f"{spine_id}.obj")
polylines_path = get_skeleton_path(f"{spine_id}.polylines.txt")
precomputed_swc_path = get_swc_path(f"{spine_id}.swc", units="pixels")
print("mesh:     ", mesh_path)
print("skeleton: ", polylines_path)
print("SWC:      ", precomputed_swc_path)

# %% [markdown]
# ## `fit_swc` arguments
#
# | Argument | Default | Role |
# |----------|---------|------|
# | `max_edge_length_frac` | `0.08` | `FitOptions.max_edge_length` as a fraction of the mesh bounding-box diagonal. Larger → coarser compartments; smaller → finer. Prefer the largest value that still matches mesh structure, volume, and surface area. |
# | `radius_strategy` | `"equivalent_area"` | How mascaf assigns a radius at each sample from the local mesh. |
# | `scale_radii` | `True` | After the fit, call `scale_radii_to_match_mesh`. CLI `--no-scale-radii` turns this off. |
# | `scale_metric` | `"surface_area"` | Metric used when `scale_radii` is True. |
# | `basis_optimize` | `False` | If True, run mascaf `BasisOptimizer` before radius fitting (`FitOptions.basis_optimizer_options`). CLI `--basis-optimize`. |
# | `basis_optimizer_options` | `None` | Optional dict of kwargs for `BasisOptimizerOptions` when `basis_optimize` is True. |
#
# Matching CLI: `scripts/fit_swc.py` (`--max-edge-length-frac`,
# `--radius-strategy`, `--no-scale-radii`, `--basis-optimize`).
#
# Combined mesh → polylines → SWC is `mesh_to_swc` (`mascaf_basic`); it forwards
# these same fit arguments.

# %%
print("precomputed SWC exists:", precomputed_swc_path.is_file())
figure_mesh_and_swc(mesh_path, precomputed_swc_path).show()

# %% [markdown]
# ## Optional comparison: coarser maximum edge length
#
# `max_edge_length_frac=0.16` doubles the default edge-length cap. Writes
# only under `_artifacts/`. Compare visually to the precomputed `0.08` fit
# above.

# %%
if RECOMPUTE:
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    coarse_swc_path = tutorial_output_dir / f"{spine_id}_maxedge_0p16.swc"
    fit_swc(
        mesh_path,
        polylines_path,
        coarse_swc_path,
        max_edge_length_frac=0.16,
        radius_strategy="equivalent_area",
        scale_radii=True,
        scale_metric="surface_area",
        basis_optimize=False,
    )
    print("wrote", coarse_swc_path)
    figure_mesh_and_swc(mesh_path, coarse_swc_path).show()
else:
    print(
        "RECOMPUTE is False; skip extra fit.\n"
        "Equivalent CLI:\n"
        "  uv run python scripts/fit_swc.py TS1.obj --max-edge-length-frac 0.16"
    )

# %% [markdown]
# ## Basis optimizer
#
# `--basis-optimize` / `basis_optimize=True` runs mascaf’s basis refinement
# before radius fitting. It is off in the package default path. Pass extra
# optimizer kwargs only if you already know the mascaf `BasisOptimizerOptions`
# fields:
#
# ```python
# fit_swc(
#     mesh_path,
#     polylines_path,
#     swc_path,
#     basis_optimize=True,
#     basis_optimizer_options={...},
# )
# ```
#
# `# CYCLE_BREAK` pairs still come from the tree export; `tsmodel_and_tsrecipe` restores
# them as gap junctions.
