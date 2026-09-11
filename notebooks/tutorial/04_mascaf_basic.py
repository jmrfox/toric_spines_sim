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
# # 04 — mascaf (basic)
#
# To fit a 1D cable to a mesh and a pymcfs skeleton, use `fit_swc`, the
# package wrapper around [mascaf](https://mascaf.readthedocs.io/en/latest/).
# The result is a pixel SWC plus `# CYCLE_BREAK` headers that later become
# gap junctions (`tsmodel_and_tsrecipe`). Option-by-option detail is
# `mascaf_advanced`.
#
# SWC is a tree, so loop-forming node pairs are stored as comments.
# Matching command-line script:
#
# ```bash
# uv run python scripts/fit_swc.py TS1.obj
# ```
#
# `FitOptions.max_edge_length` is a fraction of the mesh bounding-box diagonal
# (`--max-edge-length-frac`, default 0.08). A larger maximum edge length
# produces coarser compartments. Set `RECOMPUTE = True` to refit into
# `notebooks/tutorial/_artifacts/`.
#
# Algorithm internals stay in the mascaf docs.

# %%
from toric_spines_sim.geometry import (
    fit_swc,
    mesh_to_swc,
    parse_cycle_breaks,
    parse_multi_neck_reconnects,
)
from toric_spines_sim.geometry.mesh_pipeline import (
    MeshToSwcResult,
    default_polylines_path,
    default_swc_path,
    list_ts_meshes,
)
from toric_spines_sim.geometry.neckpoint import default_neckpoint_path
from toric_spines_sim.paths import (
    NOTEBOOKS_DIR,
    get_mesh_path,
    get_skeleton_path,
    get_swc_path,
)
from toric_spines_sim.utils import load_xyz_points
from toric_spines_sim.viz import figure_mesh_and_swc

spine_id = "TS1"
RECOMPUTE = False  # re-run mascaf into tutorial_output_dir
SHOW_ALL = False  # plot every mesh/SWC pair (slow)
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"
MAX_EDGE_LENGTH_FRAC = 0.08

print("fit_swc:", fit_swc.__name__)
print(
    "defaults used here: max_edge_length_frac="
    f"{MAX_EDGE_LENGTH_FRAC}, radius_strategy='equivalent_area', "
    "scale_radii=True, basis_optimize=False"
)

mesh_path = get_mesh_path(f"{spine_id}.obj")
polylines_path = tutorial_output_dir / f"{spine_id}.polylines.txt"
if not polylines_path.is_file():
    polylines_path = get_skeleton_path(f"{spine_id}.polylines.txt")

# %%
if RECOMPUTE:
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    swc_path = tutorial_output_dir / f"{spine_id}.swc"
    fit_swc(
        mesh_path,
        polylines_path,
        swc_path,
        max_edge_length_frac=MAX_EDGE_LENGTH_FRAC,
        basis_optimize=False,
    )
else:
    swc_path = get_swc_path(f"{spine_id}.swc", units="pixels")

if not swc_path.is_file():
    raise FileNotFoundError(
        f"No SWC at {swc_path}. Run scripts/fit_swc.py TS1.obj or set RECOMPUTE."
    )

print("skeleton:", polylines_path)
print("SWC:     ", swc_path)

# %% [markdown]
# ## Cycle breaks
#
# Each `# CYCLE_BREAK reconnect i j` pair is two SWC nodes that should be
# electrically coupled. `# MULTI_NECK` is the same idea for extra necks.
# `tsmodel_and_tsrecipe` turns those headers into `GapJunctionPoint`s.

# %%
cycle_breaks = parse_cycle_breaks(swc_path)
multi_neck = parse_multi_neck_reconnects(swc_path)
print(f"CYCLE_BREAK pairs: {len(cycle_breaks)}")
for pair in cycle_breaks[:8]:
    print(" ", pair)
if len(cycle_breaks) > 8:
    print(f"  ... ({len(cycle_breaks) - 8} more)")
print(f"MULTI_NECK pairs: {len(multi_neck)}")

header_lines = [
    line for line in swc_path.read_text().splitlines() if line.startswith("#")
]
print("header (first 12 lines):")
for line in header_lines[:12]:
    print(" ", line)

# %%
neck_path = default_neckpoint_path(spine_id)
neck_points = load_xyz_points(neck_path) if neck_path.is_file() else None
fig = figure_mesh_and_swc(mesh_path, swc_path, neck_points=neck_points)
fig.show()

# %% [markdown]
# ## `mesh_to_swc`
#
# One call for mesh → polylines → SWC. Default outputs are
# `data/skeletons/<spine_id>.polylines.txt` and
# `data/swc/pixels/<spine_id>.swc`. Do not pass those defaults from a
# notebook — write under `_artifacts/` instead. Flags:
#
# - `polylines_only=True` — stop after pymcfs
# - `skip_skeletonize=True` — reuse existing polylines, fit SWC only
#
# Cable-fit parameters (`max_edge_length_frac`, `radius_strategy`, …) are
# forwarded to `fit_swc` (`mascaf_advanced`).

# %%
print("default polylines:", default_polylines_path(mesh_path))
print("default SWC:      ", default_swc_path(mesh_path))
print("MeshToSwcResult fields:", MeshToSwcResult.__dataclass_fields__.keys())

if RECOMPUTE:
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    result = mesh_to_swc(
        spine_id,
        polylines_path=tutorial_output_dir / f"{spine_id}.polylines.txt",
        swc_path=tutorial_output_dir / f"{spine_id}_mesh_to_swc.swc",
        skip_skeletonize=polylines_path.is_file(),
        max_edge_length_frac=MAX_EDGE_LENGTH_FRAC,
        basis_optimize=False,
    )
    print(result)
else:
    print(
        "RECOMPUTE is False; not writing a combined mesh_to_swc result.\n"
        "Equivalent CLI: uv run python scripts/mesh_to_swc.py TS1.obj"
    )

# %% [markdown]
# ## All spines (optional)
#
# Set `SHOW_ALL = True` after fitting every cable:
#
# ```bash
# uv run python scripts/fit_swc.py --all
# ```

# %%
if SHOW_ALL:
    pairs = []
    for other_mesh in list_ts_meshes():
        other_swc = default_swc_path(other_mesh)
        if not other_swc.is_file():
            continue
        other_neck = default_neckpoint_path(other_mesh.stem)
        pairs.append(
            (other_mesh, other_swc, other_neck if other_neck.is_file() else None)
        )
    print(f"Found {len(pairs)} mesh/SWC pair(s)")
    if not pairs:
        raise FileNotFoundError(
            "No mesh/SWC pairs found. Run scripts/fit_swc.py after pymcfs."
        )
    for other_mesh, other_swc, other_neck in pairs:
        neck_note = f"  +  {other_neck.name}" if other_neck else "  (no neckpoint)"
        print(f"  {other_mesh.name}  <->  {other_swc.name}{neck_note}")
        neck_points = load_xyz_points(other_neck) if other_neck is not None else None
        figure_mesh_and_swc(other_mesh, other_swc, neck_points=neck_points).show()
else:
    print("SHOW_ALL is False; skip gallery. Set True to plot every spine.")

# %%
