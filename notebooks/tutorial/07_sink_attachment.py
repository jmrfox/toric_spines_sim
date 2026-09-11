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
# # 07 — Sink attachment
#
# We model the downstream cell volume with a sink structure.
# To attach a straight cylinder, pass a `SinkGeometry` into
# `append_sink_write`. Simulations load the **micron** SWC. Synapse
# placement is `synapses_basic`.
#
# ```bash
# uv run python scripts/append_sink.py TS1
# ```
#
# Tags used throughout: spine=3, sink cylinder=5, sink tip=6. By default this
# notebook loads `TS1_wsink_r10um.swc`. `RECOMPUTE = True` writes under
# `notebooks/tutorial/_artifacts/` only.
#
# Radius scaling at simulation time (`sink_radii_scale`, `neck_radius_scale`
# on the parameter set) is applied by `TSModel.build_cell()` — see
# `tsmodel_and_tsrecipe`.

# %%
from collections import Counter
from pathlib import Path

from swctools import PointSet, SWCModel, plot_model

from toric_spines_sim.geometry import DEFAULT_SINK_RADIUS_UM, append_sink_write
from toric_spines_sim.geometry.sink import (
    SinkGeometry,
    append_sink_to_swc,
    neck_point_from_swc_file,
    optimal_sink_direction,
    sink_endpoint_location_from_swc_file,
)
from toric_spines_sim.paths import (
    NOTEBOOKS_DIR,
    SWC_MICRONS_DIR,
    get_pointset_path,
    get_swc_path,
)

spine_id = "TS1"
RECOMPUTE = False  # append a new sink into tutorial_output_dir
SHOW_ALL = False  # plot every spine-plus-sink SWC
UM_PER_PX = 0.005  # 5 nm / pixel
SINK_RADIUS_UM = DEFAULT_SINK_RADIUS_UM
tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"

# %% [markdown]
# ## `SinkGeometry`
#
# `SinkGeometry` is the object passed into `append_sink_write` /
# `append_sink_to_swc`. Straight cylinder: radius, length, number of frusta,
# connector stub, axis (`'x'`/`'-z'` or a 3-vector). Without an explicit
# geometry, `append_sink_write` picks a direction with
# `optimal_sink_direction`.

# %%
print("DEFAULT_SINK_RADIUS_UM:", DEFAULT_SINK_RADIUS_UM)
print("SinkGeometry fields:", SinkGeometry.__dataclass_fields__.keys())
print(SinkGeometry())

neckpoint_path_pixels = get_pointset_path(f"{spine_id}_neckpoint.txt", units="pixels")
swc_path_pixels_in = get_swc_path(f"{spine_id}.swc", units="pixels")
if neckpoint_path_pixels.is_file() and swc_path_pixels_in.is_file():
    direction = optimal_sink_direction(neckpoint_path_pixels, swc_path_pixels_in)
    print("optimal_sink_direction:", direction)

# %% [markdown]
# ## Append sink (pixel + micron)

# %%
if RECOMPUTE:
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    swc_path_pixels, swc_path_microns = append_sink_write(
        swc_path_pixels_in,
        neckpoint_path_pixels=neckpoint_path_pixels if neckpoint_path_pixels.is_file() else None,
        radius_um=SINK_RADIUS_UM,
        um_per_px=UM_PER_PX,
        output_swc_path_pixels=tutorial_output_dir / f"{spine_id}_wsink_r{SINK_RADIUS_UM:g}um.swc",
        output_swc_path_microns=tutorial_output_dir
        / f"{spine_id}_wsink_r{SINK_RADIUS_UM:g}um_microns.swc",
        output_neckpoint_path_microns=tutorial_output_dir / f"{spine_id}_neckpoint_um.txt",
    )
else:
    swc_path_pixels = get_swc_path(
        f"{spine_id}_wsink_r{SINK_RADIUS_UM:g}um.swc", units="pixels"
    )
    swc_path_microns = get_swc_path(
        f"{spine_id}_wsink_r{SINK_RADIUS_UM:g}um.swc", units="microns"
    )

print("pixel sink SWC: ", swc_path_pixels)
print("micron sink SWC:", swc_path_microns)


def tag_counts(path):
    counts: Counter[int] = Counter()
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        counts[int(line.split()[1])] += 1
    return counts


print("tags (micron):", dict(tag_counts(swc_path_microns)))
print("sink endpoint:", sink_endpoint_location_from_swc_file(swc_path_microns))
print("neck from SWC header:", neck_point_from_swc_file(swc_path_microns))

header_lines = [
    line for line in swc_path_microns.read_text().splitlines() if line.startswith("#")
]
print("header:")
for line in header_lines:
    print(" ", line)

# %%
model = SWCModel.from_swc_file(str(swc_path_microns), validate_reconnections=False)
fig = plot_model(
    swc_model=model,
    slider=False,
    title=f"{spine_id} sink SWC (µm)",
    show_axes=False,
    show_frusta=True,
    show_centroid=False,
    width=1200,
    height=900,
)
fig.show()

# %% [markdown]
# ## Toy cylinder
#
# Same `SinkGeometry` / `append_sink_to_swc` path without a mesh.

# %%
cylinder_radius_um = 10
cylinder_swc_path = get_swc_path("cylinder.swc", units="microns")
cylinder_neck_path = get_pointset_path("cylinder_neckpoint.txt", units="microns")
cylinder_synpts_path = get_pointset_path("cylinder_synpts.txt", units="microns")
cylinder_sink_swc_path = (
    tutorial_output_dir / f"cylinder_wsink_r{cylinder_radius_um}um.swc"
)
if not cylinder_swc_path.is_file():
    print(f"Skip cylinder: missing {cylinder_swc_path}")
else:
    tutorial_output_dir.mkdir(parents=True, exist_ok=True)
    geom = SinkGeometry(
        radius=cylinder_radius_um,
        length=2 * cylinder_radius_um,
        n_cylinders=5,
        axis="-z",
    )
    append_sink_to_swc(
        swc_in=cylinder_swc_path,
        swc_out=cylinder_sink_swc_path,
        neck_coords=cylinder_neck_path,
        geom=geom,
        tag=5,
    )
    cylinder_model = SWCModel.from_swc_file(
        str(cylinder_sink_swc_path), validate_reconnections=False
    )
    cylinder_points = (
        PointSet.from_txt_file(cylinder_synpts_path)
        if cylinder_synpts_path.is_file()
        else None
    )
    fig = plot_model(
        swc_model=cylinder_model,
        point_set=cylinder_points,
        point_size=0.2,
        point_color="crimson",
        slider=False,
        title="cylinder + sink (µm)",
        show_axes=False,
        show_frusta=True,
        show_centroid=False,
        width=900,
        height=700,
    )
    fig.show()

# %% [markdown]
# ## All spines (optional)
#
# ```bash
# uv run python scripts/append_sink.py --all
# ```

# %%
if SHOW_ALL:
    preferred = "10"
    by_stem: dict[str, list[Path]] = {}
    for path in sorted(SWC_MICRONS_DIR.glob("TS*_wsink_*.swc")):
        stem = path.name.split("_wsink_", 1)[0]
        by_stem.setdefault(stem, []).append(path)
    swc_paths = []
    for stem in sorted(by_stem, key=lambda s: (len(s), s)):
        paths = by_stem[stem]
        match = [p for p in paths if f"_r{preferred}um" in p.name]
        swc_paths.append(match[0] if match else paths[0])
    print(f"Found {len(swc_paths)} spine+sink SWC(s)")
    if not swc_paths:
        raise FileNotFoundError(
            f"No TS*_wsink_*.swc under {SWC_MICRONS_DIR}. Run scripts/append_sink.py --all."
        )
    for path in swc_paths:
        print(f"  {path.name}")
        model = SWCModel.from_swc_file(str(path), validate_reconnections=False)
        plot_model(
            swc_model=model,
            slider=False,
            title=path.stem,
            show_axes=False,
            show_frusta=True,
            show_centroid=False,
            width=1200,
            height=900,
        ).show()
else:
    print("SHOW_ALL is False; skip gallery. Set True to plot every wsink SWC.")
