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
# # 21 — Synthetic spiny dendrite
#
# A comparison morphology — tapered trunk with classical neck+head spines —
# is `SpinyDendriteParams` plus `build_spiny_dendrite`. It writes the same
# subsystem files as a toric spine (`SWC` + AZ + neckpoint). A sink attaches
# with `SinkGeometry` as in `sink_attachment`.
#
# Writes under `_artifacts/` only. An earlier, less structured demo remains at
# `notebooks/misc/spiny_dendrite_demo.py`.

# %%
from swctools import PointSet, SWCModel, plot_model

from toric_spines_sim.geometry.dendrite import (
    SpinyDendriteParams,
    ToricSpineMatchParams,
    build_from_toric_spine,
    build_spiny_dendrite,
    write_subsystem,
)
from toric_spines_sim.geometry.sink import SinkGeometry, append_sink_to_swc
from toric_spines_sim.paths import (
    NOTEBOOKS_DIR,
    get_pointset_path,
    get_swc_path,
)

tutorial_output_dir = NOTEBOOKS_DIR / "tutorial" / "_artifacts"
tutorial_output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Explicit parameters
#
# Trunk is a straight taper. Spines attach to trunk nodes (never the proximal
# neck). Active zones sit at head XYZ — one synapse per spine.

# %%
params = SpinyDendriteParams(
    length=20.0,
    trunk_neck_radius=0.5,
    trunk_tip_radius=0.35,
    n_spines=12,
    spine_length=1.0,
    spine_neck_radius=0.1,
    spine_head_radius=0.25,
    spine_neck_length_fraction=0.5,
    max_spines_per_node=3,
    distribution="even",
    azimuth0=0.0,
    axis="z",
)
print("SpinyDendriteParams:", type(params).__name__)
print("  length =", params.length, "n_spines =", params.n_spines)
morph = build_spiny_dendrite(params)
print("build_spiny_dendrite →", type(morph).__name__)
print(f"trunk nodes: {len(morph.trunk_node_ids)}")
print(f"spines / AZ: {morph.n_spines}")
print(f"spines per attach node: {morph.spines_per_attach_node}")
print(f"surface area: {morph.surface_area():.3f} µm²")
print(f"volume:       {morph.volume():.3f} µm³")
print(f"neck point:   {morph.neck_point}")

swc_path = tutorial_output_dir / "spiny_dendrite.swc"
az_path = tutorial_output_dir / "spiny_dendrite_AZ.txt"
neck_path = tutorial_output_dir / "spiny_dendrite_neckpoint.txt"
write_subsystem(morph, swc_path, az_path, neck_path)
print("wrote", swc_path.name, az_path.name, neck_path.name)

# %%
swc_model = SWCModel.from_swc_file(str(swc_path), validate_reconnections=False)
az_points = PointSet.from_txt_file(str(az_path))
plot_model(
    swc_model=swc_model,
    point_set=az_points,
    point_size=0.15,
    point_color="crimson",
    slider=False,
    title="spiny dendrite + AZ",
    show_axes=False,
    show_frusta=True,
    show_centroid=False,
    width=900,
    height=700,
).show()

# %% [markdown]
# ## Attach a sink
#
# Same `SinkGeometry` / `append_sink_to_swc` as a toric spine.

# %%
sink_radius_um = 5.0
swc_with_sink = tutorial_output_dir / f"spiny_dendrite_wsink_r{sink_radius_um:g}um.swc"
geom = SinkGeometry(
    radius=sink_radius_um,
    length=2 * sink_radius_um,
    n_cylinders=4,
    axis="-z",
)
print("SinkGeometry:", geom)
append_sink_to_swc(
    swc_in=swc_path,
    swc_out=swc_with_sink,
    neck_coords=neck_path,
    geom=geom,
    tag=5,
)
plot_model(
    swc_model=SWCModel.from_swc_file(str(swc_with_sink), validate_reconnections=False),
    slider=False,
    title="spiny dendrite + sink",
    show_axes=False,
    show_frusta=True,
    show_centroid=False,
    width=900,
    height=700,
).show()

# %% [markdown]
# ## Match a toric spine (optional)
#
# `build_from_toric_spine` sets `n_spines` from the AZ file and scales so
# lateral surface area matches the TS (excluding sink tags 5/6).

# %%
ts_swc = get_swc_path("TS1_wsink_r10um.swc", units="microns")
ts_az = get_pointset_path("TS1_synpts.txt", units="microns")
if ts_swc.is_file() and ts_az.is_file():
    matched = build_from_toric_spine(
        ts_swc,
        ts_az,
        match=ToricSpineMatchParams(scale_strategy="relative"),
    )
    print(f"matched n_spines: {matched.n_spines}")
    print(f"matched SA: {matched.surface_area():.1f} µm²")
    print("diagnostics:", {k: round(v, 3) if isinstance(v, float) else v for k, v in matched.diagnostics.items()})
else:
    print("skip match: need TS1 SWC and AZ/synpts")
